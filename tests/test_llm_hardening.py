"""模型调用层回归：失败计费、空 choices/围栏解析、用量缺失、超时与撤销、决策概率校验。"""
import time
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from advisor.harness import BudgetExceeded
from advisor.llm import DecisionClient, DecisionError, DeepSeekClient, LLMOutputError
from advisor.routing import RouterOut, decide_route
from advisor.routing.router import _top
from advisor.graph.nodes import Nodes
from tests.test_harness import make


class Out(BaseModel):
    n: int


class _Resp:
    def __init__(self, content: str | None = '{"n": 1}', usage: int | None = 30, choices: bool = True):
        message = SimpleNamespace(content=content) if content is not None else None
        self.choices = [SimpleNamespace(message=message)] if choices else []
        self.usage = SimpleNamespace(total_tokens=usage) if usage is not None else None


class _Completions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kw):
        self.calls.append(kw)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _client(responses):
    c = DeepSeekClient.__new__(DeepSeekClient)
    c.roles = {"draft": {"model": "m", "effort": "high"}}
    comp = _Completions(responses)
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    return c, comp


def test_failed_repair_retry_counts_tokens(policy):
    client, comp = _client([_Resp("not json", 30), _Resp("still bad", 30)])
    h, audit, budget = make(policy, [], limits={"tokens": 100, "wall_clock_s": 60})
    h.llm_client = client
    with pytest.raises(LLMOutputError):
        h.llm("draft", "s", "u", Out, purpose="x")
    assert len(comp.calls) == 2
    assert budget.used["tokens"] == 60                      # 失败调用实际消耗的 token 也要计费
    assert audit[-1]["tokens"] == 60 and audit[-1]["status"] == "LLMOutputError"


def test_api_error_after_first_attempt_counts_tokens(policy):
    client, _ = _client([_Resp("not json", 42), ConnectionError("boom")])
    h, audit, budget = make(policy, [], limits={"tokens": 100, "wall_clock_s": 60})
    h.llm_client = client
    with pytest.raises(ConnectionError):
        h.llm("draft", "s", "u", Out, purpose="x")
    assert budget.used["tokens"] == 42


def test_empty_choices_is_bad_output_and_retries(policy):
    client, comp = _client([_Resp(None, 15, choices=False), _Resp('{"n": 1}', 20)])
    h, _, budget = make(policy, [], limits={"tokens": 100, "wall_clock_s": 60})
    h.llm_client = client
    assert h.llm("draft", "s", "u", Out, purpose="x").n == 1
    assert len(comp.calls) == 2 and budget.used["tokens"] == 35   # 空 choices 不再 IndexError，走修复重试


def test_empty_output_twice_raises_output_error(policy):
    client, _ = _client([_Resp(None, 15, choices=False), _Resp("", 15)])
    h, _, budget = make(policy, [], limits={"tokens": 100, "wall_clock_s": 60})
    h.llm_client = client
    with pytest.raises(LLMOutputError):
        h.llm("draft", "s", "u", Out, purpose="x")
    assert budget.used["tokens"] == 30


def test_usage_missing_is_estimated():
    client, _ = _client([_Resp('{"n": 1}', usage=None)])
    obj, tokens = client.json("draft", "system", "user", Out)
    assert obj.n == 1 and tokens >= 1                       # 端点省略 usage 时不能按 0 计费


def test_fenced_json_accepted_in_one_call(policy):
    client, comp = _client([_Resp('```json\n{"n": 1}\n```', 10)])
    h, _, budget = make(policy, [], limits={"tokens": 100, "wall_clock_s": 60})
    h.llm_client = client
    assert h.llm("draft", "s", "u", Out, purpose="x").n == 1
    assert len(comp.calls) == 1 and budget.used["tokens"] == 10


def test_llm_call_bounded_by_wall_clock(policy):
    class _Slow:
        def json(self, **kw):
            time.sleep(2)
            return Out(n=1), 5

    h, audit, _ = make(policy, [], limits={"tokens": 100, "wall_clock_s": 0.2})
    h.llm_client = _Slow()
    started = time.monotonic()
    with pytest.raises(BudgetExceeded):
        h.llm("draft", "s", "u", Out, purpose="x")
    assert time.monotonic() - started < 1.6                 # 不再无限等待模型
    assert audit[-1]["status"] == "BudgetExceeded"


def test_overrun_after_return_is_not_committed(policy):
    class _Slowish:
        def json(self, **kw):
            time.sleep(0.1)
            return Out(n=1), 5

    h, _, budget = make(policy, [], limits={"tokens": 100, "wall_clock_s": 0.2})
    budget.started_at = time.time() - 0.15                 # 调用前未超时，返回后总时限已过
    h.llm_client = _Slowish()
    with pytest.raises(BudgetExceeded):
        h.llm("draft", "s", "u", Out, purpose="x")
    assert budget.used["tokens"] == 5                       # 返回后复查总时限，token 仍照实计费


def test_decide_checks_wall_clock(policy):
    class _Decision:
        def __init__(self):
            self.calls = 0

        def choose(self, state, questions):
            self.calls += 1
            return {}, {}

    h, _, budget = make(policy, [], limits={"tokens": 100, "wall_clock_s": 10})
    d = _Decision()
    h.decision_client = d
    budget.started_at = time.time() - 100
    with pytest.raises(BudgetExceeded):
        h.decide("state", {"q": ("i", {"a": "A"})}, purpose="x")
    assert d.calls == 0                                    # 超时任务不再发起外部请求


def test_low_remaining_tokens_makes_no_request(policy):
    client, comp = _client([_Resp('{"n": 1}', 5)])
    h, _, _ = make(policy, [], limits={"tokens": 10, "wall_clock_s": 60})
    h.llm_client = client
    with pytest.raises(BudgetExceeded):
        h.llm("draft", "s", "u", Out, purpose="x")
    assert comp.calls == []                                # 剩余不够最小合法输出：不发必败请求


def _decision_client(payload):
    c = DecisionClient.__new__(DecisionClient)
    c.model = "m"

    class _R:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    c._client = SimpleNamespace(post=lambda *a, **k: _R())
    return c


QUESTIONS = {"intent": ("q", {"research": "r", "explain": "e"})}


def test_decision_client_rejects_invalid_probabilities():
    ok = _decision_client({"answers": {"intent": {"probabilities": {"research": 0.9, "explain": 0.1}}}, "model": "m"})
    assert ok.choose("s", QUESTIONS)[0]["intent"]["research"] == 0.9
    for bad in (float("nan"), float("inf"), "Infinity", -0.1):
        client = _decision_client({"answers": {"intent": {"probabilities": {"research": bad, "explain": 0.2}}}})
        with pytest.raises(DecisionError):
            client.choose("s", QUESTIONS)
    with pytest.raises(DecisionError, match="缺少选项"):
        _decision_client({"answers": {"intent": {"probabilities": {"research": 0.9}}}}).choose("s", QUESTIONS)
    with pytest.raises(DecisionError):
        _decision_client({"answers": {"intent": {"probabilities": {"research": 0.0, "explain": 0.0}}}}).choose("s", QUESTIONS)


def test_router_top_and_small_fallback_ignore_non_finite():
    assert _top({"high": float("nan"), "low": 0.02, "mid": 0.03}) == "mid"

    class _H:
        decision_client = object()

        def decide(self, *a, **k):
            raise DecisionError("bad")

        def llm(self, *a, **k):
            return RouterOut(p_quick=0.1, p_standard=0.8, p_deep=0.1, stakes="low", uncertainty="low")

    d = decide_route(_H(), "复核 600999 业绩", "research", "auto",
                     {"classifier": "small", "temperature": 0.75, "abstain_below": 0.55, "deep_threshold_stakes": "high"})
    assert d.route == "standard" and d.classifier == "llm"   # 小模型坏输出回退大模型


def test_intake_small_ignores_non_finite(policy):
    node = Nodes.__new__(Nodes)
    node.d = SimpleNamespace(policy=policy)

    class _H:
        def decide(self, *a, **k):
            return {"intent": {"explain": float("nan"), "research": 0.9, "trade_execution": 0.05, "other": 0.05}}

    out = node._intake_small(_H(), "复核 600999", "600999", {})
    assert out.intent == "research"
