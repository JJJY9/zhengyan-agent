"""RSI 评测口径回归：评测命名空间档案不混入、S1 召回输入与线上一致、S2 主片为 evolution
且按坏案例逐条确认修复、S2/S3 成本信号不再恒为 0。"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from advisor.evals.compare import summarize
from advisor.rsi import RSIRunner
from advisor.rsi.evaluator import JudgeOut, _judge, s1_items
from advisor.rsi.gates import cost_gate
from advisor.rsi.issues import _issue
from advisor.rsi.models import LessonCard, Run
from tests.test_phase2 import make_service
from tests.test_rsi import RSILLM, seed_dossiers


def _runner(repo, policy):
    seed_dossiers(repo)
    svc = make_service(repo, policy, RSILLM())
    return RSIRunner(svc.deps, svc.deps.ledger, svc.deps.rsi)


# ---------- #9 评测命名空间档案 ----------

def test_eval_namespace_dossiers_excluded(repo, policy):
    runner = _runner(repo, policy)
    repo.db.dossiers.insert_one({
        "key": "evalns:600001", "version": 1, "ticker": "600001", "name": "甲酒业", "as_of": "2026-10-16",
        "question": "对照评测问题", "route": "deep", "status": "partial", "deliberation": None,
        "claims": [{"id": "c-eval", "text": "评测命名空间里的存疑主张", "kind": "fact",
                    "evidence_ids": ["ev_x"], "status": "disputed"}],
        "evidence_ids": [], "clue_ids": [], "review_triggers": [], "unknowns": [], "dissents": [],
        "quality": {"decision": "partial"}, "boundary_notes": [], "task_id": "t",
        "created_at": datetime.now(timezone.utc),
    })
    runner.collect()
    issues = runner.ledger.list_issues("S1")
    assert issues                                              # 真实档案照常登记
    assert all(i.source.get("dossier") != "evalns:600001" for i in issues)
    assert all(not it["dossier"].startswith("evalns:") for it in s1_items(repo, issues))


# ---------- #10 S1 召回输入与线上一致 ----------

class _JudgeH:
    def __init__(self):
        self.budget = SimpleNamespace(used={})
        self.user = ""

    def llm(self, role, system, user, schema, purpose):
        self.user = user
        return JudgeOut(items=[])


def test_judge_recall_matches_production_text(repo, policy):
    runner = _runner(repo, policy)
    runner.collect()
    items = s1_items(repo, runner.ledger.list_issues("S1"))
    bad = next(it for it in items if it["label"] == "bad")
    good = next(it for it in items if it["label"] == "good")
    assert "合同负债" in bad["recall_text"]                    # 坏主张证据的指标名进入召回文本
    assert "合同负债" not in good["recall_text"]               # 好主张引用另一份证据

    claim_only = LessonCard(id="L-001", title="t", applies_to="a", tags=["说明"], lesson="先排除其他解释")
    h = _JudgeH()
    _judge(h, [bad], [claim_only])                             # 主张原文含「说明」，召回输入不含
    assert "相关方法提示" not in h.user                        # 线上永远不会注入的卡，评测也不能借它得分
    metric = LessonCard(id="L-002", title="t", applies_to="a", tags=["合同负债"], lesson="先查打款政策")
    h2 = _JudgeH()
    _judge(h2, [bad], [metric])
    assert "相关方法提示" in h2.user                           # 指标名命中的卡照常注入


# ---------- #11/#12 S2 主片与成本估计 ----------

def test_s2_primary_is_evolution_and_tokens_estimated(repo, policy):
    svc = make_service(repo, policy, RSILLM())
    runner = RSIRunner(svc.deps, svc.deps.ledger, svc.deps.rsi)
    now = datetime.now(timezone.utc)
    flip = {"quick": 0.1, "standard": 0.46, "deep": 0.44}
    for k in range(3):
        rows = [{"question": "复核业绩", "label": "standard", "route": "deep", "source": "abstain", "probs": flip,
                 "abstain": True, "correct": False},
                {"question": "估值分歧大", "label": "deep", "route": "deep", "source": "model",
                 "probs": {"quick": 0.05, "standard": 0.15, "deep": 0.8}, "abstain": False, "correct": True}]
        repo.db.eval_runs.insert_one({"run_id": f"rt_{k}", "type": "routing", "status": "done",
                                      "created_at": now - timedelta(minutes=k), "rows": rows})
    runner.collect()
    run = runner.evaluate(runner.propose("S2").run_id)
    assert run.eval["primary"] == "evolution"                  # 本次候选要修的错判必须进噪声/回归门
    assert run.eval["arms"]["base"]["tokens"] > 0              # 按 route 预算估算，成本门不再恒过
    assert run.eval["arms"]["cand"]["tokens"] > 0


def test_release_addresses_only_resolved_issues(repo, policy):
    svc = make_service(repo, policy, RSILLM())
    runner = RSIRunner(svc.deps, svc.deps.ledger, svc.deps.rsi)
    ledger = runner.ledger
    ledger.bootstrap(svc.deps.policy)
    now = datetime.now(timezone.utc)
    repo.db.eval_runs.insert_one({"run_id": "rt_rel", "type": "routing", "status": "done", "created_at": now, "rows": [
        {"question": "q_good", "label": "deep", "route": "standard", "source": "model",
         "probs": {"quick": 0.05, "standard": 0.05, "deep": 0.9}, "abstain": False, "correct": False},
        {"question": "q_bad", "label": "quick", "route": "deep", "source": "model",
         "probs": {"quick": 0.1, "standard": 0.45, "deep": 0.45}, "abstain": False, "correct": False},
    ]})
    good = _issue("S2", "route_mislabel", "q_good 应走 deep", ["证据"], "判别器校准偏保守需要调整",
                  {"question": "q_good", "label": "deep", "route": "standard"}, "q_good")
    bad = _issue("S2", "route_mislabel", "q_bad 应走 quick", ["证据"], "该类问题系统性偏向错误标签",
                 {"question": "q_bad", "label": "quick", "route": "deep"}, "q_bad")
    ledger.add_issue(good)
    ledger.add_issue(bad)
    base = ledger.active_asset("routing")
    run = Run(run_id="rsi_s2rel", scenario="S2", kind="routing", issue_ids=[good.id, bad.id], base_version=base.version,
              content=dict(base.content), diff={"params": {}}, change_reason="逐条确认修复的测试候选",
              direction="S2:test", status="approved", created_at=now, updated_at=now)
    ledger.insert_run(run)
    runner.release(run.run_id, "owner")
    assert ledger.get_issues([good.id])[0].status == "addressed"     # 回放确认判对
    assert ledger.get_issues([bad.id])[0].status == "open"           # 仍判错：不能被整批标记
    release_event = next(e for e in ledger.list_events(limit=100) if e["action"] == "release")
    assert "1/2" in release_event["detail"]


# ---------- #12 S3 成本代理 ----------

def _ev(bt, ct, base_mean=0.5, cand_mean=0.7):
    return {"primary": "evolution", "arms": {
        "base": {"tokens": 0, "cost_proxy": bt, "slices": {"evolution": {"mean": base_mean, "spread": 0.0, "runs": [1, 2, 3]}}},
        "cand": {"tokens": 0, "cost_proxy": ct, "slices": {"evolution": {"mean": cand_mean, "spread": 0.0, "runs": [1, 2, 3]}}}}}


def test_cost_gate_uses_kept_proxy_when_no_tokens():
    good = cost_gate(None, _ev(10, 13), None)                  # 线索量 +30%，质量 +40%：不超收益
    assert good.passed and "线索量" in good.detail
    bad = cost_gate(None, _ev(10, 30), None)                   # 线索量 +200% 远超质量收益
    assert not bad.passed and "线索量" in bad.detail


# ---------- #13 对照汇总不因 None 崩溃 ----------

def test_summarize_all_failed_route_is_insufficient():
    v = summarize([{"route": "standard", "metrics": {"score": 0.7, "tokens": 100}},
                   {"route": "deep", "error": "boom"}])["verdict"]
    assert v["decision"] == "insufficient" and v["failed_cases"] == 1
