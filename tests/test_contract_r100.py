"""R100a/R102a 回归：代码前后边界为 Unicode 数字；敏感问题被空白/零宽/连字符拆开的代码
用「相邻数字对桥接」兜底找回，且兜底不把金额/日期拼成代码、不参与模型/名称的优先级。"""
import json
import threading
import time

from langgraph.checkpoint.memory import InMemorySaver

from advisor.contracts import ResearchRequest
from advisor.graph.deps import build_deps
from advisor.graph.nodes import _codes_of, _codes_of_loose, _codes_with_dropped, _name_evidence
from advisor.graph.runner import ResearchService
from advisor.skills.research.earnings_review import IntakeOut
from tests.fakes import CODE, NAME, FakeDecision, FakeLLM, FakeSource, FakeWebTools, as_of

NBSP = " "
ZWSP = "\u200b"


def _svc(repo, policy, **kw):
    decision = kw.pop("decision", None)
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools(), decision=decision)
    return ResearchService(deps, InMemorySaver(), **kw)


def _run(repo, policy, question: str) -> dict:
    svc = _svc(repo, policy)
    tid = svc.start(ResearchRequest(question=question, as_of=as_of(), route="quick"), background=False)
    return repo.get_task(tid)


def test_non_ascii_digit_adjacency_is_not_a_code():
    """前后边界是 Unicode 数字：阿拉伯-印度数字紧邻的 6 位 ASCII 片段不能被截成代码。"""
    assert _codes_of("看看٥600519的财报") == set()
    assert _codes_of("看看600519٥的财报") == set()
    assert _codes_of("看看 600519 的财报") == {"600519"}


def test_non_ascii_digit_adjacent_not_researched(repo, policy):
    """端到端：'٥600519' 不识别为标的。"""
    task = _run(repo, policy, "看看٥600519的财报")
    assert task["status"] == "failed"
    assert "未能识别" in (task.get("error") or "")
    assert task["request"].get("ticker") is None


def test_loose_parse_bridges_pairs_but_rejects_money_and_dates():
    """宽松兜底只在『相邻数字对拼接恰为代码』时取候选：金额（元/万/亿/%）与日期对不得拼接；
    零宽/连字符/空白拆开的真代码要能找回；多个候选不猜。"""
    assert _codes_of_loose("我600" + NBSP + "519股成本价30元怎么办？") == {"600519"}
    assert _codes_of_loose("我600-519股成本价30元怎么办？") == {"600519"}
    assert _codes_of_loose("我600" + ZWSP + "519股成本价30元怎么办？") == {"600519"}
    assert _codes_of_loose("我的房贷余额600" + NBSP + "519元怎么办？") == set()          # 余额不是标的
    assert _codes_of_loose("我持仓\n2024-12-31\n2025-01-02的流水怎么看？") == set()     # 日期对不拼
    assert _codes_of_loose("我600" + NBSP + "519股和300" + NBSP + "750股都有") == {"600519", "300750"}
    assert _codes_of_loose("我持仓\n2024-12-31\n600\n519股成本价30元") == {"600519"}      # 不整串吞码


def _consent_task(repo, policy, question: str) -> dict:
    svc = _svc(repo, policy)
    tid = svc.start(ResearchRequest(question=question, as_of=as_of(), route="quick"), background=False)
    task = repo.get_task(tid)
    state = svc.graph.get_state({"configurable": {"thread_id": tid}}).values
    return {**task, "_state_ticker": state.get("ticker")}


def test_sensitive_split_code_recovered_and_state_consistent(repo, policy):
    """被 NBSP/换行拆开的真代码：宽松兜底找回，request.ticker 与图内 state.ticker 一致，
    敏感问题在 authorize 必然停在 waiting_approval。"""
    for q in ("我600" + NBSP + "519股成本价30元怎么办？", "我600\n519股成本价30元怎么办？"):
        task = _consent_task(repo, policy, q)
        assert task["status"] == "waiting_approval", task.get("error")
        assert task["request"]["ticker"] == "600519"
        assert task["_state_ticker"] == "600519"
        assert "600519" in task["interrupt"]["message"]           # 审批文案展示推断出的标的


def test_sensitive_amount_is_not_bound_as_code(repo, policy):
    """R102a important：敏感问题里的房贷/存款金额不得被拼成真实股票代码（600519/600000）。"""
    task = _run(repo, policy, "我的房贷余额600" + NBSP + "519元怎么办？")
    assert task["status"] == "failed"
    assert "未能识别" in (task.get("error") or "")
    assert task["request"].get("ticker") is None


def test_sensitive_date_pair_not_bound(repo, policy):
    task = _run(repo, policy, "我持仓\n2024-12-31\n2025-01-02的流水怎么看？")
    assert task["status"] == "failed"
    assert task["request"].get("ticker") is None


def test_loose_ambiguous_codes_not_guessed(repo, policy):
    """两个拆分候选（都带『股』）不硬猜：按未识别失败，不绑定其中之一。"""
    task = _run(repo, policy, "我600" + NBSP + "519股和300" + NBSP + "750股都有，怎么办？")
    assert task["status"] == "failed"
    assert task["request"].get("ticker") is None


def test_queued_cancel_backfills_nbsp_split_code(repo, policy):
    """排队中被取消（intake 未执行）：runner 落盘也要用同一宽松兜底回填拆开的代码，重试可用。"""
    svc = _svc(repo, policy, max_workers=1)
    gate = threading.Event()
    svc._pool.submit(lambda: gate.wait(10))
    try:
        tid = svc.start(ResearchRequest(question="我600" + NBSP + "519股成本价30元怎么办？",
                                        as_of=as_of(), route="quick"), background=True)
        task = repo.get_task(tid)
        assert task["status"] == "queued"
        assert task["request"]["ticker"] == "600519"
        svc.deps.repo.request_cancel(tid)
    finally:
        gate.set()
    deadline = time.time() + 10
    while time.time() < deadline and repo.get_task(tid)["status"] not in ("cancelled", "done", "failed"):
        time.sleep(0.02)
    assert repo.get_task(tid)["status"] == "cancelled"


def test_explicit_ticker_conflict_is_visible(repo, policy):
    """页面/请求指定的标的与问题点名的公司不一致：intake 事件必须说明（small 分类器路径也可见）。"""
    repo.db.dossiers.insert_one({"key": CODE, "ticker": CODE, "name": NAME, "version": 1,
                                 "as_of": "2026-12-01", "status": "committed"})
    svc = _svc(repo, policy)
    tid = svc.start(ResearchRequest(question=f"{NAME} 最近怎么样？", as_of=as_of(), route="quick",
                                    ticker="000001"), background=False)
    task = repo.get_task(tid)
    notes = [e["message"] for e in task["events"] if e["node"] == "intake"]
    assert any("指定标的是 000001" in m and NAME in m for m in notes)


def test_strict_money_forms_are_not_codes():
    """严格解析同样要排除金额/小数：'600519.23元'、'600 519 万元'、'600519美元' 不是标的。"""
    assert _codes_of("我房贷600 519.23元怎么办？") == set()
    assert _codes_of("我的房贷余额600519.23元怎么办？") == set()
    assert _codes_of("我存款600519美元怎么办？") == set()
    assert _codes_of("我持仓600519股成本价30元") == {"600519"}          # 『股』仍是确定性例外
    assert _codes_of("复核 600519 半年报业绩") == {"600519"}


def test_strict_amount_not_bound_end_to_end(repo, policy):
    """端到端：六位数金额不落盘成 ticker（修复前会静默研究 600519）。"""
    task = _run(repo, policy, "我的房贷余额600519.23元怎么办？")
    assert task["status"] == "failed"
    assert task["request"].get("ticker") is None


def test_amount_does_not_override_name_resolution(repo, policy):
    """金额形态的六位数不得压制名称解析：问题点名了已跟踪公司时按公司研究。"""
    repo.db.dossiers.insert_one({"key": CODE, "ticker": CODE, "name": NAME, "version": 1,
                                 "as_of": "2026-12-01", "status": "committed"})
    task = _run(repo, policy, f"我持有{NAME}，房贷300750万元怎么办？")
    assert task["status"] == "waiting_approval", task.get("error")     # 敏感问题弹 consent
    assert task["request"]["ticker"] == CODE                            # 不是金额 300750


def test_loose_unit_after_space_and_currency():
    """宽松兜底的单位判断要跳过空白：'我房贷 300\\n750 元'（R102a 原句）不能拼成 300750；
    美元/手/份 等币种与数量单位同样排除。"""
    assert _codes_of_loose("我房贷 300\n750 元，压力大怎么办？") == set()
    assert _codes_of_loose("我的房贷余额600" + NBSP + "519 元怎么办？") == set()
    assert _codes_of_loose("我的存款600" + NBSP + "519美元怎么办？") == set()
    assert _codes_of_loose("我持仓600" + NBSP + "519手") == set()
    assert _codes_of_loose("我买了600\n519份基金") == set()
    assert _codes_of_loose("我600" + NBSP + "519股成本价30元怎么办？") == {"600519"}   # 『股』例外


def test_loose_hyphen_bridge_rejects_month_year():
    """连字符桥接只认 3+3：'03-2025'、'01-2345' 这类月-年/编号不能拼成 032025/012345。"""
    assert _codes_of_loose("03-2025 年报") == set()
    assert _codes_of_loose("编号01-2345的合同") == set()
    assert _codes_of_loose("我600-519股成本价30元怎么办？") == {"600519"}


def test_loose_rejects_unicode_digit_adjacency():
    """宽松兜底与严格解析同口径：非 ASCII 数字紧邻不算独立代码。"""
    assert _codes_of_loose("我持仓٥600 519股") == set()
    assert _codes_of_loose("我持仓600 519٥股") == set()


def test_small_classifier_path_uses_loose_last_resort(repo, policy):
    """small 分类器路径：代码/名称都解析不到时走大模型，再兜底到宽松拆码。"""
    decision = FakeDecision()
    svc = _svc(repo, policy, decision=decision)
    tid = svc.start(ResearchRequest(question="我600" + NBSP + "519股成本价30元怎么办？",
                                    as_of=as_of(), route="quick"), background=False)
    task = repo.get_task(tid)
    assert task["status"] == "waiting_approval"
    assert task["request"]["ticker"] == "600519"


def test_ambiguous_loose_candidates_listed_in_event(repo, policy):
    """多拆码候选不猜，并要在 intake 事件里列出候选（便于用户改写）。"""
    svc = _svc(repo, policy)
    tid = svc.start(ResearchRequest(question="我600" + NBSP + "519股和300" + NBSP + "750股都有，怎么办？",
                                    as_of=as_of(), route="quick"), background=False)
    task = repo.get_task(tid)
    assert task["status"] == "failed"
    notes = [e["message"] for e in task["events"] if e["node"] == "intake"]
    assert any("多个候选" in m and "[已脱敏]" in m for m in notes)   # 敏感问题：候选数字不落盘
    assert "600519" not in " ".join(notes) and "300750" not in " ".join(notes)


def test_explicit_ticker_conflict_with_code_is_visible(repo, policy):
    """显式标的与问题里的严格代码冲突：写 intake 事件。"""
    svc = _svc(repo, policy)
    tid = svc.start(ResearchRequest(question="600519最近怎么样？", as_of=as_of(), route="quick",
                                    ticker="000001"), background=False)
    task = repo.get_task(tid)
    notes = [e["message"] for e in task["events"] if e["node"] == "intake"]
    assert any("指定标的是 000001" in m and "600519" in m for m in notes)


def test_money_word_prefixes_do_not_eat_codes():
    """R106a important：金额守卫按单位首字误伤——手机/手续费/千亿/元器件/万科A 不是金额。"""
    assert _codes_of("002475手机业务占比") == {"002475"}
    assert _codes_of("600519手续费多少") == {"600519"}
    assert _codes_of("600519千亿市值怎么看") == {"600519"}
    assert _codes_of("600519元器件业务和002475比") == {"600519", "002475"}
    assert _codes_of("000002万科A和600519比") == {"000002", "600519"}
    assert _codes_of("600519元") == set()                     # 真金额仍排除
    assert _codes_of("600519万元营收") == set()


def test_money_word_prefix_multi_code_not_silently_rerouted(repo, policy):
    """多代码问题里一只代码后接『万』等词首：不得被金额守卫删掉后静默研究另一只。"""
    task = _run(repo, policy, "000002万科A和600519比怎么样")
    assert task["status"] == "failed"
    assert "多个股票代码" in (task.get("error") or "")
    assert task["request"].get("ticker") is None


def test_money_comma_decimal_and_currency_prefix():
    """逗号小数、小数点后空格、货币前缀都算金额；逗号分隔的代码列表仍是代码。"""
    assert _codes_of("600519,23元") == set()
    assert _codes_of("600519,300750") == {"600519", "300750"}
    assert _codes_of("600519. 23元") == set()
    assert _codes_of("￥600519") == set()
    assert _codes_of("人民币600519") == set()


def test_loose_not_used_when_multiple_strict_codes(repo, policy):
    """严格多代码时宽松候选不得抢选：仍按『多个股票代码』显式失败。"""
    task = _run(repo, policy, "我持仓600 519股，也持有000001，怎么办？")
    assert task["status"] == "failed"
    assert "多个股票代码" in (task.get("error") or "")
    assert task["request"].get("ticker") is None


def test_loose_rejects_spaced_month_year():
    assert _codes_of_loose("03 2025 年报") == set()
    assert _codes_of_loose("2025 03 年报") == set()


def test_ticker_source_recorded(repo, policy):
    """回填/识别的 ticker 要记录来源：explicit / code / loose（前端据此标注推断）。"""
    svc = _svc(repo, policy)
    tid = svc.start(ResearchRequest(question="复核 600519 半年报业绩", as_of=as_of(), route="quick",
                                    ticker="000001"), background=False)
    assert repo.get_task(tid)["request"]["ticker_source"] == "explicit"
    tid = svc.start(ResearchRequest(question="复核 600519 半年报业绩", as_of=as_of(), route="quick"),
                    background=False)
    assert repo.get_task(tid)["request"]["ticker_source"] == "code"
    tid = svc.start(ResearchRequest(question="我600" + NBSP + "519股成本价30元怎么办？", as_of=as_of(),
                                    route="quick"), background=False)
    assert repo.get_task(tid)["request"]["ticker_source"] == "loose"


def test_explicit_conflicts_with_code_and_name_both_logged(repo, policy):
    """explicit 与问题里的代码、公司名同时冲突：两条 intake 事件都要写（不再 elif 吞掉）。"""
    repo.db.dossiers.insert_one({"key": CODE, "ticker": CODE, "name": NAME, "version": 1,
                                 "as_of": "2026-12-01", "status": "committed"})
    svc = _svc(repo, policy)
    tid = svc.start(ResearchRequest(question=f"{NAME} 和 600519 比怎么样", as_of=as_of(), route="quick",
                                    ticker="000001"), background=False)
    notes = [e["message"] for e in repo.get_task(tid)["events"] if e["node"] == "intake"]
    assert any("出现代码" in m and "600519" in m for m in notes)
    assert any("点名的是" in m and NAME in m for m in notes)

class _EchoTickerLLM(FakeLLM):
    """intake 返回固定 ticker 的模型桩（其余角色沿用 FakeLLM）。"""

    def __init__(self, ticker: str, company: str = ""):
        super().__init__()
        self._ticker = ticker
        self._company = company

    def json(self, role, system, user, schema, max_tokens=None):
        if schema is IntakeOut:
            return IntakeOut(company_name=self._company, ticker=self._ticker, intent="research", focus=""), 50
        return super().json(role, system, user, schema, max_tokens)


def _run_llm(repo, policy, llm, question: str, **req_kw) -> dict:
    deps = build_deps(repo, llm, policy, source=FakeSource(), web_tools=FakeWebTools())
    svc = ResearchService(deps, InMemorySaver())
    tid = svc.start(ResearchRequest(question=question, as_of=as_of(), route="quick", **req_kw), background=False)
    return repo.get_task(tid)


def test_chinese_syntax_after_code_kept(repo, policy):
    """代码后接中文句法的逗号/句点数字（，2季度 / ，23年 / . 2024年报）不是小数。"""
    assert _codes_of("600519，2季度营收怎么样") == {"600519"}
    assert _codes_of("600519，23年营收怎么样") == {"600519"}
    assert _codes_of("600519,23号公告") == {"600519"}
    assert _codes_of("600519. 2024年报怎么样") == {"600519"}
    assert _codes_of("600519,23元") == set()              # 真小数金额仍排除
    task = _run(repo, policy, "600519，23年营收怎么样")
    assert task["status"] == "done", task.get("error")
    assert task["request"]["ticker"] == "600519"


def test_bare_wan_amount_and_word_kept():
    """裸万在后接句末时是金额；后接汉字（万营收）不吞代码。"""
    assert _codes_of("营收600519万") == set()
    assert _codes_of("600519万营收") == {"600519"}


def test_rmb_prefix_boundary():
    """『人民币』作为独立词才算货币前缀：数字人民币/离岸人民币 的词尾不吞代码。"""
    assert _codes_of("人民币600519") == set()
    assert _codes_of("数字人民币600519") == {"600519"}


def test_dropped_candidate_is_visible_in_event(repo, policy):
    """被金额守卫丢弃的六位数（元件/元旦等词首误伤）必须在 intake 事件里可见，不得静默改标的。"""
    task = _run(repo, policy, "002475元件业务和600519比怎么样")
    assert task["status"] == "done", task.get("error")
    notes = [e["message"] for e in task["events"] if e["node"] == "intake"]
    assert any("疑似金额" in m and "002475" in m for m in notes)


def test_model_can_rescue_dropped_candidate(repo, policy):
    """模型返回被金额守卫丢弃的候选时应当被采纳（此前 gate 把补救通道封死）。"""
    task = _run_llm(repo, policy, _EchoTickerLLM("002475"), "002475元件业务和600519比怎么样")
    assert task["request"]["ticker"] == "002475"
    assert task["request"]["ticker_source"] == "model"


def test_visible_code_source_is_code_not_model(repo, policy):
    """模型回显问题中可见的代码：来源应为 code，而不是 model（否则前端误标推断）。"""
    task = _run_llm(repo, policy, _EchoTickerLLM("600519"), "复核 600519 半年报业绩")
    assert task["request"]["ticker_source"] == "code"
    # 模型给出原文没有的标的（有公司名支撑）才算推断
    task2 = _run_llm(repo, policy, _EchoTickerLLM("600519", company="贵州茅台"), "帮我看看贵州茅台这家公司的近况")
    assert task2["request"]["ticker_source"] == "model"


def test_small_classifier_path_source_code(repo, policy):
    """small 分类器路径回显本地代码：来源应为 code（此前统一记成 model）。"""
    svc = _svc(repo, policy, decision=FakeDecision())
    tid = svc.start(ResearchRequest(question="复核 600519 半年报业绩", as_of=as_of(), route="quick"),
                    background=False)
    assert repo.get_task(tid)["request"]["ticker_source"] == "code"


def test_retry_preserves_inferred_source(repo, policy):
    """重试可带回来源：loose/model 的推断标注不因『按原参数重新发起』而丢失。"""
    task = _run_llm(repo, policy, FakeLLM(), "复核 600519 半年报业绩",
                    ticker="600519", ticker_source="loose")
    assert task["request"]["ticker_source"] == "loose"

def test_wanyi_and_qianwanyuan_are_money():
    """复合单位先于裸乘数：万亿/千万元 仍是金额（修复前被裸万分支短路成代码）。"""
    assert _codes_of("营收600519万亿") == set()
    assert _codes_of("600519千万元营收") == set()
    assert _codes_of("600519万亿市值") == set()          # 与 base 一致：该形态按金额处理
    assert _codes_of("600519千亿市值怎么看") == {"600519"}   # 千亿 仍是描述，保留代码


def test_decimal_with_space_before_unit():
    """≥3 位小数：单位前有空白/零宽也要跳过再判断。"""
    assert _codes_of("600519.123 元") == set()
    assert _codes_of("600519.12百万元") == set()


def test_digit_context_starter_precision():
    """'人/美/港' 单字不再当金额词首；'百' 补齐。"""
    assert _codes_of("600519,23人生感悟") == {"600519"}
    assert _codes_of("约合人民币600519") == set()
    assert _codes_of("折合人民币600519") == set()
    assert _codes_of("数字人民币600519") == {"600519"}


def test_model_infers_name_when_only_amount_present(repo, policy):
    """问题里只有六位数金额时，模型按公司名推断出的代码仍要被接受（not codes 语义）。"""
    task = _run_llm(repo, policy, _EchoTickerLLM("300750", company="宁德时代"), "我在看宁德时代，去年营收600519万元")
    assert task["status"] in ("waiting_approval", "done", "running"), task.get("error")
    assert task["request"]["ticker"] == "300750"
    assert task["request"]["ticker_source"] == "model"


def test_model_choice_of_dropped_is_visible(repo, policy):
    """模型选中疑似金额候选时要写事件（可见但允许模型裁决）。"""
    task = _run_llm(repo, policy, _EchoTickerLLM("300750"), "600519去年营收300750万元")
    assert task["request"]["ticker"] == "300750"
    notes = [e["message"] for e in task["events"] if e["node"] == "intake"]
    assert any("疑似金额" in m and "300750" in m for m in notes)


def test_small_classifier_defers_to_llm_when_dropped(repo, policy):
    """有疑似金额候选时 small 路径让位给大模型：被守卫误伤的代码可被模型补救。"""
    deps = build_deps(repo, _EchoTickerLLM("002475"), policy, source=FakeSource(), web_tools=FakeWebTools(),
                      decision=FakeDecision())
    svc = ResearchService(deps, InMemorySaver())
    tid = svc.start(ResearchRequest(question="002475元件业务和600519比怎么样", as_of=as_of(), route="quick"),
                    background=False)
    task = repo.get_task(tid)
    assert task["request"]["ticker"] == "002475"
    notes = [e["message"] for e in task["events"] if e["node"] == "intake"]
    assert any("疑似金额" in m for m in notes)


def test_sensitive_dropped_event_masks_digits(repo, policy):
    """R110a important：敏感问题的『疑似金额』事件不得把已脱敏的数字写回任务事件。"""
    task = _run(repo, policy, "我的房贷600519元怎么办")
    assert task["status"] == "failed"
    texts = " ".join(e["message"] for e in task["events"])
    assert "600519" not in texts
    assert "疑似金额" in texts and "已脱敏" in texts


def test_dropped_event_wording_without_code(repo, policy):
    """没有采用任何代码时文案用『出现』而不是『另有』。"""
    task = _run(repo, policy, "营收600519万元级公司怎么样")
    assert task["status"] == "failed"
    notes = [e["message"] for e in task["events"] if e["node"] == "intake"]
    assert any("出现疑似金额" in m for m in notes)

def test_model_ticker_without_company_evidence_rejected(repo, policy):
    """没有任何公司名/代码依据时，模型给出的代码不得直接开跑（防幻觉标的）。"""
    task = _run_llm(repo, policy, _EchoTickerLLM("000001"), "去年营收600000万元，这只股票怎么样")
    assert task["status"] == "failed"
    assert task["request"].get("ticker") is None

def test_sensitive_whole_document_masked(repo, policy):
    """R112a important：敏感任务的整个任务文档（request/error/events/interrupt）不得复现被脱敏数字。"""
    task = _run_llm(repo, policy, _EchoTickerLLM("000001"),
                    "我持仓600" + NBSP + "519股和000001都有，怎么办？")
    doc = json.dumps(task, ensure_ascii=False, default=str)
    assert "600519" not in doc                       # 拆码候选在事件里也被遮蔽
    notes = [e["message"] for e in task["events"] if e["node"] == "intake"]
    assert any("拆码候选" in m and "[已脱敏]" in m for m in notes)


def test_sensitive_multi_code_error_masked(repo, policy):
    """敏感多代码失败：error 里的代码列表也要遮蔽。"""
    task = _run(repo, policy, "我持有600519和000001各1000股，该怎么办？")
    doc = json.dumps(task, ensure_ascii=False, default=str)
    assert "600519" not in doc and "000001" not in doc
    assert task["status"] == "failed" and "[已脱敏]" in (task.get("error") or "")


def test_decimal_unit_table_complete():
    """小数/逗号长数字紧跟完整单位表（万股/份/千万元/万元）都算金额；百姓 不是金额词首。"""
    assert _codes_of("600519.123万股") == set()
    assert _codes_of("600519.123" + ZWSP + "元") == set()
    assert _codes_of("600519,123元") == set()
    assert _codes_of("600519,23百姓生活") == {"600519"}
    assert _codes_of("600519.12百万元") == set()

def test_short_name_question_full_name_model_accepted(repo, policy):
    """简称提问（茅台）+ 模型返回规范全称：弱名称证据应放行（修复前被逐字子串门槛误拒）。"""
    task = _run_llm(repo, policy, _EchoTickerLLM("600519", company="贵州茅台"), "茅台最近怎么样，值得研究吗")
    assert task["status"] in ("waiting_approval", "done", "running"), task.get("error")
    assert task["request"]["ticker"] == "600519"
    assert task["request"]["ticker_source"] == "model"
    notes = [e["message"] for e in task["events"] if e["node"] == "intake"]
    assert any("按公司名推断" in m for m in notes)


def test_non_sensitive_split_code_as_model_evidence(repo, policy):
    """非敏感问题里被 NBSP 拆开的代码：模型回填正确时应作为证据采纳（修复前被拒成未能识别）。"""
    task = _run_llm(repo, policy, _EchoTickerLLM("600519"), "帮我看看600" + NBSP + "519这只股票")
    assert task["status"] in ("waiting_approval", "done", "running"), task.get("error")
    assert task["request"]["ticker"] == "600519"
    notes = [e["message"] for e in task["events"] if e["node"] == "intake"]
    assert any("按拆码推断" in m for m in notes)


def test_name_hits_conflict_prefers_local_name(repo, policy):
    """问题点名已跟踪公司而模型给另一只：不采纳模型码，回退本地档案名（修复前模型码覆盖名称）。"""
    repo.db.dossiers.insert_one({"key": CODE, "ticker": CODE, "name": NAME, "version": 1,
                                 "as_of": "2026-12-01", "status": "committed"})
    task = _run_llm(repo, policy, _EchoTickerLLM("000001", company="五粮液"),
                    f"我在看{NAME}，去年营收600519万元")
    assert task["request"]["ticker"] == CODE


def test_comma_code_list_with_component_word_kept(repo, policy):
    """逗号分隔的代码列表：第二个代码后接『元器件』也不得把第一个代码当千分位删掉。"""
    assert _codes_with_dropped("600519,002475元器件业务对比")[0] == {"600519", "002475"}
    task = _run(repo, policy, "600519,002475元器件业务对比")
    assert task["status"] == "failed"
    assert "多个股票代码" in (task.get("error") or "")


def test_unit_table_extras_and_yuan_exception():
    assert _codes_with_dropped("600519.123元器件")[0] == {"600519"}   # 元器件不是金额单位
    assert _codes_of("600519.123万份") == set()
    assert _codes_of("600519.123手") == set()
    assert _codes_of("600519.123万美元") == set()


def test_small_path_decide_sees_masked_question(repo, policy):
    """small 分类器路径的决策调用也必须只看脱敏文本（敏感多行金额不出域到决策服务）。"""
    decision = FakeDecision()
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools(), decision=decision)
    svc = ResearchService(deps, InMemorySaver())
    svc.start(ResearchRequest(question="我持仓600519股，成本价30元怎么办", as_of=as_of(), route="quick"),
              background=False)
    assert decision.calls
    assert "[已脱敏]" in decision.calls[0]
    assert "600519" not in decision.calls[0]


def test_sensitive_failure_error_has_no_digits(repo, policy):
    """敏感失败的静态错误文案不得带示例数字。"""
    task = _run(repo, policy, "我的房贷余额600519元怎么办")
    assert task["status"] == "failed"
    assert "600519" not in (task.get("error") or "")

def test_generic_company_words_not_evidence(repo, policy):
    """通用行业词不构成模型码证据：银行板块/中国经济类问题不得被幻觉 ticker 过关。"""
    task = _run_llm(repo, policy, _EchoTickerLLM("601398", company="工商银行"), "银行板块最近怎么走")
    assert task["status"] == "failed" and task["request"].get("ticker") is None
    task2 = _run_llm(repo, policy, _EchoTickerLLM("601318", company="中国平安"), "中国经济怎么看")
    assert task2["status"] == "failed" and task2["request"].get("ticker") is None


def test_name_hits_priority_over_overlapping_evidence(repo, policy):
    """问题点名已跟踪公司时本地档案名优先：即使模型公司名与问题有公共字也不采纳其代码。"""
    repo.db.dossiers.insert_one({"key": "000001", "ticker": "000001", "name": "平安银行", "version": 1,
                                 "as_of": "2026-12-01", "status": "committed"})
    task = _run_llm(repo, policy, _EchoTickerLLM("601318", company="中国平安"), "平安银行最近怎么样")
    assert task["request"]["ticker"] == "000001"
    notes = [e["message"] for e in task["events"] if e["node"] == "intake"]
    assert any("模型给出 601318" in m for m in notes)


def test_multi_code_with_tracked_name_fails_explicitly(repo, policy):
    """多严格码 + 档案名共现：不得用名称兜底静默选定，仍按多个股票代码失败。"""
    repo.db.dossiers.insert_one({"key": CODE, "ticker": CODE, "name": NAME, "version": 1,
                                 "as_of": "2026-12-01", "status": "committed"})
    task = _run(repo, policy, f"{NAME}、600519和000001比怎么样")
    assert task["status"] == "failed"
    assert "多个股票代码" in (task.get("error") or "")
    assert task["request"].get("ticker") is None


def test_name_fallback_syncs_company_name(repo, policy):
    """回退本地档案名时公司名同步为档案名（不再出现 600999 配模型自报『五粮液』）。"""
    repo.db.dossiers.insert_one({"key": CODE, "ticker": CODE, "name": NAME, "version": 1,
                                 "as_of": "2026-12-01", "status": "committed"})
    task = _run_llm(repo, policy, _EchoTickerLLM("000001", company="五粮液"), f"{NAME}最近怎么样")
    assert task["request"]["ticker"] == CODE
    notes = [e["message"] for e in task["events"] if e["node"] == "intake"]
    assert any(f"（{NAME}）" in m for m in notes)
    assert not any("五粮液" in m for m in notes)


def test_decimal_comma_full_code_list_pinned():
    """'600519,300750元' 的取舍钉死：第一码保留为代码、第二码按金额候选（事件可见）。"""
    kept, dropped = _codes_with_dropped("600519,300750元")
    assert kept == {"600519"} and dropped == {"300750"}


def test_small_decide_masks_multiline_amounts(repo, policy):
    """small 决策调用只看整段脱敏文本：多行敏感问题里的数字不出域。"""
    decision = FakeDecision()
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=FakeWebTools(), decision=decision)
    svc = ResearchService(deps, InMemorySaver())
    svc.start(ResearchRequest(question="我的持仓600519股，\n成本价30元，\n怎么办", as_of=as_of(), route="quick"),
              background=False)
    assert decision.calls
    text = decision.calls[0]
    assert "[已脱敏]" in text and "600519" not in text and "30" not in text.replace("[已脱敏]", "")

def test_small_path_multi_code_not_shortcut_by_name(repo, policy):
    """small 分类器路径不得把档案名当捷径绕过『多码必须显式失败』。"""
    repo.db.dossiers.insert_one({"key": CODE, "ticker": CODE, "name": NAME, "version": 1,
                                 "as_of": "2026-12-01", "status": "committed"})
    svc = _svc(repo, policy, decision=FakeDecision())
    tid = svc.start(ResearchRequest(question=f"{NAME}、600519和000001比怎么样", as_of=as_of(), route="quick"),
                    background=False)
    task = repo.get_task(tid)
    assert task["status"] == "failed"
    assert "多个股票代码" in (task.get("error") or "")
    assert task["request"].get("ticker") is None


def test_multi_code_error_mentions_tracked_names(repo, policy):
    """多码失败文案同时列出问题点名的档案名。"""
    repo.db.dossiers.insert_one({"key": CODE, "ticker": CODE, "name": NAME, "version": 1,
                                 "as_of": "2026-12-01", "status": "committed"})
    task = _run(repo, policy, f"{NAME}、600519和000001比怎么样")
    assert task["status"] == "failed"
    assert "多个股票代码" in (task.get("error") or "") and NAME in (task.get("error") or "")


def test_industry_words_not_evidence_anywhere():
    """通用行业词（汽车/保险/石油石化近名/中国中×）在任何分支都不作数。"""
    assert not _name_evidence("长城汽车", "汽车板块最近怎么走")
    assert not _name_evidence("新华保险", "保险板块最近怎么样")
    assert not _name_evidence("中国石化", "中国石油最近怎么样")      # 3 字片段『中国石』被停用词过滤
    assert not _name_evidence("中国中免", "中国中车最近怎么样")
    assert not _name_evidence("宁德时代", "宁德最近怎么样")           # 2 字前缀不再作证据（安全失败）
    assert _name_evidence("中国石油", "中石油最近怎么样")             # 『中国X→中X』惯用简称
    assert _name_evidence("360", "360这家公司怎么样")                 # 非 6 位纯数字名可用
    assert not _name_evidence("360", "有360万元闲钱怎么配置")          # 金额语境不是名称
    assert not _name_evidence("360", "1360万")


def test_industry_evidence_e2e_fails(repo, policy):
    task = _run_llm(repo, policy, _EchoTickerLLM("601633", company="长城汽车"), "汽车板块最近怎么走")
    assert task["status"] == "failed" and task["request"].get("ticker") is None
    task2 = _run_llm(repo, policy, _EchoTickerLLM("600028", company="中国石化"), "中国石油最近怎么样")
    assert task2["status"] == "failed" and task2["request"].get("ticker") is None


def test_multiple_tracked_names_fail_explicitly(repo, policy):
    """同句点名多家已跟踪公司：显式失败并列出名字，不再 next() 静默选定。"""
    repo.db.dossiers.insert_one({"key": CODE, "ticker": CODE, "name": NAME, "version": 1,
                                 "as_of": "2026-12-01", "status": "committed"})
    repo.db.dossiers.insert_one({"key": "300750", "ticker": "300750", "name": "宁德时代", "version": 1,
                                 "as_of": "2026-12-01", "status": "committed"})
    task = _run(repo, policy, f"{NAME}和宁德时代哪个好")
    assert task["status"] == "failed"
    assert "多家已跟踪公司" in (task.get("error") or "")
    assert task["request"].get("ticker") is None


def test_company_name_synced_to_known_or_cleared(repo, policy):
    """最终标的与模型代码不一致时：有档案用档案名，无档案不沿用模型的公司名。"""
    repo.db.dossiers.insert_one({"key": CODE, "ticker": CODE, "name": NAME, "version": 1,
                                 "as_of": "2026-12-01", "status": "committed"})
    task = _run_llm(repo, policy, _EchoTickerLLM("000001", company="平安银行"), f"{CODE}最近怎么样")
    notes = [e["message"] for e in task["events"] if e["node"] == "intake"]
    assert any(f"（{NAME}）" in m for m in notes)
    assert not any("平安银行" in m for m in notes)


def test_decimal_and_dot_code_runs_kept():
    """句点后长数字不再因空字符串被当金额：'600519.123' 与 '600519.300750' 都保留代码。"""
    assert _codes_with_dropped("600519.123")[0] == {"600519"}
    assert _codes_with_dropped("600519.300750")[0] == {"600519", "300750"}


def test_two_char_prefix_evidence_closed(repo, policy):
    """2 字首证据已移除：地名/集团前缀（贵州/中信/华夏）不得为模型码背书。"""
    task = _run_llm(repo, policy, _EchoTickerLLM("600519", company="贵州茅台"), "贵州板块最近怎么走")
    assert task["status"] == "failed" and task["request"].get("ticker") is None
    task2 = _run_llm(repo, policy, _EchoTickerLLM("600030", company="中信证券"), "中信银行最近怎么样")
    assert task2["status"] == "failed" and task2["request"].get("ticker") is None


def test_numeric_name_money_context_rejected(repo, policy):
    """纯数字名只在非金额语境下算证据：『360万元闲钱』不被 360 背书。"""
    task = _run_llm(repo, policy, _EchoTickerLLM("601360", company="360"), "有360万元闲钱，怎么配置比较好")
    assert task["status"] == "failed" and task["request"].get("ticker") is None


def test_zhongshi_alias_recall(repo, policy):
    """『中国X→中X』惯用简称恢复召回：中石油+模型中国石油应被采纳。"""
    task = _run_llm(repo, policy, _EchoTickerLLM("601857", company="中国石油"), "中石油最近怎么样")
    assert task["status"] in ("waiting_approval", "done", "running"), task.get("error")
    assert task["request"]["ticker"] == "601857"


def test_multi_name_conflict_no_contradictory_event(repo, policy):
    """多档案名 + 模型另一码：不得先写『将以本地档案名继续』再失败（文案自相矛盾）。"""
    repo.db.dossiers.insert_one({"key": CODE, "ticker": CODE, "name": NAME, "version": 1,
                                 "as_of": "2026-12-01", "status": "committed"})
    repo.db.dossiers.insert_one({"key": "300750", "ticker": "300750", "name": "宁德时代", "version": 1,
                                 "as_of": "2026-12-01", "status": "committed"})
    task = _run_llm(repo, policy, _EchoTickerLLM("000001", company="平安银行"), f"{NAME}和宁德时代哪个好")
    assert task["status"] == "failed"
    notes = [e["message"] for e in task["events"] if e["node"] == "intake"]
    assert not any("将以本地档案名继续" in m for m in notes)
    assert any("多家已跟踪公司" in (task.get("error") or "") for _ in [0])


def test_visible_code_model_name_dropped(repo, policy):
    """问题明写代码时模型自报的公司名不沿用（600519 不再配『五粮液』）。"""
    task = _run_llm(repo, policy, _EchoTickerLLM("600519", company="五粮液"), "600519最近怎么样")
    assert task["request"]["ticker"] == "600519"
    notes = [e["message"] for e in task["events"] if e["node"] == "intake"]
    assert not any("五粮液" in m for m in notes)


def test_zhong_alias_whitelist_only(repo, policy):
    """『中X』别名只认白名单且排除『其中石油』式词中命中：中医药/其中银行/其中石油 板块问句必须失败。"""
    task = _run_llm(repo, policy, _EchoTickerLLM("600056", company="中国医药"), "中医药板块最近怎么样")
    assert task["status"] == "failed" and task["request"].get("ticker") is None
    task2 = _run_llm(repo, policy, _EchoTickerLLM("601988", company="中国银行"),
                     "我看好金融板块，其中银行股最近表现如何")
    assert task2["status"] == "failed" and task2["request"].get("ticker") is None
    task3 = _run_llm(repo, policy, _EchoTickerLLM("601857", company="中国石油"),
                     "能源板块怎么看，其中石油价格影响大吗")
    assert task3["status"] == "failed" and task3["request"].get("ticker") is None


def test_numeric_name_range_amounts_rejected(repo, policy):
    """数字名的区间/口语金额语境：360-500元、360万块、360多块 都不是名称证据。"""
    assert not _name_evidence("360", "预算360-500元怎么配置")
    assert not _name_evidence("360", "赚了360万块")
    assert not _name_evidence("360", "360多块")
    task = _run_llm(repo, policy, _EchoTickerLLM("601360", company="360"), "预算360-500元怎么配置")
    assert task["status"] == "failed" and task["request"].get("ticker") is None


def test_semiconductor_tail_rejected(repo, policy):
    """『导体』停用：半导体板块不再为华虹半导体背书；中石油白名单不受影响。"""
    assert not _name_evidence("华虹半导体", "半导体板块最近怎么样")
    task = _run_llm(repo, policy, _EchoTickerLLM("688347", company="华虹半导体"), "半导体板块最近怎么样")
    assert task["status"] == "failed" and task["request"].get("ticker") is None
    assert _name_evidence("中国石油", "我买了中石油")
