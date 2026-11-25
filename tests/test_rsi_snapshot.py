"""RSI 任务级快照回归：运行中发布的新版经验卡/检索模板不得渗入已开始的任务，
档案记录的版本与实际注入/生效的方法必须一致。"""

from advisor.contracts import ResearchRequest
from advisor.rsi import RSIRunner
from tests.fakes import CODE, as_of
from tests.test_phase2 import make_service
from tests.test_rsi import RSILLM, seed_dossiers


def _release_search(ledger, loader, rerank_min: float) -> None:
    base = ledger.active_asset("search")
    asset = ledger.append_asset("search", {**base.content, "rerank_min": rerank_min},
                                "检索阈值中途变更的测试版本", "owner", base.version, None)
    assert ledger.activate("search", base.version, asset.version)
    ledger.event("release", "owner", kind="search", active_version=asset.version,
                 detail=f"测试发布 search v{asset.version}")
    loader.invalidate()


def _service_with_hook(repo, policy, hook_schema: str, prepare):
    """prepare(svc) 在任务启动前装配「中途发布」动作；任务跑到 hook_schema 的模型调用时触发一次。"""
    fired = {"done": False}
    hook = {}

    class LLM(RSILLM):
        def json(self, role, system, user, schema, max_tokens=None):
            if schema.__name__ == hook_schema and not fired["done"]:
                fired["done"] = True
                hook["fn"]()
            return super().json(role, system, user, schema, max_tokens)

    svc = make_service(repo, policy, LLM())
    hook["fn"] = prepare(svc)
    return svc


def _run(svc, question: str, route: str = "standard"):
    tid = svc.start(ResearchRequest(question=question, as_of=as_of(), route=route), background=False)
    task = svc.deps.repo.get_task(tid)
    assert task["status"] == "done", task
    d = svc.deps.repo.get_dossier(task["result"]["dossier_key"], task["result"]["version"])
    return task, d


def _prepare_lesson_release(svc):
    runner = RSIRunner(svc.deps, svc.deps.ledger, svc.deps.rsi)
    runner.collect()
    issue = next(i for i in runner.ledger.list_issues("S1") if i.source["ticker"] == "600001")
    run = runner.propose("S1", issue.id)
    run = runner.evaluate(run.run_id)
    assert run.status == "review", run.reject_reason
    run = runner.approve(run.run_id, "owner")

    def release():
        runner.release(run.run_id, "owner")     # 与运行任务共享同一个 loader：invalidate 立即生效
        assert runner.ledger.active_version("lessons") == 2

    return release


def _prepare_search_bump(svc, rerank_min: float = 9.0):
    svc.deps.ledger.bootstrap(svc.deps.policy)

    def bump():
        _release_search(svc.deps.ledger, svc.deps.rsi, rerank_min)

    return bump


def test_midtask_lesson_release_does_not_leak_into_running_task(repo, policy):
    """任务开始后发布的经验卡（tags 命中本问题）不能被注入本任务，档案版本与注入内容一致。"""
    seed_dossiers(repo)
    svc = _service_with_hook(repo, policy, "QueryPlan", _prepare_lesson_release)
    task, d = _run(svc, f"复核 {CODE} 半年报业绩，合同负债变化说明什么")
    assert d.rsi_versions["lessons"] == 1                  # 任务开始时快照的版本
    assert d.lessons_used == []                            # v2 的卡不能中途注入
    assert not any("<method_hints>" in p for p in svc.deps.llm.prompts.get("DraftOutput", []))
    assert svc.deps.ledger.active_version("lessons") == 2  # 发布本身成功，只是不影响本任务


def test_midtask_search_release_does_not_change_running_filters(repo, policy):
    """检索相关度阈值中途提高到 9：运行中的任务仍按快照的 1.5 过滤，网页证据不能凭空清空。"""
    svc = _service_with_hook(repo, policy, "QueryPlan", _prepare_search_bump)
    task, d = _run(svc, f"复核 {CODE} 半年报业绩与估值")
    assert list(repo.db.evidence.find({"source_type": "web"}))   # 快照阈值 1.5 下命中；实时读 9.0 会被全部丢弃
    assert d.rsi_versions["search"] == 1


def test_midtask_search_release_does_not_change_expert_filters(repo, policy):
    """深研专家的检索也读任务级快照：中途发布不改变已开始任务的过滤阈值。"""
    svc = _service_with_hook(repo, policy, "TeamPlan", _prepare_search_bump)
    task, d = _run(svc, f"复核 {CODE} 半年报业绩，多空分歧在哪里", route="deep")
    assert list(repo.db.evidence.find({"source_type": "web"}))
    assert d.rsi_versions["search"] == 1
