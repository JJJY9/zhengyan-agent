"""复核领取租约与证据元数据回归：活进程的 pending 占位不论时长不被 sync 重置、
占位被误重置后 finish_review 接回、死进程占位按时重置；证据冲突比对排除 retrieved_at、
冲突时 gather 引用库里生效的旧卡片；并发提交在档案里注明 prior_review 对照基准。"""
from datetime import datetime, timedelta, timezone

from advisor.graph.deps import build_deps
from advisor.graph.gather import gather_web
from advisor.harness import Budget
from advisor.services import Services
from advisor.watch.models import Alert
from tests.fakes import CODE, FakeLLM, FakeSource, as_of
from tests.test_diff_evals_hardening import _card
from tests.test_phase2 import make_service


def _alert_services(repo, policy):
    svc = make_service(repo, policy)
    services = Services(svc.deps, svc)
    services.watch.store.subscribe(CODE, "示例酒业", 10)
    services.watch.store.add_alert(Alert(id="al1", ticker=CODE, name="示例酒业", dossier_key=CODE, dossier_version=1,
                                         fingerprint="fp1", events=[], affected=[],
                                         created_at=datetime.now(timezone.utc)))
    return svc, services


def test_live_instance_pending_not_reset(repo, policy):
    """CLI 同步复核可能远超 120s：领取方租约还活着时，sync 不能把 pending 占位重置回 open。"""
    svc, services = _alert_services(repo, policy)
    store = services.watch.store
    repo.touch_lease(svc.instance)
    assert store.claim_review("al1", instance=svc.instance)
    store.update_alert("al1", review_started_at=(datetime.now(timezone.utc) - timedelta(seconds=900)).isoformat())
    services.watch.sync()
    assert store.get_alert("al1").status == "reviewing"


def test_finish_review_reattaches_after_misreset(repo, policy):
    """占位在任务创建期间被 sync 误判重置：任务真实创建后 finish_review 必须把提醒接回复核中。"""
    svc, services = _alert_services(repo, policy)
    store = services.watch.store
    assert store.claim_review("al1", instance=svc.instance)
    store.update_alert("al1", status="open", review_task_id=None)     # 模拟 sync 误判
    store.finish_review("al1", "t_real")
    a = store.get_alert("al1")
    assert a.status == "reviewing" and a.review_task_id == "t_real"


def test_dead_instance_pending_reset_after_grace(repo, policy):
    """属主进程已死且过了宽限：占位放回待处理，提醒不会永久卡在复核中。"""
    svc, services = _alert_services(repo, policy)
    store = services.watch.store
    assert store.claim_review("al1", instance="inst_ghost")
    store.update_alert("al1", review_started_at=(datetime.now(timezone.utc) - timedelta(seconds=900)).isoformat())
    services.watch.sync()
    assert store.get_alert("al1").status == "open"


# ---------- 证据元数据 ----------

def test_put_evidence_ignores_retrieved_at(repo):
    """同一内容再次取数（retrieved_at 变了）是幂等重放，不是元数据冲突。"""
    a = _card("body-x", title="标题")
    assert repo.put_evidence(a) is True
    b = a.model_copy(update={"retrieved_at": a.retrieved_at + timedelta(hours=1)})
    assert repo.put_evidence(b) is True


class _TitledWeb:
    def __init__(self):
        self.title = "旧标题"

    def search(self, query, count):
        return [{"title": self.title, "url": "https://www.stcn.com/a/1.html", "site_name": "证券时报",
                 "authority": 5, "rerank_score": 2.5, "date": "2026-08-16T08:00:00+00:00",
                 "summary": "来源：证券时报 示例酒业上半年实现营业收入910亿元。"}]

    def fetch(self, url):
        return {"title": "正文", "content": "<p>示例酒业上半年实现营业收入910亿元。</p>"}


def test_gather_uses_stored_card_on_conflict(repo, policy):
    """元数据冲突时保留旧卡：本次搜索结果必须引用库里生效的卡片，而不是被拒的新标题。"""
    web = _TitledWeb()
    deps = build_deps(repo, FakeLLM(), policy, source=FakeSource(), web_tools=web)
    h1 = deps.harness("t_g1", "standard", as_of(), Budget(limits=policy.route("standard").budget))
    r1 = gather_web(h1, deps, ["示例酒业 业绩"], as_of(), "standard", max_fetch=2)
    assert r1.cards and r1.cards[0].title == "旧标题"
    web.title = "新标题"
    h2 = deps.harness("t_g2", "standard", as_of(), Budget(limits=policy.route("standard").budget))
    r2 = gather_web(h2, deps, ["示例酒业 最新业绩"], as_of(), "standard", max_fetch=2)   # 换查询绕开技能缓存
    assert r2.cards[0].title == "旧标题"                              # 引用存库的旧卡
    assert any("不一致" in u for u in r2.unknowns)                    # 冲突仍然可见
