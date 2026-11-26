"""版本链与列表口径回归：并发重试重算 parent、列表按 (as_of,version) 取当前档案、
证据元数据冲突可见、行情快照保留完整窗口。"""
import io
import json
import zipfile
from datetime import date, timedelta

from fastapi.testclient import TestClient

from advisor.api.app import create_app
from advisor.contracts import ResearchRequest
from advisor.evidence import EvidenceFactory
from advisor.skills.data.structured import DailyIn, build_structured_skills
from advisor.watch.scanner import _rebase_close
from tests.fakes import CODE, as_of
from tests.test_diff_evals_hardening import _card
from tests.test_phase2 import make_service


def test_commit_retry_recomputes_parent(repo, policy):
    svc = make_service(repo, policy)
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"), background=False)
    v1 = repo.get_dossier(CODE)
    real_try = repo.try_insert_dossier
    fired = {"n": 0}

    def racing(dossier):
        if fired["n"] == 0:
            fired["n"] = 1
            repo.insert_dossier(dossier.model_copy(update={"version": 2, "parent_version": v1.version,
                                                           "task_id": "t_rival", "report_md": "竞争版本"}))
            return False                                          # 模拟竞争任务抢先提交
        return real_try(dossier)

    repo.try_insert_dossier = racing
    try:
        svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"),
                  background=False)
    finally:
        repo.try_insert_dossier = real_try
    latest = repo.get_dossier(CODE)
    assert latest.version == 3 and latest.parent_version == 2     # 重试后必须挂到并发插入的 v2，而非旧 parent
    assert any("由 v1 变为 v2" in n for n in latest.boundary_notes)   # 档案自述并发续链，对照基准可追溯


def test_list_and_exports_follow_as_of_current(repo, policy):
    svc = make_service(repo, policy)
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=date(2026, 10, 18), route="standard"),
              background=False)
    svc.start(ResearchRequest(question=f"回填 {CODE} 更早时点", as_of=date(2026, 6, 1), route="standard"),
              background=False)
    rows = repo.list_dossiers()
    assert rows[0]["version"] == 1 and str(rows[0]["as_of"])[:10] == "2026-10-18"   # 回填的高版本旧档案不算最新
    with TestClient(create_app(deps=svc.deps, checkpointer=None)) as c:
        wl = c.get("/api/watchlist").json()
        assert wl[0]["version"] == 1
        z = c.get("/api/dossiers/export.zip")
        names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
        assert any("_v1_2026-10-18" in n for n in names)          # 导出与列表同口径


def test_put_evidence_conflict_visible(repo):
    a = _card("same-body", title="检索标题A")
    assert repo.put_evidence(a) is True
    assert repo.put_evidence(a) is True                           # 同内容幂等重复
    b = _card("same-body", title="检索标题B")
    assert b.id == a.id                                           # id 不含展示元数据
    assert repo.put_evidence(b) is False                          # 元数据不同：可见地保留旧版而非静默吞掉
    assert repo.db.evidence.find_one({"id": a.id})["title"] == "检索标题A"


def test_market_snapshot_keeps_full_window(repo):
    factory = EvidenceFactory(snapshot_sink=repo.put_snapshot)
    src = __import__("tests.fakes", fromlist=["FakeSource"]).FakeSource()
    skill = next(s for s in build_structured_skills(src, factory) if s.spec.name == "market.daily")
    as_of_day = date(2025, 3, 1) + timedelta(days=719)
    out = skill.run(DailyIn(code=CODE, as_of=as_of_day, lookback_days=720))
    snap = json.loads(repo.get_snapshot(out.cards[0].snapshot_hash))
    assert len(snap) > 260                                        # 完整窗口：老档案的复权基准行不能被截掉
    oldest = str(snap[0]["date"])[:10]
    assert _rebase_close(repo, out.cards[0], oldest) is not None  # 档案交易日可对齐复权基准
