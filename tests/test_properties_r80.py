"""R80a 核心不变量性质测试（确定性随机）：
模拟盘——严格递增推进下 seq 连续递增、日期严格递增、净值恒正、同日幂等、回填被拒且状态不变；
版本链——随机并发碰撞（0~2 次重试）后版本号连续、parent 严格等于上一版、链可从最新回溯到 v1。"""
import random
from datetime import timedelta

import pytest

from advisor.contracts import ResearchRequest
from advisor.services import Services
from advisor.strategy import PaperError
from tests.fakes import CODE, as_of
from tests.test_phase2 import make_service
from tests.test_phase4 import START, spec


@pytest.fixture
def services(repo, policy):
    svc = make_service(repo, policy)
    svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"), background=False)
    return Services(svc.deps, svc)


def _assert_navs(book, v):
    navs = book.store.navs_of(v.card_id, v.version)
    assert [n.seq for n in navs] == list(range(1, len(navs) + 1))      # seq 连续递增
    assert [n.date for n in navs] == sorted(n.date for n in navs)      # 日期严格递增
    assert len({n.date for n in navs}) == len(navs)
    assert all(n.nav > 0 for n in navs)                                # 净值恒正
    return navs


def test_paper_random_forward_sequences(services):
    rng = random.Random(20261128)
    book = services.strategy
    v = book.start_paper(book.create(spec(), "owner", "性质测试：随机递增推进的模拟盘").card_id, 1, "owner", start=START)
    day = START
    for _ in range(rng.randint(8, 16)):
        day = day + timedelta(days=rng.randint(1, 3))
        p = book.paper.step(v, day)
        assert p.date == day
        assert book.paper.step(v, day) == p                            # 同日幂等：不新增净值点
        navs = _assert_navs(book, v)
        assert navs[-1].date == day


def test_paper_illegal_backfill_leaves_state_unchanged(services):
    rng = random.Random(20261129)
    book = services.strategy
    v = book.start_paper(book.create(spec(), "owner", "性质测试：非法回填不改变模拟盘状态").card_id, 1, "owner",
                         start=START)
    day = START
    for _ in range(6):
        day = day + timedelta(days=rng.randint(1, 2))
        book.paper.step(v, day)
        before = [(n.seq, n.date, n.nav) for n in book.store.navs_of(v.card_id, v.version)]
        for bad in (day - timedelta(days=1), START - timedelta(days=rng.randint(1, 5))):
            with pytest.raises(PaperError):
                book.paper.step(v, bad)
        after = [(n.seq, n.date, n.nav) for n in book.store.navs_of(v.card_id, v.version)]
        assert before == after                                        # 非法操作不写任何状态
    _assert_navs(book, v)


def _run_with_random_collisions(repo, policy, rng):
    svc = make_service(repo, policy)
    real_try = repo.try_insert_dossier
    collisions = {"left": rng.randint(0, 2)}

    def racing(dossier):
        if collisions["left"] > 0:
            collisions["left"] -= 1
            latest = repo.latest_dossier_version(dossier.key) or 0
            repo.insert_dossier(dossier.model_copy(update={"version": latest + 1, "parent_version": latest,
                                                           "task_id": "t_rival", "report_md": "竞争版本"}))
            return False
        return real_try(dossier)

    repo.try_insert_dossier = racing
    try:
        svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"),
                  background=False)
    finally:
        repo.try_insert_dossier = real_try


def test_version_chain_integrity_under_random_collisions(repo, policy):
    rng = random.Random(20261130)
    rounds = 6
    for _ in range(rounds):
        _run_with_random_collisions(repo, policy, rng)
    versions = sorted(row["version"] for row in repo.list_dossier_versions(CODE))
    assert versions == list(range(1, len(versions) + 1))              # 版本号连续、无分叉空洞
    by_version = {row["version"]: repo.get_dossier(CODE, row["version"])
                  for row in repo.list_dossier_versions(CODE)}
    for ver, d in by_version.items():
        if ver == 1:
            continue
        assert d.parent_version == ver - 1                             # 重试后续链严格挂在上一版
    # 从最新沿 parent 回溯必须到达 v1 且严格递减
    cur = by_version[max(by_version)]
    seen = [cur.version]
    while cur.parent_version:
        cur = by_version[cur.parent_version]
        seen.append(cur.version)
    assert seen == sorted(seen, reverse=True) and seen[-1] == 1
