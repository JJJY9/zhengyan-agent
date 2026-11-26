"""R83a 性质测试：RSI 经验库发布/回滚的在用版本不变式。

用直接构造的 approved 运行（绕过提案门控，专注指针状态机）随机交替发布与回滚，断言：
1) 指针版本必有资产、资产版本号连续 1..latest、sha256 与内容一致、parent 指向更早版本；
2) 任意时刻最多一个 released 运行，其 released_version == 指针且 base_version == 指针资产 parent；
3) 回滚精确退回 parent（from>to），回滚后无 released 运行（旧运行对齐为 rolled_back）；
4) 过期候选（base 与在用版本不符）发布必失败，且指针/资产/运行状态零副作用；
5) 基线版本不可回滚；loader 快照内容始终等于指针资产内容。"""
import random
from datetime import datetime, timezone

import pytest

from advisor.rsi import RSIRunner, TransitionError
from advisor.rsi.ledger import content_hash
from advisor.rsi.models import Run
from tests.test_phase2 import make_service


def _setup(repo, policy):
    svc = make_service(repo, policy)
    runner = RSIRunner(svc.deps, svc.deps.ledger, svc.deps.rsi)
    runner.ledger.bootstrap(policy)
    return svc, runner


def _candidate(runner, tag):
    """以当前在用内容为基线，追加一张新卡，直接登记为 approved 运行。"""
    base = runner.ledger.active_version("lessons")
    base_asset = runner.ledger.active_asset("lessons")
    content = dict(base_asset.content)
    cards = list(content.get("cards") or [])
    cards.append({"id": f"L-P{tag}", "title": "性质测试候选卡", "applies_to": "任何公司",
                  "tags": ["核验"], "lesson": f"先核对渠道政策与退款节奏再做因果判断（第 {tag} 版）",
                  "must_check": [], "invalid_if": "", "source": "", "rev": 1})
    content["cards"] = cards
    now = datetime.now(timezone.utc)
    run = Run(run_id=f"rsi_prop_{tag}", scenario="S1", kind="lessons", issue_ids=[], base_version=base,
              content=content, change_reason=f"性质测试：第 {tag} 版候选运行", direction=f"S1:props{tag}",
              status="approved", created_at=now, updated_at=now)
    runner.ledger.insert_run(run)
    return run


def _check(runner):
    led = runner.ledger
    av = led.active_version("lessons")
    assert av is not None
    asset = led.active_asset("lessons")
    assert asset is not None and asset.version == av
    versions = sorted(a.version for a in led.list_assets("lessons"))
    assert versions == list(range(1, len(versions) + 1))              # 资产版本连续无空洞
    assert asset.sha256 == content_hash(asset.content)
    if asset.parent_version is not None:
        assert asset.parent_version < asset.version
    released = [r for r in led.list_runs() if r.kind == "lessons" and r.status == "released"]
    assert len(released) <= 1                                          # 任意时刻至多一个在用运行
    if released:
        assert released[0].released_version == av
        assert released[0].base_version == asset.parent_version
    snap = runner.loader.snapshot()["lessons"]
    assert snap["version"] == av and snap["content"] == asset.content
    return av, asset


def test_rsi_release_rollback_pointer_invariants(repo, policy):
    rng = random.Random(20261204)
    _, runner = _setup(repo, policy)
    _check(runner)
    for i in range(6):
        run = _candidate(runner, i)
        av, asset = _check(runner)
        assert run.base_version == av                                  # 候选基于当前在用版本
        if rng.random() < 0.35 and asset.parent_version is not None:
            res = runner.rollback("lessons", "owner", f"性质测试回滚：第{i}轮理由不少于十个字")
            assert res["from"] == av and res["to"] == asset.parent_version
            to_av, to_asset = _check(runner)
            assert to_av == asset.parent_version and to_asset.version == to_av
            assert not [r for r in runner.ledger.list_runs()
                        if r.kind == "lessons" and r.status == "released"]   # 旧运行已对齐为 rolled_back
            assets_before = len(runner.ledger.list_assets("lessons"))
            with pytest.raises(TransitionError):                        # 过期候选：base 已不是在用版本
                runner.release(run.run_id, "owner")
            av2, _ = _check(runner)
            assert av2 == to_av                                        # 失败发布零副作用
            assert len(runner.ledger.list_assets("lessons")) == assets_before
            assert runner.ledger.get_run(run.run_id).status == "approved"
        else:
            runner.release(run.run_id, "owner")
            new_av, new_asset = _check(runner)
            assert new_av > av and new_asset.parent_version == av      # 指针前进且续链
            with pytest.raises(TransitionError):                        # 同一运行不能重复发布
                runner.release(run.run_id, "owner")
            assert _check(runner)[0] == new_av
    # 一路回滚到基线：每步精确退 parent，最后拒绝
    while True:
        av, asset = _check(runner)
        if asset.parent_version is None:
            with pytest.raises(TransitionError):
                runner.rollback("lessons", "owner", "性质测试：基线版本不可再回滚")
            break
        res = runner.rollback("lessons", "owner", "性质测试：逐级回滚到基线并核对指针")
        assert res["to"] == asset.parent_version
        assert _check(runner)[0] == res["to"]
