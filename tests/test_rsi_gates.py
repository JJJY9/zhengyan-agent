"""RSI 六道门控的边界行为：任何一项不满足都要拦下（纯代码、不调模型）。"""
from datetime import datetime, timezone

import pytest

from advisor.rsi import Ledger
from advisor.rsi.gates import (cost_gate, leakage_gate, max_step, noise_gate, overfit_gate,
                               protocol_gate, regression_gate, run_gates)
from advisor.rsi.models import Run


def _stat(mean: float, spread: float = 0.02) -> dict:
    return {"runs": [mean] * 3, "mean": mean, "spread": spread}


def make_run(**kw) -> Run:
    now = datetime.now(timezone.utc)
    base = dict(
        run_id="rsi_x", scenario="S1", kind="lessons", issue_ids=[], base_version=1,
        content={"cards": []},
        diff={"cards": [{"id": "L-001", "title": "先核对口径", "applies_to": "适用于任何公司",
                         "tags": ["营收"], "lesson": "先核对口径再下结论", "must_check": [],
                         "invalid_if": "", "rev": 1}]},
        change_reason="门控单元测试用的候选运行", direction="S1:class:x", created_at=now, updated_at=now,
    )
    return Run(**(base | kw))


def make_ev(**kw) -> dict:
    base = dict(
        scenario="S1", primary="evolution", repeats=3,
        arms={
            "base": {"slices": {"evolution": _stat(0.60), "replay": _stat(0.60), "holdout": _stat(0.50)},
                     "tokens": 100},
            "cand": {"slices": {"evolution": _stat(0.70), "replay": _stat(0.60), "holdout": _stat(0.50)},
                     "tokens": 110},
        },
        leakage={"evolution": ["600001"], "holdout": ["000002"]},
        protocol={"scorer": "s1-claim-review-v3", "replay_hash": "H", "slices": {}},
    )
    base.update(kw)
    return base


@pytest.fixture
def ledger(repo):
    return Ledger(repo.db)


def test_protocol_gate_requires_frozen_replay(ledger):
    run, ev = make_run(), make_ev()
    assert not protocol_gate(run, ev, ledger).passed              # 还没冻结回放集
    ledger.db.rsi_protocols.insert_one({"scenario": "S1", "replay_ids": ["a"], "hash": "H",
                                        "created_at": datetime.now(timezone.utc)})
    assert protocol_gate(run, ev, ledger).passed
    assert not protocol_gate(run, make_ev(protocol={**ev["protocol"], "replay_hash": "OTHER"}), ledger).passed


def test_leakage_gate_rejects_overlap_or_empty_holdout(ledger):
    run = make_run()
    assert leakage_gate(run, make_ev(), ledger).passed
    assert not leakage_gate(run, make_ev(leakage={"evolution": ["X"], "holdout": ["X"]}), ledger).passed
    assert not leakage_gate(run, make_ev(leakage={"evolution": ["X"], "holdout": []}), ledger).passed


def test_noise_gate_requires_margin_beyond_spread(ledger):
    run = make_run()
    assert noise_gate(run, make_ev(), ledger).passed              # +0.10 远大于波动 0.02
    flat = make_ev(arms={"base": {"slices": {"evolution": _stat(0.60)}, "tokens": 1},
                         "cand": {"slices": {"evolution": _stat(0.61)}, "tokens": 1}})
    assert not noise_gate(run, flat, ledger).passed               # +0.01 落在波动内，可能是偶然
    assert not noise_gate(run, make_ev(repeats=2), ledger).passed  # 少于 3 次重复


def test_cost_gate_fails_when_token_increase_outruns_gain(ledger):
    run = make_run()
    assert cost_gate(run, make_ev(), ledger).passed               # token +10%，质量 +16.7% 覆盖
    pricey = make_ev(arms={"base": {"slices": {"evolution": _stat(0.60)}, "tokens": 100},
                           "cand": {"slices": {"evolution": _stat(0.62)}, "tokens": 160}})
    assert not cost_gate(run, pricey, ledger).passed              # token +60%，质量只 +3.3%


def test_regression_gate_blocks_holdout_regression(ledger):
    run = make_run()
    assert regression_gate(run, make_ev(), ledger).passed
    worse = make_ev(arms={"base": {"slices": {"replay": _stat(0.60), "holdout": _stat(0.55)}, "tokens": 1},
                          "cand": {"slices": {"replay": _stat(0.60), "holdout": _stat(0.50)}, "tokens": 1}})
    assert not regression_gate(run, worse, ledger).passed         # 留出集 -0.05 超过波动 0.02


def test_overfit_gate_enforces_step_limits(ledger):
    run = make_run(kind="routing", content={"temperature": 1.75, "abstain_below": 0.55},
                   diff={"params": {"temperature": [0.75, 1.75]}})
    g = overfit_gate(run, make_ev(), ledger)
    assert not g.passed and "步长" in g.detail                    # 首次发布改动上限 0.5
    ok_run = make_run(kind="routing", content={"temperature": 1.0, "abstain_below": 0.55},
                      diff={"params": {"temperature": [0.75, 1.0]}})
    assert overfit_gate(ok_run, make_ev(), ledger).passed


def test_overfit_gate_rejects_numbers_in_lesson_card(ledger):
    bad = {"id": "L-009", "title": "渠道政策核对", "applies_to": "适用于任何公司", "tags": ["x"],
           "lesson": "营收下降 42% 时先排除渠道政策", "must_check": [], "invalid_if": "", "rev": 1}
    g = overfit_gate(make_run(diff={"cards": [bad]}), make_ev(), ledger)
    assert not g.passed and "数字" in g.detail


def test_max_step_shrinks_with_released_count():
    assert max_step("temperature", 0) == 0.5
    assert max_step("temperature", 3) == 0.125
    assert max_step("rerank_min", 0) == 0.5


def test_run_gates_turns_gate_errors_into_failures(ledger):
    gates = {g.name: g for g in run_gates(make_run(), {}, ledger)}   # 残缺评测结果
    assert set(gates) == {"protocol", "leakage", "overfit", "noise", "cost", "regression"}
    assert not gates["leakage"].passed and not gates["noise"].passed
    assert not gates["cost"].passed and not gates["regression"].passed
