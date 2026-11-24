import json
from decimal import Decimal

import pytest

from advisor.contracts import Metric
from advisor.settings import ROOT
from advisor.skills.compute.finance import (BasisMismatch, annualized_volatility, max_drawdown, moving_average,
                                            percentile_rank, price_change, ratio, same_basis_growth, single_quarter, ttm)

CASES = json.loads((ROOT / "evals" / "golden" / "compute_cases.json").read_text(encoding="utf-8"))


def m(field, value, period, basis="cumulative"):
    return Metric(name=field, field=field, value=Decimal(value), unit="元", period=period, basis=basis)


@pytest.mark.parametrize("case", CASES["growth"], ids=lambda c: c["id"])
def test_growth(case):
    cur, pri = m(*case["current"]), m(*case["prior"])
    if case.get("error"):
        with pytest.raises(BasisMismatch):
            same_basis_growth(cur, pri)
    else:
        assert same_basis_growth(cur, pri).value == Decimal(case["expected"])


@pytest.mark.parametrize("case", CASES["single_quarter"], ids=lambda c: c["id"])
def test_single_quarter(case):
    cum = m(*case["cumulative"])
    prev = m(*case["previous"]) if case["previous"] else None
    if case.get("error"):
        with pytest.raises(BasisMismatch):
            single_quarter(cum, prev)
    else:
        out = single_quarter(cum, prev)
        assert out.value == Decimal(case["expected"]) and out.basis == "single_quarter"


@pytest.mark.parametrize("case", CASES["ratio"], ids=lambda c: c["id"])
def test_ratio(case):
    num, den = m(*case["numerator"]), m(*case["denominator"])
    if case.get("error"):
        with pytest.raises(BasisMismatch):
            ratio(num, den, "r", "R")
    else:
        assert ratio(num, den, "r", "R").value == Decimal(case["expected"])


def test_price_stats():
    closes = [Decimal(100), Decimal(110), Decimal(121)]
    assert price_change(closes, 2) == Decimal("21.00")
    assert price_change(closes, 5) is None
    assert annualized_volatility(closes, 5) is None
    flat = [Decimal(10)] * 70
    assert annualized_volatility(flat, 60) == Decimal("0.00")


def test_single_quarter_rejects_unit_mismatch():
    cum = m("OPERATE_INCOME", "200", "2026-06-30")
    prev = Metric(name="OPERATE_INCOME", field="OPERATE_INCOME", value=Decimal("100"), unit="万元",
                  period="2026-03-31", basis="cumulative")
    with pytest.raises(BasisMismatch):
        single_quarter(cum, prev)


def test_single_quarter_rejects_non_cumulative_previous():
    cum = m("OPERATE_INCOME", "200", "2026-06-30")
    prev = m("OPERATE_INCOME", "100", "2026-03-31", basis="single_quarter")
    with pytest.raises(BasisMismatch):
        single_quarter(cum, prev)


def test_ttm_formats_missing_and_dirty_periods():
    dashed = {"2026-06-30": Decimal(91), "2025-12-31": Decimal(168), "2025-06-30": Decimal(90)}
    assert ttm(dashed, "2026-06-30") == Decimal(169)
    compact = {"20260630": Decimal(91), "20251231": Decimal(168), "20250630": Decimal(90)}
    assert ttm(compact, "20260630") == Decimal(169)          # 紧凑期号与短横线格式等价
    assert ttm({"2026-12-31": Decimal(168)}, "2026-12-31") == Decimal(168)
    assert ttm({"2026-06-30": Decimal(91)}, "2026-06-30") is None   # 缺上年年报/上年同期
    assert ttm({"2026-06-30": Decimal(91)}, "NaT") is None          # 脏期号按缺失处理


def test_window_guards():
    assert annualized_volatility([Decimal(10), Decimal(11)], 1) is None
    assert annualized_volatility([Decimal(10), Decimal(11), Decimal(12)], 0) is None
    assert moving_average([Decimal(1), Decimal(2)], 0) is None
    assert price_change([Decimal(100), Decimal(110)], 0) is None


def test_non_positive_prices_are_unusable():
    assert annualized_volatility([Decimal(100), Decimal(100), Decimal(0), Decimal(100)], 3) is None
    assert max_drawdown([Decimal(100), Decimal(0), Decimal(50)]) is None
    assert max_drawdown([Decimal(100), Decimal(80)]) == Decimal("-20.00")
    assert price_change([Decimal(0), Decimal(100)], 1) is None


def test_percentile_rank_midrank():
    assert percentile_rank([Decimal(5)] * 60, Decimal(5)) == Decimal("50.00")      # 平坦窗口给中性值
    assert percentile_rank([Decimal(1)] * 59 + [Decimal(10)], Decimal(10)) == Decimal("99.17")
    assert percentile_rank([Decimal(i) for i in range(1, 11)], Decimal(10)) == Decimal("95.00")
    assert percentile_rank([], Decimal(1)) is None
