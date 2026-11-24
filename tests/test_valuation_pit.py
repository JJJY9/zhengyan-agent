"""估值时点正确性：TTM 生效日、资产负债表按报告期取最新、非正价格与短历史行情。

- TTM 依赖本期累计、上年年报、上年同期三张报表；三者都披露后才能使用，生效日取披露日最大值。
- 资产负债表按报告期（MRQ）选最新一期，披露日只决定「何时可见」；一季报先于年报披露时不能回退到年报。
- 0/负收盘价（停牌脏行、复权异常）不参与行情与估值计算。
"""
from datetime import date, timedelta
from decimal import Decimal

from advisor.evidence import EvidenceFactory
from advisor.skills.data.structured import build_structured_skills
from advisor.skills.data.valuation import build_valuation_skill
from tests.fakes import CODE, FakeSource

AS_OF = date(2026, 5, 15)
B = 10 ** 9


def _factory():
    return EvidenceFactory(lambda h, c: None)


def _valuation(source):
    skill = build_valuation_skill(source, _factory())
    return skill.run(skill.spec.input_model(code=CODE, as_of=AS_OF))


def _daily(source, **kwargs):
    skill = {s.spec.name: s for s in build_structured_skills(source, _factory())}["market.daily"]
    return skill.run(skill.spec.input_model(code=CODE, as_of=AS_OF, **kwargs))


def _metric(out, field):
    return next(m for m in out.cards[0].metrics if m.field == field)


def _last_row(source):
    return max(source.daily(CODE, date(2024, 1, 1), AS_OF), key=lambda r: r["date"])


class _ReverseDisclosure(FakeSource):
    """2026 一季报（4/10）先于 2025 年报（4/28）披露；两期股本不同。"""

    def daily(self, code, start, end):
        rows, day, i = [], date(2025, 3, 1), 0
        while day <= end:
            rows.append({"date": day.isoformat(), "close": str(1000 + (i % 90) * 3),
                         "turnover": "0.3", "volume": str(20000 + (i % 30) * 100)})
            day += timedelta(days=1)
            i += 1
        return rows

    def income_rows(self, code):
        def row(period, notice, np_):
            return {"REPORT_DATE": f"{period} 00:00:00", "NOTICE_DATE": f"{notice} 00:00:00",
                    "SECURITY_NAME_ABBR": "示例酒业", "OPERATE_INCOME": str(np_), "PARENT_NETPROFIT": str(np_)}
        return [row("2025-03-31", "2025-04-30", 10 * B), row("2025-06-30", "2025-08-15", 40 * B),
                row("2025-09-30", "2025-10-30", 60 * B), row("2025-12-31", "2026-04-28", 100 * B),
                row("2026-03-31", "2026-04-10", 20 * B)]

    def balance_rows(self, code):
        return [
            {"REPORT_DATE": "2025-12-31 00:00:00", "NOTICE_DATE": "2026-04-28 00:00:00",
             "SHARE_CAPITAL": "1000000000", "TOTAL_PARENT_EQUITY": str(250 * B)},
            {"REPORT_DATE": "2026-03-31 00:00:00", "NOTICE_DATE": "2026-04-10 00:00:00",
             "SHARE_CAPITAL": "2000000000", "TOTAL_PARENT_EQUITY": str(300 * B)},
        ]


class _ZeroClose(FakeSource):
    def daily(self, code, start, end):
        return super().daily(code, start, end) + [
            {"date": end.isoformat(), "close": "0", "volume": "1", "turnover": "0.1"},
        ]


class _ShortHistory(FakeSource):
    def daily(self, code, start, end):
        return [{"date": (date(2026, 4, 1) + timedelta(days=i)).isoformat(), "close": str(100 - i),
                 "turnover": "0.3", "volume": "1000"} for i in range(22)]


def test_ttm_waits_for_all_dependency_reports():
    """年初一季报先披露但上年年报未披露时，Q1 的 TTM 不提前生效；都披露后取 Q1 口径。"""
    out = _valuation(_ReverseDisclosure())
    assert _metric(out, "np_ttm").value == Decimal(110 * B)      # 20 + 100 - 10


def test_balance_uses_latest_report_period_not_latest_notice():
    out = _valuation(_ReverseDisclosure())
    last = _last_row(_ReverseDisclosure())
    assert _metric(out, "mkt_cap").value == Decimal(last["close"]) * Decimal("2000000000")


def test_market_daily_skips_non_positive_close():
    out = _daily(_ZeroClose())
    last = _last_row(FakeSource())
    assert _metric(out, "close").value == Decimal(last["close"])
    assert _metric(out, "close").period == last["date"]


def test_valuation_skips_non_positive_close():
    out = _valuation(_ZeroClose())
    last = _last_row(FakeSource())
    mcap = _metric(out, "mkt_cap")
    assert mcap.period == last["date"]
    assert mcap.value == Decimal(last["close"]) * Decimal("1250000000")   # 0 价行的同日有效收盘仍被采用


def test_mdd_omitted_when_history_shorter_than_250():
    out = _daily(_ShortHistory(), lookback_days=30)
    assert "mdd_250d" not in {m.field for m in out.cards[0].metrics}


def test_mdd_present_with_enough_history():
    out = _daily(FakeSource())
    assert "mdd_250d" in {m.field for m in out.cards[0].metrics}
