"""数据源脏行健壮性：NaT/缺失日期不得让技能崩溃，也不得产生错误的最新价或报告期。

AKShare 在停牌、退市整理或字段缺失时会返回 NaT/None。旧实现里 daily/announcements
把 _d() 的 None 直接和日期比较（TypeError），valuation 用未过滤的 rows[-1] 当最新行情
（脏尾行崩溃，乱序时把最早一天当最新价），statements 会为报告期为 NaT 的行生成证据卡。
"""
from datetime import date
from decimal import Decimal

from advisor.evidence import EvidenceFactory
from advisor.skills.compute.finance import q
from advisor.skills.data.structured import build_structured_skills
from advisor.skills.data.valuation import build_valuation_skill
from tests.fakes import CODE, FakeSource

AS_OF = date(2026, 10, 18)
SHARES = Decimal("1250000000")


def _skills(source):
    factory = EvidenceFactory(lambda h, c: None)
    skills = {s.spec.name: s for s in build_structured_skills(source, factory)}
    skills["valuation.snapshot"] = build_valuation_skill(source, factory)
    return skills


def _run(skills, name):
    skill = skills[name]
    return skill.run(skill.spec.input_model(code=CODE, as_of=AS_OF))


def _base_daily():
    return FakeSource().daily(CODE, date(2024, 1, 1), AS_OF)


def _metric(out, field):
    return next(m for m in out.cards[0].metrics if m.field == field)


class _DirtyReport(FakeSource):
    """报告期是 NaT、披露日正常的坏行。"""

    def income_rows(self, code):
        return super().income_rows(code) + [
            {"REPORT_DATE": "NaT", "NOTICE_DATE": "2026-09-10 00:00:00", "SECURITY_NAME_ABBR": "示例酒业",
             "OPERATE_INCOME": "1", "TOTAL_OPERATE_INCOME": "1", "PARENT_NETPROFIT": "1", "DEDUCT_PARENT_NETPROFIT": "1"},
        ]


class _DirtyDaily(FakeSource):
    def daily(self, code, start, end):
        return super().daily(code, start, end) + [
            {"date": "NaT", "close": "8888", "volume": "1", "turnover": "0.1"},
            {"date": None, "close": "7777", "volume": "1", "turnover": "0.1"},
        ]


class _ReversedDaily(FakeSource):
    def daily(self, code, start, end):
        return list(reversed(super().daily(code, start, end)))


class _DirtyAnn(FakeSource):
    def announcements(self, code, start, end):
        return super().announcements(code, start, end) + [
            {"name": "示例酒业", "title": "脏日期公告", "date": "NaT", "url": "http://www.cninfo.com.cn/x/3"},
        ]


class _DirtyTailDaily(FakeSource):
    def daily(self, code, start, end):
        return super().daily(code, start, end) + [
            {"date": "NaT", "close": "9999", "volume": "1", "turnover": "0.1"},
        ]


def test_statements_skip_rows_with_unparseable_period():
    out = _run(_skills(_DirtyReport()), "financials.statements")
    periods = [m.period for c in out.cards for m in c.metrics]
    assert periods and "NaT" not in periods
    assert len(out.cards) == 6                         # 坏行不算卡片；6 行在时点前已披露


def test_daily_skips_dirty_rows_and_uses_latest_date():
    out = _run(_skills(_DirtyDaily()), "market.daily")
    last = max(_base_daily(), key=lambda r: r["date"])
    close = _metric(out, "close")
    assert close.period == last["date"]
    assert close.value == Decimal(last["close"])


def test_daily_sorts_rows_from_source():
    out = _run(_skills(_ReversedDaily()), "market.daily")
    last = max(_base_daily(), key=lambda r: r["date"])
    assert _metric(out, "close").period == last["date"]
    assert _metric(out, "close").value == Decimal(last["close"])


def test_announcements_skip_dirty_date_rows():
    out = _run(_skills(_DirtyAnn()), "announcements.list")
    assert len(out.cards) == 2
    assert all("脏日期" not in c.title for c in out.cards)


def test_valuation_ignores_dirty_tail_row():
    out = _run(_skills(_DirtyTailDaily()), "valuation.snapshot")
    last = max(_base_daily(), key=lambda r: r["date"])
    mcap = _metric(out, "mkt_cap")
    assert mcap.period == last["date"]
    assert mcap.value == q(Decimal(last["close"]) * SHARES)


def test_valuation_sorts_rows_before_taking_last():
    out = _run(_skills(_ReversedDaily()), "valuation.snapshot")
    last = max(_base_daily(), key=lambda r: r["date"])
    assert _metric(out, "mkt_cap").period == last["date"]


class _CloseNoneDaily(FakeSource):
    """日期合法、收盘为 None（停牌/字段缺失）的行情行。"""

    def daily(self, code, start, end):
        return super().daily(code, start, end) + [
            {"date": AS_OF.isoformat(), "close": None, "volume": "1", "turnover": "0.1"},
        ]


class _MissingReportKeys(FakeSource):
    """REPORT_DATE 键缺失（不是 NaT）：旧实现合并 cash/balance 时直接 KeyError。"""

    def income_rows(self, code):
        return super().income_rows(code) + [
            {"NOTICE_DATE": "2026-09-10 00:00:00", "SECURITY_NAME_ABBR": "示例酒业", "OPERATE_INCOME": "1"},
        ]

    def cashflow_rows(self, code):
        return super().cashflow_rows(code) + [{"NOTICE_DATE": "2026-08-15 00:00:00", "NETCASH_OPERATE": "1"}]

    def balance_rows(self, code):
        return super().balance_rows(code) + [{"NOTICE_DATE": "2026-08-15 00:00:00", "SHARE_CAPITAL": "1"}]


class _BalanceDirtyReport(FakeSource):
    def balance_rows(self, code):
        return super().balance_rows(code) + [
            {"REPORT_DATE": "NaT", "NOTICE_DATE": "2026-08-15 00:00:00",
             "SHARE_CAPITAL": "2000000000", "TOTAL_PARENT_EQUITY": "250000000000"},
        ]


class _BalanceNoneReport(FakeSource):
    """同日披露两行、其中一行 REPORT_DATE 为 None：排序键比较不能抛 TypeError。"""

    def balance_rows(self, code):
        day = "2026-09-30 00:00:00"
        return super().balance_rows(code) + [
            {"REPORT_DATE": None, "NOTICE_DATE": day, "SHARE_CAPITAL": "3000000000"},
            {"REPORT_DATE": "2026-09-30 00:00:00", "NOTICE_DATE": day, "SHARE_CAPITAL": "4000000000"},
        ]


def test_daily_skips_close_none_row():
    out = _run(_skills(_CloseNoneDaily()), "market.daily")
    last = max(_base_daily(), key=lambda r: r["date"])
    close = _metric(out, "close")
    assert close.period == last["date"]
    assert close.value == Decimal(last["close"])


def test_valuation_skips_close_none_tail():
    out = _run(_skills(_CloseNoneDaily()), "valuation.snapshot")
    last = max(_base_daily(), key=lambda r: r["date"])
    mcap = _metric(out, "mkt_cap")
    assert mcap.period == last["date"]
    assert mcap.value == q(Decimal(last["close"]) * SHARES)


def test_valuation_skips_dirty_income_report_date():
    out = _run(_skills(_DirtyReport()), "valuation.snapshot")
    metrics = {m.field: m for m in out.cards[0].metrics}
    assert "pe_ttm" in metrics                    # 脏行不进 TTM，其余数据照常估值


def test_statements_skip_rows_missing_report_date():
    out = _run(_skills(_MissingReportKeys()), "financials.statements")
    assert len(out.cards) == 6                    # cash/balance 缺 REPORT_DATE 的行不参与合并、不崩


def test_valuation_prefers_dated_balance_rows_over_undated():
    """报告期可解析的行优先；REPORT_DATE=NaT 的行不覆盖已知报告期（只能作兜底）。"""
    out = _run(_skills(_BalanceDirtyReport()), "valuation.snapshot")
    last = max(_base_daily(), key=lambda r: r["date"])
    assert _metric(out, "mkt_cap").value == q(Decimal(last["close"]) * SHARES)


class _BalanceOnlyUndated(FakeSource):
    """所有资产负债表行的 REPORT_DATE 都缺失：退化为按披露日取最新。"""

    def balance_rows(self, code):
        return [
            {"REPORT_DATE": "NaT", "NOTICE_DATE": "2025-12-01 00:00:00", "SHARE_CAPITAL": "3000000000"},
            {"REPORT_DATE": None, "NOTICE_DATE": "2026-08-15 00:00:00", "SHARE_CAPITAL": "2000000000"},
        ]


def test_valuation_uses_undated_balance_rows_as_fallback():
    out = _run(_skills(_BalanceOnlyUndated()), "valuation.snapshot")
    last = max(_base_daily(), key=lambda r: r["date"])
    assert _metric(out, "mkt_cap").value == q(Decimal(last["close"]) * Decimal("2000000000"))


def test_valuation_tolerates_none_report_date_tie_break():
    out = _run(_skills(_BalanceNoneReport()), "valuation.snapshot")
    last = max(_base_daily(), key=lambda r: r["date"])
    assert _metric(out, "mkt_cap").value == q(Decimal(last["close"]) * Decimal("4000000000"))
