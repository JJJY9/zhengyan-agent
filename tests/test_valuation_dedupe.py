"""估值快照回归：同一报告期多行（追溯调整/更正重披）按最新披露取数，行序不影响结果、PIT 不放松。"""
from datetime import date

from advisor.evidence import EvidenceFactory
from advisor.skills.data.valuation import ValuationIn, build_valuation_skill
from tests.fakes import CODE, FakeSource

B = 10 ** 9
AS_OF = date(2026, 10, 18)


class _Source(FakeSource):
    """在标准行情/报表之上追加重复报告期的行（顺序由用例指定）。"""

    def __init__(self, extra: list[dict]):
        super().__init__()
        self._extra = extra

    def income_rows(self, code: str) -> list[dict]:
        return super().income_rows(code) + self._extra


def _dups() -> tuple[dict, dict]:
    """2025-06-30 期在 FakeSource 里是追溯调整行（notice 2025-08-13 / update 2026-08-15）。"""
    dup = next(r for r in FakeSource().income_rows(CODE) if r["REPORT_DATE"].startswith("2025-06-30"))
    stale = {**dup, "NOTICE_DATE": "2025-06-20 00:00:00", "UPDATE_DATE": "2025-06-20 00:00:00",
             "PARENT_NETPROFIT": str(30 * B)}          # 同一报告期的旧披露值
    future = {**dup, "NOTICE_DATE": "2027-01-05 00:00:00", "UPDATE_DATE": "2027-01-05 00:00:00",
              "PARENT_NETPROFIT": str(1 * B)}           # 研究时点之后才披露的更正：不可见
    return stale, future


def _np_pe(src, repo) -> tuple:
    skill = build_valuation_skill(src, EvidenceFactory(snapshot_sink=repo.put_snapshot))
    metrics = {m.field: m.value for c in skill.run(ValuationIn(code=CODE, as_of=AS_OF)).cards for m in c.metrics}
    return metrics["np_ttm"], metrics["pe_ttm"]


def test_duplicate_report_rows_follow_latest_disclosure_not_row_order(repo):
    stale, future = _dups()
    control = _np_pe(FakeSource(), repo)
    for extra in ([stale, future], [future, stale]):   # 旧披露在前/在后：结果必须与无重复行一致
        assert _np_pe(_Source(extra), repo) == control
