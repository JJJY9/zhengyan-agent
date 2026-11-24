"""估值快照：按披露日逐日对齐 TTM 净利润与股本，得到时点正确的 PE 区间与倍数情景。完全确定性。"""
import json
import statistics
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from pydantic import BaseModel, Field

from advisor.contracts import Metric
from advisor.contracts.skill import Skill, SkillSpec
from advisor.evidence import EvidenceFactory
from advisor.harness.errors import SkillFailed
from advisor.skills.compute.finance import percentile_rank, q, ttm
from advisor.skills.data.ashare import AShareSource
from advisor.skills.data.structured import CardsOut, to_decimal


class ValuationIn(BaseModel):
    code: str = Field(pattern=r"^\d{6}$")
    as_of: date
    window_days: int = Field(default=730, ge=180, le=1500)


def _d(s: str | None) -> date | None:
    try:
        return date.fromisoformat(s[:10]) if s else None
    except ValueError:          # 数据源偶发 "NaT" 等脏值：当作缺失，不让整个技能失败
        return None


def _latest(points: list[tuple[date, Decimal]], day: date) -> Decimal | None:
    """points 按生效日升序；返回 day 当天已生效的最新值。"""
    value = None
    for notice, v in points:
        if notice > day:
            break
        value = v
    return value


def _latest_balance(points: list[tuple[date, date, Decimal]], day: date) -> Decimal | None:
    """资产负债表时点取值：取 day 当天可见的各期里报告期最新的一期；同一报告期多次披露取晚披露的修订值。"""
    best: tuple[date, date] | None = None
    value = None
    for notice, period, v in points:
        if notice > day or period > day:
            continue
        key = (period, notice)
        if best is None or key > best:
            best, value = key, v
    return value


def build_valuation_skill(source: AShareSource, factory: EvidenceFactory) -> Skill:
    def run(inp: ValuationIn) -> CardsOut:
        income = []
        for r in source.income_rows(inp.code):
            notice, period = _d(r.get("NOTICE_DATE")), _d(r.get("REPORT_DATE"))
            np_ = to_decimal(r.get("PARENT_NETPROFIT"))
            if notice is None or period is None or notice > inp.as_of or np_ is None:
                continue
            income.append((notice, period.isoformat(), np_, r))
        if not income:
            raise SkillFailed("没有可用利润表，无法估值")
        name = income[0][3].get("SECURITY_NAME_ABBR") or inp.code
        cum = {p: v for _, p, v, _ in income}
        notice = {p: n for n, p, _, _ in income}

        def effective(period: str, value: Decimal) -> date | None:
            """TTM 依赖的报表全部披露后才生效：生效日取依赖报表披露日的最大值。"""
            y, md = period[:4], period[4:]
            deps = [period] if md == "-12-31" else [period, f"{int(y) - 1}-12-31", f"{int(y) - 1}{md}"]
            days = [notice.get(dep) for dep in deps]
            return None if any(x is None for x in days) else max(days)

        ttm_points = []
        for p in cum:
            v = ttm(cum, p)
            eff = effective(p, v) if v is not None else None
            if v is not None and eff is not None:
                ttm_points.append((eff, p, v))
        ttm_points.sort(key=lambda x: (x[0], x[1]))
        ttm_points = [(d, v) for d, _, v in ttm_points]

        # 资产负债表按报告期取最新一期（MRQ）：披露日决定「何时可见」，报告期决定「哪期更新」。
        # REPORT_DATE 不可解析的行只能作兜底候选（排在所有可解析报告期之后）。
        shares: list[tuple[date, date, Decimal]] = []
        equity: list[tuple[date, date, Decimal]] = []
        for r in source.balance_rows(inp.code):
            notice_d = _d(r.get("NOTICE_DATE"))
            sh = to_decimal(r.get("SHARE_CAPITAL"))
            if notice_d is None or notice_d > inp.as_of or sh is None or sh <= 0:
                continue
            period_d = _d(r.get("REPORT_DATE")) or date.min
            shares.append((notice_d, period_d, sh))
            eq = to_decimal(r.get("TOTAL_PARENT_EQUITY"))
            if eq is not None and eq > 0:
                equity.append((notice_d, period_d, eq))
        if not ttm_points or not shares:
            raise SkillFailed("TTM 净利润或股本不足，无法估值")

        rows = source.daily(inp.code, inp.as_of - timedelta(days=inp.window_days), inp.as_of)
        valid: list[tuple[date, Decimal]] = []
        for r in rows:
            day, price = _d(r.get("date")), to_decimal(r.get("close"))
            if day is None or day > inp.as_of or price is None or price <= 0:
                continue
            valid.append((day, price))
        valid.sort(key=lambda x: x[0])          # 数据源顺序不作假设；脏尾行不能当最新价
        series = []
        for day, price in valid:
            np_ttm, sh = _latest(ttm_points, day), _latest_balance(shares, day)
            if np_ttm is None or sh is None or np_ttm <= 0:
                continue
            series.append((day, price * sh / np_ttm))
        np_now, sh_now, eq_now = _latest(ttm_points, inp.as_of), _latest_balance(shares, inp.as_of), _latest_balance(equity, inp.as_of)
        if not valid or np_now is None or sh_now is None:
            raise SkillFailed("行情或 TTM 缺失，无法估值")
        last_day, close = valid[-1]
        period = last_day.isoformat()
        mcap = close * sh_now
        metrics = [
            Metric(name="总市值", field="mkt_cap", value=q(mcap), unit="元", period=period, basis="point",
                   note="收盘价×股本（按面值 1 元折算股数）"),
            Metric(name="归母净利润(TTM)", field="np_ttm", value=np_now, unit="元", period=period, basis="ttm",
                   note="本期累计+上年年报-上年同期累计"),
        ]
        pe_caveats: list[str] = []
        if np_now > 0:
            pe_now = mcap / np_now
            metrics.append(Metric(name="市盈率PE(TTM)", field="pe_ttm", value=q(pe_now), unit="倍", period=period, basis="ratio"))
        else:
            pe_now = None
            pe_caveats.append("TTM 净利润为负，PE 不适用（未给出 pe_ttm，避免 0 被当作真实倍数）")
        if eq_now and eq_now > 0:
            metrics.append(Metric(name="市净率PB(MRQ)", field="pb_mrq", value=q(mcap / eq_now), unit="倍", period=period, basis="ratio"))
        pes = [v for _, v in series]
        if pe_now is not None and len(pes) >= 60:
            lo, mid, hi = min(pes), Decimal(str(statistics.median(pes))), max(pes)
            span_note = f"{series[0][0]}~{series[-1][0]}，{len(pes)} 个交易日"
            metrics += [
                Metric(name="PE(TTM)窗口分位", field="pe_pct", value=percentile_rank(pes, pe_now), unit="%", period=period,
                       basis="ratio", note=span_note),
                Metric(name="PE(TTM)窗口最低", field="pe_min", value=q(lo), unit="倍", period=period, basis="ratio", note=span_note),
                Metric(name="PE(TTM)窗口中位", field="pe_median", value=q(mid), unit="倍", period=period, basis="ratio", note=span_note),
                Metric(name="PE(TTM)窗口最高", field="pe_max", value=q(hi), unit="倍", period=period, basis="ratio", note=span_note),
            ]
            for label, band in (("最低", lo), ("中位", mid), ("最高", hi)):
                metrics.append(Metric(
                    name=f"情景：PE回到窗口{label}时价格变化", field=f"scn_pe_{label}", value=q((band / pe_now - 1) * 100),
                    unit="%", period=period, basis="change", note="假设盈利不变、只改变估值倍数；不是预测",
                ))
        card = factory.make(
            content=json.dumps({"pe_series": [[d.isoformat(), str(q(v))] for d, v in series],
                                "ttm_points": [[d.isoformat(), str(v)] for d, v in ttm_points]}, ensure_ascii=False),
            level="A", source_type="calc", title=f"{name} 估值快照与 PE 区间（确定性计算，截至 {period}）",
            site_name="系统计算", origin_id=f"valuation:{inp.code}:{period}", as_of=inp.as_of,
            published_at=datetime(last_day.year, last_day.month, last_day.day, tzinfo=timezone.utc),
            excerpt="；".join(f"{m.name} {m.value}{m.unit}" for m in metrics[:6]), metrics=metrics,
            caveats=["股数按股本/面值 1 元折算", "历史市值用前复权价格，仅作相对比较",
                     "资产负债表按报告期取最新一期（MRQ），TTM 在依赖报表都披露后才生效",
                     "PE 区间只覆盖窗口期，不代表合理估值", "情景只改变估值倍数，盈利假设不变，不是预测"] + pe_caveats,
            skill="valuation.snapshot", skill_version="1.0.0",
        )
        return CardsOut(cards=[card])

    return Skill(
        spec=SkillSpec(name="valuation.snapshot", version="1.0.0", category="data", permission="read",
                       description="时点正确的市值、PE(TTM)、PB 与 PE 窗口区间、倍数情景",
                       input_model=ValuationIn, output_model=CardsOut, timeout_s=90,
                       retryable=(ConnectionError, TimeoutError, OSError)),
        run=run,
    )
