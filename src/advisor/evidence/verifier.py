"""独立核验器（规则，非模型）：来源存在、时点合法、证据等级足够、数字可在 A 级证据中复算。

数字核验不只看数量级：涨跌方向（“下降/亏损” → 负号）、报告期年份、以及无单位的十进制数
（如“营收 907.03”隐含亿元）都要与引用证据对得上；对不上就判存疑。
中文数字（如“一千亿”“三千五百万”）会解析为金额参与复算；无单位的零散中文数字不处理。
"""
import re
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation

from advisor.contracts import Claim, EvidenceCard

_NUM = re.compile(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(亿元|万元|亿|万|元|%|个百分点|倍)")
_BARE = re.compile(r"(?<![\d.])(\d+\.\d+)(?![\d.%])")
_SCALE = {"亿元": Decimal(10) ** 8, "亿": Decimal(10) ** 8, "万元": Decimal(10) ** 4, "万": Decimal(10) ** 4, "元": Decimal(1)}
_DOWN = re.compile(r"下降|下滑|减少|降低|回落|负增长|下跌|降幅|亏损|下调|收窄")
_UP = re.compile(r"上升|上涨|增长|提升|提高|增加|增幅|改善|回升")
_YEAR = re.compile(r"(20\d{2})\s*年")
_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}
_CN = re.compile(r"([零〇一二两三四五六七八九十百千万亿]{1,16})\s*(亿元|万元|亿|万|元)")


def cn_decimal(text: str) -> Decimal | None:
    """中文数字 → Decimal：支持 十/百/千 与 万/亿 分段（“一千亿”“三千五百万”的数字部分）。"""
    total = section = number = Decimal(0)
    for ch in text:
        if ch in _CN_DIGIT:
            number = Decimal(_CN_DIGIT[ch])
        elif ch in _CN_UNIT:
            section += (number if number else Decimal(1)) * _CN_UNIT[ch]
            number = Decimal(0)
        elif ch == "万":
            section = (section + number) * Decimal(10000)
            total += section
            section = number = Decimal(0)
        elif ch == "亿":
            section = (section + number) * Decimal(10) ** 8
            total += section
            section = number = Decimal(0)
        else:
            return None
    return total + section + number
_DIR_WINDOW = 10                 # 方向词只在此字数内生效
_SCALED_SCALES = (Decimal(1), Decimal(10) ** 4, Decimal(10) ** 8)   # 无单位数字按 个/万/亿 三档刻度匹配


def _direction(text: str, start: int) -> int:
    window = text[max(0, start - _DIR_WINDOW):start]
    if _DOWN.search(window) and not _UP.search(window):
        return -1
    return 1


def extract_numbers(text: str) -> list[tuple[Decimal, Decimal, str, str]]:
    """返回 [(带符号的值, 允许误差, 单位类别, 原文)]；误差按书写精度的半个单位。

    - 涨跌方向：数字前 10 个字内出现“下降/亏损”等取负，出现“增长/提升”等保持正；
    - 百分比差值（个百分点）只比较幅度，不带方向；
    - 无单位的十进制数按 scaled 类别记录（按 1/万/亿 三档刻度与证据比对）。
    """
    out = []
    spans = [m.span() for m in _NUM.finditer(text)]
    for m in _NUM.finditer(text):
        raw, unit = m.group(1), m.group(2)
        try:
            value = Decimal(raw.replace(",", ""))
        except InvalidOperation:
            continue
        places = len(raw.split(".")[1]) if "." in raw else 0
        half = Decimal(5) * Decimal(10) ** (-(places + 1))
        sign = -1 if value < 0 else _direction(text, m.start())
        if unit in _SCALE:
            out.append((sign * abs(value) * _SCALE[unit], half * _SCALE[unit] + Decimal("0.0001"), "amount", raw))
        elif unit == "%":
            out.append((sign * abs(value), half + Decimal("0.0001"), "pct", raw))
        elif unit == "倍":
            out.append((sign * abs(value), half + Decimal("0.0001"), "multiple", raw))
        else:
            out.append((abs(value), half + Decimal("0.0001"), "pp", raw))
    for m in _CN.finditer(text):
        if any(s <= m.start() < e for s, e in spans):        # 属于阿拉伯数字的单位（如“907.03亿元”）不重复解析
            continue
        if not any(ch in _CN_DIGIT for ch in m.group(1)):    # 纯“亿/万”等量词不算数字
            continue
        value = cn_decimal(m.group(1))
        if value is None:
            continue
        scale = _SCALE.get(m.group(2), Decimal(1))
        tol = Decimal("0.5") * scale + Decimal("0.0001")      # 约数：半个最小单位
        sign = _direction(text, m.start())
        out.append((sign * value * scale, tol, "amount", m.group(0)))
    for m in _BARE.finditer(text):
        if any(s <= m.start() < e for s, e in spans):     # 已带单位的数字不重复解析
            continue
        raw = m.group(1)
        try:
            value = Decimal(raw)
        except InvalidOperation:
            continue
        places = len(raw.split(".")[1]) if "." in raw else 0
        half = Decimal(5) * Decimal(10) ** (-(places + 1))
        sign = -1 if value < 0 else _direction(text, m.start())
        out.append((sign * value, half, "scaled", raw))
    return out


def _bigrams(text: str) -> set[str]:
    s = "".join(ch for ch in text if not ch.isspace() and ch not in "，。、；：（）()%,.;:+-—…/·")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _candidates(cards: list[EvidenceCard], years: set[str], claim_text: str = "") -> dict[str, list[Decimal]]:
    """按单位分池；claim 提到年份时只保留该年份报告期的指标（无报告期的保留）。

    指标归属：主张文本与某个指标名共享 2 字片段时，只用这些「同口径」指标做候选，
    避免拿营收去给「扣非净利润 907.03 亿」背书；一个都对不上时退回全部指标（避免误伤）。
    """
    metrics = []
    for c in cards:
        for m in c.metrics:
            period_year = str(m.period)[:4]
            if years and period_year and period_year not in years:
                continue
            metrics.append(m)
    if claim_text and metrics:
        claim_bg = _bigrams(claim_text)
        named = [m for m in metrics if _bigrams(m.name) & claim_bg]
        if named:
            metrics = named
    amounts, pcts, multiples = [], [], []
    for m in metrics:
        {"%": pcts, "倍": multiples}.get(m.unit, amounts).append(m.value)
    pps = [abs(a - b) for i, a in enumerate(pcts) for b in pcts[i + 1:]]
    return {"amount": amounts, "pct": pcts, "pp": pps + pcts, "multiple": multiples + [p / 100 for p in pcts]}


def _matches(value: Decimal, tol: Decimal, pool: list[Decimal]) -> bool:
    return any(abs(value - v) <= tol for v in pool)


def _matches_scaled(value: Decimal, tol: Decimal, pools: dict[str, list[Decimal]]) -> bool:
    """无单位数字：直接匹配比率/倍数，或按 个/万/亿 刻度匹配金额。"""
    if _matches(value, tol, pools["pct"]) or _matches(value, tol, pools["multiple"]):
        return True
    for scale in _SCALED_SCALES:
        t = tol * scale
        if any(abs(value * scale - v) <= t for v in pools["amount"]):
            return True
    return False


def verify_claim(claim: Claim, evidence: dict[str, EvidenceCard], as_of: date) -> Claim:
    notes: list[str] = []
    known = [evidence[i] for i in claim.evidence_ids if i in evidence]
    missing = [i for i in claim.evidence_ids if i not in evidence]
    if missing:
        notes.append(f"引用了不存在的证据 {missing}，已忽略")

    if claim.kind in ("assumption", "scenario"):
        notes.append("假设/情景，不是事实")
        return claim.model_copy(update={"status": "verified", "evidence_ids": [c.id for c in known], "verify_notes": notes})

    def done(status: str) -> Claim:
        return claim.model_copy(update={"status": status, "evidence_ids": [c.id for c in known], "verify_notes": notes})

    if not known:
        notes.append("没有可用证据")
        return done("rejected")
    future = [c.id for c in known if c.published_at and c.published_at.date() > as_of]
    if future:
        notes.append(f"证据晚于研究时点: {future}")
        return done("rejected")
    usable = [c for c in known if c.level in ("A", "B")]
    if not usable:
        notes.append("仅有 C 级线索，不能支撑结论")
        return done("rejected")

    numbers = extract_numbers(claim.text)
    if numbers:
        # 血缘：引用计算卡时，其来源财报也算作被引用
        lineage = [evidence[i] for c in usable if c.source_type == "calc" for i in c.derived_from if i in evidence]
        years = set(_YEAR.findall(claim.text))
        a_pool = _candidates([c for c in usable + lineage if c.level == "A"], years, claim.text)
        b_text = " ".join(c.excerpt for c in usable if c.level == "B")
        for value, tol, kind, raw in numbers:
            ok = _matches_scaled(value, tol, a_pool) if kind == "scaled" else _matches(value, tol, a_pool[kind])
            if ok:
                continue
            if raw in b_text:
                notes.append(f"数字 {value.normalize():f} 仅见于网页来源，须回到财报/公告复核")
            else:
                notes.append(f"数字 {value.normalize():f} 无法由引用证据复算")
            return done("disputed")

    web = [c for c in usable if c.source_type == "web"]
    if web and len(web) == len(usable) and len(web) > 1:
        groups = Counter(c.cluster_id or c.origin_publisher or c.origin_id for c in web)
        if len(groups) == 1:
            notes.append("多个网页来源属同一事件/转载，按单一来源计")
    if claim.kind == "fact" and all(c.level == "B" for c in usable):
        notes.append("依据为 B 级网页来源，仅说明发生了什么")
    return done("verified")


def verify_claims(claims: list[Claim], evidence: dict[str, EvidenceCard], as_of: date) -> list[Claim]:
    return [verify_claim(c, evidence, as_of) for c in claims]
