"""R127a 性质测试：标的识别线的收敛不变量（固定种子随机组合）。

不变量（对应 R94a-R125a 对抗审计收敛后的定稿口径）：
P1 纯金额串（六位数 + 单位/货币前缀/区间/逗号小数）永远不产生代码；
P2 代码列表（逗号/顿号/空格分隔、含『元器件』等尾词）两边都保留；
P3 `_codes_with_dropped` 的 kept 与 `_codes_of` 一致、kept∩dropped=∅、并集覆盖全部严格匹配，
   产出的代码永远满足 ASCII A 股形态；
P4 停用行业词整名永远不构成名称证据；数字名在金额/区间语境不成立、在「XX这家公司」成立；
   『其中石油』不算证据而『尤其中石油』算；
P5 敏感遮蔽 `_mask_six`：≥6 位数字串必被遮蔽、<6 位不变、幂等；
P6 宽松拆码：合法代码按合法拆分与分隔符 + 『股』尾可找回，同样的拆分后接『元』找回为空。
"""
import random
import re

from advisor.graph.nodes import (
    _CODE, _CODE_SPACED, _NAME_STOP_TAILS, _codes_of, _codes_of_loose, _codes_with_dropped,
    _mask_six, _name_evidence, _norm_code,
)

RNG = random.Random(127)
VALID_PREFIXES = ("0", "3", "6", "4", "8")
UNITS = ["元", "万元", "亿元", "千元", "百万元", "千万元", "万手", "万股", "亿股", "万亿",
         "份", "块", "%", "％", "手", "美元", "港元", "港币"]
MONEY_PREFIXES = ["￥", "¥", "$", "€", "人民币"]
RANGE_SEPS = "-–—~～至到/／－−〜、"
LOOSE_SEPS = [" ", "　", " ", "​", "\n", "-"]


def _six(rng: random.Random) -> str:
    return rng.choice(VALID_PREFIXES) + "".join(rng.choice("0123456789") for _ in range(5))


def _pair(rng: random.Random):
    a = _six(rng)
    b = _six(rng)
    while b == a:
        b = _six(rng)
    return a, b


def _is_ashare(code: str) -> bool:
    return bool(re.fullmatch(r"[0-9]{6}", code)) and _CODE.fullmatch(code) is not None


def test_p1_money_only_never_a_code():
    """P1：纯金额串永远不产生代码。"""
    for _ in range(40):
        num = _six(RNG)
        kind = RNG.randrange(5)
        if kind == 0:                       # 数字 + （可选空白）+ 单位
            s = num + RNG.choice(["", " ", "　"]) + RNG.choice(UNITS) + "怎么办"
        elif kind == 1:                     # 货币前缀
            s = RNG.choice(MONEY_PREFIXES) + num + "怎么用"
        elif kind == 2:                     # 区间
            s = num + RNG.choice(RANGE_SEPS) + str(RNG.randint(1, 999)) + "元怎么配置"
        elif kind == 3:                     # 逗号小数
            s = num + "，" + str(RNG.randint(1, 99)) + "元怎么办"
        else:                               # 小数 + 单位
            s = num + "." + str(RNG.randint(1, 999)) + RNG.choice(["", " "]) + "万元营收"
        assert _codes_of(s) == set(), s


def test_p2_code_lists_survive():
    """P2：代码列表两边都保留（尾随『元器件』行业词也不吞）。"""
    for _ in range(30):
        a, b = _pair(RNG)
        joiner = RNG.choice([",", "，", "、", " "])
        assert _codes_of(f"{a}{joiner}{b}哪个好") == {a, b}
        assert _codes_of(f"{a},{b}元器件业务对比") == {a, b}


def test_p3_kept_and_dropped_partition():
    """P3：kept 与 _codes_of 一致、与 dropped 不交、覆盖全部严格匹配且形态合法。"""
    pool = ["600519", "300750", "002475", "920001", "600 519", "６００５１９",
            "360万元", "年薪600519元", "营收23万元", "元器件", "茅台", "半导体",
            "，", "、", " ", "怎么样", "和", "比", "营收600\n519", "600\t519"]
    for _ in range(60):
        s = "".join(RNG.choice(pool) for _ in range(RNG.randint(4, 8)))
        kept, dropped = _codes_with_dropped(s)
        assert kept == _codes_of(s)
        # kept 与 dropped 可以交叠：同一代码在不同出现位置可一次按代码、一次按金额
        strict = {_norm_code(m.group(1)) for m in _CODE_SPACED.finditer(s)}
        assert kept | dropped == strict
        for code in kept | dropped:
            assert _is_ashare(code), (s, code)


def test_p4_name_evidence_invariants():
    """P4：停用行业词整名不作证据；数字名的金额语境不成立、命名语境成立；其-词中排除。"""
    for word in _NAME_STOP_TAILS:
        assert not _name_evidence(word, f"{word}板块最近怎么走"), word
    for _ in range(20):
        n = str(RNG.randint(100, 99999))
        assert len(n) != 6
        assert not _name_evidence(n, f"预算{n}-500元怎么配置"), n
        assert not _name_evidence(n, f"赚了{n}多万怎么理财"), n
        assert _name_evidence(n, f"{n}这家公司怎么样"), n
    assert not _name_evidence("中国石油", "其中石油价格影响大吗")
    assert _name_evidence("中国石油", "尤其中石油表现最好")


def test_p5_mask_six_invariants():
    """P5：≥6 位数字串必遮蔽、<6 位不变、幂等。"""
    for _ in range(20):
        long = "".join(RNG.choice("0123456789") for _ in range(RNG.randint(6, 10)))
        short = "".join(RNG.choice("0123456789") for _ in range(RNG.randint(1, 5)))
        s = f"金额{long}与{short}"
        masked = _mask_six(s)
        assert long not in masked and short in masked
        assert _mask_six(masked) == masked


def test_p6_loose_bridging_symmetry():
    """P6：合法拆分 + 『股』尾可找回；同样的拆分后接『元』找回为空。"""
    for _ in range(20):
        code = _six(RNG)
        sep = RNG.choice(LOOSE_SEPS)
        cut = (3, 3) if sep == "-" else RNG.choice([(2, 4), (3, 3)])   # 连字符只认 3+3（避开月-年日期）
        a, b = code[:cut[0]], code[cut[0]:]
        assert _codes_of_loose(f"我持仓{a}{sep}{b}股") == {code}
        assert _codes_of_loose(f"我余额{a}{sep}{b}元") == set()
