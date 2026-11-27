"""主图节点（共用 + 单 Agent 标准研究）。模型提议，Harness 许可，核验器检查，提交器在复验授权后写入档案。"""
import math
import re
from datetime import date, datetime, timezone

from langgraph.types import interrupt

from advisor.contracts import Claim, Dossier, EvidenceCard, QualityResult
from advisor.contracts.deliberation import RouteDecision
from advisor.evidence.verifier import verify_claims
from advisor.graph.deps import Deps
from advisor.graph.gather import gather_web
from advisor.graph.report import render_report
from advisor.graph.state import ResearchState
from advisor.harness import Budget, BudgetExceeded, HarnessDenied, SkillFailed, TaskCancelled
from advisor.harness.egress import EgressGuard
from advisor.routing import decide_route, should_escalate
from advisor.rsi import LessonCard, recall, render_lessons
from advisor.skills.research.earnings_review import (
    DRAFT_SYSTEM, INTAKE_SYSTEM, INTENT_QUESTIONS, QUERY_SYSTEM, DraftOutput, IntakeOut, QueryPlan, derive_metrics,
    draft_user_prompt,
)

_CODE = re.compile(r"(?<!\d)([036][0-9]{5}|[48][0-9]{5}|92[0-9]{4})(?!\d)")
# 代码内只允许一个半角/全角空格：\s 还会匹配制表符/换行/NBSP，会把粘贴表格、折行里的
# 两个无关数字拼成一个"合法"代码，失败路径因此变成静默研究错误公司。
# 数字主体用 ASCII [0-9]（Unicode \d 会放行阿拉伯-印度等数字），前后边界仍用 \d：
# 非 ASCII 数字紧邻时不算代码（'٥600519' 不能被截成 600519）。
_CODE_SPACED = re.compile(r"(?<!\d)([036][0-9]{2}[ 　]?[0-9]{3}|[036][0-9]{5}|[48][0-9]{5}|92[0-9]{4})(?!\d)")
_FW_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
_DIGIT_TOKEN = re.compile(r"[0-9]+")
_LOOSE_SEP_GAP = re.compile(r"[\s​-‍﻿\-–—－~·•]+")
_UNIT_COMPOUND = ("万元", "亿元", "千元", "百万元", "千万元", "万手", "万股", "亿股", "万亿")
_MONEY_WORDS = ("美元", "港元", "港币", "人民币")
_UNIT_ALWAYS = "份块%％"
_FORMAT_CHARS = "\u200b\u200c\u200d\ufeff"
_MONEY_HEADS = "￥¥$€"
_MONEY_STARTERS = "元万亿千%％手份块股美港人"


def _is_cjk(ch: str) -> bool:
    return bool(ch) and "一" <= ch <= "鿿"


def _next_visible(s: str, j: int) -> int:
    while j < len(s) and (s[j].isspace() or s[j] in _FORMAT_CHARS):
        j += 1
    return j


def _digits_money_context(s: str, k: int) -> bool:
    """数字串之后的上下文是否金额化：句末/空白/数字/标点/非汉字符，或金额单位词首
    （元/万/亿/千/%/手/份/块/股/美元/港元/人民币）。'23年'、'2季度' 不算。"""
    k = _next_visible(s, k)
    if k >= len(s):
        return True
    ch = s[k]
    if not _is_cjk(ch):
        return True
    return ch in _MONEY_STARTERS


def _money_tail(s: str, i: int) -> bool:
    """匹配结束后的上下文是否表明这是金额/数量而不是代码：
    小数（句点/逗号后 1-2 位且后接金额语境；句点后多位数仅在紧跟金额单位时算）；
    复合单位/币种恒算；万/亿/千后接句末/标点/数字或 元/币种/份手股 才算是金额
    （'万科A' 不受影响）；份/块/% 恒为数量；手/元 的词首例外会先跳过空白与零宽字符。"""
    j = _next_visible(s, i)
    if j >= len(s):
        return False
    ch = s[j]
    if ch in ".．":
        k = _next_visible(s, j + 1)
        m = re.match(r"[0-9]+", s[k:])
        if not m:
            return False
        after = k + m.end()
        if re.match(r"(元|万元|亿元|千元|美元|港元|港币|人民币)", s[after:]):
            return True
        return len(m.group()) <= 2 and _digits_money_context(s, after)
    if ch in ",，":
        m = re.match(r"[,，]([0-9]+)", s[j:])
        if not m or len(m.group(1)) > 2:
            return False
        return _digits_money_context(s, j + m.end())
    if ch in "万亿千":
        k = _next_visible(s, j + 1)
        if k >= len(s) or not _is_cjk(s[k]):
            return True
        return s.startswith(("元", "美元", "港元", "港币", "人民币", "份", "手", "股"), k)
    if any(s.startswith(w, j) for w in _MONEY_WORDS) or any(s.startswith(w, j) for w in _UNIT_COMPOUND):
        return True
    if ch in _UNIT_ALWAYS:
        return True
    if ch in "手元":
        nxt = s[_next_visible(s, j + 1):][:1]
        if ch == "手":
            return not _is_cjk(nxt)
        return nxt not in ("器", "宇", "素")
    return False


def _money_head(s: str, i: int) -> bool:
    """匹配之前的上下文是否表明这是金额：货币符号前缀，或作为独立词出现的『人民币』
    （数字人民币/离岸人民币 这类词尾不算）。"""
    j = i - 1
    while j >= 0 and (s[j].isspace() or s[j] in _FORMAT_CHARS):
        j -= 1
    if j >= 0 and s[j] in _MONEY_HEADS:
        return True
    start = j + 1 - 3
    if start >= 0 and s[start:j + 1] == "人民币":
        before = s[start - 1:start]
        return not (before and _is_cjk(before))
    return False


def _norm_digits(text: str) -> str:
    """全角数字归一为半角（不合并词间空格，避免把「600999 10-18」粘成一个长数字）。"""
    return (text or "").translate(_FW_DIGITS)


def _norm_code(text: str) -> str:
    """代码归一：全角数字转半角并去掉全部空白（问题文本与模型 ticker 两个来源共用同一口径）。"""
    return re.sub(r"\s+", "", _norm_digits(text))


def _codes_with_dropped(text: str) -> tuple[set[str], set[str]]:
    """严格解析：返回（代码，被金额/货币上下文排除的六位数候选）。
    后者用于让『疑似金额』在 intake 里可见——多代码问题不会因为守卫误伤而静默改标的。"""
    s = text or ""
    kept: set[str] = set()
    dropped: set[str] = set()
    for m in _CODE_SPACED.finditer(s):
        code = _norm_code(m.group(1))
        if _money_tail(s, m.end()) or _money_head(s, m.start()):
            dropped.add(code)
            continue
        kept.add(code)
    return kept, dropped


def _codes_of(text: str) -> set[str]:
    """问题中出现的股票代码：允许代码内部一个空格（'600 519'），不把相邻数字误并。
    六位数金额/小数（'600519.23元'、'600 519 万元'、'600519美元'、'￥600519'）不是标的，按上下文排除。"""
    return _codes_with_dropped(text)[0]


def _codes_of_loose(text: str) -> set[str]:
    """敏感问题的兜底解析：相邻两个数字片段只由空白/全角空格/零宽/连字符等分隔、
    拼接后恰为合法代码、且尾随上下文不是金额/小数（与 _money_tail 同口径）时视为标的。
    不做整串合并——那样会把日期对（2024-12-31 / 2025-01-02）或股数粘成假代码；
    '我600\xa0519股' 能找回，'我的房贷余额600\xa0519元' / '我房贷 300\n750 元' 不会把余额当标的。"""
    s = _norm_digits(text or "")
    toks = [(m.start(), m.end()) for m in _DIGIT_TOKEN.finditer(s)]
    date_seps = "-–—－/．."
    out: set[str] = set()
    for (a1, a2), (b1, b2) in zip(toks, toks[1:]):
        gap = s[a2:b1]
        if not _LOOSE_SEP_GAP.fullmatch(gap):
            continue
        join = s[a1:a2] + s[b1:b2]
        if not _CODE.fullmatch(join):
            continue
        # 拆分形态：常规 3+3；允许 2+4/4+2（92 段代码），连字符分隔只认 3+3（'03-2025' 是月-年）
        split = (a2 - a1, b2 - b1)
        hyphen = any(ch in gap for ch in "-–—－")
        if split not in ((3, 3), (2, 4), (4, 2)) or (hyphen and split != (3, 3)):
            continue
        # 月-年/年-月形状（'03 2025'、'2025 03'）不是标的
        if split == (2, 4) and 1 <= int(s[a1:a2]) <= 12 and 1900 <= int(s[b1:b2]) <= 2100:
            continue
        if split == (4, 2) and 1900 <= int(s[a1:a2]) <= 2100 and 1 <= int(s[b1:b2]) <= 12:
            continue
        if (a1 and s[a1 - 1].isdecimal()) or (b2 < len(s) and s[b2].isdecimal()):
            continue                     # 邻接（含非 ASCII）数字：不是独立代码（isdecimal 对齐 \d/Nd）
        if (a1 and s[a1 - 1] in date_seps) or (b2 < len(s) and s[b2] in date_seps):
            continue                     # 日期片段：'2024-12-31\n2025-01-02' 的 31+2025 不能拼成 312025
        if _money_tail(s, b2) or _money_head(s, a1):
            continue                     # 金额/小数：'300\n750 元'、'￥600 519'
        out.add(join)
    return out


_RISK = re.compile(r"风险|负面|下滑|质疑|问题|处罚|诉讼|减持|下调|反方")
MIN_VERIFIED_FACTS = 3

# 执行过程里给用户看的说法（不出现内部术语）
INTENT_LABEL = {"research": "研究分析", "explain": "解释说明", "trade_execution": "交易指令（只研究，不执行交易）", "other": "其他"}
ROUTE_LABEL = {"quick": "快查", "standard": "标准研究", "deep": "争议深研"}
ROUTE_SOURCE_LABEL = {"user": "你指定的", "rule": "按规则判断", "model": "系统根据问题判断", "abstain": "系统拿不准，选了更稳妥的方式",
                      "escalation": "中途发现争议较多，升级", "fallback": "系统判断失败，按默认方式"}
ROLE_LABEL = {"reason": "深度推理模型", "draft": "快速生成模型", "classify": "快速判断模型"}
GATE_LABEL = {"pass": "质量检查通过", "partial": "质量检查：证据不足的部分标为未完成，其余保留",
              "need_more": "质量检查：证据不够，再补充搜索一轮", "escalate": "质量检查：争议较多，升级为多专家深研"}
DOSSIER_STATUS_LABEL = {"committed": "完整结论", "partial": "部分结论"}


def _cards(state: ResearchState, ids: list[str] | None = None) -> list[EvidenceCard]:
    ev = state.get("evidence", {})
    keys = ids if ids is not None else list(ev)
    return [EvidenceCard.model_validate(ev[i]) for i in keys if i in ev]


def _dump(cards: list[EvidenceCard]) -> dict[str, dict]:
    return {c.id: c.model_dump(mode="json") for c in cards}


def lesson_query(question: str, cards: list[EvidenceCard]) -> str:
    """经验卡召回文本：问题 + 证据里出现的指标名。"""
    return question + " " + " ".join({m.name for c in cards for m in c.metrics})


def dossier_key(state: ResearchState) -> str:
    ns = state.get("namespace")
    return f"{ns}:{state['ticker']}" if ns else state["ticker"]


class Nodes:
    def __init__(self, deps: Deps):
        self.d = deps
        self.guard = EgressGuard(
            deps.policy.egress.get("sensitive_patterns", []), deps.policy.egress.get("trade_intent_patterns", [])
        )

    # ---------- 工具 ----------
    def _rsi_snapshot(self, state: ResearchState) -> dict:
        return state.get("rsi_snapshot") or {}

    def _lessons(self, state: ResearchState, text: str) -> list:
        """经验卡召回：任务内只用 route 时点的快照，运行中发布不影响本任务。"""
        snap = self._rsi_snapshot(state).get("lessons")
        if snap is not None:
            cards = [LessonCard.model_validate(c) for c in snap.get("content", {}).get("cards", [])]
            return recall(cards, text)
        return self.d.rsi.lessons(text) if self.d.rsi else []

    def _search_cfg(self, state: ResearchState) -> dict:
        """检索参数：同样只读快照（rerank_min/权威阈值/追加站点/专家侧重）。"""
        snap = self._rsi_snapshot(state).get("search")
        if snap is not None:
            return {**self.d.policy.search, **snap.get("content", {})}
        return self.d.rsi.search() if self.d.rsi else dict(self.d.policy.search)
    def _h(self, state: ResearchState, node: str):
        budget = Budget.from_dict(state["budget"])
        # 任务级账本：节点被审批中断时不会返回它算出的预算，计费只落在这里才不会丢；
        # 重启/重放后按账本恢复已用，route 上限仍被真实消耗守住。
        for k, v in self.d.repo.task_runtime_budget(state["task_id"]).items():
            budget.used[k] = max(float(budget.used.get(k, 0.0)), float(v))
        budget.paused_s = self.d.repo.task_paused_s(state["task_id"])   # 审批等待不算执行时限
        route = state.get("route") or "quick"
        return self.d.harness(state["task_id"], route, date.fromisoformat(state["as_of"]), budget, node=node,
                              task_scoped_cache=True, namespace=state.get("namespace", "")), budget

    def _event(self, state: ResearchState, node: str, msg: str) -> None:
        self.d.repo.push_event(state["task_id"], node, msg)

    # ---------- L0：任务识别 ----------
    def intake(self, state: ResearchState) -> dict:
        q = state["question"]
        qn = _norm_digits(q)
        if not state.get("budget"):
            state = {**state, "budget": Budget(limits=self.d.policy.route("quick").budget).to_dict()}
        h, budget = self._h(state, "intake")
        explicit = state.get("ticker")
        codes, dropped = _codes_with_dropped(qn)   # dropped：被金额/货币上下文排除的六位数（仅用于可见性）
        sensitive_hit = bool(self.guard.find_sensitive(q))
        # 宽松拆码只作为最后的兜底候选：不并入 codes，不参与模型/名称的优先级；
        # 敏感问题下始终计算（即使已有严格码），仅用于冲突可见性与多候选提示。
        loose = _codes_of_loose(q) if sensitive_hit else set()
        code = explicit or (next(iter(codes)) if len(codes) == 1 else None)   # 多个代码不硬猜，交给模型/澄清
        code_src = "explicit" if explicit else ("code" if code else None)
        known = {d["ticker"]: d.get("name") or "" for d in self.d.repo.list_dossiers() if d.get("ticker")}
        name_hits = {t for t, n in known.items() if n and n in qn}   # 名字先本地解析，脱敏后模型看不到也能识别
        name_code = next((t for t, n in known.items() if n and n in qn), None)
        out = None
        small_src = None
        if self.d.policy.intake.get("classifier", "llm") == "small" and h.decision_client is not None:
            try:
                out = self._intake_small(h, q, code or name_code, known)
                if out is not None:
                    small_src = "code" if code else "name"   # 小模型路径回显的是本地解析结果
            except TaskCancelled:
                raise
            except Exception:  # 小模型不可用不致命：退回大模型
                out = None
        if out is None:
            # 入口识别需要保留公司名：用与 find_sensitive 同口径的整块脱敏（跨行拆分的手机号/卡号
            # 也会被合并命中并遮蔽）；股票代码已在上方从原文本地解析，不依赖模型看见原始数字。
            masked = self.guard.redact_question(q)
            try:
                out = h.llm("classify", INTAKE_SYSTEM, masked, IntakeOut, purpose="intake")
            except TaskCancelled:
                raise
            except Exception as e:  # 识别失败不致命：退回规则
                self._event(state, "intake", f"自动识别失败，改用规则识别：{e}")
                out = IntakeOut()
        model_ticker = _norm_code(out.ticker)
        candidates = codes | dropped
        if explicit is None:
            if small_src and _CODE.fullmatch(model_ticker):
                code, code_src = model_ticker, small_src
            elif _CODE.fullmatch(model_ticker) and (not candidates or model_ticker in candidates):
                # 模型给了原文里没有的标的才算『推断』；回显可见代码仍记 code
                code = model_ticker
                code_src = "code" if model_ticker in codes else "model"
            elif len(codes) == 1:
                code, code_src = next(iter(codes)), "code"
            elif name_code:
                code, code_src = name_code, "name"
            elif not codes and len(loose) == 1:
                # 数字被整段脱敏后模型也看不到原文：拆码兜底放到最后，且审批时会展示标的。
                # 严格代码存在时（哪怕多个）一律不用兜底——多码必须显式失败而不是静默选定。
                code, code_src = next(iter(loose)), "loose"
                self._event(state, "intake", f"问题中的数字已被脱敏；从拆分数字中识别到标的 {code}，将以它继续（审批时可拒绝）")
        if explicit:
            if codes and explicit not in codes:
                self._event(state, "intake", f"注意：指定标的是 {explicit}，但问题里出现代码 "
                                             f"{'、'.join(sorted(codes))}，将按指定标的继续")
            if name_hits and explicit not in name_hits:
                names = "、".join(f"{known.get(t) or ''}（{t}）" for t in sorted(name_hits))
                self._event(state, "intake", f"注意：指定标的是 {explicit}，但问题里点名的是 {names}，将按指定标的继续")
            if len(loose) == 1 and next(iter(loose)) != explicit:
                self._event(state, "intake", f"注意：问题中被脱敏的数字里包含标的 {next(iter(loose))}，"
                                             f"将按指定标的 {explicit} 继续")
        elif code and len(loose) == 1 and next(iter(loose)) != code:
            self._event(state, "intake", f"注意：问题中还出现拆码候选 {next(iter(loose))}，未采用（将按 {code} 继续）")
        stray = dropped - ({code} if code else set())
        if stray:
            # 金额守卫是启发式的（如『元件/元旦』词首）：被排除的六位数必须可见，避免改写标的时静默
            self._event(state, "intake", f"注意：问题里另有疑似金额的六位数（未采用）：{'、'.join(sorted(stray))}")
        intent = "trade_execution" if self.guard.has_trade_intent(q) else out.intent
        sensitive = sensitive_hit
        upd = {
            "ticker": code, "company_name": out.company_name or "", "intent": intent, "focus": out.focus,
            "sensitive": sensitive, "budget": budget.to_dict(), "iteration": 0, "status": "running",
            # 预脱敏问题：后续所有提示词拼装都用它，证据区数字因此不会被问题区的命中波及
            "safe_question": self.guard.redact_question(q),
        }
        if not explicit and code:
            # 回写识别出的股票代码：失败/取消后的「按原参数重试」不再依赖问题文本里仍带代码；
            # 同时记录来源（code/model/name/loose），前端据此标注「推断」
            self.d.repo.update_task(state["task_id"], **{"request.ticker": code, "request.ticker_source": code_src})
        if not code:
            if len(codes) > 1:
                if loose:
                    self._event(state, "intake", f"问题里的数字被拆开且出现候选（{'、'.join(sorted(loose))}），未自动选择")
                upd.update(status="failed", error=f"问题里出现多个股票代码（{'、'.join(sorted(codes))}），"
                                                   "请只保留一个或直接写明要研究哪一只")
            elif len(loose) > 1:
                self._event(state, "intake", f"问题里的数字被拆开且出现多个候选（{'、'.join(sorted(loose))}），未自动选择")
                upd.update(status="failed", error="未能识别 A 股代码，请在问题中写明 6 位代码（如 600519）")
            else:
                upd.update(status="failed", error="未能识别 A 股代码，请在问题中写明 6 位代码（如 600519）")
        self._event(state, "intake", f"股票 {code or '未识别'}{'（' + out.company_name + '）' if out.company_name else ''}，"
                                     f"问题类型：{INTENT_LABEL.get(intent, intent)}" + ("；问题里含个人信息，已脱敏" if sensitive else ""))
        return upd

    def _intake_small(self, h, q: str, code: str | None, known: dict[str, str]) -> IntakeOut | None:
        """小模型路径：意图由决策小模型四选一；代码靠正则/已有档案的简称（由调用方解析）。
        识别不到代码时返回 None，交给大模型（它能从"茅台"这类简称推出代码）。"""
        if not code:
            return None
        probs = h.decide(q, INTENT_QUESTIONS, purpose="intake_small")["intent"]
        finite = {k: float(v) for k, v in probs.items()
                  if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0}
        words = [w for w in self.d.policy.intake.get("focus_keywords", []) if w in q]
        words = [w for w in words if not any(w != o and w in o for o in words)]   # 去掉被更长词包含的
        return IntakeOut(company_name=known.get(code) or None, ticker=code,
                         intent=max(finite, key=finite.get) if finite else "research",
                         focus="、".join(words)[:20])

    def authorize(self, state: ResearchState) -> dict:
        notes = []
        if state.get("intent") == "trade_execution":
            notes.append("本系统仅做研究与模拟，不执行任何买卖或资金动作；以下只解释事实与情景。")
        if state.get("sensitive"):
            ticker = state.get("ticker")
            message = ("问题中包含个人财务/账户信息。当前不做个人情境分析：这些信息会被脱敏，不会发往搜索，"
                       "也不会用于结论。是否继续只做公开信息研究？")
            if ticker:
                # 标的可能是从被脱敏的数字里识别出来的：审批时展示，错了可以直接拒绝并改写问题
                message += f"（本次将以 {ticker} 为标的继续；如不对请拒绝并改写问题。）"
            decision = interrupt({"type": "consent", "message": message})
            if not (isinstance(decision, dict) and decision.get("approve")):
                self._event(state, "authorize", "用户拒绝，任务停止")
                return {"status": "cancelled"}
            notes.append("问题中的个人信息已脱敏，本次只做公开信息研究，不构成个性化建议。")
        self._event(state, "authorize", "检查通过：只做公开信息研究，不涉及交易")
        return {"boundary_notes": notes}

    # ---------- L1：双轴路由（任务轴 + 模型轴） ----------
    def route(self, state: ResearchState) -> dict:
        h, budget = self._h(state, "route")
        # 任务开始时一次性取 RSI 各组件快照：路由参数、经验卡、检索模板此后都只读快照
        snap = self.d.rsi.snapshot() if self.d.rsi else {}
        cfg = {**self.d.policy.routing, **snap.get("routing", {}).get("content", {})} if snap else self.d.policy.routing
        decision = decide_route(h, state.get("safe_question") or state["question"], state.get("intent", "research"),
                                state.get("route_requested", "auto"), cfg)
        rp = self.d.policy.route(decision.route)
        budget = Budget(limits=rp.budget, used=budget.used, started_at=budget.started_at, paused_s=budget.paused_s)
        self.d.repo.save_working(state["task_id"], "routing", "decision", decision.model_dump(mode="json"))
        how = ROUTE_SOURCE_LABEL.get(decision.source, "")
        self._event(state, "route", f"采用「{ROUTE_LABEL[decision.route]}」{'（' + how + '）' if how else ''}，"
                                    f"使用{ROLE_LABEL.get(rp.llm_role, rp.llm_role)}")
        return {"route": decision.route, "routing": decision.model_dump(mode="json"), "budget": budget.to_dict(),
                "rsi_snapshot": snap,
                "rsi_versions": {k: v.get("version") for k, v in snap.items()} if snap
                else (self.d.rsi.versions() if self.d.rsi else {})}   # 任务开始时的方法版本快照

    # ---------- 计划 ----------
    def plan(self, state: ResearchState) -> dict:
        h, budget = self._h(state, "plan")
        quick = state["route"] == "quick"
        limit = 2 if quick else 3
        name = state.get("company_name") or state["ticker"]
        if state.get("iteration", 0) == 0:
            try:
                plan = h.llm("draft", QUERY_SYSTEM, f"公司：{name}（{state['ticker']}）\n关注：{state.get('focus') or '最新业绩'}",
                             QueryPlan, purpose="plan_queries")
                queries = plan.queries
            except TaskCancelled:
                raise
            except Exception as e:
                self._event(state, "plan", f"查询规划失败，使用默认查询：{e}")
                queries = []
            queries = queries or [f"{name} 业绩", f"{name} 行业 竞争", f"{name} 风险"]
        else:
            queries = list((state.get("draft") or {}).get("search_gaps", []))
        done = set(state.get("searched", []))
        clean = []
        for q in queries:
            q = re.sub(r"\s+", " ", q).strip()[:40]
            if len(q) < 2 or q in done or self.guard.find_sensitive(q):
                continue
            clean.append(q)
        clean = clean[:limit]
        if state.get("iteration", 0) == 0 and not any(_RISK.search(q) for q in clean):
            clean = clean[: limit - 1] + [f"{name} 风险 质疑"]   # 反证：至少一条风险/负面查询
        self.d.repo.save_working(state["task_id"], "plan", f"round{state.get('iteration', 0) + 1}", clean)
        self._event(state, "plan", f"第 {state.get('iteration', 0) + 1} 轮要搜索：{'；'.join(clean) or '无'}")
        return {"queries": clean, "budget": budget.to_dict()}

    # ---------- L3：结构化取数与计算 + 工作记忆 ----------
    def collect(self, state: ResearchState) -> dict:
        if state.get("statement_ids"):
            return {}                      # 补证轮次或升级深研：复用已取的事实快照
        h, budget = self._h(state, "collect")
        as_of = date.fromisoformat(state["as_of"])
        code = state["ticker"]
        upd: dict = {"evidence": {}, "unknowns_sys": []}
        try:
            st = h.call("financials.statements", code=code, as_of=as_of, periods=12)
        except (SkillFailed, HarnessDenied, BudgetExceeded) as e:
            self._event(state, "collect", f"财报取数失败：{e}")
            return {"status": "failed", "error": f"财报取数失败：{e}", "budget": budget.to_dict()}
        upd["evidence"].update(_dump(st.cards))
        upd["statement_ids"] = [c.id for c in st.cards]
        name = state.get("company_name") or ""
        if name and st.name not in name and name not in st.name:
            upd["boundary_notes"] = [f"用户提到的“{name}”与代码 {code} 对应的“{st.name}”不一致，以代码为准"]
        upd["company_name"] = st.name
        calc = derive_metrics(h, st.cards, self.d.factory, as_of)
        for skill, key, label in (("market.daily", "market_ids", "行情"), ("valuation.snapshot", "calc_extra", "估值"),
                                  ("announcements.list", "announcement_ids", "公告")):
            try:
                out = h.call(skill, code=code, as_of=as_of)
                upd["evidence"].update(_dump(out.cards))
                if key == "calc_extra":
                    calc += out.cards
                else:
                    upd[key] = [c.id for c in out.cards]
            except (SkillFailed, HarnessDenied, BudgetExceeded) as e:
                upd["unknowns_sys"].append(f"{label}数据缺失：{e}")
                if key != "calc_extra":
                    upd[key] = []
        upd["evidence"].update(_dump(calc))
        upd["calc_ids"] = [c.id for c in calc]
        for c in _cards({"evidence": upd["evidence"]}):
            self.d.repo.put_evidence(c)
        upd["prior"] = self._load_prior(state)
        upd["budget"] = budget.to_dict()
        self._event(state, "collect", f"读取{st.name}财报 {len(st.cards)} 期、公告 {len(upd['announcement_ids'])} 条，算出指标 {len(calc)} 组"
                                      + (f"；载入上一版档案 v{upd['prior']['version']} 用于对照" if upd["prior"] else ""))
        return upd

    def _load_prior(self, state: ResearchState) -> dict | None:
        """上一版档案：只取研究时点不晚于本次 as_of 的最新版本，避免回填研究被未来档案前视污染。"""
        key = dossier_key(state)
        version = self.d.repo.latest_dossier_version_at(key, date.fromisoformat(state["as_of"]))
        prev = self.d.repo.get_dossier(key, version) if version is not None else None
        if prev is None:
            return None
        claims = [{"kind": c.kind, "text": c.text} for c in prev.claims
                  if c.status == "verified" and c.kind in ("fact", "inference")][:12]
        return {"version": prev.version, "as_of": prev.as_of.isoformat(), "claims": claims,
                "triggers": [t.condition for t in prev.review_triggers], "focus": state.get("review_focus", [])}

    # ---------- 搜索子系统 ----------
    def search(self, state: ResearchState) -> dict:
        h, budget = self._h(state, "search")
        res = gather_web(h, self.d, state.get("queries", []), date.fromisoformat(state["as_of"]), state["route"],
                         max_fetch=int(self.d.policy.search.get("max_fetch_per_round", 4)),
                         log=lambda m: self._event(state, "search", m), search_cfg=self._search_cfg(state))
        nb = sum(c.level == "B" for c in res.cards)
        self._event(state, "search", f"搜索 {len(res.searched)} 次，{res.hits} 条结果，合并转载后 {res.clusters} 件事；"
                                     f"读取原文 {res.fetched} 篇，可信来源 {nb} 条、仅作线索 {len(res.cards) - nb} 条")
        return {"evidence": _dump(res.cards), "searched": res.searched, "unknowns_sys": res.unknowns, "budget": budget.to_dict()}

    # ---------- L2：单 Agent 研究员起草 ----------
    def draft(self, state: ResearchState) -> dict:
        h, budget = self._h(state, "draft")
        quick = state["route"] == "quick"
        all_cards = _cards(state)
        statements = sorted(_cards(state, state.get("statement_ids", [])), key=lambda c: c.metrics[0].period if c.metrics else "", reverse=True)[:5]
        web = sorted([c for c in all_cards if c.source_type == "web"], key=lambda c: (c.level, -(c.rerank_score or 0)))
        web = [c for c in web if c.level == "B"] + [c for c in web if c.level == "C"][:10]
        context = (statements + _cards(state, state.get("calc_ids", [])) + _cards(state, state.get("market_ids", []))
                   + _cards(state, state.get("announcement_ids", []))[:15] + web)
        role = self.d.policy.route(state["route"]).llm_role
        safe_q = state.get("safe_question") or state["question"]
        lessons = self._lessons(state, lesson_query(safe_q, context))
        if lessons:
            self._event(state, "draft", f"参考了 {len(lessons)} 条以往经验（{', '.join(c.id for c in lessons)}），只作方法提示，不作为证据")
        try:
            out = h.llm(role, DRAFT_SYSTEM, draft_user_prompt(safe_q, state.get("company_name", ""), state["ticker"],
                        date.fromisoformat(state["as_of"]), context, quick, state.get("prior"), render_lessons(lessons)),
                        DraftOutput, purpose="draft")
        except TaskCancelled:
            raise
        except Exception as e:
            self._event(state, "draft", f"起草失败：{e}")
            return {"status": "failed", "error": f"起草失败：{e}", "budget": budget.to_dict()}
        claims = [Claim(id=f"c{i + 1}", text=c.text, kind=c.kind, evidence_ids=c.evidence_ids).model_dump(mode="json")
                  for i, c in enumerate(out.claims)]
        self.d.repo.save_working(state["task_id"], "draft", f"round{state.get('iteration', 0) + 1}", out.model_dump(mode="json"))
        self._event(state, "draft", f"写出 {len(claims)} 条结论，{len(out.unknowns)} 个问题暂时无法确定")
        return {"draft": out.model_dump(mode="json"), "claims": claims, "budget": budget.to_dict(),
                "lessons_used": [f"{c.id} r{c.rev}" for c in lessons]}

    # ---------- 独立核验 ----------
    def verify(self, state: ResearchState) -> dict:
        evidence = {c.id: c for c in _cards(state)}
        claims = verify_claims([Claim.model_validate(c) for c in state.get("claims", [])], evidence,
                               date.fromisoformat(state["as_of"]))
        counts = {s: sum(c.status == s for c in claims) for s in ("verified", "disputed", "rejected")}
        self._event(state, "verify", f"逐条对照数据核对结论：成立 {counts['verified']}，存疑 {counts['disputed']}，不成立 {counts['rejected']}")
        return {"claims": [c.model_dump(mode="json") for c in claims]}

    # ---------- 质量门控（规则）+ 检查点升级 ----------
    def quality_gate(self, state: ResearchState) -> dict:
        claims = [Claim.model_validate(c) for c in state.get("claims", [])]
        verified_facts = sum(c.status == "verified" and c.kind == "fact" for c in claims)
        disputed = [c for c in claims if c.status == "disputed"]
        rejected = [c for c in claims if c.status == "rejected"]
        draft = state.get("draft") or {}
        gaps = [f"复核：{c.text[:60]}" for c in disputed] + list(draft.get("search_gaps", []))
        rp = self.d.policy.route(state["route"])
        budget = Budget.from_dict(state["budget"])
        notes = [] if state.get("calc_ids") else ["缺少同口径派生指标"]
        weak = disputed or verified_facts < MIN_VERIFIED_FACTS or notes
        upd: dict = {}
        # 升级不看 weak：反方观点达到阈值本身就要升级（此前 weak 短路导致 min_dissents 永远无法单独触发）
        escalate = should_escalate(state.get("routing") or {}, len(disputed), len(draft.get("dissents", [])), self.d.policy.routing)
        if escalate:
            decision = "escalate"
            routing = RouteDecision.model_validate(state["routing"])
            routing = routing.model_copy(update={
                "route": "deep", "source": "escalation", "escalated_from": "standard",
                "reasons": routing.reasons + [f"质量门控检查点：存疑 {len(disputed)}、反方观点 {len(draft.get('dissents', []))}，升级到争议深研"],
            })
            deep = self.d.policy.route("deep")
            upd.update(route="deep", escalated=True, routing=routing.model_dump(mode="json"),
                       budget=Budget(limits=deep.budget, used=budget.used, started_at=budget.started_at).to_dict())
        elif weak and state.get("iteration", 0) < rp.max_iterations and budget.remaining("search_calls") > 0 and draft.get("search_gaps"):
            decision = "need_more"
            upd["iteration"] = state.get("iteration", 0) + 1
        else:
            decision = "partial" if weak else "pass"
        q = QualityResult(decision="partial" if decision == "escalate" else decision,
                          verified=sum(c.status == "verified" for c in claims),
                          disputed=len(disputed), rejected=len(rejected), gaps=gaps, notes=notes)
        upd["quality"] = q.model_dump(mode="json") | {"next": decision}
        self._event(state, "quality_gate", f"{GATE_LABEL[decision]}（核实的事实 {verified_facts} 条，存疑 {len(disputed)} 条）")
        return upd

    # ---------- 提交：复验授权；高后果未解争议需人工复核 ----------
    def commit(self, state: ResearchState) -> dict:
        if self.d.repo.is_cancel_requested(state["task_id"]):
            return {"status": "cancelled"}
        delib = state.get("deliberation")
        boundary = []
        if delib and delib.get("needs_human_review"):
            open_items = [ch["text"] for ch in delib.get("challenges", [])
                          if any(r["challenge_id"] == ch["id"] and r["verdict"] == "unresolved" and r["severity"] == "high"
                                 for r in delib.get("rulings", []))]
            decision = interrupt({
                "type": "review",
                "message": "存在高后果且未解决的争议。确认后将以“部分结论”提交（争议保留在档案中）；拒绝则不提交。",
                "items": open_items,
            })
            if not (isinstance(decision, dict) and decision.get("approve")):
                self._event(state, "commit", "人工复核未通过，档案未提交")
                return {"status": "cancelled"}
            boundary.append("存在未解决的高后果争议，经人工确认以部分结论提交。")
        draft = state["draft"]
        quality = QualityResult.model_validate(state["quality"])
        key = dossier_key(state)
        as_of = date.fromisoformat(state["as_of"])
        parent = self.d.repo.latest_dossier_version_at(key, as_of)   # 版本链按研究时点，不把未来档案当上一版
        claims = [Claim.model_validate(c) for c in state["claims"]]
        used = {i for c in claims for i in c.evidence_ids}
        clue_ids = sorted(i for i, c in state.get("evidence", {}).items() if c["source_type"] == "web")
        status = "committed" if quality.decision == "pass" and not boundary else "partial"
        dossier = Dossier(
            key=key, version=0, parent_version=parent, ticker=state["ticker"],
            name=state.get("company_name", ""), as_of=as_of, question=self.guard.redact_question(state["question"]),
            route=state["route"], status=status, summary=draft.get("summary", ""), claims=claims,
            unknowns=list(dict.fromkeys(draft.get("unknowns", []) + state.get("unknowns_sys", []))),
            dissents=draft.get("dissents", []), review_triggers=draft.get("review_triggers", []),
            evidence_ids=sorted(used | set(state.get("calc_ids", [])) | set(state.get("statement_ids", [])[:5])
                                | set(state.get("market_ids", [])) | set(clue_ids)),
            quality=quality, boundary_notes=state.get("boundary_notes", []) + boundary, task_id=state["task_id"],
            created_at=datetime.now(timezone.utc), routing=state.get("routing"), deliberation=delib,
            prior_review=draft.get("prior_review", []), clue_ids=clue_ids,
            rsi_versions=state.get("rsi_versions") or (self.d.rsi.versions() if self.d.rsi else {}),
            lessons_used=sorted(set(state.get("lessons_used", [])
                                    + [x for r in (delib or {}).get("team", []) for x in r.get("lessons", [])])),
        )
        evidence = {c.id: c for c in _cards(state)}
        prior_v = (state.get("prior") or {}).get("version")

        def _drift_note(old_version, new_version):
            """并发漂移注记：父版本为空时绝不打印 vNone；prior_review 基准只在研究期有对照时说明。"""
            if old_version is None:
                return (f"并发提交：研究开始时无上一版本，提交时已有 v{new_version}，"
                        "已按最新版本续链；本档案未做上一版本对照")
            note = (f"并发提交：上一版本在本任务研究期间由 v{old_version} 变为 v{new_version}，"
                    "已按最新版本续链")
            if prior_v is not None and dossier.prior_review:
                note += f"；prior_review 对照仍基于研究开始时载入的 v{prior_v}"
            return note

        if dossier.parent_version != prior_v:
            # 研究开始载入的对照版本与提交快照不一致：即便首次插入成功也要在档案里自述
            note = _drift_note(prior_v, dossier.parent_version)
            self._event(state, "commit", note)
            dossier = dossier.model_copy(update={"boundary_notes": [*dossier.boundary_notes, note]})
        for attempt in range(3):
            # 同 key 并发提交会同时算出版本号与父版本：撞唯一索引后两者都要按当前库重算，
            # 否则重试后的新版本挂着旧 parent，版本链分叉、diff 基准错位
            new_parent = self.d.repo.latest_dossier_version_at(key, as_of)
            if new_parent != dossier.parent_version:
                note = _drift_note(dossier.parent_version, new_parent)
                self._event(state, "commit", note)
                dossier = dossier.model_copy(update={"boundary_notes": [*dossier.boundary_notes, note]})
            dossier = dossier.model_copy(update={"version": (self.d.repo.latest_dossier_version(key) or 0) + 1,
                                                 "parent_version": new_parent})
            report = render_report(dossier, evidence, Budget.from_dict(state["budget"]))
            dossier = dossier.model_copy(update={"report_md": report})
            if self.d.repo.try_insert_dossier(dossier):
                break
            if attempt == 2:
                raise RuntimeError(f"档案 {key} 版本分配连续冲突 3 次，提交失败")
        self._event(state, "commit", f"已保存为档案 v{dossier.version}（{DOSSIER_STATUS_LABEL.get(dossier.status, dossier.status)}）")
        return {"dossier_key": key, "version": dossier.version, "report_md": report, "status": "done"}
