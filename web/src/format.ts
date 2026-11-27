export const STATUS_LABEL: Record<string, string> = {
  queued: "排队中", running: "运行中", awaiting_approval: "等待审批", cache_hit: "缓存命中", waiting_approval: "等待你确认", done: "已完成",
  failed: "失败", cancelled: "已取消", committed: "已通过质量门控", partial: "部分结论",
  verified: "已核验", disputed: "存疑", rejected: "驳回", candidate: "待核验", stale: "待复核",
};
export const KIND_LABEL: Record<string, string> = { fact: "事实", inference: "推断", assumption: "假设", scenario: "情景" };
export const NODE_LABEL: Record<string, string> = {
  intake: "理解问题", authorize: "安全检查", route: "选择研究方式", plan: "规划搜索", collect: "读取财报与行情",
  search: "搜索资讯", draft: "撰写结论", verify: "核对数据", quality_gate: "质量检查", commit: "保存档案", runner: "系统",
  team_plan: "组建专家组", blackboard: "汇总观点", cross_exam: "反证质疑", rebuttal: "专家回应", chair: "主席汇总",
  deep_gate: "质量检查", approval: "人工确认", watch: "订阅扫描",
};
export const ROUTE_LABEL: Record<string, string> = { quick: "快查", standard: "标准研究", deep: "争议深研" };
export const SOURCE_ROUTE: Record<string, string> = {
  user: "你指定的", rule: "按规则", model: "系统判断", abstain: "拿不准，选稳妥方式", escalation: "中途升级", fallback: "默认方式",
};
export const LEVEL_LABEL: Record<string, string> = { low: "低", mid: "中", high: "高" };
export const ROLE_LABEL: Record<string, string> = { reason: "深度推理", draft: "快速生成", classify: "快速判断" };
export const BUDGET_LABEL: Record<string, string> = {
  tokens: "模型用量（tokens）", search_calls: "搜索次数", fetch_calls: "读取网页原文次数", llm_calls: "模型调用次数",
};
export const MODE_LABEL: Record<string, string> = { map_reduce: "各自取证后汇总", debate: "专家互相质疑", committee: "投资委员会讨论" };
export const VERDICT_LABEL: Record<string, string> = {
  upheld: "质询成立", rejected: "质询不成立", unresolved: "未决",
  retained: "保留", weakened: "削弱", invalidated: "失效", unverifiable: "无法判断",
};
export const STANCE_LABEL: Record<string, string> = { concede: "承认", rebut: "反驳", revise: "修正" };
export const CHALLENGE_KIND: Record<string, string> = {
  counter_evidence: "反面证据", alternative_explanation: "替代解释", logic_gap: "逻辑漏洞", data_basis: "口径/数据",
};
export const EXPERT_LABEL: Record<string, string> = {
  fundamental: "基本面", industry: "行业", valuation: "估值", technical: "技术面", event: "事件", risk: "风险", red_team: "反证官",
};
export function nodeLabel(node: string) {
  if (node.startsWith("expert:")) return "专家·" + (EXPERT_LABEL[node.slice(7)] ?? node.slice(7));
  return NODE_LABEL[node] ?? node;
}
export const SOURCE_LABEL: Record<string, string> = {
  filing: "财报", market: "行情", announcement: "公告", web: "网页", calc: "计算",
};

export function fmtTime(s?: string | null) {
  if (!s) return "—";
  const d = new Date(s.endsWith("Z") || s.includes("+") ? s : s + "Z");
  return d.toLocaleString("zh-CN", { hour12: false });
}

export function fmtDate(s?: string | null) {
  return s ? s.slice(0, 10) : "未知";
}

export function fmtMetric(value: string, unit: string) {
  const n = Number(value);
  if (unit === "元" && Math.abs(n) >= 1e8) return `${(n / 1e8).toFixed(2)} 亿元`;
  return `${value}${unit === "%" ? "%" : " " + unit}`;
}

export function today() {
  const d = new Date();
  return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
}

/** 系统生成的文字里夹带的内部代号 → 页面显示的中文名（存档原文不变）。 */
export const TERM_LABEL: Record<string, string> = {
  fundamental: "基本面研究员", industry: "行业/产业链研究员", valuation: "估值研究员", technical: "技术与资金面研究员",
  event: "事件/新闻研究员", risk: "风险审阅专家", red_team: "反证官",
  quick: "快查", standard: "标准研究", deep: "争议深研",
  temperature: "校准温度", abstain_below: "拒判阈值",
  evolution: "新失败集", replay: "固定回放集", holdout: "留出集",
  lessons: "经验库", routing: "路由参数", search: "检索模板",
  parent: "基于", sha256: "校验码", token: "模型用量",
  challenge_upheld: "质询成立", verify_disputed: "核验存疑", route_mislabel: "研究方式选错",
  expert_query_weak: "专家搜索词不好用", low_quality_search: "搜索结果质量低", site_gap: "缺少可读取的网站",
  owner: "你", optimizer: "优化器", evaluator: "评测器", gates: "门控", system: "系统",
};
const TERM_RE = new RegExp(`(?<![\\w./:-])(${Object.keys(TERM_LABEL).join("|")})(?![\\w./:-])`, "g");
export function zh(text: string | null | undefined): string {
  if (!text) return "";
  return text.replace(TERM_RE, (w) => TERM_LABEL[w])
    .replace(/门禁/g, "门控").replace(/(质量|噪声|成本|回归|审批)门(?!控)/g, "$1门控");
}

// A 股代码形态（与后端 nodes._CODE_SPACED 同口径）：数字主体只认 ASCII [0-9]，前后边界是
// Unicode 数字（\p{Nd}，非 ASCII 数字紧邻时不算代码），代码内允许一个半角/全角空格。
// 注：\p{Nd} 的具体码位随 JS 引擎的 Unicode 数据版本演进，与 Python \d 理论上可能有差异。
const A_SHARE_CODE = /(?<![\p{Nd}])(?:[036][0-9]{2}[ 　]?[0-9]{3}|[036][0-9]{5}|[48][0-9]{5}|92[0-9]{4})(?![\p{Nd}])/u;
const UNIT_COMPOUND = ["万元", "亿元", "千元", "百万元", "千万元", "万手", "万股", "亿股", "万亿"];
const MONEY_WORDS = ["美元", "港元", "港币", "人民币"];
const UNIT_ALWAYS = "份块%％";
const FORMAT_CHARS = "\u200b\u200c\u200d\ufeff";
const MONEY_HEADS = "￥¥$€";
const MONEY_STARTERS = "元万亿千%％手份块股";
const isCjk = (ch: string) => !!ch && ch >= "一" && ch <= "鿿";
const skipFormat = (s: string, k: number) => {
  while (k < s.length && (/[\s\u001c-\u001f\u0085]/.test(s[k]) || FORMAT_CHARS.includes(s[k]))) k += 1;
  return k;
};
const skipFmtOnly = (s: string, k: number) => {
  while (k < s.length && FORMAT_CHARS.includes(s[k])) k += 1;
  return k;
};
const UNIT_EXTRA = ["万份", "亿份", "千份", "千股", "千万", "百亿", "千亿", "万美元", "万港元", "亿港元"];
const unitAt = (s: string, k0: number): boolean => {
  const k = skipFormat(s, k0);
  const rest = s.slice(k);
  if (UNIT_COMPOUND.some((w) => rest.startsWith(w)) || UNIT_EXTRA.some((w) => rest.startsWith(w))
    || MONEY_WORDS.some((w) => rest.startsWith(w))) return true;
  const ch = rest.slice(0, 1);
  if (ch && UNIT_ALWAYS.includes(ch)) return true;
  if (ch === "元") return !["器", "宇", "素"].includes(rest[1] ?? "");
  if (ch === "手") return !isCjk(rest[1] ?? "");
  if (ch && "万亿千".includes(ch) && (rest.length === 1 || !isCjk(rest[1] ?? ""))) return true;
  return false;
};
const digitsMoneyContext = (s: string, k0: number): boolean => {
  const k = skipFormat(s, k0);
  if (k >= s.length) return true;
  const ch = s[k];
  if (!isCjk(ch)) return true;
  if (MONEY_STARTERS.includes(ch)) return true;
  return ["百万", "百亿", "百元", "美元", "港元", "港币", "人民币"].some((w) => s.startsWith(w, k));
};
/** 与后端 nodes._money_tail 同口径：复合单位（含万亿/千万元/百万元）先于裸乘数判断；
 *  小数（1-2 位后接金额语境；多位数仅在紧跟金额单位时算）；份/块/% 恒为数量；
 *  手/元 词首例外只跳过零宽字符（'手 机' 不是词）。 */
function moneyTail(s: string, i: number): boolean {
  const j = skipFormat(s, i);
  if (j >= s.length) return false;
  const ch = s[j];
  if (ch === "." || ch === "．") {
    const k = skipFormat(s, j + 1);
    const m = /^[0-9]+/.exec(s.slice(k));
    if (!m) return false;
    const after = k + m[0].length;
    if (unitAt(s, after)) return true;
    return m[0].length <= 2 && digitsMoneyContext(s, after);
  }
  if (ch === "," || ch === "，") {
    const m = /^[,，]([0-9]+)/.exec(s.slice(j));
    if (!m) return false;
    const end = j + m[0].length;
    if (m[1].length > 2) return unitAt(s, end);   // 千分位/长小数：只有紧跟单位才算金额
    return digitsMoneyContext(s, end);
  }
  if (UNIT_COMPOUND.some((w) => s.startsWith(w, j))) return true;
  if (ch === "万" || ch === "亿" || ch === "千") {
    const k = skipFormat(s, j + 1);
    if (k >= s.length || !isCjk(s[k])) return true;
    return ["元", "美元", "港元", "港币", "人民币", "份", "手", "股"].some((w) => s.startsWith(w, k));
  }
  if (MONEY_WORDS.some((w) => s.startsWith(w, j))) return true;
  if (UNIT_ALWAYS.includes(ch)) return true;
  if (ch === "手" || ch === "元") {
    const nxt = s.slice(skipFmtOnly(s, j + 1), skipFmtOnly(s, j + 1) + 1);
    if (ch === "手") return !isCjk(nxt);
    return !["器", "宇", "素"].includes(nxt);
  }
  return false;
}
function moneyHead(s: string, i: number): boolean {
  let j = i - 1;
  while (j >= 0 && (/[\s\u001c-\u001f\u0085]/.test(s[j]) || FORMAT_CHARS.includes(s[j]))) j -= 1;
  if (j >= 0 && MONEY_HEADS.includes(s[j])) return true;
  const start = j + 1 - 3;
  if (start >= 0 && s.slice(start, j + 1) === "人民币") {
    const before = s.slice(0, start);
    return !["数字", "离岸", "在岸", "跨境"].some((w) => before.endsWith(w));
  }
  return false;
}
export function normDigits(q: string): string {
  return q.replace(/[０-９]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0xfee0));
}
export function ashareCodes(q: string): string[] {
  const s = normDigits(q);
  const re = new RegExp(A_SHARE_CODE.source, "gu");
  const out: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(s))) {
    if (moneyTail(s, m.index + m[0].length) || moneyHead(s, m.index)) continue;
    out.push(m[0].replace(/\s+/g, ""));
  }
  return out;
}
export function hasAshareCode(q: string): boolean {
  return ashareCodes(q).length > 0;
}
/** 页面附带的 ticker 只在问题没有点名其它 A 股代码时使用（财务数字 208448 万元不是代码）。
 *  公司名维度不在这里判定：R100a 审计的两条 important 证明前端按档案全名做仲裁会因简称/
 *  网站名偏差把研究翻转到别的股票，名称与页面标的的冲突留给 intake 的事件/boundary_note 呈现。 */
export function resolveSubmitTicker(question: string, pageTicker: string | undefined): string | undefined {
  if (!pageTicker) return undefined;
  return ashareCodes(question).some((c) => c !== pageTicker) ? undefined : pageTicker;
}
