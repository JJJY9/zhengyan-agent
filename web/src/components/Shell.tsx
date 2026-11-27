import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { api } from "../api";
import { nodeLabel, resolveSubmitTicker, ROUTE_LABEL, today } from "../format";
import type { Evidence, Task, WatchRow } from "../types";
import EvidenceCard from "./EvidenceCard";
import {
  IconArrowUp, IconBell, IconChart, IconChat, IconChevron, IconClose, IconCpu, IconFlask, IconFolder, IconHome, IconList, IconLoop, IconMoon,
  IconPanel, IconPanelR, IconPlus, IconSun, IconUser, Sparkle,
} from "./Icons";

// ---------- 全局上下文：问顾问的页面语境、溯源抽屉 ----------
export interface AskContext { title: string; suggestions: string[]; ticker?: string }
interface ShellApi {
  setAskContext: (c: AskContext | null) => void;
  ask: (question: string, opts?: { ticker?: string; route?: string }) => void;
  openEvidence: (items: Evidence[], title?: string) => void;
  refresh: () => void;
}
const ShellCtx = createContext<ShellApi | null>(null);
export const useShell = () => useContext(ShellCtx)!;

/** 页面声明它的"问顾问"语境（标题 + 推荐问题）。 */
export function useAskContext(c: AskContext | null, deps: unknown[]) {
  const { setAskContext } = useShell();
  useEffect(() => {
    setAskContext(c);
    return () => setAskContext(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

const DEFAULT_ASK: AskContext = {
  title: "研究台",
  suggestions: [
    "复核贵州茅台 600519 最新一期半年报业绩，利润质量如何",
    "宁德时代 300750 当前估值是否已反映价格战风险？多空分歧在哪里",
    "合同负债大幅下降一定说明需求走弱吗？",
  ],
};

function store<T>(key: string, fallback: T): T {
  try { const v = localStorage.getItem(key); return v ? JSON.parse(v) as T : fallback; } catch { return fallback; }
}
function save(key: string, v: unknown) { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* 隐私模式 */ } }

function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark" | null>(() => store("theme", null));
  useEffect(() => {
    if (theme) document.documentElement.setAttribute("data-theme", theme);
    else document.documentElement.removeAttribute("data-theme");
  }, [theme]);
  const dark = theme ? theme === "dark" : window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  return { dark, toggle: () => { const t = dark ? "light" : "dark"; setTheme(t); save("theme", t); } };
}

export function chgClass(v?: string) {
  const n = Number(v);
  if (!v || Number.isNaN(n) || n === 0) return "";
  return n > 0 ? "up" : "down";
}
export const signed = (v?: string, unit = "%") => (v == null ? "—" : `${Number(v) > 0 ? "+" : ""}${Number(v).toFixed(2)}${unit}`);

// ---------- 侧栏 ----------
const SIDEBAR_LIMIT = 5;

function MoreToggle({ total, open, onToggle }: { total: number; open: boolean; onToggle: () => void }) {
  if (total <= SIDEBAR_LIMIT) return null;
  return <button className="sb-more" onClick={onToggle}>{open ? "收起" : `显示更多（还有 ${total - SIDEBAR_LIMIT} 个）`}</button>;
}

function Sidebar({ collapsed, onCollapse, watch, tasks, pendingRsi, watchErr, onRetryWatch }: {
  collapsed: boolean; onCollapse: () => void; watch: WatchRow[] | null; tasks: Task[]; pendingRsi: number;
  watchErr: string; onRetryWatch: () => void;
}) {
  const openAlerts = (watch ?? []).reduce((n, w) => n + (w.open_alerts ?? 0), 0);
  const activeTasks = tasks.filter((t) => ["queued", "running", "waiting_approval"].includes(t.status)).length;
  const loc = useLocation();
  const [moreWatch, setMoreWatch] = useState(false);
  const [moreTasks, setMoreTasks] = useState(false);
  const [loginRequired, setLoginRequired] = useState(false);
  useEffect(() => { api.session().then((x) => setLoginRequired(x.required)).catch(() => undefined); }, []);
  const logout = async () => { await api.logout().catch(() => undefined); location.assign("/"); };
  // 收起时只显示前几个；正在看的档案即使排在后面也保留
  const visibleWatch = (watch ?? []).filter((w, i) => moreWatch || i < SIDEBAR_LIMIT || loc.pathname.startsWith(`/dossiers/${w.key}/`));
  const { dark, toggle } = useTheme();
  const nav = [
    { to: "/new", label: "新研究", icon: IconChat },
    { to: "/workspace", label: "研究台", icon: IconHome },
    { to: "/dossiers", label: "决策档案", icon: IconFolder },
    { to: "/tasks", label: "研究任务", icon: IconList },
    { to: "/watch", label: "订阅与复核", icon: IconBell, count: openAlerts },
    { to: "/profile", label: "我的约束", icon: IconUser },
    { to: "/strategy", label: "策略与模拟盘", icon: IconFlask },
    { to: "/evals", label: "评测", icon: IconChart },
    { to: "/rsi", label: "RSI", icon: IconLoop, count: pendingRsi },
    { to: "/system", label: "系统介绍", icon: IconCpu },
  ];
  return (
    <aside className="sidebar">
      <div className="sb-top">
        <Link to="/" className="sb-logo" title="返回产品首页"><Sparkle className="sb-mark" /><span className="sb-label">证研</span></Link>
        <span className="sb-beta">V6</span>
        <button className="icon" onClick={onCollapse} title={collapsed ? "展开侧栏" : "收起侧栏"} aria-label="切换侧栏"><IconPanel width={18} height={18} /></button>
      </div>
      <nav className="sb-nav">
        {nav.map(({ to, label, icon: I, count }) => (
          <NavLink key={to} to={to} className={({ isActive }) => `sb-item ${isActive ? "active" : ""}`} title={label}>
            <I /><span className="sb-label">{label}</span>{!!count && <span className="count sb-label">{count}</span>}
          </NavLink>
        ))}
      </nav>
      <div className="sb-scroll">
        <div className="sb-sec"><span className="grow">我的档案</span></div>
        <div className="wl-head"><span>标的（{watch === null ? (watchErr ? "—" : "…") : watch.length}）</span><span title="研究当时取到的近 20 个交易日涨跌幅，不是实时行情">近 20 日涨跌</span></div>
        <div className="wl-list">
          {watchErr && (
            <div className="wl-head" style={{ color: "var(--danger, #c33)" }} title={watchErr}>档案列表加载失败（稍后自动重试）
              <button className="ghost small" onClick={onRetryWatch}>重试</button></div>
          )}
          {watch === null ? (watchErr ? null : <div className="wl-head">加载中…</div>) : watch.length === 0 ? (
            watchErr ? null : <div className="wl-head">还没有档案</div>
          ) : visibleWatch.map((w) => {
            const chg = w.quote.chg_20d?.value;
            return (
              <Link key={w.key} to={`/dossiers/${w.key}/${w.version}`}
                className={`wl-row ${loc.pathname.startsWith(`/dossiers/${w.key}/`) ? "active" : ""}`}
                title={`行情截至 ${w.quote.close?.period ?? w.as_of}（A 级证据，非实时）`}>
                <div><div className="wl-name">{w.name || w.ticker}{w.open_alerts > 0 && <span className="count" title="有待复核的变化">{w.open_alerts}</span>}</div><div className="wl-code">{w.ticker} · v{w.version}</div></div>
                <div><div className={`wl-chg ${chgClass(chg)}`}>{signed(chg)}</div>
                  <div className="wl-px">{w.quote.close ? Number(w.quote.close.value).toFixed(2) : "—"}</div></div>
              </Link>
            );
          })}
          <MoreToggle total={watch?.length ?? 0} open={moreWatch} onToggle={() => setMoreWatch(!moreWatch)} />
        </div>
        <div className="sb-sec"><span className="grow">最近研究</span>
          {activeTasks > 0 && <span className="count" title="进行中或等待确认的任务">{activeTasks}</span>}
        </div>
        <div className="sb-hist">
          {tasks.slice(0, moreTasks ? 20 : SIDEBAR_LIMIT).map((t) => <Link key={t.task_id} to={`/tasks/${t.task_id}`} title={t.request.question}>{t.request.question}</Link>)}
          <MoreToggle total={Math.min(tasks.length, 20)} open={moreTasks} onToggle={() => setMoreTasks(!moreTasks)} />
        </div>
      </div>
      <div className="sb-foot">
        <div className="avatar">我</div><span className="name">本地研究员</span>
        {loginRequired && <button className="sb-logout" onClick={logout} title="退出登录">退出</button>}
        <button className="icon theme" onClick={toggle} title="切换明暗" aria-label="切换明暗">{dark ? <IconSun width={16} /> : <IconMoon width={16} />}</button>
      </div>
    </aside>
  );
}

// ---------- 问顾问 ----------
interface Turn { task_id: string; question: string; route: string; as_of: string; ts: number }
const ROUTES = [["auto", "自动"], ["quick", "快查"], ["standard", "标准"], ["deep", "深研"]] as const;
const ACTIVE = new Set(["queued", "running"]);

function TurnView({ turn, onDone }: { turn: Turn; onDone: () => void }) {
  const [task, setTask] = useState<Task | null>(null);
  const [summary, setSummary] = useState<string>("");
  const [decideErr, setDecideErr] = useState("");
  const [taskErr, setTaskErr] = useState("");
  const [tick, setTick] = useState(0);
  const doneRef = useRef(false);
  useEffect(() => {
    let alive = true, timer = 0;
    const load = async () => {
      try {
        const t = await api.task(turn.task_id);
        if (!alive) return;
        setTask(t);
        if (ACTIVE.has(t.status)) timer = window.setTimeout(load, 2000);
        else if (t.status === "waiting_approval") timer = window.setTimeout(load, 5000);
        else if (t.status === "done" && t.result && !doneRef.current) {
          doneRef.current = true;
          onDone();
          api.dossier(t.result.dossier_key, t.result.version).then((d) => alive && setSummary(d.dossier.summary)).catch(() => undefined);
        }
      } catch (e) {
        if ((e as Error & { status?: number }).status === 404) {
          setTaskErr("任务不存在或已清理（可能已被归档），请重新发起");
          return;                          // 404 不重试：永久轮询没有意义
        }
        timer = window.setTimeout(load, 4000);
      }
    };
    load();
    return () => { alive = false; window.clearTimeout(timer); };
  }, [turn.task_id, onDone, tick]);

  const decide = async (approve: boolean) => {
    setDecideErr("");
    try { await api.resume(turn.task_id, approve); } catch (e) { setDecideErr((e as Error).message); }
    setTask((t) => t && { ...t, status: "running", interrupt: null });
    setTick((x) => x + 1);        // 审批后任务回到 running：重新接上轮询，spinner 和结果才能继续更新
  };
  const events = task?.events ?? [];
  const shown = events.slice(-6);
  return (
    <>
      <div className="msg-user">{turn.question}</div>
      <div className="msg-meta">{ROUTE_LABEL[turn.route] ?? "自动选择方式"} · 数据截至 {turn.as_of}</div>
      <div className="msg-bot">
        <div className="who"><Sparkle width={16} height={16} />顾问
          {task && ACTIVE.has(task.status) && <span className="spinner inline" />}
          {task?.result?.route && <span className={`route-pill rt-${task.result.route}`}>{ROUTE_LABEL[task.result.route]}</span>}
        </div>
        {!task ? <span className={taskErr ? "error" : "muted"}>{taskErr || "提交中…"}</span> : (
          <ul className="steps">
            {events.length > shown.length && <li className="more">… 已完成 {events.length - shown.length} 步</li>}
            {shown.map((e, i) => <li key={i}><b>{nodeLabel(e.node)}</b>{e.message}</li>)}
          </ul>
        )}
        {task?.status === "waiting_approval" && task.interrupt && (() => {
          const it = task.interrupt!;
          const expired = !!(it.ttl_s && it.issued_at && Date.now() / 1000 > it.issued_at + it.ttl_s);
          return (
            <div className="card approval" style={{ margin: 0 }}>
              <div>{it.message}</div>
              {it.items?.length ? <ul>{it.items.map((x, i) => <li key={i}>{x}</li>)}</ul> : null}
              {expired && <div className="warn-text">审批已过期：此时点同意会被按拒绝处理，请拒绝后重新发起。</div>}
              {decideErr && <div className="error">{decideErr}</div>}
              <div className="row-btns" style={{ marginTop: 8 }}>
                <button className="primary" disabled={expired} onClick={() => decide(true)}>同意，继续</button>
                <button onClick={() => decide(false)}>拒绝</button>
              </div>
            </div>
          );
        })()}
        {task?.status === "done" && task.result && (
          <div className="result-card">
            <div className="title-row"><b>档案 {task.result.dossier_key} v{task.result.version}</b>
              <Link className="primary-link small" to={`/dossiers/${task.result.dossier_key}/${task.result.version}`}>查看底稿 →</Link></div>
            {summary && <div className="summary">{summary}</div>}
          </div>
        )}
        {task && ["failed", "cancelled"].includes(task.status) && (
          <div className="error" style={{ margin: 0 }}>{task.status === "failed" ? task.error : "任务已取消"} · <Link to={`/tasks/${turn.task_id}`}>详情</Link></div>
        )}
        {task && ACTIVE.has(task.status) && <Link className="small" to={`/tasks/${turn.task_id}`}>查看完整过程 →</Link>}
      </div>
    </>
  );
}

function AskPanel({ ctx, onClose, turns, onSubmit, onDone, onClear }: {
  ctx: AskContext; onClose: () => void; turns: Turn[]; onSubmit: (q: string, route: string, asOf: string) => Promise<void>;
  onDone: () => void; onClear: () => void;
}) {
  const [q, setQ] = useState("");
  const [route, setRoute] = useState("auto");
  const [asOf, setAsOf] = useState(today());
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [glow, setGlow] = useState(true);
  const body = useRef<HTMLDivElement>(null);
  useEffect(() => { body.current?.scrollTo({ top: body.current.scrollHeight, behavior: "smooth" }); }, [turns.length]);

  const send = async (text = q) => {
    if (text.trim().length < 2 || busy) return;
    setBusy(true); setErr("");
    try { await onSubmit(text.trim(), route, asOf || today()); setQ(""); } catch (e) { setErr((e as Error).message); }
    setBusy(false);
  };
  return (
    <aside className="ask">
      <div className="ask-head"><Sparkle width={16} height={16} />问顾问
        <span className="sp">
          <button className="icon" title="新对话" aria-label="新对话" onClick={() => { onClear(); setGlow(true); }}><IconPlus width={18} /></button>
          <button className="icon" title="收起" aria-label="收起问顾问" onClick={onClose}><IconPanelR width={18} /></button>
        </span>
      </div>
      <div className="ask-body" ref={body}>
        {turns.length === 0 ? (
          <div className="ask-empty">
            <Sparkle className="sparkle" />
            <h3>关于{ctx.title}，你可能想问</h3>
            {ctx.suggestions.map((s) => (
              <button key={s} className="fu" onClick={() => send(s)} disabled={busy}><span>{s}</span><IconChevron /></button>
            ))}
          </div>
        ) : turns.map((t) => <TurnView key={t.task_id} turn={t} onDone={onDone} />)}
      </div>
      {err && <div className="error" style={{ margin: "0 14px" }}>{err}</div>}
      <div className={`composer ${glow ? "glow" : ""}`} onAnimationEnd={() => setGlow(false)}>
        <textarea value={q} onChange={(e) => setQ(e.target.value)} placeholder={`对${ctx.title}有疑问，问顾问…`} rows={2} maxLength={500}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(); } }} />
        <div className="composer-bar">
          <div className="seg" role="group" aria-label="研究流程">
            {ROUTES.map(([k, l]) => <button key={k} className={route === k ? "on" : ""} onClick={() => setRoute(k)} type="button">{l}</button>)}
          </div>
          <input type="date" value={asOf} max={today()} onChange={(e) => setAsOf(e.target.value)} title="数据截至哪一天（默认今天）" aria-label="数据截至" />
          <button className="send" onClick={() => send()} disabled={busy || q.trim().length < 2} aria-label="发送"><IconArrowUp /></button>
        </div>
      </div>
      <div className="ask-note">结论由 AI 生成并经规则核验，仅供研究，不构成投资建议</div>
    </aside>
  );
}

// ---------- 溯源抽屉 ----------
function EvidenceDrawer({ items, title, onClose }: { items: Evidence[]; title: string; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <>
      <div className="drawer-mask" onClick={onClose} />
      <div className="drawer" role="dialog" aria-label="溯源">
        <div className="drawer-h"><h3>溯源</h3><span className="muted small">{title}</span>
          <button className="icon" onClick={onClose} aria-label="关闭"><IconClose width={18} /></button></div>
        <div className="drawer-b">
          {items.length === 0 ? <div className="empty">没有可展示的证据</div> : items.map((e) => <EvidenceCard key={e.id} ev={e} />)}
          <p className="muted small">A 级＝结构化财报/行情/公告/系统计算，可支撑结论；B 级＝权威网页且有快照；C 级＝线索，只能触发补证。</p>
        </div>
      </div>
    </>
  );
}

// ---------- 外壳 ----------
export default function Shell({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState<boolean>(() => store("sb-collapsed", false));
  const [askOpen, setAskOpen] = useState<boolean>(() => store("ask-open", window.innerWidth > 1180));
  const [ctx, setCtx] = useState<AskContext | null>(null);
  const [turns, setTurns] = useState<Turn[]>(() => store("ask-turns", []));
  const [drawer, setDrawer] = useState<{ items: Evidence[]; title: string } | null>(null);
  const [watch, setWatch] = useState<WatchRow[] | null>(null);
  const [watchErr, setWatchErr] = useState("");
  const [watchTick, setWatchTick] = useState(0);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [pendingRsi, setPendingRsi] = useState(0);

  const loadWatch = useCallback(() => {
    api.watchlist().then((w) => { setWatch(w); setWatchErr(""); })
      .catch((e) => { setWatchErr((e as Error).message); setWatchTick((x) => x + 1); });   // 失败保留旧状态，不能伪装成「还没有档案」
  }, []);
  const refresh = useCallback(() => {
    loadWatch();
    api.tasks().then(setTasks).catch(() => undefined);
    api.rsi.runs().then((rs) => setPendingRsi(rs.filter((r) => r.status === "review" || r.status === "approved").length)).catch(() => undefined);
  }, [loadWatch]);
  useEffect(refresh, [refresh]);
  // 档案列表加载失败：5 秒后自动重试（恢复后侧栏自动纠正）。
  // watchTick 每次失败递增：同一错误文案不会触发 React 同值 bail out，定时器才能持续重新武装。
  useEffect(() => {
    if (!watchErr) return;
    const timer = window.setTimeout(loadWatch, 5000);
    return () => window.clearTimeout(timer);
  }, [watchErr, watchTick, loadWatch]);
  // 有进行中/待确认任务时每 4 秒刷新一次侧栏计数；全部结束后自动停止
  useEffect(() => {
    if (!tasks.some((t) => ["queued", "running", "waiting_approval"].includes(t.status))) return;
    const timer = window.setTimeout(refresh, 4000);
    return () => window.clearTimeout(timer);
  }, [tasks, refresh]);
  useEffect(() => save("sb-collapsed", collapsed), [collapsed]);
  useEffect(() => save("ask-open", askOpen), [askOpen]);
  useEffect(() => save("ask-turns", turns.slice(-8)), [turns]);

  const submit = useCallback(async (question: string, route: string, as_of: string, ticker?: string) => {
    // 页面附带的 ticker 只在问题没有点名其它 A 股代码时使用；财务数字（208448 万元）不是代码。
    // 名称与页面标的的冲突不在这里仲裁（R100a 审计：前端按档案全名仲裁会因简称/网站名翻转研究标的）。
    const t = resolveSubmitTicker(question, ticker);
    const { task_id } = await api.research({ question, route, as_of, ticker: t });
    setTurns((ts) => [...ts, { task_id, question, route, as_of, ts: Date.now() }]);
    setAskOpen(true);
    refresh();
  }, [refresh]);

  const apiValue = useMemo<ShellApi>(() => ({
    setAskContext: setCtx,
    ask: (q, o) => { submit(q, o?.route ?? "auto", today(), o?.ticker).catch(() => navigate("/new")); },
    openEvidence: (items, title = "") => setDrawer({ items, title }),
    refresh,
  }), [submit, refresh, navigate]);

  return (
    <ShellCtx.Provider value={apiValue}>
      <div className={`ws ${collapsed ? "side-collapsed" : ""} ${askOpen ? "" : "no-ask"}`}>
        <Sidebar collapsed={collapsed} onCollapse={() => setCollapsed((c) => !c)} watch={watch} tasks={tasks}
                 pendingRsi={pendingRsi} watchErr={watchErr} onRetryWatch={loadWatch} />
        <main className="main">
          {!askOpen && (
            <button className="ask-fab" onClick={() => setAskOpen(true)} title="展开右侧问顾问面板" aria-label="打开问顾问">
              <Sparkle width={16} height={16} />问顾问<IconChevron width={14} style={{ transform: "rotate(180deg)" }} /></button>
          )}
          {children}
          <footer className="footer">仅为研究与信息整理，不构成投资建议，不承诺收益；系统不执行任何交易或资金动作。</footer>
        </main>
        <AskPanel ctx={ctx ?? DEFAULT_ASK} onClose={() => setAskOpen(false)} turns={turns}
          onSubmit={(q, r, a) => submit(q, r, a, ctx?.ticker)} onDone={refresh} onClear={() => setTurns([])} />
      </div>
      {drawer && <EvidenceDrawer {...drawer} onClose={() => setDrawer(null)} />}
    </ShellCtx.Provider>
  );
}

export function PageHead({ title, sub, right }: { title: React.ReactNode; sub?: React.ReactNode; right?: React.ReactNode }) {
  return <div className="mk-head"><h1>{title}</h1>{sub && <span className="sub">{sub}</span>}{right && <div className="right">{right}</div>}</div>;
}
