import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import { StatusBadge } from "../components/Badge";
import ExpertTeam from "../components/ExpertTeam";
import RoutingCard from "../components/RoutingCard";
import TraceView from "../components/TraceView";
import { BUDGET_LABEL, fmtTime, MODE_LABEL, nodeLabel, ROUTE_LABEL } from "../format";
import type { AuditRecord, ExpertResult, RouteDecision, Task, WorkingItem } from "../types";

const ACTIVE = new Set(["queued", "running"]);
const NAMES: Record<string, string> = {
  fundamental: "基本面研究员", industry: "行业/产业链研究员", valuation: "估值研究员", technical: "技术与资金面研究员",
  event: "事件/新闻研究员", risk: "风险审阅专家", red_team: "反证官（红方）",
};

export default function TaskDetail() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const [task, setTask] = useState<Task | null>(null);
  const [work, setWork] = useState<WorkingItem[]>([]);
  const [audit, setAudit] = useState<AuditRecord[] | null>(null);
  const [error, setError] = useState<{ msg: string; id: string } | null>(null);
  const [actionErr, setActionErr] = useState("");
  const [retrying, setRetrying] = useState(false);
  const [tick, setTick] = useState(0);

  // 失败/取消的任务按已保存的请求（脱敏后的原参数）重新发起一个新任务
  const retry = async () => {
    if (!task || retrying) return;
    setRetrying(true);
    setActionErr("");
    try {
      const { task_id } = await api.research({
        question: task.request.question,
        ticker: task.request.ticker ?? undefined,
        as_of: task.request.as_of ?? undefined,
        route: task.request.route,
      });
      nav(`/tasks/${task_id}`);
    } catch (e) {
      setActionErr((e as Error).message);
    } finally {
      setRetrying(false);          // 成功跳转也要复位：/tasks/:id 参数变化不会重挂载组件
    }
  };

  // 切换到另一条任务时清掉上一条的重试/操作错误（不能并入 [id, tick] 主 effect，否则轮询会清掉刚显示的错误）
  useEffect(() => { setRetrying(false); setActionErr(""); }, [id]);
  const taskRef = useRef<Task | null>(null);

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    taskRef.current = null;
    setTask(null);          // 切换任务时清空旧内容，避免旧任务的慢响应/旧数据挂在新任务页上
    setWork([]);
    setAudit(null);
    setError(null);
    const load = async () => {
      try {
        const [t, w] = await Promise.all([api.task(id), api.workspace(id)]);
        if (!alive) return;
        taskRef.current = t;
        setTask(t);
        setWork(w);
        if (ACTIVE.has(t.status)) timer = window.setTimeout(load, 2000);
        else if (t.status === "waiting_approval") timer = window.setTimeout(load, 5000);   // 跨页面审批后卡片能自己更新
      } catch (e) {
        if (!alive) return;
        if (!taskRef.current) setError({ msg: (e as Error).message, id });   // 首屏失败：错误页
        else timer = window.setTimeout(load, 4000);                          // 轮询抖动：保留内容并重试
      }
    };
    load();
    return () => { alive = false; window.clearTimeout(timer); };
  }, [id, tick]);

  const decide = async (approve: boolean) => {
    setActionErr("");
    try { await api.resume(id, approve); } catch (e) { setActionErr((e as Error).message); }
    setTimeout(() => setTick((x) => x + 1), 800);
  };
  const cancel = async () => {
    if (!confirm("撤销授权并停止这个任务？已完成的步骤不会回滚，未提交的结论不会写入档案。")) return;
    setActionErr("");
    try { await api.cancel(id); } catch (e) { setActionErr((e as Error).message); }
    setTick((x) => x + 1);
  };
  const openAudit = async () => {
    try {
      setAudit(await api.audit(id));
    } catch (e) {
      setError({ msg: (e as Error).message, id });
    }
  };

  if (error && error.id === id) return <div className="page"><div className="error">{error.msg}</div></div>;
  if (!task) return <div className="page muted">加载中…</div>;
  const budget = task.result?.budget;
  const routing = (work.find((w) => w.kind === "routing")?.data ?? task.result?.routing) as RouteDecision | undefined;
  const escalated = task.events?.some((e) => e.node === "quality_gate" && e.message.includes("升级"));
  const finalRouting = task.result?.routing ?? (escalated && routing ? { ...routing, route: "deep", escalated_from: "standard", source: "escalation" } as RouteDecision : routing);
  const team = work.find((w) => w.kind === "team")?.data as { members: { expert: string; questions: string[] }[]; mode: string; rationale: string } | undefined;
  const results = work.filter((w) => w.kind === "expert_result").map((w) => w.data as ExpertResult);
  const pending = team ? [...team.members.map((m) => m.expert), "red_team"].filter((k) => !results.some((r) => r.expert === k)).map((k) => NAMES[k] ?? k) : [];

  return (
    <div className="page narrow">
      <div className="title-row">
        <h1>研究任务</h1>
        <StatusBadge status={task.status} />
        {ACTIVE.has(task.status) && <span className="spinner" aria-label="运行中" />}
        {(ACTIVE.has(task.status) || task.status === "waiting_approval") && (
          <button className="danger" onClick={cancel}>取消 / 撤销授权</button>
        )}
      </div>
      {actionErr && <div className="error">{actionErr}</div>}
      <div className="card">
        <div className="kv"><span>问题</span><b>{task.request.question}</b></div>
        <div className="kv"><span>数据截至</span>{task.request.as_of ? `${task.request.as_of}（只用这天及之前公开的信息）` : "今天（使用最新可得的信息）"}</div>
        <div className="kv"><span>研究方式</span>{task.request.route === "auto" ? "由系统根据问题自动选择" : `${ROUTE_LABEL[task.request.route]}（你指定的）`}</div>
        <div className="kv"><span>任务编号</span><span className="muted small">{task.task_id}</span></div>
      </div>

      {task.status === "waiting_approval" && task.interrupt && (() => {
        const it = task.interrupt!;
        const expired = !!(it.ttl_s && it.issued_at && Date.now() / 1000 > it.issued_at + it.ttl_s);
        return (
        <div className="card approval">
          <h3>{it.type === "review" ? "人工复核：高后果争议未解决" : it.type === "approval" ? "调用审批" : "需要你确认"}</h3>
          <p>{it.message}</p>
          {task.interrupt.items && task.interrupt.items.length > 0 && <ul>{task.interrupt.items.map((x, i) => <li key={i}>{x}</li>)}</ul>}
          {task.interrupt.type === "approval" && (
            <p className="muted small">审批绑定 {task.interrupt.skill} v{task.interrupt.version} 与本次参数，{task.interrupt.ttl_s}s 内有效；参数或版本变化即失效。</p>
          )}
          {expired && <div className="warn-text">审批已过期（超过 {it.ttl_s}s）：此时点同意会被按拒绝处理，请点「拒绝并停止」后重新发起任务。</div>}
          <div className="row-btns">
            <button className="primary" disabled={expired} onClick={() => decide(true)}>
              {it.type === "review" ? "确认以部分结论提交" : "同意，继续"}
            </button>
            <button onClick={() => decide(false)}>{it.type === "review" ? "不提交" : "拒绝并停止"}</button>
          </div>
        </div>
        );
      })()}

      {task.status === "done" && task.result && (
        <div className="card success">
          研究完成，已写入决策档案 <b>{task.result.dossier_key} v{task.result.version}</b>。
          <Link className="primary-link" to={`/dossiers/${task.result.dossier_key}/${task.result.version}`}>查看底稿 →</Link>
        </div>
      )}
      {task.error && (task.status === "failed" || task.status === "cancelled")
        && <div className="error">{task.error}</div>}
      {(task.status === "failed" || task.status === "cancelled") && (
        (task.request.ticker || /(?<!\d)\d{6}(?!\d)/.test(task.request.question)) ? (
          <div className="row-btns" style={{ margin: "8px 0" }}>
            <button className="primary" disabled={retrying} onClick={retry}>
              {retrying ? "正在发起…" : "按原参数重新发起"}
            </button>
            <span className="muted small">用已保存的请求（问题为脱敏后的文本）发起一个新任务</span>
          </div>
        ) : (
          <div className="row-btns" style={{ margin: "8px 0" }}>
            <Link className="button" to="/new">去新研究补填股票代码</Link>
            <span className="muted small">该任务的问题已被脱敏且不含股票代码，无法按原参数重试</span>
          </div>
        )
      )}

      {finalRouting && <RoutingCard r={finalRouting} />}

      {team && (
        <>
          <h2>专家团队 <span className="muted small">{MODE_LABEL[team.mode] ?? team.mode} · 首轮独立取证，互不可见</span></h2>
          <ExpertTeam team={results} pending={ACTIVE.has(task.status) ? pending : []} />
        </>
      )}

      <h2>执行过程</h2>
      <ol className="timeline">
        {(task.events ?? []).map((e, i) => (
          <li key={i} className={e.node.startsWith("expert:") ? "tl-expert" : ""}>
            <span className="tl-node">{nodeLabel(e.node)}</span>
            <span className="tl-msg">{e.message}</span>
            <span className="tl-time muted">{fmtTime(e.ts).slice(-8)}</span>
          </li>
        ))}
        {ACTIVE.has(task.status) && <li className="muted">…</li>}
      </ol>

      {budget && (
        <>
          <h2>资源用量 <span className="muted small">每种研究方式都有上限，超出会停下，避免无限消耗</span></h2>
          <div className="budget">
            {Object.entries(budget.limits).filter(([k]) => k !== "wall_clock_s").map(([k, lim]) => {
              const used = budget.used[k] ?? 0;
              return (
                <div key={k} className="budget-row">
                  <span>{BUDGET_LABEL[k] ?? k}</span>
                  <div className="bar"><div style={{ width: `${Math.min(100, (used / Math.max(lim, 1)) * 100)}%` }} /></div>
                  <span className="num">{Math.round(used).toLocaleString()} / {lim.toLocaleString()}</span>
                </div>
              );
            })}
          </div>
        </>
      )}

      <h2>运行明细 <span className="muted small">每一步调用了哪些数据工具和模型、用了多久，用于事后追查</span></h2>
      <div className="row-btns">
        <a className="button ghost" href={`/api/tasks/${id}/export.json`} download>下载任务明细 (.json)</a>
      </div>
      {audit === null ? (
        <button onClick={openAudit}>展开运行明细</button>
      ) : <TraceView audit={audit} />}
    </div>
  );
}
