import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { StatusBadge } from "../components/Badge";
import { chgClass, PageHead, signed, useAskContext } from "../components/Shell";
import { fmtTime, ROUTE_LABEL } from "../format";
import type { Alert, RsiOverview, RsiRun, StrategyCard, Task, WatchRow } from "../types";

const KIND_LABEL: Record<string, string> = { lessons: "经验库", routing: "路由参数", search: "检索模板" };

export function VersionPills({ v }: { v: Partial<Record<string, number | null>> | undefined }) {
  if (!v || !Object.keys(v).length) return <span className="ver-pill base">未记录版本（RSI 前）</span>;
  return (
    <span className="ver-pills">
      {(["lessons", "routing", "search"] as const).map((k) => (
        <span key={k} className={`ver-pill ${v[k] && v[k]! > 1 ? "" : "base"}`}>{KIND_LABEL[k]} {v[k] ? `v${v[k]}` : "基线"}</span>
      ))}
    </span>
  );
}

export default function Workspace() {
  const [watch, setWatch] = useState<WatchRow[] | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [rsi, setRsi] = useState<RsiOverview | null>(null);
  const [runs, setRuns] = useState<RsiRun[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [cards, setCards] = useState<StrategyCard[]>([]);
  const [err, setErr] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const failed: string[] = [];
    const track = (what: string, p: Promise<unknown>) =>
      p.catch((e) => { failed.push(`${what}：${(e as Error).message}`); });
    Promise.all([
      track("提醒", api.watch.overview().then((w) => alive && setAlerts(w.alerts.filter((a) => a.status === "open")))),
      track("策略卡", api.strategy.overview().then((o) => alive && setCards(o.cards.filter((c) => c.status === "review")))),
      track("档案列表", api.watchlist().then((w) => alive && setWatch(w))),   // 失败保留 null：不能伪装成「还没有档案」
      track("任务列表", api.tasks().then((t) => alive && setTasks(t))),
      track("方法版本", api.rsi.overview().then((r) => alive && setRsi(r))),
      track("RSI 运行", api.rsi.runs().then((r) => alive && setRuns(r))),
    ]).then(() => {
      if (!alive) return;
      if (failed.length) {
        setErr(`部分数据加载失败（${failed.join("；")}），稍后自动重试`);
        timer = window.setTimeout(() => setAttempt((x) => x + 1), 5000);
      } else {
        setErr("");
      }
    });
    return () => { alive = false; window.clearTimeout(timer); };
  }, [attempt]);
  const top = watch?.[0];
  useAskContext({
    title: "研究台",
    suggestions: [
      ...(top ? [`复核${top.name} ${top.ticker} 最新一期业绩，与上一版本结论相比有什么变化`] : []),
      ...(watch?.[1] ? [`${watch[1].name} ${watch[1].ticker} 当前估值是否已反映主要风险？多空分歧在哪里`] : []),
      "扣非净利润和归母净利润有什么区别？",
    ],
  }, [top?.key, watch?.[1]?.key]);

  const focus = (watch ?? []).filter((w) => w.status === "partial");
  const review = runs.filter((r) => r.status === "review" || r.status === "approved");
  const running = tasks.filter((t) => ["queued", "running", "waiting_approval"].includes(t.status));

  return (
    <>
      <PageHead title="研究台" sub="行情与估值取自各档案的 A 级证据，截至研究时点，非实时" />
      <div className="page">
        {err && <div className="error">{err}
          <button className="ghost" style={{ marginLeft: 8 }} onClick={() => setAttempt((x) => x + 1)}>立即重试</button>
        </div>}
        <div className="tiles">
          {watch === null ? <div className="muted">加载中…</div> : watch.map((w) => {
            const c = chgClass(w.quote.chg_20d?.value);
            return (
              <Link key={w.key} to={`/dossiers/${w.key}/${w.version}`} className={`tile ${c === "up" ? "is-up" : c === "down" ? "is-down" : "is-flat"}`}>
                <span className="t-name">{w.name} · {w.ticker}</span>
                <span className="t-px">{w.quote.close ? Number(w.quote.close.value).toFixed(2) : "—"}</span>
                <span className="t-chg">近 20 日 {signed(w.quote.chg_20d?.value)}{w.quote.pe_ttm ? ` · PE ${w.quote.pe_ttm.value}` : ""}</span>
                <span className="t-asof">截至 {w.quote.close?.period ?? w.as_of}</span>
              </Link>
            );
          })}
          <Link to="/new" className="tile is-flat" style={{ justifyContent: "center", alignItems: "center", color: "var(--action)" }}>＋ 新研究</Link>
        </div>

        <div className="home-grid" style={{ marginTop: 18 }}>
          <div>
            <section className="card">
              <h3>需要关注</h3>
              {running.map((t) => (
                <Link key={t.task_id} to={`/tasks/${t.task_id}`} className="focus-item">
                  <StatusBadge status={t.status} /><span className="fi-title">{t.request.question}</span>
                  <span className="fi-sub">{t.status === "waiting_approval" ? "需要你确认后才会继续" : "研究进行中"}</span>
                </Link>
              ))}
              {review.map((r) => (
                <Link key={r.run_id} to={`/rsi/runs/${r.run_id}`} className="focus-item">
                  <span className={`badge ${r.status === "approved" ? "st-approved" : "st-review"}`}>{r.status === "approved" ? "已批准 · 待发布" : "待你审批"}</span>
                  <span className="fi-title">{r.change_reason}</span>
                  <span className="fi-sub">{r.status === "approved" ? "已批准，去运行页执行发布" : "六道门控已通过，发布前必须由你批准"}</span>
                </Link>
              ))}
              {alerts.map((a) => (
                <Link key={a.id} to="/watch" className="focus-item">
                  <span className="badge st-open">待复核</span><span className="fi-title">{a.name}：{a.events.map((e) => e.title).join("；")}</span>
                  <span className="fi-sub">订阅发现前提变化，影响 {a.affected.length} 条旧结论</span>
                </Link>
              ))}
              {cards.map((c) => (
                <Link key={`${c.card_id}-${c.version}`} to={`/strategy/${c.card_id}/${c.version}`} className="focus-item">
                  <span className="badge st-review">待你批准</span><span className="fi-title">策略卡 {c.card_id} v{c.version} {c.name}</span>
                  <span className="fi-sub">晋级门控已通过，晋级前必须由你批准</span>
                </Link>
              ))}
              {focus.map((w) => (
                <Link key={w.key} to={`/dossiers/${w.key}/${w.version}`} className="focus-item">
                  <StatusBadge status={w.status} /><span className="fi-title">{w.name} v{w.version}</span>
                  <span className="fi-sub">{w.summary}</span>
                </Link>
              ))}
              {!running.length && !review.length && !focus.length && !alerts.length && !cards.length && <div className="empty">暂无需要关注的事项</div>}
            </section>

            <section className="card">
              <div className="title-row"><h3 style={{ margin: 0 }}>决策档案</h3><Link className="primary-link small" to="/dossiers">全部 →</Link></div>
              {(watch ?? []).map((w) => (
                <Link key={w.key} to={`/dossiers/${w.key}/${w.version}`} className="focus-item">
                  <span className="tag">v{w.version}</span>
                  <span className="fi-title">{w.name} <span className="muted small">{w.ticker} · {w.route ? ROUTE_LABEL[w.route] : ""} · 数据截至 {w.as_of}</span></span>
                  <span className="fi-sub">{w.summary}</span>
                </Link>
              ))}
            </section>
          </div>

          <div>
            <section className="card">
              <div className="title-row"><h3 style={{ margin: 0 }}>方法版本</h3><Link className="primary-link small" to="/rsi">RSI →</Link></div>
              <p className="muted small" style={{ margin: "6px 0 10px" }}>下一次研究会读取这些在用版本，并记入档案。</p>
              {rsi ? (
                <>
                  <VersionPills v={{ lessons: rsi.active.lessons?.version, routing: rsi.active.routing?.version, search: rsi.active.search?.version }} />
                  <div className="stat-row" style={{ gridTemplateColumns: "repeat(3, 1fr)", marginTop: 14 }}>
                    {(["S1", "S2", "S3"] as const).map((s) => (
                      <div key={s} className="stat"><div className="muted small">{s} 待处理坏案例</div><b>{rsi.issues[s].open}</b></div>
                    ))}
                  </div>
                </>
              ) : err ? <div className="muted">方法版本加载失败，稍后自动重试</div> : <div className="muted">加载中…</div>}
            </section>
            <section className="card">
              <h3>最近研究</h3>
              {tasks.slice(0, 6).map((t) => (
                <Link key={t.task_id} to={`/tasks/${t.task_id}`} className="focus-item">
                  <StatusBadge status={t.status} /><span className="fi-title" style={{ fontWeight: 400 }}>{t.request.question}</span>
                  <span className="fi-sub muted small">{fmtTime(t.created_at)}</span>
                </Link>
              ))}
              {!tasks.length && <div className="empty">{err ? "任务列表加载失败，稍后自动重试" : "还没有研究，右侧问顾问试试"}</div>}
            </section>
          </div>
        </div>
      </div>
    </>
  );
}
