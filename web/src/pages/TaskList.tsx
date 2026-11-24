import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { StatusBadge } from "../components/Badge";
import { fmtTime } from "../format";
import type { Task } from "../types";

const RUNNING = new Set(["queued", "running"]);
const WAITING = new Set(["waiting_approval"]);

type Filter = "all" | "running" | "waiting" | "finished";

export default function TaskList() {
  const [tasks, setTasks] = useState<Task[] | null>(null);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState<Filter>("all");

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const load = async () => {
      try {
        const rows = await api.tasks();
        if (!alive) return;
        setTasks(rows);
        setError("");
        if (rows.some((t) => RUNNING.has(t.status) || WAITING.has(t.status))) {
          timer = window.setTimeout(load, 3000);      // 有活跃任务时自动刷新，进度不用手动刷页面
        }
      } catch (e) {
        if (!alive) return;
        setError((e as Error).message);
        timer = window.setTimeout(load, 5000);        // 拉取抖动：保留列表并稍后重试
      }
    };
    setTasks(null);
    load();
    return () => { alive = false; window.clearTimeout(timer); };
  }, []);

  const counts = useMemo(() => {
    const all = tasks ?? [];
    return {
      all: all.length,
      running: all.filter((t) => RUNNING.has(t.status)).length,
      waiting: all.filter((t) => WAITING.has(t.status)).length,
      finished: all.filter((t) => !RUNNING.has(t.status) && !WAITING.has(t.status)).length,
    };
  }, [tasks]);

  const shown = useMemo(() => (tasks ?? []).filter((t) =>
    filter === "all" ? true :
    filter === "running" ? RUNNING.has(t.status) :
    filter === "waiting" ? WAITING.has(t.status) :
    !RUNNING.has(t.status) && !WAITING.has(t.status)), [tasks, filter]);

  const TABS: [Filter, string, number][] = [
    ["all", "全部", counts.all],
    ["running", "运行中", counts.running],
    ["waiting", "待确认", counts.waiting],
    ["finished", "已结束", counts.finished],
  ];

  return (
    <div className="page narrow">
      <h1>任务</h1>
      {error && <div className="error">{error}（稍后自动重试）</div>}
      {tasks === null ? <p className="muted">加载中…</p> : tasks.length === 0 ? (
        <p className="muted">还没有任务。<Link to="/new">新建研究</Link></p>
      ) : (
        <>
          <div className="tabs">
            {TABS.map(([key, label, n]) => (
              <button key={key} className={filter === key ? "active" : ""} onClick={() => setFilter(key)}>
                {label} ({n})
              </button>
            ))}
          </div>
          {shown.length === 0 ? <p className="muted">没有匹配的任务。</p> : (
            <div className="list">
              {shown.map((t) => (
                <Link key={t.task_id} to={`/tasks/${t.task_id}`} className="list-item">
                  <StatusBadge status={t.status} />
                  <span className="grow">{t.request.question}</span>
                  <span className="muted small">{fmtTime(t.created_at)}</span>
                </Link>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
