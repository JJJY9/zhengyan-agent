import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { StatusBadge } from "../components/Badge";
import { fmtTime } from "../format";

type Row = Awaited<ReturnType<typeof api.dossiers>>[number];

const STATUS_TABS: [string, string][] = [
  ["all", "全部"],
  ["committed", "已通过质量门控"],
  ["partial", "部分结论"],
];

export default function DossierList() {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [error, setError] = useState("");
  const [q, setQ] = useState("");
  const [status, setStatus] = useState("all");

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const load = async () => {
      try {
        const data = await api.dossiers();
        if (!alive) return;
        setRows(data);
        setError("");
      } catch (e) {
        if (!alive) return;
        setError((e as Error).message);
        timer = window.setTimeout(load, 5000);          // 拉取抖动：保留列表并稍后重试
      }
    };
    load();
    return () => { alive = false; window.clearTimeout(timer); };
  }, []);

  const counts = useMemo(() => {
    const all = rows ?? [];
    return {
      all: all.length,
      committed: all.filter((r) => r.status === "committed").length,
      partial: all.filter((r) => r.status === "partial").length,
    };
  }, [rows]);

  const shown = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return (rows ?? []).filter((r) =>
      (status === "all" || r.status === status) &&
      (!needle || r.ticker.includes(needle) || (r.name ?? "").toLowerCase().includes(needle)
        || (r.summary ?? "").toLowerCase().includes(needle)));
  }, [rows, q, status]);

  return (
    <div className="page">
      <h1>决策档案</h1>
      <p className="muted">每个标的一份档案，每次研究生成新版本，旧版本保留。</p>
      <div className="row-btns">
        <a className="button ghost" href="/api/dossiers/export.zip" download>导出全部底稿 (.zip)</a>
        <input placeholder="按代码、名称或结论搜索…" value={q} onChange={(e) => setQ(e.target.value)} style={{ minWidth: 240 }} />
      </div>
      {error && <div className="error">{error}（稍后自动重试）</div>}
      {rows === null ? <p className="muted">加载中…</p> : rows.length === 0 ? (
        <p className="muted">还没有档案。<Link to="/new">新建研究</Link></p>
      ) : (
        <>
          <div className="tabs">
            {STATUS_TABS.map(([key, label]) => (
              <button key={key} className={status === key ? "active" : ""} onClick={() => setStatus(key)}>
                {label} ({counts[key as keyof typeof counts]})
              </button>
            ))}
          </div>
          {shown.length === 0 ? <p className="muted">没有匹配的档案。</p> : (
            <div className="grid">
              {shown.map((d) => (
                <Link key={d.key} to={`/dossiers/${d.key}/${d.version}`} className="card dossier-card">
                  <div className="title-row">
                    <b>{d.name}</b><code>{d.ticker}</code><span className="tag">v{d.version}</span>
                    <StatusBadge status={d.status} />
                  </div>
                  <p className="clamp">{d.summary}</p>
                  <div className="muted small">数据截至 {d.as_of} · 更新于 {fmtTime(d.created_at)}</div>
                </Link>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
