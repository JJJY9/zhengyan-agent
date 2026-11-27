import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, waitJob } from "../api";
import { GateDots } from "../components/Rsi";
import { PageHead, useAskContext } from "../components/Shell";
import { RuleChips, RuleEditor } from "../components/Strategy";
import type { Rule, StrategyOverview } from "../types";

const DEFAULT_ENTRY: Rule[] = [{ field: "status", op: "==", value: "committed" }, { field: "pe_pct", op: "<=", value: 40 }];
const DEFAULT_EXIT: Rule[] = [{ field: "open_alerts", op: ">=", value: 1 }];

/** 策略卡列表 + 新建。策略卡只在前向模拟盘上运行，系统不执行任何真实交易。 */
export default function Strategy() {
  const nav = useNavigate();
  const [ov, setOv] = useState<StrategyOverview | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [uniErr, setUniErr] = useState("");
  const [log, setLog] = useState("");
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const [form, setForm] = useState<null | { name: string; thesis: string; universe: string; entry: Rule[]; exit: Rule[];
    max_positions: string; stop_loss_pct: string; max_drawdown_limit: string; change_reason: string }>(null);
  useAskContext({ title: "策略与模拟盘", suggestions: ["PE 分位低一定意味着便宜吗？还要核对哪些口径"] }, []);

  const load = useCallback(() => { api.strategy.overview().then(setOv).catch((e) => setError(e.message)); }, []);
  useEffect(load, [load]);

  const step = async () => {
    setBusy(true); setError("");
    try {
      const { job_id } = await api.strategy.step();
      const j = await waitJob(job_id, (x) => setLog(x.log[x.log.length - 1] ?? "推进中…"), () => !alive.current);
      setLog(j.status === "failed" ? `推进失败：${j.error ?? "未知原因"}` : (j.log.join("；") || "没有模拟中的策略"));
    } catch (e) { setError((e as Error).message); }
    setBusy(false); load();
  };
  const create = async () => {
    if (!form || busy) return;                       // 双击创建两张卡：提交期间禁止重入
    setBusy(true);
    setError("");
    setUniErr("");                                   // 上一份表单的候选股校验不能带进本次提交
    try {
      const { universe, ...rest } = form;
      // 空串不能被 Number("") 当成 0 通过校验：编辑期保留字符串，提交前统一转换
      const num = (s: string): number | null => {
        const t = String(s).trim();
        return t === "" || !Number.isFinite(Number(t)) ? null : Number(t);
      };
      const maxPos = num(rest.max_positions), stop = num(rest.stop_loss_pct), mdd = num(rest.max_drawdown_limit);
      if (maxPos === null || !Number.isInteger(maxPos) || maxPos < 1 || maxPos > 20) {
        setError("最多持有必须是 1–20 的整数"); setBusy(false); return;
      }
      if (stop === null || stop <= 0 || stop > 80) {   // 后端约束 gt=0 le=80：前端同口径
        setError("止损 % 必须是 (0,80] 的数字"); setBusy(false); return;
      }
      if (mdd === null || mdd <= 0 || mdd > 100) {
        setError("晋级回撤上限 % 必须是 (0,100] 的数字"); setBusy(false); return;
      }
      const uni = universe.split(/[,，、\s]+/).filter(Boolean);
      const badUni = uni.filter((k) => !/^\d{6}$/.test(k));
      if (badUni.length) {
        setUniErr("候选股票需为 6 位代码，用逗号或顿号分隔"); setBusy(false); return;
      }
      const cast = (rules: Rule[]): Rule[] => rules.map((r) => (r.field === "status" ? r : { ...r, value: Number(r.value) }));
      for (const r of [...rest.entry, ...rest.exit]) {
        if (r.field !== "status" && (String(r.value).trim() === "" || !Number.isFinite(Number(r.value)))) {
          setError(`阈值必须是数字（${r.field} 当前为「${String(r.value)}」）`);
          setBusy(false);
          return;
        }
      }
      const v = await api.strategy.create({ ...rest, max_positions: maxPos, stop_loss_pct: stop, max_drawdown_limit: mdd,
                                           entry: cast(rest.entry), exit: cast(rest.exit),
                                           universe: uni });
      nav(`/strategy/${v.card_id}/${v.version}`);
    } catch (e) { setError((e as Error).message); setBusy(false); }
  };

  return (
    <>
      <PageHead title="策略与模拟盘" sub="规则显式、只向前模拟、晋级须过门控并由你批准；不执行任何真实交易"
        right={<>
          <button disabled={busy} onClick={step}>{busy ? "推进中…" : "推进模拟盘到今天"}</button>
          <button className="primary" onClick={() => { setUniErr(""); setForm({ name: "", thesis: "", universe: "", entry: DEFAULT_ENTRY, exit: DEFAULT_EXIT,
            max_positions: "5", stop_loss_pct: "15", max_drawdown_limit: "20", change_reason: "" }); }}>新建策略卡</button>
        </>} />
      <div className="page">
        {error && <div className="error">{error}</div>}
        {log && <p className="muted small">{log}</p>}

        {form && ov && (
          <section className="card form">
            <h3 style={{ margin: 0 }}>新建策略卡</h3>
            <div className="two-col">
              <label>名称<input value={form.name} maxLength={40} onChange={(e) => setForm({ ...form, name: e.target.value })} /></label>
              <label>候选股票（空 = 全部有档案的股票）
                <input value={form.universe} placeholder="600519, 300750"
                       onChange={(e) => { setUniErr(""); setForm({ ...form, universe: e.target.value }); }} />
                {uniErr && <span className="warn-text small">{uniErr}</span>}
              </label>
            </div>
            <label>策略逻辑：为什么这些规则有意义<textarea rows={2} value={form.thesis} onChange={(e) => setForm({ ...form, thesis: e.target.value })} /></label>
            <div><div className="muted small" style={{ marginBottom: 6 }}>纳入条件（全部满足）</div>
              <RuleEditor rules={form.entry} fields={ov.fields} onChange={(entry) => setForm({ ...form, entry })} /></div>
            <div><div className="muted small" style={{ marginBottom: 6 }}>移出条件（任一满足）</div>
              <RuleEditor rules={form.exit} fields={ov.fields} onChange={(exit) => setForm({ ...form, exit })} /></div>
            <div className="stat-row" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
              <label>最多持有<input type="number" min={1} max={20} value={form.max_positions} onChange={(e) => setForm({ ...form, max_positions: e.target.value })} /></label>
              <label>止损 %<input type="number" min={1} max={80} value={form.stop_loss_pct} onChange={(e) => setForm({ ...form, stop_loss_pct: e.target.value })} /></label>
              <label>晋级回撤上限 %<input type="number" value={form.max_drawdown_limit} onChange={(e) => setForm({ ...form, max_drawdown_limit: e.target.value })} /></label>
            </div>
            <label>改动理由（不少于 10 个字）<input value={form.change_reason} onChange={(e) => setForm({ ...form, change_reason: e.target.value })} /></label>
            <div className="row-btns" style={{ margin: 0 }}>
              <button className="primary" disabled={busy || form.change_reason.trim().length < 10 || form.name.trim().length < 2} onClick={create}>保存为草稿</button>
              <button className="ghost" onClick={() => { setUniErr(""); setForm(null); }}>取消</button>
            </div>
          </section>
        )}

        {!ov ? <p className="muted">加载中…</p> : ov.cards.length === 0 ? (
          <div className="empty">还没有策略卡。策略卡把「什么情况下纳入 / 移出观察组合」写成规则，在前向模拟盘上检验。</div>
        ) : (
          <div className="card table">
            <table className="table">
              <thead><tr><th>策略卡</th><th>状态</th><th>纳入条件</th><th className="num">模拟</th><th className="num">收益</th><th className="num">最大回撤</th><th>门控</th></tr></thead>
              <tbody>
                {ov.cards.map((c) => (
                  <tr key={`${c.card_id}-${c.version}`} style={{ cursor: "pointer" }} onClick={() => nav(`/strategy/${c.card_id}/${c.version}`)}>
                    <td><Link to={`/strategy/${c.card_id}/${c.version}`}>{c.card_id} v{c.version}</Link> {c.name}
                      <div className="muted small">{c.change_reason}</div></td>
                    <td><span className={`badge st-${c.status}`}>{ov.status_label[c.status]}</span></td>
                    <td><RuleChips rules={c.entry} fields={ov.fields} /></td>
                    <td className="num">{c.stats.days} 天</td>
                    <td className={`num ${c.stats.return_pct > 0 ? "up" : c.stats.return_pct < 0 ? "down" : ""}`}>{c.stats.return_pct.toFixed(2)}%</td>
                    <td className="num">{c.stats.max_drawdown_pct.toFixed(2)}%</td>
                    <td><GateDots gates={c.gates} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {ov && (
          <p className="muted small">晋级门控：前向模拟 ≥ {ov.gates_cfg.min_paper_days} 个交易日、无未来数据、回撤不超上限、每笔可追溯、与在用版本同期收益落后不超过 {ov.gates_cfg.regression_tolerance_pct}%。
            模拟资金 {ov.initial_cash.toLocaleString()}（虚拟），等权、整手。</p>
        )}
      </div>
    </>
  );
}
