"""二期出口标准：对照"单 Agent + SOP"基线，多专家质量不退化且成本可接受，否则保留单 Agent。

统一材料（同一问题、同一 as_of、同一数据）、统一核验规则；档案写入隔离命名空间，不污染用户档案。
"""
import json
import uuid
from datetime import date, datetime, timezone
from statistics import mean

from advisor.contracts import ResearchRequest
from advisor.evals.metrics import dossier_metrics
from advisor.settings import ROOT

DEFAULT_CASES = ROOT / "evals" / "cases" / "compare_cases.json"
NON_INFERIOR_MARGIN = 0.02
MAX_COST_RATIO = 5.0


def load_cases(path=DEFAULT_CASES) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def summarize(results: list[dict], baseline: str = "standard", candidate: str = "deep") -> dict:
    by_route: dict[str, list[dict]] = {}
    for r in results:
        by_route.setdefault(r["route"], []).append(r)
    keys = ("score", "verified_rate", "coverage_rate", "numeric_accuracy", "dissents", "tokens", "search_calls", "seconds")
    agg = {}
    for route, rs in by_route.items():
        ok = [r["metrics"] for r in rs if r.get("metrics")]
        entry = {}
        for k in keys:
            vals = [m[k] for m in ok if m.get(k) is not None]
            entry[k] = round(mean(vals), 3) if vals else None
        entry["cases"] = len(rs)
        entry["failed"] = len(rs) - len(ok)
        entry["success_rate"] = round(len(ok) / len(rs), 3) if rs else 0.0
        # 失败用例按 0 分计入有效分：只在成功子集上取均值会奖励「躲开难题」
        entry["score_all"] = round(sum(m["score"] for m in ok) / len(rs), 3) if rs else 0.0
        agg[route] = entry
    verdict = {"decision": "insufficient", "text": "样本不足，无法判断"}
    if baseline in agg and candidate in agg:
        failed = agg[baseline]["failed"] + agg[candidate]["failed"]
        b_score, c_score = agg[baseline]["score"], agg[candidate]["score"]
        if failed or agg[baseline]["cases"] != agg[candidate]["cases"] or b_score is None or c_score is None:
            # 某条 route 全部失败时 mean 为 None：先判不完整再算差值，不能拿 None 做减法
            verdict = {"decision": "insufficient", "failed_cases": failed,
                       "text": f"{failed} 个用例没有产出档案（失败或未完成），评测不完整，先排查后再比较"}
        else:
            delta = c_score - b_score
            ratio = agg[candidate]["tokens"] / max(agg[baseline]["tokens"], 1) if agg[candidate]["tokens"] is not None else None
            verdict = {
                "decision": "insufficient",
                "quality_delta": round(delta, 3), "cost_ratio": round(ratio, 2) if ratio is not None else None,
                "failed_cases": failed,
                "criteria": {"non_inferior_margin": NON_INFERIOR_MARGIN, "max_cost_ratio": MAX_COST_RATIO},
            }
            non_inferior, affordable = delta >= -NON_INFERIOR_MARGIN, ratio is not None and ratio <= MAX_COST_RATIO
            ok = non_inferior and affordable
            verdict.update({
                "decision": "adopt_multi_expert" if ok else "keep_single_agent",
                "non_inferior": non_inferior, "affordable": affordable,
                "text": ("多专家通过准入：质量不退化且成本在上限内（仅用于争议深研）" if ok else
                         "保留单 Agent 基线：" + ("质量退化" if not non_inferior else "") +
                         ("、" if not non_inferior and not affordable else "") + ("成本超过上限" if not affordable else "")),
            })
    return {"routes": agg, "verdict": verdict}


def run_compare(svc, cases: list[dict], routes: list[str], run_id: str | None = None) -> str:
    repo = svc.deps.repo
    run_id = run_id or "ev_" + uuid.uuid4().hex[:8]
    run = repo.get_eval(run_id) or {"run_id": run_id, "type": "compare", "routes": routes, "cases": cases,
                                    "results": [], "created_at": datetime.now(timezone.utc)}
    run.update(status="running", routes=routes, cases=cases)
    run.setdefault("results", [])
    repo.save_eval(run)
    try:
        for case in cases:
            for route in routes:
                entry = {"case": case["id"], "route": route}
                try:
                    req = ResearchRequest(question=case["question"], ticker=case.get("ticker"), route=route,
                                          as_of=date.fromisoformat(case["as_of"]) if case.get("as_of") else None)
                    tid = svc.start(req, background=False, namespace=f"eval-{run_id}-{route}")
                    entry["task_id"] = tid
                    task = repo.get_task(tid)
                    if task["status"] == "waiting_approval":
                        # 评测任务在隔离命名空间里，用户看不到也无法审批：评测内自动批准继续
                        entry["auto_approved"] = True
                        (getattr(svc, "research", None) or svc).resume(tid, True, background=False)
                        task = repo.get_task(tid)
                    if task["status"] == "done":
                        d = repo.get_dossier(task["result"]["dossier_key"], task["result"]["version"])
                        ev = {c.id: c for c in repo.get_evidence(d.evidence_ids + [i for c in d.claims for i in c.evidence_ids])}
                        entry.update(dossier_key=d.key, version=d.version, metrics=dossier_metrics(d, ev, task))
                    else:
                        entry["error"] = f"{task['status']}: {task.get('error')}"
                except Exception as e:  # noqa: BLE001  单例失败记录在案，不中断评测
                    entry["error"] = f"{type(e).__name__}: {e}"
                run["results"].append(entry)
                run["summary"] = summarize(run["results"])
                repo.save_eval(run)
        run["status"] = "done"
    except Exception as e:  # noqa: BLE001  任何意外都落成 failed：评测页不会永远停在「进行中」
        run["status"] = "failed"
        run["error"] = f"{type(e).__name__}: {e}"
    repo.save_eval(run)
    return run_id
