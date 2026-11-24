"""评测：路由校准、单 Agent 与多专家对照。"""
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from advisor.api.common import Svc
from advisor.graph.runner import PoolSaturated

router = APIRouter(prefix="/api/evals")

ROUTE_CHOICES = {"quick", "standard", "deep"}
RouteName = Literal["quick", "standard", "deep"]


def _require_llm(svc: Svc) -> None:
    if getattr(svc.deps.llm, "configured", True) is False:
        raise HTTPException(400, "未配置 DEEPSEEK_API_KEY：路由校准与对照评测都需要模型参与，请先在 .env 中填写后重启服务")


class CompareIn(BaseModel):
    case_ids: list[str] | None = Field(default=None, max_length=200)
    routes: list[RouteName] = Field(default=["standard", "deep"], max_length=3)


@router.get("")
def evals(svc: Svc):
    return [{k: v for k, v in r.items() if k not in ("rows", "update_results")} for r in svc.deps.repo.list_evals()]


@router.get("/{run_id}")
def eval_run(run_id: str, svc: Svc):
    r = svc.deps.repo.get_eval(run_id)
    if not r:
        raise HTTPException(404, "评测不存在")
    return r


@router.post("/routing")
def eval_routing(svc: Svc):
    _require_llm(svc)
    from functools import partial

    from advisor.evals.routing import run_routing_eval
    run_id = "rt_" + uuid.uuid4().hex[:8]
    svc.deps.repo.save_eval({"run_id": run_id, "type": "routing", "status": "running",
                             "created_at": datetime.now(timezone.utc)})
    try:
        svc.submit(partial(run_routing_eval, svc.deps, run_id=run_id))
    except PoolSaturated as e:          # 队列满：评测文档落成 failed，前端看得到原因
        svc.deps.repo.save_eval({"run_id": run_id, "type": "routing", "status": "failed",
                                 "error": str(e), "created_at": datetime.now(timezone.utc)})
        raise HTTPException(429, str(e)) from e
    return {"run_id": run_id}


@router.post("/update")
def eval_update(svc: Svc):
    """持续更新评测：纯代码打分，几秒内完成。"""
    from advisor.evals.update import run_update_eval
    return {"run_id": run_update_eval(svc.deps)}


@router.post("/compare")
def eval_compare(body: CompareIn, svc: Svc):
    from advisor.evals.compare import load_cases, run_compare
    routes = list(dict.fromkeys(body.routes))          # 去重：重复 route 不再放大后台研究次数
    if not routes:
        raise HTTPException(400, f"routes 只能是 {'/'.join(sorted(ROUTE_CHOICES))} 的非空子集")
    cases = [c for c in load_cases() if not body.case_ids or c["id"] in body.case_ids]
    if not cases:
        raise HTTPException(400, "没有匹配的评测用例")
    _require_llm(svc)
    run_id = "ev_" + uuid.uuid4().hex[:8]
    # 先登记评测文档再丢给后台线程：前端立即 GET /api/evals/{run_id} 不会 404
    svc.deps.repo.save_eval({"run_id": run_id, "type": "compare", "status": "running", "routes": routes,
                             "cases": cases, "results": [], "created_at": datetime.now(timezone.utc)})
    try:
        svc.submit(run_compare, svc.research, cases, routes, run_id)
    except PoolSaturated as e:
        svc.deps.repo.save_eval({"run_id": run_id, "type": "compare", "status": "failed", "routes": routes,
                                 "cases": cases, "results": [], "error": str(e),
                                 "created_at": datetime.now(timezone.utc)})
        raise HTTPException(429, str(e)) from e
    return {"run_id": run_id}
