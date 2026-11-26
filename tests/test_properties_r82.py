"""R82a 性质测试：评测回收的状态语义表（随机化文档形状）。

对每个 running 文档按 (instance 归属, started_at 形态, 进展新鲜度) 组合断言：
- 活实例：排队（started_at=None）永不被年龄回收；已开跑/历史缺字段则只在停滞时回收（归因「停滞」）；
- 死实例：无论新旧一律回收（归因「进程已不在」）；无实例归属的历史文档仅在停滞时回收；
- 停机 fail_instance_evals：本实例所有 running（含排队/新鲜）标失败（归因停机），他实例不动；
- 失败文档统一形状：summary 保留原字段、routes 存在、verdict 为「评测中断」且例数=结果数；
  results 原样；非 running 文档完全不动；回收幂等（第二次为 0）。"""
import random
import uuid
from datetime import datetime, timedelta, timezone

from tests.test_phase2 import make_service


def _doc(repo, run_id, *, instance, started, stale, results, summary):
    now = datetime.now(timezone.utc)
    doc = {"run_id": run_id, "type": "routing", "status": "running",
           "updated_at": now - timedelta(hours=3 if stale else 0),
           "results": [{"case": f"c{i}"} for i in range(results)]}
    if instance is not None:
        doc["instance"] = instance
    if started == "timestamp":
        doc["started_at"] = now - timedelta(minutes=5)
    elif started == "none":
        doc["started_at"] = None
    if summary is not None:
        doc["summary"] = summary
    repo.db.eval_runs.insert_one(doc)


def _expected(instance_kind, started, stale):
    """返回 recovery 后的期望状态：running / failed:<reasonsub>。"""
    if instance_kind == "self":
        if started == "none":
            return "running"
        return "failed:停滞" if stale else "running"
    if instance_kind == "ghost":
        return "failed:进程已不在"
    # 无实例归属：历史文档，只在停滞时回收
    if started == "none":
        return "running"
    return "failed:进程已不在" if stale else "running"


def _check_failed(repo, doc):
    got = repo.get_eval(doc["run_id"])
    assert got["status"] == "failed"
    s = got["summary"]
    assert "routes" in s and isinstance(s["routes"], dict)          # 前端按 s.routes 渲染，缺了会崩
    v = s["verdict"]
    assert v["provisional"] is True and "中断" in v["text"]
    assert f"已完成 {len(doc.get('results') or [])} 例" in v["text"]
    assert got["results"] == doc.get("results")                     # 结果列表原样保留
    if doc.get("summary") and "routes" in doc["summary"]:
        assert s["routes"] == doc["summary"]["routes"]              # 已有 routes 绝不被覆盖


def test_eval_recovery_semantic_table(repo, policy):
    rng = random.Random(20261202)
    svc = make_service(repo, policy)
    repo.touch_lease(svc.instance)
    ghost = f"inst_ghost_{uuid.uuid4().hex[:8]}"
    kinds = ["self", "ghost", "absent"]
    expect_changed = 0
    docs = []
    for i in range(40):
        kind = rng.choice(kinds)
        started = rng.choice(["timestamp", "none", "missing"])
        stale = rng.random() < 0.5
        results = rng.randint(0, 3)
        summary = rng.choice([None, {"routes": {"standard": {"cases": results}}}, {"verdict": {"x": 1}}])
        instance = {"self": svc.instance, "ghost": ghost, "absent": None}[kind]
        doc = {"run_id": f"ev_r82_{i}", "results": [{"case": f"c{k}"} for k in range(results)]}
        _doc(repo, doc["run_id"], instance=instance, started=started, stale=stale,
             results=results, summary=summary)
        doc.update(summary=summary)
        docs.append((doc, _expected(kind, started, stale)))
        if docs[-1][1] != "running":
            expect_changed += 1
    # 非 running 文档不受影响
    repo.db.eval_runs.insert_one({"run_id": "ev_r82_done", "type": "routing", "status": "done",
                                  "updated_at": datetime.now(timezone.utc) - timedelta(hours=9)})
    changed = repo.recover_stale_evals()
    assert changed == expect_changed
    for doc, want in docs:
        got = repo.get_eval(doc["run_id"])
        if want == "running":
            assert got["status"] == "running", (doc["run_id"], got)
        else:
            assert got["status"] == "failed" and want.split(":")[1] in got["error"], (doc["run_id"], got)
            _check_failed(repo, doc)
    assert repo.get_eval("ev_r82_done")["status"] == "done"
    assert repo.recover_stale_evals() == 0                          # 幂等：失败文档不再变化


def test_fail_instance_then_recover_composition(repo, policy):
    """停机收尾与僵尸回收组合：本实例所有 running 统一停机归因；他实例按回收语义处理。"""
    rng = random.Random(20261203)
    svc = make_service(repo, policy)
    repo.touch_lease(svc.instance)
    ghost = f"inst_g2_{uuid.uuid4().hex[:8]}"
    mine, others, failed_ids = 0, 0, []
    for i in range(30):
        kind = rng.choice(["self", "ghost", "absent"])
        started = rng.choice(["timestamp", "none", "missing"])
        stale = rng.random() < 0.5
        instance = {"self": svc.instance, "ghost": ghost, "absent": None}[kind]
        _doc(repo, f"ev_c82_{i}", instance=instance, started=started, stale=stale, results=1, summary=None)
        if kind == "self":
            mine += 1
            failed_ids.append(f"ev_c82_{i}")
        elif _expected(kind, started, stale) != "running":
            others += 1
            failed_ids.append(f"ev_c82_{i}")
    assert repo.fail_instance_evals(svc.instance, "服务停止，评测中断；可重新发起") == mine
    assert repo.recover_stale_evals() == others
    for i in range(30):
        got = repo.get_eval(f"ev_c82_{i}")
        if got.get("instance") == svc.instance:
            assert got["status"] == "failed" and "服务停止" in got["error"]
    for rid in failed_ids:
        _check_failed(repo, {"run_id": rid, "results": [{"case": "c0"}]})
