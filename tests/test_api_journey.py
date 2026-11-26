"""离线全链路旅程（FastAPI 层）：研究 → 档案/导出 → 订阅/扫描 → 个人约束/情境 →
策略卡/模拟盘推进 → 评测（持续更新/路由校准）→ RSI 总览。

这是跨模块的冒烟护栏：任何一段接线断裂（路由前缀、payload 契约、后台作业状态机）都会在这里先失败，
而不是等到浏览器里才发现。全部虚构数据 + 假模型，不联网、不需要 Key。
"""
import io
import time
import zipfile

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from advisor.api.app import create_app
from tests.fakes import CODE
from tests.test_phase2 import make_service


def _wait(fn, ok, timeout=60.0, interval=0.05):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = fn()
        if ok(last):
            return last
        time.sleep(interval)
    raise AssertionError(f"等待超时：{last}")


def test_full_offline_user_journey(repo, policy):
    svc = make_service(repo, policy)
    with TestClient(create_app(deps=svc.deps, checkpointer=InMemorySaver())) as c:
        # 1) 健康与元信息
        health = c.get("/api/health").json()
        assert health["mongo"] is True and health["skills"]
        meta = c.get("/api/meta").json()
        assert set(meta["expert_cfg"]) <= {"max_team", "max_challenges", "search_per_expert", "fetch_per_expert"}

        # 2) 发起标准研究并等到完成（默认审批等级自动通过）
        r = c.post("/api/research", json={"question": f"复核 {CODE} 半年报业绩与估值",
                                          "ticker": CODE, "as_of": "2026-10-18", "route": "standard"})
        assert r.status_code == 200, r.text
        tid = r.json()["task_id"]
        task = _wait(lambda: c.get(f"/api/tasks/{tid}").json(), lambda t: t["status"] in ("done", "failed", "cancelled"))
        assert task["status"] == "done", task.get("error")
        assert task["result"]["dossier_key"] == CODE and task["result"]["version"] == 1
        assert c.get(f"/api/tasks/{tid}/audit").json()            # 审计可见
        export = c.get(f"/api/tasks/{tid}/export.json")
        assert export.status_code == 200 and export.json()["task"]["task_id"] == tid

        # 3) 档案列表/版本/详情/差异/导出
        rows = c.get("/api/dossiers").json()
        assert rows and rows[0]["key"] == CODE and rows[0]["version"] == 1
        assert c.get(f"/api/dossiers/{CODE}/versions").json()[0]["version"] == 1
        detail = c.get(f"/api/dossiers/{CODE}/1").json()
        assert detail["dossier"]["task_id"] == tid and detail["evidence"]
        md = c.get(f"/api/dossiers/{CODE}/1/markdown")
        assert md.status_code == 200 and "不构成投资建议" in md.text
        z = c.get("/api/dossiers/export.zip")
        assert z.status_code == 200
        assert any(CODE in n for n in zipfile.ZipFile(io.BytesIO(z.content)).namelist())

        # 4) 订阅与扫描（有档案才能订阅；扫描无新变化时返回空提醒）
        sub = c.post("/api/watch/subscriptions", json={"ticker": CODE, "price_move_pct": 10})
        assert sub.status_code == 200 and sub.json()["active"] is True
        overview = c.get("/api/watch").json()
        assert any(s["ticker"] == CODE and s["active"] for s in overview["subscriptions"])
        scan = c.post(f"/api/watch/scan/{CODE}")
        assert scan.status_code == 200 and "alert" in scan.json()
        assert c.delete(f"/api/watch/subscriptions/{CODE}").json()["ok"] is True

        # 5) 个人约束确认与情境官（只在本地比对）
        prof = c.post("/api/profile", json={
            "risk": {"max_drawdown_pct": 20, "horizon_months": 36, "liquidity_note": "两年后换房首付"},
            "holdings": [{"ticker": CODE, "name": "", "thesis": "高端白酒现金流稳定，渠道库存是主要风险", "horizon_months": 24}],
            "change_note": "离线旅程测试确认个人约束", "confirmed": True})
        assert prof.status_code == 200 and prof.json()["version"] == 1
        fit = c.get(f"/api/profile/fit/{CODE}")
        assert fit.status_code == 200 and "fit" in fit.json()

        # 6) 策略卡 → 启动模拟盘 → 推进一天（后台作业状态机）
        card = c.post("/api/strategy", json={
            "name": "离线旅程策略", "thesis": "只在前一版档案提交后检验趋势跟随是否成立",
            "universe": [], "entry": [{"field": "status", "op": "==", "value": "committed"}],
            "exit": [{"field": "open_alerts", "op": ">=", "value": 1}],
            "max_positions": 1, "stop_loss_pct": 10, "max_drawdown_limit": 20,
            "change_reason": "离线全链路旅程的策略卡用例"})
        assert card.status_code == 200, card.text
        cv = card.json()
        started = c.post(f"/api/strategy/{cv['card_id']}/{cv['version']}/paper")
        assert started.status_code == 200, started.text
        step = c.post("/api/strategy/step", json={"day": "2026-10-19"})
        assert step.status_code == 200, step.text
        job = _wait(lambda: c.get(f"/api/jobs/{step.json()['job_id']}").json(),
                    lambda j: j["status"] in ("done", "failed", "cancelled"))
        assert job["status"] == "done", job
        sdetail = c.get(f"/api/strategy/{cv['card_id']}/{cv['version']}").json()
        assert sdetail["navs"] and sdetail["navs"][0]["nav"] > 0

        # 7) 评测：持续更新（同步）与路由校准（后台）都落 done
        upd = c.post("/api/evals/update").json()["run_id"]
        assert c.get(f"/api/evals/{upd}").json()["status"] == "done"
        rt = c.post("/api/evals/routing").json()["run_id"]
        run = _wait(lambda: c.get(f"/api/evals/{rt}").json(),
                    lambda e: e["status"] in ("done", "failed", "cancelled"))
        assert run["status"] == "done" and run["accuracy"] is not None
        evals = c.get("/api/evals").json()
        assert {rt, upd} <= {e["run_id"] for e in evals}

        # 8) RSI 总览可用（账本已由启动引导建好）
        rsi = c.get("/api/rsi/overview").json()
        assert {"S1", "S2", "S3"} <= set(rsi["issues"])
