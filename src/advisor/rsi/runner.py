"""⑤ 状态机：草稿 → 待审 → 已批准 → 已发布 → 归档 / 回滚。每次推进都写事件；发布必须由人批准。

单人开发阶段，触发人和批准人是同一个人：审批记录会明确标出「自批」，不伪装成已经做到了角色分离。
"""
from typing import Callable

from advisor.rsi.evaluator import evaluate_s1, evaluate_s2, evaluate_s3
from advisor.rsi.gates import run_gates
from advisor.rsi.issues import collect
from advisor.rsi.ledger import Ledger, LedgerError
from advisor.rsi.loader import RSILoader
from advisor.rsi.models import KIND_LABEL, LessonCard, Run
from advisor.rsi.proposer import PROPOSERS

MIN_ROUTING_RUNS = 3


class TransitionError(Exception):
    pass


class RSIRunner:
    def __init__(self, deps, ledger: Ledger, loader: RSILoader):
        self.deps, self.ledger, self.loader = deps, ledger, loader

    def _get(self, run_id: str, *allowed: str) -> Run:
        run = self.ledger.get_run(run_id)
        if not run:
            raise TransitionError(f"运行 {run_id} 不存在")
        if allowed and run.status not in allowed:
            raise TransitionError(f"{run_id} 当前为 {run.status}，不能执行该操作（需要 {'/'.join(allowed)}）")
        return run

    def _move(self, run: Run, status: str, action: str, actor: str, detail: str, **extra) -> Run:
        old = run.status
        run.status = status
        # 乐观锁：后台评测跑几分钟，期间人工 reject/approve 可能已改变状态；
        # 状态未变才允许整文档写回，避免用旧副本静默覆盖人工决定
        if not self.ledger.save_run(run, expect_status=old):
            raise TransitionError(f"{run.run_id} 的状态已从「{old}」被其他操作改变，请刷新后重试")
        self.ledger.event(action, actor, run_id=run.run_id, kind=run.kind, scenario=run.scenario,
                          from_status=old, to_status=status, detail=detail, **extra)
        return run

    # ① 收集
    def collect(self, actor: str = "owner") -> dict:
        self.ledger.bootstrap(self.deps.policy)
        return collect(self.ledger, self.deps.repo, actor, self.deps.policy)

    # ② 提议
    def propose(self, scenario: str, issue_id: str | None = None, actor: str = "owner") -> Run:
        self.ledger.bootstrap(self.deps.policy)
        run = PROPOSERS[scenario](self.deps, self.ledger, issue_id, actor)
        self.ledger.insert_run(run)
        self.ledger.event("propose", "optimizer", run_id=run.run_id, kind=run.kind, scenario=scenario,
                          to_status="draft", detail=run.change_reason, triggered_by=actor)
        return run

    # ③ 对比 + ④ 门控
    def evaluate(self, run_id: str, actor: str = "owner", log: Callable[[str], None] = lambda m: None) -> Run:
        run = self._get(run_id, "draft")
        base = self.ledger.get_asset(run.kind, run.base_version)
        if run.scenario == "S1":
            cards = lambda c: [LessonCard.model_validate(x) for x in c.get("cards", [])]   # noqa: E731
            ev = evaluate_s1(self.deps, self.ledger, run, cards(base.content), cards(run.content), log)
        elif run.scenario == "S2":
            self._top_up_routing(log)
            ev = evaluate_s2(self.deps, self.ledger, run, base.content, run.content, log)
        else:
            ev = evaluate_s3(self.deps, self.ledger, run, base.content, run.content, log)
        run.eval = ev
        run.gates = run_gates(run, ev, self.ledger)
        current = self.ledger.get_run(run.run_id)
        if current is None or current.status != "draft":
            # 评测耗时期间运行已被人工拒绝/批准：人工操作优先，本次结果不写回
            self.ledger.event("evaluate.discard", "evaluator", run_id=run.run_id, kind=run.kind, scenario=run.scenario,
                              detail=f"评测完成时运行状态为「{current.status if current else '缺失'}」，结果未写入")
            return current or run
        failed = [g for g in run.gates if not g.passed]
        self.ledger.event("evaluate", "evaluator", run_id=run.run_id, kind=run.kind, scenario=run.scenario,
                          detail="六道门控：" + " ".join(f"{g.label}{'✓' if g.passed else '✗'}" for g in run.gates),
                          gates={g.name: g.passed for g in run.gates})
        if failed:
            run.reject_reason = "；".join(f"{g.label}：{g.detail}" for g in failed)
            return self._move(run, "rejected", "reject", "gates", f"门控未通过：{run.reject_reason}")
        return self._move(run, "review", "submit", actor, "六道门控全部通过，提交人工审批")

    def _top_up_routing(self, log: Callable[[str], None]) -> None:
        """噪声门控至少需要 3 次独立的判别器运行。"""
        from advisor.evals.routing import run_routing_eval
        have = self.deps.repo.db.eval_runs.count_documents(
            {"type": "routing", "status": "done", "rows.0": {"$exists": True}})   # 失败/无数据的评测不算一次重复
        for i in range(max(MIN_ROUTING_RUNS - have, 0)):
            log(f"补跑路由评测 {i + 1}/{MIN_ROUTING_RUNS - have}（噪声门控需要 ≥{MIN_ROUTING_RUNS} 次独立运行）")
            run_routing_eval(self.deps)

    # ⑤ 人工批准 / 发布 / 回滚
    def approve(self, run_id: str, actor: str, note: str = "") -> Run:
        run = self._get(run_id, "review")
        if run.risk == "high" and len(note.strip()) < 10:
            raise TransitionError("高风险改动（改抓取站点名单）必须写明批准理由，不少于 10 个字")
        self_approved = actor == run.triggered_by
        run.approvals.append({"by": actor, "note": note, "self_approved": self_approved})
        return self._move(run, "approved", "approve", actor,
                          f"人工批准{'（自批：触发人与批准人相同，单人阶段未做到角色分离）' if self_approved else ''}"
                          + (f"：{note}" if note else ""), self_approved=self_approved)

    def reject(self, run_id: str, actor: str, reason: str) -> Run:
        run = self._get(run_id, "draft", "review", "approved")
        run.reject_reason = reason
        return self._move(run, "rejected", "reject", actor, f"人工拒绝：{reason}")

    def release(self, run_id: str, actor: str) -> Run:
        run = self._get(run_id, "approved")
        # 先用状态迁移占位：双击发布、发布与并发拒绝在这一点上串行化；失败方不留任何副作用
        run = self._move(run, "released", "release.begin", actor,
                         f"发布{KIND_LABEL[run.kind]}：校验在用版本后激活新版本")
        asset = None
        activated = False
        try:
            active = self.ledger.active_version(run.kind)
            if active != run.base_version:
                raise TransitionError(f"{KIND_LABEL[run.kind]}在用版本已从 v{run.base_version} 变为 v{active}，候选需重新评测")
            try:
                resolved = self._resolved_issues(run)
                asset = self.ledger.append_asset(run.kind, run.content, run.change_reason, actor, run.base_version, run.run_id)
            except LedgerError as e:
                raise TransitionError(str(e)) from e
            # 激活 = 对在用版本指针的 CAS：并发发布同一 base 只有一个成功，回滚并发也会被拦下
            if not self.ledger.activate(run.kind, run.base_version, asset.version):
                raise TransitionError(f"{KIND_LABEL[run.kind]}在用版本已被并发修改，候选需重新评测")
            activated = True
            for prev in self.ledger.runs_of(run.kind, "released"):
                if prev.run_id != run.run_id:
                    self._move(prev, "archived", "archive", "system", f"被 {run.run_id} 发布的 v{asset.version} 取代")
            self.ledger.set_issue_status(resolved, "addressed")   # 只标记回放确认已判对的坏案例；其余留在 open
            self.loader.invalidate()
            run.released_version = asset.version
            self.ledger.save_run(run, expect_status="released")
            self.ledger.event("release", actor, run_id=run.run_id, kind=run.kind, scenario=run.scenario,
                              from_status="released", to_status="released",
                              detail=(f"发布{KIND_LABEL[run.kind]} v{asset.version}（parent v{run.base_version}，"
                                      f"sha256 {asset.sha256[:10]}）；坏案例回放确认 {len(resolved)}/{len(run.issue_ids)} 条已修复"),
                              active_version=asset.version)
            return run
        except Exception:
            # 失败补偿：从未激活的资产删除、占位退回 approved，保证失败方不留副作用
            try:
                if asset is not None and not activated:
                    self.ledger.delete_asset(run.kind, asset.version)
                if not activated:
                    run.status = "approved"
                    run.released_version = None
                    self.ledger.save_run(run, expect_status="released")
            except Exception:      # noqa: BLE001  补偿尽力而为，不掩盖原始异常
                pass
            raise

    def rollback(self, kind: str, actor: str, reason: str) -> dict:
        current = self.ledger.active_asset(kind)
        if not current:
            raise TransitionError(f"{KIND_LABEL.get(kind, kind)} 没有在用版本")
        if current.parent_version is None:
            # 幂等自愈：上次回滚可能在标记运行记录前中断——把不在用的已发布运行对齐为已回滚
            self._reconcile_rollback(kind, current.version, actor)
            raise TransitionError(f"{KIND_LABEL[kind]}已是基线版本，无法回滚")
        if len(reason.strip()) < 10:
            raise TransitionError("回滚理由不少于 10 个字")
        # 先用 CAS 把在用版本退回 parent：与并发 release 互斥，失败方不改任何其它文档
        if not self.ledger.activate(kind, current.version, current.parent_version):
            raise TransitionError(f"{KIND_LABEL[kind]}在用版本已被并发修改，请刷新后重试")
        self.ledger.event("rollback", actor, kind=kind, active_version=current.parent_version,
                          detail=f"{KIND_LABEL[kind]} v{current.version} → v{current.parent_version}：{reason}")
        self._reconcile_rollback(kind, current.parent_version, actor, reason)
        self.loader.invalidate()
        return {"kind": kind, "from": current.version, "to": current.parent_version}

    def _resolved_issues(self, run: Run) -> list[str]:
        """发布前按评测回放逐条确认坏案例是否真的被判对：路由类候选只把已修复的标记 addressed。"""
        if run.kind != "routing":
            return list(run.issue_ids)
        from advisor.routing import route_from_probs
        cfg = {**self.deps.policy.routing, **run.content}
        resolved = []
        eval_runs = list(self.deps.repo.db.eval_runs.find({"type": "routing"}, projection={"_id": 0})
                         .sort("created_at", -1).limit(5))
        for issue in self.ledger.get_issues(run.issue_ids):
            q = (issue.source or {}).get("question")
            rows = [r for rr in eval_runs for r in rr.get("rows", []) if r.get("question") == q]
            if not q or not rows:
                continue
            fixed = True
            for r in rows:
                if r.get("source") in ("model", "abstain"):
                    route = route_from_probs(r["probs"], cfg, r.get("stakes"), r.get("uncertainty"),
                                             r.get("intent", "research")).route
                else:
                    route = r["route"]
                fixed = fixed and route == r["label"]
            if fixed:
                resolved.append(issue.id)
        return resolved

    def _reconcile_rollback(self, kind: str, active_version: int, actor: str, reason: str = "") -> None:
        """账本对齐：指针已不在用的已发布运行标记回滚、坏案例重开；可安全重试（幂等）。"""
        for run in self.ledger.runs_of(kind, "released"):
            if run.released_version == active_version:
                continue
            run.reject_reason = f"发布后回滚：{reason}" if reason else (run.reject_reason or "发布后回滚（账本对齐）")
            try:
                self._move(run, "rolled_back", "rolled_back", actor, run.reject_reason)
                self.ledger.set_issue_status(run.issue_ids, "open")
            except TransitionError:      # 并发下运行状态已变化：下次重试再对齐
                continue
