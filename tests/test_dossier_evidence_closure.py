"""档案接口证据闭包：反方观点/专家回应/修订主张/上一版复核引用的证据都必须随档案返回。"""
from datetime import date, datetime, timezone

from advisor.api.dossiers import all_evidence_ids
from advisor.contracts import Dissent, Dossier, PriorReview, QualityResult


def test_evidence_closure_includes_dissent_responses_and_prior_review():
    d = Dossier(
        key="600999", version=2, parent_version=1, ticker="600999", name="示例酒业", as_of=date(2026, 10, 18),
        question="复核 600999 半年报", route="deep", status="partial", summary="",
        dissents=[Dissent(text="反对意见", evidence_ids=["ev_dissent"])],
        prior_review=[PriorReview(prior_claim="旧结论", verdict="retained", reason="仍成立", evidence_ids=["ev_prior"])],
        quality=QualityResult(decision="partial", verified=1, disputed=0, rejected=0),
        task_id="t1", created_at=datetime.now(timezone.utc),
        deliberation={"responses": [{"evidence_ids": ["ev_resp"], "revised_claim": {"evidence_ids": ["ev_rev"]}}]},
    )
    ids = set(all_evidence_ids(d))
    assert {"ev_dissent", "ev_prior", "ev_resp", "ev_rev"} <= ids   # 缺任何一类，前端证据芯片点开就是空的
