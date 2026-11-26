"""intake 识别出的股票代码回写任务请求：失败/取消后的「按原参数重新发起」不依赖问题文本里仍带代码。"""
from advisor.contracts import ResearchRequest
from tests.fakes import CODE, as_of
from tests.test_phase2 import make_service


def test_intake_writes_back_resolved_ticker(repo, policy):
    svc = make_service(repo, policy)
    tid = svc.start(ResearchRequest(question=f"复核 {CODE} 半年报业绩", as_of=as_of(), route="standard"),
                    background=False)                     # 未显式给 ticker：由 intake 从问题解析
    task = repo.get_task(tid)
    assert task["status"] == "done"
    assert task["request"]["ticker"] == CODE              # 解析结果回写，重试可直接复用
