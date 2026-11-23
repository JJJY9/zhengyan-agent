from datetime import date

import httpx
import pytest
import respx

from advisor.harness import SkillFailed
from advisor.skills.data.web import DeepSeekTools, build_query, build_web_skills
from tests.fakes import FakeWebTools

AS_OF = date(2026, 10, 18)


def test_build_query_enforces_time_and_length(policy):
    q = build_query("示例酒业 业绩", AS_OF, 120, policy.block_sites)
    assert "before:2026-10-19" in q and "after:2026-06-20" in q and len(q) <= 128


def test_search_filters(policy, repo):
    web = FakeWebTools()
    search = {s.spec.name: s for s in build_web_skills(web, repo, policy)}["web.search"]
    out = search.run(search.spec.input_model(query="示例酒业 业绩", as_of=AS_OF))
    urls = {h.url for h in out.hits}
    assert "https://guba.eastmoney.com/x" not in urls                # 黑名单
    assert "https://www.cls.cn/f" not in urls                       # 晚于 as_of
    assert "https://www.yicai.com/n" not in urls                    # 低相关
    assert out.dropped == {"blocked_site": 1, "after_as_of": 1, "low_relevance": 1, "invalid_row": 0}
    search.run(search.spec.input_model(query="示例酒业 业绩", as_of=AS_OF))
    assert len(web.searches) == 1                                   # 第二次命中缓存


def test_fetch_whitelist(policy, repo):
    fetch = {s.spec.name: s for s in build_web_skills(FakeWebTools(), repo, policy)}["web.fetch"]
    with pytest.raises(SkillFailed, match="白名单"):
        fetch.run(fetch.spec.input_model(url="https://www.example-blog.com/p"))
    assert fetch.run(fetch.spec.input_model(url="https://www.stcn.com/a/1.html")).content


@respx.mock
def test_http_error_mapping():
    tools = DeepSeekTools("k", qps=100)
    respx.post("https://api.deepseek.com/v1/tools/search").mock(side_effect=[
        httpx.Response(429, text="slow down"), httpx.Response(402, text="no money"),
        httpx.Response(200, json={"results": [{"title": "t"}]}),
    ])
    with pytest.raises(SkillFailed) as e1:
        tools.search("q", 5)
    assert e1.value.retryable
    with pytest.raises(SkillFailed) as e2:
        tools.search("q", 5)
    assert not e2.value.retryable and "余额" in str(e2.value)
    assert tools.search("q", 5) == [{"title": "t"}]


def test_fetch_url_with_long_digits_not_treated_as_sensitive(policy, repo):
    from advisor.harness import Budget, Harness, RunContext
    from advisor.skills.registry import SkillRegistry

    h = Harness(policy, SkillRegistry(build_web_skills(FakeWebTools(), repo, policy)), lambda r: None,
                Budget(limits={"fetch_calls": 1, "wall_clock_s": 60}), RunContext("t", "standard", AS_OF))
    assert h.call("web.fetch", url="https://data.eastmoney.com/report/info/AP202608161828030058.html").content


def test_host_matches_rejects_userinfo_bypass():
    from advisor.skills.data.web import host_matches
    allow = ("stcn.com",)
    assert host_matches("https://www.stcn.com/a/1.html", allow)
    assert not host_matches("https://stcn.com:443@evil.com/x", allow)       # userinfo 冒充白名单主机
    assert host_matches("https://evil.com@stcn.com/x", allow)               # 真实主机仍是 stcn.com
    assert not host_matches("https://stcn.com.evil.com/x", allow)
    assert host_matches("https://STCN.COM./x", allow)                       # 大小写与尾点规范化


def test_fetch_refuses_blocked_domain_even_if_parent_allowed(policy, repo):
    fetch = {s.spec.name: s for s in build_web_skills(FakeWebTools(), repo, policy)}["web.fetch"]
    with pytest.raises(SkillFailed, match="黑名单"):
        fetch.run(fetch.spec.input_model(url="https://guba.eastmoney.com/x"))


def test_fetch_allow_add_cannot_smuggle_blocked_subdomains(policy):
    from advisor.skills.data.web import fetch_allowlist
    extra = fetch_allowlist(policy, {"fetch_allow_add": [
        "r.guba.eastmoney.com", "GUBA.eastmoney.com", "guba.eastmoney.com.", "wt.cfi.cn"]})
    assert extra[len(policy.fetch_allow):] == ("wt.cfi.cn",)


def test_search_skips_invalid_rows_without_failing(policy, repo):
    class Tools:
        def __init__(self):
            self.searches = 0

        def search(self, q, c):
            self.searches += 1
            return [
                {"title": "好结果", "url": "https://www.stcn.com/ok.html", "authority": 4, "rerank_score": 2.0,
                 "date": "2026-08-16T00:00:00+00:00", "summary": "正常"},
                {"title": "坏结果", "url": "https://www.stcn.com/bad.html", "authority": "4.5",
                 "rerank_score": 2.0, "date": None, "summary": "脏数据"},
            ]

        def fetch(self, url):
            return {"title": "", "content": "x"}

    tools = Tools()
    search = {s.spec.name: s for s in build_web_skills(tools, repo, policy)}["web.search"]
    out = search.run(search.spec.input_model(query="示例酒业 业绩", as_of=AS_OF))
    assert {h.url for h in out.hits} == {"https://www.stcn.com/ok.html"}
    assert out.dropped["invalid_row"] == 1
    again = search.run(search.spec.input_model(query="示例酒业 业绩", as_of=AS_OF))
    assert again.dropped["invalid_row"] == 1 and tools.searches == 1      # 脏数据进了缓存也不拖垮后续
