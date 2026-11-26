"""配置 fail-closed 回归：egress 缺段/坏正则/必需流程/预算校验启动即失败、
auth_user 默认与文档一致、演示数据导入跟随 MONGO_DB。"""
import pytest
import yaml

from advisor.harness.policy import Policy
from advisor.settings import ROOT, Settings


def _doc(**over) -> dict:
    routes = {name: {"llm_role": "draft", "allowed_skills": ["*"],
                     "budget": {"tokens": 10, "search_calls": 1, "fetch_calls": 0, "wall_clock_s": 10},
                     "max_iterations": 0} for name in ("quick", "standard", "deep", "mcp")}
    doc = {"routes": routes,
           "egress": {"sensitive_patterns": [r"\d{6,}"], "trade_intent_patterns": ["买入"]},
           "approvals": {"default": "auto"}}
    doc.update(over)
    return doc


def _load(tmp_path, doc: dict) -> Policy:
    (tmp_path / "policies.yaml").write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    (tmp_path / "sites.yaml").write_text("fetch_allow: [example.com]\nblock: []\n", encoding="utf-8")
    return Policy.load(tmp_path)


def test_egress_missing_or_bad_regex_is_fatal(tmp_path):
    doc = _doc()
    doc.pop("egress")
    with pytest.raises(ValueError, match="egress"):
        _load(tmp_path, doc)                               # 脱敏段缺失会静默失效：拒绝启动
    with pytest.raises(ValueError, match="egress"):
        _load(tmp_path, _doc(egress={"sensitive_patterns": [], "trade_intent_patterns": ["买入"]}))
    with pytest.raises(ValueError, match="egress|正则"):
        _load(tmp_path, _doc(egress={"sensitive_patterns": ["(("], "trade_intent_patterns": []}))
    _load(tmp_path, _doc())                                # 合法配置正常加载


def test_required_routes_validated(tmp_path):
    doc = _doc()
    doc["routes"].pop("mcp")
    with pytest.raises(ValueError, match="必需流程"):
        _load(tmp_path, doc)                               # 代码硬编码使用的流程不能悄悄缺失


def test_budget_keys_and_types_validated(tmp_path):
    doc = _doc()
    doc["routes"]["quick"]["budget"].pop("tokens")
    with pytest.raises(ValueError, match="预算键"):
        _load(tmp_path, doc)
    zero = _doc()
    zero["routes"]["quick"]["budget"]["wall_clock_s"] = 0
    with pytest.raises(ValueError, match="必须为正"):
        _load(tmp_path, zero)
    bad = _doc()
    bad["routes"]["quick"]["budget"]["tokens"] = "abc"
    with pytest.raises(ValueError, match="无效"):
        _load(tmp_path, bad)                               # 字符串/负数/非有限数启动即拒


def test_auth_user_default_matches_docs():
    assert Settings(_env_file=None).auth_user == "admin"   # README 承诺「用户名默认 admin」
    assert "默认 `admin`" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_demo_restore_follows_mongo_db():
    script = (ROOT / "deploy/mongo-init/01-restore-demo.sh").read_text(encoding="utf-8")
    assert "${MONGO_DB" in script and "--nsTo" in script   # 改库名时演示数据不能导入旧库
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "MONGO_DB" in compose
