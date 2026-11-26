"""R72a 配置/部署/安全巡检回归：Docker 测试镜像内容、compose 兼容与健康检查、
requirements-dev 与 pyproject 一致、死配置清理、模型角色/站点结构启动校验、
跨行脱敏、决策模型错误串不含凭据、CI 权限与 .env 文档。"""
import shutil
from types import SimpleNamespace

import pytest
import yaml

from advisor.api.research import meta
from advisor.harness.egress import EgressGuard
from advisor.harness.policy import Policy
from advisor.llm.decision import DecisionClient, DecisionError
from advisor.settings import ROOT
from tests.test_phase2 import make_service

CONFIG = ROOT / "config"


def test_docker_test_image_has_fixture_files():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    test_stage = dockerfile.split("# ---------- 3. 测试")[1]
    assert "COPY deploy/mongo-init/01-restore-demo.sh" in test_stage   # 配置校验用例要读它
    assert "COPY docker-compose.yml" in test_stage
    ignored = [ln.strip() for ln in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()]
    assert "deploy" not in ignored                                    # 不能整目录排除出构建上下文


def test_compose_env_file_and_healthcheck():
    doc = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    env_file = doc["services"]["app"]["env_file"]
    assert isinstance(env_file, list) and all(isinstance(x, str) for x in env_file)   # 旧版 Compose v2 也能解析
    health = " ".join(doc["services"]["mongo"]["healthcheck"]["test"])
    assert "/proc/1/comm" in health                                    # 初始化临时 mongod 不能提前 healthy


def test_requirements_dev_matches_pyproject():
    req = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "ruff>=0.17.6" in pyproject
    assert "ruff==" in req                                            # 未重新导出时 pip 路径/Docker test 镜像缺 ruff


def test_meta_expert_cfg_whitelist(repo, policy):
    svc = make_service(repo, policy)
    svc.deps.policy.experts["dead_config_key"] = 123                  # 任何未消费的键都不该被当成权威配置展示
    cfg = meta(SimpleNamespace(deps=svc.deps))["expert_cfg"]
    assert "dead_config_key" not in cfg
    assert "debate_rounds" not in cfg
    assert set(cfg) <= {"max_team", "max_challenges", "search_per_expert", "fetch_per_expert"}
    assert "debate_rounds" not in (yaml.safe_load((CONFIG / "policies.yaml").read_text(encoding="utf-8"))["experts"])


def test_policy_rejects_unknown_model_role(tmp_path):
    for name in ("policies.yaml", "sites.yaml", "models.yaml"):
        shutil.copy(CONFIG / name, tmp_path / name)
    doc = yaml.safe_load((tmp_path / "policies.yaml").read_text(encoding="utf-8"))
    doc["routes"]["standard"]["llm_role"] = "reasoning"              # 拼错：首个研究任务才会 KeyError
    (tmp_path / "policies.yaml").write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="未定义这些角色"):
        Policy.load(tmp_path)


def test_policy_rejects_bad_sites(tmp_path):
    for name in ("policies.yaml", "models.yaml"):
        shutil.copy(CONFIG / name, tmp_path / name)
    (tmp_path / "sites.yaml").write_text("fetch_allow_misspelled: [a.com]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fetch_allow"):
        Policy.load(tmp_path)
    (tmp_path / "sites.yaml").write_text("fetch_allow: [a.com]\nfetch_alow: [b.com]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未知键"):                    # 拼错的多余键会让白名单语义漂移
        Policy.load(tmp_path)
    (tmp_path / "sites.yaml").write_text("fetch_allow: []\nblock: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fetch_allow"):
        Policy.load(tmp_path)


def test_env_example_documents_proxy_and_decision_model():
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "FORWARDED_ALLOW_IPS" in env
    assert "DECISION_MODEL=" in env and "DECISION_API_KEY=" in env
    assert "docs/小模型决策节点实施方案.md" not in env                   # 引用了不存在的文档


def test_redact_masks_cross_line_personal_amounts(policy):
    """个人描述跨行时（我持有 + 换行 + 成本金额），被跨过的行也必须整体脱敏；证据行不动。"""
    guard = EgressGuard(policy.egress.get("sensitive_patterns", []), policy.egress.get("trade_intent_patterns", []))
    text = "我的 600519\n成本 120元，占我资产的 30%\n证据：营收 910亿元"
    assert guard.find_sensitive(text)
    red = guard.redact(text)
    assert "600519" not in red and "120" not in red and "30" not in red
    assert "910" in red                                                # 未命中的证据行保持原样


def test_decision_error_scrubs_credentials():
    import httpx

    client = DecisionClient(base_url="http://user:secret@127.0.0.1:9", api_key="k", timeout_s=2)
    client._client = httpx.Client(base_url=client._client.base_url,
                                  transport=httpx.MockTransport(lambda req: httpx.Response(500, json={})))
    with pytest.raises(DecisionError) as ei:
        client.choose("s", {"q": ("i", {"a": "x"})})                  # HTTPStatusError 消息含带凭据的完整 URL
    assert "secret" not in str(ei.value)                               # 错误串会进入任务事件/审计/导出
