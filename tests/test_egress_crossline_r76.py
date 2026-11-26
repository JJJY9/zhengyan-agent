"""R76a 回归：跨行拆分号码整块脱敏、空行屏障不过度遮蔽、intake 提示词不含拆行号码、
Docker test 镜像补齐测试直读文件、models.yaml roles 严格校验（含代码硬编码角色）、README 顺序。"""
import shutil

import pytest
import yaml

from advisor.contracts import ResearchRequest
from advisor.harness.egress import EgressGuard
from advisor.harness.policy import Policy
from advisor.settings import ROOT
from tests.fakes import CODE, FakeLLM, as_of
from tests.test_phase2 import make_service

CONFIG = ROOT / "config"


def _guard(policy):
    return EgressGuard(policy.egress.get("sensitive_patterns", []), policy.egress.get("trade_intent_patterns", []))


@pytest.mark.parametrize("text,parts", [
    ("我的手机号 1381234\n5678 请联系我", ["1381234", "5678"]),
    ("我的卡号 622202020000\n123456789012", ["622202020000", "123456789012"]),
    ("身份证 11010119900307\n123X", ["11010119900307", "123X"]),
    ("我的手机号 1381234\r\n5678 请联系我", ["1381234", "5678"]),
])
def test_split_numbers_are_masked(policy, text, parts):
    """数字分隔符含换行：find_sensitive 整段命中后，redact 必须把两行都整体遮蔽，而不是原样出域。"""
    guard = _guard(policy)
    assert guard.find_sensitive(text)
    red = guard.redact(text)
    for p in parts:
        assert p not in red, red


def test_cross_blank_personal_description_masked(policy):
    """跨空行的个人描述与 find_sensitive 同口径命中：两块的数字都要遮蔽，不能只遮第一块。"""
    guard = _guard(policy)
    text = "我持有600519 1000股\n\n成本1500元，该怎么办"
    assert guard.find_sensitive(text)
    red = guard.redact(text)
    assert "600519" not in red and "1000" not in red and "1500" not in red


def test_cross_blank_split_numbers_masked(policy):
    """空行拆分的手机号/卡号同样按整段归一化命中并遮蔽。"""
    guard = _guard(policy)
    for text in ("我的手机号 1381234\n\n5678 请联系我",
                 "我的卡号 622202020000\n\n123456789012"):
        assert guard.find_sensitive(text)
        red = guard.redact(text)
        assert guard.find_sensitive(red) == []


def test_intake_prompt_has_no_split_phone(repo, policy):
    """intake 的大模型提示词用与 find_sensitive 同口径的整块脱敏：换行拆开的手机号不会原文发给模型。"""
    llm = FakeLLM()
    svc = make_service(repo, policy, llm=llm)
    tid = svc.start(ResearchRequest(question=f"我的手机号 1381234\n5678，想了解 {CODE} 半年报业绩前景",
                                    as_of=as_of(), route="standard"), background=False)
    task = svc.deps.repo.get_task(tid)
    assert task["status"] == "waiting_approval" and task["interrupt"]["type"] == "consent"
    prompt = "\n".join(llm.prompts.get("IntakeOut", []))
    assert prompt, "intake 应至少调用一次分类模型"
    assert "1381234" not in prompt and "5678" not in prompt
    svc.resume(tid, approve=True, background=False)
    assert svc.deps.repo.get_task(tid)["status"] == "done"


def test_docker_test_image_copies_all_read_repo_files():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    test_stage = dockerfile.split("# ---------- 3. 测试")[1]
    for name in ("deploy/mongo-init/01-restore-demo.sh", "docker-compose.yml"):
        assert f"COPY {name}" in test_stage
    assert "COPY Dockerfile .dockerignore .env.example ./" in test_stage


def test_readme_way2_creates_env_before_compose():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("### 方式二")[1]
    assert section.index("cp .env.example .env") < section.index("docker compose up -d mongo")
    assert "即使只启动 mongo 也要求它存在" in section


def _copy_config(tmp_path, roles_yaml):
    for name in ("policies.yaml", "sites.yaml"):
        shutil.copy(CONFIG / name, tmp_path / name)
    (tmp_path / "models.yaml").write_text(roles_yaml, encoding="utf-8")


@pytest.mark.parametrize("roles_yaml", ["roles: {}\n", "roles: []\n", "roles:\n", "classify: x\n"])
def test_policy_rejects_empty_or_invalid_roles(tmp_path, roles_yaml):
    """models.yaml 存在但 roles 空/非映射：没有任何可用角色，必须启动即失败。"""
    _copy_config(tmp_path, roles_yaml)
    with pytest.raises(ValueError, match="roles"):
        Policy.load(tmp_path)


def test_policy_rejects_missing_code_required_role(tmp_path):
    """代码硬编码使用的 classify/draft/reason 缺失时，启动即拒绝（否则运行期静默退化）。"""
    roles = yaml.safe_load((CONFIG / "models.yaml").read_text(encoding="utf-8"))["roles"]
    roles.pop("classify")
    _copy_config(tmp_path, "roles:\n" + "".join(f"  {k}: {v}\n" for k, v in roles.items()))
    with pytest.raises(ValueError, match="classify"):
        Policy.load(tmp_path)
