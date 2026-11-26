"""路由金标结构校验：标签/意图取值、归一化去重、三档样本量与均衡——防止后续编辑把校准集改坏。"""
import json
import re

from advisor.settings import ROOT

ALLOWED_INTENTS = {None, "explain", "research", "trade_execution"}


def _norm(q: str) -> str:
    return re.sub(r"[\s，。？?！!,.、：:；;]+", "", q).lower()


def test_routing_golden_structure():
    doc = json.loads((ROOT / "evals" / "golden" / "routing_cases.json").read_text(encoding="utf-8"))
    cases = doc["cases"]
    assert len(cases) >= 30                                    # 三档各 ≥10 才能做校准
    labels = [c["label"] for c in cases]
    assert set(labels) <= {"quick", "standard", "deep"}
    for c in cases:
        assert isinstance(c["question"], str) and len(c["question"].strip()) >= 4
        assert c.get("intent") in ALLOWED_INTENTS
    normalized = [_norm(c["question"]) for c in cases]
    assert len(normalized) == len(set(normalized))             # 归一化后唯一：近似重复题会虚抬准确率
    for label in ("quick", "standard", "deep"):
        assert labels.count(label) >= 10
        assert labels.count(label) >= len(cases) * 0.2          # 三档相对均衡，校准不被单档主导
