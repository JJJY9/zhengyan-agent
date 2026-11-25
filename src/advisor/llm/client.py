"""DeepSeek（OpenAI 兼容接口）。代码按角色取模型，输出必须通过 Pydantic Schema。"""
import json
import re
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from advisor.settings import Settings, load_yaml

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


class LLMOutputError(Exception):
    def __init__(self, message: str, tokens: int = 0):
        super().__init__(message)
        self.tokens = tokens            # 失败前已真实消耗的 token：Harness 据此计费


def _json_payload(content: str) -> str:
    """取出 JSON 主体：去掉 markdown 代码块围栏与前后缀说明文字。"""
    t = (content or "").strip()
    if m := _FENCE.search(t):
        t = m.group(1).strip()
    if t and not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if 0 <= i < j:
            t = t[i:j + 1]
    return t


class LLMNotConfigured(RuntimeError):
    """没有配置大模型 Key：服务可以启动、浏览已有数据，但不能发起新研究。"""


class DeepSeekClient:
    def __init__(self, settings: Settings, roles: dict[str, Any] | None = None):
        self.roles = roles or load_yaml("models.yaml", settings.config_dir)["roles"]
        # 没配 Key 也允许服务启动（可以浏览已有档案、RSI 账本和评测结果）；真正调用模型时才报错
        self._client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=300,
            max_retries=2,
        ) if settings.deepseek_api_key else None

    @property
    def configured(self) -> bool:
        return self._client is not None

    def json(self, role: str, system: str, user: str, schema: type[BaseModel],
             max_tokens: int | None = None) -> tuple[BaseModel, int]:
        if self._client is None:
            raise LLMNotConfigured("未配置 DEEPSEEK_API_KEY：请在 .env 中填写后重启服务")
        cfg = self.roles[role]
        schema_hint = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        messages = [
            {"role": "system", "content": f"{system}\n\n只输出一个 JSON 对象，符合以下 JSON Schema：\n{schema_hint}"},
            {"role": "user", "content": user},
        ]
        extra = {"max_tokens": max_tokens} if max_tokens else {}
        tokens = 0
        for attempt in range(2):  # 一次修复重试
            try:
                resp = self._client.chat.completions.create(
                    model=cfg["model"],
                    messages=messages,
                    response_format={"type": "json_object"},
                    extra_body={"reasoning_effort": cfg.get("effort", "high")},
                    **extra,
                )
            except Exception as e:      # 供应商可能已计费：把累计用量带给上层
                setattr(e, "advisor_tokens", tokens)
                raise
            choice = resp.choices[0] if getattr(resp, "choices", None) else None
            message = getattr(choice, "message", None)
            content = (getattr(message, "content", "") or "").strip()
            usage = getattr(resp, "usage", None)
            total = getattr(usage, "total_tokens", None) if usage is not None else None
            if total is not None:
                tokens += int(total)
            else:                       # 端点省略 usage：保守估算，避免预算静默失效
                tokens += max(1, (len(system) + len(user) + len(content)) // 4)
            if not content:             # 200 + 空 choices / 空内容：按坏输出走修复重试
                if attempt == 1:
                    raise LLMOutputError("模型返回空内容", tokens=tokens)
                messages += [
                    {"role": "assistant", "content": "(空输出)"},
                    {"role": "user", "content": "上一次没有返回内容。请只输出一个 JSON 对象。"},
                ]
                continue
            try:
                return schema.model_validate_json(_json_payload(content)), tokens
            except ValidationError as e:
                if attempt == 1:
                    raise LLMOutputError(f"模型输出不符合 Schema: {e.errors()[:2]}", tokens=tokens) from e
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"上面的 JSON 不符合 Schema：{e.errors()[:3]}。请修正后只输出 JSON。"},
                ]
        raise LLMOutputError("unreachable", tokens=tokens)
