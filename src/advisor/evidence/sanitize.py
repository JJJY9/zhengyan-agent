"""外部网页只是数据：去掉脚本与疑似指令文本，防止提示注入经检索进入推理。"""
import html
import re

_SCRIPT = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_ZERO_WIDTH = re.compile(r"[​-‏‪-‮⁠﻿]")
_INJECTION = re.compile(
    r"(?:忽略|无视|忘记|忘掉|不要(?:再)?(?:理会|遵守|相信)|别(?:理会|管))[^。！？!?\n]{0,16}"
    r"(?:之前|以上|上述|前面|系统|你的|全部|所有)?[^。！？!?\n]{0,12}"
    r"(?:指令|提示词?|规则|要求|设定|限制|prompt|系统提示)"
    r"|(?:忽略|无视)[^。！？!?\n]{0,8}(?:之前|以上|上述|前面)"
    r"|ignore\s+(?:all\s+|the\s+)?(?:previous|above|prior|earlier|system|your)"
    r"|disregard\s+(?:all\s+|the\s+)?(?:previous|above|prior|instructions?|system)"
    r"|forget\s+(?:everything|all|your\s+instructions)"
    r"|system\s+prompt|developer\s+mode|<\|im_start\|>"
    r"|###\s*(?:system|instruction)"
    r"|你(?:现在)?是(?:一个)?(?:AI|人工智能|语言模型)"
    r"|请(?:将|把)(?:以下|上述|上面)[^。！？!?\n]{0,16}(?:发送|输出|转发|告诉)",
    re.I,
)


def sanitize(text: str, max_chars: int = 4000) -> tuple[str, bool]:
    """返回（清洗后文本，是否发现疑似注入）。"""
    text = _ZERO_WIDTH.sub("", text or "")
    text = _SCRIPT.sub(" ", text)
    text = _MD_IMAGE.sub(" ", text)
    text = html.unescape(_TAG.sub(" ", text))
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t　]+", " ", line).strip()
        if line:
            lines.append(line)
    # 跨行拆写的注入（“忽略\n之前的指令”）在整段文本上判定；单行能命中的行直接剔除
    joined = re.sub(r"\s+", " ", "\n".join(lines))
    suspicious = bool(_INJECTION.search(joined))
    kept = [ln for ln in lines if not _INJECTION.search(ln)]
    if suspicious and len(kept) == len(lines):
        return "", True          # 整段命中但没有任何单行命中：按注入处理整段
    return "\n".join(kept)[:max_chars], suspicious
