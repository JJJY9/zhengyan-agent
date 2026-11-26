"""出域检查：搜索查询词禁止含个人敏感信息；发往模型的文本先脱敏。

匹配前先归一：剥掉零宽/双向控制等格式字符、全角数字转半角、数字之间的空格/连字符合并——
这些写法能拆开关键词或拆分号码，绕过未归一的模式。一旦命中个人敏感信息，
按 consent 承诺对命中所在行的全部数字遮蔽（金额/股数/成本/账号同属个人财务信息），
而不是只替换命中片段；其他行不动，避免把提示词里的证据区一并毁掉。
"""
import re

from advisor.harness.errors import HarnessDenied

_FORMAT_CHARS = dict.fromkeys(map(ord, "​‌‍‎‏‪‫‬‭‮"
                                       "⁠⁦⁧⁨⁩﻿"), None)
_FULLWIDTH = str.maketrans("０１２３４５６７８９＋", "0123456789+")
_SEP_BETWEEN_DIGITS = re.compile(r"(?<=\d)[\s\-–—－]+(?=\d)")
_DIGITS = re.compile(r"\d+(?:\.\d+)?")


def _normalize(text: str) -> str:
    t = (text or "").translate(_FORMAT_CHARS).translate(_FULLWIDTH)
    return _SEP_BETWEEN_DIGITS.sub("", t)


class EgressGuard:
    def __init__(self, sensitive_patterns: list[str], trade_patterns: list[str] | None = None):
        self._sensitive = [re.compile(p) for p in sensitive_patterns]
        self._trade = [re.compile(p) for p in (trade_patterns or [])]

    def _spans(self, text: str) -> list[list[int]]:
        spans = sorted((m.start(), m.end()) for p in self._sensitive for m in p.finditer(text))
        merged: list[list[int]] = []
        for start, end in spans:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return merged

    def find_sensitive(self, text: str) -> list[str]:
        t = _normalize(text)
        return [m.group(0) for p in self._sensitive for m in p.finditer(t)]

    def has_trade_intent(self, text: str) -> bool:
        t = _normalize(text)
        return any(p.search(t) for p in self._trade)

    def redact_amounts(self, text: str) -> str:
        """只脱敏命中区间里的数字/账号，保留公司名等上下文（入口识别用；随后仍有常规 redact 兜底）。"""
        out = []
        for line in (text or "").split("\n"):
            norm = _normalize(line)
            for start, end in reversed(self._spans(norm)):
                chunk = norm[start:end]
                masked = re.sub(r"\d+(?:\.\d+)?\s*(?:万|千|元|股|手)?", "[已脱敏]", chunk)
                norm = norm[:start] + (masked if masked != chunk else "[已脱敏]") + norm[end:]
            out.append(norm if self._spans(_normalize(line)) else line)
        return "\n".join(out)

    def check_outbound(self, text: str) -> None:
        if self.find_sensitive(text):
            raise HarnessDenied("出域内容含个人敏感信息，已阻断")

    def redact(self, text: str) -> str:
        """命中个人敏感信息时，把命中覆盖到的行的数字整体遮蔽。

        consent 承诺「这些信息会被脱敏」：同一条个人描述里的成本、股数、账号常常落在正则命中片段之外，
        只替换片段会原样出域。定位必须与 find_sensitive 完全同口径：按「空行分块」，块内对整段做
        归一化（数字之间的空白/连字符、包括换行都会被合并——手机号「1381234\n5678」按整段才算命中），
        命中 span 覆盖到哪几行就整体遮蔽哪几行的数字；空行是硬屏障，问题区与证据区不会互相桥接。
        """
        lines = (text or "").splitlines() or [""]
        out = list(lines)
        blocks: list[list[int]] = []
        cur: list[int] = []
        for i, line in enumerate(lines):
            if not line.strip():
                if cur:
                    blocks.append(cur)
                    cur = []
                continue
            cur.append(i)
        if cur:
            blocks.append(cur)
        sep_chars = set("-–—－")
        for idxs in blocks:
            group = [lines[i] for i in idxs]
            joined = "\n".join(group)
            line_of: list[int] = []
            for li, part in enumerate(group):
                line_of.extend([li] * len(part))
                if li < len(group) - 1:
                    line_of.append(li)          # 换行符归属上一行
            norm_chars: list[str] = []
            norm_line: list[int] = []
            j, n = 0, len(joined)
            prev_digit = False
            while j < n:
                ch = joined[j]
                if ord(ch) in _FORMAT_CHARS:
                    j += 1                      # 零宽等格式字符不参与归一化
                    continue
                t = ch.translate(_FULLWIDTH)
                if prev_digit and (t in sep_chars or t.isspace()):
                    k = j + 1
                    while k < n and (joined[k] in sep_chars or joined[k].isspace()
                                     or ord(joined[k]) in _FORMAT_CHARS):
                        k += 1
                    if k < n and joined[k].translate(_FULLWIDTH).isdigit():
                        j = k                       # 数字之间的分隔（含换行）整体合并
                        continue
                norm_chars.append(t)
                norm_line.append(line_of[j])
                prev_digit = t.isdigit()
                j += 1
            norm = "".join(norm_chars)
            hit = set()
            for start, end in self._spans(norm):
                for k in range(start, min(end, len(norm_line))):
                    hit.add(norm_line[k])
            for li in hit:
                gline = _normalize(group[li])
                for start, end in reversed(self._spans(gline)):
                    gline = gline[:start] + "[已脱敏]" + gline[end:]
                out[idxs[li]] = _DIGITS.sub("[已脱敏]", gline)
        return "\n".join(out)
