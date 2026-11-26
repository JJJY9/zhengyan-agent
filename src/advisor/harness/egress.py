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
# 跨行分隔只在「两侧都是完整日期」时不合并：证据摘录里相邻日期行不能被拼成假银行卡号
_DATE_TAIL = re.compile(r"\d{4}[-–—－/.]\d{1,2}[-–—－/.]\d{1,2}\s*$")
_DATE_HEAD = re.compile(r"^\s*\d{4}[-–—－/.]\d{1,2}[-–—－/.]\d{1,2}(?!\d)")
_SECRET_HINT = re.compile(r"(?i)sk-|bearer|akia|xox|@")
_CJK = re.compile(r"[\u4e00-\u9fff]")


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

    def _scan(self, text: str) -> tuple[str, list[int], list[int]]:
        """整段归一化（与 find_sensitive 同口径），返回 (归一化串, 归一化索引→原字符索引, 原字符索引→行号)。

        数字之间的空白/连字符/换行都合并（拆行号码按整段才算命中）；但两侧都是完整日期时不合并。
        """
        text = text or ""
        line_of: list[int] = []
        for li, part in enumerate(text.split("\n")):
            line_of.extend([li] * len(part))
            line_of.append(li)
        line_of = line_of[:len(text)]
        norm_chars: list[str] = []
        norm_orig: list[int] = []
        j, n = 0, len(text)
        prev_digit = False
        sep_chars = "-–—－"
        while j < n:
            ch = text[j]
            if ord(ch) in _FORMAT_CHARS:
                j += 1
                continue
            t = ch.translate(_FULLWIDTH)
            if prev_digit and (t in sep_chars or t.isspace()):
                k = j + 1
                while k < n and (text[k] in sep_chars or text[k].isspace()
                                 or ord(text[k]) in _FORMAT_CHARS):
                    k += 1
                if k < n and text[k].translate(_FULLWIDTH).isdigit():
                    bridge_dates = ("\n" in text[j:k]
                                    and _DATE_TAIL.search(text[:j]) and _DATE_HEAD.match(text[k:]))
                    if not bridge_dates:
                        j = k                       # 数字间分隔（含换行）整体合并
                        continue
            norm_chars.append(t)
            norm_orig.append(j)
            prev_digit = t.isdigit()
            j += 1
        return "".join(norm_chars), norm_orig, line_of

    def find_sensitive(self, text: str) -> list[str]:
        norm, _, _ = self._scan(text)
        return [norm[s:e] for s, e in self._spans(norm)]

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

    def redact(self, text: str, keep_names: bool = True) -> str:
        """命中个人敏感信息时，把命中覆盖到的行上的数字整体遮蔽；凭据形状的片段整段替换。

        定位与 find_sensitive 完全同一套整段归一化（数字间空白/连字符/换行合并，拆行号码按整段命中），
        杜绝「检测到了但脱敏没覆盖」的口径分裂。遮蔽粒度：
        - span 覆盖到的行里有数字的，整行数字替换为 [已脱敏]（成本/股数/账号同属个人财务信息）；
        - span 文本是凭据形状（邮箱/sk-/Bearer/AKIA/无中文的字母数字串）时整段替换，避免邮箱等无数字泄露；
        - 含中文的描述（如「我有贵州茅台1000股」）保留文字只抹数字，入口识别仍能读到公司名。
        相邻日期行不会被拼成假号码；证据区只有在与个人描述真正连成一个命中时才受影响——提示词拼装
        一律使用预脱敏的问题文本，证据数字不受问题区影响。
        """
        text = text or ""
        norm, norm_orig, line_of = self._scan(text)
        drop: dict[int, int] = {}                 # 原字符索引 → 替换段起点（起点插占位符）
        hit_lines: set[int] = set()
        for start, end in self._spans(norm):
            chunk = norm[start:end]
            secret_like = (_SECRET_HINT.search(chunk)
                           or (re.search(r"[A-Za-z]", chunk) and not _CJK.search(chunk)))
            if secret_like or not keep_names:
                origs = norm_orig[start:end]
                if origs:
                    head = min(origs)
                    for o in origs:
                        drop[o] = head                 # 凭据/严格模式：整段替换为占位符
            for k in range(start, min(end, len(norm))):
                if norm[k].isdigit():
                    hit_lines.add(line_of[norm_orig[k]])
        if not drop and not hit_lines:
            return text
        out_lines = []
        for li, line in enumerate(text.split("\n")):
            if li in hit_lines or any(0 <= o < len(text) and line_of[o] == li for o in drop):
                out_lines.append(_DIGITS.sub("[已脱敏]", line))
            else:
                out_lines.append(line)
        if drop:
            rebuilt_chars: list[str] = []
            for o, ch in enumerate(text):
                if o in drop:
                    if o == drop[o] and ch != "\n":
                        rebuilt_chars.append("[已脱敏]")
                    continue
                rebuilt_chars.append(ch)
            out_lines = []
            for li, line in enumerate("".join(rebuilt_chars).split("\n")):
                out_lines.append(_DIGITS.sub("[已脱敏]", line) if li in hit_lines else line)
        return "\n".join(out_lines)
