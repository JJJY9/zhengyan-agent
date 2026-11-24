"""来源独立性：转载数不等于证据数。按原始来源与内容指纹聚成事件簇。

同标题或正文指纹接近视为同一事件；但指纹接近还不够：正文的数字集合或涨跌方向不一致
（“营收 910 亿” vs “9100 亿”、“同比增长” vs “同比下降”）时不能合并——否则冲突信息会被
当成转载丢掉。标题先剥掉站点后缀（“-新浪财经”“（东方财富网）”等）再比较。
"""
import hashlib
import re
from urllib.parse import urlparse

from advisor.contracts import SearchHit

_PUBLISHER = re.compile(r"(?:^|[\n。；;])\s*来源[:：]\s*([0-9A-Za-z一-龥]{2,20}?)(?=记者|通讯员|编辑|\s|$)")
_SITE_SUFFIX = re.compile(r"\s*[-–—_|]\s*[一-龥A-Za-z]{2,12}$")
_PAREN_SUFFIX = re.compile(r"\s*[（(][一-龥A-Za-z]{2,12}[)）]\s*$")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_DOWN = re.compile(r"下降|下滑|减少|降低|回落|负增长|下跌|降幅|亏损|下调|收窄|减持")
_UP = re.compile(r"上升|上涨|增长|提升|提高|增加|增幅|改善|回升|增持")


def origin_id(url: str) -> str:
    p = urlparse(url)
    host = re.sub(r"^(www|m|wap)\.", "", p.netloc.lower())
    return f"{host}{p.path.rstrip('/')}"


def extract_publisher(text: str, fallback: str | None) -> str | None:
    """只认行首/句首的“来源：媒体名”；记者名、正文里的“资金来源：”不算署名。"""
    m = _PUBLISHER.search(text or "")
    return m.group(1) if m else fallback


def simhash(text: str, bits: int = 64) -> int:
    text = re.sub(r"\s+", "", text)
    grams = [text[i:i + 2] for i in range(max(len(text) - 1, 1))]
    vec = [0] * bits
    for g in grams:
        h = int.from_bytes(hashlib.md5(g.encode()).digest()[:8], "big")
        for i in range(bits):
            vec[i] += 1 if (h >> i) & 1 else -1
    return sum(1 << i for i in range(bits) if vec[i] > 0)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _norm_title(title: str, site_name: str | None = None) -> str:
    t = (title or "").strip()
    if site_name:
        for sep in ("-", "–", "—", "_", "|"):
            suffix = f"{sep}{site_name.strip()}"
            if t.endswith(suffix):
                t = t[: -len(suffix)].strip()
        for open_, close in (("（", "）"), ("(", ")")):
            suffix = f"{open_}{site_name.strip()}{close}"
            if t.endswith(suffix):
                t = t[: -len(suffix)].strip()
    t = t.split("_")[0].split("|")[0]
    t = _SITE_SUFFIX.sub("", t).strip()
    t = _PAREN_SUFFIX.sub("", t).strip()
    return re.sub(r"[\s\W_]+", "", t)


def _direction(text: str) -> str:
    down, up = bool(_DOWN.search(text)), bool(_UP.search(text))
    return "down" if down and not up else "up" if up and not down else ""


def _body_key(hit: SearchHit) -> tuple[int | None, set[str], str]:
    body = re.sub(r"\s+", "", (hit.summary or hit.snippet)[:400])
    if not body:
        return None, set(), ""
    return simhash(body), set(_NUM.findall(body)), _direction(body)


def _body_mergeable(a: tuple[int | None, set[str], str], b: tuple[int | None, set[str], str],
                    threshold: int) -> bool:
    sa, na, da = a
    sb, nb, db = b
    if sa is None or sb is None or hamming(sa, sb) > threshold:
        return False
    # 数字集合或涨跌方向不一致时，指纹再近也不是同一事件
    return na == nb and da == db


def _title_mergeable(a: tuple[int | None, set[str], str], b: tuple[int | None, set[str], str]) -> bool:
    """同标题可合并，但双方都有正文且数字/方向冲突时仍视为不同事件。"""
    if a[0] is None or b[0] is None:
        return True
    return a[1] == b[1] and a[2] == b[2]


def assign_clusters(hits: list[SearchHit], threshold: int = 10) -> dict[str, str]:
    """返回 url -> cluster_id。标题相同或正文指纹接近（且数字/方向一致）视为同一事件。

    空标题/空正文不参与相似判断：否则所有「只有链接没有摘要」的结果会因指纹相同被
    误聚成一簇，gather 只保留一个代表，独立来源被当成转载丢掉。
    每个新 hit 与簇内任一成员比较（单链接），避免只和代表比较造成同稿多截断版本拆簇。
    """
    clusters: list[dict] = []                    # {"cid": str, "members": [(title, body_key)]}
    result: dict[str, str] = {}
    for hit in sorted(hits, key=lambda h: (-h.authority, -h.rerank_score)):
        title, body = _norm_title(hit.title, hit.site_name), _body_key(hit)
        for cl in clusters:
            if any((bool(title) and title == m[0] and _title_mergeable(body, m[1]))
                   or _body_mergeable(body, m[1], threshold)
                   for m in cl["members"]):
                result[hit.url] = cl["cid"]
                cl["members"].append((title, body))
                break
        else:
            cid = "cl_" + hashlib.sha1(hit.url.encode()).hexdigest()[:8]
            clusters.append({"cid": cid, "members": [(title, body)]})
            result[hit.url] = cid
    return result
