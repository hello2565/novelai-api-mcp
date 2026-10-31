"""Booru tag 检索（`tag_vectors.db`：tag/中文/别名/嵌入向量矩阵）。

向量列由用户自行填充（同模型族）；未填充或未配置 `NAI_EMBEDDING_URL` 时
自动退化为中文/别名/标签名的词面匹配。数据库由 `tool/build_tag_vector_db.py`
从 Launcher 的 `tag_catalog.db` 生成。

性能设计：
- 向量以 numpy 矩阵缓存（首查一次 266MB float32），余弦为一次矩阵-向量
  乘法；未装 numpy 时回退纯 Python（慢数秒，仅建议安装 numpy）。
- 词面匹配的 zh/tag/别名小写字符串在加载时预计算，查询不再逐行 lower()。
"""

import json
import os
import sqlite3
import struct
import urllib.request
from pathlib import Path

try:  # 可选加速：numpy 走矩阵余弦，缺失时回退纯 Python
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

DB_PATH = Path(
    os.environ.get(
        "NAI_TAG_DB", str(Path(__file__).resolve().parent / "tag_vectors.db")
    )
)

CATEGORY_NAMES = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta",
}

_CACHE: dict[str, object] = {
    "mtime": None,
    "tags": [],
    "dim": 0,
    "matrix": None,
    "norms": None,
}


def _load_cache():
    if not DB_PATH.exists():
        return [], 0
    mtime = DB_PATH.stat().st_mtime
    if _CACHE["mtime"] == mtime:
        return _CACHE["tags"], _CACHE["dim"]  # type: ignore[return-value]

    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    dim = 0
    meta = {
        row["key"]: row["value"]
        for row in con.execute("SELECT key, value FROM meta")
    }
    dim = int(meta.get("dim") or 0)
    tags: list[dict] = []
    blobs: list[tuple[int, bytes]] = []
    for row in con.execute(
        "SELECT tag, category, post_count, zh_cn, aliases, vector "
        "FROM tags_vec ORDER BY post_count DESC"
    ):
        entry = {
            "tag": row["tag"],
            "category": row["category"],
            "post_count": row["post_count"],
            "zh": row["zh_cn"],
            "zh_lower": (row["zh_cn"] or "").lower(),
            "tag_lower": row["tag"].lower(),
            "aliases_lower": [
                alias.lower()
                for alias in (
                    (row["aliases"] or "").split("|") if row["aliases"] else []
                )
            ],
            "aliases": (row["aliases"] or "").split("|") if row["aliases"] else [],
        }
        if row["vector"] is not None:
            blob = bytes(row["vector"])
            if dim == 0:
                dim = len(blob) // 4
            blobs.append((len(tags), blob))
            if _np is None:  # 无 numpy 兜底：挂到行上按需解包
                entry["vector"] = blob
        tags.append(entry)
    con.close()

    matrix = None
    norms = None
    if _np is not None and blobs and dim:
        matrix = _np.empty((len(blobs), dim), dtype=_np.float32)
        for offset, (_, blob) in enumerate(blobs):
            matrix[offset] = _np.frombuffer(blob, dtype=_np.float32, count=dim)
        norms = _np.linalg.norm(matrix, axis=1)
        norms[norms == 0.0] = 1.0
    _CACHE["mtime"] = mtime
    _CACHE["tags"] = tags
    _CACHE["dim"] = dim
    _CACHE["matrix"] = matrix
    _CACHE["norms"] = norms
    return tags, dim


_QUERY_VEC_CACHE: dict[str, object] = {}  # (model, text) → 查询向量（LRU ≤256）


def _query_vector(text: str):
    url = os.environ.get("NAI_EMBEDDING_URL", "").strip()
    if not url:
        return None
    model = os.environ.get("NAI_EMBEDDING_MODEL", "")
    cache_key = f"{model}\x00{text}"
    cached = _QUERY_VEC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    payload = json.dumps({"input": [text], "model": model}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("NAI_EMBEDDING_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['NAI_EMBEDDING_API_KEY']}"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if _np is not None:
        vector = _np.asarray(data["data"][0]["embedding"], dtype=_np.float32)
    else:
        vector = list(data["data"][0]["embedding"])
    if len(_QUERY_VEC_CACHE) >= 256:
        _QUERY_VEC_CACHE.pop(next(iter(_QUERY_VEC_CACHE)))
    _QUERY_VEC_CACHE[cache_key] = vector
    return vector


def _cosine_fallback(query_vec, tags: list[dict], scored: dict, add) -> None:
    """无 numpy 时的逐行余弦（慢，仅为兜底）。"""
    for entry in tags:
        blob = entry.get("vector")
        if blob is None:
            continue
        vector_floats = entry.get("_vec")
        if vector_floats is None:
            vector_floats = list(
                struct.unpack(f"<{len(blob) // 4}f", blob)
            )
            entry["_vec"] = vector_floats
        if len(vector_floats) != len(query_vec):
            continue
        similarity = _cosine(query_vec, vector_floats)
        if similarity > 0.3:
            existing = scored.get(entry["tag"])
            if existing is None or similarity > existing[0]:
                add(entry, similarity, "向量检索")


def _cosine(a, b) -> float:
    return float(
        sum(x * y for x, y in zip(a, b))
        / max(sum(x * x for x in a) ** 0.5 * sum(y * y for y in b) ** 0.5, 1e-30)
    )


def _has_vectors(tags: list[dict]) -> bool:
    return _CACHE["matrix"] is not None or any(t.get("vector") for t in tags)


def semantic_ready() -> bool:
    """向量已填充且运行环境配置了 NAI_EMBEDDING_URL 时为 True。"""
    if not DB_PATH.exists() or not os.environ.get("NAI_EMBEDDING_URL"):
        return False
    tags, _ = _load_cache()
    return _has_vectors(tags)


def search(query: str, limit: int = 10, use_vector: bool = False) -> list[dict] | str:
    """检索标准 booru tag。返回结果列表；库不存在时返回提示文本。

    use_vector=False（默认）只做词面匹配；True 时代码检索向量余弦
    （需已填充向量且配置 NAI_EMBEDDING_URL）。"""
    query = (query or "").strip()
    if not query:
        raise ValueError("query 不能为空")
    limit = max(1, min(int(limit), 30))
    if not DB_PATH.exists():
        return (
            f"标签库 {DB_PATH} 不存在：先运行 "
            "tool/build_tag_vector_db.py 生成后再使用本工具"
        )

    tags, dim = _load_cache()
    lowered = query.lower()
    underscored = lowered.replace(" ", "_")
    scored: dict[str, tuple[float, dict, str]] = {}

    def add(entry: dict, score: float, source: str) -> None:
        scored[entry["tag"]] = (score, entry, source)

    for entry in tags:
        zh = entry["zh_lower"]
        synonyms = entry["aliases_lower"]
        best_source = None
        best_score = 0.0
        if zh == lowered:
            best_source, best_score = "zh 精确", 5.0
        elif any(s == underscored for s in synonyms) or entry["tag_lower"] == underscored:
            best_source, best_score = "tag/别名精确", 4.8
        elif lowered and lowered in zh:
            best_source, best_score = "zh 包含", 2.0
        elif any(underscored and underscored in s for s in synonyms):
            best_source, best_score = "别名包含", 1.8
        elif underscored and underscored in entry["tag_lower"]:
            best_source, best_score = "tag 包含", 1.5
        if best_source:
            # 博热度做轻微加成，避免冷门重名 tag 抢占
            weight = entry["post_count"] / (entry["post_count"] + 1_000_000)
            add(entry, best_score + weight, best_source)

    matrix = _CACHE["matrix"]
    query_vec = None
    if use_vector and ((matrix is not None and dim) or _has_vectors(tags)):
        if os.environ.get("NAI_EMBEDDING_URL"):
            query_vec = _query_vector(query)
    if query_vec is not None:
        if matrix is not None and _np is not None:
            # 矩阵-向量乘：一次算完所有行（float32 ≈ 66M FLOPs，毫秒级）
            vec = _np.asarray(query_vec, dtype=_np.float32)
            if vec.shape[0] == matrix.shape[1]:
                norms = _CACHE["norms"]
                sims = (matrix @ vec) / norms
                hits = _np.nonzero(sims > 0.3)[0]
                for offset in hits.tolist():
                    entry = tags[offset]
                    similarity = float(sims[offset])
                    existing = scored.get(entry["tag"])
                    if existing is None or similarity > existing[0]:
                        add(entry, similarity, "向量检索")
        else:
            _cosine_fallback(query_vec, tags, scored, add)
    results = sorted(scored.values(), key=lambda item: -item[0])[:limit]
    out = []
    for score, entry, source in results:
        out.append(
            {
                "tag": entry["tag"],
                "category": CATEGORY_NAMES.get(entry["category"], str(entry["category"])),
                "zh_cn": entry["zh"],
                "aliases": entry["aliases"],
                "score": round(score, 4),
                "match": source,
                "post_count": entry["post_count"],
            }
        )
    return out
