"""生成向量检索库 tag_vectors.db（tag/中文/别名/可选嵌入向量）。

tag_catalog 来源：
- 默认从 Launcher 仓库 Aaalice233/Aaalice_NAI_Launcher 自动获取：读取 main 分支
  assets/databases/manifest.json（普通 git 文件）拿到官方发布 release 的
  tag_catalog.db 与锁定 SHA-256/大小，下载后校验，缓存在 tool/.tmp/。
- `--source` 可手动指定本地 tag_catalog.db 路径跳过网络。

翻译来源（可选叠加，优先级与 Launcher 一致：override > ffdkj）：
- catalog 自带的 zh_translations（别名、热度与 override/missing 中文修订）
- `--with-ffdkj`：ffdkj-Danbooru_Tag-Chinese-English-Translation-Table 的
  tag.sqlite（32 万条全量中文）。Launcher 在 lib/core/autocomplete/
  zh_dictionary_download.dart 锁定来源提交与 SHA-256，此处默认校验同一哈希。

用法（在 nai_mcp 目录执行）：

  # tag_catalog 自动下载（首次约 47 MB），只用少量中文
  .venv/Scripts/python tool/build_tag_vector_db.py --db tag_vectors.db

  # 叠加 ffdkj 全量中文（推荐；文件不存在时脚本会先下载并校验 SHA-256）
  .venv/Scripts/python tool/build_tag_vector_db.py --db tag_vectors.db --with-ffdkj

vector 初始为 NULL，由 tool/fill_tag_vectors.py 按配置的 embeddings 端点填充；
填充前词面检索，填充后 search_tag 自动启用语义检索。
"""

import argparse
import hashlib
import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

# 所有默认路径锚定到项目根，不随执行 cwd 漂移
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Launcher 仓库 main 分支的产物清单（普通 git 文件，含 tag_catalog.db 的
# 官方 release 下载地址与锁定 SHA-256/大小，与 Launcher
# tool/tag_catalog/verify_bundled_databases.dart 同源校验）
CATALOG_MANIFEST_URL = (
    "https://raw.githubusercontent.com/Aaalice233/Aaalice_NAI_Launcher/main/"
    "assets/databases/manifest.json"
)
CATALOG_CACHE = Path(PROJECT_ROOT / "tool" / ".tmp" / "tag_catalog.db")

# 与 Launcher zh_dictionary_download.dart 的 ffdkjPinnedSource 一致
FFDKJ_SHA256 = "cd7aa3fa9a2b6153b04a1bce88f98ab3381c20bdabb6348b5fbf1005dad00870"
FFDKJ_SIZE = 22437888
FFDKJ_URL = (
    "https://raw.githubusercontent.com/ffdkj/"
    "ffdkj-Danbooru_Tag-Chinese-English-Translation-Table/"
    "3049e201002a64702873d7f9f3cc7cb8c45ea325/tag.sqlite"
)


def ensure_ffdkj(path: Path) -> Path:
    """返回本地 ffdkj tag.sqlite；不存在或哈希不符时下载锁定版本。"""
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == FFDKJ_SHA256 and path.stat().st_size == FFDKJ_SIZE:
            return path
        print("已有 tag.sqlite 哈希不匹配，重新下载锁定版本…", file=sys.stderr)
        path.unlink()
    print(f"下载 ffdkj tag.sqlite（锁定版本，{FFDKJ_SIZE} 字节）…", file=sys.stderr)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".downloading")
    try:
        urllib.request.urlretrieve(FFDKJ_URL, tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"ffdkj tag.sqlite 下载失败：{exc}") from exc
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != FFDKJ_SHA256:
        tmp.unlink()
        raise SystemExit(f"ffdkj tag.sqlite SHA-256 校验失败：{digest}")
    tmp.rename(path)
    return path


def ensure_catalog(path: Path | None) -> tuple[Path, str]:
    """返回本地 tag_catalog.db 与来源描述。

    path 为空时走 GitHub 自动获取：读 main 分支 manifest.json 锁定的
    release 资产（避开仓库内该文件的 LFS pointer），下载后按清单校验
    SHA-256 与大小，缓存于 tool/.tmp/tag_catalog.db。清单拉取失败但本地
    缓存存在时继续用缓存（离线可用）。
    """
    if path is not None:
        return path, "本地文件"

    try:
        with urllib.request.urlopen(CATALOG_MANIFEST_URL, timeout=30) as resp:
            manifest = json.loads(resp.read().decode("utf-8"))
        entry = manifest["databases"]["tag_catalog.db"]
        want_sha = str(entry["sha256"]).lower()
        want_size = int(entry["size"])
        url = entry["release"]["url"]
    except Exception as exc:
        if CATALOG_CACHE.exists():
            print(
                f"manifest 拉取失败（{exc}），复用本地缓存副本",
                file=sys.stderr,
            )
            return CATALOG_CACHE, "缓存副本（manifest 拉取失败）"
        raise SystemExit(
            f"manifest 拉取失败且无本地缓存：{exc}（或用 --source 指定本地 tag_catalog.db）"
        ) from exc

    if CATALOG_CACHE.exists():
        digest = hashlib.sha256(CATALOG_CACHE.read_bytes()).hexdigest()
        if digest == want_sha and CATALOG_CACHE.stat().st_size == want_size:
            return CATALOG_CACHE, "缓存副本（SHA-256 已校验）"
        print("已缓存 tag_catalog.db 与 manifest 不匹配，重新下载…", file=sys.stderr)
        CATALOG_CACHE.unlink()

    print(f"下载 Launcher tag_catalog.db（{want_size} 字节）…", file=sys.stderr)
    CATALOG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CATALOG_CACHE.with_suffix(".downloading")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"tag_catalog.db 下载失败：{exc}") from exc
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != want_sha or tmp.stat().st_size != want_size:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"tag_catalog.db 校验失败：SHA-256={digest}（期望 {want_sha}）"
        )
    tmp.rename(CATALOG_CACHE)
    return CATALOG_CACHE, "Launcher release（SHA-256 已校验）"


def load_ffdkj_zh(path: Path) -> dict[str, str]:
    con = sqlite3.connect(str(path))
    try:
        columns = [row[1] for row in con.execute("PRAGMA table_info(tags)")]
        required = {"name", "category", "cn_name", "post_count"}
        if not required.issubset(set(columns)):
            raise SystemExit(f"ffdkj tags schema 不受支持：{columns}")
        rows = con.execute(
            "SELECT name, cn_name FROM tags "
            "WHERE cn_name IS NOT NULL AND TRIM(cn_name) <> ''"
        )
        return {name: cn.strip() for name, cn in rows}
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=None,
        help="本地 tag_catalog.db 路径（默认自动从 Launcher GitHub 下载并校验）",
    )
    parser.add_argument(
        "--with-ffdkj",
        action="store_true",
        help="用 ffdkj tag.sqlite 叠加全量中文（override 优先）",
    )
    parser.add_argument(
        "--zh-source",
        default=str(PROJECT_ROOT / "tool" / ".tmp" / "ffdkj_tag.sqlite"),
        help="ffdkj tag.sqlite 本地路径（--with-ffdkj 时默认在此自动下载）",
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "tag_vectors.db"),
        help="输出库路径（默认项目根的 tag_vectors.db）",
    )
    parser.add_argument(
        "--min-post-count",
        type=int,
        default=0,
        help="按 post_count 过滤，减小后续填充向量的工作量",
    )
    args = parser.parse_args()

    source, source_label = ensure_catalog(
        Path(args.source) if args.source else None
    )
    if source.open("rb").read(15)[:15] != b"SQLite format 3":
        print(f"错误：{source} 不是真实 SQLite 数据库（可能还是 LFS pointer）", file=sys.stderr)
        return 1

    zh_map: dict[str, str] = {}
    zh_source_label = ""
    if args.with_ffdkj:
        ffdkj_path = ensure_ffdkj(Path(args.zh_source))
        zh_map = load_ffdkj_zh(ffdkj_path)
        zh_source_label = f"ffdkj tag.sqlite（{len(zh_map)} 条，SHA-256 已校验）"

    src = sqlite3.connect(str(source))
    src.row_factory = sqlite3.Row
    rows = src.execute(
        """
        SELECT t.id, t.name AS tag, t.category, t.post_count,
               z.zh_cn,
               (SELECT GROUP_CONCAT(a.alias, '|') FROM aliases a WHERE a.tag_id = t.id) AS aliases
        FROM tags t
        LEFT JOIN zh_translations z ON z.tag = t.name
        WHERE t.post_count >= ?
        ORDER BY t.post_count DESC
        """,
        (args.min_post_count,),
    ).fetchall()
    src.close()

    # Launcher 预处理顺序：catalog 中文（override/missing 手工修订）优先于 ffdkj
    applied_ffdkj = 0
    output_rows = []
    for r in rows:
        zh_cn = r["zh_cn"]
        if not zh_cn and zh_map:
            candidate = zh_map.get(r["tag"])
            if candidate:
                zh_cn = candidate
                applied_ffdkj += 1
        output_rows.append((r["id"], r["tag"], r["category"], r["post_count"], zh_cn, r["aliases"]))

    out_path = Path(args.db)
    out_path.unlink(missing_ok=True)
    dst = sqlite3.connect(str(out_path))
    dst.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE tags_vec(
          id INTEGER PRIMARY KEY,
          tag TEXT NOT NULL,
          category INTEGER NOT NULL,
          post_count INTEGER NOT NULL DEFAULT 0,
          zh_cn TEXT,
          aliases TEXT,
          vector BLOB
        );
        CREATE INDEX idx_tags_vec_post ON tags_vec(post_count DESC);
        CREATE INDEX idx_tags_vec_zh ON tags_vec(zh_cn COLLATE NOCASE);
        """
    )
    dst.executemany(
        "INSERT INTO tags_vec(id, tag, category, post_count, zh_cn, aliases, vector) "
        "VALUES (?, ?, ?, ?, ?, ?, NULL)",
        output_rows,
    )
    dst.executemany(
        "INSERT INTO meta(key, value) VALUES (?, ?)",
        (
            ("source", str(source.resolve())),
            ("catalog_source", source_label),
            ("zh_source", zh_source_label),
            ("built_at", time.strftime("%Y-%m-%d %H:%M:%S")),
            ("count", str(len(output_rows))),
            ("filled", "0"),
            ("model", ""),
            ("dim", "0"),
        ),
    )
    dst.commit()

    zh_count = sum(1 for row in output_rows if row[4])
    alias_count = sum(1 for row in output_rows if row[5])
    print(
        f"已生成 {out_path}：{len(output_rows)} tags（含中文 {zh_count}、"
        f"含别名 {alias_count}{('，其中来自 ffdkj ' + str(applied_ffdkj)) if args.with_ffdkj else ''}）；"
        f"catalog 来源：{source_label}"
    )
    if args.with_ffdkj:
        print("下一步：tool/fill_tag_vectors.py 填充 vector 后启用语义检索。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
