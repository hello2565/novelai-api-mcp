"""按环境变量配置的 OpenAI 兼容 embeddings 接口填充 tag_vectors.db 的 vector 列。

性能设计：
- 按 token 预算与单批条目数双上限打包请求（NAI_EMBED_BATCH_TOKENS /
  NAI_EMBED_MAX_BATCH_ROWS，默认 24000 / 256，适配 32k 输入模型）。
- 并发请求 + 429/5xx 指数退避重试；响应带 index 时按 index 回填。
- 批次起点即打印日志，避免大批次长时间无输出；启动前先用 1 条样本做
  预热探针，端点不可达/超慢立刻暴露。
- Ctrl+C：取消未开始批次并立即退出（os._exit），不再等待卡在 socket 里的
  worker 线程。

用法（在 nai_mcp 目录执行）：

  .venv/Scripts/python tool/fill_tag_vectors.py                      # 全量
  .venv/Scripts/python tool/fill_tag_vectors.py --min-post-count 1000  # 只填热门

要求环境变量：
  NAI_EMBEDDING_URL       embeddings 端点（POST {"input":[...], "model":...}
                          → {"data":[{"embedding":[...]}]}）
  NAI_EMBEDDING_MODEL     模型名（可选，取决于端点）
  NAI_EMBEDDING_API_KEY   Bearer 密钥（可选）

server 的 search_tag 运行时也用同一组变量编码查询词——两端的模型必须一致。
"""

import argparse
import json
import os
import queue
import sqlite3
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

# tag_vectors.db 默认定位在项目根，不随执行 cwd 漂移
PROJECT_ROOT = Path(__file__).resolve().parent.parent

STOP = threading.Event()


def _embed_inputs(
    inputs: list[str], url: str, headers: dict, model: str, timeout: float
) -> list[list[float]]:
    payload = {"input": inputs}
    if model:
        payload["model"] = model
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(4):
        if STOP.is_set():
            raise RuntimeError("已中止")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items = data["data"]
            if len(items) != len(inputs):
                raise RuntimeError(
                    f"端点返回数量不匹配：期望 {len(inputs)}，得到 {len(items)}"
                )
            vectors: list[list[float] | None] = [None] * len(inputs)
            for position, item in enumerate(items):
                index = item.get("index", position)
                vectors[index] = item["embedding"]
            if any(v is None for v in vectors):
                raise RuntimeError("端点返回的 index 不完整")
            return vectors  # type: ignore[return-value]
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                break
            time.sleep(2**attempt * 10)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            break
    raise RuntimeError(f"embeddings 请求失败：{last_error}")


def _build_batches(
    rows: list[tuple[dict, str]],
    batch_tokens: int,
    chars_per_token: float,
    max_rows: int,
) -> list[list[tuple[dict, str]]]:
    """按 token 预算与条目数上限装批，token 估算 = 字符数 / chars_per_token。"""
    batches: list[list[tuple[dict, str]]] = []
    current: list[tuple[dict, str]] = []
    current_tokens = 0
    for row in rows:
        tokens = max(1, int(len(row[1]) / chars_per_token) + 1)
        if current and (
            current_tokens + tokens > batch_tokens
            or len(current) >= max_rows
        ):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(row)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="tag_vectors.db", help="标签库路径")
    parser.add_argument("--min-post-count", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0, help="本次最多填充条数（0=不限）")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="并发请求数（默认取 NAI_EMBED_CONCURRENCY，无则 4）",
    )
    parser.add_argument(
        "--batch-tokens",
        type=int,
        default=0,
        help="单请求 token 预算（默认取 NAI_EMBED_BATCH_TOKENS，无则 24000）",
    )
    parser.add_argument(
        "--max-batch-rows",
        type=int,
        default=0,
        help="单请求条目数上限（默认取 NAI_EMBED_MAX_BATCH_ROWS，无则 256）",
    )
    args = parser.parse_args()

    url = os.environ.get("NAI_EMBEDDING_URL", "").strip()
    if not url:
        print("请先设置 NAI_EMBEDDING_URL（OpenAI 兼容 embeddings 端点）", file=sys.stderr)
        return 1
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    if not db_path.exists():
        print(f"标签库不存在：{db_path}，先运行 tool/build_tag_vector_db.py", file=sys.stderr)
        return 1

    batch_tokens = args.batch_tokens or int(
        os.environ.get("NAI_EMBED_BATCH_TOKENS", "24000")
    )
    max_batch_rows = args.max_batch_rows or int(
        os.environ.get("NAI_EMBED_MAX_BATCH_ROWS", "256")
    )
    concurrency = args.concurrency or max(
        1, int(os.environ.get("NAI_EMBED_CONCURRENCY", "4"))
    )
    chars_per_token = float(os.environ.get("NAI_EMBED_CHARS_PER_TOKEN", "1.0"))
    request_timeout = float(os.environ.get("NAI_EMBED_TIMEOUT", "600"))

    model = os.environ.get("NAI_EMBEDDING_MODEL", "")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("NAI_EMBEDDING_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['NAI_EMBEDDING_API_KEY']}"

    # 预热探针：1 条样本，端点可达性与响应速度立刻可见
    print(f"预热探针 -> {url}", flush=True)
    started = time.monotonic()
    try:
        probe = _embed_inputs(["warmup"], url, headers, model, min(request_timeout, 60))
    except RuntimeError as exc:
        print(f"端点预热失败：{exc}", file=sys.stderr)
        return 1
    print(
        f"端点可达，样本维度 {len(probe[0])}，耗时 {time.monotonic() - started:.1f}s",
        flush=True,
    )

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = [
        (dict(r), None)
        for r in con.execute(
            "SELECT id, tag, zh_cn, aliases FROM tags_vec "
            "WHERE post_count >= ? AND vector IS NULL ORDER BY post_count DESC",
            (args.min_post_count,),
        )
    ]

    def input_of(row: dict) -> str:
        return " | ".join(
            part for part in (row["zh_cn"], row["tag"].replace("_", " "), row["aliases"]) if part
        )

    prepared = [(row, input_of(row)) for row, _ in rows]
    if args.max_rows:
        prepared = prepared[: args.max_rows]
    total = len(prepared)
    batches = _build_batches(prepared, batch_tokens, chars_per_token, max_batch_rows)
    print(
        f"待填充 {total} 条，分 {len(batches)} 批（≤{batch_tokens} token 且 "
        f"≤{max_batch_rows} 行/批），并发 {concurrency}",
        flush=True,
    )

    work: "queue.Queue[tuple[int, list]]" = queue.Queue()
    futures = []
    results: dict[int, list[tuple[int, list[float]]]] = {}
    first_dim = [0]
    lock = threading.Lock()
    done_counter = [0]

    def worker() -> None:
        while not STOP.is_set():
            try:
                batch_index, batch = work.get_nowait()
            except queue.Empty:
                return
            inputs = [item[1] for item in batch]
            started = time.monotonic()
            print(
                f"批 {batch_index + 1}/{len(batches)} 开始：{len(batch)} 行 → "
                f"端点…",
                flush=True,
            )
            try:
                vectors = _embed_inputs(inputs, url, headers, model, request_timeout)
            except RuntimeError as exc:
                print(f"\n批 {batch_index + 1} 失败：{exc}", file=sys.stderr, flush=True)
                work.task_done()
                if not STOP.is_set():
                    STOP.set()  # 一批失败即停，避免整场空转
                return
            elapsed = time.monotonic() - started
            with lock:
                if first_dim[0] == 0 and vectors:
                    first_dim[0] = len(vectors[0])
                results[batch_index] = [
                    (batch[i][0]["id"], vectors[i]) for i in range(len(batch))
                ]
                done_counter[0] += len(batch)
            print(
                f"批 {batch_index + 1}/{len(batches)} 完成："
                f"{len(batch)} 行 / {elapsed:.1f}s（累计 {done_counter[0]}/{total}）",
                flush=True,
            )
            work.task_done()

    for index, batch in enumerate(batches):
        work.put((index, batch))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for _ in range(concurrency):
            futures.append(pool.submit(worker))
        try:
            # 主线程只等未来完成；Ctrl+C在这里被捕获
            while not STOP.is_set():
                done, _pending = wait(futures, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                if all(f.done() for f in futures):
                    break
        except KeyboardInterrupt:
            print("\n收到 Ctrl+C：中止未完成的批次…", flush=True)
            STOP.set()
            print("已退出（进行中的网络请求被放弃）", flush=True)
            os._exit(130)

    if not results:
        print("没有返回结果", file=sys.stderr)
        return 1

    flat = [item for index in sorted(results) for item in results[index]]
    con.executemany(
        "UPDATE tags_vec SET vector = ? WHERE id = ?",
        [(_pack(values), row_id) for row_id, values in flat],
    )
    con.commit()
    dim = first_dim[0]
    con.execute("UPDATE meta SET value = ? WHERE key = 'dim'", (str(dim),))
    con.execute("UPDATE meta SET value = ? WHERE key = 'model'", (model,))
    con.execute(
        "UPDATE meta SET value = ? WHERE key = 'filled'",
        (time.strftime("%Y-%m-%d %H:%M:%S"),),
    )
    con.commit()
    con.close()
    print(f"完成：{len(flat)} 条，维度 {dim}。search_tag 已可启用语义检索。")
    return 0


def _pack(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


if __name__ == "__main__":
    sys.exit(main())
