"""stdio 冒烟 + 离线断言：握手、工具列表、工作流流程、请求构造与 tag 检索。"""

import io
import json
import os
import sqlite3
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import shutil

ROOT = Path(__file__).resolve().parent
TMP = ROOT / "tool" / ".tmp_smoke"

ENV = dict(os.environ)
ENV["NAI_MCP_WORKFLOW_DIR"] = str(TMP / "workflows")
ENV["NAI_TAG_DB"] = str(TMP / "tags.db")
ENV["NAI_OUTPUT_DIR"] = str(TMP / "images")
ENV.pop("NAI_TOKEN", None)


def make_png(width=8, height=8) -> bytes:
    import zlib

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data))
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def make_tag_db() -> None:
    DB = TMP / "tags.db"
    DB.unlink(missing_ok=True)
    con = sqlite3.connect(str(DB))
    con.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE tags_vec(
          id INTEGER PRIMARY KEY, tag TEXT NOT NULL, category INTEGER NOT NULL,
          post_count INTEGER NOT NULL DEFAULT 0, zh_cn TEXT, aliases TEXT, vector BLOB
        );
        """
    )
    con.executemany(
        "INSERT INTO tags_vec VALUES (?,?,?,?,?,?,NULL)",
        [
            (1, "looking at viewer", 0, 999_999_999, "面向观众", "ltd|looking_at_viewer"),
            (2, "silver hair", 0, 500_000, "银发", None),
            (3, "1girl", 0, 10_000_000, "一名少女", "girl"),
        ],
    )
    con.commit()
    con.close()


def main() -> None:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    make_tag_db()

    proc = subprocess.Popen(
        [sys.executable, "server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=open(TMP / "stderr.txt", "wb"),
        text=True,
        encoding="utf-8",
        env=ENV,
        cwd=str(ROOT),
    )

    def send(payload):
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        return json.loads(proc.stdout.readline())

    resp = send(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "smoke", "version": "0"},
            },
        }
    )
    print("initialize ->", resp["result"]["serverInfo"])
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    proc.stdin.flush()

    expected = {
        "set_prompt", "set_gen_param", "set_characters", "set_image",
        "set_mask", "set_precise_reference", "create_workflow",
        "delete_workflow", "delete_all_workflows", "clear_saved_images",
        "read_workflow", "estimate_anlas", "list_workflows",
        "list_gen_options", "generate_image", "enhance_image",
        "upscale_image", "search_tag",
    }
    resp = send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = [t["name"] for t in resp["result"]["tools"]]
    print("tools ->", names)
    assert set(names) == expected, names

    def call(cid, tool, args):
        result = send(
            {
                "jsonrpc": "2.0",
                "id": cid,
                "method": "tools/call",
                "params": {"name": tool, "arguments": args},
            }
        )["result"]
        content = result.get("content") or []
        if content:
            return content[0]["text"]
        # 数组返回值（如 list_workflows）无 text content，走结构化字段
        return json.dumps(
            result.get("structuredContent", result), ensure_ascii=False, indent=2
        )

    # generate 缺工作流
    out = call(3, "generate_image", {"name": "nope"})
    assert "不存在" in out, out
    # generate 有工作流但缺 prompt
    out = call(4, "set_gen_param", {"name": "w1", "steps": 30})
    out = call(5, "generate_image", {"name": "w1"})
    assert "prompt" in out, out
    # 完成 prompt 后 generate 走到 token 检查
    out = call(6, "set_prompt", {"name": "w1", "prompt": "1girl, silver hair"})
    assert "w1" in out
    out = call(7, "generate_image", {"name": "w1"})
    assert "NAI_TOKEN" in out, out
    # read / estimate
    out = call(8, "read_workflow", {"name": "w1"})
    assert "nai-diffusion-5-full" in out and '"steps": 30' in out, out
    assert "预计 Anlas" in out, out
    out = call(9, "estimate_anlas", {"name": "w1"})
    assert "预计 Anlas：32" in out, out  # 默认模型 V5-full：832×1216@30 步 = ceil(20.43)*1.5 = 32
    # 角色
    out = call(10, "set_characters", {"name": "w1", "characters": [{"prompt": "2girls"}]})
    assert "已设置 1 个角色" in out, out
    # img2img
    sample = TMP / "sample.png"
    sample.write_bytes(make_png())
    out = call(11, "set_image", {"name": "w1", "image_path": str(sample)})
    assert "已设置底图" in out, out
    out = call(12, "read_workflow", {"name": "w1"})
    assert "sample.png" in out and "0.7" in out, out
    # 底图清掉继续
    out = call(13, "set_image", {"name": "w1", "image_path": "clear"})
    assert "清除" in out, out
    # tag 检索
    out = call(14, "search_tag", {"query": "面向观众"})
    print("tag ->", out[:200])
    assert "looking at viewer" in out, out
    out = call(15, "search_tag", {"query": "looking at viewer"})
    assert "looking at viewer" in out, out
    # upscale 缺文件
    out = call(16, "upscale_image", {"image_path": str(TMP / "missing.png")})
    assert "图片不存在" in out, out
    # enhance 会走网络（无 token），期望因缺 token 报错而不是参数错误
    # enhance 无底图
    out = call(17, "enhance_image", {"name": "w1"})
    assert "还没有底图" in out, out
    # 恢复底图后 enhance 走到网络（无 token），期望缺 token 报错
    out = call(18, "set_image", {"name": "w1", "image_path": str(sample)})
    assert "已设置底图" in out, out
    out = call(19, "enhance_image", {"name": "w1"})
    assert "NAI_TOKEN" in out, out
    # inpaint 蒙版
    out = call(20, "set_mask", {"name": "w1"})
    assert "mask_path" in out and "rects" in out, out
    # create_workflow：新建 / 重复创建报错
    out = call(20, "create_workflow", {"name": "w_new"})
    assert "已创建" in out, out
    out = call(20, "create_workflow", {"name": "w_new"})
    assert "已存在" in out, out
    out = call(20, "create_workflow", {"name": "w_model", "model": "nai-diffusion-3"})
    assert "已创建" in out and "采样预设" in out, out
    out = call(21, "set_mask", {"name": "w1", "rects": [[0.1, 0.1, 0.5, 0.4]]})
    assert "已设置 inpaint 蒙版" in out and "矩形 1 个" in out, out
    out = call(22, "read_workflow", {"name": "w1"})
    assert '"rects"' in out and "strength" in out, out
    out = call(23, "enhance_image", {"name": "w1"})
    assert "蒙版" in out, out  # 蒙版与增强互斥提示
    # 带蒙版生成走网络前需 token（离线安全断言 infill 路径已通到传输层）
    out = call(24, "generate_image", {"name": "w1"})
    assert "NAI_TOKEN" in out, out
    out = call(25, "set_mask", {"name": "w1", "mask_path": "clear"})
    assert "清除" in out, out
    # V5 配额用量查看：离线（无 token）时估价不附带配额行
    out = call(26, "create_workflow", {"name": "w_v5", "model": "nai-diffusion-5-full"})
    assert "已创建" in out, out
    out = call(27, "set_prompt", {"name": "w_v5", "prompt": "1girl"})
    assert "w_v5" in out, out
    out = call(28, "estimate_anlas", {"name": "w_v5"})
    assert "预计 Anlas" in out, out  # 1024×1024@28 步
    assert "V5 配额" not in out, out
    # 纯格式化断言（anlas.v5_usage_view，不发请求）
    import anlas

    v5 = "nai-diffusion-5-full"
    area_1mp = 1024 * 1024
    # 非 V5 模型 / 非 Opus / 订阅失败 → 不显示
    assert anlas.v5_usage_view("nai-diffusion-4-5-full", {"percent": 50}, is_opus=True, area=area_1mp) == ""
    assert anlas.v5_usage_view(v5, {"percent": 50}, is_opus=False, area=area_1mp) == ""
    assert anlas.v5_usage_view(v5, {"percent": 50}, is_opus=None, area=area_1mp) == ""
    # Opus 但响应缺 usage 字段 → 明确提示
    out = anlas.v5_usage_view(v5, None, is_opus=True, area=area_1mp)
    assert "usage" in out, out
    # 正常配额：1% ≈ 17.3 张 1MP 口径；>100% 保留真实值
    out = anlas.v5_usage_view(v5, {"percent": 97.3, "isNegative": False}, is_opus=True, area=area_1mp)
    assert "97.3%" in out, out
    assert "1683" in out, out  # round(97.3 * 17.3 / 1)
    assert "回充" in out and "正常价" in out, out
    out = anlas.v5_usage_view(v5, {"percent": 102.0, "isNegative": False}, is_opus=True, area=area_1mp)
    assert "102%" in out, out
    # 大图按面积扣份折算（1216×1664=2.02MP → 3 份/张）
    out = anlas.v5_usage_view(v5, {"percent": 97.3, "isNegative": False}, is_opus=True, area=1216 * 1664)
    assert "3 份/张" in out and "561" in out, out  # round(97.3 * 17.3 / 3)
    # 透支：percent 与张数不再展示
    out = anlas.v5_usage_view(v5, {"percent": -0.5, "isNegative": True}, is_opus=True, area=area_1mp)
    assert "透支" in out and "%" not in out, out

    # 工作流删除 / 全部删除 / 清空已存图片（均离线）
    out = call(29, "delete_workflow", {"name": "w_ghost"})
    assert "不存在" in out, out
    out = call(30, "delete_workflow", {"name": "w_v5"})
    assert "已删除工作流 w_v5" in out, out
    out = call(31, "list_workflows", {})
    assert "w_v5" not in out and "w1" in out, out
    (TMP / "images").mkdir(exist_ok=True)
    (TMP / "images" / "nai_test.png").write_bytes(b"PNG")
    (TMP / "images" / "notes.txt").write_text("keep")
    out = call(32, "clear_saved_images", {})
    assert "已清空 1 个 PNG" in out, out
    assert (TMP / "images" / "notes.txt").exists()  # 非 PNG 不动
    out = call(33, "clear_saved_images", {})
    assert "没有 PNG" in out, out
    out = call(34, "delete_all_workflows", {})
    assert ("w1" in out and "w_model" in out) and "已删除" in out, out
    out = call(35, "list_workflows", {})
    assert "w1" not in out, out

    # 精准参考（V4.5 支持路径 / 非法类型 / 非 PNG / V5 明确报错 / 清空）
    out = call(36, "create_workflow", {"name": "w_ref", "model": "nai-diffusion-4-5-full"})
    assert "已创建" in out, out
    out = call(37, "set_prompt", {"name": "w_ref", "prompt": "1girl"})
    assert "w_ref" in out, out
    ref_png = TMP / "ref.png"
    ref_png.write_bytes(make_png())
    out = call(38, "set_precise_reference", {"name": "w_ref", "image_path": str(ref_png), "type": "character"})
    assert "已新增精准参考" in out, out
    out = call(39, "read_workflow", {"name": "w_ref"})
    assert '"precise_refs"' in out and '"type": "character"' in out, out
    out = call(40, "generate_image", {"name": "w_ref"})
    assert "NAI_TOKEN" in out, out  # 请求构建通过，网络前才失败
    out = call(41, "set_precise_reference", {"name": "w_ref", "image_path": str(ref_png), "type": "wrong"})
    assert "type 无效" in out, out
    bad_png = TMP / "ref.bad"
    bad_png.write_bytes(b"JPEG")
    out = call(42, "set_precise_reference", {"name": "w_ref", "image_path": str(bad_png)})
    assert "需为 PNG" in out, out
    out = call(43, "set_precise_reference", {"name": "w_ref", "image_path": "clear"})
    assert "已清空" in out, out
    out = call(44, "create_workflow", {"name": "w_ref5", "model": "nai-diffusion-5-full"})
    assert "已创建" in out, out
    out = call(45, "set_prompt", {"name": "w_ref5", "prompt": "1girl"})
    assert "w_ref5" in out, out
    out = call(46, "set_precise_reference", {"name": "w_ref5", "image_path": str(ref_png)})
    assert "已新增精准参考" in out, out
    out = call(47, "generate_image", {"name": "w_ref5"})
    assert "precise reference" in out, out  # V5 生成前明确报错

    # 精准参考图规范化：官方只认三档黑底画布几何（否则 HTTP 500）
    from nai_mask import decode_png_rgb, write_png

    import precise_ref

    norm = precise_ref.normalize(make_png(8, 4))  # aspect 2.0 → 横档
    nw, nh, _ = decode_png_rgb(norm)
    assert (nw, nh) == (1536, 1024), (nw, nh)
    norm = precise_ref.normalize(make_png(4, 8))  # aspect 0.5 → 竖档
    nw, nh, _ = decode_png_rgb(norm)
    assert (nw, nh) == (1024, 1536), (nw, nh)

    proc.kill()
    print("smoke test passed")


if __name__ == "__main__":
    main()
