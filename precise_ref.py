"""精准参考图规范化（Launcher nai_api_utils.ensurePngFormat 同口径）。

官网 director_reference_images 只接受规格化几何：按最接近宽高比选
1024×1536 / 1536×1024 / 1472×1472 之一，等比缩放后居中贴到黑底目标
画布再编码 PNG。发送任意尺寸原图会被服务端 HTTP 500。
与 Launcher 的差异：缩放插值用 bilinear（numpy）而非 cubic。
"""

import numpy as np

_TARGETS = ((1024, 1536), (1536, 1024), (1472, 1472))


def _bilinear(src: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    h, w, _ = src.shape
    ys = np.linspace(0, h - 1, new_h)
    xs = np.linspace(0, w - 1, new_w)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    a = src[y0][:, x0]
    b = src[y0][:, x1]
    c = src[y1][:, x0]
    d = src[y1][:, x1]
    out = (
        a * (1 - wy) * (1 - wx)
        + b * (1 - wy) * wx
        + c * wy * (1 - wx)
        + d * wy * wx
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize(png_bytes: bytes) -> bytes:
    """按官方三档规格化参考图黑底画布，返回规范化后的 PNG 字节。"""
    # 延迟导入避免模块环（nai_mask 不依赖本模块）
    from nai_mask import decode_png_rgb, write_png

    width, height, rows = decode_png_rgb(png_bytes)
    src = np.frombuffer(rows, np.uint8).reshape(height, width, 3)
    aspect = width / height
    target_w, target_h = min(
        _TARGETS, key=lambda t: abs(t[0] / t[1] - aspect)
    )
    if aspect > target_w / target_h:
        new_w = target_w
        new_h = max(1, round(target_w / aspect))
    else:
        new_h = target_h
        new_w = max(1, round(target_h * aspect))
    resized = src if (new_w, new_h) == (width, height) else _bilinear(src, new_w, new_h)
    canvas = np.zeros((target_h, target_w, 3), np.uint8)  # 黑底
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return write_png(target_w, target_h, canvas.tobytes())
