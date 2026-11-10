"""Inpaint 蒙版构造（纯 stdlib，不引入图像库）。

蒙版语义与官网一致：白色 = 重绘区域，黑色 = 保留。
- `mask_path`：解码现成蒙版 PNG（8-bit RGB/RGBA/灰度，非隔行），
  按亮度 ≥128 二值化后缩放到工作流尺寸。
- `rects`：0-1 归一矩形 [x, y, w, h]（左上原点），直接渲染出蒙版 PNG。
"""

import struct
import zlib


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(width: int, height: int, rgb_rows: bytes) -> bytes:
    """打包 8-bit RGB PNG（行 filter 0）。rgb_rows 为连续 RGB 像素。"""
    stride = width * 3
    raw = bytearray()
    for row in range(height):
        raw.append(0)  # filter type 0
        raw += rgb_rows[row * stride : (row + 1) * stride]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _chunk(b"IEND", b"")
    )


def _paeth(a: int, b: int, c: int) -> int:
    """PNG filter 4 的标准 Paeth 预测器：取最近邻（左/上/左上）中预测误差最小者。"""
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def decode_png_rgb(png_bytes: bytes) -> tuple[int, int, bytes]:
    """解码非隔行 8-bit PNG（RGB/RGBA/灰度/灰度+alpha）为 RGB。
    返回 (width, height, rgb_rows)。"""
    if png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("蒙版必须是 PNG 文件")
    width = height = bit_depth = color_type = None
    interlace = 0
    idat = bytearray()
    position = 8
    while position + 8 <= len(png_bytes):
        length, tag = struct.unpack(">I4s", png_bytes[position : position + 8])
        data = png_bytes[position + 8 : position + 8 + length]
        position += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", data
            )
        elif tag == b"IDAT":
            idat += data
        elif tag == b"IEND":
            break
    if width is None or not idat:
        raise ValueError("蒙版 PNG 缺少 IHDR/IDAT")
    if bit_depth != 8 or interlace != 0:
        raise ValueError("暂不支持该蒙版 PNG 编码（需要 8-bit 非隔行）")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ValueError(f"暂不支持的蒙版 PNG 颜色类型 {color_type}")
    palette: bytes | None = None
    if color_type == 3:
        # 调色板：解析 PLTE chunk
        position = 8
        while position + 8 <= len(png_bytes):
            length, tag = struct.unpack(">I4s", png_bytes[position : position + 8])
            data = png_bytes[position + 8 : position + 8 + length]
            position += 12 + length
            if tag == b"PLTE":
                palette = data
                break
            if tag == b"IDAT":
                break
        if not palette or len(palette) % 3 != 0:
            raise ValueError("调色板 PNG 缺少 PLTE 数据")

    stride = width * channels
    raw = bytearray(zlib.decompress(bytes(idat)))
    if len(raw) != (stride + 1) * height:
        raise ValueError("蒙版 PNG 解压尺寸不匹配")
    pixels = bytearray(width * height * channels)
    previous = bytearray(stride)
    for row in range(height):
        offset = row * (stride + 1)
        filter_type = raw[offset]
        line = bytearray(raw[offset + 1 : offset + 1 + stride])
        if filter_type == 1:  # Sub
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif filter_type == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + previous[i]) & 0xFF
        elif filter_type == 3:  # Average
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 0xFF
        elif filter_type == 4:  # Paeth
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                up_left = previous[i - channels] if i >= channels else 0
                line[i] = (line[i] + _paeth(left, previous[i], up_left)) & 0xFF
        elif filter_type != 0:
            raise ValueError(f"蒙版 PNG 压缩 filter {filter_type} 不支持")
        pixels[row * stride : (row + 1) * stride] = line
        previous = line

    if channels == 3:
        return width, height, bytes(pixels)
    rgb = bytearray(width * height * 3)
    if palette is not None:
        # 调色板：索引 → RGB
        for index in range(width * height):
            entry = pixels[index] * 3
            rgb[index * 3 : index * 3 + 3] = palette[entry : entry + 3]
        return width, height, bytes(rgb)
    if color_type == 6:  # RGBA：直接取 RGB 三个通道
        for index in range(width * height):
            rgb[index * 3 : index * 3 + 3] = pixels[index * 4 : index * 4 + 3]
        return width, height, bytes(rgb)
    # 灰度(0)与灰度+alpha(4)：单通道复制为 RGB
    for index in range(width * height):
        value = pixels[index * channels]
        rgb[index * 3 : index * 3 + 3] = bytes((value, value, value))
    return width, height, bytes(rgb)


def _resize_nearest(rgb_rows: bytes, src_w: int, src_h: int, width: int, height: int) -> bytes:
    """最近邻缩放 RGB 像素（16.16 定点缩放比；蒙版/合成时尺寸对齐用）。"""
    out = bytearray(width * height * 3)
    x_ratio = ((src_w << 16) // width) or 1
    y_ratio = ((src_h << 16) // height) or 1
    stride = src_w * 3
    for y in range(height):
        src_y = min(src_h - 1, (y * y_ratio) >> 16)
        row_base = src_y * stride
        dst_base = y * width * 3
        for x in range(width):
            src_x = min(src_w - 1, (x * x_ratio) >> 16)
            src_index = row_base + src_x * 3
            dst_index = dst_base + x * 3
            out[dst_index : dst_index + 3] = rgb_rows[src_index : src_index + 3]
    return bytes(out)


def render_rects_mask(rects: list, width: int, height: int) -> bytes:
    """按 0-1 归一矩形 [x, y, w, h] 渲染蒙版：白 = 重绘区。"""
    canvas = bytearray(width * height * 3)  # 全黑
    for index, rect in enumerate(rects):
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            raise ValueError(f"rects[{index}] 需为 [x, y, w, h]")
        x0, y0, w, h = (float(v) for v in rect)
        if not all(0.0 <= v <= 1.0 for v in (x0, y0, w, h)):
            raise ValueError(
                f"rects[{index}]={rect} 越界：坐标与宽高需在 0-1（左上原点归一值）"
            )
        if w <= 0 or h <= 0:
            raise ValueError(f"rects[{index}] 宽高必须为正")
        left = min(width, max(0, round(x0 * width)))
        top = min(height, max(0, round(y0 * height)))
        right = min(width, max(left, round((x0 + w) * width)))
        bottom = min(height, max(top, round((y0 + h) * height)))
        for y in range(top, bottom):
            row_base = y * width * 3
            for x in range(left, right):
                pixel = row_base + x * 3
                canvas[pixel : pixel + 3] = b"\xff\xff\xff"
    return write_png(width, height, bytes(canvas))


def _box_blur_gray(values: bytes, width: int, height: int, radius: int) -> bytes:
    """灰度蒙版的分离式 box blur（官网 infill worker 用半径 20 羽化）。"""
    radius = max(1, int(radius))
    # 行方向
    temp = bytearray(width * height)
    for y in range(height):
        row = values[y * width : (y + 1) * width]
        prefix = [0] * (width + 1)
        for x in range(width):
            prefix[x + 1] = prefix[x] + row[x]
        base = y * width
        for x in range(width):
            lo, hi = max(0, x - radius), min(width, x + radius + 1)
            temp[base + x] = (prefix[hi] - prefix[lo]) // (hi - lo)
    # 列方向
    out = bytearray(width * height)
    for x in range(width):
        col = [temp[y * width + x] for y in range(height)]
        prefix = [0] * (height + 1)
        for y in range(height):
            prefix[y + 1] = prefix[y] + col[y]
        for y in range(height):
            lo, hi = max(0, y - radius), min(height, y + radius + 1)
            out[y * width + x] = (prefix[hi] - prefix[lo]) // (hi - lo)
    return bytes(out)


def _latent_quantize(binary: bytes, width: int, height: int, grid: int = 8) -> bytes:
    """官网 infill 语义：二值蒙版先按 latent 网格（8px）采样并阈值化，
    再最近邻放大回请求尺寸——发给模型的 mask 边缘就是这种 8px 量化硬边。"""
    latent_w = max(1, width // grid)
    latent_h = max(1, height // grid)
    sampled = bytearray(latent_w * latent_h)
    x_ratio = ((width << 16) // latent_w) or 1
    y_ratio = ((height << 16) // latent_h) or 1
    for ly in range(latent_h):
        src_y = min(height - 1, (ly * y_ratio) >> 16)
        for lx in range(latent_w):
            src_x = min(width - 1, (lx * x_ratio) >> 16)
            sampled[ly * latent_w + lx] = binary[src_y * width + src_x]
    quantized = bytearray(width * height)
    grid_ratio_x = ((latent_w << 16) // width) or 1
    grid_ratio_y = ((latent_h << 16) // height) or 1
    for y in range(height):
        src_y = min(latent_h - 1, (y * grid_ratio_y) >> 16)
        for x in range(width):
            src_x = min(latent_w - 1, (x * grid_ratio_x) >> 16)
            quantized[y * width + x] = 255 if sampled[src_y * latent_w + src_x] else 0
    return bytes(quantized)


def _binary_like_rgb(values: bytes) -> bytes:
    """灰度字节流展开为黑/白 RGB 三通道（请求 mask PNG 按三通道编码）。"""
    return bytes(v for value in values for v in (value, value, value))


def prepare_mask(
    mask_path: str | None,
    rects: list | None,
    width: int,
    height: int,
) -> tuple[bytes, bytes] | None:
    """按工作流尺寸产出 (请求用硬边缘蒙版 PNG, 合成用软蒙版灰度值)。

    对齐官网前端 infill worker：二值蒙版（白=重绘）→ 8px latent 网格量化 →
    - 请求 `mask`：量化后的硬边缘蒙版（模型不见羽化边，否则会画出矩形框）
    - 合成蒙版：同一量化蒙版 box blur 半径 20 × 2 次（仅生成后本地合成用）
    空蒙版直接报错（官方 EmptyInpaintMaskException 语义）。"""
    binary: bytes
    if mask_path:
        src_w, src_h, rgb = decode_png_rgb(open(mask_path, "rb").read())
        binarized = bytearray()
        for i in range(0, len(rgb), 3):
            binarized.append(
                255
                if (rgb[i] * 299 + rgb[i + 1] * 587 + rgb[i + 2] * 114) // 1000
                >= 128
                else 0
            )
        binary = _resize_nearest(bytes(binarized), src_w, src_h, width, height)
    elif rects:
        src_w, src_h, rgb = decode_png_rgb(render_rects_mask(rects, width, height))
        binary = bytes(rgb[::3])  # 渲染图只有纯黑/纯白
    else:
        return None
    quantized = _latent_quantize(binary, width, height)
    if not any(quantized):
        raise ValueError("蒙版区域为空：rects/蒙版文件没有覆盖任何像素")
    request_mask = write_png(width, height, _binary_like_rgb(quantized))
    soft = _box_blur_gray(quantized, width, height, 20)
    soft = _box_blur_gray(soft, width, height, 20)
    return request_mask, soft


def composite(
    source_bytes: bytes, generated_bytes: bytes, soft: bytes, width: int, height: int
) -> bytes:
    """官网局部重绘后处理：`原图×(1-m) + 生成×m`，m 为请求尺寸软蒙版。"""
    src_w, src_h, src = decode_png_rgb(source_bytes)
    if (src_w, src_h) != (width, height):
        src = _resize_nearest(src, src_w, src_h, width, height)
    gen_w, gen_h, gen = decode_png_rgb(generated_bytes)
    if (gen_w, gen_h) != (width, height):
        gen = _resize_nearest(gen, gen_w, gen_h, width, height)
    blended = bytearray(width * height * 3)
    index = 0
    for i in range(width * height):
        weight = soft[i]
        if weight == 0:
            blended[index : index + 3] = src[index : index + 3]
        elif weight == 255:
            blended[index : index + 3] = gen[index : index + 3]
        else:
            for c in range(3):
                blended[index + c] = (
                    src[index + c] * (255 - weight) + gen[index + c] * weight + 127
                ) // 255
        index += 3
    return write_png(width, height, bytes(blended))
