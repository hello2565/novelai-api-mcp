"""NovelAI 文生图 MCP Server（stdio）。

工作流式出图：set_* 工具维护一个命名工作流（不触发请求、不消耗 Anlas），
read_workflow / estimate_anlas 以工作流名称读取并给出 Anlas 预估供确认，
generate_image / enhance_image / upscale_image 按名称或图片实际出图。
另有 search_tag 检索标准 booru tag（tool/build_tag_vector_db.py 生成词库）。

配置方式：向 MCP 客户端注册本脚本的绝对路径，并在 env 中提供
`NAI_TOKEN`（持久令牌，形态如 `pst-...`）。
"""

import base64
import json
import os
import time
from mcp.server.fastmcp import FastMCP, Image

import anlas
import nai_request
import tagsearch
import workflow

mcp = FastMCP(
    "nai-mcp",
    instructions=(
        "默认模型为 nai-diffusion-5-full（新建/未显式设 model 的工作流即用此模型）；"
        "仅当用户明确指名其它模型时才切换。"
        "出图默认以 0 Anlas（Opus 免费档：steps≤28、≤1024×1024）为目标；"
        "未先获用户明确要求时不要选择会产生 Anlas 消耗的参数或工具。"
        "写提示词前先用 search_tag 检索标准 booru tag（支持中文；"
        "默认词面约 0.1s，词面查不到时才 use_vector=True 走语义，约 1-4s），"
        "角色/主体尽量使用 category=character 的角色 tag。"
    ),
)


def _save_images(images: list[bytes], save_dir: str | None) -> list:
    from pathlib import Path

    # 默认落在 nai_mcp 项目的 images/ 目录（NAI_OUTPUT_DIR 环境变量可覆盖）
    out_dir = (
        Path(save_dir).expanduser()
        if save_dir
        else Path(os.environ.get(
            "NAI_OUTPUT_DIR",
            str(Path(__file__).resolve().parent / "images"),
        ))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    paths = []
    for index, data in enumerate(images):
        path = out_dir / f"nai_{stamp}_{index}.png"
        path.write_bytes(data)
        paths.append(path)
    return paths


def _summarize(wf: dict) -> str:
    return json.dumps(wf, ensure_ascii=False, indent=2)


def estimate_image_payload(resolved: dict, strength: float | None = None) -> float:
    """估价用的 strength：infill 用蒙版 strength，img2img 用底图 strength。"""
    if strength is not None:
        return strength
    mask = resolved.get("mask")
    if mask:
        return mask["strength"]
    img = resolved.get("image")
    return img["strength"] if img else 1.0


def _estimate(
    resolved: dict,
    strength: float | None = None,
    size: tuple[int, int] | None = None,
) -> str:
    """Anlas 估价文本；订阅查询失败时附带原因。"""
    img = resolved.get("image")
    size = size or (resolved["width"], resolved["height"])
    strength = estimate_image_payload(resolved, strength)
    model = resolved.get("model")
    is_opus, usage = nai_request.fetch_subscription()
    text = anlas.estimate_generation_cost(
        width=size[0],
        height=size[1],
        steps=resolved["steps"],
        n_samples=resolved.get("n_samples", 1),
        strength=strength,
        is_opus=is_opus,
        model=model,
    )
    reason = nai_request.subscription_failure()
    if reason and "订阅查询失败" in text:
        text += f"（原因：{reason}）"
    quota = anlas.v5_usage_view(
        model, usage, is_opus=is_opus, area=size[0] * size[1]
    )
    if quota:
        text += "\n" + quota
    return text


# ---------------------------------------------------------------------------
# set_*：写入工作流（不生成）
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "设置工作流的正面提示词与负面提示词（uc）。必须提供工作流名称；"
        "uc 可选（省略时保持已保存的 uc 或清除）。不生成、不消耗 Anlas。"
        "注意：uc 只写必要的目标性负面词（如想规避的特定内容/缺陷/构图问题），"
        "不要写 lowres、worst quality、bad quality 之类质量负面词——"
        "uc_preset（默认 heavy）已自带整套官方质量负面词，重复写会导致其相对权重上升。"
    )
)
def set_prompt(name: str, prompt: str, uc: str = "") -> str:
    """写入 prompt（必需）与 uc（可选负面提示词）。"""
    if not prompt.strip():
        raise ValueError("prompt 不能为空")
    patch = {"prompt": {"text": prompt.strip(), "uc": uc or ""}}
    wf = workflow.write_patch(name, patch)
    return f"工作流 {name} 已更新 prompt：\n{_summarize(wf.get('prompt'))}"


@mcp.tool(
    description=(
        "设置工作流的生成参数（模型/尺寸/步数/引导系数/采样器/种子/数量/"
        "uc 预设/cfg_rescale/噪声调度/use_coords）。必须提供工作流名称，"
        "参数里至少给一项；未给的字段保持原值。不生成、不消耗 Anlas。"
        "uc_preset 为官方负面词预设（heavy/light/humanFocus/none），"
        "已包含整套质量负面词；自定义 uc 无需再写质量词。"
        "注意：n_samples（数量）尽量保持 1，一次跑一张；除非用户特别要求"
        "一次出多张，否则不要改成 1 以上。"
    )
)
def set_gen_param(
    name: str,
    model: str | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    scale: float | None = None,
    sampler: str | None = None,
    seed: int | None = None,
    n_samples: int | None = None,
    uc_preset: str | None = None,
    cfg_rescale: float | None = None,
    noise_schedule: str | None = None,
    use_coords: bool | None = None,
) -> str:
    """部分更新生成参数；省略时按各模型采样预设补默认（见 README）。"""
    given = {
        key: value
        for key, value in {
            "model": model,
            "width": width,
            "height": height,
            "steps": steps,
            "scale": scale,
            "sampler": sampler,
            "seed": seed,
            "n_samples": n_samples,
            "uc_preset": uc_preset,
            "cfg_rescale": cfg_rescale,
            "noise_schedule": noise_schedule,
            "use_coords": use_coords,
        }.items()
        if value is not None
    }
    if not given:
        raise ValueError("至少提供一项生成参数")
    existing = (
        workflow.load(name).get("gen") or {}
        if workflow.workflow_path(name).exists()
        else {}
    )
    gen = {**existing, **given}
    wf = workflow.write_patch(name, {"gen": gen})
    return f"工作流 {name} 已更新 gen 参数：\n{_summarize(wf.get('gen'))}"


@mcp.tool(
    description=(
        "设置工作流的角色提示词（V4+ 模型），含每个角色的负面词 uc 与可选"
        "位置 center [x, y]（0-1，x 左右、y 上下；省略时按人数取官方默认构图）。"
        "角色 uc 与 uc_preset 是叠加关系，同样只写必要的目标性负面词，不重复质量词。"
        "必须提供工作流名称与角色列表；传空列表 [] 可清除全部角色。"
        "use_coords 由 set_gen_param 控制，默认 false。"
    )
)
def set_characters(name: str, characters: list) -> str:
    """characters: [{"prompt": "...", "uc": "...", "center": [x, y]}, ...]"""
    normalized = []
    for index, character in enumerate(characters):
        if not isinstance(character, dict):
            raise ValueError(f"characters[{index}] 应为对象")
        prompt = (character.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"characters[{index}].prompt 不能为空")
        normalized.append(
            {
                "prompt": prompt,
                "uc": character.get("uc") or "",
                "center": (
                    list(character["center"]) if character.get("center") else None
                ),
            }
        )
    wf = workflow.write_patch(name, {"characters": normalized})
    return (
        f"工作流 {name} 已设置 {len(normalized)} 个角色 Prompt：\n"
        + _summarize(wf.get("characters"))
    )


@mcp.tool(
    description=(
        "设置工作流的 img2img 底图（本地 PNG 文件路径），之后 generate_image "
        "自动按 img2img 动作出图。strength/noise 可选（默认 0.7/0.0）。"
        "必须提供工作流名称与 image_path；image_path 传 \"\" 或 \"clear\" 可"
        "清除底图，恢复文生图。"
    )
)
def set_image(name: str, image_path: str, strength: float | None = None, noise: float | None = None) -> str:
    path = image_path.strip() if isinstance(image_path, str) else ""
    if path in ("", "clear"):
        wf = workflow.write_patch(name, {"image": None})
        return f"工作流 {name} 已清除底图，回到文生图。"
    import os

    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        raise ValueError(f"底图不存在：{path}")
    payload = {"path": expanded, "strength": strength, "noise": noise}
    payload = {k: v for k, v in payload.items() if v is not None}
    wf = workflow.write_patch(name, {"image": payload})
    return (
        f"工作流 {name} 已设置底图 {expanded}"
        f"（strength={strength if strength is not None else nai_request.DEFAULT_STRENGTH}"
        f", noise={noise if noise is not None else nai_request.DEFAULT_NOISE}）"
    )


@mcp.tool(
    description=(
        "设置工作流的 inpaint 蒙版（需要先 set_image 底图）。两种来源二选一："
        "1) mask_path：现成蒙版 PNG（白色=重绘区，自动按亮度二值化并缩放到"
        "工作流尺寸）；2) rects：0-1 归一矩形列表 [[x, y, w, h]]（左上原点，"
        "宽高为归一值），server 自行渲染蒙版，agent 无需图像编辑能力。"
        "strength 可选（默认 1.0 完全重绘，<1 时按 Launcher 规则 discounts）。"
        "传 mask_path=\"clear\" 或 rects=[] 清除蒙版；设置蒙版后 "
        "generate_image 自动按 infill 动作出图并切换 inpainting 模型。"
    )
)
def set_mask(
    name: str,
    mask_path: str = "",
    rects: list | None = None,
    strength: float | None = None,
    noise: float | None = None,
) -> str:
    """写入蒙版（mask_path 或 rects 二选一）；"clear" 清除。"""
    import os

    path = (mask_path or "").strip()
    normalized_rects = [list(r) for r in rects] if rects else []
    if path == "clear":
        wf = workflow.write_patch(name, {"mask": None})
        return f"工作流 {name} 已清除蒙版。"
    if not path and not normalized_rects:
        raise ValueError("提供 mask_path（蒙版 PNG）或 rects（[[x, y, w, h], …]）之一")
    if path and not os.path.isfile(os.path.expanduser(path)):
        raise ValueError(f"蒙版文件不存在：{path}")
    if path:
        payload = {"path": os.path.expanduser(path), "rects": []}
    else:
        import nai_mask

        # 预渲染校验坐标合法性与蒙版非空
        raw = workflow.load(name) if workflow.workflow_path(name).exists() else {}
        gen = raw.get("gen") or {}
        width, height = nai_request.sanitize_resolution(
            int(gen.get("width") or 832), int(gen.get("height") or 1216)
        )
        nai_mask.prepare_mask(None, normalized_rects, width, height)
        payload = {"path": "", "rects": normalized_rects}
    payload["strength"], payload["noise"] = strength, noise
    payload = {k: v for k, v in payload.items() if v not in (None, "", [])}
    wf = workflow.write_patch(name, {"mask": payload})
    source = (
        f"蒙版文件 {payload['path']}"
        if payload.get("path")
        else f"矩形 {len(payload['rects'])} 个 {payload['rects']}"
    )
    return (
        f"工作流 {name} 已设置 inpaint 蒙版：{source}"
        f"（strength={strength if strength is not None else 1.0}）。"
        "工作流需已 set_image 底图，生成时按 infill 出图。"
    )


# ---------------------------------------------------------------------------
# read / estimate（必须提供工作流名称，不生成）
# ---------------------------------------------------------------------------


@mcp.tool(description="读取命名工作流（含已设参数与 Anlas 预估），供确认。")
def read_workflow(name: str) -> str:
    """读取工作流内容与预估消耗；不生成。"""
    raw = workflow.load(name)
    resolved = workflow.resolve_generation(raw)
    return _summarize(resolved) + "\n" + _estimate(resolved)


@mcp.tool(description="估算命名工作流一次出图的 Anlas 消耗（不请求生成）。")
def estimate_anlas(name: str) -> str:
    """只做估价，不生成。"""
    resolved = workflow.resolve_generation(workflow.load(name))
    return _estimate(resolved)


@mcp.tool(description="列出全部工作流名称。")
def list_workflows() -> list:
    return workflow.list_names()


@mcp.tool(
    description=(
        "列出 set_gen_param / create_workflow 的合法取值：模型（各模型默认"
        "步数/引导系数、Anlas 倍率、是否支持 cfg_rescale、角色 prompt 上限、"
        "inpainting 变体）、采样器、噪声调度、uc_preset、尺寸约束（64 倍数"
        "网格与最大正方形）。调用 set_gen_param 前先查本工具选合法值，避免"
        "猜错模型名/采样器被拒；不生成、不消耗 Anlas。"
    )
)
def list_gen_options() -> dict:
    """set_gen_param 可选值与各模型采样预设（源：nai_request 常量）。"""
    grid = nai_request.GRID
    models = {}
    for name, profile in nai_request.MODELS.items():
        models[name] = {
            "default_steps": profile.steps,
            "default_scale": profile.scale,
            "max_characters": profile.max_characters or None,
            "anlas_multiplier": profile.anlas_multiplier,
            "supports_cfg_rescale": profile.supports_cfg_rescale,
            "inpainting_variant": nai_request.INPAINTING_MODELS.get(name),
        }
    return {
        "models": models,
        "samplers": list(nai_request.SAMPLERS),
        "default_sampler": nai_request.DEFAULT_SAMPLER,
        "noise_schedules": list(nai_request.NOISE_SCHEDULES),
        "uc_presets": list(nai_request.UC_PRESET_IDS),
        "size": {
            "grid_multiples": grid,
            "max_pixels": nai_request.MAX_PIXELS,
            "max_square_side": int(nai_request.MAX_PIXELS ** 0.5) // grid * grid,
        },
        "notes": (
            "uc_preset 默认 heavy，预设文本随模型不同（生成时自动拼进官方"
            "质量负面词，可在 read_workflow 的 uc 预览）；furryFocus 未开放"
            "（set_gen_param 会拒绝）；ddim/ddim_v3 仅 V3 系模型可用：V4+ "
            f"自动回退 {nai_request.DEFAULT_SAMPLER}，nai-diffusion-3 换 "
            "ddim_v3。"
        ),
    }


@mcp.tool(
    description=(
        "创建空工作流（可选 model 预填模型名并同步写入该模型的采样预设："
        "步数/引导系数，gen 其余留空按预设取默认）。名称规则与 set_* 相同；"
        "已存在的同名工作流会报错。后续用 set_prompt / set_gen_param / "
        "set_characters / set_image / set_mask 填内容。"
    )
)
def create_workflow(name: str, model: str | None = None) -> str:
    """创建空工作流文件；重复创建报错。"""
    if workflow.workflow_path(name).exists():
        raise ValueError(
            f"工作流 {name} 已存在（先 list_workflows 查看，或直接用 set_* 更新）"
        )
    gen: dict = {}
    if model:
        if model not in nai_request.MODELS:
            raise ValueError(
                f"不支持的模型 {model}，可选：{', '.join(nai_request.MODELS)}"
            )
        profile = nai_request.MODELS[model]
        gen.update({"model": model, "steps": profile.steps, "scale": profile.scale})
    workflow.write_patch(name, {"gen": gen})
    detail = f"（model={model}，含采样预设）" if gen else "（全空，生成时按默认模型预设）"
    return (
        f"工作流 {name} 已创建{detail}。"
        "下一步：set_prompt 写提示词，之后 read_workflow/estimate_anlas 确认、"
        "generate_image 出图。"
    )


# ---------------------------------------------------------------------------
# 生成类：按工作流名实际出图（消耗 Anlas）
# ---------------------------------------------------------------------------


def _dispatch_generation(
    name: str, extra_prompt_addition: bool = False
) -> tuple[list[bytes], dict]:
    """按工作流构造并调用生成接口，返回（图片字节, 用于展示的参数）。"""
    raw = workflow.load(name)
    resolved = workflow.resolve_generation(raw)
    img_info = resolved["image"]
    mask_info = resolved.get("mask")
    image_bytes = (
        open(img_info["path"], "rb").read() if img_info else None
    )
    mask_bytes = None
    soft_mask = None
    if mask_info:
        if not image_bytes:
            raise ValueError(
                f"工作流 {name} 设置了蒙版但没有底图：先 set_image 再生图"
            )
        import nai_mask

        # 官网式羽化蒙版：请求用软蒙版，生成后还要按它合成回原图
        mask_bytes, soft_mask = nai_mask.prepare_mask(
            mask_info.get("path") or None,
            mask_info.get("rects") or None,
            resolved["width"],
            resolved["height"],
        )
    seed = resolved["seed"]
    if seed == -1:
        import secrets

        seed = secrets.randbelow(4294967295)
    size = (resolved["width"], resolved["height"])
    request = nai_request.build_request(
        prompt=resolved["prompt"],
        negative_prompt=resolved["negative_prompt"],
        model=resolved["model"],
        width=size[0],
        height=size[1],
        steps=resolved["steps"],
        scale=resolved["scale"],
        sampler=resolved["sampler"],
        seed=seed,
        n_samples=resolved["n_samples"],
        uc_preset=resolved["uc_preset"],
        cfg_rescale=resolved["cfg_rescale"],
        noise_schedule=resolved["noise_schedule"],
        characters=resolved["characters"],
        use_coords=resolved["use_coords"],
        source_image_b64=(
            base64.b64encode(image_bytes).decode() if image_bytes else None
        ),
        strength=img_info["strength"] if img_info else None,
        noise=img_info["noise"] if img_info else None,
        enhance_addition=extra_prompt_addition,
        mask_b64=(
            base64.b64encode(mask_bytes).decode() if mask_bytes else None
        ),
        inpaint_strength=mask_info["strength"] if mask_info else None,
    )
    action = (
        "infill" if mask_bytes
        else "img2img" if image_bytes
        else "generate"
    )
    params_strength = (
        mask_info["strength"] if mask_info else
        img_info["strength"] if img_info else None
    )
    response = nai_request.generate_image(request)
    images = nai_request.pngs_from_response(response)
    if soft_mask:
        # 官网后处理：生成结果按软蒙版合成回原图，蒙版外像素保持原图
        import nai_mask

        images = [
            nai_mask.composite(image_bytes, png, soft_mask, size[0], size[1])
            for png in images
        ]
    return images, (
        {
            "name": name,
            "model": nai_request.INPAINTING_MODELS.get(resolved["model"], resolved["model"])
            if action == "infill" else resolved["model"],
            "size": size,
            "steps": resolved["steps"],
            "scale": resolved["scale"],
            "sampler": resolved["sampler"],
            "uc_preset": resolved["uc_preset"],
            "seed": seed,
            "n_samples": resolved["n_samples"],
            "action": action,
            "strength": params_strength if img_info else None,
            "noise": img_info["noise"] if img_info else None,
            "mask": mask_info if mask_info else None,
            "characters": len(resolved["characters"]),
            "use_coords": resolved["use_coords"],
        }
    )


@mcp.tool(
    description=(
        "按命名工作流实际生成图片（消耗 Anlas 或 Opus 免费额度）。"
        "必须提供工作流名称；先用 read_workflow / estimate_anlas 确认，"
        "未获用户明确要求时保持 0 Anlas（steps≤28、≤1024×1024 即免费）。"
        "生成前建议先用 search_tag 把提示词规范化为标准 booru tag。"
        "工作流若设置了蒙版则按 infill（局部重绘）出图并自动切换 inpainting "
        "模型；只有底图则 img2img；设置了角色 prompt 则走 V4 多角色结构。"
        "返回 PNG 内容与实际参数。"
        "注意：不要自行检查/分析返回的图片内容，否则很容易被画面带偏，"
        "无法专注于用户内容和提示词；判断好坏以此前协商的提示词为准。"
    )
)
def generate_image(name: str) -> list:
    """按当前工作流出图；返回图片内容、参数摘要与 Anlas 预估。"""
    images, params = _dispatch_generation(name)
    blocks = [Image(data=data, format="png").to_image_content() for data in images]
    blocks.append(
        "生成完成：" + json.dumps(params, ensure_ascii=False)
        + "\n"
        + _estimate({"width": params["size"][0], "height": params["size"][1],
                     "steps": params["steps"], "n_samples": params["n_samples"],
                     "model": params["model"],
                     "image": None if params["strength"] is None
                             else {"strength": params["strength"]}})
        + "；已保存:\n"
        + "\n".join(str(p) for p in _save_images(images, None))
    )
    return blocks


@mcp.tool(
    description=(
        "增强（Enhance）：按命名工作流的提示词与采样参数重绘底图并放大。"
        "必须提供工作流名称（底图用 set_image 设置）；level 1-5 官方档位 "
        "（默认 3），factor 期望放大倍率 2.0/1.5/1.0（超出面积上限时自动"
        "回落到下一档）。请求尺寸 = 底图尺寸×factor 对齐 64 网格。"
        "消耗 Anlas。"
    )
)
def enhance_image(name: str, level: int = 3, factor: float = 2.0) -> list:
    """按增强档位 level 与倍率 factor 重绘底图。"""
    raw = workflow.load(name)
    resolved = workflow.resolve_generation(raw)
    if resolved.get("mask"):
        raise ValueError(
            f"工作流 {name} 设置了蒙版：Enhance 走 img2img 全图重绘，"
            "与 infill 不兼容；先 set_mask(name, \"clear\") 再增强"
        )
    if not resolved["image"]:
        raise ValueError(
            f"工作流 {name} 还没有底图：先调用 set_image(name, image_path)"
        )
    if not (1 <= int(level) <= 5):
        raise ValueError("level 取值 1-5")
    # EnhanceLevels：1-5 档固定 strength/noise（最高档 0.7/0.1）
    strengths = (0.2, 0.4, 0.5, 0.6, 0.7)
    noises = (0.0, 0.0, 0.0, 0.0, 0.1)
    level = int(level)
    if resolved["image"]:
        resolved["image"] = {
            **resolved["image"],
            "strength": strengths[level - 1],
            "noise": noises[level - 1],
        }
    source_bytes = open(resolved["image"]["path"], "rb").read()
    source_w, source_h = nai_request.read_png_size(source_bytes)
    width, height = nai_request.sanitize_resolution(
        *nai_request.resolve_enhance_target(source_w, source_h, factor)
    )
    # 临时替换源图尺寸: enhance n_samples 固定 1
    resolved["width"], resolved["height"] = width, height
    resolved["n_samples"] = 1
    images, params = _enhance_dispatch(resolved)
    blocks = [Image(data=data, format="png").to_image_content() for data in images]
    blocks.append(
        "增强完成：" + json.dumps(params, ensure_ascii=False)
        + "\n"
        + _estimate({"width": width, "height": height,
                     "steps": resolved["steps"], "n_samples": 1,
                     "model": resolved["model"],
                     "image": {"strength": strengths[level - 1]}})
        + "；已保存:\n"
        + "\n".join(str(p) for p in _save_images(images, None))
    )
    return blocks


def _enhance_dispatch(resolved: dict) -> tuple[list[bytes], dict]:
    """增强请求与 img2img 相同，n_samples 固定 1。"""
    from secrets import randbelow

    seed = resolved["seed"]
    if seed == -1:
        seed = randbelow(4294967295)
    image_bytes = open(resolved["image"]["path"], "rb").read()
    request = nai_request.build_request(
        prompt=resolved["prompt"],
        negative_prompt=resolved["negative_prompt"],
        model=resolved["model"],
        width=resolved["width"],
        height=resolved["height"],
        steps=resolved["steps"],
        scale=resolved["scale"],
        sampler=resolved["sampler"],
        seed=seed,
        n_samples=1,
        uc_preset=resolved["uc_preset"],
        cfg_rescale=resolved["cfg_rescale"],
        noise_schedule=resolved["noise_schedule"],
        source_image_b64=base64.b64encode(image_bytes).decode(),
        strength=resolved["image"]["strength"],
        noise=resolved["image"]["noise"],
        enhance_addition=True,
    )
    zip_bytes = nai_request.generate_image(request)
    images = nai_request.pngs_from_response(zip_bytes)
    params = {
        "action": "enhance",
        "model": resolved["model"],
        "size": f"{resolved['width']}x{resolved['height']}",
        "strength": resolved["image"]["strength"],
        "noise": resolved["image"]["noise"],
        "seed": seed,
    }
    return images, params


@mcp.tool(
    description=(
        "云端超分（/ai/upscale）：直接放大输入图片，不重绘。必须提供本地 "
        "PNG 路径。Opus 用户输入 ≤ 640×640 免费，否则按输入面积 1-4 Anlas "
        "分档（与官网一致）。返回 PNG 内容与保存路径。"
    )
)
def upscale_image(image_path: str) -> list:
    """cloud upscale 单张图片。"""
    import os

    expanded = os.path.expanduser(image_path)
    if not os.path.isfile(expanded):
        raise ValueError(f"图片不存在：{image_path}")
    source_bytes = open(expanded, "rb").read()
    input_w, input_h = nai_request.read_png_size(source_bytes)
    cost = anlas.estimate_upscale_cost(input_w * input_h, nai_request.fetch_is_opus())
    reason = nai_request.subscription_failure()
    if reason and "订阅查询失败" in cost:
        cost += f"（原因：{reason}）"
    zip_bytes = nai_request.upscale_image(source_bytes)
    images = nai_request.pngs_from_response(zip_bytes)
    blocks = [Image(data=data, format="png").to_image_content() for data in images]
    blocks.append(
        f"放大完成：输入 {expanded}（{input_w}x{input_h}）；{cost}；已保存:\n"
        + "\n".join(str(p) for p in _save_images(images, None))
    )
    return blocks


@mcp.tool(
    description=(
        "检索标准 booru tag（供提示词用），支持中文/别名/tag 词面匹配；"
        "例如输入“面向观众”可返回 looking at viewer。词库由 "
        "tool/build_tag_vector_db.py 从 Launcher tag_catalog.db 生成。"
        "use_vector 默认 False（仅词面，约 0.1s）；改 True 时走 embeddings "
        "语义检索（需已配置 NAI_EMBEDDING_URL 并填充向量，全新查询约 1-4s）——"
        "词面查不到或中文描述比较绕时再用。"
    )
)
def search_tag(query: str, limit: int = 10, use_vector: bool = False) -> list:
    """search_tags([...]) → tag 列表；use_vector 控制语义检索。"""
    results = tagsearch.search(query, limit, use_vector=use_vector)
    # 请求语义但未配置/未填充时给出明确提示，避免静默错误结果
    if (
        use_vector
        and isinstance(results, list)
        and not tagsearch.semantic_ready()
    ):
        message = "（提示：语义检索未生效"
        if not tagsearch.DB_PATH.exists():
            message += "：标签库不存在，请先运行 tool/build_tag_vector_db.py"
        else:
            tags, _ = tagsearch._load_cache()
            if tagsearch._has_vectors(tags):
                message += "：本 MCP server 未配置 NAI_EMBEDDING_URL/NAI_EMBEDDING_MODEL"
            else:
                message += "：词库向量尚未填充，请先运行 tool/fill_tag_vectors.py"
        message += "；已回退词面匹配结果）"
        results.append(message)
    return results


if __name__ == "__main__":
    mcp.run(transport="stdio")
