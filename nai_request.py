"""NovelAI 图像请求构造与传输（对齐官方 OpenAPI 文档 /docs/doc.json）。

- 生图：POST https://image.novelai.net/ai/generate-image（纯 JSON body，
  parameters.image / parameters.mask 为 base64 内联）
- 云端放大：POST https://image.novelai.net/ai/upscale（纯 JSON body；
  404/405 时回退主 API 旧版 JSON → ZIP）
- 订阅档位：GET https://image.novelai.net/user/subscription
"""

import base64
import io
import json
import struct
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass

IMAGE_BASE_URL = "https://image.novelai.net"
MAIN_BASE_URL = "https://api.novelai.net"
GENERATE_PATH = "/ai/generate-image"
UPSCALE_PATH = "/ai/upscale"
USER_SUBSCRIPTION_PATH = "/user/subscription"

MAX_PIXELS = 3145728  # 官网最大 3.14 MP
GRID = 64  # 尺寸必须为 64 的倍数
PRESET_MODEL = "nai-diffusion-5-full"  # 与 Launcher 默认（ImageModels.animeDiffusionV5Full）一致
UPSCALE_MODEL = "nai-diffusion-5-curated"  # /ai/upscale v2 固定模型

UC_PRESET_IDS = ("heavy", "light", "humanFocus", "none")
# UC 预设的官网数字提示（tag_hint_uc_preset）
UC_TAG_HINT = {"none": 0, "heavy": 2, "light": 3, "humanFocus": 4}
# 官方质量词 standard 档提示
QUALITY_TAG_HINT = 1

SAMPLERS = (
    "k_lms",
    "k_euler",
    "k_euler_ancestral",
    "k_heun",
    "k_dpm_2",
    "k_dpm_2_ancestral",
    "k_dpmpp_2m",
    "k_dpmpp_2m_sde",
    "k_dpmpp_2s_ancestral",
    "k_dpmpp_sde",
    "ddim",
    "ddim_v3",
)
NOISE_SCHEDULES = ("karras", "native", "exponential", "polyexponential")

# Launcher ImageParams 通用默认采样器（各模型未设 fixedSettings）
DEFAULT_SAMPLER = "k_euler_ancestral"
# Launcher 默认 img2img 强度/噪声
DEFAULT_STRENGTH = 0.7
DEFAULT_NOISE = 0.0

ENHANCE_PROMPT_ADDITION = ", -2::upscaled, blurry::,"

# 角色 Prompt 相对布局默认位置（CharacterPositionLayout），column=x * 纵向 y
CHARACTER_DEFAULT_POSITIONS = {
    1: [(0.5, 0.5)],
    2: [(0.25, 0.5), (0.75, 0.5)],
    3: [(0.2, 0.5), (0.5, 0.5), (0.8, 0.5)],
    4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
    5: [(0.2, 0.2), (0.8, 0.2), (0.5, 0.5), (0.2, 0.8), (0.8, 0.8)],
    6: [(0.2, 0.25), (0.5, 0.25), (0.8, 0.25),
        (0.2, 0.75), (0.5, 0.75), (0.8, 0.75)],
}
MAX_CHARACTERS = 6

# V4 结构（带 v4_prompt / characterPrompts）的模型
_V4_PREFIX = ("nai-diffusion-4", "nai-diffusion-5")
# 支持增强降权词追加的模型（capabilities.supportsEnhancePromptAdd）
_ENHANCE_ADDITION_MODELS = (
    "nai-diffusion-5-full",
    "nai-diffusion-5-curated",
    "nai-diffusion-5-full-medium",
    "nai-diffusion-4-5-full",
    "nai-diffusion-4-5-curated",
)

# infill 模型变体映射（Launcher ImageModels.resolveInpaintingModel；
# V5 Curated 的重绘权重尚未就绪，官网映射到 V4.5 Curated Inpainting）
INPAINTING_MODELS = {
    "nai-diffusion-5-full": "nai-diffusion-5-full-inpainting",
    "nai-diffusion-5-full-medium": "nai-diffusion-5-full-medium-inpainting",
    "nai-diffusion-5-curated": "nai-diffusion-4-5-curated-inpainting",
    "nai-diffusion-4-5-full": "nai-diffusion-4-5-full-inpainting",
    "nai-diffusion-4-5-curated": "nai-diffusion-4-5-curated-inpainting",
    "nai-diffusion-4-full": "nai-diffusion-4-full-inpainting",
    "nai-diffusion-3": "nai-diffusion-3-inpainting",
}
DEFAULT_INPAINT_STRENGTH = 1.0  # Launcher ImageParams.inpaintStrength 默认


@dataclass
class ModelProfile:
    """模型采样预设（Launcher ModelCapabilities 的 defaultScale/defaultSteps）
    与官方质量词、UC 预设文本。"""

    quality: str
    uc_presets: dict
    scale: float
    steps: int
    max_characters: int = 0  # 0 = 不支持角色 Prompt
    # Anlas 模型倍率（V5 为 1.5）与是否发送 cfg_rescale（V5 Medium 不支持）
    anlas_multiplier: float = 1.0
    supports_cfg_rescale: bool = True


_V4_UC = {
    "heavy": (
        "lowres, artistic error, film grain, scan artifacts, worst quality, "
        "bad quality, jpeg artifacts, very displeasing, chromatic aberration, "
        "dithering, halftone, screentone, multiple views, logo, "
        "too many watermarks, negative space, blank page"
    ),
    "light": (
        "lowres, artistic error, scan artifacts, worst quality, bad quality, "
        "jpeg artifacts, multiple views, very displeasing, "
        "too many watermarks, negative space, blank page"
    ),
    "humanFocus": (
        "lowres, artistic error, film grain, scan artifacts, worst quality, "
        "bad quality, jpeg artifacts, very displeasing, chromatic aberration, "
        "dithering, halftone, screentone, multiple views, logo, "
        "too many watermarks, negative space, blank page, @_@, "
        "mismatched pupils, glowing eyes, bad anatomy"
    ),
    "none": "",
}

_FURRY_UC = (
    "{worst quality}, distracting watermark, unfinished, bad quality, "
    "{widescreen}, upscale, {sequence}, {{grandfathered content}}, "
    "blurred foreground, chromatic aberration, sketch, everyone, "
    "[sketch background], simple, [flat colors], ych (character), "
    "outline, multiple scenes, [[horror (theme)]], comic"
)

# V5 预设（Curated/Full 共用；heavy/humanFocus 与 V4.5 Full 同文案，light 为 V5 新文案）
_V5_UC = {
    "heavy": (
        "lowres, artistic error, film grain, scan artifacts, worst quality, "
        "bad quality, jpeg artifacts, very displeasing, chromatic aberration, "
        "dithering, halftone, screentone, multiple views, logo, "
        "too many watermarks, negative space, blank page"
    ),
    "light": (
        "lowres, bad hands, bad anatomy, artistic error, sepia, white haze, "
        "worst quality, very displeasing, jpeg artifacts, 0::ai-generated::"
    ),
    "humanFocus": (
        "lowres, artistic error, film grain, scan artifacts, worst quality, "
        "bad quality, jpeg artifacts, very displeasing, chromatic aberration, "
        "dithering, halftone, screentone, multiple views, logo, "
        "too many watermarks, negative space, blank page, @_@, "
        "mismatched pupils, glowing eyes, bad anatomy"
    ),
    "furryFocus": _FURRY_UC,
    "none": "",
}

# V5 质量词（官网 standard 档；另有 light 档 'very aesthetic, amazing quality, no text'）
_V5_QUALITY = "very aesthetic, masterpiece, no text"

MODELS: dict[str, ModelProfile] = {
    "nai-diffusion-4-5-full": ModelProfile(
        quality="location, very aesthetic, masterpiece, no text",
        uc_presets=_V4_UC,
        scale=5.0,
        steps=23,
        max_characters=MAX_CHARACTERS,
    ),
    "nai-diffusion-4-5-curated": ModelProfile(
        quality="location, masterpiece, no text, -0.8::feet::, rating:general",
        uc_presets={
            "heavy": (
                "blurry, lowres, upscaled, artistic error, film grain, "
                "scan artifacts, worst quality, bad quality, jpeg artifacts, "
                "very displeasing, chromatic aberration, halftone, "
                "multiple views, logo, too many watermarks, negative space, "
                "blank page"
            ),
            "light": (
                "blurry, lowres, upscaled, artistic error, scan artifacts, "
                "jpeg artifacts, logo, too many watermarks, negative space, "
                "blank page"
            ),
            "humanFocus": (
                "blurry, lowres, upscaled, artistic error, film grain, "
                "scan artifacts, bad anatomy, bad hands, worst quality, "
                "bad quality, jpeg artifacts, very displeasing, "
                "chromatic aberration, halftone, multiple views, logo, "
                "too many watermarks, @_@, mismatched pupils, glowing eyes, "
                "negative space, blank page"
            ),
            "none": "",
            "furryFocus": _FURRY_UC,
        },
        scale=5.0,
        steps=23,
        max_characters=MAX_CHARACTERS,
    ),
    "nai-diffusion-4-full": ModelProfile(
        quality="no text, best quality, very aesthetic, absurdres",
        uc_presets={
            "heavy": (
                "blurry, lowres, error, film grain, scan artifacts, "
                "worst quality, bad quality, jpeg artifacts, very displeasing, "
                "chromatic aberration, multiple views, logo, "
                "too many watermarks"
            ),
            "light": (
                "blurry, lowres, error, worst quality, bad quality, "
                "jpeg artifacts, very displeasing"
            ),
            "humanFocus": (
                "blurry, lowres, error, film grain, scan artifacts, "
                "worst quality, bad quality, jpeg artifacts, very displeasing, "
                "chromatic aberration, multiple views, logo, "
                "too many watermarks, bad anatomy, bad hands"
            ),
            "furryFocus": _FURRY_UC,
            "none": "",
        },
        scale=5.5,
        steps=23,
        max_characters=MAX_CHARACTERS,
    ),
    "nai-diffusion-5-full": ModelProfile(
        quality=_V5_QUALITY,
        uc_presets=_V5_UC,
        scale=4.0,
        steps=28,
        max_characters=MAX_CHARACTERS,
        anlas_multiplier=1.5,
    ),
    "nai-diffusion-5-curated": ModelProfile(
        quality=_V5_QUALITY,
        uc_presets=_V5_UC,
        scale=4.0,
        steps=28,
        max_characters=MAX_CHARACTERS,
        anlas_multiplier=1.5,
    ),
    "nai-diffusion-5-full-medium": ModelProfile(
        quality=_V5_QUALITY,
        uc_presets=_V5_UC,
        scale=4.0,
        steps=14,  # fixedSettings：Medium 档固定 14 步
        max_characters=MAX_CHARACTERS,
        anlas_multiplier=1.5,
        supports_cfg_rescale=False,
    ),
    "nai-diffusion-3": ModelProfile(
        quality="best quality, amazing quality, very aesthetic, absurdres",
        uc_presets={
            "heavy": (
                "lowres, {bad}, error, fewer, extra, missing, worst quality, "
                "jpeg artifacts, bad quality, watermark, unfinished, "
                "displeasing, chromatic aberration, signature, extra digits, "
                "artistic error, username, scan, [abstract]"
            ),
            "light": (
                "lowres, jpeg artifacts, worst quality, watermark, blurry, "
                "very displeasing"
            ),
            "humanFocus": (
                "lowres, {bad}, error, fewer, extra, missing, worst quality, "
                "jpeg artifacts, bad quality, watermark, unfinished, "
                "displeasing, chromatic aberration, signature, extra digits, "
                "artistic error, username, scan, [abstract], bad anatomy, "
                "bad hands, @_@, mismatched pupils, heart-shaped pupils, "
                "glowing eyes"
            ),
            "furryFocus": _FURRY_UC,
            "none": "lowres",
        },
        scale=5.0,
        steps=23,
    ),
}


def is_v4_model(model: str) -> bool:
    return model.startswith(_V4_PREFIX)


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------


def sanitize_resolution(width: int, height: int) -> tuple[int, int]:
    width = max(GRID, width // GRID * GRID)
    height = max(GRID, height // GRID * GRID)
    if width * height > MAX_PIXELS:
        factor = (MAX_PIXELS / (width * height)) ** 0.5
        width = max(GRID, int(width * factor) // GRID * GRID)
        height = max(GRID, int(height * factor) // GRID * GRID)
    return width, height


def resolve_sampler(sampler: str | None, model: str) -> str:
    """按 Launcher `Samplers.resolveForModel` 规则换算采样器。"""
    resolved = sampler or DEFAULT_SAMPLER
    if resolved in ("ddim", "ddim_v3"):
        if is_v4_model(model):
            return DEFAULT_SAMPLER  # V4 起不支持 DDIM
        if "diffusion-3" in model:
            return "ddim_v3"
    return resolved


def normalize_dimension(value: int) -> int:
    normalized = ((value + GRID // 2) // GRID) * GRID
    return max(GRID, min(4096, normalized))


def resolve_enhance_target(source_w: int, source_h: int, factor: float) -> tuple[int, int]:
    """增强目标尺寸：Launcher EnhanceScales 语义——首选倍率可用则用之，
    否则回落到面积上限内最大的候选倍率（2.0/1.5/1.0），均不可用回 1.0。"""
    candidates = (2.0, 1.5, 1.0)

    def target_of(candidate: float) -> tuple[int, int]:
        return (
            normalize_dimension(round(source_w * candidate)),
            normalize_dimension(round(source_h * candidate)),
        )

    available = [
        c for c in candidates if target_of(c)[0] * target_of(c)[1] <= MAX_PIXELS
    ]
    chosen = factor if factor in available else (available[0] if available else 1.0)
    return target_of(chosen)


def character_center(index: int, total: int, use_coords: bool, center):
    """角色中心：显式给定优先，否则按数量取默认布局。"""
    if center is not None:
        x, y = center
        if 0 <= x <= 1 and 0 <= y <= 1:
            return float(x), float(y)
        raise ValueError(f"角色中心越界: {center}（x/y 需在 0-1）")
    if use_coords:
        raise ValueError("use_coords 开启时每个角色必须显式提供 center")
    fallback = CHARACTER_DEFAULT_POSITIONS[max(total, 1)][min(index, total - 1)]
    return fallback


def supports_enhance_addition(model: str) -> bool:
    return model in _ENHANCE_ADDITION_MODELS


# ---------------------------------------------------------------------------
# 请求体构造
# ---------------------------------------------------------------------------


def build_request(
    *,
    prompt: str,
    negative_prompt: str,
    model: str,
    width: int,
    height: int,
    steps: int,
    scale: float,
    sampler: str,
    seed: int,
    n_samples: int,
    uc_preset: str,
    cfg_rescale: float = 0.0,
    noise_schedule: str = "karras",
    characters: list | None = None,
    use_coords: bool = False,
    source_image_b64: str | None = None,
    strength: float | None = None,
    noise: float | None = None,
    enhance_addition: bool = False,
    mask_b64: str | None = None,
    inpaint_strength: float | None = None,
) -> dict:
    """构造 generate-image 请求体（含 V4 角色结构、img2img 与 infill 字段）。"""
    profile = MODELS[model]
    # infill 时按官网规则切换 inpainting 变体（采样预设仍按基础模型取值）
    model_for_preset = model
    if mask_b64 and model in INPAINTING_MODELS:
        model = INPAINTING_MODELS[model]
    is_v4 = is_v4_model(model_for_preset)

    effective_prompt = f"{prompt}, {profile.quality}" if prompt else profile.quality
    if enhance_addition and supports_enhance_addition(model):
        effective_prompt = effective_prompt.rstrip() + ENHANCE_PROMPT_ADDITION
    if is_v4:
        base_negative = negative_prompt
        preset_text = profile.uc_presets.get(uc_preset, "")
        if preset_text and base_negative:
            effective_negative = f"{preset_text}, {base_negative}"
        else:
            effective_negative = preset_text or base_negative
    else:
        profile_uc = profile.uc_presets.get(uc_preset, "")
        if profile_uc and negative_prompt:
            effective_negative = f"{profile_uc}, {negative_prompt}"
        else:
            effective_negative = profile_uc or negative_prompt

    if characters and not is_v4:
        raise ValueError(f"模型 {model} 不支持角色 Prompt")
    if characters and len(characters) > profile.max_characters:
        raise ValueError(
            f"模型 {model} 最多支持 {profile.max_characters} 个角色"
        )

    params: dict = {
        "params_version": 4,
        "width": width,
        "height": height,
        "scale": scale,
        "sampler": sampler,
        "steps": steps,
        "n_samples": n_samples,
        "ucPresetId": uc_preset,
        "qualityPresetId": "standard",
        "controlnet_strength": 1,
        "legacy": False,
        "add_original_image": False,
        "noise_schedule": noise_schedule,
        "seed": seed,
        "image_format": "png",
        "tag_hint_qt": QUALITY_TAG_HINT,
        "tag_hint_uc_preset": UC_TAG_HINT[uc_preset],
    }

    if is_v4:
        params["autoSmea"] = False
        if profile.supports_cfg_rescale:
            params["cfg_rescale"] = cfg_rescale
        params["use_coords"] = use_coords
        params.update(
            {
                "use_order": True,
                "legacy_v3_extend": False,
                "legacy_uc": False,
                "normalize_reference_strength_multiple": True,
                "v4_prompt": {
                    "caption": {
                        "base_caption": effective_prompt,
                        "char_captions": [],
                    },
                    "use_coords": use_coords,
                    "use_order": True,
                },
                "v4_negative_prompt": {
                    "caption": {
                        "base_caption": effective_negative,
                        "char_captions": [],
                    },
                    "legacy_uc": False,
                },
                "characterPrompts": [],
            }
        )
        for index, character in enumerate(characters or []):
            x, y = character_center(
                index, len(characters), use_coords, character.get("center")
            )
            char_caption = {
                "centers": [{"x": x, "y": y}],
                "char_caption": character["prompt"],
            }
            negative_caption = {
                "centers": [{"x": x, "y": y}],
                "char_caption": character.get("uc") or "",
            }
            params["v4_prompt"]["caption"]["char_captions"].append(char_caption)
            params["v4_negative_prompt"]["caption"]["char_captions"].append(
                negative_caption
            )
            params["characterPrompts"].append(
                {
                    "center": {"x": x, "y": y},
                    "prompt": character["prompt"],
                    "uc": character.get("uc") or "",
                    "enabled": True,
                }
            )
    else:
        params["sm"] = False
        params["sm_dyn"] = False
        params["dynamic_thresholding"] = False

    if effective_negative:
        params["negative_prompt"] = effective_negative
    else:
        params["uc"] = ""

    if (
        sampler == "k_euler_ancestral"
        and noise_schedule != "native"
    ):
        params["deliberate_euler_ancestral_bug"] = False
        params["prefer_brownian"] = True

    if source_image_b64:
        params["image"] = source_image_b64
        params["strength"] = strength if strength is not None else DEFAULT_STRENGTH
        params["noise"] = noise if noise is not None else DEFAULT_NOISE
        params["color_correct"] = False
        # 官网对带底图的请求固定 extra_noise_seed = seed - 1
        params["extra_noise_seed"] = seed - 1

    action = "generate"
    if mask_b64:
        # Launcher nai_image_request_builder infill 分支：image + mask + 强度
        if not source_image_b64:
            raise ValueError("infill 需要底图：先用 set_image 设置底图")
        params["mask"] = mask_b64
        params["strength"] = strength if strength is not None else DEFAULT_INPAINT_STRENGTH
        params["noise"] = noise if noise is not None else DEFAULT_NOISE
        action = "infill"
        # Launcher buildBaseParameters：V4 请求固定携带 inpaintImg2ImgStrength
        if is_v4:
            params["inpaintImg2ImgStrength"] = (
                inpaint_strength if inpaint_strength is not None else DEFAULT_INPAINT_STRENGTH
            )
        if inpaint_strength is not None and inpaint_strength != DEFAULT_INPAINT_STRENGTH:
            params["img2img"] = {"strength": inpaint_strength, "color_correct": True}
    elif source_image_b64:
        action = "img2img"

    return {
        "input": effective_prompt,
        "model": model,
        "action": action,
        "parameters": params,
        "use_new_shared_trial": True,
    }


# ---------------------------------------------------------------------------
# 传输
# ---------------------------------------------------------------------------


def _normalize_token(value: str) -> str:
    """与 Launcher 一致的 token 规范化：去引号、循环剥 Bearer 前缀、去空白。"""
    normalized = value.strip()
    if len(normalized) >= 2 and normalized[0] in "\"'" and normalized[-1] == normalized[0]:
        normalized = normalized[1:-1].strip()
    previous = ""
    while normalized != previous:
        previous = normalized
        head, normalized = (
            normalized[:7],
            normalized[7:],
        ) if normalized[:7].lower() == "bearer " else ("", normalized)
        normalized = normalized.strip()
    return "".join(normalized.split())


def _token() -> str:
    import os

    token = _normalize_token(os.environ.get("NAI_TOKEN", ""))
    if not token:
        raise RuntimeError(
            "缺少 NovelAI token：请在 MCP 客户端配置的 env 中设置 NAI_TOKEN"
        )
    return token


def _auth_headers() -> dict:
    return {
        "Authorization": f"Bearer {_token()}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    }


def generate_image(request: dict, timeout: float = 300.0) -> dict:
    """按官方 API 文档（/docs 的 OpenAPI）格式发送生成请求：

    纯 JSON body，`parameters.image` / `parameters.mask` 为 base64 内联；
    `Accept: application/json` → 响应 {"images": [{"image": <b64>, ...}]}。
    返回解析后的响应 dict。"""
    body = json.dumps(
        request, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{IMAGE_BASE_URL}{GENERATE_PATH}",
        data=body,
        headers={
            **_auth_headers(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:600].decode("utf-8", "replace")
        raise RuntimeError(f"NovelAI 请求失败 HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 NovelAI: {exc.reason}") from exc


def pngs_from_response(response: dict | bytes) -> list[bytes]:
    """解析生成/放大响应：JSON images 数组（或旧版 ZIP 兜底）→ PNG 字节列表。"""
    images: list[bytes] = []
    if isinstance(response, dict):
        for item in response.get("images") or []:
            payload = item["image"] if isinstance(item, dict) else item
            images.append(base64.b64decode(payload))
    else:  # 旧版兜底：ZIP 字节
        with zipfile.ZipFile(io.BytesIO(response)) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".png"):
                    images.append(zf.read(name))
    if not images:
        raise RuntimeError("响应中没有 PNG 图片")
    return images


def read_png_size(png_bytes: bytes) -> tuple[int, int]:
    """从 PNG IHDR 读取宽高；非 PNG 抛错。"""
    if png_bytes[:8] != b"\x89PNG\r\n\x1a\n" or len(png_bytes) < 26:
        raise ValueError("源图必须是 PNG（其他格式请先转换）")
    width, height = struct.unpack(">II", png_bytes[16:24])
    return int(width), int(height)


def upscale_image(png_bytes: bytes, timeout: float = 300.0) -> dict:
    """云端超分（官方 OpenAPI）：纯 JSON body，image 为 base64 内联；
    `Accept: application/json` → {"images": [...]}。404/405 时回退主 API
    旧版 JSON（{image, scale, width, height} → ZIP）。"""
    width, height = read_png_size(png_bytes)
    body = json.dumps(
        {
            "image": base64.b64encode(png_bytes).decode(),
            "model": UPSCALE_MODEL,
            "declared_blur_sigma": 0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{IMAGE_BASE_URL}{UPSCALE_PATH}",
        data=body,
        headers={
            **_auth_headers(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code not in (404, 405):
            detail = exc.read()[:600].decode("utf-8", "replace")
            raise RuntimeError(f"NovelAI 放大失败 HTTP {exc.code}: {detail}") from None
    # 旧版格式：主 API 域
    json_body = json.dumps(
        {
            "image": base64.b64encode(png_bytes).decode(),
            "scale": 2,
            "width": width,
            "height": height,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{MAIN_BASE_URL}{UPSCALE_PATH}",
        data=json_body,
        headers={
            **_auth_headers(),
            "Content-Type": "application/json",
            "Accept": "application/x-zip-compressed",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:600].decode("utf-8", "replace")
        raise RuntimeError(f"NovelAI 放大失败 HTTP {exc.code}: {detail}") from None


_SUB_CACHE: dict[str, object] | None = None


def subscription_failure() -> str | None:
    """返回最近一次订阅查询失败的原因（无失败时为 None）。"""
    return None if _SUB_CACHE is None else _SUB_CACHE.get("reason")


def _fetch_subscription(
    url: str, token: str
) -> tuple[bool | None, str | None, dict | None]:
    req = urllib.request.Request(
        url,
        headers={**_auth_headers(), "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        tier = data.get("tier", 0) or 0
        active = data.get("active", tier > 0)
        usage = data.get("usage")
        # usage：Opus 免费生成配额（V5 起，随时间回充；Launcher OpusUsageInfo）
        return (
            bool(active and tier == 3),
            None,
            usage if isinstance(usage, dict) else None,
        )
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}: {exc.read()[:300].decode('utf-8', 'replace')}", None
    except Exception as exc:  # 网络/解析问题
        return None, f"{type(exc).__name__}: {exc}"[:80], None


def fetch_subscription() -> tuple[bool | None, dict | None]:
    """查询订阅：返回 (is_opus, V5 配额 usage 对象)，失败返回 (None, None)。

    官方 user 端点由 image host 提供（Launcher NaiApiEndpointConfig.userUrl
    的注释与 NAI 400 文案 "update to the image URL" 一致）；旧 main host
    仅作为第三方/旧部署的回退。
    """
    global _SUB_CACHE
    try:
        token = _token()
    except RuntimeError:
        return None, None
    if _SUB_CACHE is not None and _SUB_CACHE.get("token") == token:
        return _SUB_CACHE.get("is_opus"), _SUB_CACHE.get("usage")
    is_opus: bool | None
    reason: str | None = None
    usage: dict | None = None

    is_opus, reason, usage = _fetch_subscription(
        f"{IMAGE_BASE_URL}{USER_SUBSCRIPTION_PATH}", token
    )
    if is_opus is None and reason and (
        f"HTTP {404}" in reason or f"HTTP {405}" in reason
    ):
        is_opus, reason, usage = _fetch_subscription(
            f"{MAIN_BASE_URL}{USER_SUBSCRIPTION_PATH}", token
        )
    _SUB_CACHE = {
        "token": token,
        "is_opus": is_opus,
        "reason": reason,
        "usage": usage,
    }
    return is_opus, usage


def fetch_is_opus() -> bool | None:
    """查询订阅档位判断是否 Opus（tier==3），失败返回 None。"""
    return fetch_subscription()[0]
