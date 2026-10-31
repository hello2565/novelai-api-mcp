"""Anlas 消耗估算（移植 Launcher AnlasCalculator，现代模型公式）。"""

import math

# 现代（V3 起）公式系数
_ANLAS_AREA = 2.951823174884865e-6
_ANLAS_STEP_AREA = 5.753298233447344e-7
_MAX_PER_SAMPLE = 140
_OPUS_FREE_PIXELS = 1024 * 1024
# 官方 Character Reference（精准参考）单价：每张参考每样本 5 Anlas
# （Launcher AnlasCalculator._preciseReferenceCost），不参与任何免档抵扣
_PRECISE_REF_COST = 5

# 云端超分输入面积计费分档；Opus 输入 ≤ 640×640 免费
_UPSCALE_FREE_INPUT = 640 * 640
_UPSCALE_TIERS = (
    (1048576, 1),
    (1747627, 2),
    (2446678, 3),
    (3145728, 4),
)


def estimate_generation_cost(
    *,
    width: int,
    height: int,
    steps: int,
    n_samples: int,
    strength: float = 1.0,
    is_opus: bool | None = None,
    model: str | None = None,
    precise_ref_count: int = 0,
) -> str:
    """估算一次 generate-image 请求的 Anlas 消耗，返回面向用户的文本。

    Launcher AnlasCalculator 现代公式：面积/步数部分先取整，再乘模型倍率
    （V5 = 1.5），最后按重绘强度取整。V5 的 Opus 免费额度受官方配额池限制。
    每张精准参考按官方口径加收 `_PRECISE_REF_COST`（5）Anlas/样本，
    该费用不在 Opus 免费档抵扣范围内（含免费首张）。
    """
    pixels = width * height
    multiplier = 1.5 if model and model.startswith("nai-diffusion-5") else 1.0
    per_sample = max(
        math.ceil(_ANLAS_AREA * pixels + _ANLAS_STEP_AREA * pixels * steps),
        2,
    ) * multiplier
    quota_note = (
        "；V5 Opus 免费额度受官方配额池限制，超额后计费"
        if is_opus is True and multiplier > 1.0
        else ""
    )
    ref_fee = _PRECISE_REF_COST * max(precise_ref_count, 0)
    if per_sample > _MAX_PER_SAMPLE:
        return (
            f"预计每张 ≥{_MAX_PER_SAMPLE} Anlas：超出单张可估上限，"
            "不作合计估计（服务端计费为准）"
        )
    cost = max(math.ceil(per_sample * max(strength, 0.0)), 2)
    free = (
        is_opus is True
        and steps <= 28
        and pixels <= _OPUS_FREE_PIXELS
    )
    # Opus 免费档只免首张图像样本费，参考图附加费每张样本照收
    total = (
        cost * max(n_samples - (1 if free else 0), 0)
        + ref_fee * n_samples
    )
    note = (
        "Opus 免费档"
        if free
        else "订阅查询失败，按非 Opus 估算"
        if is_opus is None
        else "非 Opus"
    )
    ref_note = (
        f"；精准参考 {precise_ref_count} 张：每样本 +{ref_fee}"
        if precise_ref_count
        else ""
    )
    return f"预计 Anlas：{total}（{note}{quota_note}{ref_note}；服务端计费为准)"


def estimate_upscale_cost(input_pixels: int, is_opus: bool | None = None) -> str:
    if input_pixels > 3145728:
        return "输入面积超过官网最高计费分档，无法估计；服务端计费为准"
    if is_opus is True and input_pixels <= _UPSCALE_FREE_INPUT:
        cost = 0
    else:
        cost = next(c for max_p, c in _UPSCALE_TIERS if input_pixels <= max_p)
    note = (
        "Opus 免费档"
        if cost == 0 and is_opus is True
        else "订阅查询失败，按非 Opus 估算"
        if is_opus is None
        else "非 Opus"
    )
    return f"预计 Anlas：{cost}（{note}；服务端计费为准）"


# V5 配额池口径（Launcher OpusUsageChip）：1% ≈ 17.3 张免费图；
# 面积扣份档位与 upscale 计费分档数值一致（≤1MP 1 份，最大 4 份/张）
_QUOTA_IMAGES_PER_PERCENT = 17.3

_V5_PREFIX = "nai-diffusion-5"


def _quota_units_for_area(area: int) -> int:
    return next(c for max_p, c in _UPSCALE_TIERS if area <= max_p)


def v5_usage_view(
    model: str | None,
    usage: dict | None,
    *,
    is_opus: bool | None,
    area: int | None = None,
) -> str:
    """V5 的 Opus 免费配额用量文本（源：/user/subscription 响应的 usage 字段）。

    usage: {percent: 剩余配额百分比（回充后可 >100）、
            isNegative: 配额是否透支、
            timeUntilNextPercent: 距下一次 1% 回充的秒数（官网端不展示，略）}。
    非 V5 模型或非 Opus 订阅返回 ""（配额仅 V5 且 Opus 享有，非 Opus 提示
    已由估价文本承担）。
    """
    if not model or not model.startswith(_V5_PREFIX) or is_opus is not True:
        return ""
    if not isinstance(usage, dict):
        return "V5 配额：订阅响应未携带 usage 字段，无法查看用量"
    try:
        percent = float(usage.get("percent", 0) or 0)
    except (TypeError, ValueError):
        percent = 0.0
    exhausted = usage.get("isNegative") is True or percent < 0
    if exhausted:
        return "V5 配额：已透支（V5 生成将按正常价扣 Anlas；配额随时间自动回充，达标后恢复免费）"
    percent = percent if percent > 0 else 0.0
    # 服务端允许回充后超过 100%，文本保留真实值（Launcher 同口径）
    shown = f"{percent:g}%"
    if area:
        units = _quota_units_for_area(area)
        images = round(percent * _QUOTA_IMAGES_PER_PERCENT / units)
        return (
            f"V5 配额：剩余 {shown}，约可免费生成 {images} 张 1MP 图"
            f"（1% ≈ {_QUOTA_IMAGES_PER_PERCENT:g} 张口径，当前尺寸按 {units} 份/张折算；"
            "配额随时间自动回充，超额后 V5 按正常价扣 Anlas）"
        )
    return (
        f"V5 配额：剩余 {shown}（1% ≈ {_QUOTA_IMAGES_PER_PERCENT:g} 张 1MP 免费图口径；"
        "配额随时间自动回充，超额后 V5 按正常价扣 Anlas）"
    )
