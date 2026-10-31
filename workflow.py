"""命名工作流的持久化与解析。

目录默认在 server.py 旁的 `workflows/`（env `NAI_MCP_WORKFLOW_DIR` 可改），
每个工作流一个 `<name>.json`。文件结构：

```json
{
  "name": "...", "updated_at": "...",
  "prompt":  {"text": "...", "uc": "..."},
  "gen":     {"model": ..., "width": ..., "height": ..., ...},
  "characters": [{"prompt": ..., "uc": "...", "center": [x, y]}],
  "image":   {"path": "...", "strength": 0.7, "noise": 0.0},
  "mask":    {"source": "rects|path", "rects": [[x, y, w, h]], "strength": 1.0, "noise": 0.0}
}
```
"""

import json
import os
import re
import time
from pathlib import Path

import nai_request

WORKFLOW_DIR = Path(
    os.environ.get(
        "NAI_MCP_WORKFLOW_DIR",
        str(Path(__file__).resolve().parent / "workflows"),
    )
)
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

GEN_KEYS = (
    "model",
    "width",
    "height",
    "steps",
    "scale",
    "sampler",
    "seed",
    "n_samples",
    "uc_preset",
    "cfg_rescale",
    "noise_schedule",
    "use_coords",
    "precise_refs",
)


def sanitize_name(name: str) -> str:
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        raise ValueError(
            "工作流名称只能包含字母/数字/下划线/短横线，长度 1-64"
        )
    return name


def workflow_path(name: str) -> Path:
    return WORKFLOW_DIR / (sanitize_name(name) + ".json")


def write_patch(name: str, patch: dict) -> dict:
    """按部分更新合并写回工作流，返回合并后的文件内容。"""
    path = workflow_path(name)
    wf = {}
    if path.exists():
        wf = json.loads(path.read_text(encoding="utf-8"))
    wf.update(patch)
    wf["name"] = name
    wf["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
    return wf


def load(name: str) -> dict:
    path = workflow_path(name)
    if not path.exists():
        raise RuntimeError(f"工作流 {name} 不存在（{path}），请先用 set_* 工具创建")
    return json.loads(path.read_text(encoding="utf-8"))


def list_names() -> list[str]:
    if not WORKFLOW_DIR.exists():
        return []
    return sorted(p.stem for p in WORKFLOW_DIR.glob("*.json"))


def delete(name: str) -> bool:
    """删除工作流文件；返回是否存在过。"""
    path = workflow_path(name)
    if not path.exists():
        return False
    path.unlink()
    return True


def delete_all() -> list[str]:
    """删除目录下全部工作流，返回被删除的名称列表。"""
    if not WORKFLOW_DIR.exists():
        return []
    removed = []
    for path in sorted(WORKFLOW_DIR.glob("*.json")):
        path.unlink()
        removed.append(path.stem)
    return removed


# ---------------------------------------------------------------------------
# 出图前解析：校验 + 按模型预设补默认值
# ---------------------------------------------------------------------------


def resolve_generation(wf: dict) -> dict:
    """把工作流文件解析成一次出图所需的完整参数（文本生图与 img2img 通用）。"""
    prompt = (wf.get("prompt") or {}).get("text") or ""
    if not prompt.strip():
        raise ValueError("工作流还没有 prompt，请先调用 set_prompt")
    negative_prompt = (wf.get("prompt") or {}).get("uc") or ""

    gen = wf.get("gen") or {}

    model = gen.get("model") or nai_request.PRESET_MODEL
    if model not in nai_request.MODELS:
        raise ValueError(
            f"不支持的模型 {model}，可选：{', '.join(nai_request.MODELS)}"
        )
    profile = nai_request.MODELS[model]

    uc_preset = gen.get("uc_preset") or "heavy"
    if uc_preset not in nai_request.UC_PRESET_IDS:
        raise ValueError(f"uc_preset 只能是 {', '.join(nai_request.UC_PRESET_IDS)}")
    schedule = gen.get("noise_schedule") or "karras"
    if schedule not in nai_request.NOISE_SCHEDULES:
        raise ValueError(
            f"noise_schedule 只能是 {', '.join(nai_request.NOISE_SCHEDULES)}"
        )
    sampler_input = gen.get("sampler")
    sampler = nai_request.resolve_sampler(sampler_input, model)
    if sampler_input and sampler not in nai_request.SAMPLERS:
        raise ValueError(
            f"不支持的采样器 {sampler_input}，可选：{', '.join(nai_request.SAMPLERS)}"
        )
    if sampler_input in ("ddim", "ddim_v3") and sampler not in ("ddim", "ddim_v3"):
        raise ValueError(
            f"模型 {model} 不支持 DDIM，按官网规则回退为 {sampler}"
        )

    n_samples = int(gen.get("n_samples") or 1)
    if not (1 <= n_samples <= 4):
        raise ValueError("n_samples 取值 1-4")
    raw_steps = gen.get("steps")
    steps = max(1, min(50, int(raw_steps))) if raw_steps else profile.steps
    width, height = nai_request.sanitize_resolution(
        int(gen.get("width") or 832), int(gen.get("height") or 1216)
    )
    scale = (
        float(gen["scale"]) if gen.get("scale") is not None else profile.scale
    )
    cfg_rescale = float(gen.get("cfg_rescale") or 0.0)
    seed = int(gen.get("seed", -1))
    use_coords = bool(gen.get("use_coords"))

    # 精准参考（V4.5 特有；解析类型/强度与保真度，路径延后到出图时读取）
    precise_refs = []
    for index, ref in enumerate(gen.get("precise_refs") or []):
        if not isinstance(ref, dict) or not str(ref.get("path") or "").strip():
            raise ValueError(f"precise_refs[{index}] 缺少 path")
        ref_type = ref.get("type") or nai_request.DEFAULT_PRECISE_REF_TYPE
        if ref_type not in nai_request.PRECISE_REF_TYPES:
            raise ValueError(
                f"precise_refs[{index}].type 无效 {ref_type}，可选："
                f"{', '.join(nai_request.PRECISE_REF_TYPES)}"
            )
        precise_refs.append(
            {
                "path": os.path.expanduser(str(ref["path"])),
                "type": ref_type,
                "strength": float(
                    ref.get("strength", nai_request.DEFAULT_PRECISE_REF_STRENGTH)
                ),
                "fidelity": float(
                    ref.get("fidelity", nai_request.DEFAULT_PRECISE_REF_FIDELITY)
                ),
            }
        )

    characters = wf.get("characters") or []
    if characters and not nai_request.is_v4_model(model):
        raise ValueError(f"模型 {model} 不支持角色 Prompt，请换 V4 及以上模型")
    if characters and len(characters) > profile.max_characters:
        raise ValueError(
            f"模型 {model} 最多支持 {profile.max_characters} 个角色"
        )
    for index, character in enumerate(characters):
        if not isinstance(character, dict) or not (character.get("prompt") or "").strip():
            raise ValueError(f"characters[{index}] 缺少 prompt")
        center = character.get("center")
        if center is not None:
            if (
                not isinstance(center, (list, tuple))
                or len(center) != 2
                or not all(isinstance(v, (int, float)) for v in center)
                or not (0 <= center[0] <= 1 and 0 <= center[1] <= 1)
            ):
                raise ValueError(f"characters[{index}].center 需为 [x, y]，x/y ∈ 0-1")

    image_info = None
    raw_image = wf.get("image")
    if raw_image:
        image_path = raw_image.get("path") or ""
        if not image_path or not Path(os.path.expanduser(image_path)).is_file():
            raise RuntimeError(
                f"工作流 {wf.get('name')} 的底图不存在：{image_path or '(空)'}，"
                "请重新调用 set_image"
            )
        image_info = {
            "path": os.path.expanduser(image_path),
            "strength": float(raw_image.get("strength", nai_request.DEFAULT_STRENGTH)),
            "noise": float(raw_image.get("noise", nai_request.DEFAULT_NOISE)),
        }
        if not (0.0 <= image_info["strength"] <= 1.0 and 0.0 <= image_info["noise"] <= 1.0):
            raise ValueError("strength/noise 取值 0-1")

    mask_info = None
    if image_info:
        raw_mask = wf.get("mask")
        if raw_mask:
            mask_path = raw_mask.get("path") or ""
            rects = raw_mask.get("rects") or []
            if mask_path and not Path(os.path.expanduser(mask_path)).is_file():
                raise RuntimeError(
                    f"工作流 {wf.get('name')} 的蒙版不存在：{mask_path}，"
                    "请重新调用 set_mask"
                )
            if mask_path:
                source = "path"
                detail = os.path.expanduser(mask_path)
            elif rects:
                if not all(
                    isinstance(r, (list, tuple)) and len(r) == 4 for r in rects
                ):
                    raise ValueError("mask.rects 每项需为 [x, y, w, h]")
                source = "rects"
                detail = [list(r) for r in rects]
            else:
                raise RuntimeError(
                    "mask 配置缺少蒙版来源：提供 mask_path 或 rects"
                )
            mask_info = {
                "source": source,
                source: detail,
                "strength": float(raw_mask.get("strength", nai_request.DEFAULT_INPAINT_STRENGTH)),
                "noise": float(raw_mask.get("noise", nai_request.DEFAULT_NOISE)),
            }
            if not (0.0 <= mask_info["strength"] <= 1.0 and 0.0 <= mask_info["noise"] <= 1.0):
                raise ValueError("mask strength/noise 取值 0-1")

    return {
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt,
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
        "noise_schedule": schedule,
        "use_coords": use_coords,
        "precise_refs": precise_refs,
        "characters": characters,
        "image": image_info,
        "mask": mask_info,
    }
