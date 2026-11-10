# nai-mcp — NovelAI 生图 API MCP Server

[English](README.md) | **简体中文**

独立的 stdio MCP Server，可接入 Claude Code / Codex / Cline 等任何支持 MCP 的 agent。请求格式对齐官方 OpenAPI 文档（`image.novelai.net/docs`）。生图始终走「**命名工作流**」：set 类工具只填参数不花钱，确认后再出图。

## 工作流思想：为什么这么设计

MCP agent 直连生图 API 有三个固有风险：**误触发生成烧钱**、**参数试错烧钱**、**看图被画面带偏跑离用户目标**。本工具的全部设计都围绕这三个风险展开：

### 1. 生成 = 两阶段：免费写参数 → 唯一计费闸口

生图被强拆成两段。`set_prompt` / `set_gen_param` / `set_characters` / `set_image` / `set_mask` / `set_precise_reference` 这类 **set 工具只写参数，不触发任何请求、不消耗 Anlas**，且都是**部分更新**（几次调用可以任意顺序拼装，重复写同一项幂等）。真正碰到钱的只有 `generate_image` / `enhance_image` / `upscale_image` 三把执行工具——它们是唯一闸口，每次试探参数的代价为零，试多少次都不扣费。

### 2. 工作流是具名的 JSON 文件，不是内存状态

每个工作流是 `workflows/<名称>.json` 一个可读的文件：参数可 diff、可手工修、**跨会话不丢**。agent 失忆重启后，`read_workflow` 读出来的仍是上次的完整现场；同名工作流还可承载多个并行创作线（如 `主角设定` / `场景试稿`）。

### 3. 估价永远先行，出图永远确认

`read_workflow` / `estimate_anlas` 永远免费：读出完整解析后参数 + Anlas 预估（含 Opus 免费档判定、V5 配额池实时用量、精准参考加费）。约定的工作次序是 **set → 估价 → （用户确认）→ generate**，server 的 `instructions` 会把这套约定直接写进 agent 的工具说明，引导 agent 按此流程执行。

### 4. 默认一切以 0 Anlas 为目标

server 指令的硬约定：默认模型 `nai-diffusion-5-full`、默认参数组合对齐 Opus 免费档（steps ≤ 28、≤ 1024×1024 首张免费）；`n_samples` 默认 1；未获用户明确同意 不选任何会产生消耗的参数或工具。估价文本同时标注「服务端计费为准」。

### 5. 不要求 agent 会图像编辑

inpaint 的最大摩擦点本来是「agent 画不出来蒙版」。本工具提供 `rects`（0-1 归一矩形列表）入口，server 自行渲染蒙版并按官网口径做 **8px latent 网格量化 → 硬边请求 → 软蒙版羽化 → 结果合成回原图** 的完整管线。agent 只需描述「把左上角 40% 宽的区域重绘」，不需要任何图像编辑能力。

### 6. 提示词先规范化，再生成

写提示词前约定先过 `search_tag`：中文/别名/语义 → 标准 booru tag（`looking at viewer` 而不是自然语言句子）。命中率与服务质量直接由 tag 词库决定，而不是 agent 词汇量。

### 7. 口径全部对齐官方与参考实现，不拍脑袋

Anlas 公式、采样预设、UC 预设文案、inpainting 变体映射、精准参考黑底画布规格（原始尺寸会 HTTP 500）、V5 配额池语义——全部一致对齐官方 OpenAPI 与参考客户端 [Aaalice_NAI_Launcher](https://github.com/Aaalice233/Aaalice_NAI_Launcher) 的公开行为，并在源码注释中标注出处。口径变了只改一个模块，server 内自洽。

## 快速开始

```powershell
cd nai_mcp
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
# 可选：构建 tag 词库（自动从 GitHub 下载 Launcher 发布的 tag_catalog.db）
.venv/Scripts/python tool/build_tag_vector_db.py --db tag_vectors.db --with-ffdkj
```

在客户端注册 stdio server，token 走环境变量：

```json
{
  "mcpServers": {
    "nai-mcp": {
      "command": "C:/Users/你的用户名/codes/nai_mcp/.venv/Scripts/python.exe",
      "args": ["C:/Users/你的用户名/codes/nai_mcp/server.py"],
      "env": { "NAI_TOKEN": "pst-你的持久令牌" }
    }
  }
}
```

`NAI_TOKEN`：NovelAI 持久令牌（`pst-` 开头，账户设置 → API 令牌创建），或短期用户访问令牌。

> Agent 全局约定（已写入 server `instructions`）：默认模型为 **nai-diffusion-5-full**；出图默认以 **0 Anlas**（Opus 免费档：steps ≤ 28、≤ 1024×1024）为目标；写提示词前先用 `search_tag` 规范化，角色/主体优先用 `category=character` 的角色 tag。

## 工具一览

| 工具 | 作用 | Anlas |
| --- | --- | --- |
| `list_workflows()` | 列出全部工作流名 | 否 |
| `create_workflow(name, model?)` | 创建空工作流；`model` 预填模型并写入其采样预设，重复创建报错 | 否 |
| `delete_workflow(name)` | 删除单个工作流（永久删除）。**仅当用户明确指明删除该工作流时才调用** | 否 |
| `delete_all_workflows()` | 删除全部工作流（永久删除）。**仅当用户明确要求清空工作流时才调用** | 否 |
| `clear_saved_images()` | 清空已保存图片目录下全部 PNG（默认 `nai_mcp/images/`，env `NAI_OUTPUT_DIR` 可改）。**仅当用户明确要求清空图片时才调用** | 否 |
| `set_prompt(name, prompt, uc?)` | 写正面提示词（`uc` 可选负面词） | 否 |
| `set_gen_param(name, ...)` | 写生成参数（模型/尺寸/步数/引导/采样器/种子/数量/uc 预设/cfg_rescale/噪声调度/use_coords），部分更新 | 否 |
| `set_characters(name, characters)` | V4+ 多角色提示词，每角色可带 `uc` 与位置 `center:[x,y]`（0-1，省略用官方默认布局）；`[]` 清空 | 否 |
| `set_image(name, image_path, strength?, noise?)` | img2img 底图（本地 PNG），`"clear"` 恢复文生图 | 否 |
| `set_mask(name, mask_path?, rects?, strength?, noise?)` | inpaint 蒙版：矩形 `[[x,y,w,h]]`（0-1 归一）或蒙版 PNG（白=重绘）；`"clear"` 清除 | 否 |
| `set_precise_reference(name, image_path, type?, strength?, fidelity?)` | 精准参考 / Character Reference：参考 PNG 锁定角色形象或画风；多次调用追加多张，`image_path="clear"` 清空；仅 V4.5 Curated / V4.5 Full 支持 | 否 |
| `read_workflow(name)` | 读取工作流 + Anlas 预估，供确认 | 否 |
| `estimate_anlas(name)` | 只估 Anlas 不生成；V5 工作流附带 Opus 免费配额实时用量 | 否 |
| `list_gen_options()` | 全部合法取值与各模型默认/约束 | 否 |
| `generate_image(name)` | 按工作流出图：有底图→img2img，有角色→V4 多角色结构 | **是** |
| `enhance_image(name, level?, factor?)` | 增强重绘并放大（level 1-5 默认 3；factor 2.0/1.5/1.0 超上限自动回落） | 是 |
| `upscale_image(image_path)` | 云端超分 `/ai/upscale`（v2 接口，服务端不认时自动回退旧版） | ≤4（Opus ≤640×640 免费） |
| `search_tag(query, limit?, use_vector?)` | 标准 booru tag 检索（中文/别名/词面；`use_vector=True` 走语义） | 否 |

工作流存储为 `nai_mcp/workflows/<name>.json`（env `NAI_MCP_WORKFLOW_DIR` 可改目录）。set 类工具只做部分更新，可任意顺序组合再生成。

危险操作（删除工作流、删图片）的工具描述里写死了「仅当用户明确要求时才可调用」，把不可逆操作交给用户决定。

## 模型与采样预设

支持模型（默认加粗）：**nai-diffusion-5-full**、nai-diffusion-5-curated、nai-diffusion-5-full-medium、nai-diffusion-4-5-full、nai-diffusion-4-5-curated、nai-diffusion-4-full、nai-diffusion-3

- 预设（来自 Launcher `ModelCapabilities`）：步数 28（V5 Medium 固定 14）；引导系数 V5 4.0 / V4.5 5.0 / V4 5.5 / V3 5.0；采样器 `k_euler_ancestral`；噪声调度 karras；UC 预设 heavy；img2img strength 0.7 / noise 0.0
- V5 特有：UC `light` 为 V5 专属文案；`nai-diffusion-5-full-medium` 不发送 `cfg_rescale`；Anlas 模型倍率 1.5，且 Opus 免费额度改为**官方配额池**（见下文计费节；估价文本自动提示）
- 官方质量词与 UC 预设文本自动补全；DDIM 仅 V3（V4+ 按官网规则回退 Euler Ancestral）
- 精准参考（请求以 `director_reference*` 字段发送）：仅 V4.5 Curated / V4.5 Full 支持（同 Launcher `supportsPreciseReference`）；类型 `character` / `style` / `character&style`（默认）；`strength` 默认 1.0 透传强度，`fidelity` 默认 1.0 以 `1.0 - fidelity` 作为 secondary 强度；固定 `normalize_reference_strength_multiple=true`。参考图在请求时自动按官网口径规范化：按最接近宽高比选 `1024×1536` / `1536×1024` / `1472×1472` 目标、等比缩放（bilinear）后居中贴黑底画布——发送原始尺寸会被服务端 HTTP 500（对齐 Launcher `ensurePngFormat`，插值算法有差异）
- UC 约定：预设（heavy/light/humanFocus/none）已自带整套官方质量负面词（lowres、worst quality 等）。自定义 `uc` 只需写想规避的**特定**缺陷/构图目标词，别重复质量词以免权重被放大

### Inpaint（局部重绘）

工作流同时有底图 + 蒙版时，`generate_image` 自动按官网 infill 规则出图：

- 模型自动切到 inpainting 变体（如 `nai-diffusion-4-5-full` → `…-inpainting`；V5 Curated 官方映射到 V4.5 Curated Inpainting），请求带 `mask` 分块，`extra_noise_seed = seed - 1`
- 蒙版二选一：`rects`（server 自行渲染归一化矩形，agent 无需图像编辑能力）或 `mask_path`（PNG 白=重绘，自动二值化 + 缩放）
- 蒙版 `strength` 默认 1.0（Launcher `inpaintStrength`）；<1 时按官网规则附带 `img2img` 复用原图潜空间。底图 `strength`（默认 0.7）控制整体变化强度，两者独立
- Enhance 与蒙版互斥（增强是全图重绘），设置蒙版时调用会明确报错
- 蒙版请求链路对齐官网前端：二值化 → latent 8px 量化（请求蒙版是硬边）→ 羽化软蒙版仅用于生成结果合成回原图

## 计费与 Anlas 估价

移植 Launcher `AnlasCalculator`（现代模型公式）。估价工具自动查询订阅档位（`GET /user/subscription`），文本均标注"服务端计费为准"：

- 文生图：Opus 且 steps ≤ 28、≤ 1024×1024 时首张免费
- 云端超分：Opus 输入 ≤ 640×640 免费，否则按输入面积 1-4 Anlas 分档
- 订阅查询失败按非 Opus 估算并注明
- 精准参考：每张参考 × 每样本 +5 Anlas（不参与 Opus 免档抵扣）

### V5 配额用量查看（read_workflow / estimate_anlas）

V5 的 Opus 免费额度是**配额池**而非无条件免费：随时间自动回充、额度与图面积挂钩（官网口径 1% ≈ 17.3 张 ≤1MP 免费图；大图按 2/3/4 份/张折算），透支后 V5 按正常价扣 Anlas。估价文本附加实时用量行（对齐 Launcher `OpusUsageChip`）：

- 正常：`V5 配额：剩余 97.3%，约可免费生成 1683 张 1MP 图（…配额随时间自动回充…）`
- 透支：`V5 配额：已透支（V5 生成将按正常价扣 Anlas；配额随时间自动回充，达标后恢复免费）`
- 数据源：`/user/subscription` 的 `usage` 字段（`percent` / `isNegative` / `timeUntilNextPercent`）；回充超过 100% 保留真实值；非 Opus 或查询失败时不附加

## tag 词库与语义检索

词库由 `tool/build_tag_vector_db.py` 生成（优先级同 Launcher：手工修订中文 > ffdkj 全量中文）：

- **tag_catalog.db**：tag 名单、别名（37,790 条）与手工修订中文。默认**自动获取**：读 Launcher 仓库 main 分支的 `assets/databases/manifest.json`，下载其官方 release 资产并按清单锁定 SHA-256/大小校验，缓存在 `tool/.tmp/tag_catalog.db`（该文件在仓库内是 LFS，单纯 raw 下载会拿到 pointer，故走 release）。`--source <路径>` 可跳过网络用本地文件
- **ffdkj 中文翻译表**（`--with-ffdkj`，推荐）：全量 32 万条中文。按 Launcher `zh_dictionary_download.dart` 锁定的 URL 下载、按固定 SHA-256 校验，缓存为 `tool/.tmp/ffdkj_tag.sqlite`

server 通过 env `NAI_TAG_DB` 指词库（默认项目根 `tag_vectors.db`）。

```powershell
# 生成词库
.venv/Scripts/python tool/build_tag_vector_db.py --db tag_vectors.db --with-ffdkj

# 填充 embeddings（启用语义检索；不带参数默认填全部未填充行，可分批）
$env:NAI_EMBEDDING_URL = "https://你的-embeddings-端点"
$env:NAI_EMBEDDING_MODEL = "text-embedding-xxx"
$env:NAI_EMBEDDING_API_KEY = "..."   # 可选
.venv/Scripts/python tool/fill_tag_vectors.py
```

`fill_tag_vectors.py` 性能：请求打满模型 token 上限 + 并发。默认按 32k 输入模型：`NAI_EMBED_BATCH_TOKENS`=24000、`NAI_EMBED_CHARS_PER_TOKEN`=1.0、`NAI_EMBED_CONCURRENCY`=4；脚本参数 `--min-post-count` / `--max-rows` 可先填高频 tag；429/5xx 指数退避，响应带 `index` 时按 index 回填。参考量级：22 万行 ≈ 200 个请求、4 并发几十分钟。`tool/mock_embeddings_server.py` 可离线验证链路。

词向量填充前，`search_tag` 为中文/别名/词面检索（约 0.1s）；配置 embeddings 后自动启用余弦语义（如"面向观众" → `looking at viewer`，向量 numpy float32 矩阵单次乘，查询耗时主要是 embeddings 网络延迟 1-4s）。未装 numpy 自动回退纯 Python。

## 环境变量

| 变量 | 用途 | 缺省 |
| --- | --- | --- |
| `NAI_TOKEN` | NovelAI 令牌（必需） | — |
| `NAI_MCP_WORKFLOW_DIR` | 工作流目录 | `nai_mcp/workflows/` |
| `NAI_OUTPUT_DIR` | 出图保存目录 | `nai_mcp/images/` |
| `NAI_TAG_DB` | tag 词库路径 | 项目根 `tag_vectors.db` |
| `NAI_EMBEDDING_URL` / `NAI_EMBEDDING_MODEL` / `NAI_EMBEDDING_API_KEY` | embeddings 端点（server 与填充脚本共用） | — |
| `NAI_EMBED_BATCH_TOKENS` / `NAI_EMBED_CHARS_PER_TOKEN` / `NAI_EMBED_CONCURRENCY` | 填充性能调优 | 24000 / 1.0 / 4 |

## 验证

`.venv/Scripts/python smoke_test.py` — 全离线：stdio 握手、18 个工具、工作流创建/读写/删除、Anlas 估价与 V5 配额格式化、角色/底图/蒙版/精准参考设置、tag 检索、错误路径（不触网、不扣费）。

## 参考实现：Aaalice_NAI_Launcher

本项目大量实现取自参考项目 [Aaalice233/Aaalice_NAI_Launcher](https://github.com/Aaalice233/Aaalice_NAI_Launcher)（Flutter 桌面客户端），按其内部实现的公开行为移植并复刻口径：

- Anlas 计费公式与 Opus 免费判断 —— `AnlasCalculator` 现代公式、云端超分分档
- 模型采样预设、UC 质量词/预设文本、inpainting 变体映射 —— `ModelCapabilities` / `ImageModels`
- V5 Opus 配额池口径与展示（1% ≈ 17.3 张、透支语义、回充） —— `OpusUsageInfo` / `/user/subscription` 解析 / `OpusUsageChip`
- tag 词库与中文翻译来源、锁定哈希校验 —— `tool/tag_catalog/` 与 `zh_dictionary_download.dart`

## License

[MIT](LICENSE)
