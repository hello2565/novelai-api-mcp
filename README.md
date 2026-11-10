# nai-mcp — NovelAI Image Generation MCP Server

**English** | [简体中文](README.zh-CN.md)

A standalone stdio MCP Server that plugs into any MCP-capable agent — Claude Code, Codex, Cline, and friends. Request formats follow the official OpenAPI docs (`image.novelai.net/docs`). Image generation always goes through a **named workflow**: `set_*` tools only fill in parameters at zero cost; generation happens only after explicit confirmation.

## Design philosophy: why named workflows

Pointing an agent straight at an image-generation API brings three built-in risks: **accidentally triggering a billed generation**, **burning money on parameter trial-and-error**, and **getting dragged off-task by the images it looks at**. Every design decision in this tool addresses one of these:

### 1. Generation = two phases: free parameter writes → the sole billing gate

Generating is deliberately split in two. The `set_prompt` / `set_gen_param` / `set_characters` / `set_image` / `set_mask` / `set_precise_reference` family are **write-only tools: they touch no API and consume no Anlas**, and they are all **partial updates** (calls compose in any order; rewriting the same field is idempotent). Only three executor tools — `generate_image` / `enhance_image` / `upscale_image` — ever spend money. They are the single gate: probing parameters costs nothing, no matter how many tries.

### 2. Workflows are named JSON files, not server memory

Each workflow is a readable file `workflows/<name>.json`: diffable, hand-editable, and **survives across sessions**. After an agent restarts and forgets, `read_workflow` reconstructs the complete previous state; parallel creative lines live in separate named workflows (e.g. `protagonist` / `sketch_runs`).

### 3. Estimate first, generate only after confirmation

`read_workflow` / `estimate_anlas` are always free: they return the fully resolved parameters plus an Anlas estimate (with Opus free-tier judgment, live V5 quota pool usage, and the precise-reference surcharge). The agreed order of operations is **set → estimate → (user confirms) → generate**, and the server's `instructions` field encodes that protocol directly into agent tool guidance.

### 4. Default everything to 0 Anlas

Hard rules baked into the server instructions: default model `nai-diffusion-5-full`; the default parameter combination aligns with the Opus free tier (first image free at steps ≤ 28 and ≤ 1024×1024); `n_samples` defaults to 1; and no cost-producing parameter or tool is chosen **without the user's explicit request**. Every estimate also says "subject to server-side billing".

### 5. The agent never needs image-editing skills

The biggest friction of inpainting is that agents can't paint masks. `set_mask` accepts `rects` — a list of 0–1-normalized rectangles — and the server renders the mask itself, then runs the full official frontend pipeline: **8px-latent-grid quantization → hard-edged mask in the request → feathered soft mask → compositing the result back onto the source image**. The agent only needs to say "repaint the top-left region spanning 40% of the width", with zero image-editing capability.

### 6. Normalize prompts before generating

The workflow protocol says: pass every prompt through `search_tag` first — Chinese / aliases / semantic queries → proper Danbooru-style tags (`looking at viewer`, not prose). Output quality is decided by the tag vocabulary, not by the agent's vocabulary.

### 7. Everything follows official docs and the reference client — nothing invented

The Anlas formula, sampling presets, UC preset texts, inpainting model mapping, the precise-reference black-canvas spec (original sizes cause HTTP 500), and the V5 quota-pool semantics all mirror the official OpenAPI and the public behavior of the reference client [Aaalice_NAI_Launcher](https://github.com/Aaalice233/Aaalice_NAI_Launcher), with source attributions in code comments. When the upstream caliber changes, exactly one module changes too — the server stays self-consistent.

## Quick start

```powershell
cd nai_mcp
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
# Optional: build the tag database (auto-downloads the Launcher's published tag_catalog.db)
.venv/Scripts/python tool/build_tag_vector_db.py --db tag_vectors.db --with-ffdkj
```

Register a stdio server in your MCP client, token via env:

```json
{
  "mcpServers": {
    "nai-mcp": {
      "command": "C:/Users/your-name/codes/nai_mcp/.venv/Scripts/python.exe",
      "args": ["C:/Users/your-name/codes/nai_mcp/server.py"],
      "env": { "NAI_TOKEN": "pst-your-persistent-token" }
    }
  }
}
```

`NAI_TOKEN`: a NovelAI persistent token (starts with `pst-`; account settings → API tokens → create), or a short-lived user access token.

> Global agent conventions (written into the server `instructions`): default model is **nai-diffusion-5-full**; aim for **0 Anlas** by default (Opus free tier: steps ≤ 28, ≤ 1024×1024); normalize prompts via `search_tag` before writing; prefer `category=character` tags for characters/subjects.

## Tool overview

| Tool | Purpose | Anlas |
| --- | --- | --- |
| `list_workflows()` | List all workflow names | No |
| `create_workflow(name, model?)` | Create an empty workflow; `model` preselects the model and writes its sampling presets; duplicate names error out | No |
| `delete_workflow(name)` | Delete one workflow (permanent). **Only call when the user explicitly names it** | No |
| `delete_all_workflows()` | Delete every workflow (permanent). **Only call when the user explicitly asks to clear all** | No |
| `clear_saved_images()` | Clear all PNGs in the output directory (default `nai_mcp/images/`, env `NAI_OUTPUT_DIR`). **Only call when the user explicitly asks** | No |
| `set_prompt(name, prompt, uc?)` | Write the positive prompt (`uc` = optional negative prompt) | No |
| `set_gen_param(name, ...)` | Write generation parameters (model/size/steps/guidance/sampler/seed/samples/UC preset/cfg_rescale/noise schedule/use_coords), partial update | No |
| `set_characters(name, characters)` | V4+ multi-character prompts; per-character `uc` and optional `center:[x,y]` (0–1; official default layout when omitted); `[]` clears | No |
| `set_image(name, image_path, strength?, noise?)` | img2img source image (local PNG); `"clear"` returns to txt2img | No |
| `set_mask(name, mask_path?, rects?, strength?, noise?)` | inpaint mask: rectangles `[[x,y,w,h]]` (0–1 normalized) or mask PNG (white = repaint); `"clear"` removes | No |
| `set_precise_reference(name, image_path, type?, strength?, fidelity?)` | Precise reference / Character Reference: a reference PNG locking a character's look or a style; multiple calls append; `image_path="clear"` empties; V4.5 Curated / V4.5 Full only | No |
| `read_workflow(name)` | Read a workflow + Anlas estimate, for confirmation | No |
| `estimate_anlas(name)` | Estimate-only, no generation; V5 workflows additionally show live Opus quota usage | No |
| `list_gen_options()` | All legal values plus per-model defaults/constraints | No |
| `generate_image(name)` | Generate per the workflow: source image → img2img; characters set → V4 multi-character structure | **Yes** |
| `enhance_image(name, level?, factor?)` | Enhance (full repaint + upscale); level 1–5, default 3; factor 2.0/1.5/1.0 auto-falls-back under the pixel cap | Yes |
| `upscale_image(image_path)` | Cloud upscaling via `/ai/upscale` (v2 endpoint, falls back to the legacy version when the server rejects it) | ≤4 (free ≤640×640 on Opus) |
| `search_tag(query, limit?, use_vector?)` | Danbooru tag search (Chinese/aliases/literal; `use_vector=True` for semantic) | No |

Workflows are stored as `nai_mcp/workflows/<name>.json` (directory overridable via env `NAI_MCP_WORKFLOW_DIR`). `set_*` tools only do partial updates; compose them in any order, then generate.

Dangerous tools (workflow deletion, image clearing) hard-code "**only call when the user explicitly asks**" in their descriptions, leaving irreversible actions to the user.

## Models & sampling presets

Supported models (default in bold): **nai-diffusion-5-full**, nai-diffusion-5-curated, nai-diffusion-5-full-medium, nai-diffusion-4-5-full, nai-diffusion-4-5-curated, nai-diffusion-4-full, nai-diffusion-3

- Presets (from Launcher `ModelCapabilities`): 28 steps (V5 Medium fixed at 14); guidance V5 4.0 / V4.5 5.0 / V4 5.5 / V3 5.0; sampler `k_euler_ancestral`; noise schedule karras; UC preset heavy; img2img strength 0.7 / noise 0.0
- V5 specifics: UC `light` uses V5-exclusive copy; `nai-diffusion-5-full-medium` does not send `cfg_rescale`; Anlas model multiplier 1.5, and the Opus free tier becomes an **official quota pool** (see the Billing section; estimate text notes it automatically)
- Official quality tags and UC preset text are completed automatically; DDIM is V3-only (V4+ falls back to Euler Ancestral per the official rules)
- Precise reference (sent as `director_reference*` fields): V4.5 Curated / V4.5 Full only (matching Launcher `supportsPreciseReference`); types `character` / `style` / `character&style` (default); `strength` defaults to 1.0 (passed through as strength), `fidelity` defaults to 1.0 (sent as `1.0 - fidelity` secondary strength); `normalize_reference_strength_multiple=true` always. Reference images are auto-normalized per the official spec at request time: pick the closest-aspect target among `1024×1536` / `1536×1024` / `1472×1472`, scale to fit (bilinear), and center on a black canvas — raw-size images get HTTP 500 (matching Launcher `ensurePngFormat`; interpolation differs)
- UC convention: presets (heavy/light/humanFocus/none) already carry the full official quality negative list (lowres, worst quality, …). A custom `uc` should list only the **specific** defects/composition issues to avoid — repeating quality words would amplify their relative weight

### Inpaint (partial repaint)

When a workflow has both a source image and a mask, `generate_image` follows the official infill rules automatically:

- The model switches to the inpainting variant (e.g. `nai-diffusion-4-5-full` → `…-inpainting`; V5 Curated maps officially to V4.5 Curated Inpainting); the request carries the `mask` field and `extra_noise_seed = seed - 1`
- Two mask sources: `rects` (the server renders the normalized rectangles — the agent needs no image-editing skills) or `mask_path` (a PNG where white = repaint; auto-binarized and resized)
- Mask `strength` defaults to 1.0 (Launcher `inpaintStrength`); below 1.0 the request attaches an `img2img` block reusing the original latent per official rules. Source-image `strength` (default 0.7) controls overall change; the two are independent
- Enhance and masks are mutually exclusive (enhance repaints the whole image); calling with a mask set errors out explicitly
- The mask pipeline mirrors the official frontend: binarize → latent 8px quantization (the request mask has hard edges) → feathered soft mask, only used for compositing results back afterwards

## Billing & Anlas estimation

Ported from the Launcher `AnlasCalculator` (modern-model formula). Estimation tools query the subscription tier automatically (`GET /user/subscription`); every estimate notes "subject to server-side billing":

- Generation: first image free on Opus at steps ≤ 28 and ≤ 1024×1024
- Cloud upscaling: free ≤640×640 input on Opus, otherwise 1–4 Anlas by input-area tiers
- When the subscription query fails, estimating assumes non-Opus and says so
- Precise references: +5 Anlas per reference per sample (not covered by the Opus free tier)

### V5 quota usage (read_workflow / estimate_anlas)

On V5 the Opus free tier is a **quota pool**, not unconditional freeness: it refills over time and is tied to image area (official caliber: 1% ≈ 17.3 free ≤1MP images; larger images count as 2/3/4 units each), and an overdrawn pool bills V5 generations at normal Anlas prices. Estimate text appends a live usage line (matching Launcher `OpusUsageChip`):

- Normal: `V5 quota: 97.3% left, ~1683 free 1MP images (…quota refills over time…)`
- Overdrawn: `V5 quota: overdrawn (V5 generation bills Anlas at normal price; quota refills over time and restores freeness once back above zero)`
- Source: the `usage` field of `/user/subscription` (`percent` / `isNegative` / `timeUntilNextPercent`); refills above 100% are shown as-is; not appended for non-Opus or on query failure

## Tag database & semantic search

The vocabulary is built by `tool/build_tag_vector_db.py` (priority matches the Launcher: hand-edited Chinese > full ffdkj Chinese):

- **tag_catalog.db**: tag list, aliases (37,790 entries), and hand-edited Chinese. Auto-fetched **by default**: reads the Launcher repo's `assets/databases/manifest.json` on the main branch, downloads the official release asset, and verifies it against the manifest's SHA-256/size, cached at `tool/.tmp/tag_catalog.db` (that file is LFS inside the repo, so raw downloads only fetch a pointer — hence going through release). Pass `--source <path>` to use a local file offline
- **ffdkj Chinese translation table** (`--with-ffdkj`, recommended): full 320k-entry Chinese set. Downloads the URL pinned in Launcher `zh_dictionary_download.dart`, verified against a fixed SHA-256, cached as `tool/.tmp/ffdkj_tag.sqlite`

The server picks the database via env `NAI_TAG_DB` (default `tag_vectors.db` in the project root).

```powershell
# Build the vocabulary
.venv/Scripts/python tool/build_tag_vector_db.py --db tag_vectors.db --with-ffdkj

# Fill embeddings (enables semantic search; no args fills every unfilled row, batchable)
$env:NAI_EMBEDDING_URL = "https://your-embeddings-endpoint"
$env:NAI_EMBEDDING_MODEL = "text-embedding-xxx"
$env:NAI_EMBEDDING_API_KEY = "..."   # optional
.venv/Scripts/python tool/fill_tag_vectors.py
```

`fill_tag_vectors.py` performance: requests packed to the model's token limit + concurrency. Defaults assume a 32k-input model: `NAI_EMBED_BATCH_TOKENS`=24000, `NAI_EMBED_CHARS_PER_TOKEN`=1.0, `NAI_EMBED_CONCURRENCY`=4; script flags `--min-post-count` / `--max-rows` let you fill hot tags first; 429/5xx exponential backoff; when responses carry `index`, results are written back by index. Order of magnitude: 220k rows ≈ 200 requests, tens of minutes at 4-way concurrency. `tool/mock_embeddings_server.py` validates the pipeline offline.

Before vectors are filled, `search_tag` does Chinese/alias/literal matching (~0.1s); once embeddings are configured, cosine semantic search switches on automatically (e.g. "facing the viewer" → `looking at viewer`; the vectors form a single numpy float32 matmul, so latency is dominated by the 1–4s embeddings network round-trip). Falls back to pure Python if numpy is missing.

## Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `NAI_TOKEN` | NovelAI token (required) | — |
| `NAI_MCP_WORKFLOW_DIR` | Workflow directory | `nai_mcp/workflows/` |
| `NAI_OUTPUT_DIR` | Image output directory | `nai_mcp/images/` |
| `NAI_TAG_DB` | Tag database path | project-root `tag_vectors.db` |
| `NAI_EMBEDDING_URL` / `NAI_EMBEDDING_MODEL` / `NAI_EMBEDDING_API_KEY` | embeddings endpoint (shared by server and filler script) | — |
| `NAI_EMBED_BATCH_TOKENS` / `NAI_EMBED_CHARS_PER_TOKEN` / `NAI_EMBED_CONCURRENCY` | filler performance tuning | 24000 / 1.0 / 4 |

## Verification

`.venv/Scripts/python smoke_test.py` — fully offline: stdio handshake, all 18 tools, workflow create/read/delete flow, Anlas estimation & V5 quota formatting, character/source-image/mask/precise-reference setup, tag search, and error paths (no network, no billing).

## Reference implementation: Aaalice_NAI_Launcher

Large parts of this project are ported from the reference project [Aaalice233/Aaalice_NAI_Launcher](https://github.com/Aaalice233/Aaalice_NAI_Launcher) (a Flutter desktop client), mirroring the public behavior of its internals:

- Anlas billing formula & Opus free-tier judgment — `AnlasCalculator` modern formula, cloud upscaling tiers
- Model sampling presets, UC quality tags/preset texts, inpainting variant mapping — `ModelCapabilities` / `ImageModels`
- V5 Opus quota pool caliber & display (1% ≈ 17.3 images, overdraw semantics, refill) — `OpusUsageInfo` / `/user/subscription` parsing / `OpusUsageChip`
- Tag vocabulary & Chinese translation sources, pinned-hash verification — `tool/tag_catalog/` and `zh_dictionary_download.dart`

## License

[MIT](LICENSE)
