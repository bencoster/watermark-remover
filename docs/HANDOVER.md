# Handover

Snapshot for any future session.

## What this project is

FastAPI + vanilla-JS web app for removing watermarks from images. Repo:
[github.com/bencoster/watermark-remover](https://github.com/bencoster/watermark-remover).
Runs on `localhost:8091`. See `ARCHITECTURE.md` for components.

## Test fixture

Dreamstime stock photo `dreamstime_18829755_watermarked.jpg`. **Two watermark layers**:
- semi-translucent spiral logos in a regular tiled grid (~50 instances)
- diagonal `dreamstime.com` text rendered at low alpha across each tile

Plus a solid blue bottom-strip bar with the URL + `ID 18829755 © Igor Mojzes`.

## Pipeline

```
upload → /api/auto
  ├─ library auto-match (skip detect if a user-blessed mask matches dimensions)
  ├─ classifier gate (ConvNeXt-tiny binary — clean → bail)
  ├─ body detector (one of):
  │    "off"            — ConvNeXt + Grad-CAM           IoU 0.248  (default)
  │    "grounding_dino" — GD ∩ pixel + 3×3 dilate       IoU 0.310
  │    "sam3"           — SAM 3 hybrid                  IoU 0.337  ← BEST
  ├─ strip handler:
  │    "inpaint" + "telea" — cv2.inpaint TELEA on text glyphs
  │    "inpaint" + "sdxl"  — diffusers SDXL on cropped strip region
  │    "crop"              — physically cut the bar above the strip
  └─ LaMa inpaint on body mask + (TELEA-prefilled or SDXL-prefilled) image
```

## Detector benchmark on canonical fixture

| Tool | Recall | Precision | IoU | Notes |
|---|---|---|---|---|
| ConvNeXt + Grad-CAM | 0.64 | 0.29 | 0.248 | default |
| Florence-2 (any prompt) | 0.02 | — | 0.022 | fails on tiled |
| SAM 2 | 0.16 | 0.35 | 0.123 | single-mark only |
| Grounding DINO ∩ pixel | 0.55 | 0.42 | 0.310 | no auth gate |
| **SAM 3 hybrid** | 0.51 | 0.50 | **0.337** | needs HF auth |
| Library-cached GT mask | — | — | — | **PSNR 28.29 dB** |

LaMa-with-perfect-mask ceiling on this image: PSNR 28.08 dB.

## SAM 3 specifics

Repo: `facebook/sam3` (NOT `sam3.1` — that one only ships `sam3.1_multiplex.pt`
which needs the GitHub `sam3` package's loader). Native
`transformers.Sam3Model` + `Sam3Processor` (transformers ≥ 5.0).

Two prompts in parallel:
- `"watermark."` @ thr 0.05 → ~97 tight per-instance masks (precision 0.81)
- `"tiled stock photo watermark."` @ thr 0.05 → 38 loose boxes (recall 0.99)

Combined: `precise OR (loose AND pixel_heuristic) + 3×3 dilate ×1`.

**Critical**: default `threshold=0.30` returns 0 detections — watermark concept
fires at low confidence, 0.05 is the sweet spot.

**Critical**: `target_sizes` must be `[(h, w)]` (list of tuples), not a tensor —
transformers 5.x + torch 2.10 type bug.

## Empirical finding (worth porting)

**Upscaling 2–4× in the Diff Mask tool helps detection** — the working-scale
slider (0.1×–10× log-spaced) on the Diff Mask tab makes diagonal text strokes
occupy more pixels, giving the diff cleaner edges. Not yet ported to the auto
pipeline. Expected to add 0.02–0.05 IoU on tiled-grid masks.

## Inpainters

- **LaMa** (big-lama JIT, IOPaint origin) — body inpaint, ~5 s on CPU. Always live.
- **SDXL Inpaint** (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`) —
  strip-bar refill only, gated by `is_available()` (needs diffusers + 11 GB free
  VRAM). Runs on cuda:0 (4090, 21 GB free). ~7 GB first-run download.
- **TELEA** — fallback for thin glyphs / when SDXL unavailable.
- **ProPainter** — stubbed for video, not implemented.

## Environment

The preview server runs from **`.venv-sam3`** (Python 3.12 + PyTorch 2.10 cu128
+ transformers 5.8 + diffusers 0.38). System Python 3.10 doesn't have
diffusers/transformers-5; the launch.json points at the venv. Don't switch back
— system Python is shared with BC_LocalLLM and ComfyUI.

`.venv*/` is in `.gitignore`. The 3.5 GB `sam3.1_multiplex.pt` + 2 GB
`facebook/sam3` safetensors live in the HF cache (`~/.cache/huggingface/hub/`),
not the project tree.

## Auth

HF token authed via env var (no token written to disk). Account `bcoster`.
To persist, run `hf auth login` in a terminal once. Otherwise pass
`HF_TOKEN=...` to `python` invocations.

## Library / Diff Mask flow

`/api/auto` checks for a library mask matching input dimensions FIRST. The
library was polluted earlier (every detection run was auto-saving its mask, and
Auto-match picked the newest by-dimension match regardless of source). Fixed:

- Detection no longer auto-saves
- Auto-match ranks by `(p_full ≥ 0.999, smallest coverage, newest)`.
  `p_full=1.0` is the Diff Mask save marker; detection saves use the classifier
  score (~0.97)
- `POST /api/masks/purge-auto` + Library tab "Clean auto-saved masks" button —
  removes everything that wasn't user-blessed

For repeat watermarks (same layout, same dimensions), Diff Mask + Save → next
image of that layout hits library auto-match → 28+ dB result without detection.

## UI tabs

- **Upload** — Default-load sample, click/drop to replace, batch queue with
  thumbnail strip, sequential processing, per-item result cards with **Compare**
  button (4-mode before/after viewer)
- **Diff Mask** — A vs B images (default-loaded), live diff with
  threshold/dilate/blur sliders, **0.1×–10× working-scale slider**, "Run
  Detector at this Scale" button (cyan overlay), Save/Apply/Download
- **Library** — Saved diff masks, inline rename, Use button, "Clean auto-saved"
- **Jobs** — Background video jobs (ProPainter stub)
- **Status** — GPU info, loaded model, VRAM

## Tests

`python -m pytest tests/ --ignore=tests/test_e2e_image.py` → **34/34 passing**,
~2 min on CPU. Includes regression tests for PSNR/SSIM/IoU on the canonical
fixture and robustness sweeps across image sizes + watermark alpha.

## Quick start

```bash
cd "C:\Users\Ben Coster\watermark-remover"
.venv-sam3\Scripts\activate
$env:HF_TOKEN = "<paste from secret manager>"   # for SAM 3 download
uvicorn server:app --host 127.0.0.1 --port 8091
# or via Claude preview: launch.json already points at the venv
```

Default One-Click Remove: TELEA strip + Off detector + Auto library-match →
~24 dB on first hit, ~28 dB once a library mask exists for the layout. Pick
**SAM 3 hybrid** for tiled stock-photo images on first encounter (best one-shot
IoU 0.337).
