# BC_WatermarkRemover — Architecture

## High-level

FastAPI server, vanilla-JS frontend, SQLite for jobs + mask library. Single-user
local tool — no auth, no multi-tenancy. Runs from a project-local Python 3.12 venv
(`.venv-sam3`) with PyTorch 2.10 cu128 + transformers 5.8 + diffusers 0.38.

```
┌────────────────────────────────────────────────────────────────────┐
│                         FastAPI server                             │
│   ┌──────────────────────┐    ┌─────────────────────────────────┐  │
│   │  POST /api/auto      │ ─► │  detect_split() → body+strip    │  │
│   │  POST /api/inpaint   │    │     (CAM | GD | SAM 3 hybrid)   │  │
│   │  POST /api/detect    │    │  prefill_strip() → TELEA/SDXL   │  │
│   │  POST /api/video     │    │  inpaint() → LaMa               │  │
│   │  GET  /api/jobs/*    │    └─────────────────────────────────┘  │
│   │  GET  /api/masks/*   │                                         │
│   │  GET  /api/sample-*  │                                         │
│   └──────────────────────┘                                         │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
              ┌───────────────────┼─────────────────────┐
              ▼                   ▼                     ▼
   ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────────┐
   │  jobs.db         │  │  masks/         │  │  HF cache            │
   │  (SQLite)        │  │  (PNG + JPEG    │  │  ~/.cache/huggingface│
   │  - jobs          │  │   thumbs)       │  │  - LaMa, SAM, GD,    │
   │  - masks         │  └─────────────────┘  │    SDXL, ConvNeXt    │
   └──────────────────┘                       └──────────────────────┘
```

## Pipeline (POST /api/auto)

```
1. Save upload to tmp/, read image
2. Library short-circuit
   ├─ if library_mask == "auto":
   │    find mask whose dimensions match input within ±5%
   │    rank: (p_full ≥ 0.999, smallest coverage, newest)
   │    if hit → LaMa(input, cached_mask) → return (X-Mask-Source: library)
   └─ else: continue
3. Classifier gate
   ConvNeXt-tiny binary classifier (boomb0om/watermark-detectors)
   if P(watermarked) < 0.30 → return 422 {"detected": false}
4. Body mask
   Grad-CAM heatmap from the classifier
   ├─ tiled_detector == "sam3": SAM 3 hybrid replaces CAM mask
   │    "watermark." prompt @ 0.05 → ~97 precise per-instance masks
   │    "tiled stock photo watermark." @ 0.05 → 38 loose boxes
   │    body = precise OR (loose AND pixel_heuristic) + 3×3 dilate
   ├─ tiled_detector == "grounding_dino": GD ∩ pixel + dilate
   └─ tiled_detector == "off": CAM blob + morphology
   detect_mode = auto | recall | precision (chooses CAM thresholding)
5. Strip mask
   _strip_text_mask() — bottom-band glyph detection
   extends to image bottom on solid-colour bars (Dreamstime blue, etc.)
6. Strip handling
   ├─ strip_mode == "crop": cut image above strip top
   ├─ strip_engine == "sdxl": SDXL Inpaint on strip region only
   │    cropped + 60 px context, 1024 px long edge, prompt "seamless ..."
   │    blends back with 3 px Gaussian feather
   └─ strip_engine == "telea": cv2.inpaint TELEA on glyph mask
7. LaMa inpaint
   model_manager.get("lama") → big-lama JIT
   pad to mod-8, normalise, forward, unpad, BGR
8. Optional SAM refinement (off by default; sam2 over-segments tiled)
9. Return PNG
```

## Components

### Services

| File | Purpose |
|---|---|
| `cuda_policy.py` | NVML-based GPU selection. `BC_WMR_DEVICE` env override. |
| `model_manager.py` | One model in VRAM at a time; LaMa ↔ SDXL swap. |
| `lama_service.py` | LaMa JIT inpaint, mod-8 pad, IOPaint-derived. |
| `watermark_classifier.py` | ConvNeXt-tiny binary, boomb0om checkpoint. |
| `watermark_localizer.py` | Grad-CAM with overlapping 256² tiles + Hann blend. |
| `detector_service.py` | Top-level `detect_split()` — gate, body, strip. |
| `lattice_completion.py` | `complete_lines()` (Hough). Used by Diff Editor. |
| `grounding_dino_service.py` | `IDEA-Research/grounding-dino-base`. |
| `sam_refine_service.py` | SAM 2 (point prompts). SAM 3.1 multiplex stub. |
| `sam3_service.py` | **SAM 3 hybrid (current best detector)** |
| `sdxl_inpaint_service.py` | SDXL strip-bar refill, cropped region only. |

### Routes

| File | Endpoints |
|---|---|
| `routes/inpaint.py` | `/api/auto`, `/api/inpaint`, `/api/sample-image`, `/api/sample-clean-image`, `_find_matching_library_mask` |
| `routes/masks.py` | List/get/file/thumb/rename/delete + `/from-diff` + `/purge-auto` + `/complete-lines` |
| `routes/detect.py` | `/api/detect` — auto-detect mask only |
| `routes/video.py` | `/api/video` — submits ProPainter job (stubbed) |
| `routes/jobs.py` | `/api/jobs/{id}`, `/api/jobs/{id}/result` |
| `routes/status.py` | GPU / model status |

### Frontend

| File | Owns |
|---|---|
| `app.js` | Panel router |
| `upload.js` | Batch queue, sample preload, click-to-browse, drop, "Compare" |
| `mask-editor.js` | Brush/eraser/undo/redo/invert manual editor |
| `library.js` | Mask grid, rename, "Clean auto-saved" |
| `diff-mask.js` | Live diff, threshold/dilate/blur sliders, **0.1×–10× working-scale slider**, "Run Detector at this Scale" overlay |
| `before-after.js` | Modal viewer (slider/SBS/toggle/diff) |
| `job-dashboard.js`, `gpu-status.js` | Self-explanatory |

## Data flow for a tiled stock-photo image

```
                                 ┌───────────────────┐
   user drops image  ──────►     │  POST /api/auto   │
                                 └─────────┬─────────┘
                                           │
            ┌──────────────────────────────┼──────────────────────────────┐
            │                              │                              │
            ▼                              ▼                              ▼
  ┌──────────────────┐      ┌──────────────────────┐      ┌─────────────────────┐
  │  Library auto-   │      │  ConvNeXt classifier │      │  SAM 3 / GD / CAM   │
  │  match by dims   │ NO ► │  P(watermarked)      │ ─►   │  → body mask        │
  │  user-blessed    │ HIT  │  > 0.30?             │      │  → strip mask       │
  │  first           │ ──►  │                      │      └──────────┬──────────┘
  └────────┬─────────┘      └──────────────────────┘                 │
           │ HIT                                                     │
           ▼                                                         │
  ┌──────────────────┐                                               │
  │  LaMa(input,     │                                               │
  │       cached)    │ ◄─────────────────────────────────────────────┘
  │  ~28 dB PSNR     │              prefill_strip(SDXL or TELEA)
  └────────┬─────────┘                       │
           │                                 ▼
           ▼                       ┌──────────────────┐
       ┌────────┐                  │  LaMa(prefilled, │
       │ result │ ◄──────────────  │       body_mask) │
       └────────┘                  └──────────────────┘
```

## Test fixture

`tests/fixtures/dreamstime_18829755_*.{jpg,png}` — Dreamstime stock photo:
- watermarked.jpg — original
- reference.jpg — watermarkremover.io clean version (used as PSNR/SSIM target)
- groundtruth_mask.png — diff mask, user-blessed (p_full=1.0, used for IoU)

Watermark structure: **semi-translucent spiral logos** in a regular grid + **diagonal "dreamstime.com" text** rendered at low alpha, plus a **solid blue bottom-strip** with URL/ID text.

LaMa-with-perfect-mask ceiling on this image: PSNR 28.08 dB.

## Detector benchmark

Run `tools/bench_*.py` from `.venv-sam3`:

| Tool | Recall | Precision | IoU | Notes |
|---|---|---|---|---|
| ConvNeXt + Grad-CAM | 0.64 | 0.29 | 0.248 | default, fastest |
| Florence-2 (any prompt) | 0.02 | — | 0.022 | one-detection-per-concept fails on tiled |
| SAM 2 (per-centroid) | 0.16 | 0.35 | 0.123 | over-segments, single-mark only |
| Grounding DINO ∩ pixel | 0.55 | 0.42 | 0.310 | Apache, ~700 MB |
| **SAM 3 hybrid** | 0.51 | 0.50 | **0.337** | needs HF auth, ~2 GB |
| Library-cached GT mask | — | — | — | **PSNR 28.29 dB** |
