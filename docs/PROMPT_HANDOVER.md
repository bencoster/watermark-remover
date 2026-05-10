# Session handover prompt

Paste the block below into a new Claude session to continue this project from
where the previous session left off. The block is self-contained — it points
at the docs in this repo for everything beyond the absolute essentials.

---

## Paste this into a new session

> I'm continuing work on `bencoster/watermark-remover` (FastAPI + JS web app
> for removing watermarks from images, runs on `localhost:8091`). Previous
> session left a clean state — 34/34 tests passing, latest commit on `master`.
>
> **Read these first** in `C:\Users\Ben Coster\watermark-remover\docs\`:
>
> 1. `HANDOVER.md` — current state, pipeline, env, auth, detector bench numbers
> 2. `ARCHITECTURE.md` — components, routes, data flow
> 3. `IMPROVEMENTS.md` — prioritised backlog
>
> **Critical environment facts** (don't break these):
>
> - Server runs from `.venv-sam3` (Python 3.12, PyTorch 2.10 cu128,
>   transformers 5.8, diffusers 0.38). System Python 3.10 is shared with
>   BC_LocalLLM/ComfyUI — don't pip-install into it.
> - `~/.claude/launch.json` already points at the venv Python.
> - HF auth was via env var only (no token on disk). Account `bcoster`. To
>   re-auth: `$env:HF_TOKEN = "<token>"` before running anything that needs
>   `facebook/sam3`.
> - `.venv*/` is in `.gitignore`; HF model weights live in
>   `~/.cache/huggingface/hub/`, not the project tree.
>
> **Current pipeline best**:
>
> | Detector | IoU on canonical fixture |
> |---|---|
> | ConvNeXt + Grad-CAM (default) | 0.248 |
> | Grounding DINO ∩ pixel + dilate | 0.310 |
> | **SAM 3 hybrid** (facebook/sam3, not 3.1) | **0.337** |
> | Library-cached GT mask (PSNR vs SaaS ref) | 28.29 dB |
>
> SAM 3 hybrid uses two prompts in parallel (`"watermark."` + `"tiled stock
> photo watermark."` both at threshold 0.05) and combines as
> `precise OR (loose ∩ pixel_heuristic) + 3×3 dilate`. Wired into
> `/api/auto?tiled_detector=sam3`.
>
> **Watermark structure on the canonical fixture** (Dreamstime stock photo
> at `tests/fixtures/dreamstime_18829755_*.{jpg,png}`):
>
> - semi-translucent spiral logos in a regular tiled grid (~50 instances)
> - diagonal `dreamstime.com` text at low alpha across each tile
> - solid blue bottom-strip with URL + ID/copyright text
>
> **Tools in `tools/` for benchmarking** — all run from the venv:
>
> - `bench_sam3.py` — SAM 3 prompt sweep
> - `bench_grounding_dino.py` / `bench_gd_hybrid.py` — GD comparisons
> - `bench_florence2.py` — confirms Florence-2 fails on tiled (one detection)
> - `iterate.py` — generic per-iteration metric harness
> - `sweep_settings.py` — runs `/api/auto` across a preset grid, ranks by
>   SSIM/PSNR vs a target. Run with `--presets fast` for ~6 combos.
>
> **What I want done next**, in order:
>
> 1. **Port the upscale-before-detect insight to the auto pipeline.** The
>    Diff Mask working-scale slider (0.1×–10× log-spaced) showed empirically
>    that 2–4× linear upscale gives detectors more pixels per text stroke.
>    In `services/detector_service.detect_split()`, optionally upscale
>    `img_bgr` before running SAM 3 / GD / CAM, then `INTER_NEAREST`
>    downsample the resulting mask back. Add a `pre_upscale: float = 1.0`
>    parameter, expose as `/api/auto?upscale=2.0`, expose in the UI as a
>    "Pre-upscale before detect: 1× / 2× / 4×" radio. Bench-validate against
>    the canonical fixture's GT mask; expected gain +0.02–0.05 IoU.
>
> 2. **Per-image-class winning-preset memory.** Add a `presets` table to
>    jobs.db with `(class_signature, best_preset_json, last_score, n_uses)`.
>    `class_signature` can start as `(width, height, aspect bucket)` then
>    upgrade to a perceptual hash later. After each successful sweep
>    (sweep_settings.py), record the winner. On Upload, surface "previous
>    best for this class" as a one-click suggestion above the radios.
>
> 3. **Startup library audit.** Drop library entries whose mask file is
>    missing on disk; demote `p_full < 1.0` entries that have been
>    superseded by a higher-quality entry of the same fingerprint family.
>    Prevents drift back to polluted state. ~10 min of work.
>
> Don't auto-save detection masks to the library — that pollution issue
> was already fixed; only Diff Mask saves should populate the library.
>
> Don't push if any test fails. Always commit clean (`git status` should
> not show `.venv-sam3/` ever; if it does, the gitignore broke).

---

## Sanity checklist for the new session

Before changing anything, the new session should be able to confirm:

```bash
cd "C:\Users\Ben Coster\watermark-remover"
git log --oneline -3              # latest commit visible
git status                         # clean working tree
.venv-sam3\Scripts\activate
python -c "from server import app; print(len(app.routes), 'routes')"
                                   # → 25 routes
python -m pytest tests/ --ignore=tests/test_e2e_image.py -q
                                   # → 34 passed
```

If any of those four checks fail, fix that before starting on the backlog.

## Where to find things fast

| Need | Path |
|---|---|
| Detector entry point | `services/detector_service.py` → `detect_split()` |
| SAM 3 implementation | `services/sam3_service.py` → `detect_hybrid_mask()` |
| Library short-circuit | `routes/inpaint.py` → `_find_matching_library_mask()` |
| Strip handling | `services/detector_service.py` → `_strip_text_mask()` |
| SDXL strip refill | `services/sdxl_inpaint_service.py` |
| Frontend radios | `web/js/upload.js` (look for `tiledDetector`, `stripEngine`) |
| Diff Mask scale slider | `web/js/diff-mask.js` → `_setupDiffCanvases` |
| Auto-sweep tool | `tools/sweep_settings.py` |
| Bench harness | `tools/iterate.py` |
| Test fixtures | `tests/fixtures/dreamstime_18829755_*.{jpg,png}` |
