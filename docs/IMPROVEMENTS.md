# Improvement backlog

Ordered by **(value × likelihood-of-success) ÷ effort**. Numbers in
parentheses are rough estimates from observed bench behaviour, not measured.

## P0 — high value, modest effort

### 1. Auto-sweep settings + diff-rank → "best for this image"
Empirically the right setting combo varies per image class. Build a tool
(`tools/sweep_settings.py`, **shipped**) that:
- runs `/api/auto` across a curated grid of presets
- diff-compares each result to a target image (ground-truth, SaaS reference,
  or library mask preview)
- ranks by SSIM + PSNR + IoU vs the target

Output: a sorted markdown table + a contact-sheet PNG so you can eyeball
which setting wins for an image you've never processed. For image classes
you batch-process (Dreamstime, Shutterstock, Getty) just run once and
remember the winning preset.

**Status**: shipped as `tools/sweep_settings.py`. See section *Auto-sweep tool*
below for usage.

### 2. Upscale 2–4× before detection
User finding from the Diff Mask working-scale slider: at 2–4× linear scale,
diagonal text strokes occupy more pixels and detectors pick up edges that
were sub-pixel at native resolution. Easy port: in `detect_split()`, optionally
upscale `img_bgr` before the SAM 3 / GD / CAM detectors, run them at the larger
res, then downsample the resulting mask back to native dims (`INTER_NEAREST`
to preserve binary edges). Memory cost on a 1600×1157 image at 4×: 100 MB
working canvas, fine. Inference time +~3× for the detectors.

Expected gain: **+0.02–0.05 IoU** on tiled-grid masks, especially the SAM 3
hybrid case where each "watermark." instance gains pixels.

**Status**: not shipped. Estimated 1 hour of work.

### 3. Auto-purge polluted masks on startup
`startup` lifespan in `server.py` should call `_find_matching_library_mask`
in audit mode and drop entries whose mask file is missing OR whose `p_full < 1`
AND have been superseded by a better entry of the same fingerprint. Prevents
the library from drifting back to polluted via accidental saves. ~10 min of
work.

**Status**: not shipped.

## P1 — meaningful but more effort

### 4. SAM 3.1 multiplex via the GitHub `sam3` package
`facebook/sam3.1` ships `sam3.1_multiplex.pt` only — needs
`pip install -e git+https://github.com/facebookresearch/sam3.git`. Set up a
SECOND venv (`.venv-sam3p1`) so the GitHub package's pinned versions don't
collide with diffusers in the main venv. Expected gain over SAM 3 base:
**+0.01–0.03 IoU**, possibly **+0.05** on harder layouts. Worth the install
only if you process enough varied layouts that the small-but-consistent gain
adds up.

### 5. Per-image-class winning-preset memory
The library tracks watermark-removal *masks* but not which *settings* worked
best on which image class. Add a `presets` table:
- `class_signature` (perceptual hash or dimension+aspect bucket)
- `best_preset` (JSON of detect_mode/tiled_detector/strip_engine/...)
- `last_score` (PSNR/SSIM)
- `n_uses`

After each successful sweep (item 1), record the winner. On subsequent uploads
of similar images, surface "previous best for this class: SAM 3 hybrid + SDXL
strip" as a one-click suggestion in the Upload UI.

### 6. Cross-image-family detector validation
Current SAM 3 hybrid prompt set was tuned on Dreamstime. Test on:
- Shutterstock (SS or 123RF watermarks — usually thicker, less tiled)
- Getty (single corner stamp, large)
- Alamy (corner + diagonal)
- Adobe Stock (diagonal text only)

Probably need per-family prompts. Build a `prompts/` directory with named
preset files like `prompts/dreamstime.json`, `prompts/shutterstock.json`.

### 7. Z-Image-Edit when it lands
Currently `Z-Image-Edit` is unreleased (per Tongyi-MAI roadmap). When it does
land, it's the right tool for strip-bar refill (better texture continuation
than SDXL on natural image regions). Architecture is ready — drop in a
`zimage_inpaint_service.py` mirroring `sdxl_inpaint_service.py` and add `zimage`
to the `strip_engine` enum.

## P2 — nice-to-have

### 8. Frequency-domain pattern detection
Tiled watermarks have a regular spatial period → strong peaks in 2D FFT.
Detect peaks → reconstruct the grid mathematically → mask the entire pattern
without any model. Potentially better recall than any detector for periodic
patterns. Doesn't help non-periodic watermarks. Build as
`tools/bench_fft_detect.py` first to validate before integrating.

### 9. Mask-quality auto-fallback
If detect_split returns a body mask with coverage > 0.50 (definitely
over-mask), auto-fall-back to a stricter detect_mode and re-run. Today the
detector returns `None` past the cap; instead it should retry with `precision`
mode before giving up.

### 10. Video pipeline (ProPainter)
`services/propainter_service.py` is stubbed. Implementing requires:
- RAFT optical flow checkpoint
- Recurrent flow completion checkpoint
- ProPainter inpainting checkpoint
- 80-frame chunk processing with 10-frame overlap
- ffmpeg encode/decode plumbing
A meaningful weekend-or-two project. Hold until you have actual video to
remove watermarks from.

## P3 — quality-of-life

### 11. Drag-and-drop directly into Diff Mask tab
Diff Mask currently uses the file inputs. Mirror Upload's drag-drop pattern.

### 12. Save-as-preset button on Upload tab
Once you find a working setting combo, "Save current settings as preset
'dreamstime'" + dropdown to recall. Not the same as item 5 — that's automatic;
this is manual.

### 13. Comparing different inpainter outputs side-by-side
Currently the Compare button shows before vs after. Extend to show "best 4
results from the latest sweep" so you can pick which one to download.

---

## Auto-sweep tool — usage

```bash
cd "C:\Users\Ben Coster\watermark-remover"
.venv-sam3\Scripts\activate
$env:HF_TOKEN = "<your HF token>"

python tools/sweep_settings.py \
    --image tests/fixtures/dreamstime_18829755_watermarked.jpg \
    --target tests/fixtures/dreamstime_18829755_reference.jpg \
    --presets fast
```

Presets bundle named setting combos. `fast` runs ~6 combos in ~3 min; `all`
runs the full grid (~18 combos, ~10 min depending on SDXL availability).

Output: ranked table + `tmp/sweep_out/` with a result PNG per preset and a
single contact-sheet PNG showing all of them.
