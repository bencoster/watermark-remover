"""Auto-sweep /api/auto across a curated grid of settings.

For each preset combo it hits the live server, scores the result against
a target image (SaaS reference, ground-truth diff, or the user's chosen
"this is what good looks like" PNG), and ranks by SSIM + PSNR. Saves a
contact sheet so you can eyeball which preset wins for an image class
you've never processed before.

Usage:
    python tools/sweep_settings.py [--image PATH] [--target PATH]
                                   [--presets fast|all] [--server URL]

Defaults to the canonical Dreamstime fixture vs the SaaS reference.

Run from the .venv-sam3 venv so SAM 3 / SDXL / GD are available:
    .venv-sam3\\Scripts\\python.exe tools/sweep_settings.py
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import requests

FIXTURES = ROOT / "tests" / "fixtures"
DEFAULT_IMAGE = FIXTURES / "dreamstime_18829755_watermarked.jpg"
DEFAULT_TARGET = FIXTURES / "dreamstime_18829755_reference.jpg"
DEFAULT_GT_MASK = FIXTURES / "dreamstime_18829755_groundtruth_mask.png"
OUT_DIR = ROOT / "tmp" / "sweep_out"


# ─── Presets ──────────────────────────────────────────────────────────────────

# Each preset is the form-data dict sent to /api/auto. We curate ~6 useful
# combos for `fast`, then the full cross product for `all`. Sequential —
# parallel would OOM on the 4090 between LaMa and SDXL.

FAST_PRESETS = [
    {"name": "default",
     "fields": {"detect_mode": "auto", "library_mask": "auto",
                "strip_mode": "inpaint", "strip_engine": "telea",
                "tiled_detector": "off", "sam_refine": "off"}},
    {"name": "library_off+gd",
     "fields": {"detect_mode": "auto", "library_mask": "off",
                "strip_mode": "inpaint", "strip_engine": "telea",
                "tiled_detector": "grounding_dino", "sam_refine": "off"}},
    {"name": "library_off+sam3",
     "fields": {"detect_mode": "auto", "library_mask": "off",
                "strip_mode": "inpaint", "strip_engine": "telea",
                "tiled_detector": "sam3", "sam_refine": "off"}},
    {"name": "library_off+sam3+sdxl",
     "fields": {"detect_mode": "auto", "library_mask": "off",
                "strip_mode": "inpaint", "strip_engine": "sdxl",
                "tiled_detector": "sam3", "sam_refine": "off"}},
    {"name": "library_off+sam3+crop",
     "fields": {"detect_mode": "auto", "library_mask": "off",
                "strip_mode": "crop", "strip_engine": "telea",
                "tiled_detector": "sam3", "sam_refine": "off"}},
    {"name": "force_recall+sam3",
     "fields": {"detect_mode": "recall", "library_mask": "off",
                "strip_mode": "inpaint", "strip_engine": "telea",
                "tiled_detector": "sam3", "sam_refine": "off"}},
    {"name": "library_only",
     "fields": {"detect_mode": "auto", "library_mask": "auto",
                "strip_mode": "inpaint", "strip_engine": "telea",
                "tiled_detector": "off", "sam_refine": "off"}},
]


def all_presets():
    """Cartesian product over the meaningful axes. Skip combos that are
    ruled-out (e.g. SAM 2 on tiled, library on with detection sweep)."""
    out = []
    for det_mode, tiled, strip_mode, strip_eng, sam in product(
        ("auto", "recall", "precision"),
        ("off", "sam3", "grounding_dino"),
        ("inpaint", "crop"),
        ("telea", "sdxl"),
        ("off",),  # SAM 2 over-segments; skip
    ):
        if strip_mode == "crop" and strip_eng == "sdxl":
            continue  # SDXL is for refilling the strip; crop drops it
        out.append({
            "name": f"{det_mode}+{tiled}+{strip_mode}+{strip_eng}",
            "fields": {"detect_mode": det_mode, "library_mask": "off",
                       "strip_mode": strip_mode, "strip_engine": strip_eng,
                       "tiled_detector": tiled, "sam_refine": sam}
        })
    return out


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    K1, K2, L = 0.01, 0.03, 255
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sa2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sb2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sab + C2)
    den = (mu_a * mu_a + mu_b * mu_b + C1) * (sa2 + sb2 + C2)
    return float(np.mean(num / den))


def _ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([_ssim(a[..., c], b[..., c]) for c in range(3)]))


def score(result: np.ndarray, target: np.ndarray, body_frac: float = 0.86) -> dict:
    """Crop both to the body region (default top 86%) so strip-handling
    differences don't dominate the score, then compute PSNR + SSIM."""
    if result.shape[:2] != target.shape[:2]:
        target = cv2.resize(target, (result.shape[1], result.shape[0]))
    h = result.shape[0]
    rb = result[: int(h * body_frac)]
    tb = target[: int(h * body_frac)]
    return {
        "psnr": float(cv2.PSNR(tb, rb)),
        "ssim": float(_ssim_rgb(rb, tb)),
    }


# ─── Sweep runner ─────────────────────────────────────────────────────────────

def run_preset(server: str, image_path: Path, fields: dict,
               timeout_s: float = 600) -> tuple[bytes | None, str, float]:
    """POST /api/auto with the given form fields. Returns (png_bytes, mask_source, elapsed)."""
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        t0 = time.time()
        try:
            r = requests.post(server.rstrip("/") + "/api/auto",
                              data=fields, files=files, timeout=timeout_s)
        except requests.RequestException as e:
            return None, f"request error: {type(e).__name__}: {e}", time.time() - t0
    elapsed = time.time() - t0
    if r.status_code == 422:
        return None, "no watermark detected", elapsed
    if not r.ok:
        return None, f"HTTP {r.status_code}: {r.text[:200]}", elapsed
    return r.content, r.headers.get("X-Mask-Source", "detect"), elapsed


def contact_sheet(rows: list[dict], target: np.ndarray, out_path: Path,
                  cell_w: int = 540) -> None:
    """Build a single PNG showing all results next to the target. One
    row per preset with the preset name + score overlaid."""
    n = len(rows)
    if not n:
        return
    th, tw = target.shape[:2]
    cell_h = int(round(cell_w * th / tw))
    target_thumb = cv2.resize(target, (cell_w, cell_h))
    panel_h = cell_h + 56
    sheet = np.full((panel_h * (n + 1), cell_w, 3), 12, dtype=np.uint8)

    def stamp(img, text, y_offset):
        bar = np.full((56, cell_w, 3), 25, dtype=np.uint8)
        cv2.putText(bar, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (220, 220, 220), 1, cv2.LINE_AA)
        sheet[y_offset:y_offset + 56] = bar
        sheet[y_offset + 56: y_offset + 56 + cell_h] = img

    # Top: target
    stamp(target_thumb, "TARGET", 0)
    # Each preset row
    for i, row in enumerate(rows, start=1):
        img = row.get("img")
        if img is None:
            label = f"{row['name']}  FAILED  ({row['note']})"
            placeholder = np.full((cell_h, cell_w, 3), 32, dtype=np.uint8)
            stamp(placeholder, label, i * panel_h)
            continue
        thumb = cv2.resize(img, (cell_w, cell_h))
        s = row["score"]
        label = (f"{row['name']:<32}  "
                 f"PSNR {s['psnr']:5.2f}  SSIM {s['ssim']:.3f}  "
                 f"{row['mask_source']}  {row['elapsed']:.1f}s")
        stamp(thumb, label, i * panel_h)
    cv2.imwrite(str(out_path), sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(DEFAULT_IMAGE),
                    help="Input image to process")
    ap.add_argument("--target", default=str(DEFAULT_TARGET),
                    help="Target image to score against (SaaS reference / known-good)")
    ap.add_argument("--presets", default="fast", choices=("fast", "all"))
    ap.add_argument("--server", default="http://127.0.0.1:8091",
                    help="Running watermark-remover server URL")
    args = ap.parse_args()

    image_path = Path(args.image)
    target_path = Path(args.target)
    target = cv2.imread(str(target_path))
    if target is None:
        print(f"ERROR: cannot read target {target_path}")
        return 2

    presets = FAST_PRESETS if args.presets == "fast" else all_presets()
    print(f"Running {len(presets)} presets against {image_path.name} → {target_path.name}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, preset in enumerate(presets, 1):
        name = preset["name"]
        fields = preset["fields"]
        flat = " ".join(f"{k}={v}" for k, v in fields.items())
        print(f"[{i}/{len(presets)}] {name}")
        print(f"          {flat}")
        png, mask_source, elapsed = run_preset(args.server, image_path, fields)
        if png is None:
            print(f"          FAILED ({mask_source}, {elapsed:.1f}s)\n")
            rows.append({"name": name, "img": None, "note": mask_source,
                         "elapsed": elapsed, "score": {"psnr": 0, "ssim": 0},
                         "mask_source": ""})
            continue
        out_png = OUT_DIR / f"{i:02d}_{name}.png"
        out_png.write_bytes(png)
        result = cv2.imread(str(out_png))
        s = score(result, target)
        print(f"          PSNR {s['psnr']:5.2f}  SSIM {s['ssim']:.3f}  "
              f"mask={mask_source}  {elapsed:.1f}s\n")
        rows.append({"name": name, "img": result, "score": s,
                     "elapsed": elapsed, "mask_source": mask_source,
                     "note": ""})

    # Rank
    ranked = sorted(rows, key=lambda r: (-r["score"]["ssim"], -r["score"]["psnr"]))
    print("\n" + "=" * 80)
    print("RANKED RESULTS (best SSIM first)")
    print("=" * 80)
    print(f"{'rank':>4}  {'preset':<32}  {'PSNR':>6}  {'SSIM':>6}  {'mask src':<10}  {'time':>6}")
    print("-" * 80)
    for i, r in enumerate(ranked, 1):
        if r["img"] is None:
            print(f"{i:>4}  {r['name']:<32}  {'-':>6}  {'-':>6}  {'failed':<10}  {r['elapsed']:>5.1f}s")
        else:
            print(f"{i:>4}  {r['name']:<32}  "
                  f"{r['score']['psnr']:>5.2f}   {r['score']['ssim']:>.3f}  "
                  f"{r['mask_source']:<10}  {r['elapsed']:>5.1f}s")

    contact_sheet(ranked, target, OUT_DIR / "contact_sheet.png")
    print(f"\nContact sheet: {OUT_DIR / 'contact_sheet.png'}")
    print(f"Per-preset PNGs: {OUT_DIR}\\")

    # Persist a markdown ranking next to the sheet
    md = OUT_DIR / "ranking.md"
    with md.open("w", encoding="utf-8") as f:
        f.write(f"# Sweep — `{image_path.name}` vs `{target_path.name}`\n\n")
        f.write(f"| Rank | Preset | PSNR | SSIM | Mask | Time |\n")
        f.write(f"|---:|---|---:|---:|---|---:|\n")
        for i, r in enumerate(ranked, 1):
            if r["img"] is None:
                f.write(f"| {i} | {r['name']} | — | — | failed: {r['note']} | {r['elapsed']:.1f}s |\n")
            else:
                f.write(f"| {i} | {r['name']} | {r['score']['psnr']:.2f} dB | "
                        f"{r['score']['ssim']:.3f} | {r['mask_source']} | {r['elapsed']:.1f}s |\n")
    print(f"Markdown:        {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
