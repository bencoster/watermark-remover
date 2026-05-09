"""Offline E2E test: detector + (TELEA pre-fill) + LaMa on CPU, no server."""
import os
import shutil
import sys
import time
from pathlib import Path

# Force CPU before any service import - safety net against GPU freeze.
os.environ["BC_WMR_DEVICE"] = "cpu"

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from services.detector_service import detect_split, prefill_strip
from services.lama_service import inpaint, load_model
from services.cuda_policy import get_device


def main(image_path: str, out_dir: str = "tmp/e2e_out"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Detecting watermark in {image_path}")
    t0 = time.time()
    result = detect_split(image_path)
    print(f"      detector took {time.time() - t0:.2f}s")
    if result is None:
        print("      detector returned None - aborting")
        return 1
    body_mask_path, strip_mask_path, p_full = result
    print(f"      P(watermarked) = {p_full:.3f}")
    print(f"      body mask: {body_mask_path}")
    print(f"      strip mask: {strip_mask_path or '(none)'}")

    # Visual overlay: red = body (LaMa), blue = strip text (TELEA)
    img = cv2.imread(image_path)
    body = cv2.imread(body_mask_path, cv2.IMREAD_GRAYSCALE)
    overlay = img.copy()
    overlay[body > 0] = (0, 0, 255)
    if strip_mask_path:
        strip = cv2.imread(strip_mask_path, cv2.IMREAD_GRAYSCALE)
        overlay[strip > 0] = (255, 0, 0)
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    cv2.imwrite(str(out / "mask_overlay.png"), blended)
    shutil.copy(body_mask_path, out / "mask_body.png")
    if strip_mask_path:
        shutil.copy(strip_mask_path, out / "mask_strip.png")
    print(f"      overlay saved: {out / 'mask_overlay.png'}")

    # Pass 1: TELEA pre-fill of the thin strip-text glyphs (no learned prior).
    image_for_lama = image_path
    if strip_mask_path is not None:
        print(f"[2/4] TELEA pre-fill on strip text")
        t0 = time.time()
        image_for_lama = prefill_strip(image_path, strip_mask_path, radius=4)
        print(f"      prefill took {time.time() - t0:.2f}s")
        shutil.copy(image_for_lama, out / "step_prefilled.png")
    else:
        print(f"[2/4] No strip mask - skipping TELEA pre-fill")

    print(f"[3/4] Loading LaMa model on {get_device()}")
    t0 = time.time()
    model = load_model(get_device())
    print(f"      load took {time.time() - t0:.2f}s")

    print(f"      Running LaMa inpaint on body mask")
    t0 = time.time()
    result_path = inpaint(image_for_lama, body_mask_path, get_device(), model)
    print(f"      inpaint took {time.time() - t0:.2f}s")

    final_result = out / "result.png"
    shutil.copy(result_path, final_result)
    print(f"[4/4] Result saved: {final_result}")
    return 0


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_image.jpg"
    sys.exit(main(img))
