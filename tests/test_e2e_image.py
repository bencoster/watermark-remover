"""Offline E2E test: detector + LaMa on CPU, no server."""
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

from services.detector_service import detect
from services.lama_service import inpaint, load_model
from services.cuda_policy import get_device


def main(image_path: str, out_dir: str = "tmp/e2e_out"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Detecting watermark in {image_path}")
    t0 = time.time()
    mask_path = detect(image_path)
    print(f"      detector took {time.time() - t0:.2f}s")
    if mask_path is None:
        print("      detector returned None - aborting")
        return 1

    final_mask = out / "mask.png"
    shutil.copy(mask_path, final_mask)
    print(f"      mask saved: {final_mask}")

    img = cv2.imread(image_path)
    mask = cv2.imread(str(final_mask), cv2.IMREAD_GRAYSCALE)
    overlay = img.copy()
    overlay[mask > 0] = (0, 0, 255)
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    overlay_path = out / "mask_overlay.png"
    cv2.imwrite(str(overlay_path), blended)
    print(f"      overlay saved: {overlay_path}")

    print(f"[2/4] Loading LaMa model on {get_device()}")
    t0 = time.time()
    model = load_model(get_device())
    print(f"      load took {time.time() - t0:.2f}s")

    print(f"[3/4] Running inpaint")
    t0 = time.time()
    result_path = inpaint(image_path, str(final_mask), get_device(), model)
    print(f"      inpaint took {time.time() - t0:.2f}s")

    final_result = out / "result.png"
    shutil.copy(result_path, final_result)
    print(f"[4/4] Result saved: {final_result}")
    return 0


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_image.jpg"
    sys.exit(main(img))
