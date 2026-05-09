"""Download model weights with MD5 verification."""
import hashlib
import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).parent / "weights"

MODELS = {
    "big-lama.pt": {
        "url": "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        "md5": "e3aa4aaa15225a33ec84f9f4bc47e500",
    },
}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, force: bool = False) -> Path:
    info = MODELS[name]
    dest = WEIGHTS_DIR / name
    if dest.exists() and not force:
        if md5_file(dest) == info["md5"]:
            logger.info("%s already downloaded and verified", name)
            return dest
        logger.warning("%s MD5 mismatch - re-downloading", name)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from %s", name, info["url"])
    urllib.request.urlretrieve(info["url"], str(dest))

    actual = md5_file(dest)
    if actual != info["md5"]:
        dest.unlink()
        raise RuntimeError(f"MD5 mismatch for {name}: expected {info['md5']}, got {actual}")
    logger.info("%s downloaded and verified", name)
    return dest


def download_all():
    for name in MODELS:
        download(name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_all()
