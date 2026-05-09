# BC_WatermarkRemover

GPU-accelerated watermark removal for still images and video.

- **Stills**: LaMa inpainting (big-lama JIT checkpoint)
- **Video**: ProPainter temporal-consistent inpainting (stub - WIP)
- **Detection**: Auto-detect watermark regions, manual mask fallback
- **Web UI**: FastAPI + vanilla JS - upload, draw mask, download result

## Quick Start

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8090
```

Visit `http://localhost:8090`. Model weights (~200MB) auto-download in the background on first run. Set `BC_WMR_AUTO_DOWNLOAD=0` to disable and download manually with `python download_models.py`.

## GPU Selection

Automatically picks the GPU with the most free VRAM. Set `BC_WMR_SKIP_GPUS=1` to skip GPU 1 (default - reserved for LLM inference). Falls back to CPU when no CUDA is available.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/inpaint` | POST | Still image inpainting (file + mask) |
| `/api/video` | POST | Video inpainting job (file + mask, returns job ID) |
| `/api/detect` | POST | Auto-detect watermark mask |
| `/api/jobs/{id}` | GET | Poll job status |
| `/api/jobs/{id}/result` | GET | Download completed result |
| `/api/status` | GET | GPU info, loaded model |

## Status

- [x] Project scaffold
- [x] LaMa still image inpainting
- [x] Web UI with mask editor
- [x] Job queue for async video processing
- [ ] ProPainter video inpainting (model extraction WIP)
- [ ] Watermark auto-detection (model integration WIP)
