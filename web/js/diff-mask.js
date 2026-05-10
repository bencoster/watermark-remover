// Live diff-mask builder.
//
// User loads:
//   A = watermarked image
//   B = clean reference (e.g. from another tool, or the original
//       before-watermark version if available)
//
// We resize B to A's dimensions, compute per-pixel BGR delta, then
// apply user-controlled threshold / dilate / blur to produce a binary
// mask. Everything runs in <canvas> so the preview is instant.
//
// "Save to Library" uploads the source image + final mask to
// /api/masks/from-diff which persists it as a reusable library entry.

let _diff = {
  imgA: null, imgB: null,
  canvasA: null, canvasB: null, canvasMask: null, canvasDetector: null,
  ctxA: null, ctxB: null, ctxMask: null, ctxDetector: null,
  // Resampled native-resolution data for image B (size matches A's
  // current working scale). Recomputed when scale changes.
  dataA: null, dataB: null,
  // Working scale: resample image A and B to this multiple before
  // computing the diff. 1.0 = native; 2.0 = 2x linear (4x area);
  // 0.25 = quarter linear. The mask is downsized to native A dims
  // when saving.
  scale: 1.0,
  // tunables — synced to slider DOM each render
  threshold: 30,
  dilate: 3,
  blur: 1,
  channels: 'lum',
  // Last detector mask overlay (bounded to working scale)
  detectorMask: null,
};

// Slider exposes 0.1x..10x logarithmically: slider value 0..100 maps
// to 10**(value/100*2 - 1) = 0.1..10. Linear sliders feel too uneven
// across a 100x range.
function _scaleFromSlider(v) { return Math.pow(10, (+v) / 50 - 1); }
function _sliderFromScale(s) { return Math.round((Math.log10(s) + 1) * 50); }

function initDiffMask() {
  const panel = document.getElementById('panel-diff');
  panel.innerHTML = `
    <div class="diff-header">
      <h2>Diff Mask Builder</h2>
      <p class="library-hint">Load a watermarked image and a clean reference of the same scene. The pixel difference becomes a precise mask. Tune the sliders to dial in coverage, then save to the library.</p>
    </div>
    <div class="diff-loaders">
      <label class="diff-loader">
        <span>A — Watermarked</span>
        <input type="file" id="diffFileA" accept="image/*">
        <div class="diff-thumb" id="diffThumbA"></div>
      </label>
      <label class="diff-loader">
        <span>B — Clean reference</span>
        <input type="file" id="diffFileB" accept="image/*">
        <div class="diff-thumb" id="diffThumbB"></div>
      </label>
    </div>
    <div class="diff-controls" id="diffControls" style="display:none">
      <label title="Resample image A and B to this fraction of A's native resolution before computing the diff. Higher resolutions give the detector more pixels per text stroke; lower resolutions are faster.">Working scale <input type="range" id="diffScale" min="0" max="100" value="50"> <span id="diffScaleValue">1.00x</span></label>
      <label>Threshold <input type="range" id="diffThreshold" min="2" max="120" value="30"> <span id="diffThresholdValue">30</span></label>
      <label>Dilate <input type="range" id="diffDilate" min="0" max="15" value="3"> <span id="diffDilateValue">3</span></label>
      <label>Smooth <input type="range" id="diffBlur" min="0" max="10" value="1"> <span id="diffBlurValue">1</span></label>
      <label>Channels
        <select id="diffChannels">
          <option value="lum" selected>Luminance (robust)</option>
          <option value="rgb">RGB max (sensitive)</option>
        </select>
      </label>
      <span class="diff-coverage" id="diffCoverage"></span>
      <span class="diff-resolution" id="diffResolution"></span>
    </div>
    <div class="diff-stage" id="diffStage" style="display:none">
      <div class="diff-canvas-wrap">
        <canvas id="diffCanvasA"></canvas>
        <canvas id="diffCanvasMask"></canvas>
        <canvas id="diffCanvasDetector"></canvas>
      </div>
      <div class="diff-actions">
        <button class="btn btn-secondary" id="diffCompleteLinesBtn">Complete Partial Lines</button>
        <button class="btn btn-secondary" id="diffRunDetectorBtn">Run Current Detector at this Scale</button>
        <button class="btn btn-primary" id="diffSaveBtn">Save to Library</button>
        <button class="btn btn-secondary" id="diffApplyBtn">Apply (manual editor)</button>
        <button class="btn btn-secondary" id="diffDownloadBtn">Download PNG</button>
      </div>
      <div class="diff-legend">
        <span><span class="legend-swatch swatch-diff"></span> Diff mask (red)</span>
        <span><span class="legend-swatch swatch-detector"></span> Current detector (cyan)</span>
      </div>
    </div>
  `;

  document.getElementById('diffFileA').addEventListener('change', e => loadDiffImage(e.target.files[0], 'A'));
  document.getElementById('diffFileB').addEventListener('change', e => loadDiffImage(e.target.files[0], 'B'));

  // Working-scale slider — debounced because resampling at 10x is heavy
  const scaleEl = document.getElementById('diffScale');
  let scaleDebounce = null;
  scaleEl.addEventListener('input', () => {
    const s = _scaleFromSlider(scaleEl.value);
    _diff.scale = s;
    document.getElementById('diffScaleValue').textContent = s.toFixed(2) + 'x';
    if (scaleDebounce) clearTimeout(scaleDebounce);
    scaleDebounce = setTimeout(() => {
      // Detector mask is invalidated whenever scale changes — its
      // pixel coordinates only make sense at the resolution it was
      // computed at.
      _diff.detectorMask = null;
      const ctx = _diff.ctxDetector;
      if (ctx) ctx.clearRect(0, 0, _diff.canvasDetector.width, _diff.canvasDetector.height);
      _setupDiffCanvases();
      renderDiffMask();
    }, 200);
  });

  ['Threshold', 'Dilate', 'Blur'].forEach(k => {
    const el = document.getElementById('diff' + k);
    el.addEventListener('input', () => {
      document.getElementById(`diff${k}Value`).textContent = el.value;
      _diff[k.toLowerCase()] = +el.value;
      renderDiffMask();
    });
  });
  document.getElementById('diffChannels').addEventListener('change', e => {
    _diff.channels = e.target.value;
    renderDiffMask();
  });
  document.getElementById('diffSaveBtn').addEventListener('click', saveDiffMask);
  document.getElementById('diffApplyBtn').addEventListener('click', applyDiffMaskToEditor);
  document.getElementById('diffDownloadBtn').addEventListener('click', downloadDiffMask);
  document.getElementById('diffCompleteLinesBtn').addEventListener('click', completePartialLines);
  document.getElementById('diffRunDetectorBtn').addEventListener('click', runDetectorAtScale);

  // Default-load both fixtures so the diff workflow runs end-to-end
  // without the user needing to upload anything. This matches the
  // Upload tab's behaviour where the watermarked sample is preloaded.
  loadDefaultDiffSamples();
}

async function loadDefaultDiffSamples() {
  try {
    const [rA, rB] = await Promise.all([
      fetch(API + '/api/sample-image'),
      fetch(API + '/api/sample-clean-image'),
    ]);
    if (!rA.ok || !rB.ok) return;
    const blobA = await rA.blob();
    const blobB = await rB.blob();
    const fileA = new File([blobA], 'dreamstime_watermarked.jpg', { type: blobA.type });
    const fileB = new File([blobB], 'dreamstime_clean.jpg', { type: blobB.type });
    loadDiffImage(fileA, 'A');
    loadDiffImage(fileB, 'B');
  } catch (e) { /* silent — user can still upload */ }
}

async function completePartialLines() {
  // Server-side: detect partial line segments in the current mask via
  // Hough on the skeleton, then extend each one along its own slope to
  // span the bounding box of nearby co-linear pixels. Returns a new
  // mask with the gaps filled. Works much better in OpenCV than in JS.
  if (!_diff.lastMaskBin) { showToast('Adjust sliders first'); return; }
  const blob = await _maskBinToBlob();
  const fd = new FormData();
  fd.append('mask', blob, 'mask.png');
  showToast('Completing lines...');
  try {
    const r = await fetch(API + '/api/masks/complete-lines', { method: 'POST', body: fd });
    if (!r.ok) { showToast('Complete failed: ' + r.status); return; }
    const out = await r.blob();
    // Re-load the completed mask back into the canvas + overlay.
    const im = new Image();
    im.onload = () => {
      const w = _diff.canvasMask.width, h = _diff.canvasMask.height;
      const tmp = document.createElement('canvas');
      tmp.width = w; tmp.height = h;
      tmp.getContext('2d').drawImage(im, 0, 0, w, h);
      const data = tmp.getContext('2d').getImageData(0, 0, w, h).data;
      const updated = new Uint8ClampedArray(w * h);
      for (let p = 0, q = 0; p < updated.length; p++, q += 4) {
        updated[p] = data[q] > 127 ? 255 : 0;
      }
      _diff.lastMaskBin = updated;
      // Repaint translucent red overlay
      const rgba = new Uint8ClampedArray(w * h * 4);
      let count = 0;
      for (let p = 0, q = 0; p < updated.length; p++, q += 4) {
        if (updated[p]) { rgba[q]=255; rgba[q+1]=64; rgba[q+2]=64; rgba[q+3]=140; count++; }
      }
      _diff.ctxMask.putImageData(new ImageData(rgba, w, h), 0, 0);
      const cov = (count / (w*h) * 100).toFixed(1);
      document.getElementById('diffCoverage').textContent = `Coverage: ${cov}% (lines completed)`;
      showToast('Lines completed');
    };
    im.src = URL.createObjectURL(out);
  } catch (e) { showToast('Error: ' + e.message); }
}

function loadDiffImage(file, slot) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    _diff[slot === 'A' ? 'imgA' : 'imgB'] = img;
    if (slot === 'A') _diff.fileA = file;
    document.getElementById('diffThumb' + slot).innerHTML = `<img src="${url}">`;
    if (_diff.imgA && _diff.imgB) {
      document.getElementById('diffControls').style.display = '';
      document.getElementById('diffStage').style.display = '';
      _setupDiffCanvases();
      renderDiffMask();
    }
  };
  img.src = url;
}

function _setupDiffCanvases() {
  const a = _diff.imgA;
  // Working-scale dimensions. Cap absolute area at 64 MP so a careless
  // 10x slider on a 4000x3000 source doesn't lock the browser.
  const nativeW = a.naturalWidth, nativeH = a.naturalHeight;
  let sx = nativeW * _diff.scale, sy = nativeH * _diff.scale;
  const MAX_AREA = 64 * 1024 * 1024;
  if (sx * sy > MAX_AREA) {
    const k = Math.sqrt(MAX_AREA / (sx * sy));
    sx *= k; sy *= k;
    document.getElementById('diffResolution').textContent = ` (clamped from ${(_diff.scale).toFixed(2)}x)`;
  } else {
    document.getElementById('diffResolution').textContent = '';
  }
  const W = Math.max(8, Math.round(sx));
  const H = Math.max(8, Math.round(sy));

  const c = document.getElementById('diffCanvasA');
  const m = document.getElementById('diffCanvasMask');
  const d = document.getElementById('diffCanvasDetector');
  c.width = W; c.height = H;
  m.width = W; m.height = H;
  d.width = W; d.height = H;
  _diff.canvasA = c; _diff.canvasMask = m; _diff.canvasDetector = d;
  _diff.ctxA = c.getContext('2d', { willReadFrequently: true });
  _diff.ctxMask = m.getContext('2d');
  _diff.ctxDetector = d.getContext('2d');
  _diff.ctxA.imageSmoothingEnabled = true;
  _diff.ctxA.imageSmoothingQuality = 'high';
  _diff.ctxA.drawImage(a, 0, 0, W, H);

  // Resample image B to working dims so the diff only sees aligned pixels.
  const tmp = document.createElement('canvas');
  tmp.width = W; tmp.height = H;
  const tctx = tmp.getContext('2d');
  tctx.imageSmoothingEnabled = true;
  tctx.imageSmoothingQuality = 'high';
  tctx.drawImage(_diff.imgB, 0, 0, W, H);
  _diff.dataA = _diff.ctxA.getImageData(0, 0, W, H);
  _diff.dataB = tctx.getImageData(0, 0, W, H);

  document.getElementById('diffResolution').textContent =
    `${W}x${H}` + (document.getElementById('diffResolution').textContent || '');
}

async function runDetectorAtScale() {
  // Send the current working-resolution image A to /api/detect and
  // overlay the returned mask in cyan. Lets the user A/B the diff
  // mask vs the auto-detector at the chosen working scale.
  if (!_diff.canvasA) { showToast('Load both images first'); return; }
  showToast('Running detector at working scale...');
  const blob = await new Promise(r => _diff.canvasA.toBlob(r, 'image/png'));
  const fd = new FormData();
  fd.append('file', blob, 'a_scaled.png');
  try {
    const r = await fetch(API + '/api/detect', { method: 'POST', body: fd });
    if (!r.ok) {
      const ct = r.headers.get('content-type') || '';
      if (ct.includes('image')) {
        // unexpected — should be JSON if no detection
      } else {
        showToast('Detector returned ' + r.status);
        return;
      }
    }
    const ct = r.headers.get('content-type') || '';
    if (!ct.includes('image')) {
      showToast('Detector found no watermark at this scale');
      return;
    }
    const maskBlob = await r.blob();
    const im = new Image();
    im.onload = () => {
      const W = _diff.canvasDetector.width, H = _diff.canvasDetector.height;
      const tmp = document.createElement('canvas');
      tmp.width = W; tmp.height = H;
      tmp.getContext('2d').drawImage(im, 0, 0, W, H);
      const data = tmp.getContext('2d').getImageData(0, 0, W, H).data;
      const rgba = new Uint8ClampedArray(W * H * 4);
      let count = 0;
      for (let p = 0, q = 0; p < W * H; p++, q += 4) {
        if (data[q] > 127) {
          rgba[q] = 64; rgba[q + 1] = 200; rgba[q + 2] = 255; rgba[q + 3] = 130;
          count++;
        }
      }
      _diff.ctxDetector.putImageData(new ImageData(rgba, W, H), 0, 0);
      _diff.detectorMask = data;
      const cov = (count / (W * H) * 100).toFixed(1);
      showToast(`Detector covered ${cov}% at ${_diff.scale.toFixed(2)}x`);
    };
    im.src = URL.createObjectURL(maskBlob);
  } catch (e) { showToast('Error: ' + e.message); }
}

function renderDiffMask() {
  if (!_diff.dataA || !_diff.dataB) return;
  const a = _diff.dataA.data, b = _diff.dataB.data;
  const w = _diff.dataA.width, h = _diff.dataA.height;
  const mask = new Uint8ClampedArray(w * h);
  const t = _diff.threshold;

  if (_diff.channels === 'lum') {
    for (let i = 0, p = 0; i < a.length; i += 4, p++) {
      const la = 0.299 * a[i] + 0.587 * a[i+1] + 0.114 * a[i+2];
      const lb = 0.299 * b[i] + 0.587 * b[i+1] + 0.114 * b[i+2];
      mask[p] = Math.abs(la - lb) > t ? 255 : 0;
    }
  } else {
    for (let i = 0, p = 0; i < a.length; i += 4, p++) {
      const dr = Math.abs(a[i]-b[i]);
      const dg = Math.abs(a[i+1]-b[i+1]);
      const db = Math.abs(a[i+2]-b[i+2]);
      mask[p] = Math.max(dr,dg,db) > t ? 255 : 0;
    }
  }

  // Box-blur smooth (cheap) before threshold to suppress speckles
  let smoothed = mask;
  if (_diff.blur > 0) smoothed = _boxBlur(smoothed, w, h, _diff.blur);

  // Dilate iteratively (3x3 + repeat). Cheap dilation via 2-pass max.
  let dilated = smoothed;
  for (let i = 0; i < _diff.dilate; i++) {
    dilated = _dilate3x3(dilated, w, h);
  }

  // Pack into RGBA for canvas, with a translucent red overlay.
  const rgba = new Uint8ClampedArray(w * h * 4);
  let count = 0;
  for (let p = 0, q = 0; p < dilated.length; p++, q += 4) {
    if (dilated[p] > 127) {
      rgba[q] = 255; rgba[q+1] = 64; rgba[q+2] = 64; rgba[q+3] = 140;
      count++;
    }
  }
  _diff.ctxMask.putImageData(new ImageData(rgba, w, h), 0, 0);
  _diff.lastMaskBin = dilated;  // keep a copy for save/download

  const cov = (count / (w * h) * 100).toFixed(1);
  document.getElementById('diffCoverage').textContent = `Coverage: ${cov}%`;
}

function _boxBlur(src, w, h, radius) {
  // Two-pass box blur on a binary mask -> soft gray then threshold>127
  const tmp = new Uint8ClampedArray(src.length);
  const out = new Uint8ClampedArray(src.length);
  const r = Math.max(1, radius | 0);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, n = 0;
      for (let dx = -r; dx <= r; dx++) {
        const xx = x + dx;
        if (xx < 0 || xx >= w) continue;
        sum += src[y * w + xx]; n++;
      }
      tmp[y * w + x] = sum / n;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, n = 0;
      for (let dy = -r; dy <= r; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        sum += tmp[yy * w + x]; n++;
      }
      out[y * w + x] = sum / n > 64 ? 255 : 0;
    }
  }
  return out;
}

function _dilate3x3(src, w, h) {
  const out = new Uint8ClampedArray(src.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let dy = -1; dy <= 1 && !v; dy++) {
        for (let dx = -1; dx <= 1 && !v; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
          if (src[yy * w + xx]) v = 255;
        }
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

function _maskBinToBlob() {
  // The mask was computed at the working scale. For Save / Apply /
  // Download we always emit the mask at A's native resolution so it
  // pairs cleanly with the original image (which is what /api/inpaint
  // and the library consumers expect). Resize via nearest-neighbour
  // so the binary mask stays binary.
  const wW = _diff.canvasMask.width, wH = _diff.canvasMask.height;
  const nativeW = _diff.imgA.naturalWidth, nativeH = _diff.imgA.naturalHeight;
  const work = document.createElement('canvas');
  work.width = wW; work.height = wH;
  const wctx = work.getContext('2d');
  const rgba = new Uint8ClampedArray(wW * wH * 4);
  for (let p = 0, q = 0; p < _diff.lastMaskBin.length; p++, q += 4) {
    const v = _diff.lastMaskBin[p] > 127 ? 255 : 0;
    rgba[q] = v; rgba[q+1] = v; rgba[q+2] = v; rgba[q+3] = 255;
  }
  wctx.putImageData(new ImageData(rgba, wW, wH), 0, 0);

  if (wW === nativeW && wH === nativeH) {
    return new Promise(r => work.toBlob(r, 'image/png'));
  }
  const out = document.createElement('canvas');
  out.width = nativeW; out.height = nativeH;
  const octx = out.getContext('2d');
  octx.imageSmoothingEnabled = false;  // preserve binary edges
  octx.drawImage(work, 0, 0, nativeW, nativeH);
  return new Promise(r => out.toBlob(r, 'image/png'));
}

async function saveDiffMask() {
  if (!_diff.lastMaskBin || !_diff.fileA) { showToast('Need both images loaded'); return; }
  const blob = await _maskBinToBlob();
  const fd = new FormData();
  fd.append('mask', blob, 'mask.png');
  fd.append('source_image', _diff.fileA);
  showToast('Saving to library...');
  try {
    const r = await fetch(API + '/api/masks/from-diff', { method: 'POST', body: fd });
    if (r.ok) {
      const data = await r.json();
      showToast(`Saved as "${data.name}"`);
    } else {
      showToast('Save failed: ' + r.status);
    }
  } catch (e) { showToast('Save error: ' + e.message); }
}

async function applyDiffMaskToEditor() {
  if (!_diff.lastMaskBin || !_diff.fileA) { showToast('Load both images first'); return; }
  _uploadedFile = _diff.fileA;
  _uploadedType = 'image';
  const blob = await _maskBinToBlob();
  showPanel('mask');
  initMaskEditor(URL.createObjectURL(_diff.fileA), URL.createObjectURL(blob));
  showToast('Mask loaded — refine and apply');
}

async function downloadDiffMask() {
  if (!_diff.lastMaskBin) { showToast('Adjust the sliders first'); return; }
  const blob = await _maskBinToBlob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'diff_mask.png';
  a.click();
}
