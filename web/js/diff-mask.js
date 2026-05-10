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
  canvasA: null, canvasB: null, canvasMask: null,
  ctxA: null, ctxB: null, ctxMask: null,
  // tunables — synced to slider DOM each render
  threshold: 30,
  dilate: 3,
  blur: 1,
  channels: 'lum',  // 'lum' or 'rgb' — luminance is more robust against tiny color shifts
};

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
    </div>
    <div class="diff-stage" id="diffStage" style="display:none">
      <div class="diff-canvas-wrap">
        <canvas id="diffCanvasA"></canvas>
        <canvas id="diffCanvasMask"></canvas>
      </div>
      <div class="diff-actions">
        <button class="btn btn-secondary" id="diffCompleteLinesBtn">Complete Partial Lines</button>
        <button class="btn btn-primary" id="diffSaveBtn">Save to Library</button>
        <button class="btn btn-secondary" id="diffApplyBtn">Apply (manual editor)</button>
        <button class="btn btn-secondary" id="diffDownloadBtn">Download PNG</button>
      </div>
    </div>
  `;

  document.getElementById('diffFileA').addEventListener('change', e => loadDiffImage(e.target.files[0], 'A'));
  document.getElementById('diffFileB').addEventListener('change', e => loadDiffImage(e.target.files[0], 'B'));
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
  const c = document.getElementById('diffCanvasA');
  const m = document.getElementById('diffCanvasMask');
  c.width = a.naturalWidth; c.height = a.naturalHeight;
  m.width = a.naturalWidth; m.height = a.naturalHeight;
  _diff.canvasA = c; _diff.canvasMask = m;
  _diff.ctxA = c.getContext('2d', { willReadFrequently: true });
  _diff.ctxMask = m.getContext('2d');
  _diff.ctxA.drawImage(a, 0, 0);

  // Cache image B resampled to A's resolution so render is fast.
  const tmp = document.createElement('canvas');
  tmp.width = a.naturalWidth; tmp.height = a.naturalHeight;
  tmp.getContext('2d').drawImage(_diff.imgB, 0, 0, a.naturalWidth, a.naturalHeight);
  _diff.dataA = _diff.ctxA.getImageData(0, 0, a.naturalWidth, a.naturalHeight);
  _diff.dataB = tmp.getContext('2d').getImageData(0, 0, a.naturalWidth, a.naturalHeight);
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
  const w = _diff.canvasMask.width, h = _diff.canvasMask.height;
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  const ctx = tmp.getContext('2d');
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let p = 0, q = 0; p < _diff.lastMaskBin.length; p++, q += 4) {
    const v = _diff.lastMaskBin[p] > 127 ? 255 : 0;
    rgba[q] = v; rgba[q+1] = v; rgba[q+2] = v; rgba[q+3] = 255;
  }
  ctx.putImageData(new ImageData(rgba, w, h), 0, 0);
  return new Promise(r => tmp.toBlob(r, 'image/png'));
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
