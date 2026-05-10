// Upload + batch queue. State design:
//   _queue : list of { id, file, type, url, status, resultUrl, error,
//                       maskSource, elapsedMs }
//     - status flows: 'queued' -> 'running' -> 'done' | 'failed'
//   _currentIndex : index of the focused queue item (preview / mask
//                    editor / library-apply target)
//
// Default load: on init we pull /api/sample-image so the preview is
// populated immediately and the user can hit One-Click Remove without
// uploading anything.
//
// Drop / click behaviour: the preview area is also a drop target, and
// clicking the displayed image (or anywhere in the preview wrap) opens
// the file picker. This means there's no "go back to upload" step —
// new images replace or append to the queue in place.

let _queue = [];
let _currentIndex = 0;
// Back-compat shims for other JS modules that read these names.
let _uploadedFile = null;
let _uploadedType = null;

function _activeItem() { return _queue[_currentIndex] || null; }

function initUpload() {
  const panel = document.getElementById('panel-upload');
  panel.innerHTML = `
    <div class="upload-zone" id="uploadZone" style="display:none">
      <div class="upload-icon">&#128206;</div>
      <p>Drag &amp; drop image(s) here</p>
      <p class="upload-hint">or click to browse — multi-select queues a batch</p>
      <input type="file" id="fileInput" accept="image/*,video/*" hidden multiple>
    </div>
    <div id="uploadPreview" class="upload-preview">
      <div class="preview-container" id="previewContainer" title="Click to browse, or drag images here">
        <img id="previewImg" style="display:none">
        <video id="previewVid" controls style="display:none"></video>
      </div>
      <div id="batchStrip" class="batch-strip"></div>
      <div class="preview-actions">
        <button class="btn btn-primary" id="btnOneClick">One-Click Remove</button>
        <button class="btn btn-secondary" id="btnAutoDetect">Review Mask First</button>
        <button class="btn btn-secondary" id="btnManualMask">Draw Mask Manually</button>
        <button class="btn btn-secondary" id="btnAddMore">Add More</button>
        <button class="btn btn-danger" id="btnClearUpload">Clear All</button>
      </div>
      <div class="strip-mode-toggle">
        <span class="strip-mode-label" title="Auto picks based on whether the watermark forms a 2D grid. Override to force a specific mode if Auto picks wrong.">Watermark type:</span>
        <label title="Detects whether the watermark is a tiled grid or a single mark and picks the right mode automatically."><input type="radio" name="detectMode" value="auto" checked> Auto-detect</label>
        <label title="Force tiled / stock-photo mode (Dreamstime, Shutterstock, Getty). Catches every tile."><input type="radio" name="detectMode" value="recall"> Force tiled</label>
        <label title="Force single mark / corner stamp mode. Tighter mask, won't catch tiled patterns."><input type="radio" name="detectMode" value="precision"> Force single</label>
      </div>
      <div class="strip-mode-toggle">
        <span class="strip-mode-label" title="When ON, the server checks if any saved library mask matches the input dimensions and uses it directly. Library masks come from the Diff Mask tool and give SaaS-equivalent quality (+4 dB PSNR vs auto-detect on the canonical fixture).">Library masks:</span>
        <label title="Use a saved library mask if its dimensions match the input. Best for repeat watermarks."><input type="radio" name="libraryMask" value="auto" checked> Auto-match</label>
        <label title="Always run detection from scratch."><input type="radio" name="libraryMask" value="off"> Always detect</label>
      </div>
      <div class="strip-mode-toggle">
        <span class="strip-mode-label">Bottom strip bar:</span>
        <label><input type="radio" name="stripMode" value="inpaint" checked> Inpaint (rebuild content)</label>
        <label><input type="radio" name="stripMode" value="crop"> Crop (cut bar off)</label>
      </div>
      <div class="strip-mode-toggle">
        <span class="strip-mode-label" title="SDXL re-synthesises plausible texture under the strip bar; TELEA only propagates colour from the rows above. SDXL needs ~11 GB free VRAM and adds ~15 s on GPU; falls back to TELEA when unavailable.">Strip refill engine:</span>
        <label title="Fast pixel-propagation. Works without GPU."><input type="radio" name="stripEngine" value="telea" checked> Fast (TELEA)</label>
        <label title="Diffusion-based texture synthesis. Better on bars covering carpet/wood/sky. Needs GPU."><input type="radio" name="stripEngine" value="sdxl"> Quality (SDXL)</label>
      </div>
      <div class="strip-mode-toggle">
        <span class="strip-mode-label" title="Grounding DINO is the only multi-instance text-prompted detector that handles tiled stock-photo watermarks (97% recall vs 64% for the default detector). When enabled, GD's loose bounding boxes are intersected with the existing pixel heuristic to give tight pixel masks. ~4 s extra on CPU, ~700 MB first-run download.">Tiled-pattern detector:</span>
        <label title="Default ConvNeXt + Grad-CAM only. Faster, no extra download."><input type="radio" name="groundingDino" value="off" checked> Off (default)</label>
        <label title="Add Grounding DINO refinement. Best for tiled stock-photo watermarks. Empirically improves IoU 0.25 -> 0.31 on the canonical fixture."><input type="radio" name="groundingDino" value="on"> On (Grounding DINO)</label>
      </div>
      <div class="strip-mode-toggle">
        <span class="strip-mode-label" title="SAM 2 over-segments tiled stock-photo watermarks (it isolates each logo but misses the diagonal text between them). Recommended only for single-mark / corner-stamp images. SAM 3.1 may handle tiled cases better but requires Python 3.12 + Meta access request.">Mask refinement (advanced):</span>
        <label title="No SAM refinement. Default Grad-CAM blob mask."><input type="radio" name="samRefine" value="off" checked> Off (default)</label>
        <label title="SAM 2 with per-centroid point prompts. Best for single watermarks. WARNING: drops recall on tiled stock-photo watermarks."><input type="radio" name="samRefine" value="sam2"> SAM 2 (single-mark only)</label>
        <label title="Text-prompted SAM 3.1. Requires Python 3.12 venv + Meta checkpoint access. Falls back to Off when unavailable."><input type="radio" name="samRefine" value="sam3.1"> SAM 3.1 (when available)</label>
      </div>
      <div id="oneClickResult" class="result-preview" style="display:none"></div>
      <div id="batchResults" class="batch-results"></div>
    </div>
  `;

  const input = document.getElementById('fileInput');
  const previewContainer = document.getElementById('previewContainer');
  const zone = document.getElementById('uploadZone');

  // Click-to-browse on the preview AND zone. The hidden input has the
  // multiple flag so users can pick a batch in one shot.
  previewContainer.addEventListener('click', () => input.click());
  zone.addEventListener('click', () => input.click());
  input.addEventListener('change', () => {
    if (input.files && input.files.length) handleFiles(Array.from(input.files));
    input.value = '';  // allow re-selecting same files
  });

  // Drop handling everywhere in the preview area + the zone.
  for (const dropTarget of [previewContainer, zone, document.getElementById('panel-upload')]) {
    dropTarget.addEventListener('dragover', e => {
      e.preventDefault();
      previewContainer.classList.add('dragover');
      zone.classList.add('dragover');
    });
    dropTarget.addEventListener('dragleave', () => {
      previewContainer.classList.remove('dragover');
      zone.classList.remove('dragover');
    });
    dropTarget.addEventListener('drop', e => {
      e.preventDefault();
      previewContainer.classList.remove('dragover');
      zone.classList.remove('dragover');
      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length) handleFiles(Array.from(files));
    });
  }

  document.getElementById('btnOneClick').addEventListener('click', oneClickRemove);
  document.getElementById('btnAutoDetect').addEventListener('click', autoDetect);
  document.getElementById('btnManualMask').addEventListener('click', () => openMaskEditor());
  document.getElementById('btnClearUpload').addEventListener('click', clearUpload);
  document.getElementById('btnAddMore').addEventListener('click', () => input.click());

  // Default-load the canonical sample so the user can hit One-Click
  // Remove without uploading anything.
  loadDefaultSample();
}

async function loadDefaultSample() {
  try {
    const r = await fetch(API + '/api/sample-image');
    if (!r.ok) return;
    const blob = await r.blob();
    const file = new File([blob], 'dreamstime_18829755_sample.jpg', { type: blob.type });
    handleFiles([file], { isSample: true });
  } catch (e) {
    // No big deal — preview just stays empty until the user drops a file.
  }
}

function handleFiles(files, opts = {}) {
  // Filter to image/video, dedup by name+size+lastModified
  const accepted = files.filter(f => f.type.startsWith('image/') || f.type.startsWith('video/'));
  if (!accepted.length) {
    showToast('No image/video files in drop');
    return;
  }
  // If the existing queue is just the default sample, replace it on
  // first real drop so the sample doesn't pollute a real batch.
  if (_queue.length === 1 && _queue[0].isSample && !opts.isSample) {
    _queue = [];
    _currentIndex = 0;
  }
  for (const f of accepted) {
    _queue.push({
      id: Math.random().toString(36).slice(2, 10),
      file: f,
      type: f.type.startsWith('video') ? 'video' : 'image',
      url: URL.createObjectURL(f),
      status: 'queued',
      resultUrl: null,
      error: null,
      maskSource: null,
      isSample: !!opts.isSample,
    });
  }
  _currentIndex = _queue.length - 1;
  renderBatchStrip();
  renderActivePreview();
}

function renderActivePreview() {
  const item = _activeItem();
  const img = document.getElementById('previewImg');
  const vid = document.getElementById('previewVid');
  if (!item) {
    img.style.display = 'none';
    vid.style.display = 'none';
    _uploadedFile = null;
    _uploadedType = null;
    return;
  }
  _uploadedFile = item.file;
  _uploadedType = item.type;
  if (item.type === 'video') {
    vid.src = item.url; vid.style.display = 'block';
    img.style.display = 'none';
  } else {
    img.src = item.url; img.style.display = 'block';
    vid.style.display = 'none';
  }
}

function renderBatchStrip() {
  const strip = document.getElementById('batchStrip');
  if (!strip) return;
  if (_queue.length <= 1 && _queue[0]?.isSample) {
    strip.innerHTML = `<div class="batch-hint">${_queue[0] ? '<em>Sample image loaded — click or drop to replace.</em>' : ''}</div>`;
    return;
  }
  if (_queue.length === 0) {
    strip.innerHTML = '';
    return;
  }
  strip.innerHTML = _queue.map((it, i) => `
    <div class="batch-item ${i === _currentIndex ? 'active' : ''} batch-${it.status}" data-i="${i}">
      <img src="${it.url}" alt="">
      <span class="batch-label">${it.status === 'done' ? '✓' : it.status === 'failed' ? '!' : (i + 1)}</span>
    </div>
  `).join('');
  strip.querySelectorAll('.batch-item').forEach(el => {
    el.addEventListener('click', () => {
      _currentIndex = +el.dataset.i;
      renderActivePreview();
      renderBatchStrip();
    });
  });
}

async function oneClickRemove() {
  if (_queue.length === 0) {
    showToast('No images loaded');
    return;
  }
  if (_queue.some(it => it.type === 'video')) {
    showToast('Video files require manual mask — use Draw Mask Manually');
    return;
  }

  const stripMode = (document.querySelector('input[name=stripMode]:checked') || {}).value || 'inpaint';
  const detectMode = (document.querySelector('input[name=detectMode]:checked') || {}).value || 'auto';
  const libraryMask = (document.querySelector('input[name=libraryMask]:checked') || {}).value || 'auto';
  const stripEngine = (document.querySelector('input[name=stripEngine]:checked') || {}).value || 'telea';
  const samRefine = (document.querySelector('input[name=samRefine]:checked') || {}).value || 'off';
  const groundingDino = (document.querySelector('input[name=groundingDino]:checked') || {}).value || 'off';

  const batchEl = document.getElementById('batchResults');
  batchEl.innerHTML = '';

  const pending = _queue.filter(it => it.type === 'image');
  if (!pending.length) return;

  showToast(`Processing ${pending.length} image${pending.length > 1 ? 's' : ''}...`);

  // Sequential — the server holds one LaMa instance and SDXL would OOM
  // if we fired requests in parallel.
  for (const item of pending) {
    item.status = 'running';
    renderBatchStrip();
    appendBatchCard(item);

    const fd = new FormData();
    fd.append('file', item.file);
    fd.append('strip_mode', stripMode);
    fd.append('detect_mode', detectMode);
    fd.append('library_mask', libraryMask);
    fd.append('strip_engine', stripEngine);
    fd.append('sam_refine', samRefine);
    fd.append('grounding_dino', groundingDino);

    const t0 = performance.now();
    try {
      const r = await fetch(API + '/api/auto', { method: 'POST', body: fd });
      item.elapsedMs = Math.round(performance.now() - t0);
      if (r.ok) {
        const blob = await r.blob();
        item.resultUrl = URL.createObjectURL(blob);
        item.maskSource = r.headers.get('X-Mask-Source') || 'detect';
        item.status = 'done';
      } else if (r.status === 422) {
        item.status = 'failed';
        item.error = 'No watermark detected';
      } else {
        item.status = 'failed';
        item.error = 'HTTP ' + r.status;
      }
    } catch (e) {
      item.status = 'failed';
      item.error = e.message;
    }
    renderBatchStrip();
    updateBatchCard(item);
  }

  const ok = pending.filter(it => it.status === 'done').length;
  const fail = pending.length - ok;
  showToast(fail ? `${ok} done, ${fail} failed` : `Done — ${ok}/${ok}`);
}

function appendBatchCard(item) {
  const batchEl = document.getElementById('batchResults');
  const card = document.createElement('div');
  card.className = 'batch-card';
  card.id = 'batchcard-' + item.id;
  card.innerHTML = `
    <img class="batch-card-thumb" src="${item.url}">
    <div class="batch-card-body">
      <div class="batch-card-name">${escapeText(item.file.name)}</div>
      <div class="batch-card-status">Running...</div>
    </div>
  `;
  batchEl.appendChild(card);
}

function updateBatchCard(item) {
  const card = document.getElementById('batchcard-' + item.id);
  if (!card) return;
  const status = card.querySelector('.batch-card-status');
  if (item.status === 'done') {
    const tag = item.maskSource === 'library'
      ? '<span class="badge badge-strip" title="Used a saved library mask">library</span>'
      : '';
    status.innerHTML = `Done in ${(item.elapsedMs / 1000).toFixed(1)}s ${tag} <a href="${item.resultUrl}" download="cleaned_${escapeAttr(item.file.name)}.png" class="btn btn-sm btn-primary">Download</a>`;
    const thumb = card.querySelector('.batch-card-thumb');
    thumb.src = item.resultUrl;
    thumb.title = 'Click to compare original / result';
    let showingResult = true;
    thumb.addEventListener('click', () => {
      thumb.src = showingResult ? item.url : item.resultUrl;
      showingResult = !showingResult;
    });
  } else if (item.status === 'failed') {
    status.innerHTML = `<span class="batch-error">Failed: ${escapeText(item.error || 'unknown')}</span>`;
  }
}

async function autoDetect() {
  const item = _activeItem();
  if (!item) { showToast('Load an image first'); return; }
  const fd = new FormData();
  fd.append('file', item.file);
  showToast('Detecting watermark...');
  try {
    const r = await fetch(API + '/api/detect', { method: 'POST', body: fd });
    if (r.ok) {
      const ct = r.headers.get('content-type');
      if (ct && ct.includes('image')) {
        const blob = await r.blob();
        openMaskEditor(URL.createObjectURL(blob));
        showToast('Watermark detected - review mask');
      } else {
        const data = await r.json();
        if (!data.detected) {
          showToast('No watermark detected - draw mask manually');
          openMaskEditor();
        }
      }
    } else {
      showToast('Detection failed');
      openMaskEditor();
    }
  } catch (e) {
    showToast('Detection error: ' + e.message);
    openMaskEditor();
  }
}

function openMaskEditor(preloadedMaskUrl) {
  const item = _activeItem();
  if (!item) { showToast('Load an image first'); return; }
  showPanel('mask');
  initMaskEditor(item.url, preloadedMaskUrl);
}

function clearUpload() {
  for (const it of _queue) URL.revokeObjectURL(it.url);
  _queue = [];
  _currentIndex = 0;
  document.getElementById('batchResults').innerHTML = '';
  renderBatchStrip();
  renderActivePreview();
  // Reload the default sample so the preview isn't empty.
  loadDefaultSample();
}

function escapeText(s) {
  const div = document.createElement('div');
  div.textContent = String(s);
  return div.innerHTML;
}
function escapeAttr(s) { return escapeText(s).replace(/"/g, '&quot;').replace(/'/g, '&#39;'); }
