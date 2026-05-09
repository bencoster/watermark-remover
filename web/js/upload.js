let _uploadedFile = null;
let _uploadedType = null;

function initUpload() {
  const panel = document.getElementById('panel-upload');
  panel.innerHTML = `
    <div class="upload-zone" id="uploadZone">
      <div class="upload-icon">&#128206;</div>
      <p>Drag &amp; drop an image or video here</p>
      <p class="upload-hint">or click to browse</p>
      <input type="file" id="fileInput" accept="image/*,video/*" hidden>
    </div>
    <div id="uploadPreview" class="upload-preview" style="display:none">
      <div class="preview-container">
        <img id="previewImg" style="display:none">
        <video id="previewVid" controls style="display:none"></video>
      </div>
      <div class="preview-actions">
        <button class="btn btn-primary" id="btnOneClick">One-Click Remove</button>
        <button class="btn btn-secondary" id="btnAutoDetect">Review Mask First</button>
        <button class="btn btn-secondary" id="btnManualMask">Draw Mask Manually</button>
        <button class="btn btn-danger" id="btnClearUpload">Clear</button>
      </div>
      <div class="strip-mode-toggle">
        <span class="strip-mode-label" title="Tiled = repeating pattern across whole image (stock photos). Single = one isolated logo or stamp.">Watermark type:</span>
        <label title="Best for stock photos with the watermark repeating in a grid (Dreamstime, Shutterstock, Getty). Catches every tile but masks ~50% of the image."><input type="radio" name="detectMode" value="recall" checked> Tiled / stock-photo grid</label>
        <label title="Best for a single isolated watermark — corner stamp, one brand logo. Uses a tighter mask to avoid touching surrounding subject content."><input type="radio" name="detectMode" value="precision"> Single mark / corner stamp</label>
      </div>
      <div class="strip-mode-toggle">
        <span class="strip-mode-label">Bottom strip bar:</span>
        <label><input type="radio" name="stripMode" value="inpaint" checked> Inpaint (rebuild content)</label>
        <label><input type="radio" name="stripMode" value="crop"> Crop (cut bar off)</label>
      </div>
      <div id="oneClickResult" class="result-preview" style="display:none"></div>
    </div>
  `;

  const zone = document.getElementById('uploadZone');
  const input = document.getElementById('fileInput');

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  input.addEventListener('change', () => { if (input.files.length) handleFile(input.files[0]); });

  document.getElementById('btnOneClick').addEventListener('click', oneClickRemove);
  document.getElementById('btnAutoDetect').addEventListener('click', autoDetect);
  document.getElementById('btnManualMask').addEventListener('click', () => openMaskEditor());
  document.getElementById('btnClearUpload').addEventListener('click', clearUpload);
}

async function oneClickRemove() {
  if (!_uploadedFile) return;
  if (_uploadedType === 'video') {
    showToast('Video one-click not supported yet - use mask editor');
    return;
  }
  const fd = new FormData();
  fd.append('file', _uploadedFile);
  const stripMode = (document.querySelector('input[name=stripMode]:checked') || {}).value || 'inpaint';
  const detectMode = (document.querySelector('input[name=detectMode]:checked') || {}).value || 'recall';
  fd.append('strip_mode', stripMode);
  fd.append('detect_mode', detectMode);
  showToast(`Removing watermark (${detectMode}, strip:${stripMode})... ~20s on CPU`);

  try {
    const r = await fetch(API + '/api/auto', { method: 'POST', body: fd });
    if (r.ok) {
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const el = document.getElementById('oneClickResult');
      el.style.display = 'block';
      el.innerHTML = `
        <h3>Result</h3>
        <img src="${url}" class="result-img">
        <a href="${url}" download="cleaned.png" class="btn btn-primary">Download</a>
      `;
      showToast('Done!');
    } else if (r.status === 422) {
      showToast('No watermark detected - try manual mask');
    } else {
      showToast('Error: ' + r.status);
    }
  } catch (e) {
    showToast('Error: ' + e.message);
  }
}

function handleFile(file) {
  _uploadedFile = file;
  _uploadedType = file.type.startsWith('video') ? 'video' : 'image';

  document.getElementById('uploadZone').style.display = 'none';
  const preview = document.getElementById('uploadPreview');
  preview.style.display = 'block';

  if (_uploadedType === 'image') {
    const img = document.getElementById('previewImg');
    img.src = URL.createObjectURL(file);
    img.style.display = 'block';
    document.getElementById('previewVid').style.display = 'none';
  } else {
    const vid = document.getElementById('previewVid');
    vid.src = URL.createObjectURL(file);
    vid.style.display = 'block';
    document.getElementById('previewImg').style.display = 'none';
  }
}

async function autoDetect() {
  if (!_uploadedFile) return;
  const fd = new FormData();
  fd.append('file', _uploadedFile);

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
  showPanel('mask');
  const imgUrl = URL.createObjectURL(_uploadedFile);
  initMaskEditor(imgUrl, preloadedMaskUrl);
}

function clearUpload() {
  _uploadedFile = null;
  _uploadedType = null;
  document.getElementById('uploadZone').style.display = '';
  document.getElementById('uploadPreview').style.display = 'none';
  document.getElementById('previewImg').style.display = 'none';
  document.getElementById('previewVid').style.display = 'none';
}
