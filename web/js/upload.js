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
        <button class="btn btn-primary" id="btnAutoDetect">Auto-Detect Watermark</button>
        <button class="btn btn-secondary" id="btnManualMask">Draw Mask Manually</button>
        <button class="btn btn-danger" id="btnClearUpload">Clear</button>
      </div>
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

  document.getElementById('btnAutoDetect').addEventListener('click', autoDetect);
  document.getElementById('btnManualMask').addEventListener('click', () => openMaskEditor());
  document.getElementById('btnClearUpload').addEventListener('click', clearUpload);
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
