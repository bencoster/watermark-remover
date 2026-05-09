let _maskCanvas = null;
let _maskCtx = null;
let _maskDrawing = false;
let _maskBrushSize = 20;

function initMaskEditor(imageUrl, preloadedMaskUrl) {
  const panel = document.getElementById('panel-mask');
  panel.innerHTML = `
    <div class="mask-editor">
      <div class="mask-toolbar">
        <label>Brush: <input type="range" id="brushSize" min="5" max="100" value="20"></label>
        <button class="btn btn-sm" id="btnClearMask">Clear Mask</button>
        <button class="btn btn-primary" id="btnSubmitMask">Remove Watermark</button>
      </div>
      <div class="mask-canvas-wrap" id="maskCanvasWrap">
        <img id="maskBgImg" src="${imageUrl}">
        <canvas id="maskCanvas"></canvas>
      </div>
    </div>
  `;

  const bgImg = document.getElementById('maskBgImg');
  bgImg.onload = () => {
    const canvas = document.getElementById('maskCanvas');
    canvas.width = bgImg.naturalWidth;
    canvas.height = bgImg.naturalHeight;
    _maskCanvas = canvas;
    _maskCtx = canvas.getContext('2d');
    _maskCtx.lineCap = 'round';
    _maskCtx.lineJoin = 'round';

    if (preloadedMaskUrl) {
      const maskImg = new Image();
      maskImg.onload = () => { _maskCtx.drawImage(maskImg, 0, 0); };
      maskImg.src = preloadedMaskUrl;
    }

    canvas.addEventListener('mousedown', e => { _maskDrawing = true; _maskDraw(e); });
    canvas.addEventListener('mousemove', e => { if (_maskDrawing) _maskDraw(e); });
    canvas.addEventListener('mouseup', () => _maskDrawing = false);
    canvas.addEventListener('mouseleave', () => _maskDrawing = false);
  };

  document.getElementById('brushSize').addEventListener('input', e => _maskBrushSize = +e.target.value);
  document.getElementById('btnClearMask').addEventListener('click', () => {
    if (_maskCtx) _maskCtx.clearRect(0, 0, _maskCanvas.width, _maskCanvas.height);
  });
  document.getElementById('btnSubmitMask').addEventListener('click', submitWithMask);
}

function _maskDraw(e) {
  const rect = _maskCanvas.getBoundingClientRect();
  const scaleX = _maskCanvas.width / rect.width;
  const scaleY = _maskCanvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;
  _maskCtx.fillStyle = 'rgba(255, 255, 255, 1)';
  _maskCtx.beginPath();
  _maskCtx.arc(x, y, _maskBrushSize * scaleX / 2, 0, Math.PI * 2);
  _maskCtx.fill();
}

async function submitWithMask() {
  if (!_uploadedFile || !_maskCanvas) return;

  _maskCanvas.toBlob(async (maskBlob) => {
    const fd = new FormData();
    fd.append('file', _uploadedFile);
    fd.append('mask', maskBlob, 'mask.png');

    if (_uploadedType === 'video') {
      fd.append('auto_detect', 'false');
      showToast('Submitting video job...');
      try {
        const r = await fetch(API + '/api/video', { method: 'POST', body: fd });
        const data = await r.json();
        if (data.job_id) {
          showToast('Video job submitted: ' + data.job_id);
          showPanel('jobs');
        } else {
          showToast('Error: ' + (data.error || 'unknown'));
        }
      } catch (e) {
        showToast('Submit error: ' + e.message);
      }
    } else {
      showToast('Inpainting...');
      try {
        const r = await fetch(API + '/api/inpaint', { method: 'POST', body: fd });
        if (r.ok) {
          const blob = await r.blob();
          const url = URL.createObjectURL(blob);
          _showResult(url);
          showToast('Done!');
        } else {
          const err = await r.json();
          showToast('Error: ' + (err.error || 'unknown'));
        }
      } catch (e) {
        showToast('Inpaint error: ' + e.message);
      }
    }
  }, 'image/png');
}

function _showResult(url) {
  const panel = document.getElementById('panel-mask');
  panel.innerHTML += `
    <div class="result-preview">
      <h3>Result</h3>
      <img src="${url}" class="result-img">
      <a href="${url}" download="inpainted.png" class="btn btn-primary">Download</a>
    </div>
  `;
}
