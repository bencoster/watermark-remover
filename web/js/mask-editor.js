let _maskCanvas = null;
let _maskCtx = null;
let _maskDrawing = false;
let _maskBrushSize = 20;
let _maskTool = 'brush';   // 'brush' | 'eraser'
let _maskHistory = [];     // ImageData stack
let _maskRedo = [];
const _MAX_HISTORY = 30;

function initMaskEditor(imageUrl, preloadedMaskUrl) {
  const panel = document.getElementById('panel-mask');
  panel.innerHTML = `
    <div class="mask-editor">
      <div class="mask-toolbar">
        <div class="tool-group">
          <button class="btn btn-sm tool-btn active" data-tool="brush" id="toolBrush">Brush</button>
          <button class="btn btn-sm tool-btn" data-tool="eraser" id="toolEraser">Eraser</button>
        </div>
        <label>Size <input type="range" id="brushSize" min="5" max="200" value="20"> <span id="brushSizeVal">20</span></label>
        <div class="tool-group">
          <button class="btn btn-sm" id="btnUndo" disabled>Undo</button>
          <button class="btn btn-sm" id="btnRedo" disabled>Redo</button>
        </div>
        <button class="btn btn-sm btn-secondary" id="btnInvertMask">Invert</button>
        <button class="btn btn-sm" id="btnClearMask">Clear</button>
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
    _maskHistory = [];
    _maskRedo = [];

    const finishLoad = () => { _pushHistory(); _updateUndoRedo(); };
    if (preloadedMaskUrl) {
      const maskImg = new Image();
      maskImg.onload = () => { _maskCtx.drawImage(maskImg, 0, 0); finishLoad(); };
      maskImg.src = preloadedMaskUrl;
    } else {
      finishLoad();
    }

    canvas.addEventListener('mousedown', e => { _maskDrawing = true; _maskDraw(e); });
    canvas.addEventListener('mousemove', e => { if (_maskDrawing) _maskDraw(e); });
    const endStroke = () => { if (_maskDrawing) { _maskDrawing = false; _pushHistory(); _maskRedo = []; _updateUndoRedo(); } };
    canvas.addEventListener('mouseup', endStroke);
    canvas.addEventListener('mouseleave', endStroke);
  };

  document.getElementById('brushSize').addEventListener('input', e => {
    _maskBrushSize = +e.target.value;
    document.getElementById('brushSizeVal').textContent = e.target.value;
  });
  document.getElementById('toolBrush').addEventListener('click', () => _setMaskTool('brush'));
  document.getElementById('toolEraser').addEventListener('click', () => _setMaskTool('eraser'));
  document.getElementById('btnUndo').addEventListener('click', _undoMask);
  document.getElementById('btnRedo').addEventListener('click', _redoMask);
  document.getElementById('btnInvertMask').addEventListener('click', _invertMask);
  document.getElementById('btnClearMask').addEventListener('click', () => {
    if (!_maskCtx) return;
    _maskCtx.clearRect(0, 0, _maskCanvas.width, _maskCanvas.height);
    _pushHistory(); _maskRedo = []; _updateUndoRedo();
  });
  document.getElementById('btnSubmitMask').addEventListener('click', submitWithMask);

  // Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z keyboard support while panel is open
  document.addEventListener('keydown', _maskHotkeys);
}

function _setMaskTool(tool) {
  _maskTool = tool;
  document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(tool === 'brush' ? 'toolBrush' : 'toolEraser').classList.add('active');
}

function _maskDraw(e) {
  const rect = _maskCanvas.getBoundingClientRect();
  const scaleX = _maskCanvas.width / rect.width;
  const scaleY = _maskCanvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;
  const radius = _maskBrushSize * scaleX / 2;
  // Eraser uses destination-out so it punches transparent holes;
  // brush uses source-over so strokes are additive.
  _maskCtx.globalCompositeOperation = _maskTool === 'eraser' ? 'destination-out' : 'source-over';
  _maskCtx.fillStyle = 'rgba(255, 255, 255, 1)';
  _maskCtx.beginPath();
  _maskCtx.arc(x, y, radius, 0, Math.PI * 2);
  _maskCtx.fill();
  _maskCtx.globalCompositeOperation = 'source-over';
}

function _pushHistory() {
  if (!_maskCtx) return;
  const snap = _maskCtx.getImageData(0, 0, _maskCanvas.width, _maskCanvas.height);
  _maskHistory.push(snap);
  if (_maskHistory.length > _MAX_HISTORY) _maskHistory.shift();
}

function _undoMask() {
  if (_maskHistory.length < 2) return;
  _maskRedo.push(_maskHistory.pop());
  _maskCtx.putImageData(_maskHistory[_maskHistory.length - 1], 0, 0);
  _updateUndoRedo();
}

function _redoMask() {
  if (_maskRedo.length === 0) return;
  const snap = _maskRedo.pop();
  _maskHistory.push(snap);
  _maskCtx.putImageData(snap, 0, 0);
  _updateUndoRedo();
}

function _updateUndoRedo() {
  const u = document.getElementById('btnUndo');
  const r = document.getElementById('btnRedo');
  if (u) u.disabled = _maskHistory.length < 2;
  if (r) r.disabled = _maskRedo.length === 0;
}

function _invertMask() {
  if (!_maskCtx) return;
  const w = _maskCanvas.width, h = _maskCanvas.height;
  const data = _maskCtx.getImageData(0, 0, w, h);
  // We treat any non-zero alpha as masked, and produce an inverted mask:
  // unmasked pixels become opaque white, masked pixels become transparent.
  for (let i = 0; i < data.data.length; i += 4) {
    const a = data.data[i + 3];
    if (a > 0) {
      data.data[i + 3] = 0;
    } else {
      data.data[i] = 255; data.data[i + 1] = 255; data.data[i + 2] = 255; data.data[i + 3] = 255;
    }
  }
  _maskCtx.putImageData(data, 0, 0);
  _pushHistory(); _maskRedo = []; _updateUndoRedo();
}

function _maskHotkeys(e) {
  // Only react when the mask panel is the active one.
  const panel = document.getElementById('panel-mask');
  if (!panel || !panel.classList.contains('active')) return;
  if (e.ctrlKey && (e.key === 'z' || e.key === 'Z') && !e.shiftKey) {
    e.preventDefault(); _undoMask();
  } else if (e.ctrlKey && (e.key === 'y' || e.key === 'Y' || (e.shiftKey && (e.key === 'z' || e.key === 'Z')))) {
    e.preventDefault(); _redoMask();
  } else if (e.key === 'b' || e.key === 'B') {
    _setMaskTool('brush');
  } else if (e.key === 'e' || e.key === 'E') {
    _setMaskTool('eraser');
  }
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
  const existing = panel.querySelector('.result-preview');
  if (existing) existing.remove();
  const div = document.createElement('div');
  div.className = 'result-preview';
  div.innerHTML = `
    <h3>Result</h3>
    <img src="${url}" class="result-img">
    <a href="${url}" download="inpainted.png" class="btn btn-primary">Download</a>
  `;
  panel.appendChild(div);
}
