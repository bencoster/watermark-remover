// Reusable before/after viewer modal.
//
// Public API:
//   openBeforeAfter(beforeUrl, afterUrl, opts?)
//      opts: { titleBefore, titleAfter, fileName }
//
// Four view modes (toggle via tabs at top):
//   1. Slider     — drag a vertical handle to wipe between B and A
//   2. Side-by-side — both at half-width
//   3. Toggle     — click anywhere to flip; useful for spot-the-diff
//   4. Difference — pixel diff overlay heat-mapped (red = changed)
//
// Modal closes on Esc or by clicking the overlay outside the panel.

(function () {
  let _modal = null;
  let _state = null;

  function ensureModal() {
    if (_modal) return _modal;
    _modal = document.createElement('div');
    _modal.id = 'baModal';
    _modal.className = 'ba-modal';
    _modal.style.display = 'none';
    _modal.innerHTML = `
      <div class="ba-overlay"></div>
      <div class="ba-panel" role="dialog" aria-modal="true" aria-label="Before/after viewer">
        <div class="ba-toolbar">
          <div class="ba-tabs">
            <button class="ba-tab active" data-mode="slider" title="Drag the handle to wipe between before and after">Slider</button>
            <button class="ba-tab" data-mode="sbs" title="Show both images side by side">Side-by-side</button>
            <button class="ba-tab" data-mode="toggle" title="Click to flip between before and after">Toggle</button>
            <button class="ba-tab" data-mode="diff" title="Show what pixels changed (red = changed)">Difference</button>
          </div>
          <div class="ba-actions">
            <a class="btn btn-sm btn-primary" id="baDownload" download>Download result</a>
            <button class="btn btn-sm btn-danger" id="baClose" title="Close (Esc)">Close</button>
          </div>
        </div>
        <div class="ba-stage" id="baStage"></div>
        <div class="ba-footer">
          <span id="baLabel"></span>
          <span class="ba-hint" id="baHint"></span>
        </div>
      </div>
    `;
    document.body.appendChild(_modal);

    _modal.querySelectorAll('.ba-tab').forEach(t => {
      t.addEventListener('click', () => setMode(t.dataset.mode));
    });
    _modal.querySelector('#baClose').addEventListener('click', closeModal);
    _modal.querySelector('.ba-overlay').addEventListener('click', closeModal);
    document.addEventListener('keydown', onKey);
    return _modal;
  }

  function onKey(e) {
    if (!_modal || _modal.style.display === 'none') return;
    if (e.key === 'Escape') closeModal();
    if (e.key === '1') setMode('slider');
    if (e.key === '2') setMode('sbs');
    if (e.key === '3') setMode('toggle');
    if (e.key === '4') setMode('diff');
  }

  function closeModal() {
    if (!_modal) return;
    _modal.style.display = 'none';
    _state = null;
  }

  window.openBeforeAfter = function (beforeUrl, afterUrl, opts = {}) {
    ensureModal();
    _state = {
      before: beforeUrl, after: afterUrl,
      titleBefore: opts.titleBefore || 'Before',
      titleAfter: opts.titleAfter || 'After',
      fileName: opts.fileName || 'cleaned.png',
      mode: 'slider',
    };
    const dl = document.getElementById('baDownload');
    dl.href = afterUrl;
    dl.download = (opts.fileName || 'cleaned') + '.png';
    document.getElementById('baLabel').textContent = `${_state.titleBefore} ↔ ${_state.titleAfter}`;
    setMode('slider');
    _modal.style.display = 'flex';
  };

  function setMode(mode) {
    if (!_state) return;
    _state.mode = mode;
    _modal.querySelectorAll('.ba-tab').forEach(t => t.classList.toggle('active', t.dataset.mode === mode));
    const stage = document.getElementById('baStage');
    stage.innerHTML = '';
    const hint = document.getElementById('baHint');
    if (mode === 'slider') { renderSlider(stage); hint.textContent = 'Drag the handle, or click any column.'; }
    else if (mode === 'sbs') { renderSideBySide(stage); hint.textContent = ''; }
    else if (mode === 'toggle') { renderToggle(stage); hint.textContent = 'Click image to flip. Hold Space.'; }
    else if (mode === 'diff') { renderDiff(stage); hint.textContent = 'Red = pixels that changed.'; }
  }

  function renderSlider(stage) {
    stage.innerHTML = `
      <div class="ba-slider-wrap">
        <img class="ba-base" src="${_state.after}" alt="${_state.titleAfter}">
        <div class="ba-clip"><img src="${_state.before}" alt="${_state.titleBefore}"></div>
        <div class="ba-handle" id="baHandle"><span class="ba-handle-bar"></span><span class="ba-handle-grip">⇆</span></div>
        <span class="ba-corner ba-corner-l">${_state.titleBefore}</span>
        <span class="ba-corner ba-corner-r">${_state.titleAfter}</span>
      </div>
    `;
    const wrap = stage.querySelector('.ba-slider-wrap');
    const clip = stage.querySelector('.ba-clip');
    const handle = stage.querySelector('#baHandle');
    let dragging = false;
    let pct = 50;
    function set(p) {
      pct = Math.max(0, Math.min(100, p));
      clip.style.width = pct + '%';
      handle.style.left = pct + '%';
    }
    function fromEvent(e) {
      const r = wrap.getBoundingClientRect();
      const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
      set((x / r.width) * 100);
    }
    set(50);
    handle.addEventListener('pointerdown', e => { dragging = true; handle.setPointerCapture(e.pointerId); });
    handle.addEventListener('pointerup', e => { dragging = false; handle.releasePointerCapture(e.pointerId); });
    handle.addEventListener('pointermove', e => { if (dragging) fromEvent(e); });
    wrap.addEventListener('click', e => { if (e.target === wrap || e.target.classList.contains('ba-base') || e.target.parentElement === clip) fromEvent(e); });
  }

  function renderSideBySide(stage) {
    stage.innerHTML = `
      <div class="ba-sbs">
        <figure><figcaption>${_state.titleBefore}</figcaption><img src="${_state.before}"></figure>
        <figure><figcaption>${_state.titleAfter}</figcaption><img src="${_state.after}"></figure>
      </div>
    `;
  }

  function renderToggle(stage) {
    stage.innerHTML = `
      <div class="ba-toggle">
        <img id="baToggleImg" src="${_state.after}" alt="${_state.titleAfter}">
        <span class="ba-corner" id="baToggleLabel">${_state.titleAfter}</span>
      </div>
    `;
    const img = stage.querySelector('#baToggleImg');
    const label = stage.querySelector('#baToggleLabel');
    let showingAfter = true;
    const flip = () => {
      showingAfter = !showingAfter;
      img.src = showingAfter ? _state.after : _state.before;
      label.textContent = showingAfter ? _state.titleAfter : _state.titleBefore;
    };
    img.addEventListener('click', flip);
    // Hold Space for the alternate
    const keyHandler = (e) => {
      if (e.code === 'Space' && stage.isConnected) {
        e.preventDefault();
        if (e.type === 'keydown' && showingAfter) flip();
        if (e.type === 'keyup' && !showingAfter) flip();
      }
    };
    document.addEventListener('keydown', keyHandler);
    document.addEventListener('keyup', keyHandler);
    // Cleanup on stage replacement
    new MutationObserver(() => {
      if (!stage.contains(img)) {
        document.removeEventListener('keydown', keyHandler);
        document.removeEventListener('keyup', keyHandler);
      }
    }).observe(stage, { childList: true });
  }

  function renderDiff(stage) {
    stage.innerHTML = `
      <div class="ba-diff">
        <canvas id="baDiffCanvas"></canvas>
        <div class="ba-diff-controls">
          <label>Threshold <input id="baDiffThreshold" type="range" min="2" max="80" value="20"> <span id="baDiffThresholdValue">20</span></label>
          <label>Overlay <input id="baDiffAlpha" type="range" min="0" max="100" value="60"> <span id="baDiffAlphaValue">60%</span></label>
          <span id="baDiffStats" class="ba-diff-stats"></span>
        </div>
      </div>
    `;
    const canvas = stage.querySelector('#baDiffCanvas');
    const ctx = canvas.getContext('2d');
    const thresholdEl = stage.querySelector('#baDiffThreshold');
    const alphaEl = stage.querySelector('#baDiffAlpha');
    const stats = stage.querySelector('#baDiffStats');
    let imgB = new Image(), imgA = new Image();
    let dataB = null, dataA = null;
    let loaded = 0;
    function tryRender() {
      if (loaded < 2) return;
      const W = imgB.naturalWidth, H = imgB.naturalHeight;
      canvas.width = W; canvas.height = H;
      const c1 = document.createElement('canvas'); c1.width = W; c1.height = H;
      const c2 = document.createElement('canvas'); c2.width = W; c2.height = H;
      c1.getContext('2d').drawImage(imgB, 0, 0, W, H);
      c2.getContext('2d').drawImage(imgA, 0, 0, W, H);
      dataB = c1.getContext('2d').getImageData(0, 0, W, H).data;
      dataA = c2.getContext('2d').getImageData(0, 0, W, H).data;
      paint();
    }
    function paint() {
      const W = imgB.naturalWidth, H = imgB.naturalHeight;
      const t = +thresholdEl.value;
      const aPct = +alphaEl.value;
      thresholdEl.nextElementSibling.textContent = t;
      alphaEl.nextElementSibling.textContent = aPct + '%';
      const out = ctx.createImageData(W, H);
      let changed = 0;
      for (let i = 0, j = 0; i < dataB.length; i += 4, j += 4) {
        // Luminance diff
        const lb = 0.299*dataB[i] + 0.587*dataB[i+1] + 0.114*dataB[i+2];
        const la = 0.299*dataA[i] + 0.587*dataA[i+1] + 0.114*dataA[i+2];
        const d = Math.abs(lb - la);
        // Base = after image, attenuated; overlay red where changed
        const att = 1 - aPct / 200;  // never fully black
        out.data[i]   = dataA[i]   * att;
        out.data[i+1] = dataA[i+1] * att;
        out.data[i+2] = dataA[i+2] * att;
        if (d > t) {
          out.data[i]   = Math.min(255, out.data[i]   + 220 * (aPct / 100));
          out.data[i+1] = out.data[i+1] * 0.5;
          out.data[i+2] = out.data[i+2] * 0.5;
          changed++;
        }
        out.data[i+3] = 255;
      }
      ctx.putImageData(out, 0, 0);
      stats.textContent = `${(100 * changed / (W*H)).toFixed(2)}% pixels changed`;
    }
    imgB.onload = () => { loaded++; tryRender(); };
    imgA.onload = () => { loaded++; tryRender(); };
    imgB.crossOrigin = 'anonymous';
    imgA.crossOrigin = 'anonymous';
    imgB.src = _state.before;
    imgA.src = _state.after;
    thresholdEl.addEventListener('input', () => { if (dataB) paint(); });
    alphaEl.addEventListener('input', () => { if (dataB) paint(); });
  }
})();
