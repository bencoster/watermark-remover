function initGpuStatus() {
  const panel = document.getElementById('panel-status');
  panel.innerHTML = '<h2>System Status</h2><div id="gpuCards">Loading...</div>';
  loadGpuStatus();
}

async function loadGpuStatus() {
  try {
    const r = await fetch(API + '/api/status');
    if (!r.ok) return;
    const data = await r.json();
    renderGpuStatus(data);
  } catch (e) {
    document.getElementById('gpuCards').innerHTML = '<p>Failed to load status</p>';
  }
}

function renderGpuStatus(data) {
  const el = document.getElementById('gpuCards');
  if (!el) return;

  let html = `
    <div class="status-card">
      <h3>Inference</h3>
      <p>Device: <strong>${data.device}</strong></p>
      <p>Loaded model: <strong>${data.loaded_model || 'none'}</strong></p>
    </div>
  `;

  if (data.gpus && data.gpus.length > 0) {
    html += '<h3>GPUs</h3>';
    html += data.gpus.map(g => `
      <div class="status-card ${g.skipped ? 'gpu-skipped' : ''}">
        <h4>GPU ${g.index}: ${g.name} ${g.skipped ? '(skipped)' : ''}</h4>
        <div class="vram-bar">
          <div class="vram-used" style="width:${((g.used_mb / g.total_mb) * 100).toFixed(0)}%"></div>
        </div>
        <p>${g.used_mb} / ${g.total_mb} MB (${g.free_mb} MB free)</p>
      </div>
    `).join('');
  } else {
    html += '<p>No GPUs detected - running on CPU</p>';
  }

  el.innerHTML = html;
}
