function initLibrary() {
  const panel = document.getElementById('panel-library');
  panel.innerHTML = `
    <div class="library-header">
      <h2>Mask Library</h2>
      <p class="library-hint">Auto-saved masks from /api/auto runs. Rename, delete, or apply to a new image.</p>
    </div>
    <div id="libraryGrid" class="library-grid">Loading...</div>
  `;
  loadLibrary();
}

async function loadLibrary() {
  try {
    const r = await fetch(API + '/api/masks');
    if (!r.ok) {
      document.getElementById('libraryGrid').innerHTML = '<p class="empty">Library unavailable</p>';
      return;
    }
    const data = await r.json();
    renderLibrary(data.masks || []);
  } catch (e) {
    document.getElementById('libraryGrid').innerHTML = '<p class="empty">Library load failed</p>';
  }
}

function renderLibrary(masks) {
  const el = document.getElementById('libraryGrid');
  if (!el) return;
  if (masks.length === 0) {
    el.innerHTML = '<p class="empty">No masks saved yet — run a one-click watermark removal to populate the library.</p>';
    return;
  }
  el.innerHTML = masks.map(m => {
    const dt = m.created_at ? new Date(m.created_at).toLocaleString() : '';
    const conf = m.p_full != null ? (m.p_full * 100).toFixed(0) + '%' : '';
    const cov = m.body_coverage != null ? (m.body_coverage * 100).toFixed(1) + '%' : '';
    const stripBadge = m.has_strip ? '<span class="badge badge-strip">strip</span>' : '';
    return `
      <div class="library-card" data-id="${m.id}">
        <div class="card-thumb-wrap">
          <img class="card-thumb" src="${API}/api/masks/${m.id}/thumb" alt="" onerror="this.style.display='none'">
          <img class="card-mask" src="${API}/api/masks/${m.id}/file" alt="">
        </div>
        <div class="card-body">
          <input class="card-name" type="text" value="${escapeAttr(m.name)}" data-id="${m.id}">
          <div class="card-meta">
            <span>${dt}</span>
            <span>conf ${conf}</span>
            <span>cov ${cov}</span>
            ${stripBadge}
          </div>
          <div class="card-source">${escapeText(m.source_filename || '')}</div>
          <div class="card-actions">
            <button class="btn btn-sm btn-secondary" onclick="useMaskFromLibrary('${m.id}')">Use</button>
            <a class="btn btn-sm btn-secondary" href="${API}/api/masks/${m.id}/file" download="${escapeAttr(m.name)}.png">Download</a>
            <button class="btn btn-sm btn-danger" onclick="deleteMaskFromLibrary('${m.id}')">Delete</button>
          </div>
        </div>
      </div>
    `;
  }).join('');

  // Wire up rename-on-blur for each name input
  el.querySelectorAll('.card-name').forEach(inp => {
    inp.addEventListener('blur', async () => {
      const id = inp.dataset.id;
      const name = inp.value.trim();
      if (!name) return;
      try {
        await fetch(`${API}/api/masks/${id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name }),
        });
        showToast('Renamed');
      } catch (e) {
        showToast('Rename failed');
      }
    });
  });
}

async function deleteMaskFromLibrary(id) {
  if (!confirm('Delete this mask?')) return;
  try {
    const r = await fetch(`${API}/api/masks/${id}`, { method: 'DELETE' });
    if (r.ok) {
      showToast('Deleted');
      loadLibrary();
    } else {
      showToast('Delete failed');
    }
  } catch (e) {
    showToast('Delete error: ' + e.message);
  }
}

async function useMaskFromLibrary(id) {
  if (!_uploadedFile) {
    showToast('Upload an image first');
    showPanel('upload');
    return;
  }
  try {
    const r = await fetch(`${API}/api/masks/${id}/file`);
    if (!r.ok) { showToast('Failed to fetch mask'); return; }
    const blob = await r.blob();
    showPanel('mask');
    initMaskEditor(URL.createObjectURL(_uploadedFile), URL.createObjectURL(blob));
    showToast('Mask loaded — review and apply');
  } catch (e) {
    showToast('Load error: ' + e.message);
  }
}

function escapeText(s) {
  const div = document.createElement('div');
  div.textContent = String(s);
  return div.innerHTML;
}
function escapeAttr(s) { return escapeText(s).replace(/"/g, '&quot;'); }
