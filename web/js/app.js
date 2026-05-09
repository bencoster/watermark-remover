const API = '';
const NAV_ITEMS = ['upload', 'mask', 'jobs', 'status'];

function showPanel(name) {
  if (!NAV_ITEMS.includes(name)) name = 'upload';
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const panel = document.getElementById('panel-' + name);
  if (panel) panel.classList.add('active');
  const idx = NAV_ITEMS.indexOf(name);
  if (idx >= 0) document.querySelectorAll('.nav-item')[idx]?.classList.add('active');
  if (name === 'jobs') initJobDashboard();
  if (name === 'status') initGpuStatus();
}

function showToast(msg) {
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2500);
}

document.addEventListener('DOMContentLoaded', () => {
  showPanel('upload');
  initUpload();
});
