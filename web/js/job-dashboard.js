let _jobPollTimer = null;

function initJobDashboard() {
  const panel = document.getElementById('panel-jobs');
  panel.innerHTML = '<h2>Jobs</h2><div id="jobList">Loading...</div>';
  loadJobs();
  if (_jobPollTimer) clearInterval(_jobPollTimer);
  _jobPollTimer = setInterval(loadJobs, 3000);
}

async function loadJobs() {
  try {
    const r = await fetch(API + '/api/jobs?limit=20');
    if (!r.ok) return;
    const data = await r.json();
    renderJobs(data.jobs || []);
  } catch (e) { /* ignore */ }
}

function renderJobs(jobs) {
  const el = document.getElementById('jobList');
  if (!el) return;
  if (jobs.length === 0) {
    el.innerHTML = '<p class="empty">No jobs yet</p>';
    return;
  }
  el.innerHTML = jobs.map(j => `
    <div class="job-card job-${j.status}">
      <div class="job-header">
        <span class="job-id">${j.id}</span>
        <span class="job-status">${j.status}</span>
      </div>
      <div class="job-body">
        <span class="job-kind">${j.kind}</span>
        ${j.status === 'running' ? `<div class="progress-bar"><div class="progress-fill" style="width:${(j.progress * 100).toFixed(0)}%"></div></div><span class="job-stage">${j.stage || ''}</span>` : ''}
        ${j.status === 'succeeded' ? `<a href="${API}/api/jobs/${j.id}/result" class="btn btn-sm btn-primary" download>Download</a>` : ''}
        ${j.status === 'failed' ? `<span class="job-error">${j.error || 'Unknown error'}</span>` : ''}
      </div>
    </div>
  `).join('');
}
