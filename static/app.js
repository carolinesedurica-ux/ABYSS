'use strict';

// ── Tab switching ─────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => {
      p.classList.remove('active');
      p.classList.add('hidden');
    });
    tab.classList.add('active');
    const panel = document.getElementById(`tab-${tab.dataset.tab}`);
    panel.classList.add('active');
    panel.classList.remove('hidden');
    if (tab.dataset.tab === 'sources') loadSources();
  });
});

// ── Query ─────────────────────────────────────────────────────
const qInput   = document.getElementById('query-input');
const qBtn     = document.getElementById('query-btn');
const qLabel   = document.getElementById('query-btn-label');
const qLoading = document.getElementById('query-loading');
const qResult  = document.getElementById('query-result');
const qError   = document.getElementById('query-error');
const steps    = document.getElementById('pipeline-steps');

qBtn.addEventListener('click', runQuery);
qInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runQuery();
});

function setStep(id) {
  document.querySelectorAll('.step').forEach(s => {
    if (s.id === id) s.classList.add('active');
    else { s.classList.remove('active'); if (s.getBoundingClientRect()) s.classList.add('done'); }
  });
}

async function runQuery() {
  const q = qInput.value.trim();
  if (!q || qBtn.classList.contains('busy')) return;

  qBtn.classList.add('busy');
  qLabel.textContent = 'RUNNING...';
  qLoading.classList.remove('hidden');
  qResult.classList.add('hidden');
  qError.classList.add('hidden');
  steps.classList.remove('hidden');

  // simulate pipeline step animation
  const stepOrder = ['step-router','step-retrieve','step-evaluate','step-synthesize'];
  let si = 0;
  const stepTimer = setInterval(() => {
    if (si < stepOrder.length) { setStep(stepOrder[si++]); }
    else clearInterval(stepTimer);
  }, 900);

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status} — ${await res.text()}`);
    const data = await res.json();

    clearInterval(stepTimer);
    document.querySelectorAll('.step').forEach(s => {
      s.classList.remove('active');
      s.classList.add('done');
    });

    document.getElementById('answer-text').textContent = data.answer;
    document.getElementById('result-meta').textContent =
      `${data.citations.length} source${data.citations.length !== 1 ? 's' : ''}`;

    const citBox = document.getElementById('citations');
    const citHdr = document.getElementById('cit-header');
    citBox.innerHTML = '';

    if (data.citations.length === 0) {
      citHdr.classList.add('hidden');
    } else {
      citHdr.classList.remove('hidden');
      data.citations.forEach(c => {
        const el = document.createElement('div');
        el.className = 'cit-item';
        el.innerHTML = `
          <span class="cit-idx">[${c.index}]</span>
          <span class="cit-lbl">${escHtml(c.label)}</span>
          <span class="cit-score">score: ${c.score}</span>
        `;
        citBox.appendChild(el);
      });
    }

    qResult.classList.remove('hidden');
  } catch (err) {
    clearInterval(stepTimer);
    qError.textContent = `ERROR: ${err.message}`;
    qError.classList.remove('hidden');
  } finally {
    qBtn.classList.remove('busy');
    qLabel.textContent = 'RUN QUERY';
    qLoading.classList.add('hidden');
  }
}

// ── Ingest ────────────────────────────────────────────────────
const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const fileQueue  = document.getElementById('file-queue');
const iLoading   = document.getElementById('ingest-loading');
const iResult    = document.getElementById('ingest-result');
const iError     = document.getElementById('ingest-error');

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFiles([...e.dataTransfer.files]);
});
fileInput.addEventListener('change', () => handleFiles([...fileInput.files]));

function fileExt(name) { return name.split('.').pop().toLowerCase(); }

function badgeFor(ext) {
  if (['mp4','mkv','mov','avi','webm'].includes(ext))
    return `<span class="badge badge-video">${ext.toUpperCase()}</span>`;
  if (['mp3','wav','m4a','ogg','flac'].includes(ext))
    return `<span class="badge badge-audio">${ext.toUpperCase()}</span>`;
  return `<span class="badge badge-pdf">PDF</span>`;
}

async function handleFiles(files) {
  if (!files.length) return;

  fileQueue.innerHTML = '';
  files.forEach(f => {
    const ext = fileExt(f.name);
    const row = document.createElement('div');
    row.className = 'file-item';
    row.innerHTML = `${badgeFor(ext)}<span class="file-nm">${escHtml(f.name)}</span><span class="file-sz">${fmtSize(f.size)}</span>`;
    fileQueue.appendChild(row);
  });
  fileQueue.classList.remove('hidden');

  iResult.classList.add('hidden');
  iError.classList.add('hidden');
  iLoading.classList.remove('hidden');

  const form = new FormData();
  files.forEach(f => form.append('files', f));

  try {
    // 1. Store in vault
    const upRes = await fetch('/vault/upload', { method: 'POST', body: form });
    if (!upRes.ok) throw new Error(`HTTP ${upRes.status} — ${await upRes.text()}`);
    const upData = await upRes.json();

    // 2. Kick off mining for every newly stored file
    await fetch('/vault/mine-all', { method: 'POST' });

    const n = upData.count;
    document.getElementById('ingest-output').textContent =
      `✓  ${n} file${n !== 1 ? 's' : ''} added to vault\n` +
      `✓  mining queued — switching to VAULT tab…`;
    iResult.classList.remove('hidden');

    // 3. Auto-switch to vault so the user sees live mining status
    setTimeout(() => document.querySelector('[data-tab="vault"]').click(), 1200);
  } catch (err) {
    iError.textContent = `ERROR: ${err.message}`;
    iError.classList.remove('hidden');
  } finally {
    iLoading.classList.add('hidden');
    fileInput.value = '';
  }
}

// ── Sources ───────────────────────────────────────────────────
async function loadSources() {
  const list = document.getElementById('sources-list');
  list.innerHTML = '<div class="empty">scanning...</div>';

  try {
    const res  = await fetch('/sources');
    const data = await res.json();
    list.innerHTML = '';

    if (!data.answer || data.answer.includes('No sources')) {
      list.innerHTML = '<div class="empty">// no sources indexed yet</div>';
      return;
    }

    const lines = data.answer.split('\n').filter(l => l.includes('•'));
    if (!lines.length) {
      list.innerHTML = '<div class="empty">// no sources indexed yet</div>';
      return;
    }

    lines.forEach(line => {
      const name = line.replace(/.*•\s*/, '').trim();
      const ext  = fileExt(name);
      const row  = document.createElement('div');
      row.className = 'src-item';
      row.innerHTML = `${badgeFor(ext)}<span class="src-nm">${escHtml(name)}</span>`;
      list.appendChild(row);
    });
  } catch (err) {
    list.innerHTML = `<div class="errbox">ERROR: ${err.message}</div>`;
  }
}

// ── Helpers ───────────────────────────────────────────────────
function fmtSize(b) {
  if (b < 1024)        return `${b} B`;
  if (b < 1048576)     return `${(b/1024).toFixed(1)} KB`;
  return `${(b/1048576).toFixed(1)} MB`;
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// auto-load sources if starting on that tab
window.addEventListener('load', () => {
  if (document.querySelector('.tab.active')?.dataset.tab === 'sources') loadSources();
});

// ── Vault ─────────────────────────────────────────────────────
const vaultDrop   = document.getElementById('vault-drop');
const vaultInput  = document.getElementById('vault-file-input');
const vaultUpErr  = document.getElementById('vault-upload-error');
const vaultUpLoad = document.getElementById('vault-upload-loading');

let vaultPollTimer = null;

// switch-to-vault: start polling
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    if (tab.dataset.tab === 'vault') {
      loadVault();
      startVaultPoll();
    } else {
      stopVaultPoll();
    }
  });
});

vaultDrop.addEventListener('dragover',  e => { e.preventDefault(); vaultDrop.classList.add('drag-over'); });
vaultDrop.addEventListener('dragleave', () => vaultDrop.classList.remove('drag-over'));
vaultDrop.addEventListener('drop', e => {
  e.preventDefault(); vaultDrop.classList.remove('drag-over');
  uploadToVault([...e.dataTransfer.files]);
});
vaultInput.addEventListener('change', () => uploadToVault([...vaultInput.files]));

async function uploadToVault(files) {
  if (!files.length) return;
  vaultUpErr.classList.add('hidden');
  vaultUpLoad.classList.remove('hidden');

  const form = new FormData();
  files.forEach(f => form.append('files', f));

  try {
    const res = await fetch('/vault/upload', { method: 'POST', body: form });
    if (!res.ok) throw new Error(`HTTP ${res.status} — ${await res.text()}`);
    await loadVault();
  } catch (err) {
    vaultUpErr.textContent = `UPLOAD ERROR: ${err.message}`;
    vaultUpErr.classList.remove('hidden');
  } finally {
    vaultUpLoad.classList.add('hidden');
    vaultInput.value = '';
  }
}

async function loadVault() {
  try {
    const res  = await fetch('/vault/files');
    if (!res.ok) return;
    const data = await res.json();

    // stats
    const stats = data.stats;
    const statEl = document.getElementById('vault-stats');
    statEl.classList.remove('hidden');
    document.querySelector('#stat-total  .stat-num').textContent = stats.total;
    document.querySelector('#stat-stored .stat-num').textContent = stats.stored;
    document.querySelector('#stat-mining .stat-num').textContent = stats.mining;
    document.querySelector('#stat-mined  .stat-num').textContent = stats.mined;
    document.querySelector('#stat-error  .stat-num').textContent = stats.error;

    const list  = document.getElementById('vault-list');
    const empty = document.getElementById('vault-empty');

    if (!data.files.length) {
      list.classList.add('hidden');
      empty.classList.remove('hidden');
      return;
    }

    empty.classList.add('hidden');
    list.classList.remove('hidden');
    list.innerHTML = '';

    data.files.forEach(f => {
      const row = document.createElement('div');
      row.className = 'vault-item';
      row.id = `vault-row-${f.id}`;

      const isMining  = f.status === 'mining';
      const isMined   = f.status === 'mined';
      const dateStr   = f.uploaded_at ? f.uploaded_at.slice(0,10) : '';
      const chunks    = isMined ? `${f.chunk_count} chunks` : '';

      row.innerHTML = `
        ${badgeFor(fileExt(f.original_name))}
        <span class="vault-nm" title="${escHtml(f.original_name)}">${escHtml(f.original_name)}</span>
        <span class="vault-sz">${fmtSize(f.file_size)}</span>
        <span class="vault-date">${escHtml(dateStr)}</span>
        <span class="status-badge status-${f.status}">${f.status.toUpperCase()}${chunks ? ' · ' + chunks : ''}</span>
        <span class="vault-btns">
          <button class="icon-btn mine-btn" onclick="mineOne('${f.id}')" ${isMining ? 'disabled' : ''}>⚡ MINE</button>
          <button class="icon-btn del-btn"  onclick="deleteVault('${f.id}')">✕</button>
        </span>
      `;
      list.appendChild(row);
    });
  } catch (_) {}
}

async function mineOne(fileId) {
  try {
    const res = await fetch(`/vault/mine/${fileId}`, { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await loadVault();
  } catch (err) {
    vaultUpErr.textContent = `MINE ERROR: ${err.message}`;
    vaultUpErr.classList.remove('hidden');
  }
}

async function mineAll() {
  const btn = document.getElementById('mine-all-btn');
  btn.disabled = true;
  try {
    const res = await fetch('/vault/mine-all', { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await loadVault();
  } catch (err) {
    vaultUpErr.textContent = `MINE-ALL ERROR: ${err.message}`;
    vaultUpErr.classList.remove('hidden');
  } finally {
    btn.disabled = false;
  }
}

async function deleteVault(fileId) {
  try {
    const res = await fetch(`/vault/files/${fileId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const row = document.getElementById(`vault-row-${fileId}`);
    if (row) row.remove();
    await loadVault();
  } catch (err) {
    vaultUpErr.textContent = `DELETE ERROR: ${err.message}`;
    vaultUpErr.classList.remove('hidden');
  }
}

function startVaultPoll() {
  stopVaultPoll();
  vaultPollTimer = setInterval(loadVault, 4000);
}

function stopVaultPoll() {
  if (vaultPollTimer) { clearInterval(vaultPollTimer); vaultPollTimer = null; }
}
