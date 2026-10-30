// NeuroAI Web Application Client-Side Logic

let currentFile = null;
let currentSamplePath = null;
let currentScanId = null;

// Backend API URL configuration (supports Vercel frontend <-> Render backend)
function getBackendUrl() {
  const stored = localStorage.getItem('NEUROAI_BACKEND_URL');
  if (stored) return stored.trim().replace(/\/+$/, '');
  if (window.NEUROAI_BACKEND_URL) return window.NEUROAI_BACKEND_URL.trim().replace(/\/+$/, '');
  if (window.location.hostname.includes('vercel.app')) {
    return 'https://neuroai-brain-detection.onrender.com';
  }
  return '';
}

function resolveApiUrl(path) {
  if (!path) return '';
  if (path.startsWith('http://') || path.startsWith('https://') || path.startsWith('data:') || path.startsWith('blob:')) {
    return path;
  }
  const base = getBackendUrl();
  if (!base) return path;
  const cleanPath = path.startsWith('/') ? path : '/' + path;
  return `${base}${cleanPath}`;
}

document.addEventListener('DOMContentLoaded', () => {
  initDropzone();
  loadSampleList();
  loadPatientHistory();
  checkBackendConnection();
});

// Check & Ping Backend Connection Status
async function checkBackendConnection() {
  const badge = document.getElementById('btn-backend-settings');
  const text = document.getElementById('backend-status-text');
  if (!badge || !text) return;

  const url = getBackendUrl();
  text.innerText = url ? 'Connecting to Render...' : 'Checking Backend...';
  badge.className = 'backend-badge checking';

  try {
    const healthUrl = resolveApiUrl('/api/health');
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 9000);
    const res = await fetch(healthUrl, { signal: controller.signal });
    clearTimeout(timeoutId);

    if (res.ok) {
      badge.className = 'backend-badge online';
      text.innerText = 'Backend: Online';
    } else {
      badge.className = 'backend-badge error';
      text.innerText = 'Backend Error';
    }
  } catch (err) {
    if (url && (url.includes('onrender.com') || window.location.hostname.includes('vercel.app'))) {
      badge.className = 'backend-badge waking';
      text.innerText = 'Waking up Render (free tier)...';
      // Render free tier spins down on idle; automatically re-check every 8s
      setTimeout(checkBackendConnection, 8000);
    } else {
      badge.className = 'backend-badge error';
      text.innerText = 'Backend Offline';
    }
  }
}

// Modal Settings for Backend API URL
function openBackendModal() {
  const modal = document.getElementById('backend-modal');
  const input = document.getElementById('input-backend-url');
  if (input) input.value = getBackendUrl();
  if (modal) modal.style.display = 'flex';
}

function closeBackendModal() {
  const modal = document.getElementById('backend-modal');
  if (modal) modal.style.display = 'none';
}

async function testBackendConnection() {
  const input = document.getElementById('input-backend-url');
  const status = document.getElementById('backend-ping-status');
  if (!input || !status) return;

  const targetUrl = input.value.trim().replace(/\/+$/, '');
  status.style.display = 'block';
  status.style.background = 'rgba(2, 132, 199, 0.12)';
  status.style.color = '#0284c7';
  status.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Pinging ${targetUrl || 'current origin'}... (Render takes ~30s if waking up)`;

  try {
    const pingEndpoint = targetUrl ? `${targetUrl}/api/health` : '/api/health';
    const res = await fetch(pingEndpoint);
    if (res.ok) {
      status.style.background = 'rgba(34, 197, 94, 0.15)';
      status.style.color = '#15803d';
      status.innerHTML = '<i class="fa-solid fa-circle-check"></i> Connection successful! Render backend is live and ready.';
    } else {
      status.style.background = 'rgba(239, 68, 68, 0.15)';
      status.style.color = '#b91c1c';
      status.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> Server returned status code: ${res.status}`;
    }
  } catch (err) {
    status.style.background = 'rgba(245, 158, 11, 0.15)';
    status.style.color = '#b45309';
    status.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Backend not reachable yet. If it was sleeping, please wait ~30s for Render to wake up and test again.';
  }
}

function saveBackendUrl() {
  const input = document.getElementById('input-backend-url');
  if (input) {
    const val = input.value.trim();
    if (val) {
      localStorage.setItem('NEUROAI_BACKEND_URL', val);
    } else {
      localStorage.removeItem('NEUROAI_BACKEND_URL');
    }
    closeBackendModal();
    checkBackendConnection();
    loadSampleList();
    loadPatientHistory();
  }
}

// Tab Navigation
function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
  
  const target = document.getElementById(tabId);
  if (target) target.classList.add('active');

  const clickedBtn = Array.from(document.querySelectorAll('.nav-btn')).find(b => 
    b.getAttribute('onclick') && b.getAttribute('onclick').includes(tabId)
  );
  if (clickedBtn) clickedBtn.classList.add('active');

  if (tabId === 'tab-database') {
    loadPatientHistory();
  }
}

// Drag & Drop Setup
function initDropzone() {
  const dropzone = document.getElementById('mri-dropzone');
  if (!dropzone) return;

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  });
}

function handleFileUpload(file) {
  if (!file) return;
  currentFile = file;
  currentSamplePath = null;
  
  // Clear sample active states
  document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
  
  // Preview
  const reader = new FileReader();
  reader.onload = (e) => {
    const rawView = document.getElementById('view-raw');
    const placeholder = document.getElementById('viewer-placeholder');
    
    rawView.src = e.target.result;
    rawView.style.display = 'block';
    if (placeholder) placeholder.style.display = 'none';
  };
  reader.readAsDataURL(file);
}

// Load Benchmark Samples
let sampleMap = {};
async function loadSampleList() {
  try {
    const res = await fetch(resolveApiUrl('/api/sample-images'));
    const data = await res.json();
    if (data.status === 'success') {
      sampleMap = data.samples;
    }
  } catch (err) {
    console.error('Error fetching sample images:', err);
  }
}

function loadSampleMRI(className) {
  const samples = sampleMap[className];
  if (!samples || samples.length === 0) return;
  
  // Pick random sample from the class list
  const randomSample = samples[Math.floor(Math.random() * samples.length)];
  currentSamplePath = randomSample;
  currentFile = null;

  document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
  try {
    const ev = (typeof event !== 'undefined' ? event : window.event);
    if (ev && ev.currentTarget) {
      ev.currentTarget.classList.add('active');
    }
  } catch (e) {}

  const rawView = document.getElementById('view-raw');
  const placeholder = document.getElementById('viewer-placeholder');
  
  rawView.src = resolveApiUrl(randomSample);
  rawView.style.display = 'block';
  if (placeholder) placeholder.style.display = 'none';

  // Automatically trigger diagnosis for fast demo
  runDiagnosis();
}

// Run Deep Learning & XAI Diagnosis
async function runDiagnosis() {
  if (!currentFile && !currentSamplePath) {
    alert('Please upload an MRI scan or select a sample image first.');
    return;
  }

  const btn = document.getElementById('btn-diagnose');
  const originalBtnText = btn.innerHTML;
  btn.innerHTML = '<span class="spinner"></span> Analyzing Scan...';
  btn.disabled = true;

  const formData = new FormData();
  if (currentFile) {
    formData.append('file', currentFile);
  } else if (currentSamplePath) {
    formData.append('sample_path', currentSamplePath);
  }

  formData.append('patient_id', document.getElementById('patient-id').value);
  formData.append('patient_name', document.getElementById('patient-name').value);
  formData.append('apply_crop', document.getElementById('prep-crop').checked);
  formData.append('apply_clahe', document.getElementById('prep-clahe').checked);
  formData.append('apply_denoise', document.getElementById('prep-denoise').checked);
  formData.append('model_name', document.getElementById('model-select').value);
  formData.append('xai_method', document.getElementById('xai-select').value);

  try {
    const response = await fetch(resolveApiUrl('/api/predict'), {
      method: 'POST',
      body: formData
    });
    const data = await response.json();

    if (data.status === 'success') {
      currentScanId = data.scan_id;
      displayDiagnosisResults(data);
    } else {
      alert('Diagnosis Error: ' + data.message);
    }
  } catch (err) {
    console.error('Diagnosis request failed:', err);
    alert('Error running AI diagnosis. Please try again.');
  } finally {
    btn.innerHTML = originalBtnText;
    btn.disabled = false;
  }
}

// Display Diagnostic Results on UI
function displayDiagnosisResults(data) {
  // 1. Update MRI Viewer Layers
  const rawView = document.getElementById('view-raw');
  const prepView = document.getElementById('view-prep');
  const xaiView = document.getElementById('view-xai');
  const placeholder = document.getElementById('viewer-placeholder');
  const opacityControl = document.getElementById('opacity-control');
  const exportBtn = document.getElementById('btn-export-pdf');

  if (placeholder) placeholder.style.display = 'none';
  if (opacityControl) opacityControl.style.display = 'flex';
  if (exportBtn) exportBtn.style.display = 'inline-flex';

  rawView.src = resolveApiUrl(data.images.preprocessed);
  rawView.style.display = 'block';
  xaiView.src = resolveApiUrl(data.images.gradcam);
  xaiView.style.display = 'block';
  xaiView.style.opacity = '0.65';
  document.getElementById('xai-opacity').value = 65;
  document.getElementById('opacity-val').innerText = '65%';

  // 2. Update Verdict Banner
  const resultsCard = document.getElementById('diagnostic-results');
  resultsCard.style.display = 'block';

  const verdictCard = document.getElementById('verdict-card');
  const verdictTitle = document.getElementById('verdict-title');
  const verdictConf = document.getElementById('verdict-confidence');

  verdictCard.className = 'verdict-banner ' + (data.is_tumor ? (data.predicted_class === 'glioma' ? 'malignant' : 'benign') : 'normal');
  verdictTitle.innerText = data.class_details.title + (data.is_tumor ? ' Detected' : '');
  verdictConf.innerText = `${data.confidence}% Confidence`;

  // 3. Update Probability Bars
  const probs = data.probabilities;
  for (const cname of ['glioma', 'meningioma', 'pituitary', 'notumor']) {
    const val = probs[cname] || 0;
    const bar = document.getElementById(`prob-bar-${cname}`);
    const valText = document.getElementById(`prob-val-${cname}`);
    if (bar) bar.style.width = `${val}%`;
    if (valText) valText.innerText = `${val.toFixed(1)}%`;
  }

  // 4. Update Quantitative Lesion Metrics
  const metrics = data.tumor_metrics;
  const areaElem = document.getElementById('metric-area');
  const quadElem = document.getElementById('metric-quadrant');

  if (areaElem) areaElem.innerText = `${metrics.tumor_area_percentage}% of brain`;
  if (quadElem) quadElem.innerText = metrics.anatomical_quadrant;

  // 5. Update Clinical Guidance, Curability & Recommended Medicines
  const guidance = data.clinical_guidance;
  if (guidance) {
    const curable = guidance.curability || {};
    const curablePct = curable.curable_percentage || 0;
    const nonCurablePct = curable.non_curable_percentage || 0;
    
    // Curability Gauge
    const meterCurable = document.getElementById('meter-curable');
    const meterNonCurable = document.getElementById('meter-non-curable');
    const curableBadge = document.getElementById('curability-badge');
    const textCurable = document.getElementById('text-curable-pct');
    const textNonCurable = document.getElementById('text-non-curable-pct');
    const curableSummary = document.getElementById('curability-summary');

    if (meterCurable) meterCurable.style.width = `${curablePct}%`;
    if (meterNonCurable) meterNonCurable.style.width = `${nonCurablePct}%`;
    if (textCurable) textCurable.innerText = `${curablePct.toFixed(1)}%`;
    if (textNonCurable) textNonCurable.innerText = `${nonCurablePct.toFixed(1)}%`;

    if (curableBadge) {
      curableBadge.innerText = `${curablePct.toFixed(1)}% Curable`;
      const isHighCurable = curablePct >= 75;
      curableBadge.style.color = isHighCurable ? '#059669' : '#dc2626';
      curableBadge.style.borderColor = isHighCurable ? '#059669' : '#dc2626';
      curableBadge.style.background = isHighCurable ? '#ecfdf5' : '#fef2f2';
    }

    if (curableSummary) {
      curableSummary.innerHTML = `<strong>${curable.prognosis_category || ''}</strong><br/>${curable.curability_summary || ''}<br/><span style="color: var(--text-muted); font-size: 0.72rem;"><i class="fa-solid fa-chart-line"></i> ${curable.survival_benchmark || ''}</span>`;
    }

    // Recommended Treatment Solution
    const solution = guidance.primary_solution || {};
    const solTitle = document.getElementById('solution-title');
    const solDesc = document.getElementById('solution-desc');
    const solUrgency = document.getElementById('solution-urgency');

    if (solTitle) solTitle.innerText = solution.title || 'Standard Clinical Management';
    if (solUrgency) solUrgency.innerText = solution.urgency || 'Standard Protocol';
    if (solDesc) {
      solDesc.innerHTML = `<strong>Surgical / Clinical Pathway:</strong> ${solution.surgical_protocol || ''}<br/><strong>Radiation Protocol:</strong> ${solution.radiotherapy_protocol || ''}`;
    }

    // Recommended Prescribed Medicines
    const medContainer = document.getElementById('medicines-container');
    if (medContainer) {
      medContainer.innerHTML = '';
      const medicines = guidance.recommended_medicines || [];
      medicines.forEach(med => {
        const item = document.createElement('div');
        item.className = 'med-card-item';
        item.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
            <div style="font-weight: 800; color: #0284c7; font-size: 0.84rem;"><i class="fa-solid fa-capsules" style="color: #d97706; margin-right: 0.35rem;"></i>${med.name}</div>
            <span style="font-size: 0.68rem; font-weight: 700; background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; padding: 0.15rem 0.45rem; border-radius: 4px;">${med.category}</span>
          </div>
          <div style="font-size: 0.76rem; color: var(--text-primary); margin-bottom: 0.15rem;"><strong>Dosage:</strong> ${med.dosage}</div>
          <div style="font-size: 0.74rem; color: var(--text-secondary); line-height: 1.35;">${med.purpose}</div>
        `;
        medContainer.appendChild(item);
      });
    }
  }
}

// Opacity Slider for Grad-CAM
function updateHeatmapOpacity(val) {
  const xaiView = document.getElementById('view-xai');
  const opacityVal = document.getElementById('opacity-val');
  if (xaiView) xaiView.style.opacity = val / 100.0;
  if (opacityVal) opacityVal.innerText = `${val}%`;
}

// PDF Clinical Report Export
async function exportPDFReport() {
  if (!currentScanId) {
    alert('No active scan record to export.');
    return;
  }

  const exportBtn = document.getElementById('btn-export-pdf');
  const originalText = exportBtn.innerHTML;
  exportBtn.innerHTML = '<span class="spinner"></span> Generating PDF...';
  exportBtn.disabled = true;

  try {
    const res = await fetch(resolveApiUrl('/api/export-report'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scan_id: currentScanId })
    });

    if (res.ok) {
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Diagnostic_Report_${currentScanId}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } else {
      alert('Failed to generate PDF report.');
    }
  } catch (err) {
    console.error('PDF export error:', err);
    alert('Error exporting PDF report.');
  } finally {
    exportBtn.innerHTML = originalText;
    exportBtn.disabled = false;
  }
}

// Patient Database
async function loadPatientHistory() {
  try {
    const res = await fetch(resolveApiUrl('/api/history'));
    const data = await res.json();
    if (data.status !== 'success') return;

    const tbody = document.getElementById('history-table-body');
    if (!tbody) return;

    tbody.innerHTML = '';
    if (data.scans.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: #64748b;">No scan records found. Run a diagnosis above.</td></tr>`;
      return;
    }

    data.scans.forEach(s => {
      const tr = document.createElement('tr');
      const badgeClass = s.is_tumor ? (s.predicted_class === 'glioma' ? 'badge-red' : 'badge-amber') : 'badge-green';
      
      tr.innerHTML = `
        <td><b>#${s.id}</b></td>
        <td><b>${s.patient_id}</b><br/><span style="color: #64748b; font-size: 0.75rem;">${s.patient_name}</span></td>
        <td>${s.created_at}</td>
        <td><span class="badge ${badgeClass}">${s.predicted_class}</span></td>
        <td><b>${s.confidence}%</b></td>
        <td>${s.severity_index}</td>
        <td><span style="font-size: 0.78rem; color: #94a3b8;">${s.model_used}</span></td>
        <td>
          <button class="btn-secondary" style="padding: 0.35rem 0.65rem; font-size: 0.75rem;" onclick="downloadHistoryPDF(${s.id})">
            <i class="fa-solid fa-file-pdf" style="color: #f87171;"></i> PDF
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error('Failed to load history:', err);
  }
}

function downloadHistoryPDF(scanId) {
  currentScanId = scanId;
  exportPDFReport();
}
