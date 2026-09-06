#!/usr/bin/env python3
"""HTML Frontend — single page app, embedded as a raw string."""

# HTML Frontend (single page app, embedded)
# ═══════════════════════════════════════════════

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ollama Benchmark</title>
<style>
@keyframes blink-cursor { 0%,100% { opacity: 1; } 50% { opacity: 0; } }
@keyframes pulse-glow { 0%,100% { box-shadow: 0 0 4px rgba(108,140,255,.3), inset 0 0 4px rgba(108,140,255,.05); } 50% { box-shadow: 0 0 12px rgba(108,140,255,.6), inset 0 0 8px rgba(108,140,255,.1); } }
@keyframes scan-move { 0% { top: -2px; } 100% { top: 100%; } }
@keyframes live-pulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }
@keyframes bar-pulse { 0% { filter: brightness(1.3); } 100% { filter: brightness(1); } }
@keyframes answer-flash { 0% { background: rgba(81,207,102,.2); } 100% { background: transparent; } }
@keyframes answer-flash-wrong { 0% { background: rgba(255,107,107,.2); } 100% { background: transparent; } }
@keyframes panel-appear { 0% { opacity: 0; transform: translateY(10px); } 100% { opacity: 1; transform: translateY(0); } }
:root {
  --bg: #0f1117;
  --card: #1a1d27;
  --border: #2a2d3a;
  --text: #e1e4ed;
  --muted: #7a7f8e;
  --accent: #6c8cff;
  --accent2: #4ecdc4;
  --danger: #ff6b6b;
  --success: #51cf66;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; min-height:100vh; }
.app { max-width:1200px; margin:0 auto; padding:20px; }
h1 { font-size:1.8em; font-weight:700; margin-bottom:4px; }
h1 span { color:var(--accent); }
.subtitle { color:var(--muted); margin-bottom:24px; font-size:.95em; }
.card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; margin-bottom:16px; }
.card h2 { font-size:1.1em; margin-bottom:12px; color:var(--accent2); }
.btn { display:inline-block; padding:8px 20px; border-radius:8px; border:none; cursor:pointer; font-size:.95em; font-weight:600; transition:all .15s; }
.btn-primary { background:var(--accent); color:#fff; }
.btn-primary:hover { background:#5a7de6; }
.btn-primary:disabled { opacity:.5; cursor:not-allowed; }
.btn-danger { background:var(--danger); color:#fff; }
.btn-danger:hover { background:#e05555; }
.btn-warning { background:#e6a237; color:#fff; }
.btn-warning:hover { background:#cf9235; }
.btn-secondary { background:var(--border); color:var(--text); }
.btn-secondary:hover { background:#3a3d4a; }
.btn-success { background:var(--success); color:#000; }

.row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
.col { flex:1; min-width:200px; }

/* Model & test selection */
.select-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(220px,1fr)); gap:8px; }
.select-item { display:flex; align-items:center; gap:8px; padding:8px 12px; border-radius:8px; background:var(--bg); border:1px solid var(--border); cursor:pointer; transition:all .15s; font-size:.9em; }
.select-item:hover { border-color:var(--accent); }
.select-item.selected { border-color:var(--accent); background:rgba(108,140,255,.1); }
.select-item input[type=checkbox] { accent-color:var(--accent); width:16px; height:16px; }
.model-size { color:var(--muted); font-size:.8em; }

/* Progress */
.progress-bar { height:6px; background:var(--border); border-radius:3px; overflow:hidden; margin:8px 0; }
.progress-fill { height:100%; background:linear-gradient(90deg,var(--accent),var(--accent2)); transition:width .3s; border-radius:3px; }
.status-text { color:var(--muted); font-size:.9em; }

/* Results table */
.results-table { width:100%; border-collapse:collapse; font-size:.85em; }
.results-table th { text-align:left; padding:8px 10px; border-bottom:2px solid var(--border); color:var(--accent2); font-weight:600; white-space:nowrap; }
.results-table td { padding:6px 10px; border-bottom:1px solid var(--border); }
.results-table tr:hover td { background:rgba(108,140,255,.05); }

/* Rating */
.rating-slider { width:200px; accent-color:var(--accent); }
.rating-val { font-weight:700; color:var(--accent); min-width:30px; }
.rating-row { display:flex; align-items:center; gap:12px; padding:6px 0; }
.rating-label { min-width:180px; font-size:.9em; }

/* Response viewer */
.response-box { background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:16px; margin:8px 0; max-height:400px; overflow-y:auto; white-space:pre-wrap; word-break:break-word; font-family:'SF Mono',Monaco,Consolas,monospace; font-size:.85em; line-height:1.5; color:#c5c8d4; }
.thinking-toggle { color:var(--muted); font-size:.85em; cursor:pointer; }
.thinking-box { background:rgba(78,205,196,.05); border-left:3px solid var(--accent2); padding:12px; margin:8px 0; border-radius:0 8px 8px 0; max-height:250px; overflow-y:auto; white-space:pre-wrap; word-break:break-word; font-size:.85em; font-family:inherit; }

/* Tabs */
.tabs { display:flex; gap:4px; margin-bottom:16px; }
.tab { padding:8px 16px; border-radius:8px 8px 0 0; cursor:pointer; color:var(--muted); font-size:.9em; border:1px solid transparent; border-bottom:none; }
.tab.active { color:var(--text); background:var(--card); border-color:var(--border); }
.tab:hover { color:var(--text); }

/* Toast */
.toast { position:fixed; bottom:20px; right:20px; background:var(--card); border:1px solid var(--accent); border-radius:8px; padding:12px 20px; font-size:.9em; z-index:999; opacity:0; transition:opacity .3s; }
.toast.show { opacity:1; }

.hidden { display:none !important; }

.badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:.75em; font-weight:600; }
.badge-thinking { background:rgba(78,205,196,.15); color:var(--accent2); }
.badge-error { background:rgba(255,107,107,.15); color:var(--danger); }
.badge-ok { background:rgba(81,207,102,.15); color:var(--success); }
</style>
</head>
<body>
<div class="app">
  <h1><span>Ollama</span> Benchmark</h1>
  <p class="subtitle">Model performance &amp; quality testing</p>

  <!-- TABS -->
  <div class="tabs">
    <div class="tab active" onclick="showTab('benchmarks')">Benchmarks</div>
    <div class="tab" onclick="showTab('setup')">Setup &amp; Run</div>
    <div class="tab" onclick="showTab('tests')">Tests</div>
    <div class="tab" onclick="showTab('results')">Results</div>
    <div class="tab" onclick="showTab('history')">History</div>
    <div class="tab" onclick="showTab('compare')">Compare</div>
  </div>

  <!-- SETUP TAB -->
  <div id="tab-setup" class="hidden">
    <div class="card">
      <h2>1. Ollama Host</h2>
      <div class="row">
        <input id="ollama-host" type="text" value="http://localhost:11434" style="flex:1;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.95em;">
        <button class="btn btn-secondary" onclick="fetchModels()">Connect</button>
      </div>
      <div id="gpu-info" style="margin-top:8px;font-size:.85em;color:var(--muted);"></div>
      <div style="margin-top:6px;display:flex;align-items:center;gap:8px;">
        <label style="font-size:.85em;color:var(--muted);white-space:nowrap;">GPU Override:</label>
        <input id="gpu-override" type="text" placeholder="e.g. 4x Tesla P40 (96GB)" style="flex:1;padding:4px 8px;border-radius:6px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85em;">
        <span style="font-size:.75em;color:var(--muted);">Auto-detects CPU/RAM/GPU on save</span>
      </div>
    </div>

    <div class="card">
      <h2>2. Select Models</h2>
      <div id="models-loading" class="status-text">Click Connect to load models...</div>
      <div id="models-grid" class="select-grid"></div>
    </div>

    <div class="card">
      <h2>3. Select Tests</h2>
      <div id="tests-grid" class="select-grid"></div>
    </div>

    <div class="card">
      <div class="row" style="justify-content:space-between;">
        <div>
          <h2 style="margin-bottom:0">4. Run Benchmark</h2>
          <div id="run-summary" class="status-text">Select models and tests above</div>
        </div>
        <div>
          <button id="btn-run" class="btn btn-primary" onclick="startBenchmark()" disabled>Start</button>
          <button id="btn-stop" class="btn btn-danger" onclick="stopBenchmark()" style="display:none">Stop</button>
        </div>
      </div>
      <div id="progress-section" class="hidden" style="margin-top:12px;">
        <div class="progress-bar"><div id="progress-fill" class="progress-fill" style="width:0%"></div></div>
        <div id="progress-text" class="status-text"></div>
        <div id="progress-log" style="margin-top:8px; font-size:.85em; color:var(--muted); max-height:200px; overflow-y:auto;"></div>
      </div>
    </div>
  </div>

  <!-- TESTS TAB -->
  <div id="tab-tests" class="hidden">
    <div class="card">
      <h2 style="margin-bottom:4px">Test Prompts</h2>
      <p style="color:var(--muted);font-size:.85em;margin-bottom:12px">View, edit (custom tests), or copy as new test. Default tests are read-only.</p>
      <div id="tests-list"></div>
      <button id="btn-new-test" class="btn btn-success" style="margin-top:16px" onclick="openTestEditor()">+ New Test</button>
    </div>

    <!-- Test Editor Modal -->
    <div id="test-editor-overlay" class="hidden" style="position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.6);z-index:1000;display:flex;align-items:center;justify-content:center">
      <div style="background:var(--bg-alt);border:1px solid var(--border);border-radius:12px;padding:24px;width:640px;max-height:80vh;overflow-y:auto">
        <h3 id="test-editor-title" style="margin:0 0 16px">New Test</h3>
        <label style="display:block;margin-bottom:8px"><span style="color:var(--muted);font-size:.85em">Name</span>
          <input id="te-name" type="text" style="display:block;width:100%;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);margin-top:4px" placeholder="e.g. Finnish Coding">
        </label>
        <label style="display:block;margin-bottom:8px"><span style="color:var(--muted);font-size:.85em">Prompt</span>
          <textarea id="te-prompt" rows="8" style="display:block;width:100%;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);margin-top:4px;font-family:monospace;font-size:.9em" placeholder="Enter the test prompt..."></textarea>
        </label>
        <label style="display:block;margin-bottom:8px"><span style="color:var(--muted);font-size:.85em">Expected Answer (optional, for objective/math tests)</span>
          <input id="te-expected" type="text" style="display:block;width:100%;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);margin-top:4px" placeholder="e.g. 240 — leave empty for subjective tests">
        </label>
        <label style="display:block;margin-bottom:16px"><span style="color:var(--muted);font-size:.85em">Rating Type</span>
          <select id="te-rating" style="display:block;width:100%;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);margin-top:4px">
            <option value="subjective">Subjective — needs human/LLM quality rating</option>
            <option value="objective">Objective — has a known correct answer</option>
            <option value="numerical">Numerical — pure speed metric, no quality rating</option>
          </select>
        </label>
        <div id="te-error" style="color:var(--danger);font-size:.85em;margin-bottom:8px"></div>
        <div style="display:flex;gap:8px;justify-content:flex-end">
          <button class="btn btn-secondary" onclick="closeTestEditor()">Cancel</button>
          <button id="te-save-btn" class="btn btn-success" onclick="saveTestEditor()">Save Test</button>
        </div>
      </div>
    </div>
  </div>

  <!-- RESULTS TAB -->
  <div id="tab-results" class="hidden">
    <div class="card">
      <h2>Benchmark Results</h2>
      <div id="results-content">
        <p class="status-text">No results yet. Run a benchmark first.</p>
      </div>
    </div>

    <!-- Rating section -->
    <div id="rating-section" class="card hidden">
      <h2>Quality Rating (4-10)</h2>
      <p class="status-text" style="margin-bottom:12px;">Rate subjective tests only. Numerical and objective tests are auto-evaluated.</p>
      <div id="rating-grid"></div>
      <div style="margin-top:16px;">
        <button class="btn btn-success" onclick="submitRatings()">Save Ratings</button>
      </div>
    </div>

    <!-- Judge section -->
    <div id="judge-section" class="card hidden">
      <h2>LLM-as-Judge</h2>
      <p class="status-text" style="margin-bottom:12px;">Select a judge model to evaluate response quality automatically (score 4-10). Numerical tests (pure speed metrics) are skipped.</p>
      <div class="row" style="margin-bottom:12px;">
        <select id="judge-model-select" style="flex:1;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.9em;">
          <option value="">-- Select judge model --</option>
        </select>
        <button id="btn-judge" class="btn btn-primary" onclick="runJudge()" disabled>Run Judge</button>
      </div>
      <div id="judge-progress" class="hidden">
        <div class="progress-bar"><div id="judge-progress-fill" class="progress-fill" style="width:0%"></div></div>
        <div id="judge-progress-text" class="status-text"></div>
      </div>
      <div id="judge-results"></div>
    </div>
  </div>

  <!-- HISTORY TAB -->
  <div id="tab-history" class="hidden">
    <div class="card">
      <h2>Past Benchmark Runs</h2>
      <div id="history-list">
        <p class="status-text">Loading...</p>
      </div>
    </div>
  </div>

  <div id="tab-compare" class="hidden">
    <div class="card">
      <h2>Cross-Run Comparison</h2>
      <p style="font-size:.85em;color:var(--muted);margin-bottom:12px;">Compare model performance across multiple benchmark runs. Select computers, models, tests, and a metric to visualize.</p>

      <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px;">
        <div style="flex:1;min-width:220px;">
          <label style="font-size:.85em;color:var(--muted);display:block;margin-bottom:4px;">Computers:</label>
          <div id="compare-computers" style="display:flex;gap:8px;flex-wrap:wrap;max-height:120px;overflow-y:auto;padding:4px;border:1px solid var(--border);border-radius:6px;"></div>
        </div>
        <div style="flex:1;min-width:220px;">
          <label style="font-size:.85em;color:var(--muted);display:block;margin-bottom:4px;">Models:</label>
          <div id="compare-models" style="display:flex;gap:8px;flex-wrap:wrap;max-height:120px;overflow-y:auto;padding:4px;border:1px solid var(--border);border-radius:6px;"></div>
        </div>
        <div style="flex:1;min-width:220px;">
          <label style="font-size:.85em;color:var(--muted);display:block;margin-bottom:4px;">Tests:</label>
          <div id="compare-tests" style="display:flex;gap:8px;flex-wrap:wrap;max-height:120px;overflow-y:auto;padding:4px;border:1px solid var(--border);border-radius:6px;"></div>
        </div>
      </div>

      <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin-bottom:16px;">
        <div>
          <label style="font-size:.85em;color:var(--muted);">Metric:</label>
          <select id="compare-metric" style="padding:4px 8px;border-radius:6px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.9em;min-width:140px;">
            <option value="tps">tok/s</option>
            <option value="wall_time">Wall Time</option>
            <option value="quality_rating">Quality Rating</option>
            <option value="judge_score">Judge Score</option>
            <option value="ttft">TTFT</option>
          </select>
        </div>
        <div style="display:flex;gap:8px;">
          <button class="btn btn-primary" onclick="runCompare()">Compare</button>
          <button class="btn btn-secondary" onclick="exportCompareCSV()">CSV</button>
          <button class="btn btn-secondary" onclick="exportComparePDF()">PDF</button>
          <button class="btn btn-secondary" onclick="exportCompareHTML()">HTML</button>
        </div>
      </div>

      <div id="compare-chart" style="margin-bottom:16px;"></div>
      <div id="compare-results"></div>
    </div>
  </div>

  <!-- BENCHMARKS TAB -->
  <div id="tab-benchmarks">
    <div class="card">
      <h2>Open Benchmark Suites</h2>
      <p style="font-size:.85em;color:var(--muted);margin-bottom:12px;">Run standardized LLM benchmarks (MMLU-Pro, IFEval, BFCL v3) with automatic scoring. Queue multiple models/suites to run sequentially.</p>

      <!-- 1. Select Suites -->
      <h3 style="margin-bottom:8px;">1. Select Suites</h3>
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;">
        <label style="font-size:.85em;color:var(--muted);">Apply sample size to all:</label>
        <input id="bm-sample-all" type="number" value="" min="1" max="999" placeholder="(use defaults)" style="width:80px;padding:4px 8px;border-radius:4px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.9em;">
        <button class="btn btn-secondary" style="font-size:.8em;padding:3px 10px;" onclick="applySampleSizeToAll()">Set All</button>
      </div>
      <div id="bm-suite-list" class="select-grid" style="margin-bottom:16px;"></div>

      <!-- 2. Select Models -->
      <h3 style="margin-bottom:8px;">2. Select Models</h3>
      <div style="display:flex;gap:8px;margin-bottom:8px;">
        <select id="bm-model-add" style="flex:1;padding:6px 10px;border-radius:6px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.95em;"></select>
        <button class="btn btn-secondary" onclick="bmAddModel()">Add</button>
      </div>
      <div id="bm-models-list" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px;min-height:32px;padding:8px;border:1px solid var(--border);border-radius:6px;">
        <span style="color:var(--muted);font-size:.85em;">No models selected</span>
      </div>

      <!-- 3. Config -->
      <h3 style="margin-bottom:8px;">3. Configuration</h3>
      <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px;">
        <div style="min-width:160px;">
          <label style="font-size:.85em;color:var(--muted);display:block;margin-bottom:4px;">Max parallel jobs:</label>
          <input id="bm-parallel" type="number" value="1" min="1" max="8" style="width:100%;padding:6px 10px;border-radius:6px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.95em;" onchange="setBmParallel(this.value)" title="Set to 1 unless you have separate GPU hosts — GPU time-slicing corrupts results">
        </div>
      </div>

      <!-- 4. Actions & Queue -->
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:16px;">
        <button id="btn-bm-run" class="btn btn-primary" onclick="runBenchmarkSuite()">Queue &amp; Run</button>
        <button id="btn-bm-stop" class="btn btn-danger" onclick="stopBenchmarkSuite()" style="display:none">Stop All</button>
        <button class="btn btn-secondary" onclick="profileModels()" title="Cold-load each model one at a time, measure VRAM/RAM/GPU%, then unload. Use this to populate scheduling data.">Profile Models</button>
        <button class="btn btn-secondary" onclick="downloadSuiteData()">Download Datasets</button>
        <button class="btn btn-secondary" onclick="clearBmQueue()" style="margin-left:auto;">Clear Queue</button>
      </div>

      <!-- Progress (Command Center HUD) -->
      <div id="bm-progress" style="display:none;margin-bottom:16px;">
        <div style="background:linear-gradient(180deg,#0a0c12,#12141c);border:1px solid rgba(108,140,255,.25);border-radius:4px;height:24px;overflow:hidden;position:relative;animation:pulse-glow 3s ease-in-out infinite;">
          <div id="bm-progress-bar" style="background:linear-gradient(90deg,#3a5fff,#6c8cff,#4ecdc4);height:100%;width:0%;transition:width 0.3s;position:relative;"></div>
          <div style="position:absolute;top:0;left:0;right:0;bottom:0;background:linear-gradient(180deg,rgba(255,255,255,.04) 0%,transparent 50%,rgba(0,0,0,.1) 100%);pointer-events:none;"></div>
        </div>
        <div id="bm-progress-text" style="font-size:.85em;color:#6c8cff;margin-top:6px;text-align:center;font-family:monospace;letter-spacing:.5px;"></div>
      </div>

      <!-- Live Streaming Panels Container (one panel per parallel job) -->
      <div id="bm-streams-container" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px;margin-bottom:16px;"></div>

      <!-- Queue -->
      <div id="bm-queue-section" style="display:none;margin-bottom:16px;">
        <h3 style="margin-bottom:8px;">Queue <span id="bm-queue-count" style="font-size:.85em;color:var(--muted);"></span></h3>
        <div id="bm-queue-list" style="border:1px solid rgba(108,140,255,.12);border-radius:4px;overflow:hidden;background:#080a10;"></div>
      </div>

      <!-- Profile Progress -->
      <div id="profile-progress-section" style="display:none;margin-bottom:16px;">
        <h3 style="margin-bottom:8px;">Profiling Progress <span id="profile-status-text" style="font-size:.85em;color:var(--muted);"></span></h3>
        <div id="profile-current" style="font-size:.9em;color:var(--accent);margin-bottom:4px;"></div>
      </div>

      <!-- Model Profiles -->
      <div id="profile-results-section" style="margin-bottom:16px;">
        <h3 style="margin-bottom:8px;cursor:pointer;" onclick="document.getElementById('profile-table').style.display=document.getElementById('profile-table').style.display==='none'?'':'none'">Model Profiles <span style="font-size:.75em;color:var(--muted);">◀ click to toggle</span></h3>
        <div id="profile-table" style="display:none;overflow-x:auto;"></div>
      </div>

      <h3 style="margin-top:20px;">Suite Results</h3>
      <div id="bm-results" style="overflow-x:auto;"></div>
    </div>
  </div>
  </div>

<div id="toast" class="toast"></div>

<script>
let currentRunId = null;
let currentResults = null;
let ollamaHost = '';

// ── Tabs ──
function showTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    t.classList.toggle('active', ['benchmarks','setup','tests','results','history','compare'][i] === name);
  });
  document.getElementById('tab-setup').classList.toggle('hidden', name !== 'setup');
  document.getElementById('tab-tests').classList.toggle('hidden', name !== 'tests');
  document.getElementById('tab-results').classList.toggle('hidden', name !== 'results');
  document.getElementById('tab-history').classList.toggle('hidden', name !== 'history');
  document.getElementById('tab-compare').classList.toggle('hidden', name !== 'compare');
  document.getElementById('tab-benchmarks').classList.toggle('hidden', name !== 'benchmarks');
  if (name === 'history') loadHistory();
  if (name === 'tests') loadTestsList();
  if (name === 'compare') loadCompare();
  if (name === 'benchmarks') { loadBenchmarks(); autoConnectSuiteSSE(); _initDragHandles(); }
}

// ── Toast ──
function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3000);
}

// ── Fetch Models ──
async function fetchModels() {
  ollamaHost = document.getElementById('ollama-host').value.replace(/\/+$/, '');
  document.getElementById('models-loading').textContent = 'Loading...';
  // Fetch system info in parallel
  fetch('/api/gpu').then(r => r.json()).then(d => {
    const gpuEl = document.getElementById('gpu-info');
    const parts = [];
    if (d.computer_id) parts.push('<strong>ID:</strong> ' + d.computer_id);
    if (d.gpu_summary) parts.push('<strong>GPU:</strong> ' + d.gpu_summary);
    if (d.cpu_summary) parts.push('<strong>CPU:</strong> ' + d.cpu_summary);
    if (d.ram_summary) parts.push('<strong>RAM:</strong> ' + d.ram_summary);
    if (d.cuda_version) parts.push('<strong>CUDA:</strong> ' + d.cuda_version);
    if (d.nvidia_driver) parts.push('<strong>Driver:</strong> ' + d.nvidia_driver);
    if (parts.length) {
      gpuEl.innerHTML = parts.join(' &nbsp;|&nbsp; ');
      gpuEl.style.color = 'var(--text)';
    } else {
      // Fallback to just gpu_summary if available
      gpuEl.textContent = '';
    }
  }).catch(() => {});
  try {
    const res = await fetch('/api/models?host=' + encodeURIComponent(ollamaHost));
    const models = await res.json();
    const grid = document.getElementById('models-grid');
    grid.innerHTML = '';
    models.forEach(m => {
      const sizeStr = m.size ? (m.size >= 1e9 ? Math.round(m.size/1e9) + 'GB' : Math.round(m.size/1e6) + 'MB') : 'cloud';
      grid.innerHTML += `<label class="select-item"><input type="checkbox" value="${m.name}" class="model-cb" onchange="updateSummary()"><span>${m.name}</span><span class="model-size">${sizeStr}</span></label>`;
    });
    populateJudgeModels(models);
    document.getElementById('models-loading').textContent = models.length + ' models found';
    updateSummary();
  } catch(e) {
    document.getElementById('models-loading').textContent = 'Error: ' + e.message;
  }
}

// ── Load Tests ──
async function loadTests() {
  const res = await fetch('/api/tests');
  const tests = await res.json();
  const grid = document.getElementById('tests-grid');
  grid.innerHTML = '';
  tests.forEach(t => {
    const tag = t.default ? ' (default)' : ' (custom)';
    const ratingBadge = t.rating_type === 'numerical' ? ' <span style="font-size:.7em;background:var(--muted);color:#fff;padding:1px 4px;border-radius:3px;">NUM</span>' :
                        t.rating_type === 'objective' ? ' <span style="font-size:.7em;background:var(--success);color:#fff;padding:1px 4px;border-radius:3px;">OBJ</span>' : '';
    grid.innerHTML += `<label class="select-item"><input type="checkbox" value="${t.name}" class="test-cb" ${t.default?'checked':''} onchange="updateSummary()"><span>${t.name}${tag}${ratingBadge}</span></label>`;
  });
  updateSummary();
}

// ── Tests Tab ──
async function loadTestsList() {
  const res = await fetch('/api/tests');
  const tests = await res.json();
  const list = document.getElementById('tests-list');
  list.innerHTML = '';
  tests.forEach(t => {
    const isDefault = t.default;
    const ratingBadge = t.rating_type === 'numerical' ? '<span style="font-size:.65em;padding:1px 5px;border-radius:3px;background:#2196f3;color:#fff;margin-left:6px">NUM</span>' :
                        t.rating_type === 'objective' ? '<span style="font-size:.65em;padding:1px 5px;border-radius:3px;background:#ff9800;color:#fff;margin-left:6px">OBJ</span>' : '';
    const typeTag = isDefault ? '<span style="font-size:.7em;padding:2px 6px;border-radius:4px;background:var(--border);color:var(--muted);margin-left:6px">default</span>' :
                                 '<span style="font-size:.7em;padding:2px 6px;border-radius:4px;background:#2a4a2a;color:#81c784;margin-left:6px">custom</span>';
    const promptPreview = t.prompt;
    const isLong = t.prompt.length > 200;
    const actions = isDefault
      ? `<button class="btn btn-secondary" style="padding:3px 10px;font-size:.75em" onclick="copyAsNewTest('${t.name.replace(/'/g,"\\'")}')">Copy as New</button>`
      : `<button class="btn btn-secondary" style="padding:3px 10px;font-size:.75em" onclick="editTest('${t.name.replace(/'/g,"\\'")}')">Edit</button>
         <button class="btn btn-secondary" style="padding:3px 10px;font-size:.75em" onclick="copyAsNewTest('${t.name.replace(/'/g,"\\'")}')">Copy as New</button>
         <button class="btn btn-danger" style="padding:3px 10px;font-size:.75em" onclick="deleteTest('${t.name.replace(/'/g,"\\'")}')">Delete</button>`;

    list.innerHTML += `
      <div style="margin-bottom:12px;padding:12px 16px;background:var(--bg);border:1px solid var(--border);border-radius:8px">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
          <strong style="font-size:1.05em">${t.name}</strong>${typeTag}${ratingBadge}
        </div>
        <div style="color:var(--text);font-size:.88em;line-height:1.5;margin-bottom:8px;padding:8px;background:var(--bg-alt);border-radius:6px;white-space:pre-wrap;font-family:monospace;${isLong ? 'max-height:120px;overflow-y:auto;' : ''}">${escHtml(promptPreview)}</div>
        ${t.expected_answer ? `<div style="font-size:.8em;color:var(--muted);margin-bottom:6px">Expected answer: <strong style="color:var(--success)">${t.expected_answer}</strong></div>` : ''}
        <div style="display:flex;gap:6px">${actions}</div>
      </div>`;
  });
}

function escHtml(s) {
  const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
}

// Store tests data for editing
let _testsData = [];
const _origLoadTests = loadTests;
loadTests = async function() {
  const res = await fetch('/api/tests');
  _testsData = await res.json();
  // Call original logic for checkboxes
  const grid = document.getElementById('tests-grid');
  grid.innerHTML = '';
  _testsData.forEach(t => {
    const tag = t.default ? ' (default)' : ' (custom)';
    const ratingBadge = t.rating_type === 'numerical' ? ' <span style="font-size:.7em;background:var(--muted);color:#fff;padding:1px 4px;border-radius:3px;">NUM</span>' :
                        t.rating_type === 'objective' ? ' <span style="font-size:.7em;background:var(--success);color:#fff;padding:1px 4px;border-radius:3px;">OBJ</span>' : '';
    grid.innerHTML += `<label class="select-item"><input type="checkbox" value="${t.name}" class="test-cb" ${t.default?'checked':''} onchange="updateSummary()"><span>${t.name}${tag}${ratingBadge}</span></label>`;
  });
  updateSummary();
};

function openTestEditor(data = null) {
  document.getElementById('te-name').value = data ? data.name : '';
  document.getElementById('te-name').disabled = false;
  document.getElementById('te-prompt').value = data ? data.prompt : '';
  document.getElementById('te-expected').value = data ? (data.expected_answer || '') : '';
  document.getElementById('te-rating').value = data ? data.rating_type : 'subjective';
  document.getElementById('te-error').textContent = '';
  document.getElementById('test-editor-title').textContent = data ? `Edit: ${data.name}` : 'New Test';
  document.getElementById('te-save-btn').textContent = data && !data.default ? 'Save Changes' : 'Create Test';
  document.getElementById('test-editor-overlay').classList.remove('hidden');
}

function closeTestEditor() {
  document.getElementById('test-editor-overlay').classList.add('hidden');
}

function editTest(name) {
  const t = _testsData.find(x => x.name === name);
  if (t) openTestEditor(t);
}

function copyAsNewTest(name) {
  const t = _testsData.find(x => x.name === name);
  if (!t) return;
  openTestEditor({name: name + ' (copy)', prompt: t.prompt, expected_answer: t.expected_answer, rating_type: t.rating_type, default: false});
}

async function saveTestEditor() {
  const name = document.getElementById('te-name').value.trim();
  const prompt = document.getElementById('te-prompt').value.trim();
  const expected_answer = document.getElementById('te-expected').value.trim();
  const rating_type = document.getElementById('te-rating').value;
  const errEl = document.getElementById('te-error');
  if (!name) { errEl.textContent = 'Name is required'; return; }
  if (!prompt) { errEl.textContent = 'Prompt is required'; return; }
  errEl.textContent = '';
  try {
    const res = await fetch('/api/tests/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name, prompt, expected_answer: expected_answer || null, rating_type})
    });
    const data = await res.json();
    if (data.error) { errEl.textContent = data.error; return; }
    closeTestEditor();
    loadTestsList();
    // Also refresh the test checkboxes on setup tab
    loadTests();
    toast(`Test "${name}" ${data.action}`);
  } catch(e) {
    errEl.textContent = 'Error: ' + e.message;
  }
}

async function deleteTest(name) {
  if (!confirm(`Delete custom test "${name}"?`)) return;
  try {
    const res = await fetch('/api/tests/delete', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name})
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    loadTestsList();
    loadTests();
    toast(`Deleted test "${name}"`);
  } catch(e) {
    alert('Error: ' + e.message);
  }
}

function updateSummary() {
  const models = [...document.querySelectorAll('.model-cb:checked')].map(c => c.value);
  const tests = [...document.querySelectorAll('.test-cb:checked')].map(c => c.value);
  const total = models.length * tests.length;
  document.getElementById('run-summary').textContent = `${models.length} models x ${tests.length} tests = ${total} runs`;
  document.getElementById('btn-run').disabled = total === 0;
}

// ── Start Benchmark ──
async function startBenchmark() {
  const models = [...document.querySelectorAll('.model-cb:checked')].map(c => c.value);
  const tests = [...document.querySelectorAll('.test-cb:checked')].map(c => c.value);
  if (!models.length || !tests.length) return;

  document.getElementById('btn-run').style.display = 'none';
  document.getElementById('btn-stop').style.display = '';
  document.getElementById('progress-section').classList.remove('hidden');
  document.getElementById('progress-log').innerHTML = '';

  try {
    const res = await fetch('/api/benchmark', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({host: ollamaHost, models, tests, gpu_override: document.getElementById('gpu-override').value.trim()})
    });
    const data = await res.json();
    if (data.error) { toast(data.error); resetBmButtons(); return; }
    toast('Benchmark started!');
  } catch(e) {
    toast('Error: ' + e.message);
    resetBmButtons();
    return;
  }

  // Subscribe to SSE for progress
  const evtSource = new EventSource('/api/stream');
  evtSource.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'progress') {
      const pct = (msg.progress / msg.total * 100).toFixed(0);
      document.getElementById('progress-fill').style.width = pct + '%';
      document.getElementById('progress-text').textContent = `[${msg.progress}/${msg.total}] ${msg.model} / ${msg.test}`;
    } else if (msg.type === 'test_done') {
      const errTag = msg.error ? ' <span class="badge badge-error">ERROR</span>' : '';
      document.getElementById('progress-log').innerHTML += `<div>${msg.model} / ${msg.test} ${msg.error ? 'ERROR' : msg.tps + ' tok/s, ' + msg.output_tokens + ' tok, ' + msg.wall_time + 's'}${errTag}</div>`;
    } else if (msg.type === 'done') {
      evtSource.close();
      currentRunId = msg.run_id;
      const stopped = msg.stopped === true;
      if (stopped) {
        document.getElementById('progress-text').textContent = 'Stopped! Partial results saved. Run #' + msg.run_id;
        toast('Benchmark stopped. Partial results saved. Run #' + msg.run_id);
      } else {
        document.getElementById('progress-text').textContent = 'Complete! Run #' + msg.run_id;
        document.getElementById('progress-fill').style.width = '100%';
        toast('Benchmark complete! Run #' + msg.run_id);
      }
      resetBmButtons();
      // Auto-load results
      if (msg.run_id) {
        loadResults(msg.run_id);
        showTab('results');
      }
    } else if (msg.type === 'stopping') {
      document.getElementById('progress-text').textContent = 'Stopping... saving partial results';
    }
  };
}

function resetBmButtons() {
  document.getElementById('btn-run').style.display = '';
  document.getElementById('btn-stop').style.display = 'none';
  document.getElementById('btn-stop').disabled = false;
  updateSummary();
}

// ── Stop Benchmark ──
async function stopBenchmark() {
  try {
    const res = await fetch('/api/benchmark/stop', {method: 'POST'});
    const data = await res.json();
    if (data.error) { toast(data.error); return; }
    toast('Stopping benchmark — will save partial results...');
    document.getElementById('btn-stop').disabled = true;
  } catch(e) {
    toast('Error: ' + e.message);
  }
}

// ── Load Results ──
async function loadResults(runId) {
  currentRunId = runId;
  try {
    const res = await fetch('/api/runs/' + runId);
    currentResults = await res.json();
    renderResults(currentResults);
  } catch(e) {
    document.getElementById('results-content').innerHTML = '<p class="status-text">Error loading results</p>';
  }
}

function renderResults(data) {
  if (!data || !data.models || !data.models.length) {
    document.getElementById('results-content').innerHTML = '<p class="status-text">No results to display.</p>';
    return;
  }

  // Summary table
  // GPU & run info header
  let runMeta = '';
  if (data.gpu_info || data.host) {
    runMeta = '<div style="margin-bottom:12px;padding:8px 12px;background:var(--bg-alt);border-radius:6px;font-size:.9em;display:flex;gap:20px;flex-wrap:wrap">';
    if (data.computer_id) runMeta += `<span><strong>ID:</strong> ${data.computer_id}</span>`;
    if (data.gpu_info) runMeta += `<span><strong>GPU:</strong> ${data.gpu_info}</span>`;
    const si = data.system_info || {};
    if (si.cpu_summary) runMeta += `<span><strong>CPU:</strong> ${si.cpu_summary}</span>`;
    if (si.ram_summary) runMeta += `<span><strong>RAM:</strong> ${si.ram_summary}</span>`;
    if (si.cuda_version) runMeta += `<span><strong>CUDA:</strong> ${si.cuda_version}</span>`;
    if (si.nvidia_driver) runMeta += `<span><strong>Driver:</strong> ${si.nvidia_driver}</span>`;
    if (data.host) runMeta += `<span><strong>Host:</strong> ${data.host}</span>`;
    if (data.timestamp) runMeta += `<span><strong>Run:</strong> #${data.run_id} &mdash; ${new Date(data.timestamp).toLocaleString()}</span>`;
    runMeta += '</div>';
  }
  const allTestNames = [];
  data.models.forEach(m => m.tests.forEach(t => { if (!allTestNames.includes(t.name)) allTestNames.push(t.name); }));

  let html = runMeta + '<table class="results-table"><thead><tr><th>Model</th>';
  allTestNames.forEach(tn => html += `<th>${tn}</th>`);
  html += '<th>Avg Rating</th><th>Judge</th><th>Math</th></tr></thead><tbody>';

  data.models.forEach(m => {
    let modelRatings = [];
    let judgeScores = [];
    let mathCorrect = 0;
    let mathTotal = 0;
    html += `<tr><td><strong>${m.name}</strong>${m.is_thinking ? ' <span class="badge badge-thinking">thinking</span>' : ''}</td>`;
    const tdict = {};
    m.tests.forEach(t => tdict[t.name] = t);
    allTestNames.forEach(tn => {
      const t = tdict[tn];
      if (!t) { html += '<td>-</td>'; }
      else if (t.stop_reason && t.stop_reason.startsWith('ERROR')) { html += '<td><span class="badge badge-error">ERROR</span></td>'; }
      else if (t.rating_type === 'numerical') {
        // Numerical test: just show tok/s and time, no quality column relevance
        html += `<td><strong>${t.tps}</strong> tok/s<br><span style="color:var(--muted);font-size:.8em">${t.output_tokens || '?'} tok, ${t.wall_time ? t.wall_time.toFixed(1) : '?'}s</span></td>`;
      }
      else {
        html += `<td><strong>${t.tps}</strong> tok/s<br><span style="color:var(--muted);font-size:.8em">${t.output_tokens || '?'} tok, ${t.wall_time ? t.wall_time.toFixed(1) : '?'}s</span></td>`;
        if (t.quality_rating) modelRatings.push(t.quality_rating);
        if (t.judge_score) judgeScores.push(t.judge_score);
        if (t.math_expected) { mathTotal++; if (t.math_correct) mathCorrect++; }
      }
    });
    const avgR = modelRatings.length ? (modelRatings.reduce((a,b)=>a+b,0)/modelRatings.length).toFixed(1) : '-';
    const avgJ = judgeScores.length ? (judgeScores.reduce((a,b)=>a+b,0)/judgeScores.length).toFixed(1) : '-';
    const mathStr = mathTotal ? (mathCorrect + '/' + mathTotal) : '-';
    html += `<td>${avgR}</td><td>${avgJ}</td><td>${mathStr}</td></tr>`;
  });
  html += '</tbody></table>';

  // Per-model detail
  data.models.forEach(m => {
    html += `<div style="margin-top:20px;"><h3 style="margin-bottom:8px;">${m.name}${m.is_thinking ? ' <span class="badge badge-thinking">thinking</span>' : ''}</h3>`;
    m.tests.forEach(t => {
      if (t.stop_reason && t.stop_reason.startsWith('ERROR')) {
        html += `<div style="margin:8px 0;"><strong>${t.name}</strong> <span class="badge badge-error">ERROR</span></div>`;
        return;
      }
      const isNumerical = t.rating_type === 'numerical';
      const isObjective = t.rating_type === 'objective';
      const metricsLine = `${t.tps} tok/s | ${t.output_tokens || '?'} out | ${t.wall_time ? t.wall_time.toFixed(1) : '?'}s`;
      const ratingLine = (!isNumerical && !isObjective && t.quality_rating) ? ' | Rating: '+t.quality_rating : '';
      html += `<div style="margin:12px 0;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <strong>${t.name}</strong>${isNumerical ? ' <span class="badge" style="background:var(--muted);color:#fff;font-size:.7em;">NUMERICAL</span>' : ''}${isObjective ? ' <span class="badge" style="background:var(--success);color:#fff;font-size:.7em;">OBJECTIVE</span>' : ''}
          <span style="color:var(--muted);font-size:.85em;">${metricsLine}${ratingLine}</span>
        </div>`;

      // Judge score display (skip for numerical tests)
      if (t.judge_score && !isNumerical) {
        html += `<div style="margin-top:4px;"><span style="color:var(--accent);">Judge (${t.judge_model}):</span> <strong>${t.judge_score}/10</strong>`;
        if (t.judge_reasoning) html += ` — ${escapeHtml(t.judge_reasoning.substring(0, 200))}`;
        html += `</div>`;
      }
      // Math verification display
      if (t.math_expected) {
        const icon = t.math_correct ? '✓' : '✗';
        const color = t.math_correct ? 'var(--success)' : 'var(--danger)';
        html += `<div style="margin-top:2px;"><span style="color:${color};">Math ${icon}</span> Expected: ${t.math_expected}, Got: ${t.math_answer || 'N/A'}</div>`;
      }

      // Thinking toggle
      if (t.thinking_excerpt) {
        html += `<div class="thinking-toggle" onclick="this.nextElementSibling.classList.toggle('hidden')">[+Show thinking]</div>
        <div class="thinking-box hidden">${escapeHtml(t.thinking_excerpt)}${t.thinking_excerpt.length >= 10000 ? '...(truncated)' : ''}</div>`;
      }

      // Full response
      const responseText = t.response_text || '(empty)';
      html += `<div class="response-box">${escapeHtml(responseText)}</div>`;
      html += `</div>`;
    });
    html += '</div>';
  });

  document.getElementById('results-content').innerHTML = html;

  // Show rating section (renderRatingGrid already hides it if no subjective tests)
  renderRatingGrid(data);

  // Show judge section
  document.getElementById('judge-section').classList.remove('hidden');
  document.getElementById('btn-judge').disabled = false;
}

function escapeHtml(text) {
  if (!text) return '';
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

// ── Rating ──
function renderRatingGrid(data) {
  let html = '';
  let hasRatingItems = false;
  data.models.forEach(m => {
    m.tests.forEach(t => {
      // Skip numerical and objective tests — they don't need quality ratings
      if (t.rating_type === 'numerical' || t.rating_type === 'objective') return;
      hasRatingItems = true;
      const key = `${m.name}|||${t.name}`;
      const currentRating = t.quality_rating || '';
      html += `<div class="rating-row">
        <span class="rating-label">${m.name} / ${t.name}</span>
        <input type="range" min="4" max="10" value="${currentRating || 7}" class="rating-slider" id="rate-${key}" oninput="document.getElementById('ratev-${key}').textContent=this.value">
        <span class="rating-val" id="ratev-${key}">${currentRating || '7'}</span>
      </div>`;
    });
  });
  document.getElementById('rating-grid').innerHTML = html;
  // Hide entire rating section if nothing to rate
  document.getElementById('rating-section').classList.toggle('hidden', !hasRatingItems);
}

async function submitRatings() {
  if (!currentResults || !currentRunId) return;
  const ratings = [];
  currentResults.models.forEach(m => {
    m.tests.forEach(t => {
      const key = `${m.name}|||${t.name}`;
      const slider = document.getElementById('rate-' + key);
      if (slider) {
        ratings.push({model: m.name, test: t.name, score: parseInt(slider.value)});
      }
    });
  });
  try {
    const res = await fetch('/api/rate/' + currentRunId, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ratings})
    });
    const data = await res.json();
    toast(data.saved + ' ratings saved!');
  } catch(e) {
    toast('Error saving ratings: ' + e.message);
  }
}

// ── History ──
async function loadHistory() {
  try {
    const res = await fetch('/api/runs');
    const runs = await res.json();
    if (!runs.length) {
      document.getElementById('history-list').innerHTML = '<p class="status-text">No benchmark runs yet.</p>';
      return;
    }
    let html = '<table class="results-table"><thead><tr><th>ID</th><th>Time</th><th>Computer</th><th>Hardware</th><th>Models</th><th>Tests</th><th>Actions</th></tr></thead><tbody>';
    runs.forEach(r => {
      const dt = new Date(r.timestamp);
      const timeStr = dt.toLocaleString();
      const si = r.system_info || {};
      const hwParts = [];
      if (r.gpu_info) hwParts.push(r.gpu_info);
      if (si.cpu_summary) hwParts.push(si.cpu_summary);
      if (si.ram_summary) hwParts.push(si.ram_summary);
      const hwStr = hwParts.length ? hwParts.join(' | ') : '-';
      const cid = r.computer_id || '-';
      html += `<tr>
        <td>#${r.id}</td>
        <td>${timeStr}</td>
        <td style="font-size:.75em;font-family:monospace;">${cid}</td>
        <td style="font-size:.75em;">${hwStr}</td>
        <td>${r.models}</td>
        <td>${r.tests}</td>
        <td>
          <button class="btn btn-secondary" style="padding:4px 10px;font-size:.8em;" onclick="loadResults(${r.id});showTab('results')">View</button>
          <button class="btn btn-secondary" style="padding:4px 10px;font-size:.8em;" onclick="exportCSV(${r.id})">CSV</button>
          <button class="btn btn-secondary" style="padding:4px 10px;font-size:.8em;" onclick="exportPDF(${r.id})">PDF</button>
          <button class="btn btn-secondary" style="padding:4px 10px;font-size:.8em;" onclick="exportHTML(${r.id})">HTML</button>
          <button class="btn btn-danger" style="padding:4px 10px;font-size:.8em;" onclick="deleteRun(${r.id})">Del</button>
        </td>
      </tr>`;
    });
    html += '</tbody></table>';
    document.getElementById('history-list').innerHTML = html;
  } catch(e) {
    document.getElementById('history-list').innerHTML = '<p class="status-text">Error loading history</p>';
  }
}

async function deleteRun(id) {
  if (!confirm('Delete run #' + id + '?')) return;
  await fetch('/api/runs/' + id, {method: 'DELETE'});
  toast('Run #' + id + ' deleted');
  loadHistory();
}

function exportCSV(id) {
  window.open('/api/export/csv/' + id, '_blank');
}

function exportPDF(id) {
  window.open('/api/export/pdf/' + id, '_blank');
}

function exportHTML(id) {
  window.open('/api/export/html/' + id, '_blank');
}

// ── Cross-Run Comparison ──
let compareComputerIds = [];
let compareModelNames = [];
let compareTestNames = [];
let compareData = null;  // store last fetch for exports

const COMPARE_COLORS = ['#7c3aed','#06b6d4','#f59e0b','#10b981','#ef4444','#3b82f6','#ec4899','#84cc16','#f97316','#6366f1'];

function cidColor(cid) {
  let h = 0;
  for (let i = 0; i < cid.length; i++) h = ((h << 5) - h + cid.charCodeAt(i)) | 0;
  return COMPARE_COLORS[Math.abs(h) % COMPARE_COLORS.length];
}

function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function loadCompare() {
  try {
    const res = await fetch('/api/compare');
    const data = await res.json();
    compareComputerIds = data.computer_ids || [];
    compareModelNames = data.model_names || [];
    compareTestNames = data.test_names || [];

    // Populate computer checkboxes
    const compDiv = document.getElementById('compare-computers');
    compDiv.innerHTML = '';
    compareComputerIds.forEach(cid => {
      compDiv.innerHTML += `<label style="font-size:.85em;display:flex;align-items:center;gap:4px;"><input type="checkbox" class="compare-cid-cb" value="${escHtml(cid)}" checked><span style="color:${cidColor(cid)}">${escHtml(cid)}</span></label>`;
    });
    if (!compareComputerIds.length) compDiv.innerHTML = '<span style="font-size:.8em;color:var(--muted)">No computers yet</span>';

    // Populate model checkboxes
    const modDiv = document.getElementById('compare-models');
    modDiv.innerHTML = '';
    compareModelNames.forEach(mn => {
      modDiv.innerHTML += `<label style="font-size:.85em;display:flex;align-items:center;gap:4px;"><input type="checkbox" class="compare-model-cb" value="${escHtml(mn)}" checked>${escHtml(mn)}</label>`;
    });
    if (!compareModelNames.length) modDiv.innerHTML = '<span style="font-size:.8em;color:var(--muted)">No models yet</span>';

    // Populate test checkboxes
    const testsDiv = document.getElementById('compare-tests');
    testsDiv.innerHTML = '';
    compareTestNames.forEach(tn => {
      testsDiv.innerHTML += `<label style="font-size:.85em;display:flex;align-items:center;gap:4px;"><input type="checkbox" class="compare-test-cb" value="${escHtml(tn)}" checked>${escHtml(tn)}</label>`;
    });
    if (!compareTestNames.length) testsDiv.innerHTML = '<span style="font-size:.8em;color:var(--muted)">No tests yet</span>';
  } catch(e) {
    console.error('Error loading compare data', e);
  }
}

async function runCompare() {
  const cids = Array.from(document.querySelectorAll('.compare-cid-cb:checked')).map(cb => cb.value);
  const models = Array.from(document.querySelectorAll('.compare-model-cb:checked')).map(cb => cb.value);
  const tests = Array.from(document.querySelectorAll('.compare-test-cb:checked')).map(cb => cb.value);
  if (!tests.length) {
    document.getElementById('compare-results').innerHTML = '<p class="status-text">Select at least one test.</p>';
    return;
  }

  let url = '/api/compare?';
  cids.forEach(c => url += 'computer_id=' + encodeURIComponent(c) + '&');
  models.forEach(m => url += 'model=' + encodeURIComponent(m) + '&');
  tests.forEach(t => url += 'test=' + encodeURIComponent(t) + '&');

  try {
    const res = await fetch(url);
    const data = await res.json();
    compareData = data;  // store for exports
    renderCompare(data, tests);
  } catch(e) {
    document.getElementById('compare-results').innerHTML = '<p class="status-text">Error loading comparison data.</p>';
  }
}

const METRIC_LABELS = {tps:'tok/s', wall_time:'Wall Time (s)', quality_rating:'Quality Rating', judge_score:'Judge Score', ttft:'TTFT (s)'};

function getMetricVal(e, metric) {
  // Return numeric value for the selected metric
  if (metric === 'tps') return e.tps;
  if (metric === 'wall_time') return e.wall_time;
  if (metric === 'quality_rating') return e.quality_rating;
  if (metric === 'judge_score') return e.judge_score;
  if (metric === 'ttft') return e.ttft || e.ttft_response;
  return null;
}

function fmtMetric(v, metric) {
  if (v == null) return '-';
  if (metric === 'tps' || metric === 'wall_time') return v.toFixed(1);
  if (metric === 'ttft') return v.toFixed(2);
  return v.toFixed(1);
}

function renderCompare(data, selectedTests) {
  const metric = document.getElementById('compare-metric').value;
  const results = data.results || {};
  const container = document.getElementById('compare-results');
  const chartDiv = document.getElementById('compare-chart');

  if (!Object.keys(results).length) {
    container.innerHTML = '<p class="status-text">No matching results found.</p>';
    chartDiv.innerHTML = '';
    return;
  }

  // Filter to selected tests
  const filtered = {};
  for (const [tn, entries] of Object.entries(results)) {
    if (selectedTests.includes(tn)) filtered[tn] = entries;
  }

  // Draw chart
  chartDiv.innerHTML = buildCompareChart(filtered, metric);

  // Draw table
  let html = '';
  const metricLabel = METRIC_LABELS[metric] || metric;
  for (const [testName, entries] of Object.entries(filtered)) {
    html += `<h3 style="color:var(--accent);margin:16px 0 8px;">${escHtml(testName)}</h3>`;
    html += `<table class="results-table"><thead><tr><th>Model</th><th>Computer</th><th>Run</th><th>${escHtml(metricLabel)}</th>`;
    // Add secondary metrics
    if (metric !== 'tps') html += '<th>tok/s</th>';
    if (metric !== 'wall_time') html += '<th>Wall Time</th>';
    html += '<th>Rating</th><th>Judge</th><th>Math</th></tr></thead><tbody>';

    // Sort by selected metric descending
    entries.sort((a, b) => {
      const va = getMetricVal(a, metric), vb = getMetricVal(b, metric);
      return (vb || 0) - (va || 0);
    });

    entries.forEach(e => {
      const val = getMetricVal(e, metric);
      const ratingStr = e.quality_rating ? e.quality_rating + '/10' : '-';
      const judgeStr = e.judge_score ? e.judge_score + '/10' : '-';
      const mathStr = e.math_correct != null ? (e.math_correct ? '&#10003;' : '&#10007;') : '-';
      html += `<tr>
        <td><strong>${escHtml(e.model)}</strong>${e.is_thinking ? ' <span class="badge badge-thinking">thinking</span>' : ''}</td>
        <td style="font-family:monospace;font-size:.8em;">${escHtml(e.computer_id || '-')}</td>
        <td style="font-size:.8em;">#${e.run_id}</td>
        <td><strong>${fmtMetric(val, metric)}</strong></td>`;
      if (metric !== 'tps') html += `<td>${e.tps != null ? e.tps.toFixed(1) : '-'}</td>`;
      if (metric !== 'wall_time') html += `<td>${e.wall_time != null ? e.wall_time.toFixed(1) + 's' : '-'}</td>`;
      html += `<td>${ratingStr}</td><td>${judgeStr}</td><td>${mathStr}</td></tr>`;
    });

    html += '</tbody></table>';
  }

  container.innerHTML = html || '<p class="status-text">No results.</p>';
}

function buildCompareChart(filtered, metric) {
  // Build a simple horizontal bar chart using SVG
  // Group by test, bars for each (model, computer) combination
  const testNames = Object.keys(filtered);
  if (!testNames.length) return '';

  // Collect all data points with their metric values
  const allEntries = [];
  let maxVal = 0;
  for (const [tn, entries] of Object.entries(filtered)) {
    for (const e of entries) {
      const v = getMetricVal(e, metric);
      if (v != null) {
        allEntries.push({test: tn, model: e.model, cid: e.computer_id || '-', value: v, run_id: e.run_id});
        if (v > maxVal) maxVal = v;
      }
    }
  }
  if (!allEntries.length) return '<p style="color:var(--muted);font-size:.85em;">No data for this metric.</p>';

  const barH = 22;
  const labelW = 200;
  const valW = 60;
  const gap = 4;
  const chartW = 700;
  const barMaxW = chartW - labelW - valW - 20;
  // Group entries by test
  const groups = {};
  allEntries.forEach(e => {
    if (!groups[e.test]) groups[e.test] = [];
    groups[e.test].push(e);
  });
  const groupKeys = Object.keys(groups);
  const totalBars = allEntries.length;
  const headerH = 24;
  const svgH = totalBars * (barH + gap) + groupKeys.length * headerH + 30;

  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${chartW + 20}" height="${svgH}" style="font-family:-apple-system,sans-serif;font-size:12px;">`;
  let y = 10;

  for (const tn of testNames) {
    const entries = groups[tn] || [];
    if (!entries.length) continue;
    // Test header
    svg += `<text x="10" y="${y + 14}" fill="var(--accent)" font-weight="bold" font-size="13">${escHtml(tn)}</text>`;
    y += headerH;

    // Sort entries by value descending
    entries.sort((a, b) => b.value - a.value);

    for (const e of entries) {
      const barW = maxVal > 0 ? (e.value / maxVal) * barMaxW : 0;
      const color = cidColor(e.cid);
      // Model label
      svg += `<text x="${labelW}" y="${y + 15}" text-anchor="end" fill="var(--text)" font-size="11">${escHtml(e.model.length > 22 ? e.model.substring(0,22)+'…' : e.model)}</text>`;
      // Bar
      svg += `<rect x="${labelW + 5}" y="${y}" width="${Math.max(barW, 2)}" height="${barH}" rx="3" fill="${color}" opacity="0.85"/>`;
      // Value label
      svg += `<text x="${labelW + 5 + Math.max(barW, 2) + 4}" y="${y + 15}" fill="var(--text)" font-size="10">${fmtMetric(e.value, metric)}</text>`;
      // Computer ID as small label below
      svg += `<text x="10" y="${y + 15}" fill="var(--muted)" font-size="9">${escHtml(e.cid)}</text>`;
      y += barH + gap;
    }
    y += 6;
  }

  svg += '</svg>';
  return svg;
}

function exportCompareCSV() {
  if (!compareData) return alert('Run a comparison first');
  postDownload('/api/compare/export/csv', compareData);
}
function exportCompareHTML() {
  if (!compareData) return alert('Run a comparison first');
  postDownload('/api/compare/export/html', compareData);
}
function exportComparePDF() {
  if (!compareData) return alert('Run a comparison first');
  postDownload('/api/compare/export/pdf', compareData);
}

function postDownload(url, data) {
  const blob = new Blob([JSON.stringify(data)], {type: 'application/json'});
  fetch(url, {method: 'POST', body: blob, headers: {'Content-Type': 'application/json'}})
    .then(r => {
      const ct = r.headers.get('content-type') || '';
      const cd = r.headers.get('content-disposition') || '';
      let filename = 'comparison';
      const m = cd.match(/filename="?([^";\n]+)"?/);
      if (m) filename = m[1];
      return r.blob().then(b => {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(b);
        a.download = filename;
        a.click();
      });
    })
    .catch(e => alert('Export failed: ' + e.message));
}

// ── Benchmark Suites ──
let bmSuites = {};
let bmSelectedSuites = [];  // multiple suites
let bmSelectedModels = [];  // multiple models
let bmPollInterval = null;
let bmSuiteSource = null;
let bmStreamStates = new Map();  // Map<job_id, {model, suite, question, text, thinking, tokens, tps, answers, waitCount, lastTokenTime}>
let _streamCursors = new Map();  // Map<job_id, cursor_element>

// Vintage modem speed labels
function _modemLabel(bps) {
  const tiers = [
    [300, '300 baud'],
    [1200, '1200 bps'],
    [2400, '2400 bps'],
    [9600, '9600 bps'],
    [14400, '14.4k'],
    [28800, '28.8k'],
    [33600, '33.6k'],
    [56000, '56k'],
    [128000, '128k ISDN'],
    [1544000, 'T1'],
    [10000000, '10M'],
    [100000000, '100M'],
    [1000000000, '1G'],
  ];
  for (const [speed, label] of tiers) {
    if (bps < speed) return '📡 ' + label;
  }
  return '📡 ' + (bps >= 1000000000 ? (bps/1000000000).toFixed(1)+'G' : (bps/1000000).toFixed(0)+'M');
}

// Blinking cursor for streaming panel (per-job)
function _updateStreamCursor(respEl, jobId) {
  const old = _streamCursors.get(jobId);
  if (old && old.parentNode) old.remove();
  const cursor = document.createElement('span');
  cursor.textContent = '▌';
  cursor.style.cssText = 'animation: blink-cursor 1s step-end infinite;color:#4ade80;';
  respEl.appendChild(cursor);
  _streamCursors.set(jobId, cursor);
  respEl.scrollTop = respEl.scrollHeight;
}

// ── Drag-handle resizing for streaming panel sections ──
function _initDragHandles() {
  // Drag handles are now created dynamically per-job in _createStreamPanel
  // This function is kept for backward compatibility but does nothing
}

function _initDrag(handleId, targetId, direction) {
  const handle = document.getElementById(handleId);
  const target = document.getElementById(targetId);
  if (!handle || !target) return;
  let startY, startH;
  handle.addEventListener('mousedown', (e) => {
    e.preventDefault();
    startY = e.clientY;
    startH = target.offsetHeight;
    const onMove = (ev) => {
      const dy = ev.clientY - startY;
      target.style.height = Math.max(60, startH + dy) + 'px';
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
}

// ── Command Center helpers ──
let _timers = new Map();        // Map<jobId, {start, interval}>
let _barHistory = new Map();    // Map<jobId, number[]> last N tps samples

// Idle indicator — shows "⏳ waiting for tokens..." when a job hasn't received tokens for 10+ seconds
function _showIdleIndicator(jobId) {
  let el = document.getElementById('bm-stream-idle-' + jobId);
  if (el) return;
  const respEl = document.getElementById('bm-stream-response-' + jobId);
  if (!respEl) return;
  el = document.createElement('div');
  el.id = 'bm-stream-idle-' + jobId;
  el.style.cssText = 'padding:6px 10px;margin:4px 0;border-radius:4px;background:rgba(108,140,255,0.08);color:#6c8cff;font-size:.85em;font-style:italic;';
  el.textContent = '⏳ Waiting for tokens… (model may be loading or GPU-shared)';
  respEl.appendChild(el);
}
function _hideIdleIndicator(jobId) {
  const el = document.getElementById('bm-stream-idle-' + jobId);
  if (el) el.remove();
}
// Check all active streams for idle state every 5s
setInterval(() => {
  const now = Date.now();
  for (const [jobId, state] of bmStreamStates.entries()) {
    if (state.lastTokenTime && (now - state.lastTokenTime) > 10000) {
      _showIdleIndicator(jobId);
    }
  }
}, 5000);

function _startTimer(jobId) {
  _stopTimer(jobId);
  const el = document.getElementById('bm-stream-elapsed-' + jobId);
  if (!el) return;
  const start = Date.now();
  const tick = () => {
    const s = Math.round((Date.now() - start) / 1000);
    const m = Math.floor(s / 60);
    const sec = s % 60;
    if (el) el.textContent = (m > 0 ? m + 'm ' : '') + sec + 's';
  };
  tick();
  const iv = setInterval(tick, 1000);
  _timers.set(jobId, {start, interval: iv});
}

function _stopTimer(jobId) {
  const t = _timers.get(jobId);
  if (t) { clearInterval(t.interval); _timers.delete(jobId); }
}

function _updateBarGraph(jobId, tps) {
  if (!tps || tps <= 0) return;
  const el = document.getElementById('bm-stream-bars-' + jobId);
  if (!el) return;
  let history = _barHistory.get(jobId) || [];
  history.push(tps);
  if (history.length > 20) history.shift();
  _barHistory.set(jobId, history);
  const maxTps = Math.max(...history, 1);
  let html = '';
  for (const v of history) {
    const h = Math.max(2, Math.round(v / maxTps * 14));
    const ratio = v / maxTps;
    const color = ratio > .66 ? '#4ade80' : ratio > .33 ? '#facc15' : '#f87171';
    html += '<div style="width:4px;height:' + h + 'px;background:' + color + ';border-radius:1px;flex-shrink:0;"></div>';
  }
  el.innerHTML = html;
}

function _resetBarGraph(jobId) {
  _barHistory.delete(jobId);
  const el = document.getElementById('bm-stream-bars-' + jobId);
  if (el) el.innerHTML = '';
}

// Create a streaming panel for a job — COMMAND CENTER EDITION
function _createStreamPanel(jobId, model, suite) {
  if (document.getElementById('bm-stream-' + jobId)) {
    document.getElementById('bm-stream-' + jobId).style.display = '';
    return;
  }
  const container = document.getElementById('bm-streams-container');
  const panel = document.createElement('div');
  panel.id = 'bm-stream-' + jobId;
  panel.className = 'cc-panel';
  panel.style.cssText = 'border-radius:4px;overflow:hidden;border:1px solid rgba(108,140,255,.2);box-shadow:0 0 20px rgba(108,140,255,.08),0 4px 24px rgba(0,0,0,.5);animation:panel-appear .4s ease-out;position:relative;background:#0a0c14;';
  panel.innerHTML = `
    <!-- Corner brackets (decorative HUD frame) -->
    <div style="position:absolute;top:0;left:0;width:20px;height:20px;border-top:2px solid rgba(108,140,255,.5);border-left:2px solid rgba(108,140,255,.5);pointer-events:none;z-index:2;"></div>
    <div style="position:absolute;top:0;right:0;width:20px;height:20px;border-top:2px solid rgba(108,140,255,.5);border-right:2px solid rgba(108,140,255,.5);pointer-events:none;z-index:2;"></div>
    <div style="position:absolute;bottom:0;left:0;width:20px;height:20px;border-bottom:2px solid rgba(108,140,255,.5);border-left:2px solid rgba(108,140,255,.5);pointer-events:none;z-index:2;"></div>
    <div style="position:absolute;bottom:0;right:0;width:20px;height:20px;border-bottom:2px solid rgba(108,140,255,.5);border-right:2px solid rgba(108,140,255,.5);pointer-events:none;z-index:2;"></div>

    <!-- Header bar -->
    <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 14px;background:linear-gradient(135deg,#0d1020 0%,#141830 100%);border-bottom:1px solid rgba(108,140,255,.15);">
      <span style="display:flex;align-items:center;gap:10px;">
        <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#ff4444;animation:live-pulse 1.5s ease-in-out infinite;" title="LIVE"></span>
        <span id="bm-stream-model-${jobId}" style="font-weight:700;font-size:.95em;color:#c8d0e8;font-family:monospace;letter-spacing:.3px;">${model}</span>
        <span style="color:rgba(108,140,255,.4);font-size:.8em;">|</span>
        <span style="font-size:.8em;color:rgba(108,140,255,.7);font-family:monospace;">${suite}</span>
      </span>
      <span style="display:flex;gap:12px;align-items:center;font-size:.85em;font-family:monospace;">
        <span id="bm-stream-score-${jobId}" style="color:#4ade80;font-weight:700;font-variant-numeric:tabular-nums;font-size:.85em;display:none;" title="Correct answers">0/0</span>
        <span id="bm-stream-elapsed-${jobId}" style="color:rgba(108,140,255,.6);font-variant-numeric:tabular-nums;font-size:.8em;"></span>
        <span id="bm-stream-wait-${jobId}" style="color:#f87171;font-weight:600;font-variant-numeric:tabular-nums;display:none;font-size:.8em;" title="Wait tokens in thinking"></span>
        <span id="bm-stream-tok-${jobId}" style="color:#556;font-variant-numeric:tabular-nums;font-size:.8em;"></span>
        <span id="bm-stream-tps-${jobId}" style="color:#4ade80;font-weight:700;font-variant-numeric:tabular-nums;"></span>
        <span id="bm-stream-modem-${jobId}" style="color:#facc15;font-size:.7em;font-variant-numeric:tabular-nums;opacity:.7;" title="Vintage modem speed"></span>
        <!-- Mini bar graph for tps history -->
        <span id="bm-stream-bars-${jobId}" style="display:flex;align-items:flex-end;gap:1px;height:16px;min-width:30px;"></span>
        <button class="btn btn-warning" onclick="skipQuestion('${jobId}')" style="font-size:.75em;padding:2px 8px;border-radius:3px;" title="Skip this question">&#9193; Skip</button>
        <button class="btn btn-danger" onclick="stopJob('${jobId}')" style="font-size:.75em;padding:2px 8px;border-radius:3px;" title="Stop this job">■ Stop</button>
      </span>
    </div>

    <!-- Question area -->
    <div style="position:relative;">
      <div style="position:absolute;top:6px;left:8px;font-size:.65em;color:rgba(108,140,255,.35);font-family:monospace;letter-spacing:1px;z-index:1;">PROMPT</div>
      <div id="bm-stream-question-${jobId}" style="padding:20px 14px 8px;font-size:.83em;color:#8090a8;border-bottom:1px solid rgba(108,140,255,.1);height:120px;overflow-y:auto;white-space:pre-wrap;font-family:monospace;background:#080a10;resize:vertical;"></div>
    </div>
    <div id="bm-stream-drag-q-${jobId}" style="height:3px;background:rgba(108,140,255,.08);cursor:ns-resize;"></div>

    <!-- Response area with scan-line overlay -->
    <div style="position:relative;">
      <div style="position:absolute;top:6px;left:8px;font-size:.65em;color:rgba(78,205,196,.3);font-family:monospace;letter-spacing:1px;z-index:1;">RESPONSE</div>
      <!-- Scan-line effect -->
      <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(180deg,transparent,rgba(108,140,255,.06),transparent);animation:scan-move 4s linear infinite;pointer-events:none;z-index:1;"></div>
      <div id="bm-stream-response-${jobId}" style="padding:20px 14px 12px;font-family:monospace;font-size:.92em;line-height:1.6;height:350px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;background:linear-gradient(180deg,#080a10,#0b0d16);color:#d4dae8;"></div>
    </div>
    <div id="bm-stream-drag-a-${jobId}" style="height:3px;background:rgba(108,140,255,.08);cursor:ns-resize;"></div>

    <!-- Answers log -->
    <div style="position:relative;">
      <div style="position:absolute;top:5px;left:8px;font-size:.65em;color:rgba(108,140,255,.3);font-family:monospace;letter-spacing:1px;z-index:1;">RESULTS</div>
      <div id="bm-stream-answers-${jobId}" style="padding:16px 14px 6px;border-top:1px solid rgba(108,140,255,.1);font-size:.83em;height:100px;overflow-y:auto;background:#080a10;font-family:monospace;"></div>
    </div>
  `;
  container.appendChild(panel);
  setTimeout(() => {
    _initDrag('bm-stream-drag-q-' + jobId, 'bm-stream-question-' + jobId, 'vertical');
    _initDrag('bm-stream-drag-a-' + jobId, 'bm-stream-response-' + jobId, 'vertical');
    _startTimer(jobId);
  }, 0);
}

// Remove a streaming panel for a completed job
function _removeStreamPanel(jobId) {
  _stopTimer(jobId);
  _barHistory.delete(jobId);
  const panel = document.getElementById('bm-stream-' + jobId);
  if (panel) {
    panel.style.transition = 'opacity 0.6s, max-height 0.6s';
    panel.style.opacity = '0';
    panel.style.maxHeight = '0';
    panel.style.overflow = 'hidden';
    setTimeout(() => panel.remove(), 700);
  }
  bmStreamStates.delete(jobId);
  _streamCursors.delete(jobId);
}

// Count "wait" variants in thinking text
function _countWaitTokens(text) {
  // Match "wait", "Wait", "w ait", etc. — models sometimes emit fragmented wait tokens
  const matches = text.match(/\b[Ww]\s*[Aa]\s*[Ii]\s*[Tt]\s*/g);
  return matches ? matches.length : 0;
}

function bmAddModel() {
  const sel = document.getElementById('bm-model-add');
  const model = sel.value;
  if (!model || bmSelectedModels.includes(model)) return;
  bmSelectedModels.push(model);
  renderBmModels();
}

function bmRemoveModel(model) {
  bmSelectedModels = bmSelectedModels.filter(m => m !== model);
  renderBmModels();
}

function renderBmModels() {
  const el = document.getElementById('bm-models-list');
  if (!bmSelectedModels.length) {
    el.innerHTML = '<span style="color:var(--muted);font-size:.85em;">No models selected</span>';
    return;
  }
  el.innerHTML = bmSelectedModels.map(m =>
    `<span style="display:inline-flex;align-items:center;gap:4px;padding:4px 10px;border-radius:6px;background:var(--accent);color:var(--bg);font-size:.85em;">${m} <span style="cursor:pointer;font-weight:bold;margin-left:2px;" onclick="bmRemoveModel('${m}')">&times;</span></span>`
  ).join('');
}

async function loadBenchmarks() {
  // Load suite list
  try {
    const res = await fetch('/api/suites');
    bmSuites = await res.json();
  } catch(e) { bmSuites = {}; }

  const el = document.getElementById('bm-suite-list');
  el.innerHTML = '';
  bmSelectedSuites = [];
  for (const [key, s] of Object.entries(bmSuites)) {
    const defaultSample = s.sample_size || 0;
    const div = document.createElement('div');
    div.className = 'select-item';
    div.dataset.key = key;
    div.innerHTML = `<input type="checkbox" id="bm-suite-${key}" style="margin-right:6px;">` +
      `<label for="bm-suite-${key}" style="cursor:pointer;flex:1;"><strong>${s.name}</strong><br><span style="font-size:.8em;color:var(--muted);">${s.description}</span></label>` +
      `<div style="display:flex;align-items:center;gap:4px;margin-left:auto;" id="bm-suite-ss-wrap-${key}">
        <label style="font-size:.75em;color:var(--muted);white-space:nowrap;">N=</label>
        <input type="number" id="bm-suite-ss-${key}" value="${defaultSample}" min="0" max="999" style="width:52px;padding:2px 4px;border-radius:4px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.85em;text-align:center;display:none;" title="Sample size (0=all)">
      </div>` +
      `<span style="margin-left:4px;font-size:.75em;color:${s.cached ? '#4caf50' : 'var(--muted)'}">${s.cached ? '✓' : ''}</span>`;
    const cb = div.querySelector('input[type="checkbox"]');
    const ssInput = div.querySelector(`#bm-suite-ss-${key}`);
    cb.addEventListener('change', () => {
      bmSelectedSuites = [...document.querySelectorAll('#bm-suite-list input[type="checkbox"]:checked')].map(i => i.closest('.select-item').dataset.key);
      ssInput.style.display = cb.checked ? '' : 'none';
    });
    // Default: mmlu_pro checked
    if (key === 'mmlu_pro') {
      cb.checked = true;
      ssInput.style.display = '';
      bmSelectedSuites.push(key);
    }
    el.appendChild(div);
  }

  // Populate model-add dropdown
  const modelSel = document.getElementById('bm-model-add');
  let models = [];
  const mainSelect = document.getElementById('model-select');
  if (mainSelect && mainSelect.options.length > 0) {
    for (const opt of mainSelect.options) {
      if (opt.value) models.push(opt.value);
    }
  }
  if (!models.length) {
    const ollamaHost = document.getElementById('ollama-host').value;
    try {
      const mRes = await fetch('/api/models?host=' + encodeURIComponent(ollamaHost));
      const mData = await mRes.json();
      if (mData.length) models = mData.map(m => m.name);
    } catch(e) {}
  }
  modelSel.innerHTML = '<option value="">-- Select model --</option>' + models.map(m => `<option value="${m}">${m}</option>`).join('');

  renderBmModels();

  // Load results
  await loadSuiteResults();

  // Check if suite is currently running — reconnect SSE
  try {
    const st = await fetch('/api/suites/status').then(r => r.json());
    if (st.max_parallel) {
      document.getElementById('bm-parallel').value = st.max_parallel;
    }
    if (st.active) {
      connectBmSSE();
      await refreshBmQueue();
    }
  } catch(e) {}

  // Load model profiles
  loadModelProfiles();

  // Check if profiling is running — reconnect polling
  try {
    const ps = await fetch('/api/profile/status').then(r => r.json());
    if (ps.active) {
      document.getElementById('profile-progress-section').style.display = 'block';
      pollProfileStatus();
    }
  } catch(e) {}
}

async function downloadSuiteData() {
  if (!bmSelectedSuites.length) return alert('Select at least one suite');
  for (const key of bmSelectedSuites) {
    try {
      const res = await fetch(`/api/suites/download/${key}`, {method: 'POST'});
      const data = await res.json();
      toast(data.error || `Downloaded ${data.rows} questions for ${key}`);
    } catch(e) { toast(`Download failed for ${key}: ${e.message}`); }
  }
  loadBenchmarks(); // refresh cached status
}

async function refreshBmQueue() {
  try {
    const res = await fetch('/api/suites/queue');
    const data = await res.json();
    const queueEl = document.getElementById('bm-queue-section');
    const listEl = document.getElementById('bm-queue-list');
    const countEl = document.getElementById('bm-queue-count');

    const items = [];
    // Show active jobs (may be multiple in parallel)
    if (data.active_jobs && data.active_jobs.length) {
      for (const job of data.active_jobs) {
        const pct = job.total_questions > 0 ? Math.round(job.current_question / job.total_questions * 100) : 0;
        items.push(`<div style="padding:6px 12px;background:rgba(108,140,255,.15);color:#c8d0e8;font-size:.85em;display:flex;justify-content:space-between;font-family:monospace;border-left:3px solid var(--accent);">
          <span style="color:#4ade80;">&#9654;</span> ${job.model} — ${job.suite_key} (${job.current_question}/${job.total_questions})
          <span style="color:#6c8cff;">${pct}%</span>
        </div>`);
      }
    }
    if (data.queue && data.queue.length) {
      for (let qi = 0; qi < data.queue.length; qi++) {
        const job = data.queue[qi];
        items.push(`<div style="padding:6px 12px;border-bottom:1px solid rgba(108,140,255,.08);font-size:.85em;display:flex;justify-content:space-between;align-items:center;font-family:monospace;color:#556;">
          <span>${job.model} — ${job.suite_key}</span>
          <div style="display:flex;gap:6px;align-items:center;">
            <span style="color:#556;">queued</span>
            <button onclick="deleteQueueItem(${qi})" style="background:none;border:1px solid rgba(248,113,113,.3);color:#f87171;border-radius:3px;padding:1px 6px;cursor:pointer;font-size:.8em;">✕</button>
          </div>
        </div>`);
      }
    }
    const total = (data.active_jobs ? data.active_jobs.length : 0) + (data.queue ? data.queue.length : 0);
    if (items.length) {
      queueEl.style.display = 'block';
      listEl.innerHTML = items.join('');
      countEl.textContent = `(${total} job${total > 1 ? 's' : ''}${data.max_parallel > 1 ? ', ' + data.max_parallel + ' parallel' : ''})`;
    } else {
      queueEl.style.display = 'none';
    }
  } catch(e) {}
}

async function clearBmQueue() {
  try {
    await fetch('/api/suites/queue', {method: 'DELETE'});
    toast('Queue cleared');
    await refreshBmQueue();
  } catch(e) { toast('Failed to clear queue'); }
}

async function deleteQueueItem(index) {
  try {
    const res = await fetch('/api/suites/queue/' + index, {method: 'DELETE'});
    const data = await res.json();
    if (data.removed) {
      toast(`Removed ${data.removed.model} — ${data.removed.suite_key}`);
    } else {
      toast(data.error || 'Failed to delete');
    }
    await refreshBmQueue();
  } catch(e) { toast('Failed to delete'); }
}

async function stopBenchmarkSuite() {
  try {
    const res = await fetch('/api/suites/stop', {method: 'POST'});
    const data = await res.json();
    if (data.error) { toast(data.error); return; }
    toast('Stopping suite — clearing queue and saving partial results...');
    document.getElementById('btn-bm-stop').disabled = true;
  } catch(e) { toast('Error: ' + e.message); }
}

async function skipQuestion(jobId) {
  try {
    const res = await fetch('/api/suites/skip/' + jobId, {method: 'POST'});
    const data = await res.json();
    if (data.skipped) {
      toast('Skipping Q' + data.question + ' for ' + data.model + '...');
      // Disable the skip button for this specific panel
      const panel = document.getElementById('bm-stream-' + jobId);
      if (panel) {
        const skipBtn = panel.querySelector('.btn-warning');
        if (skipBtn) { skipBtn.disabled = true; skipBtn.textContent = 'Skipping...'; }
        setTimeout(() => { if (skipBtn) { skipBtn.disabled = false; skipBtn.textContent = '⏭ Skip'; } }, 3000);
      }
    } else {
      toast(data.error || 'Skip failed');
    }
  } catch(e) { toast('Error: ' + e.message); }
}

async function stopJob(jobId) {
  try {
    const res = await fetch('/api/suites/stop/' + jobId, {method: 'POST'});
    const data = await res.json();
    if (data.stopping || data.stopped) {
      const label = data.was_queued ? 'Removed from queue' : 'Stopping ' + data.model + '...';
      toast(label);
      const panel = document.getElementById('bm-stream-' + jobId);
      if (panel) {
        const stopBtn = panel.querySelector('.btn-danger');
        if (stopBtn) { stopBtn.disabled = true; stopBtn.textContent = 'Stopping...'; }
      }
    } else {
      toast(data.error || 'Stop failed');
    }
  } catch(e) { toast('Error: ' + e.message); }
}

async function setBmParallel(n) {
  n = Math.max(1, Math.min(8, parseInt(n) || 2));
  document.getElementById('bm-parallel').value = n;
  try {
    await fetch('/api/suites/max-parallel', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({max_parallel: n})
    });
  } catch(e) {}
}

// ── Model Profiling ──
async function profileModels() {
  if (!bmSelectedModels.length) return alert('Select at least one model first');
  const host = document.getElementById('bm-host')?.value || ollamaHost || 'http://localhost:11434';
  try {
    const res = await fetch('/api/profile/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({models: bmSelectedModels, host: host})
    });
    const data = await res.json();
    toast(`Profiling ${data.queued} model(s)...`);
    document.getElementById('profile-progress-section').style.display = 'block';
    pollProfileStatus();
  } catch(e) { toast('Failed to start profiling'); }
}

async function pollProfileStatus() {
  try {
    const res = await fetch('/api/profile/status');
    const data = await res.json();
    const el = document.getElementById('profile-current');
    const statusEl = document.getElementById('profile-status-text');
    if (data.active) {
      const stages = ['idle', 'Loading model...', 'Measuring VRAM...', 'Done'];
      el.textContent = `Profiling: ${data.current_model || '...'} (${stages[data.progress] || '...'})`;
      statusEl.textContent = `${data.queue_length} remaining`;
      setTimeout(pollProfileStatus, 3000);
    } else {
      el.textContent = '';
      statusEl.textContent = '';
      document.getElementById('profile-progress-section').style.display = 'none';
      toast('Profiling complete');
      loadModelProfiles();
    }
  } catch(e) {
    setTimeout(pollProfileStatus, 3000);
  }
}

async function loadModelProfiles() {
  try {
    const res = await fetch('/api/profile/results');
    const data = await res.json();
    if (!data.profiles || !data.profiles.length) {
      document.getElementById('profile-table').innerHTML = '<p style="color:var(--muted);font-size:.85em;">No profiles yet. Click "Profile Models" to measure VRAM usage.</p>';
      return;
    }
    let html = '<table style="width:100%;border-collapse:collapse;font-size:.85em;">';
    html += '<tr style="border-bottom:2px solid var(--border);"><th style="text-align:left;padding:6px;">Model</th><th style="text-align:right;padding:6px;">Params</th><th style="text-align:right;padding:6px;">Quant</th><th style="text-align:right;padding:6px;">VRAM</th><th style="text-align:right;padding:6px;">Context</th><th style="text-align:right;padding:6px;">GPU%</th><th style="text-align:right;padding:6px;">Profiled</th></tr>';
    for (const p of data.profiles) {
      const vram = p.vram_mib ? (p.vram_mib / 1024).toFixed(1) + ' GB' : '—';
      const ctx = p.context_length ? p.context_length.toLocaleString() : '—';
      const gpu = p.gpu_pct ? p.gpu_pct + '%' : '—';
      const ago = p.profiled_at ? new Date(p.profiled_at).toLocaleDateString() : '—';
      html += `<tr style="border-bottom:1px solid var(--border);"><td style="padding:6px;">${p.model}</td><td style="text-align:right;padding:6px;">${p.parameter_size || '—'}</td><td style="text-align:right;padding:6px;">${p.quantization || '—'}</td><td style="text-align:right;padding:6px;font-weight:bold;color:var(--accent);">${vram}</td><td style="text-align:right;padding:6px;">${ctx}</td><td style="text-align:right;padding:6px;">${gpu}</td><td style="text-align:right;padding:6px;">${ago}</td></tr>`;
    }
    html += '</table>';
    document.getElementById('profile-table').innerHTML = html;
    document.getElementById('profile-table').style.display = '';
  } catch(e) {}
}

async function autoConnectSuiteSSE() {
  // Auto-connect to suite SSE if a suite is already running (e.g. page refresh)
  if (bmSuiteSource) return;  // already connected
  try {
    const resp = await fetch('/api/suites/status');
    const st = await resp.json();
    if (st.active || (st.queue && st.queue.length > 0)) {
      connectBmSSE();
      // Create streaming panels for all active jobs
      if (st.active_jobs && st.active_jobs.length > 0) {
        document.getElementById('bm-progress').style.display = 'block';
        document.getElementById('btn-bm-stop').style.display = '';
        for (const job of st.active_jobs) {
          _createStreamPanel(job.job_id, job.model, job.suite_key);
          bmStreamStates.set(job.job_id, {model: job.model, suite: job.suite_key, question: 0, text: '', thinking: false, tokens: 0, tps: 0, answers: [], waitCount: 0, correctCount: 0, totalCount: 0, lastTokenTime: Date.now()});
        }
      }
    }
  } catch(e) {}
}

function connectBmSSE() {
  if (bmSuiteSource) { bmSuiteSource.close(); bmSuiteSource = null; }
  if (bmPollInterval) { clearInterval(bmPollInterval); bmPollInterval = null; }

  document.getElementById('bm-progress').style.display = 'block';
  document.getElementById('btn-bm-stop').style.display = '';
  document.getElementById('btn-bm-stop').disabled = false;

  bmSuiteSource = new EventSource('/api/suites/stream');

  // Create streaming panels for already-active jobs (missed job_start events)
  (async () => {
    try {
      const res = await fetch('/api/suites/queue');
      const data = await res.json();
      if (data.active_jobs) {
        for (const job of data.active_jobs) {
          if (!document.getElementById('bm-stream-' + job.job_id)) {
            _createStreamPanel(job.job_id, job.model, job.suite_key);
            bmStreamStates.set(job.job_id, {model: job.model, suite: job.suite_key, question: job.current_question, text: '', thinking: false, tokens: 0, tps: 0, answers: [], waitCount: 0, correctCount: 0, totalCount: 0, lastTokenTime: Date.now()});
          }
        }
      }
    } catch(e) {}
  })();

  bmSuiteSource.onmessage = (evt) => {
    try {
      const d = JSON.parse(evt.data);
      if (d.type === 'ping') return;

      const jobId = d.job_id;  // All events carry job_id for parallel routing

      if (d.type === 'progress') {
        // Global progress bar — update based on the first active job (or this job)
        const pct = Math.round(d.question / d.total * 100);
        document.getElementById('bm-progress-bar').style.width = pct + '%';
        document.getElementById('bm-progress-text').textContent = `${d.question}/${d.total} — ${d.model} on ${d.suite}`;
        refreshBmQueue();  // update queue display with per-job progress
      } else if (d.type === 'job_start') {
        document.getElementById('bm-progress').style.display = 'block';
        document.getElementById('bm-progress-bar').style.width = '0%';
        document.getElementById('bm-progress-text').textContent = `Starting ${d.model} on ${d.suite} (${d.total_questions} questions)...`;
        document.getElementById('btn-bm-stop').style.display = '';
        // Create streaming panel for this job
        _createStreamPanel(jobId, d.model, d.suite);
        bmStreamStates.set(jobId, {model: d.model, suite: d.suite, question: 0, text: '', thinking: false, tokens: 0, tps: 0, answers: [], waitCount: 0, correctCount: 0, totalCount: 0, lastTokenTime: Date.now()});
        refreshBmQueue();
      } else if (d.type === 'question_start') {
        // New question — reset this job's streaming state
        const state = bmStreamStates.get(jobId);
        if (!state) return;
        state.question = d.question;
        state.text = '';
        state.thinking = false;
        state.tokens = 0;
        state.tps = 0;
        state.waitCount = 0;
        state.lastTokenTime = Date.now();
        state._firstTokenSeen = false;  // don't start timer until first token
        // Show "Loading…" instead of starting timer immediately (model may be loading)
        _stopTimer(jobId);
        const elapsedEl = document.getElementById('bm-stream-elapsed-' + jobId);
        if (elapsedEl) elapsedEl.textContent = 'Loading…';
        _resetBarGraph(jobId);
        _hideIdleIndicator(jobId);
        document.getElementById('bm-stream-model-' + jobId).textContent = d.model + ' #' + d.question;
        document.getElementById('bm-stream-tok-' + jobId).textContent = '';
        document.getElementById('bm-stream-tps-' + jobId).textContent = '';
        document.getElementById('bm-stream-modem-' + jobId).textContent = '';
        const waitEl = document.getElementById('bm-stream-wait-' + jobId);
        if (waitEl) { waitEl.style.display = 'none'; waitEl.textContent = ''; }
        document.getElementById('bm-stream-question-' + jobId).textContent = d.prompt || '';
        document.getElementById('bm-stream-response-' + jobId).innerHTML = '';
        document.getElementById('bm-stream-answers-' + jobId).innerHTML = state.answers.join('');
      } else if (d.type === 'thinking') {
        const state = bmStreamStates.get(jobId);
        if (!state) return;
        state.thinking = true;
        state.tokens = d.token_count;
        state.lastTokenTime = Date.now();
        // Start the elapsed timer on first token (not during model load)
        if (!state._firstTokenSeen) { state._firstTokenSeen = true; _startTimer(jobId); }
        _hideIdleIndicator(jobId);
        // Count "wait" tokens in this chunk
        const waitInChunk = _countWaitTokens(d.text || '');
        if (waitInChunk > 0) {
          state.waitCount += waitInChunk;
          const waitEl = document.getElementById('bm-stream-wait-' + jobId);
          if (waitEl) {
            waitEl.style.display = '';
            waitEl.textContent = '⏳ wait: ' + state.waitCount;
          }
        }
        const respEl = document.getElementById('bm-stream-response-' + jobId);
        if (respEl) {
          let span = document.createElement('span');
          span.style.color = '#6b7b8d';
          span.textContent = d.text;
          respEl.appendChild(span);
          _updateStreamCursor(respEl, jobId);
        }
        // Update tok/tps counters for thinking tokens too
        const elapsed = d.elapsed || 0;
        const tps = elapsed > 0 ? Math.round(d.token_count / elapsed) : 0;
        const tokEl = document.getElementById('bm-stream-tok-' + jobId);
        const tpsEl = document.getElementById('bm-stream-tps-' + jobId);
        const modemEl = document.getElementById('bm-stream-modem-' + jobId);
        if (tokEl) tokEl.textContent = d.token_count + ' tok';
        if (tpsEl) tpsEl.textContent = tps ? tps + ' tok/s' : '';
        if (modemEl) modemEl.textContent = tps ? _modemLabel(tps * 40) : '';
        if (tps) _updateBarGraph(jobId, tps);
      } else if (d.type === 'token') {
        const state = bmStreamStates.get(jobId);
        if (!state) return;
        if (state.thinking) {
          state.thinking = false;
          // Add line break between thinking and response
          const respEl = document.getElementById('bm-stream-response-' + jobId);
          if (respEl) { let br = document.createElement('br'); respEl.appendChild(br); }
        }
        state.tokens = d.token_count;
        state.tps = d.live_tps || 0;
        state.lastTokenTime = Date.now();
        // Start the elapsed timer on first token (not during model load)
        if (!state._firstTokenSeen) { state._firstTokenSeen = true; _startTimer(jobId); }
        _hideIdleIndicator(jobId);
        const respEl = document.getElementById('bm-stream-response-' + jobId);
        if (respEl) {
          respEl.appendChild(document.createTextNode(d.text));
          _updateStreamCursor(respEl, jobId);
        }
        const tokEl = document.getElementById('bm-stream-tok-' + jobId);
        const tpsEl = document.getElementById('bm-stream-tps-' + jobId);
        const modemEl = document.getElementById('bm-stream-modem-' + jobId);
        if (tokEl) tokEl.textContent = d.token_count + ' tok';
        if (tpsEl) tpsEl.textContent = d.live_tps ? d.live_tps + ' tok/s' : '';
        if (modemEl) modemEl.textContent = d.live_tps ? _modemLabel(d.live_tps * 40) : '';
        if (d.live_tps) _updateBarGraph(jobId, d.live_tps);
      } else if (d.type === 'answer') {
        const state = bmStreamStates.get(jobId);
        if (!state) return;
        _hideIdleIndicator(jobId);
        // Remove cursor at end of question
        const cursor = _streamCursors.get(jobId);
        if (cursor && cursor.parentNode) cursor.remove();
        // Flash the response area
        const respFlash = document.getElementById('bm-stream-response-' + jobId);
        if (respFlash) {
          respFlash.style.animation = d.correct ? 'answer-flash .6s ease-out' : 'answer-flash-wrong .6s ease-out';
          setTimeout(() => { if (respFlash) respFlash.style.animation = ''; }, 600);
        }
        // Show answer with ✓/✗ indicator
        const mark = d.correct ? '\u2713' : '\u2717';
        const color = d.correct ? 'var(--success, #4ade80)' : 'var(--danger, #f87171)';
        const div = document.createElement('div');
        div.style.cssText = 'margin:2px 0;padding:2px 4px;border-radius:2px;';
        div.innerHTML = `<span style="color:${color};font-weight:700;">${mark}</span> Q${d.question}: <span style="color:#4ecdc4">${d.answer || '?'}</span>` +
          (d.expected ? ` <span style="color:#556;">exp:${d.expected}</span>` : '') +
          ` <span style="color:#556;font-size:.8em;">${d.tps ? d.tps.toFixed(1) + ' t/s' : ''}</span>`;
        state.answers.push(div.outerHTML);
        const answersEl = document.getElementById('bm-stream-answers-' + jobId);
        if (answersEl) {
          answersEl.innerHTML = state.answers.join('');
          answersEl.scrollTop = answersEl.scrollHeight;
        }
        // Update score counter in header
        state.totalCount++;
        if (d.correct) state.correctCount++;
        const scoreEl = document.getElementById('bm-stream-score-' + jobId);
        if (scoreEl) {
          scoreEl.style.display = '';
          const pct = state.totalCount > 0 ? (state.correctCount / state.totalCount * 100).toFixed(0) : 0;
          scoreEl.textContent = state.correctCount + '/' + state.totalCount;
          scoreEl.title = state.correctCount + ' correct of ' + state.totalCount + ' (' + pct + '%)';
          scoreEl.style.color = pct >= 80 ? '#4ade80' : pct >= 50 ? '#facc15' : '#f87171';
        }
      } else if (d.type === 'stream_error') {
        const respEl = document.getElementById('bm-stream-response-' + jobId);
        if (respEl) respEl.innerHTML += `<span style="color:var(--danger,red);">Error: ${d.error}</span>`;
      } else if (d.type === 'prereq_progress') {
        // Memory suite prerequisite progress — show in response area
        const state = bmStreamStates.get(jobId);
        if (!state) return;
        const respEl = document.getElementById('bm-stream-response-' + jobId);
        if (respEl) {
          const pct = d.total_prereqs > 0 ? Math.round(d.prereq / d.total_prereqs * 100) : 0;
          const bar = '█'.repeat(Math.round(pct / 5)) + '░'.repeat(20 - Math.round(pct / 5));
          respEl.innerHTML = `<span style="color:#6c8cff;">🔄 ${d.message}</span>\n<span style="color:#556;">${bar} ${pct}%</span>`;
        }
      } else if (d.type === 'question_skipped') {
        const state = bmStreamStates.get(jobId);
        if (!state) return;
        const div = document.createElement('div');
        div.style.margin = '2px 0';
        div.innerHTML = `<span style="color:#e6a237;font-weight:700;">⏭</span> Q${d.question}: <span style="color:#e6a237;">Skipped</span>`;
        state.answers.push(div.outerHTML);
        const answersEl = document.getElementById('bm-stream-answers-' + jobId);
        if (answersEl) {
          answersEl.innerHTML = state.answers.join('');
          answersEl.scrollTop = answersEl.scrollHeight;
        }
        const respEl = document.getElementById('bm-stream-response-' + jobId);
        if (respEl) respEl.innerHTML += '<span style="color:#e6a237;font-weight:600;">⏭ Question skipped</span>\n';
      } else if (d.type === 'repetition_detected') {
        const state = bmStreamStates.get(jobId);
        if (!state) return;
        _hideIdleIndicator(jobId);
        const div = document.createElement('div');
        div.style.margin = '2px 0';
        div.innerHTML = `<span style="color:#f87171;font-weight:700;">🔁</span> Q${d.question}: <span style="color:#f87171;">Repetition loop — auto-skipped</span>`;
        state.answers.push(div.outerHTML);
        const answersEl = document.getElementById('bm-stream-answers-' + jobId);
        if (answersEl) {
          answersEl.innerHTML = state.answers.join('');
          answersEl.scrollTop = answersEl.scrollHeight;
        }
        const respEl = document.getElementById('bm-stream-response-' + jobId);
        if (respEl) respEl.innerHTML += '<span style="color:#f87171;font-weight:600;">🔁 Repetition loop detected — skipping question</span>\n';
      } else if (d.type === 'thinking_cap') {
        const state = bmStreamStates.get(jobId);
        if (!state) return;
        _hideIdleIndicator(jobId);
        const div = document.createElement('div');
        div.style.margin = '2px 0';
        const tokStr = d.thinking_tokens > 1000 ? (d.thinking_tokens / 1000).toFixed(1) + 'k' : d.thinking_tokens;
        div.innerHTML = `<span style="color:#fb923c;font-weight:700;">🧠</span> Q${d.question}: <span style="color:#fb923c;">Thinking cap hit (${tokStr} tokens) — no answer produced</span>`;
        state.answers.push(div.outerHTML);
        const answersEl = document.getElementById('bm-stream-answers-' + jobId);
        if (answersEl) {
          answersEl.innerHTML = state.answers.join('');
          answersEl.scrollTop = answersEl.scrollHeight;
        }
        const respEl = document.getElementById('bm-stream-response-' + jobId);
        if (respEl) respEl.innerHTML += '<span style="color:#fb923c;font-weight:600;">🧠 Thinking cap — model used all tokens on thinking, skipping</span>\n';
      } else if (d.type === 'job_stopping') {
        // Individual job stopping — fade the panel
        const panel = document.getElementById('bm-stream-' + jobId);
        if (panel) {
          panel.style.transition = 'opacity .5s';
          panel.style.opacity = '0.4';
        }
      } else if (d.type === 'job_done') {
        const tokStr = d.total_tokens ? (d.total_tokens > 1000 ? (d.total_tokens / 1000).toFixed(1) + 'k' : d.total_tokens) : '';
        toast(`${d.model} on ${d.suite}: ${d.accuracy.toFixed(1)}% (${d.correct}/${d.total}${tokStr ? ', ' + tokStr + ' tokens' : ''})`);
        // Show final summary overlay in the panel
        _stopTimer(jobId);
        const panel = document.getElementById('bm-stream-' + jobId);
        if (panel) {
          const acc = d.accuracy.toFixed(1);
          const accColor = acc >= 80 ? '#4ade80' : acc >= 50 ? '#facc15' : '#f87171';
          const overlay = document.createElement('div');
          overlay.style.cssText = 'position:absolute;top:0;left:0;right:0;bottom:0;display:flex;flex-direction:column;align-items:center;justify-content:center;background:rgba(8,10,16,.88);z-index:10;animation:panel-appear .4s ease-out;';
          overlay.innerHTML = `<div style="font-family:monospace;font-size:2.2em;font-weight:700;color:${accColor};letter-spacing:1px;">${acc}%</div><div style="font-family:monospace;font-size:.85em;color:#6c8cff;margin-top:6px;">${d.correct}/${d.total} correct</div>${tokStr ? '<div style="font-family:monospace;font-size:.8em;color:#888;margin-top:4px;">' + tokStr + ' tokens</div>' : ''}<div style="font-family:monospace;font-size:.75em;color:#556;margin-top:4px;">${d.model}</div>`;
          panel.appendChild(overlay);
        }
        // Fade out and remove this job's panel after a delay
        if (jobId) {
          setTimeout(() => _removeStreamPanel(jobId), 3500);
        }
        refreshBmQueue();
        loadSuiteResults();
      } else if (d.type === 'queue_done') {
        bmSuiteSource.close();
        bmSuiteSource = null;
        document.getElementById('bm-progress').style.display = 'none';
        document.getElementById('btn-bm-run').style.display = '';
        document.getElementById('btn-bm-stop').style.display = 'none';
        document.getElementById('btn-bm-stop').disabled = false;
        // Clear all streaming panels + timers + bars
        _timers.forEach((_, jid) => _stopTimer(jid));
        _barHistory.clear();
        document.getElementById('bm-streams-container').innerHTML = '';
        bmStreamStates.clear();
        _streamCursors.clear();
        if (d.stopped) {
          document.getElementById('bm-progress-text').textContent = 'Stopped. Partial results saved.';
          toast('Benchmark suite stopped. Partial results saved.');
        } else {
          refreshBmQueue();
          loadSuiteResults();
        }
      } else if (d.type === 'stopping') {
        document.getElementById('bm-progress-text').textContent = 'Stopping... saving partial results';
        document.getElementById('btn-bm-stop').disabled = true;
      } else if (d.type === 'done') {
        bmSuiteSource.close();
        bmSuiteSource = null;
        document.getElementById('bm-progress').style.display = 'none';
        document.getElementById('btn-bm-run').style.display = '';
        document.getElementById('btn-bm-stop').style.display = 'none';
        // Clear all streaming panels + timers + bars
        _timers.forEach((_, jid) => _stopTimer(jid));
        _barHistory.clear();
        document.getElementById('bm-streams-container').innerHTML = '';
        bmStreamStates.clear();
        _streamCursors.clear();
        loadSuiteResults();
      } else if (d.type === 'error') {
        toast('Error: ' + (d.error || 'unknown'));
        refreshBmQueue();
        // Don't close SSE — other jobs may still be running
      }
    } catch(e) {}
  };
  bmSuiteSource.onerror = () => {
    if (bmSuiteSource) { bmSuiteSource.close(); bmSuiteSource = null; }
    // Fall back to polling
    bmPollInterval = setInterval(async () => {
      try {
        const sr = await fetch('/api/suites/status');
        const st = await sr.json();
        if (!st.active && (!st.queue || !st.queue.length)) {
          clearInterval(bmPollInterval);
          bmPollInterval = null;
          document.getElementById('bm-progress').style.display = 'none';
          // Clear all streaming panels + timers + bars
          _timers.forEach((_, jid) => _stopTimer(jid));
          _barHistory.clear();
          document.getElementById('bm-streams-container').innerHTML = '';
          bmStreamStates.clear();
          _streamCursors.clear();
          document.getElementById('btn-bm-run').style.display = '';
          document.getElementById('btn-bm-stop').style.display = 'none';
          loadSuiteResults();
          refreshBmQueue();
          return;
        }
        // Show active jobs' progress
        if (st.active_jobs && st.active_jobs.length > 0) {
          for (const job of st.active_jobs) {
            const pct = Math.round(job.current_question / job.total_questions * 100);
            // Update progress bar with first job
            if (job === st.active_jobs[0]) {
              document.getElementById('bm-progress-bar').style.width = pct + '%';
              document.getElementById('bm-progress-text').textContent = `${job.current_question}/${job.total_questions} — ${job.model} on ${job.suite_key}`;
            }
            // Ensure panel exists for this job
            if (!document.getElementById('bm-stream-' + job.job_id)) {
              _createStreamPanel(job.job_id, job.model, job.suite_key);
            }
          }
        }
        refreshBmQueue();
      } catch(e) {}
    }, 2000);
  };
}

function applySampleSizeToAll() {
  const n = parseInt(document.getElementById('bm-sample-all').value);
  if (isNaN(n) || n < 0) return;
  for (const key of bmSelectedSuites) {
    const input = document.getElementById('bm-suite-ss-' + key);
    if (input) input.value = n;
  }
}

async function runBenchmarkSuite() {
  if (!bmSelectedSuites.length) return alert('Select at least one suite');
  if (!bmSelectedModels.length) return alert('Add at least one model');
  const host = document.getElementById('ollama-host').value;

  // Connect SSE BEFORE queuing so we don't miss job_start events
  connectBmSSE();

  // Queue all combinations: each suite × each model (with per-suite sample size)
  let totalQueued = 0;
  for (const suiteKey of bmSelectedSuites) {
    const ssInput = document.getElementById('bm-suite-ss-' + suiteKey);
    const sampleSize = ssInput ? (parseInt(ssInput.value) || 0) : 0;
    try {
      const res = await fetch('/api/suites/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({host, models: bmSelectedModels, suite_key: suiteKey, sample_size: sampleSize})
      });
      const data = await res.json();
      if (data.error) { toast(`Error for ${suiteKey}: ${data.error}`); continue; }
      totalQueued += (data.queued || []).length;
      if (data.skipped && data.skipped.length) { toast(`Skipped ${data.skipped.length} duplicate(s) for ${suiteKey} — already queued or running`, 4000); }
    } catch(e) { toast(`Failed to queue ${suiteKey}: ${e.message}`); }
  }

  if (!totalQueued) {
    // All were duplicates — nothing new to queue, but don't silently return if toasts were shown
    return;
  }

  toast(`Queued ${totalQueued} job${totalQueued > 1 ? 's' : ''}`);
  await refreshBmQueue();
}

async function loadSuiteResults() {
  try {
    const res = await fetch('/api/suites/results');
    const results = await res.json();
    const el = document.getElementById('bm-results');
    if (!results.length) { el.innerHTML = '<p style="color:var(--muted);font-size:.9em;">No benchmark suite results yet.</p>'; return; }

    // Filter bar
    const suiteKeys = [...new Set(results.map(r => r.suite_key))];
    let filterHtml = '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;">';
    filterHtml += '<span style="font-size:.85em;color:var(--muted);">Filter:</span>';
    filterHtml += '<select id="results-filter-suite" style="font-size:.85em;padding:2px 6px;border-radius:4px;border:1px solid var(--border);background:var(--card);color:var(--fg);"><option value="">All suites</option>';
    for (const k of suiteKeys) { filterHtml += `<option value="${k}">${k}</option>`; }
    filterHtml += '</select>';
    filterHtml += '<button class="btn btn-secondary" style="padding:3px 10px;font-size:.78em;" onclick="deleteAllSuiteResults()">Delete All Shown</button>';
    filterHtml += '</div>';

    let html = `<table style="width:100%;border-collapse:collapse;font-size:.88em;">
      <thead><tr style="border-bottom:2px solid var(--border);text-align:left;">
        <th style="padding:6px 8px;">Date</th>
        <th style="padding:6px 8px;">Suite</th>
        <th style="padding:6px 8px;">Model</th>
        <th style="padding:6px 8px;">Computer</th>
        <th style="padding:6px 8px;">Accuracy</th>
        <th style="padding:6px 8px;">Correct / Total</th>
        <th style="padding:6px 8px;">Avg tok/s</th>
        <th style="padding:6px 8px;">Wall Time</th>
        <th style="padding:6px 8px;">Power</th>
        <th style="padding:6px 8px;">VRAM</th>
        <th style="padding:6px 8px;">GPUs</th>
        <th style="padding:6px 8px;">Energy</th>
        <th style="padding:6px 8px;">Tokens</th>
        <th style="padding:6px 8px;"></th>
      </tr></thead><tbody>`;

    for (const r of results) {
      const date = r.timestamp ? r.timestamp.slice(0, 16).replace('T', ' ') : '';
      const accColor = r.accuracy >= 60 ? '#4caf50' : r.accuracy >= 30 ? '#ff9800' : '#f44336';
      html += `<tr data-suite="${r.suite_key}" style="border-bottom:1px solid var(--border);">
        <td style="padding:6px 8px;">${date}</td>
        <td style="padding:6px 8px;">${r.suite_name}</td>
        <td style="padding:6px 8px;">${r.model}</td>
        <td style="padding:6px 8px;font-family:monospace;font-size:.82em;">${r.computer_id || '—'}</td>
        <td style="padding:6px 8px;font-weight:bold;color:${accColor};">${r.accuracy.toFixed(1)}%</td>
        <td style="padding:6px 8px;">${r.correct} / ${r.total_questions}</td>
        <td style="padding:6px 8px;">${r.avg_tps ? r.avg_tps.toFixed(1) : '—'}</td>
        <td style="padding:6px 8px;">${r.total_wall_time ? r.total_wall_time.toFixed(1) + 's' : '—'}</td>
        <td style="padding:6px 8px;">${r.avg_gpu_power ? r.avg_gpu_power.toFixed(0) + 'W' : '—'}</td>
        <td style="padding:6px 8px;">${r.gpu_mem_peak ? (r.gpu_mem_peak > 1024 ? (r.gpu_mem_peak / 1024).toFixed(1) + 'GB' : r.gpu_mem_peak + 'MB') : '—'}</td>
        <td style="padding:6px 8px;">${r.gpu_active ? r.gpu_active.join(',') : '—'}</td>
        <td style="padding:6px 8px;">${r.gpu_energy_wh ? r.gpu_energy_wh.toFixed(1) + 'Wh' : '—'}</td>
        <td style="padding:6px 8px;">${r.total_tokens ? (r.total_tokens > 1000 ? (r.total_tokens / 1000).toFixed(1) + 'k' : r.total_tokens) : '—'}</td>
        <td style="padding:6px 8px;"><button class="btn btn-secondary" style="padding:2px 8px;font-size:.75em;color:#f44336;" onclick="deleteSuiteResult(${r.id}, this)">✕</button></td>
      </tr>`;
    }
    html += '</tbody></table>';
    el.innerHTML = filterHtml + html;

    // Apply filter on change
    document.getElementById('results-filter-suite').addEventListener('change', function() {
      const val = this.value;
      document.querySelectorAll('#bm-results tbody tr').forEach(tr => {
        tr.style.display = (!val || tr.dataset.suite === val) ? '' : 'none';
      });
    });
  } catch(e) { document.getElementById('bm-results').innerHTML = '<p style="color:var(--muted);">Failed to load results</p>'; }
}

async function deleteSuiteResult(id, btn) {
  if (!confirm('Delete this result?')) return;
  const res = await fetch('/api/suites/results/' + id, {method: 'DELETE'});
  const data = await res.json();
  if (data.deleted) {
    toast('Result deleted');
    loadSuiteResults();
  } else {
    toast(data.error || 'Delete failed');
  }
}

async function deleteAllSuiteResults() {
  const suiteKey = document.getElementById('results-filter-suite')?.value || '';
  const msg = suiteKey ? `Delete ALL results for "${suiteKey}"?` : 'Delete ALL suite results?';
  if (!confirm(msg)) return;
  const url = suiteKey ? `/api/suites/results?suite_key=${encodeURIComponent(suiteKey)}` : '/api/suites/results';
  const res = await fetch(url, {method: 'DELETE'});
  const data = await res.json();
  toast(`Deleted ${data.deleted_count} results`);
  loadSuiteResults();
}

// Populate judge model selector when connecting
function populateJudgeModels(models) {
  const sel = document.getElementById('judge-model-select');
  sel.innerHTML = '<option value="">-- Select judge model (larger = better) --</option>';
  // Sort by size descending so larger models appear first
  const sorted = [...models].sort((a,b) => (b.size || 0) - (a.size || 0));
  sorted.forEach(m => {
    const sizeStr = m.size ? (m.size >= 1e9 ? Math.round(m.size/1e9) + 'GB' : Math.round(m.size/1e6) + 'MB') : 'cloud';
    sel.innerHTML += `<option value="${m.name}">${m.name} (${sizeStr})</option>`;
  });
}

function runJudge() {
  const judgeModel = document.getElementById('judge-model-select').value;
  if (!judgeModel || !currentRunId) {
    toast('Select a judge model and load results first');
    return;
  }
  document.getElementById('judge-progress').classList.remove('hidden');
  document.getElementById('btn-judge').disabled = true;

  fetch('/api/judge/' + currentRunId, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({host: ollamaHost, judge_model: judgeModel})
  }).then(r => r.json()).then(data => {
    if (data.error) { toast(data.error); return; }
    toast('Judge started with ' + judgeModel);
  });

  // Listen to judge SSE
  const evtSource = new EventSource('/api/judge-stream');
  evtSource.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'judge_progress') {
      const pct = (msg.progress / msg.total * 100).toFixed(0);
      document.getElementById('judge-progress-fill').style.width = pct + '%';
      document.getElementById('judge-progress-text').textContent = `Judging ${msg.model} / ${msg.test} (${msg.progress}/${msg.total})`;
    } else if (msg.type === 'judge_done') {
      const scoreTag = msg.judge_score ? ` — Score: ${msg.judge_score}/10` : '';
      const mathTag = msg.math_correct !== null ? (msg.math_correct ? ' ✓' : ' ✗ (got ' + msg.math_answer + ')') : '';
      document.getElementById('judge-results').innerHTML += `<div>${msg.model} / ${msg.test}${scoreTag}${mathTag}</div>`;
    } else if (msg.type === 'judge_complete') {
      evtSource.close();
      document.getElementById('judge-progress-text').textContent = 'Judge complete!';
      document.getElementById('btn-judge').disabled = false;
      toast('Judge complete! Run #' + msg.run_id);
      loadResults(currentRunId); // Refresh results to show judge scores
    }
  };
}

// ── Init ──
loadTests();
loadBenchmarks();
</script>
</body>
</html>
"""

