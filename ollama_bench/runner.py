#!/usr/bin/env python3
"""Benchmark suite runner — VRAM-aware parallel scheduling, SSE events."""

import json
import time
import threading
import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import requests as req_lib

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ollama_bench.config import (
    HIGH_NUM_PREDICT, DEFAULT_TIMEOUT, TESTPACKS_DIR, DEFAULT_OLLAMA_HOST,
)
from ollama_bench.ollama_api import stream_generate_chunks
from ollama_bench.gpu_sampler import GPUSampler
from ollama_bench.tests import stream_chat_chunks, format_bfcl_messages, format_bfcl_prompt
from ollama_bench.db import get_db, init_db, compute_computer_id, db_lock, detect_system_info
from ollama_bench.suites import (
    BENCHMARK_SUITES,
    load_benchmark_dataset, format_mmlu_question, extract_mcq_answer,
    format_ifeval_prompt, check_ifeval_constraints,
    check_bfcl_answer, check_web_search_answer, _sample_bfcl,
)
from ollama_bench.state import bench_state, judge_state

logger = logging.getLogger(__name__)

router = APIRouter()

# ═══════════════════════════════════════════════

class SuiteJob:
    """Per-job state for a single suite run (one model on one suite)."""
    def __init__(self, host, model, suite_key, sample_size, job_id):
        self.job_id = job_id
        self.host = host
        self.model = model
        self.suite_key = suite_key
        self.sample_size = sample_size
        self.current_question = 0
        self.total_questions = 0
        self.active = True
        self.run_id = None
        self.results = []
        self.estimated_vram = 0  # VRAM estimate at dispatch time, corrected after model loads
        self.skip_requested = False  # Set by skip API to abort current question

class SuiteState:
    """Track state of suite runner — queue, active jobs, event streams."""
    def __init__(self):
        self.event_queues = []
        # Queue: list of dicts {host, model, suite_key, sample_size}
        self.suite_queue = []
        # Active parallel jobs
        self.active_jobs = []  # list of SuiteJob
        self.max_parallel = 1  # how many jobs can run concurrently (1 for single-host — GPU time-slicing corrupts results)
        self._processor_running = False
        self._next_job_id = 0
        self.stop_requested = False

    @property
    def active(self):
        return len(self.active_jobs) > 0 or self._processor_running

    def next_job_id(self):
        self._next_job_id += 1
        return self._next_job_id

suite_state = SuiteState()

# Thread-safe SSE broadcast: asyncio.Queue.put_nowait() is NOT safe to call
# from synchronous threads (SuiteJob._run_question runs in threading.Thread).
# Use call_soon_threadsafe to schedule the put from the event loop thread.
_sse_loop = None  # set once at startup from @router.on_event("startup")

def _sse_broadcast(data_dict):
    """Thread-safe broadcast to all SSE subscriber queues."""
    msg = json.dumps(data_dict)
    for q_obj in list(suite_state.event_queues):
        try:
            if _sse_loop is not None and _sse_loop.is_running():
                _sse_loop.call_soon_threadsafe(q_obj.put_nowait, msg)
            else:
                # Fallback: direct put if we're already in the event loop thread
                q_obj.put_nowait(msg)
        except:
            pass

# ── Model Profile State ──
class ProfileState:
    """Track model profiling state."""
    def __init__(self):
        self.queue = []           # list of {model, host}
        self.active = False
        self.current_model = None
        self.progress = 0         # 0=idle, 1=loading, 2=measuring, 3=done
        self._lock = threading.Lock()

profile_state = ProfileState()


def _profile_model_cold(host, model):
    """Load a model cold, measure VRAM/RAM/GPU% from /api/ps, then unload it.
    
    Steps:
    1. Send a short prompt to trigger model load
    2. Wait for model to appear in /api/ps
    3. Read SIZE (total memory), context length, GPU%
    4. Also get parameter_size and quantization from /api/show
    5. Unload model (keep_alive:0)
    6. Save profile to DB
    """
    import time
    
    # 1. Get model metadata first
    param_size = ""
    quant_level = ""
    try:
        resp = req_lib.post(f"{host}/api/show", json={"name": model}, timeout=15)
        if resp.status_code == 200:
            details = resp.json().get("details", {})
            param_size = details.get("parameter_size", "")
            quant_level = details.get("quantization_level", "")
    except Exception:
        pass
    
    # 2. Load model with a minimal prompt
    profile_state.progress = 1  # loading
    profile_state.current_model = model
    
    try:
        # Use /api/generate with keep_alive=5m to load model
        for chunk in stream_generate_chunks(host, model, "Hi", num_predict=1, temperature=0.0):
            if chunk.get("type") == "done":
                break
    except Exception:
        pass
    
    # 3. Wait briefly for model to appear in /api/ps
    profile_state.progress = 2  # measuring
    time.sleep(2)
    
    vram_mib = 0
    ram_mib = 0
    gpu_pct = 0.0
    context_length = 0
    
    try:
        ps_resp = req_lib.get(f"{host}/api/ps", timeout=10)
        if ps_resp.status_code == 200:
            ps_data = ps_resp.json()
            for m in ps_data.get("models", []):
                loaded_name = m.get("name", "")
                if loaded_name == model or loaded_name.split(":")[0] == model.split(":")[0]:
                    size_bytes = m.get("size", 0)
                    vram_mib = int(size_bytes / (1024 * 1024))
                    context_length = m.get("context_length", 0)
                    # GPU% comes from processor field like "100% GPU"
                    proc = m.get("processor", "")
                    if "GPU" in proc:
                        gpu_pct = float(proc.replace("%", "").replace(" GPU", "").strip())
                    break
    except Exception:
        pass
    
    # 4. Unload model
    try:
        req_lib.post(f"{host}/api/generate", json={"model": model, "prompt": "", "keep_alive": 0}, timeout=10)
    except Exception:
        pass
    
    # 5. Save to DB
    try:
        sys_info = detect_system_info()
        computer_id = compute_computer_id(sys_info)
        with db_lock:
            db = get_db()
            db.execute("""
                INSERT INTO model_profiles (computer_id, model, vram_mib, ram_mib, gpu_pct, context_length, parameter_size, quantization, profiled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(computer_id, model) DO UPDATE SET
                    vram_mib=excluded.vram_mib, ram_mib=excluded.ram_mib, gpu_pct=excluded.gpu_pct,
                    context_length=excluded.context_length, parameter_size=excluded.parameter_size,
                    quantization=excluded.quantization, profiled_at=excluded.profiled_at
            """, (computer_id, model, vram_mib, ram_mib, gpu_pct, context_length, param_size, quant_level,
                  datetime.now(timezone.utc).isoformat()))
            db.commit()
            db.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
    
    profile_state.progress = 3  # done
    return {
        "model": model,
        "vram_mib": vram_mib,
        "ram_mib": ram_mib,
        "gpu_pct": gpu_pct,
        "context_length": context_length,
        "parameter_size": param_size,
        "quantization": quant_level,
    }


@router.get("/api/profile/results")
async def api_profile_results():
    """Return all stored model profiles."""
    with db_lock:
        db = get_db()
        rows = db.execute("SELECT model, vram_mib, ram_mib, gpu_pct, context_length, parameter_size, quantization, profiled_at FROM model_profiles ORDER BY vram_mib DESC").fetchall()
        db.close()
    results = []
    for r in rows:
        results.append({
            "model": r[0], "vram_mib": r[1], "ram_mib": r[2], "gpu_pct": r[3],
            "context_length": r[4], "parameter_size": r[5], "quantization": r[6], "profiled_at": r[7],
        })
    return {"profiles": results}


@router.post("/api/profile/run")
async def api_profile_run(request: Request):
    """Queue models for profiling (cold load + measure + unload, one at a time)."""
    data = await request.json()
    models = data.get("models", [])
    host = data.get("host", "http://localhost:11434")
    
    if not models:
        return {"error": "No models specified"}, 400
    
    with profile_state._lock:
        for m in models:
            profile_state.queue.append({"model": m, "host": host})
        count = len(profile_state.queue)
    
    # Start processor if not running
    if not profile_state.active:
        threading.Thread(target=_run_profile_queue, daemon=True).start()
    
    return {"queued": len(models), "queue_length": count}


def _run_profile_queue():
    """Process profile queue — one model at a time, cold load + measure."""
    profile_state.active = True
    try:
        while True:
            with profile_state._lock:
                if not profile_state.queue:
                    break
                item = profile_state.queue.pop(0)
            
            result = _profile_model_cold(item["host"], item["model"])
            
            # Notify SSE
            _sse_broadcast({"type": "profile_done", **result})
            
            # Brief pause between models
            import time; time.sleep(2)
    finally:
        profile_state.active = False
        profile_state.current_model = None
        profile_state.progress = 0
        _sse_broadcast({"type": "profile_queue_done"})


@router.get("/api/profile/status")
async def api_profile_status():
    """Return current profiling status."""
    return {
        "active": profile_state.active,
        "current_model": profile_state.current_model,
        "progress": profile_state.progress,
        "queue_length": len(profile_state.queue),
        "queue": profile_state.queue,
    }


@router.delete("/api/profile/queue")
async def api_profile_clear():
    """Clear the profile queue (does not stop an in-progress profile)."""
    with profile_state._lock:
        cleared = len(profile_state.queue)
        profile_state.queue.clear()
    return {"cleared": cleared}


@router.get("/api/suites")
async def api_list_suites():
    """List available benchmark suites with cached status."""
    result = {}
    for key, suite in BENCHMARK_SUITES.items():
        cached = (TESTPACKS_DIR / key / "data.parquet").exists()
        result[key] = {
            "key": key,
            "name": suite["name"],
            "description": suite["description"],
            "answer_type": suite["answer_type"],
            "sample_size": suite.get("sample_size", 0),
            "cached": cached,
            "categories": suite.get("categories", []),
        }
    return JSONResponse(result)


@router.post("/api/suites/download/{suite_key}")
async def api_download_suite(suite_key: str):
    """Download/cache a benchmark suite dataset from HuggingFace."""
    if suite_key not in BENCHMARK_SUITES:
        return JSONResponse({"error": f"Unknown suite: {suite_key}"}, status_code=404)
    try:
        data = load_benchmark_dataset(suite_key)
        return JSONResponse({"status": "ok", "rows": len(data)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/suites/run")
async def api_run_suite(request: Request):
    """Queue a benchmark suite run. Body: {host, model(s), suite_key, sample_size?}.
    Accepts 'model' (string) or 'models' (list). Queues jobs if suite is already running."""
    body = await request.json()
    ollama_host = body.get("host", DEFAULT_OLLAMA_HOST)
    # Accept model (string) or models (list)
    models = body.get("models", [])
    if not models:
        m = body.get("model", "")
        if m:
            models = [m]
    suite_key = body.get("suite_key", "")
    sample_size = body.get("sample_size", 0)

    if not models or not suite_key:
        return JSONResponse({"error": "model(s) and suite_key required"}, status_code=400)
    if suite_key not in BENCHMARK_SUITES:
        return JSONResponse({"error": f"Unknown suite: {suite_key}"}, status_code=400)

    # Load dataset (cache it)
    try:
        load_benchmark_dataset(suite_key)
    except Exception as e:
        return JSONResponse({"error": f"Failed to load dataset: {e}"}, status_code=500)

    # Enqueue jobs (one per model), skipping duplicates
    # Check both queue and active jobs for same model+suite combo
    existing_keys = set()
    for j in suite_state.suite_queue:
        existing_keys.add((j["model"], j["suite_key"]))
    for j in suite_state.active_jobs:
        existing_keys.add((j.model, j.suite_key))

    queued = []
    skipped = []
    for model in models:
        key = (model, suite_key)
        if key in existing_keys:
            skipped.append({"model": model, "suite_key": suite_key})
            continue
        job = {"host": ollama_host, "model": model, "suite_key": suite_key, "sample_size": sample_size}
        suite_state.suite_queue.append(job)
        existing_keys.add(key)
        queued.append({"model": model, "suite_key": suite_key})

    # Start processor if not already running
    suite_state.stop_requested = False  # reset any stale stop flag
    if not suite_state._processor_running:
        _start_suite_processor()

    result = {"queued": queued, "queue_length": len(suite_state.suite_queue)}
    if skipped:
        result["skipped"] = skipped
        result["message"] = f"Skipped {len(skipped)} duplicate(s) — already queued or running"
    return JSONResponse(result)


def _get_model_vram_mb(host, model):
    """Estimate VRAM needed for a model.
    
    Strategy (in order of accuracy):
    1. Stored profile (cold-measured VRAM) — best estimate.
    2. If model is currently loaded (/api/ps), use actual SIZE — ground truth.
    3. Estimate from model file size × 1.5 (weights + KV cache overhead).
    4. Estimate from parameter_size × quant × MoE ratio × 1.5.
    5. Fallback: full GPU VRAM (forces solo run).
    """
    # 0. Check stored profile first — previous cold measurement (best estimate)
    try:
        sys_info = detect_system_info()
        computer_id = compute_computer_id(sys_info)
        # Use the same DB as the rest of the server (respects --db override)
        from ollama_bench.db import _cfg
        db_path = _cfg.get("db_path")
        db = sqlite3.connect(db_path)
        row = db.execute(
            "SELECT vram_mib FROM model_profiles WHERE computer_id=? AND model=?",
            (computer_id, model)
        ).fetchone()
        db.close()
        if row and row[0] and row[0] > 0:
            return row[0]
        else:
            print(f"[VRAM] No profile for {model} (computer_id={computer_id}), falling through to estimation")
    except Exception as e:
        print(f"[VRAM] Profile lookup failed for {model}: {e}")

    try:
        # Check if model is currently loaded — use actual VRAM usage (ground truth)
        ps_resp = req_lib.get(f"{host}/api/ps", timeout=5)
        if ps_resp.status_code == 200:
            ps_data = ps_resp.json()
            for m in ps_data.get("models", []):
                loaded_name = m.get("name", "")
                # Match model name without tag (e.g. "gemma4:26b" matches "gemma4:26b")
                if loaded_name == model or loaded_name.split(":")[0] == model.split(":")[0]:
                    actual_size = m.get("size", 0)
                    if actual_size > 0:
                        return int(actual_size / (1024 * 1024))  # bytes → MiB
    except Exception:
        pass

    try:
        # Get model details + file size
        resp = req_lib.post(f"{host}/api/show", json={"name": model}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            details = data.get("details", {})
            model_info = data.get("model_info", {})
            modelfile = data.get("modelfile", "")
            
            # Strategy 3: Use actual model file size from the FROM path in modelfile
            model_file_size_mb = None
            from_match = re.search(r'^FROM\s+(\S+)', modelfile, re.MULTILINE)
            if from_match:
                blob_path = from_match.group(1)
                if os.path.exists(blob_path):
                    model_file_size_mb = int(os.path.getsize(blob_path) / (1024 * 1024))
            
            # For MoE models, adjust: only expert_used_count/total experts are active
            moe_ratio = 1.0  # default: dense (all params active)
            arch = model_info.get("general.architecture", "")
            expert_used = model_info.get(f"{arch}.expert_used_count")
            expert_total = model_info.get(f"{arch}.expert_count")
            if expert_used and expert_total and expert_total > 0:
                # The model file includes all experts, but only expert_used are active per token
                # VRAM still holds all weights, but compute is sparse — and Ollama offloads
                # expert weights to CPU/disk for large models. Use a blended ratio.
                # For scheduling: the file must fit in total system memory, but GPU only needs
                # partial weights. Use sqrt(ratio) as compromise between file-size and active-size.
                raw_ratio = expert_used / expert_total
                moe_ratio = max(raw_ratio, 0.4)  # at least 40% — GPU still holds most weights
            
            # Context overhead multiplier
            # Short-context benchmarks (BFCL: 4K, MMLU: short prompts) need minimal KV cache.
            # Long-context models default to large context, but benchmarks override it.
            # Use 1.5× instead of 4× — realistic for benchmark workloads.
            CONTEXT_OVERHEAD = 1.5
            
            if model_file_size_mb:
                # Best estimate: actual file size × MoE ratio × context overhead
                return int(model_file_size_mb * moe_ratio * CONTEXT_OVERHEAD)
            
            # Strategy 4: Estimate from parameter count and quantization
            param_str = details.get("parameter_size", "")  # e.g. "25.8B", "123.6B"
            quant = details.get("quantization_level", "")    # e.g. "Q4_K_M", "F16"

            if param_str:
                # Parse parameter count (e.g. "25.8B" → 25.8e9)
                param_str = param_str.upper().strip()
                if param_str.endswith("B"):
                    num_params = float(param_str[:-1]) * 1e9
                elif param_str.endswith("M"):
                    num_params = float(param_str[:-1]) * 1e6
                else:
                    num_params = float(param_str)

                # Estimate bytes per parameter based on quantization
                bpb = {
                    "Q3": 0.5, "Q4": 0.56, "Q4_0": 0.5, "Q4_K_M": 0.56, "Q4_K_S": 0.5,
                    "Q5": 0.68, "Q5_K_M": 0.68, "Q5_K_S": 0.65,
                    "Q6": 0.75, "Q6_K": 0.75,
                    "Q8": 1.0, "Q8_0": 1.0,
                    "F16": 2.0, "F32": 4.0,
                }.get(quant.split("_")[0] if "_" in quant else quant.split("-")[0], 0.56)

                if bpb == 0.56 and "Q4" in quant:
                    bpb = 0.56  # Q4_K_M fallback

                model_size_mb = int(num_params * bpb / (1024 * 1024))
                
                # Apply MoE ratio and context overhead
                return int(model_size_mb * moe_ratio * CONTEXT_OVERHEAD)
    except Exception:
        pass

    # Fallback: unknown size → assume full GPU VRAM (forces solo run)
    total = _get_total_vram_mb()
    print(f"[VRAM] FALLBACK for {model}: {total} MiB — all estimation strategies failed")
    return total


def _get_total_vram_mb():
    """Get total GPU VRAM in MiB from system info."""
    try:
        sys_info = detect_system_info()
        return sum(g.get("memory_mib", 0) for g in sys_info.get("gpus", []))
    except Exception:
        pass
    return 98304  # fallback: 4×24GB


def _start_suite_processor():
    """Start the VRAM-aware parallel suite processor.
    
    Dispatches up to max_parallel jobs concurrently, respecting VRAM limits.
    Each job runs in its own thread. When a job finishes, its VRAM is freed
    and the next queued job is dispatched if it fits.
    """
    SUITE_QUESTION_TIMEOUT = 120  # seconds per question
    VRAM_SAFETY_FACTOR = 0.90     # only use 90% of total VRAM for safety

    if suite_state._processor_running:
        return
    suite_state._processor_running = True

    total_vram_mb = _get_total_vram_mb()
    vram_budget_mb = int(total_vram_mb * VRAM_SAFETY_FACTOR)  # 90% of total
    vram_lock = threading.Lock()   # protects vram_in_use
    queue_lock = threading.Lock()  # protects suite_queue (processor + API handlers)
    vram_in_use = {"mb": 0}
    job_slots = threading.Semaphore(suite_state.max_parallel)
    completion_event = threading.Event()  # signaled when a job finishes
    vram_measured_event = threading.Event()  # signaled each time a job measures actual VRAM
    vram_measure_count = {"n": 0}  # how many dispatched jobs have been measured

    # Account for models already loaded in Ollama (not dispatched by us)
    try:
        ps_resp = req_lib.get(f"{suite_state.suite_queue[0]['host'] if suite_state.suite_queue else 'http://localhost:11434'}/api/ps", timeout=5)
        if ps_resp.status_code == 200:
            ps_data = ps_resp.json()
            queued_models = set(j['model'] for j in suite_state.suite_queue)
            for m in ps_data.get("models", []):
                # Don't count models that are in our queue — the job will take over that VRAM slot
                if m.get("name", "") in queued_models:
                    continue
                already_loaded_mb = int(m.get("size", 0) / (1024 * 1024))
                if already_loaded_mb > 0:
                    vram_in_use["mb"] += already_loaded_mb
    except Exception:
        pass

    def _run_single_suite(job):
        """Run one suite for one model. Updates job state. Returns True on success."""
        ollama_host = job.host
        model = job.model
        suite_key = job.suite_key
        sample_size = job.sample_size

        suite = BENCHMARK_SUITES[suite_key]
        results = []
        agentic_done = False

        # Load dataset
        try:
            all_data = load_benchmark_dataset(suite_key)
        except Exception:
            job.active = False
            return False

        # Notify: job started (must be before any suite-type branching so
        # the frontend sees "Starting..." before questions begin processing)
        _sse_broadcast({
            "type": "job_start",
            "job_id": job.job_id,
            "model": model,
            "suite": suite_key,
            "total_questions": 0  # updated below after sampling
        })

        # Sample questions
        if suite_key == "mmlu_pro":
            questions = _sample_mmlu(all_data, sample_size or BENCHMARK_SUITES["mmlu_pro"]["sample_size"])
        elif suite_key == "ifeval":
            if sample_size and sample_size < len(all_data):
                import random
                random.seed(42)
                questions = random.sample(all_data, sample_size)
            else:
                questions = all_data
        elif suite_key in ("bfcl_v3", "bfcl_v4"):
            bfcl_sample = sample_size or BENCHMARK_SUITES[suite_key]["sample_size"] or 3
            questions = _sample_bfcl(all_data, per_category=bfcl_sample, suite_key=suite_key)
        elif suite_key == "bfcl_v4_web_search":
            from ollama_bench.agentic import run_agentic_eval, create_search_backend, get_tools_for_suite
            agentic_done = True
            questions = all_data
            if sample_size and sample_size < len(questions):
                import random
                random.seed(42)
                questions = random.sample(questions, sample_size)
            job.total_questions = len(questions)
            for i, row in enumerate(questions):
                if suite_state.stop_requested or getattr(job, 'stop_requested', False):
                    break
                job.current_question = i + 1
                job.skip_requested = False
                # Signal VRAM measurement after model loads (first question only)
                if i == 0:
                    actual_vram = _get_model_vram_mb(ollama_host, model)
                    if actual_vram > 0:
                        estimated = getattr(job, 'estimated_vram', 0)
                        if estimated > 0:
                            diff = actual_vram - estimated
                            if abs(diff) > 100:
                                with vram_lock:
                                    vram_in_use["mb"] += diff
                                job.estimated_vram = actual_vram
                        vram_measured_event.set()
                        vram_measure_count["n"] += 1
                        completion_event.set()  # wake scheduler to dispatch next job
                _sse_broadcast({"type": "question_start", "job_id": job.job_id, "question": i+1, "total": len(questions), "model": model, "suite": suite_key})
                # Format prompt from web_search dataset
                prompt_data = row.get("prompt", [])
                if hasattr(prompt_data, 'tolist'):
                    prompt_data = prompt_data.tolist()
                messages = []
                if isinstance(prompt_data, list) and len(prompt_data) > 0:
                    first = prompt_data[0]
                    if isinstance(first, list):
                        for msg in first:
                            if isinstance(msg, dict):
                                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
                    elif isinstance(first, dict):
                        for msg in prompt_data:
                            if isinstance(msg, dict):
                                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
                if not messages:
                    messages = [{"role": "user", "content": str(row.get("prompt", ""))}]
                # Get ground truth answers
                expected = row.get("answer", [])
                if hasattr(expected, 'tolist'):
                    expected = expected.tolist()
                
                def _sse_cb(evt):
                    evt["job_id"] = job.job_id
                    evt["model"] = model
                    evt["suite"] = suite_key
                    _sse_broadcast(evt)
                
                def _should_stop():
                    return job.skip_requested or getattr(job, 'stop_requested', False) or suite_state.stop_requested
                
                search_backend = create_search_backend(mode=bench_state.search_mode if hasattr(bench_state, 'search_mode') else "searxng")
                tool_defs = get_tools_for_suite(suite_key)
                
                result = run_agentic_eval(host=ollama_host, model=model, messages=messages,
                                           tool_definitions=tool_defs, tool_executor=search_backend,
                                           max_turns=10, num_predict=suite.get("num_predict", 4096),
                                           num_ctx=suite.get("num_ctx", 8192),
                                           temperature=suite.get("temperature", 0.0),
                                           stop_check=_should_stop,
                                           sse_callback=_sse_cb)
                
                is_correct, matched = check_web_search_answer(result.get("answer", ""), expected if isinstance(expected, list) else [expected])
                
                results.append({
                    "question_idx": i, "correct": is_correct,
                    "model_answer": result.get("answer", "")[:200],
                    "expected": json.dumps(expected, ensure_ascii=False)[:500] if expected else "",
                    "tps": result.get("tps", 0), "wall_time": result.get("wall_time", 0),
                    "tool_calls_count": len(result.get("tool_calls", [])),
                    "turns": result.get("turns", 0),
                })
                _sse_broadcast({"type": "answer", "job_id": job.job_id, "question": i+1, "model": model, "suite": suite_key,
                                "answer": result.get("answer", "")[:50], "expected": json.dumps(expected, ensure_ascii=False)[:200] if expected else "",
                                "correct": is_correct, "tps": result.get("tps", 0), "wall_time": result.get("wall_time", 0),
                                "turns": result.get("turns", 0)})
        elif suite_key.startswith("bfcl_v4_memory"):
            from ollama_bench.agentic import run_agentic_eval, create_memory_backend, get_tools_for_suite
            agentic_done = True
            memory_type = suite.get("memory_type", "kv")
            questions = all_data
            if sample_size and sample_size < len(questions):
                import random
                random.seed(42)
                questions = random.sample(questions, sample_size)
            job.total_questions = len(questions)
            for i, row in enumerate(questions):
                if suite_state.stop_requested or getattr(job, 'stop_requested', False):
                    break
                job.current_question = i + 1
                job.skip_requested = False
                
                # Signal VRAM measurement after model loads (first question only)
                # so the scheduler can dispatch a second job in parallel
                if i == 0:
                    actual_vram = _get_model_vram_mb(ollama_host, model)
                    if actual_vram > 0:
                        estimated = getattr(job, 'estimated_vram', 0)
                        if estimated > 0:
                            diff = actual_vram - estimated
                            if abs(diff) > 100:
                                with vram_lock:
                                    vram_in_use["mb"] += diff
                                job.estimated_vram = actual_vram
                        vram_measured_event.set()
                        vram_measure_count["n"] += 1
                        completion_event.set()  # wake scheduler to dispatch next job
                
                print(f"[DEBUG] memory suite {suite_key}: starting question {i+1}/{len(questions)} for model {model}", flush=True)
                
                # Create fresh memory backend for this question
                backend = create_memory_backend(memory_type, {})
                
                # RecSum format: pre-populate memory with initial_config
                initial_config = row.get("initial_config", {})
                if hasattr(initial_config, 'tolist'):
                    initial_config = initial_config.tolist()
                if isinstance(initial_config, str):
                    try:
                        import json as _json
                        initial_config = _json.loads(initial_config)
                    except Exception:
                        initial_config = {}
                if isinstance(initial_config, dict) and initial_config:
                    backend.load_scenario(initial_config)
                
                tool_defs = get_tools_for_suite(suite_key)
                
                # Phase 1: Run prerequisite conversations to populate memory
                # (KV and Vector suites use multi-turn prereqs; RecSum uses initial_config only)
                prerequisites = row.get("prerequisites", [])
                # Convert numpy arrays from parquet cache — nested ndarrays need .tolist()
                if hasattr(prerequisites, 'tolist'):
                    prerequisites = prerequisites.tolist()
                if isinstance(prerequisites, str):
                    try:
                        prerequisites = json.loads(prerequisites)
                    except:
                        prerequisites = []
                
                prereq_tool_calls_total = 0
                print(f"[DEBUG] question {i+1}: {len(prerequisites)} prerequisites to process", flush=True)
                for pi, prereq in enumerate(prerequisites):
                    if not isinstance(prereq, dict):
                        continue
                    prereq_question = prereq.get("question", [])
                    # question structure: list of turns, each turn is list of messages
                    # e.g. [[{role: "user", content: "..."}], [{role: "user", content: "..."}]]
                    if hasattr(prereq_question, 'tolist'):
                        prereq_question = prereq_question.tolist()
                    if not prereq_question or not isinstance(prereq_question, list):
                        continue
                    
                    for turn_msgs in prereq_question:
                        if hasattr(turn_msgs, 'tolist'):
                            turn_msgs = turn_msgs.tolist()
                        if not isinstance(turn_msgs, list):
                            continue
                        # Format the turn messages
                        prereq_messages = []
                        for msg in turn_msgs:
                            if isinstance(msg, dict):
                                prereq_messages.append({
                                    "role": msg.get("role", "user"),
                                    "content": msg.get("content", "")
                                })
                        if not prereq_messages:
                            continue
                        
                        # Broadcast prerequisite progress so user can see activity
                        _sse_broadcast({
                            "type": "prereq_progress", "job_id": job.job_id,
                            "question": i+1, "total": len(questions),
                            "prereq": pi+1, "total_prereqs": len(prerequisites),
                            "model": model, "suite": suite_key,
                            "message": f"Prereq {pi+1}/{len(prerequisites)} (Q{i+1})"
                        })
                        
                        # Run agentic eval for prerequisite turn
                        def _prereq_stop():
                            return job.skip_requested or suite_state.stop_requested
                        
                        prereq_result = run_agentic_eval(
                            host=ollama_host, model=model, messages=prereq_messages,
                            tool_definitions=tool_defs, tool_executor=backend,
                            max_turns=5, num_predict=suite.get("num_predict", 4096),
                            num_ctx=suite.get("num_ctx", 8192),
                            temperature=suite.get("temperature", 0.0),
                            stop_check=_prereq_stop)
                        print(f"[DEBUG] prereq {pi+1}/{len(prerequisites)} turn done, tool_calls={len(prereq_result.get('tool_calls', []))}, turns={prereq_result.get('turns', 0)}", flush=True)
                        prereq_tool_calls_total += len(prereq_result.get("tool_calls", []))
                        
                        if suite_state.stop_requested or getattr(job, 'stop_requested', False):
                            break
                    if suite_state.stop_requested or getattr(job, 'stop_requested', False):
                        break
                
                if suite_state.stop_requested or getattr(job, 'stop_requested', False):
                    break
                
                # Phase 2: Ask the actual test question
                print(f"[DEBUG] question {i+1}: Phase 2 — asking test question", flush=True)
                # KV/Vector format: row["question"] = [[{role, content}, ...]]
                # RecSum format: row["prompt"] = string repr of messages list
                test_messages = []
                question_data = row.get("question", [])
                if hasattr(question_data, 'tolist'):
                    question_data = question_data.tolist()
                
                # RecSum: try "prompt" field if "question" is empty
                if not question_data:
                    prompt_data = row.get("prompt", [])
                    if hasattr(prompt_data, 'tolist'):
                        prompt_data = prompt_data.tolist()
                    # RecSum prompt can be a list containing stringified message lists
                    # e.g. ["[{'role': 'user', 'content': '...'}]"]
                    if isinstance(prompt_data, list) and len(prompt_data) > 0:
                        first = prompt_data[0]
                        if isinstance(first, str):
                            # Stringified Python list — parse it
                            try:
                                import ast
                                parsed = ast.literal_eval(first)
                                if isinstance(parsed, list):
                                    prompt_data[0] = parsed
                                question_data = prompt_data
                            except Exception:
                                question_data = prompt_data
                        else:
                            question_data = prompt_data
                    elif isinstance(prompt_data, str):
                        try:
                            import ast
                            question_data = ast.literal_eval(prompt_data)
                        except Exception:
                            question_data = [prompt_data]
                
                if isinstance(question_data, list) and len(question_data) > 0:
                    first = question_data[0]
                    if isinstance(first, list):
                        for msg in first:
                            if isinstance(msg, dict):
                                test_messages.append({
                                    "role": msg.get("role", "user"),
                                    "content": msg.get("content", "")
                                })
                    elif isinstance(first, dict):
                        for msg in question_data:
                            if isinstance(msg, dict):
                                test_messages.append({
                                    "role": msg.get("role", "user"),
                                    "content": msg.get("content", "")
                                })
                
                if not test_messages:
                    q_text = row.get("question", "") or row.get("prompt", "")
                    if isinstance(q_text, list):
                        q_text = str(q_text)
                    test_messages = [{"role": "user", "content": str(q_text)}]
                
                # Add system preamble with source context (KV/Vector) or memory context (RecSum)
                source = row.get("source", "")
                initial_cfg = row.get("initial_config", {})
                if hasattr(source, 'tolist'):
                    source = source.tolist()
                if not isinstance(source, str):
                    source = str(source) if source else ""
                system_preamble = ""
                if source:
                    system_preamble = (
                        f"You have access to memory tools. The user previously shared information "
                        f"with you across multiple conversations. Use your memory tools to recall "
                        f"what you know about them. Context: {source[:2000]}")
                elif isinstance(initial_config, dict) and initial_config:
                    # RecSum: tell the model its memory already has content
                    summary = initial_config.get("summary", "")
                    if summary:
                        system_preamble = (
                            f"You have access to memory tools. Your memory already contains some information. "
                            f"Use memory_retrieve to check your memory, then follow the user's instruction."
                        )
                if system_preamble:
                    test_messages.insert(0, {"role": "system", "content": system_preamble})
                
                _sse_broadcast({"type": "question_start", "job_id": job.job_id,
                    "question": i+1, "total": len(questions), "model": model, "suite": suite_key,
                    "prompt": test_messages[-1].get("content", "")[:200] if test_messages else ""})
                
                def _sse_cb(evt):
                    evt["job_id"] = job.job_id
                    evt["model"] = model
                    evt["suite"] = suite_key
                    _sse_broadcast(evt)
                
                def _should_stop():
                    return job.skip_requested or getattr(job, 'stop_requested', False) or suite_state.stop_requested
                
                result = run_agentic_eval(host=ollama_host, model=model, messages=test_messages,
                                           tool_definitions=tool_defs, tool_executor=backend,
                                           max_turns=10, num_predict=suite.get("num_predict", 4096),
                                           num_ctx=suite.get("num_ctx", 8192),
                                           temperature=suite.get("temperature", 0.0),
                                           stop_check=_should_stop,
                                           sse_callback=_sse_cb)
                
                # Memory evaluation
                # KV/Vector: compare model's text answer against ground_truth (list of acceptable strings)
                # RecSum: compare backend memory state against ground_truth_state
                is_correct = False
                expected_display = ""
                
                ground_truth_state = row.get("ground_truth_state", {})
                if hasattr(ground_truth_state, 'tolist'):
                    ground_truth_state = ground_truth_state.tolist()
                if isinstance(ground_truth_state, str):
                    try:
                        ground_truth_state = json.loads(ground_truth_state)
                    except Exception:
                        ground_truth_state = {}
                
                if isinstance(ground_truth_state, dict) and ground_truth_state:
                    # RecSum: state comparison — check if backend memory matches ground_truth_state
                    backend_state = backend.get_state()
                    # Compare summary strings (case-insensitive substring match for flexibility)
                    expected_summary = str(ground_truth_state.get("summary", "")).strip()
                    actual_summary = str(backend_state.get("summary", backend_state.get("memory", ""))).strip()
                    if expected_summary and actual_summary:
                        # Check if key facts from expected are present in actual
                        is_correct, _ = check_web_search_answer(actual_summary, [expected_summary])
                    else:
                        is_correct = False
                    expected_display = json.dumps(ground_truth_state, ensure_ascii=False)[:500]
                else:
                    # KV/Vector: text answer matching
                    expected = row.get("ground_truth", [])
                    if hasattr(expected, 'tolist'):
                        expected = expected.tolist()
                    if isinstance(expected, str):
                        try:
                            expected = json.loads(expected)
                        except:
                            expected = [expected]
                    if not isinstance(expected, list):
                        expected = list(expected) if hasattr(expected, '__iter__') else [expected]
                    # Convert numpy/pandas types to plain Python
                    expected = [str(e) if not isinstance(e, str) else e for e in expected]
                    is_correct, matched = check_web_search_answer(result.get("answer", ""), expected)
                    expected_display = json.dumps(expected, ensure_ascii=False)[:500] if expected else ""
                
                all_tool_calls = prereq_tool_calls_total + len(result.get("tool_calls", []))
                results.append({
                    "question_idx": i, "correct": is_correct,
                    "model_answer": result.get("answer", "")[:200],
                    "expected": expected_display,
                    "tps": result.get("tps", 0), "wall_time": result.get("wall_time", 0),
                    "tool_calls_count": all_tool_calls,
                    "turns": result.get("turns", 0),
                    "prereq_turns": prereq_tool_calls_total,
                })
                _sse_broadcast({"type": "answer", "job_id": job.job_id, "question": i+1, "model": model, "suite": suite_key,
                                "answer": result.get("answer", "")[:50],
                                "expected": expected_display[:200],
                                "correct": is_correct, "tps": result.get("tps", 0), "wall_time": result.get("wall_time", 0),
                                "turns": result.get("turns", 0),
                                "prereq_tool_calls": prereq_tool_calls_total})
        else:
            questions = all_data

        job.total_questions = len(questions)

        if not agentic_done:
            results = []

            try:

              for i, q in enumerate(questions):
                if suite_state.stop_requested or getattr(job, 'stop_requested', False):
                    break
                job.current_question = i + 1

                # After the first question, measure ACTUAL VRAM usage and update the tracker
                # This corrects the pre-load estimate with ground truth
                if i == 0:
                    actual_vram = _get_model_vram_mb(ollama_host, model)
                    if actual_vram > 0:
                        estimated = getattr(job, 'estimated_vram', 0)
                        if estimated > 0:
                            diff = actual_vram - estimated
                            if abs(diff) > 100:  # more than 100MB difference
                                with vram_lock:
                                    vram_in_use["mb"] += diff
                                # Store actual vram so cleanup subtracts the right amount
                                job.estimated_vram = actual_vram
                                _sse_broadcast({
                                    "type": "vram_update",
                                    "job_id": job.job_id,
                                    "estimated_mb": estimated,
                                    "actual_mb": actual_vram,
                                    "message": f"VRAM corrected: {estimated}MB → {actual_vram}MB for {model}"
                                })
                        # Signal scheduler that VRAM is now accurately measured
                        vram_measured_event.set()
                        vram_measure_count["n"] += 1
                        completion_event.set()  # wake scheduler to dispatch next job

                # Format prompt FIRST (before sending question_start) so the correct prompt is shown
                use_chat_api = False
                if suite["answer_type"] == "mcq":
                    prompt, expected = format_mmlu_question(q)
                elif suite["answer_type"] == "function_call":
                    # BFCL: use chat API with proper messages
                    bfcl_messages = format_bfcl_messages(q)
                    prompt, ground_truth = format_bfcl_prompt(q)
                    expected = ground_truth
                    use_chat_api = True
                else:
                    prompt, inst_ids, kwargs_list = format_ifeval_prompt(q)
                    expected = None

                # Notify SSE — prompt is now correct for this question
                _sse_broadcast({
                    "type": "question_start",
                    "job_id": job.job_id,
                    "question": i + 1,
                    "total": len(questions),
                    "model": model,
                    "suite": suite_key,
                    "prompt": prompt,  # full prompt — streaming panel scrolls
                    "category": str(q.get("category", ""))
                })
                _q_start_time = time.time()  # [DEBUG] question timing start
                _q_first_thinking_time = [None]  # [DEBUG] first thinking token time
                _q_first_response_time = [None]  # [DEBUG] first response token time
                _gpu_sampler = GPUSampler()
                _gpu_sampler.start()
                print(f"[Q-START] Q{i+1}/{len(questions)} {model} {suite_key} cat={q.get('category','?')}", flush=True)

                _sse_broadcast({
                    "type": "progress",
                    "job_id": job.job_id,
                    "question": i + 1,
                    "total": len(questions),
                    "model": model,
                    "suite": suite_key
                })

                # Run model with per-question timeout
                full_result = {}
                response_text = ""
                # Per-suite generation parameters
                if suite["answer_type"] == "function_call":
                    # BFCL: structured output, deterministic. Need generous num_predict
                    # because thinking/reasoning models consume output tokens for thinking
                    # before producing the actual function call response.
                    n_predict = 4096
                    temp = 0.0
                    extra_options = {"num_ctx": 8192}
                elif suite["answer_type"] == "mcq":
                    n_predict = None  # uses HIGH_NUM_PREDICT
                    temp = 0.0
                    extra_options = {}
                else:
                    n_predict = None
                    temp = 0.4
                    extra_options = {}

                # Per-suite stop sequences
                if suite["answer_type"] == "function_call":
                    stop_sequences = ["\n\n\n", "\nQuestion", "\nUser:"]
                elif suite["answer_type"] == "mcq":
                    stop_sequences = ["\n\n", "\nQuestion"]
                else:
                    stop_sequences = ["\n\n\n", "\nQuestion"]

                got_result = {"done": False}
                _stream_t0 = [time.time()]  # mutable for closure
                _stream_first_token = [True]  # reset timer on first token to exclude model loading time
                _stream_tokens = [0]  # mutable for closure — token count for live tps
                _wait_count = [0]  # mutable for closure — count "wait" tokens in thinking
                _interrupted = [None]  # mutable for closure — "skip" or "stop" when interrupted mid-stream
                _recent_chunks = []  # rolling window of last N chunks for repetition detection
                _thinking_tokens = [0]  # track thinking tokens to prevent thinking-only blowout
                MAX_THINKING_TOKENS = 8192  # if model thinks this many tokens without answering, abort
                REPETITION_WINDOW = 10  # check last 10 chunks
                REPETITION_THRESHOLD = 6  # if 6+ identical chunks in window, it's stuck

                def _check_repetition(chunk_text):
                    """Detect if model is stuck repeating the same output. Returns True if stuck."""
                    text = (chunk_text or "").strip()
                    if len(text) < 2:
                        return False  # skip whitespace/empty
                    _recent_chunks.append(text)
                    if len(_recent_chunks) > REPETITION_WINDOW:
                        _recent_chunks.pop(0)
                    if len(_recent_chunks) >= REPETITION_THRESHOLD:
                        # Check if the last REPETITION_THRESHOLD chunks are all the same
                        last_n = _recent_chunks[-REPETITION_THRESHOLD:]
                        if len(set(last_n)) == 1:
                            return True
                    return False

                def _should_stop():
                    """Check per-job skip/stop and global stop — allows interrupting streaming generation."""
                    if job.skip_requested:
                        return True
                    if getattr(job, 'stop_requested', False):
                        return True
                    if suite_state.stop_requested:
                        return True
                    return False

                def _run_question():
                    if use_chat_api:
                        stream_iter = stream_chat_chunks(ollama_host, model, bfcl_messages,
                                                          num_predict=n_predict, temperature=temp,
                                                          stop_check=_should_stop, extra_options=extra_options,
                                                          stop_sequences=stop_sequences)
                    else:
                        stream_iter = stream_generate_chunks(ollama_host, model, prompt, num_predict=n_predict, temperature=temp,
                                                             stop_check=_should_stop, extra_options=extra_options,
                                                             stop_sequences=stop_sequences)
                    # Force-close HTTP connection on abort — access generator-internal resp_ref
                    try:
                        for chunk in stream_iter:
                            # Check skip/stop — signaled by API, break stream immediately
                            if job.skip_requested or getattr(job, 'stop_requested', False) or suite_state.stop_requested:
                                if getattr(job, 'stop_requested', False) or suite_state.stop_requested:
                                    _interrupted[0] = "stop"
                                else:
                                    _interrupted[0] = "skip"
                                    _sse_broadcast({
                                        "type": "question_skipped",
                                        "job_id": job.job_id,
                                        "question": i + 1,
                                        "model": model,
                                        "suite": suite_key
                                    })
                                return
                            if chunk["type"] == "error":
                                full_result["error"] = chunk["data"]
                                _sse_broadcast({
                                    "type": "stream_error",
                                    "job_id": job.job_id,
                                    "question": i + 1,
                                    "model": model,
                                    "suite": suite_key,
                                    "error": chunk["data"]
                                })
                                return
                            elif chunk["type"] == "done":
                                full_result.update(chunk["data"])
                                got_result["text"] = full_result.get("text", "") or ""
                                got_result["done"] = True
                                # Send final answer event with evaluation
                                model_answer = got_result["text"]
                                correct = False
                                if suite["answer_type"] == "mcq":
                                    extracted = extract_mcq_answer(model_answer)
                                    correct = extracted == expected
                                elif suite["answer_type"] == "function_call":
                                    is_correct, extracted, bfcl_det = check_bfcl_answer(model_answer, expected)
                                    correct = is_correct
                                else:
                                    extracted = None
                                # Serialize expected for JS display — dicts/lists must be JSON, not Python str()
                                expected_display = expected if isinstance(expected, str) else json.dumps(expected, ensure_ascii=False)[:200] if expected else ""
                                # [DEBUG] Question timing
                                _q_now = time.time()
                                _q_ttft_think = f"{(_q_first_thinking_time[0] - _q_start_time)*1000:.0f}ms" if _q_first_thinking_time[0] else "none"
                                _q_ttft_resp = f"{(_q_first_response_time[0] - _q_start_time)*1000:.0f}ms" if _q_first_response_time[0] else "none"
                                _q_think_dur = f"{(_q_first_response_time[0] - _q_first_thinking_time[0])*1000:.0f}ms" if (_q_first_thinking_time[0] and _q_first_response_time[0]) else "N/A"
                                _q_total = f"{(_q_now - _q_start_time)*1000:.0f}ms"
                                _q_think_tok = _thinking_tokens[0]
                                _q_resp_tok = _stream_tokens[0] - _thinking_tokens[0]
                                print(f"[Q-TIMING] Q{i+1} {model} {suite_key}: total={_q_total} ttft_think={_q_ttft_think} ttft_resp={_q_ttft_resp} think_dur={_q_think_dur} think_tok={_q_think_tok} resp_tok={_q_resp_tok} answer={extracted or model_answer[:30]!r} correct={correct} done_reason={full_result.get('done_reason','')}", flush=True)
                                _sse_broadcast({
                                    "type": "answer",
                                    "job_id": job.job_id,
                                    "question": i + 1,
                                    "model": model,
                                    "suite": suite_key,
                                    "answer": extracted or model_answer[:50] or ("(thinking only)" if full_result.get("done_reason") == "length" and not model_answer else "?"),
                                    "expected": expected_display,
                                    "correct": correct,
                                    "tps": full_result.get("tps", 0),
                                    "wall_time": full_result.get("wall_time", 0),
                                    "done_reason": full_result.get("done_reason", "")
                                })
                                return
                            elif chunk["type"] == "thinking":
                                if _stream_first_token[0]:
                                    _stream_t0[0] = time.time()  # start timer from first token, not from question start
                                    _stream_first_token[0] = False
                                if _q_first_thinking_time[0] is None:
                                    _q_first_thinking_time[0] = time.time()  # [DEBUG]
                                _stream_tokens[0] += 1
                                _thinking_tokens[0] += 1
                                # Thinking cap — abort if model thinks too long without producing an answer
                                if _thinking_tokens[0] >= MAX_THINKING_TOKENS:
                                    _sse_broadcast({"type": "thinking_cap", "job_id": job.job_id, "question": i + 1, "model": model, "suite": suite_key, "thinking_tokens": _thinking_tokens[0]})
                                    _interrupted[0] = "thinking_cap"
                                    return
                                # Repetition detection
                                if _check_repetition(chunk.get("data", "")):
                                    _sse_broadcast({"type": "repetition_detected", "job_id": job.job_id, "question": i + 1, "model": model, "suite": suite_key, "repeated_text": _recent_chunks[-1][:80]})
                                    _interrupted[0] = "repetition"
                                    return
                                # Count "wait" tokens in thinking text
                                import re as _re
                                _wait_matches = _re.findall(r'\b[Ww]\s*[Aa]\s*[Ii]\s*[Tt]\s*', chunk["data"] or "")
                                _wait_count[0] += len(_wait_matches)
                                _sse_broadcast({
                                    "type": "thinking",
                                    "job_id": job.job_id,
                                    "question": i + 1,
                                    "model": model,
                                    "suite": suite_key,
                                    "text": chunk["data"],
                                    "token_count": _stream_tokens[0],
                                    "elapsed": time.time() - _stream_t0[0],
                                    "wait_count": _wait_count[0]
                                })
                            elif chunk["type"] == "token":
                                if _stream_first_token[0]:
                                    _stream_t0[0] = time.time()  # start timer from first token, not from question start
                                    _stream_first_token[0] = False
                                if _q_first_response_time[0] is None:
                                    _q_first_response_time[0] = time.time()  # [DEBUG]
                                _stream_tokens[0] += 1
                                # Repetition detection
                                if _check_repetition(chunk.get("data", "")):
                                    _sse_broadcast({"type": "repetition_detected", "job_id": job.job_id, "question": i + 1, "model": model, "suite": suite_key, "repeated_text": _recent_chunks[-1][:80]})
                                    _interrupted[0] = "repetition"
                                    return
                                elapsed = time.time() - _stream_t0[0]
                                live_tps = _stream_tokens[0] / elapsed if elapsed > 0.5 else 0
                                _sse_broadcast({
                                    "type": "token",
                                    "job_id": job.job_id,
                                    "question": i + 1,
                                    "model": model,
                                    "suite": suite_key,
                                    "text": chunk["data"],
                                    "token_count": _stream_tokens[0],
                                    "elapsed": elapsed,
                                    "live_tps": round(live_tps, 1)
                                })
                        # Generator exhausted normally — check if stop_check() aborted mid-stream
                        if not got_result.get("done") and not full_result.get("error") and (job.skip_requested or suite_state.stop_requested or getattr(job, 'stop_requested', False)):
                            if getattr(job, 'stop_requested', False) or suite_state.stop_requested:
                                _interrupted[0] = "stop"
                            else:
                                _interrupted[0] = "skip"
                        else:
                            got_result["done"] = True
                    finally:
                        # Force-close the generator to terminate the HTTP stream immediately.
                        # Without this, the Ollama connection stays open (generator not GC'd yet)
                        # and subsequent questions may go out of sync.
                        if stream_iter and hasattr(stream_iter, 'close'):
                            stream_iter.close()

                q_thread = threading.Thread(target=_run_question, daemon=True)
                q_thread.start()
                q_thread.join(timeout=SUITE_QUESTION_TIMEOUT)

                # Stop GPU sampler and collect metrics for this question
                _gpu_sampler.stop()
                _gpu_metrics = _gpu_sampler.metrics

                def _with_gpu(d):
                    """Add GPU metrics to a result dict."""
                    if _gpu_metrics and "error" not in _gpu_metrics:
                        d["gpu_power_avg_w"] = _gpu_metrics["total_power_w"].get("avg", 0)
                        d["gpu_power_peak_w"] = _gpu_metrics["total_power_w"].get("peak", 0)
                        d["gpu_mem_peak_mib"] = sum(g["mem_peak_mib"] for g in _gpu_metrics.get("gpus", []) if g["index"] in _gpu_metrics.get("active_gpus", []))
                        d["gpu_active"] = _gpu_metrics.get("active_gpus", [])
                        d["gpu_util_avg"] =round(sum(g["gpu_util_avg"] for g in _gpu_metrics.get("gpus", []) if g["index"] in _gpu_metrics.get("active_gpus", [])) / max(len(_gpu_metrics.get("active_gpus", [])), 1), 1)
                        wt = d.get("wall_time", 0)
                        if wt > 0:
                            d["gpu_energy_wh"] = round(_gpu_metrics["total_power_w"].get("avg", 0) * wt / 3600, 2)
                        # Per-GPU detail for drill-down
                        d["gpu_detail"] = [{k: v for k, v in g.items() if k != "pids"} for g in _gpu_metrics.get("gpus", []) if g["index"] in _gpu_metrics.get("active_gpus", [])]
                    return d

                # Handle skip (user pressed Skip button) — stream was interrupted mid-generation
                if _interrupted[0] == "skip":
                    results.append(_with_gpu({
                        "question_idx": i,
                        "correct": False,
                        "model_answer": "SKIPPED",
                        "expected": json.dumps(expected, ensure_ascii=False)[:500] if expected and not isinstance(expected, str) else (expected or ""),
                        "tps": 0,
                        "wall_time": 0,
                        "wait_count": _wait_count[0],
                        "error": "Skipped by user",
                    }))
                    job.skip_requested = False  # reset for next question
                    continue

                # Handle repetition loop — model got stuck repeating the same output
                if _interrupted[0] == "repetition":
                    results.append(_with_gpu({
                        "question_idx": i,
                        "correct": False,
                        "model_answer": "REPETITION_LOOP",
                        "expected": json.dumps(expected, ensure_ascii=False)[:500] if expected and not isinstance(expected, str) else (expected or ""),
                        "tps": 0,
                        "wall_time": 0,
                        "tokens": _stream_tokens[0],
                        "wait_count": _wait_count[0],
                        "error": "Repetition loop detected",
                    }))
                    continue  # move on to next question

                # Handle thinking cap — model used all tokens on thinking, no answer
                if _interrupted[0] == "thinking_cap":
                    results.append(_with_gpu({
                        "question_idx": i,
                        "correct": False,
                        "model_answer": "THINKING_ONLY",
                        "expected": json.dumps(expected, ensure_ascii=False)[:500] if expected and not isinstance(expected, str) else (expected or ""),
                        "tps": 0,
                        "wall_time": 0,
                        "tokens": _stream_tokens[0],
                        "wait_count": _wait_count[0],
                        "error": "Model used %d tokens thinking without producing an answer" % _thinking_tokens[0],
                    }))
                    continue  # move on to next question

                # Handle stop — stream was interrupted, stop processing remaining questions
                if _interrupted[0] == "stop" or suite_state.stop_requested or getattr(job, 'stop_requested', False):
                    results.append(_with_gpu({
                        "question_idx": i,
                        "correct": False,
                        "model_answer": "STOPPED",
                        "expected": json.dumps(expected, ensure_ascii=False)[:500] if expected and not isinstance(expected, str) else (expected or ""),
                        "tps": 0,
                        "wall_time": 0,
                        "wait_count": _wait_count[0],
                        "error": "Stopped by user",
                    }))
                    break  # don't continue to next question

                if q_thread.is_alive():
                    # Question timed out — abort the generation and wait for thread to finish
                    job.skip_requested = True  # signal stop_check to break the stream loop
                    # Try to close the HTTP response directly to unblock stuck threads
                    # Access generator-internal resp_ref via gi_frame
                    try:
                        _resp_ref = stream_iter.gi_frame.f_locals.get('resp_ref') if stream_iter and hasattr(stream_iter, 'gi_frame') else None
                        if _resp_ref and isinstance(_resp_ref, list) and _resp_ref[0]:
                            _resp_ref[0].close()
                    except Exception:
                        pass
                    q_thread.join(timeout=5)  # give the stream loop a moment to close the connection
                    # Force-unload model to abort any in-progress Ollama generation
                    try:
                        req_lib.post(f"{ollama_host}/api/generate", json={"model": model, "prompt": "", "keep_alive": 0}, timeout=10)
                    except Exception:
                        pass
                    if q_thread.is_alive():
                        # Thread still stuck — log warning, it's daemon so it won't block shutdown
                        import warnings
                        warnings.warn(f"Question thread for {model} still alive after abort + unload")
                    results.append(_with_gpu({
                        "question_idx": i,
                        "correct": False,
                        "model_answer": "TIMEOUT",
                        "expected": json.dumps(expected, ensure_ascii=False)[:500] if expected and not isinstance(expected, str) else (expected or ""),
                        "tps": 0,
                        "wall_time": SUITE_QUESTION_TIMEOUT,
                        "tokens": _stream_tokens[0],
                        "wait_count": _wait_count[0],
                        "error": f"Question timed out after {SUITE_QUESTION_TIMEOUT}s",
                    }))
                    job.skip_requested = False  # reset for next question
                    continue

                response_text = got_result.get("text", "")

                # Detect "thinking only" — model used all its tokens on thinking without answering
                if full_result.get("done_reason") == "length" and not response_text.strip():
                    results.append(_with_gpu({
                        "question_idx": i,
                        "correct": False,
                        "model_answer": "THINKING_ONLY",
                        "expected": json.dumps(expected, ensure_ascii=False)[:500] if expected and not isinstance(expected, str) else (expected or ""),
                        "tps": full_result.get("tps", 0),
                        "wall_time": full_result.get("wall_time", 0),
                        "tokens": _stream_tokens[0],
                        "wait_count": _wait_count[0],
                        "error": "Model hit context limit (%d tokens) thinking without producing an answer" % _stream_tokens[0],
                    }))
                    continue

                # Evaluate
                if "error" in full_result:
                    results.append(_with_gpu({
                        "question_idx": i,
                        "correct": False,
                        "model_answer": "ERROR",
                        "expected": json.dumps(expected, ensure_ascii=False)[:500] if expected and not isinstance(expected, str) else (expected or ""),
                        "tps": 0,
                        "wall_time": 0,
                        "tokens": _stream_tokens[0],
                        "wait_count": _wait_count[0],
                        "error": full_result["error"],
                    }))
                    continue

                if suite["answer_type"] == "mcq":
                    model_answer = extract_mcq_answer(response_text, suite.get("num_options", 10))
                    is_correct = (model_answer == expected) if model_answer and expected else False
                    results.append(_with_gpu({
                        "question_idx": i,
                        "correct": is_correct,
                        "model_answer": model_answer,
                        "expected": expected,
                        "tps": full_result.get("tps", 0),
                        "wall_time": full_result.get("wall_time", 0),
                        "tokens": _stream_tokens[0],
                        "wait_count": _wait_count[0],
                        "category": q.get("category", ""),
                    }))
                elif suite["answer_type"] == "function_call":
                    is_correct, model_answer, details = check_bfcl_answer(response_text, expected)
                    results.append(_with_gpu({
                        "question_idx": i,
                        "correct": is_correct,
                        "model_answer": model_answer,
                        "expected": json.dumps(expected, ensure_ascii=False)[:500] if expected else "",
                        "tps": full_result.get("tps", 0),
                        "wall_time": full_result.get("wall_time", 0),
                        "tokens": _stream_tokens[0],
                        "wait_count": _wait_count[0],
                        "category": q.get("category", ""),
                        "bfcl_details": details,
                    }))
                elif suite["answer_type"] == "instruction":
                    inst_ids, kwargs_list = q.get("instruction_id_list", []), q.get("kwargs", [])
                    constraint_results = check_ifeval_constraints(response_text, inst_ids, kwargs_list)
                    constraints_total = len(constraint_results)
                    constraints_passed = sum(1 for _, p in constraint_results if p)
                    results.append(_with_gpu({
                        "question_idx": i,
                        "correct": constraints_passed == constraints_total,
                        "constraints_total": constraints_total,
                        "constraints_passed": constraints_passed,
                        "tps": full_result.get("tps", 0),
                        "wall_time": full_result.get("wall_time", 0),
                        "tokens": _stream_tokens[0],
                        "wait_count": _wait_count[0],
                        "constraint_details": [(cid, p) for cid, p in constraint_results],
                    }))

            except Exception as e:
                import traceback
                traceback.print_exc()
                job.active = False
                _sse_broadcast({"type": "error", "job_id": job.job_id, "error": str(e)})
                return False

        # Save results
        job.results = results
        total_q = len(results)
        correct = sum(1 for r in results if r.get("correct"))
        wrong = total_q - correct - sum(1 for r in results if r.get("error"))
        skipped = sum(1 for r in results if r.get("error"))
        accuracy = (correct / total_q * 100) if total_q else 0
        avg_tps = sum(r.get("tps", 0) for r in results if r.get("wall_time", 0) > 0) / max(sum(1 for r in results if r.get("wall_time", 0) > 0), 1)
        total_wall = sum(r.get("wall_time", 0) for r in results)
        total_tokens = sum(r.get("tokens", 0) for r in results)
        # Aggregate GPU metrics across all questions
        gpu_power_vals = [r.get("gpu_power_avg_w", 0) for r in results if r.get("gpu_power_avg_w")]
        gpu_mem_peaks = [r.get("gpu_mem_peak_mib", 0) for r in results if r.get("gpu_mem_peak_mib")]
        avg_gpu_power = sum(gpu_power_vals) / len(gpu_power_vals) if gpu_power_vals else 0
        gpu_mem_peak = max(gpu_mem_peaks) if gpu_mem_peaks else 0
        gpu_energy_wh = sum(r.get("gpu_energy_wh", 0) for r in results if r.get("gpu_energy_wh"))
        # Active GPUs: union across all questions
        gpu_active_set = set()
        for r in results:
            if isinstance(r.get("gpu_active"), list):
                gpu_active_set.update(r["gpu_active"])
        avg_gpu_util = sum(r.get("gpu_util_avg", 0) for r in results if r.get("gpu_util_avg")) / max(sum(1 for r in results if r.get("gpu_util_avg")), 1)

        sys_info = detect_system_info()
        gpu_info = sys_info.get("gpu_summary", "")
        computer_id = compute_computer_id(sys_info)

        with db_lock:
            db = get_db()
            run_id = db.execute(
                "INSERT INTO runs (timestamp, host, gpu_info, system_info, computer_id) VALUES (?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), ollama_host, gpu_info,
                 json.dumps(sys_info) if sys_info else "", computer_id)
            ).lastrowid
            db.execute(
                """INSERT INTO benchmark_results
                   (run_id, model, suite_key, accuracy, total_questions, correct, wrong, skipped, avg_tps, total_wall_time, details_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, model, suite_key, accuracy, total_q, correct, wrong, skipped, avg_tps, total_wall,
                 json.dumps(results))
            )
            db.commit()
            db.close()

        job.run_id = run_id
        job.active = False

        # Notify: job completed
        _sse_broadcast({
            "type": "job_done",
            "job_id": job.job_id,
            "run_id": run_id,
            "model": model,
            "suite": suite_key,
            "accuracy": accuracy,
            "correct": correct,
            "total": total_q,
            "total_tokens": total_tokens,
        })

        # Unload model to free VRAM from Ollama
        try:
            req_lib.post(f"{ollama_host}/api/generate", json={"model": model, "prompt": "", "keep_alive": 0}, timeout=10)
        except Exception:
            pass

        return True

    def suite_processor():
        """Process queued jobs with VRAM-aware parallel dispatch.
        
        After dispatching the first job, waits for the model to load and
        measures actual VRAM before deciding to dispatch more. This prevents
        estimating too low and Ollama swapping models in and out.
        """
        try:
          suite_processor_inner()
        except Exception as e:
          import traceback
          logger.error(f"suite_processor crashed: {e}\n{traceback.format_exc()}")
        finally:
          suite_state._processor_running = False
          was_stopped = suite_state.stop_requested
          suite_state.stop_requested = False
          evt = {"type": "queue_done"}
          if was_stopped:
              evt["stopped"] = True
          _sse_broadcast(evt)

    def suite_processor_inner():
        first_job_dispatched = False
        dispatched_count = 0  # how many jobs dispatched since last measurement
        while True:
            if suite_state.stop_requested:
                break
            # Try to dispatch any queued jobs that fit
            while suite_state.suite_queue:
                # If we've dispatched jobs but they haven't measured actual VRAM yet,
                # wait before dispatching more (avoid under-estimating VRAM)
                if dispatched_count > 0 and dispatched_count > vram_measure_count["n"]:
                    break  # wait for measurement before dispatching more

                next_job_dict = suite_state.suite_queue[0]
                model_vram = _get_model_vram_mb(next_job_dict["host"], next_job_dict["model"])

                with vram_lock:
                    fits = (vram_in_use["mb"] + model_vram) <= vram_budget_mb

                if fits:
                    # Try to acquire a slot (non-blocking)
                    acquired = job_slots.acquire(blocking=False)
                    if acquired:
                        with queue_lock:
                            job_dict = suite_state.suite_queue.pop(0)
                        job = SuiteJob(
                            host=job_dict["host"],
                            model=job_dict["model"],
                            suite_key=job_dict["suite_key"],
                            sample_size=job_dict["sample_size"],
                            job_id=suite_state.next_job_id(),
                        )
                        suite_state.active_jobs.append(job)
                        job.estimated_vram = model_vram  # store estimate for later correction

                        with vram_lock:
                            vram_in_use["mb"] += model_vram

                        t = threading.Thread(target=_run_job_wrapper, args=(job, model_vram), daemon=True)
                        t.start()
                        first_job_dispatched = True
                        dispatched_count += 1
                    else:
                        # No slot available — wait for a completion
                        break
                else:
                    # Model won't fit alongside current jobs.
                    # If nothing else is running (vram_in_use == 0), it's a large model —
                    # dispatch it solo anyway. Ollama will offload to system RAM as needed.
                    with vram_lock:
                        nothing_running = vram_in_use["mb"] == 0
                    
                    if nothing_running:
                        # Large model solo dispatch — user explicitly queued it, so run it
                        acquired = job_slots.acquire(blocking=False)
                        if acquired:
                            with queue_lock:
                                job_dict = suite_state.suite_queue.pop(0)
                            job = SuiteJob(
                                host=job_dict["host"],
                                model=job_dict["model"],
                                suite_key=job_dict["suite_key"],
                                sample_size=job_dict["sample_size"],
                                job_id=suite_state.next_job_id(),
                            )
                            suite_state.active_jobs.append(job)
                            job.estimated_vram = model_vram

                            with vram_lock:
                                vram_in_use["mb"] += model_vram

                            _sse_broadcast({
                                "type": "vram_warning",
                                "model": job_dict["model"],
                                "estimated_mb": model_vram,
                                "budget_mb": vram_budget_mb,
                                "message": f"Large model {job_dict['model']} ({model_vram//1024}GB est.) exceeds parallel budget ({vram_budget_mb//1024}GB) — running solo"
                            })

                            t = threading.Thread(target=_run_job_wrapper, args=(job, model_vram), daemon=True)
                            t.start()
                            first_job_dispatched = True
                            dispatched_count += 1
                        # Whether we dispatched or not, break inner loop to wait
                        break
                    else:
                        # Other jobs are running — wait for them to finish, then try again
                        break

            if not suite_state.suite_queue and not suite_state.active_jobs:
                # Queue empty and no jobs running — we're done
                break

            # Wait for a job to complete OR VRAM measurement before trying again
            completion_event.clear()
            # Also wake when VRAM is measured (allows dispatching parallel jobs sooner)
            vram_measured_event.clear()
            completion_event.wait(timeout=5)
            if vram_measured_event.is_set():
                continue  # re-check queue immediately

        # All done (or stopped)
        # queue_done event is sent by the try/finally wrapper in suite_processor()

    def _run_job_wrapper(job, estimated_vram):
        """Thread wrapper that runs a job and cleans up."""
        try:
            _run_single_suite(job)
        finally:
            # Remove from active jobs
            if job in suite_state.active_jobs:
                suite_state.active_jobs.remove(job)
            # Unload model from Ollama to free VRAM
            try:
                req_lib.post(f"{job.host}/api/generate", json={"model": job.model, "prompt": "", "keep_alive": 0}, timeout=10)
            except Exception:
                pass
            # Use the corrected VRAM estimate (updated after model loaded)
            # If the job corrected vram_in_use during run, we've already adjusted.
            # Subtract whatever we added at dispatch time.
            freed_vram = estimated_vram
            with vram_lock:
                vram_in_use["mb"] -= freed_vram
            job_slots.release()
            completion_event.set()

    threading.Thread(target=suite_processor, daemon=True).start()


def _sample_mmlu(all_data, per_category):
    """Sample N questions per category from MMLU-Pro."""
    import random
    random.seed(42)
    by_cat = {}
    for row in all_data:
        cat = row.get("category", "other")
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(row)
    sampled = []
    for cat, rows in sorted(by_cat.items()):
        if per_category > 0 and len(rows) > per_category:
            sampled.extend(random.sample(rows, per_category))
        else:
            sampled.extend(rows)
    random.shuffle(sampled)
    return sampled


@router.get("/api/suites/stream")
async def api_suite_stream():
    """SSE stream for suite progress events."""
    queue = asyncio.Queue()
    suite_state.event_queues.append(queue)

    async def event_generator():
        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30)
                    yield {"data": data}
                except asyncio.TimeoutError:
                    yield {"data": json.dumps({"type": "ping"})}
        except asyncio.CancelledError:
            pass
        finally:
            if queue in suite_state.event_queues:
                suite_state.event_queues.remove(queue)

    return EventSourceResponse(event_generator())


@router.get("/api/suites/status")
async def api_suite_status():
    """Get current suite run status — all active jobs and queue."""
    active = [{"job_id": j.job_id, "model": j.model, "suite_key": j.suite_key,
               "current_question": j.current_question, "total_questions": j.total_questions}
              for j in suite_state.active_jobs]
    queue = [{"model": j["model"], "suite_key": j["suite_key"], "sample_size": j["sample_size"]}
             for j in suite_state.suite_queue]
    return JSONResponse({
        "active": suite_state.active,
        "processor_running": suite_state._processor_running,
        "max_parallel": suite_state.max_parallel,
        "active_jobs": active,
        "queue": queue,
        "queue_length": len(suite_state.suite_queue),
    })


@router.get("/api/suites/queue")
async def api_suite_queue():
    """View the suite run queue."""
    active = [{"job_id": j.job_id, "model": j.model, "suite_key": j.suite_key,
               "current_question": j.current_question, "total_questions": j.total_questions}
              for j in suite_state.active_jobs]
    queue = [{"model": j["model"], "suite_key": j["suite_key"], "sample_size": j["sample_size"]}
             for j in suite_state.suite_queue]
    return JSONResponse({
        "active": suite_state.active,
        "active_jobs": active,
        "queue": queue,
        "max_parallel": suite_state.max_parallel,
    })


@router.delete("/api/suites/queue")
async def api_suite_queue_clear():
    """Clear all pending jobs from the queue (does not stop running jobs)."""
    cleared = len(suite_state.suite_queue)
    suite_state.suite_queue.clear()
    return JSONResponse({"cleared": cleared})


@router.post("/api/suites/stop")
async def api_suite_stop():
    """Stop all suite jobs: clears queue, signals running jobs to stop, saves partial results.
    If jobs don't stop within 5 seconds, force-clear them."""
    cleared = len(suite_state.suite_queue)
    suite_state.suite_queue.clear()
    if suite_state.active:
        suite_state.stop_requested = True
        # Also skip all active questions to unblock agentic loops
        for job in suite_state.active_jobs:
            job.skip_requested = True
        # Notify SSE subscribers
        _sse_broadcast({"type": "stopping", "cleared": cleared})
    # Give jobs 5 seconds to gracefully stop, then force-clear
    import asyncio
    await asyncio.sleep(5)
    if suite_state.active_jobs:
        # Force-clear stuck active jobs
        for job in list(suite_state.active_jobs):
            job.active = False
        suite_state.active_jobs.clear()
        suite_state._processor_running = False
        _sse_broadcast({"type": "queue_done", "stopped": True, "force_cleared": True})
    return JSONResponse({"stopping": suite_state.active, "cleared_queue": cleared})


@router.delete("/api/suites/queue/{index}")
async def api_suite_queue_delete(index: int):
    """Delete a single item from the benchmark queue by index."""
    with queue_lock:
        if 0 <= index < len(suite_state.suite_queue):
            removed = suite_state.suite_queue.pop(index)
            _sse_broadcast({"type": "queue_updated"})
            return JSONResponse({"removed": removed})
    return JSONResponse({"error": f"Index {index} out of range (queue has {len(suite_state.suite_queue)} items)"}, status_code=404)


@router.post("/api/suites/skip/{job_id}")
async def api_suite_skip_question(job_id: int):
    """Skip the current question for a running suite job. The model continues to the next question."""
    for job in suite_state.active_jobs:
        if job.job_id == job_id and job.active:
            job.skip_requested = True
            return JSONResponse({"skipped": True, "job_id": job_id, "model": job.model, "suite": job.suite_key, "question": job.current_question})
    return JSONResponse({"error": f"No active job with id {job_id}", "skipped": False}, status_code=404)

@router.post("/api/suites/stop/{job_id}")
async def api_suite_stop_job(job_id: int):
    """Stop a single running job: signals it to stop, removes it from active jobs after grace period."""
    target = None
    for job in suite_state.active_jobs:
        if job.job_id == job_id and job.active:
            target = job
            break
    if not target:
        # Maybe it's in the queue — remove it
        with queue_lock:
            for idx, item in enumerate(suite_state.suite_queue):
                if item.get("job_id") == job_id:
                    suite_state.suite_queue.pop(idx)
                    return JSONResponse({"stopped": True, "job_id": job_id, "was_queued": True})
        return JSONResponse({"error": f"No active or queued job with id {job_id}"}, status_code=404)
    # Signal the job to stop after current question finishes
    target.stop_requested = True
    target.skip_requested = True  # also skip current question to unblock agentic loops
    # Notify SSE
    _sse_broadcast({"type": "job_stopping", "job_id": job_id, "model": target.model, "suite": target.suite_key})
    return JSONResponse({"stopping": True, "job_id": job_id, "model": target.model, "suite": target.suite_key})


@router.post("/api/suites/max-parallel")
async def api_set_max_parallel(request: Request):
    """Set max parallel jobs. Body: {max_parallel: N}. Only affects future dispatches."""
    body = await request.json()
    n = int(body.get("max_parallel", 2))
    n = max(1, min(n, 8))  # clamp 1-8
    suite_state.max_parallel = n
    return JSONResponse({"max_parallel": suite_state.max_parallel})


@router.get("/api/suites/results")
async def api_suite_results(run_id: Optional[int] = None, suite_key: str = ""):
    """Get benchmark suite results. Filter by run_id or suite_key."""
    db = get_db()
    conditions = []
    params = []
    if run_id:
        conditions.append("br.run_id = ?")
        params.append(run_id)
    if suite_key:
        conditions.append("br.suite_key = ?")
        params.append(suite_key)
    where = " AND ".join(conditions) if conditions else "1=1"

    rows = db.execute(f"""
        SELECT br.id, br.run_id, br.model, br.suite_key,
               br.accuracy, br.total_questions, br.correct, br.wrong, br.skipped,
               br.avg_tps, br.total_wall_time, br.details_json,
               r.timestamp, r.computer_id
        FROM benchmark_results br
        JOIN runs r ON br.run_id = r.id
        WHERE {where}
        ORDER BY r.timestamp DESC
    """, params).fetchall()
    db.close()

    results = []
    for row in rows:
        details = json.loads(row[11]) if row[11] else []
        total_tokens = sum(q.get("tokens", 0) for q in details)
        # Aggregate GPU metrics from per-question details
        gpu_power_vals = [q.get("gpu_power_avg_w", 0) for q in details if q.get("gpu_power_avg_w")]
        gpu_mem_peaks = [q.get("gpu_mem_peak_mib", 0) for q in details if q.get("gpu_mem_peak_mib")]
        avg_gpu_power = sum(gpu_power_vals) / len(gpu_power_vals) if gpu_power_vals else None
        gpu_mem_peak = max(gpu_mem_peaks) if gpu_mem_peaks else None
        gpu_energy_wh = sum(q.get("gpu_energy_wh", 0) for q in details if q.get("gpu_energy_wh"))
        gpu_active_set = set()
        for q in details:
            if isinstance(q.get("gpu_active"), list):
                gpu_active_set.update(q["gpu_active"])
        gpu_util_vals = [q.get("gpu_util_avg", 0) for q in details if q.get("gpu_util_avg")]
        avg_gpu_util = sum(gpu_util_vals) / len(gpu_util_vals) if gpu_util_vals else None
        results.append({
            "id": row[0],
            "run_id": row[1],
            "model": row[2],
            "suite_key": row[3],
            "suite_name": BENCHMARK_SUITES.get(row[3], {}).get("name", row[3]),
            "accuracy": row[4],
            "total_questions": row[5],
            "correct": row[6],
            "wrong": row[7],
            "skipped": row[8],
            "avg_tps": row[9],
            "total_wall_time": row[10],
            "total_tokens": total_tokens,
            "avg_gpu_power": avg_gpu_power,
            "gpu_mem_peak": gpu_mem_peak,
            "gpu_energy_wh": gpu_energy_wh if gpu_energy_wh else None,
            "gpu_active": sorted(gpu_active_set) if gpu_active_set else None,
            "avg_gpu_util": avg_gpu_util,
            "timestamp": row[12],
            "computer_id": row[13] or "",
        })
    return JSONResponse(results)


@router.delete("/api/suites/results/{result_id}")
async def api_delete_suite_result(result_id: int):
    """Delete a single suite result by its ID."""
    with db_lock:
        db = get_db()
        row = db.execute("SELECT id FROM benchmark_results WHERE id = ?", (result_id,)).fetchone()
        if not row:
            db.close()
            return JSONResponse({"error": "Result not found"}, status_code=404)
        db.execute("DELETE FROM benchmark_results WHERE id = ?", (result_id,))
        db.commit()
        db.close()
    return JSONResponse({"deleted": result_id})


@router.delete("/api/suites/results")
async def api_delete_all_suite_results(suite_key: str = ""):
    """Delete all suite results, optionally filtered by suite_key."""
    with db_lock:
        db = get_db()
        if suite_key:
            count = db.execute("SELECT COUNT(*) FROM benchmark_results WHERE suite_key = ?", (suite_key,)).fetchone()[0]
            db.execute("DELETE FROM benchmark_results WHERE suite_key = ?", (suite_key,))
        else:
            count = db.execute("SELECT COUNT(*) FROM benchmark_results").fetchone()[0]
            db.execute("DELETE FROM benchmark_results")
        db.commit()
        db.close()
    return JSONResponse({"deleted_count": count})


@router.get("/api/stream")
async def api_stream():
    queue = asyncio.Queue()
    bench_state.event_queues.append(queue)

    async def event_generator():
        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30)
                    yield {"data": data}
                except asyncio.TimeoutError:
                    yield {"data": json.dumps({"type": "ping"})}
        except asyncio.CancelledError:
            pass
        finally:
            if queue in bench_state.event_queues:
                bench_state.event_queues.remove(queue)

    return EventSourceResponse(event_generator())


@router.get("/api/judge-stream")
async def api_judge_stream():
    queue = asyncio.Queue()
    judge_state.event_queues.append(queue)

    async def event_generator():
        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30)
                    yield {"data": data}
                except asyncio.TimeoutError:
                    yield {"data": json.dumps({"type": "ping"})}
        except asyncio.CancelledError:
            pass
        finally:
            if queue in judge_state.event_queues:
                judge_state.event_queues.remove(queue)

    return EventSourceResponse(event_generator())

