#!/usr/bin/env python3
"""SQLite database layer — schema, CRUD, migrations."""

import json
import hashlib
import re
import sqlite3
import threading
import subprocess
import platform
from pathlib import Path
from datetime import datetime, timezone

from ollama_bench.config import DEFAULT_DB, TESTS_FILE
from ollama_bench.tests import load_tests

# ═══════════════════════════════════════════════

_cfg = {"db_path": str(DEFAULT_DB)}
db_lock = threading.Lock()

def get_db():
    return sqlite3.connect(_cfg["db_path"])

def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            host TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS model_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            is_thinking INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_result_id INTEGER NOT NULL,
            test_name TEXT NOT NULL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            thinking_chunks INTEGER,
            response_chunks INTEGER,
            wall_time REAL,
            ttft REAL,
            ttft_response REAL,
            thinking_duration REAL,
            response_duration REAL,
            total_duration REAL,
            load_duration REAL,
            prompt_eval_duration REAL,
            eval_duration REAL,
            tps REAL,
            prompt_tps REAL,
            stop_reason TEXT,
            response_text TEXT,
            thinking_excerpt TEXT,
            quality_rating INTEGER,
            FOREIGN KEY (model_result_id) REFERENCES model_results(id)
        )
    """)
    db.commit()
    # New columns for LLM Judge and Math Verification
    new_columns = [
        ("judge_model", "TEXT"),
        ("judge_score", "INTEGER"),
        ("judge_reasoning", "TEXT"),
        ("judge_dimensions", "TEXT"),  # JSON
        ("judge_duration", "REAL"),
        ("math_correct", "INTEGER"),  # 0 or 1, NULL if not applicable
        ("math_answer", "TEXT"),  # extracted answer
        ("math_expected", "TEXT"),  # expected answer for reference
    ]
    for col_name, col_type in new_columns:
        try:
            db.execute(f"ALTER TABLE test_results ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    # GPU info column for runs table
    try:
        db.execute("ALTER TABLE runs ADD COLUMN gpu_info TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    # System info column (JSON) for full hardware specs
    try:
        db.execute("ALTER TABLE runs ADD COLUMN system_info TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    # Computer ID column for hardware fingerprint
    try:
        db.execute("ALTER TABLE runs ADD COLUMN computer_id TEXT")
    except sqlite3.OperationalError:
        pass
    # Test definitions table
    db.execute("""
        CREATE TABLE IF NOT EXISTS tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            prompt_sha256 TEXT NOT NULL,
            prompt TEXT NOT NULL,
            rating_type TEXT DEFAULT 'subjective',
            expected_answer TEXT,
            answer_pattern TEXT,
            category TEXT,
            difficulty TEXT,
            is_default INTEGER DEFAULT 0,
            created_at TEXT,
            UNIQUE(name, prompt_sha256)
        )
    """)
    # Link test_results to test definitions
    try:
        db.execute("ALTER TABLE test_results ADD COLUMN test_id INTEGER REFERENCES tests(id)")
    except sqlite3.OperationalError:
        pass
    # Benchmark suite results (MMLU-Pro, IFEval, etc.)
    db.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            suite_key TEXT NOT NULL,
            accuracy REAL,
            total_questions INTEGER,
            correct INTEGER,
            wrong INTEGER,
            skipped INTEGER,
            avg_tps REAL,
            total_wall_time REAL,
            total_wait INTEGER DEFAULT 0,
            details_json TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
    """)

    # Model profiles — measured VRAM/RAM/GPU% per model per computer
    db.execute("""
        CREATE TABLE IF NOT EXISTS model_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            computer_id TEXT NOT NULL,
            model TEXT NOT NULL,
            vram_mib INTEGER,
            ram_mib INTEGER,
            gpu_pct REAL,
            context_length INTEGER,
            parameter_size TEXT,
            quantization TEXT,
            profiled_at TEXT NOT NULL,
            UNIQUE(computer_id, model)
        )
    """)
    db.commit()
    db.close()


def prompt_sha256(prompt: str) -> str:
    """Compute SHA-256 hash of a prompt for test identity."""
    import hashlib
    return hashlib.sha256(prompt.encode()).hexdigest()


def seed_tests_to_db():
    """Insert default and custom tests into the tests table (idempotent).
    Called on startup after init_db(). Each test is identified by (name, prompt_sha256).
    """
    import hashlib
    all_tests = load_tests()
    now = datetime.now(timezone.utc).isoformat()
    with db_lock:
        db = get_db()
        for t in all_tests:
            p = t.get("prompt", "")
            if isinstance(p, tuple):
                p = "".join(p)
            sha = hashlib.sha256(p.encode()).hexdigest()
            is_default = 1 if t.get("default", False) else 0
            try:
                db.execute(
                    """INSERT OR IGNORE INTO tests
                    (name, prompt_sha256, prompt, rating_type, expected_answer, answer_pattern,
                     category, difficulty, is_default, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        t["name"], sha, p,
                        t.get("rating_type", "objective" if t.get("expected_answer") else "subjective"),
                        t.get("expected_answer"), t.get("answer_pattern"),
                        t.get("category"), t.get("difficulty"),
                        is_default, now,
                    )
                )
            except sqlite3.OperationalError:
                pass  # Table might not exist yet during first run
        db.commit()
        db.close()


def backfill_db():
    """Backfill computer_id and test_id for existing rows that are missing them."""
    import hashlib
    with db_lock:
        db = get_db()
        # ── Backfill computer_id on runs ──
        rows = db.execute("SELECT id, system_info FROM runs WHERE computer_id IS NULL AND system_info IS NOT NULL AND system_info != ''").fetchall()
        for row in rows:
            try:
                si = json.loads(row[1])
                cid = compute_computer_id(si)
                db.execute("UPDATE runs SET computer_id = ? WHERE id = ?", (cid, row[0]))
            except (json.JSONDecodeError, Exception):
                pass
        # ── Backfill test_id on test_results ──
        orphan_rows = db.execute(
            "SELECT tr.id, tr.test_name FROM test_results tr WHERE tr.test_id IS NULL"
        ).fetchall()
        for tr_id, test_name in orphan_rows:
            # Find matching test by name (take the most recent one if multiple)
            test_row = db.execute(
                "SELECT id FROM tests WHERE name = ? ORDER BY id DESC LIMIT 1", (test_name,)
            ).fetchone()
            if test_row:
                db.execute("UPDATE test_results SET test_id = ? WHERE id = ?", (test_row[0], tr_id))
        db.commit()
        db.close()


def detect_system_info():
    """Detect full system info: GPU, CPU, RAM, OS, driver. Returns dict."""
    import subprocess, platform, re

    info = {}

    # ── GPU via nvidia-smi ──
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    # Format power limit: "250.00" -> "250W", "0" -> ""
                    pl = parts[5] if len(parts) > 5 else ""
                    try:
                        pl_val = float(pl)
                        pl = f"{pl_val:g}W" if pl_val > 0 else ""
                    except ValueError:
                        pl = ""
                    gpus.append({
                        "name": parts[0],
                        "memory_mib": int(parts[1]) if parts[1].isdigit() else 0,
                        "driver": parts[2],
                        "pcie_gen": parts[3],
                        "pcie_width": parts[4],
                        "power_limit_w": pl,
                    })
            info["gpus"] = gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # ── GPU summary string ──
    if "gpus" in info and info["gpus"]:
        from collections import Counter
        gpu_counts = Counter((g["name"], g["memory_mib"]) for g in info["gpus"])
        parts = []
        for (name, mem), count in gpu_counts.items():
            mem_gb = mem // 1024
            total_mem = mem_gb * count
            if count > 1:
                parts.append(f"{count}x {name} ({total_mem}GB)")
            else:
                parts.append(f"{name} ({mem_gb}GB)")
        info["gpu_summary"] = ", ".join(parts)
    else:
        info["gpu_summary"] = ""

    # ── CPU via lscpu ──
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            cpu = {}
            for line in result.stdout.split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    k, v = k.strip(), v.strip()
                    if k == "Model name":
                        cpu["model"] = v
                    elif k == "Socket(s)":
                        cpu["sockets"] = int(v)
                    elif k == "Core(s) per socket":
                        cpu["cores_per_socket"] = int(v)
                    elif k == "Thread(s) per core":
                        cpu["threads_per_core"] = int(v)
                    elif k == "CPU(s)":
                        cpu["total_threads"] = int(v)
                    elif k == "CPU max MHz":
                        cpu["max_mhz"] = float(v)
                    elif k in ("L2 cache", "L3 cache"):
                        cpu[k.lower().replace(" ", "_")] = v
            if cpu:
                cpu["total_cores"] = cpu.get("sockets", 1) * cpu.get("cores_per_socket", 1)
                info["cpu"] = cpu
                cpu_str = cpu.get("model", "")
                if cpu.get("total_cores"):
                    cpu_str += f", {cpu['total_cores']} cores"
                if cpu.get("max_mhz"):
                    cpu_str += f", {cpu['max_mhz']:.0f}MHz max"
                info["cpu_summary"] = cpu_str
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # ── RAM ──
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["ram_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                    info["ram_summary"] = f"{info['ram_gb']}GB"
                    break
    except Exception:
        pass

    # ── OS / kernel ──
    info["os"] = platform.platform()
    info["kernel"] = platform.release()
    info["arch"] = platform.machine()
    info["hostname"] = platform.node()

    # ── nvidia driver ──
    if "gpus" in info and info["gpus"]:
        info["nvidia_driver"] = info["gpus"][0].get("driver", "")

    # ── Storage (model files disk) ──
    try:
        result = subprocess.run(
            ["df", "-h", "/mnt/nvme" if __import__("os").path.exists("/mnt/nvme") else "/"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 6:
                    info["disk"] = {"total": parts[1], "used": parts[2], "avail": parts[3], "mount": parts[5]}
    except Exception:
        pass

    # ── CUDA version from nvidia-smi ──
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "CUDA Version" in line:
                    m = re.search(r"CUDA Version:\s*(\S+)", line)
                    if m:
                        info["cuda_version"] = m.group(1)
                    break
    except Exception:
        pass

    return info


def compute_computer_id(system_info: dict) -> str:
    """Generate a short, deterministic computer ID from hardware + driver info.
    Format: <hw_prefix>-<driver_suffix>  e.g. 'a7c3f1-b92d'
    - HW prefix: 6 hex chars from SHA-256(hostname + GPU config + CPU + RAM)
    - Driver suffix: 4 hex chars from SHA-256(CUDA version + NVIDIA driver)
    Identical hardware always produces the same prefix; driver/CUDA changes only affect the suffix.
    """
    import hashlib
    from collections import Counter

    # ── HW prefix ──
    gpus = system_info.get("gpus", [])
    gpu_counts = Counter((g.get("name", "?"), g.get("memory_mib", 0)) for g in gpus)
    gpu_desc_parts = []
    for (name, mem), count in sorted(gpu_counts.items()):
        gpu_desc_parts.append(f"{name}:{mem}x{count}" if count > 1 else f"{name}:{mem}")
    gpu_str = "|".join(gpu_desc_parts)

    cpu = system_info.get("cpu", {})
    cpu_str = f"{cpu.get('model', '?')}:{cpu.get('total_cores', 0)}:{cpu.get('total_threads', 0)}"

    ram_gb = int(system_info.get("ram_gb", 0))  # round to int to avoid float jitter
    hostname = system_info.get("hostname", "unknown")

    hw_input = f"{hostname}|{gpu_str}|{cpu_str}|{ram_gb}"
    hw_hash = hashlib.sha256(hw_input.encode()).hexdigest()[:6].lower()

    # ── Driver suffix ──
    cuda = system_info.get("cuda_version", "unknown")
    driver = system_info.get("nvidia_driver", "unknown")
    drv_input = f"{cuda}|{driver}"
    drv_hash = hashlib.sha256(drv_input.encode()).hexdigest()[:4].lower()

    return f"{hw_hash}-{drv_hash}"


def save_results_to_db(results, host, gpu_info="", system_info=None):
    with db_lock:
        db = get_db()
        import json
        si_json = json.dumps(system_info) if system_info else ""
        computer_id = compute_computer_id(system_info) if system_info else "unknown-unknown"
        run_id = db.execute(
            "INSERT INTO runs (timestamp, host, gpu_info, system_info, computer_id) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), host, gpu_info, si_json, computer_id)
        ).lastrowid

        # Build test_id lookup: name + prompt_sha256 -> test_id
        all_db_tests = {}
        for row in db.execute("SELECT id, name, prompt_sha256 FROM tests").fetchall():
            all_db_tests[(row[1], row[2])] = row[0]

        for model_name, data in results.items():
            model_result_id = db.execute(
                "INSERT INTO model_results (run_id, model, is_thinking) VALUES (?, ?, ?)",
                (run_id, model_name, 1 if data.get("is_thinking") else 0)
            ).lastrowid

            for test_entry in data["tests"]:
                test_name = test_entry["name"]
                # Look up or insert test_id
                prompt_text = test_entry.get("prompt", "")
                if isinstance(prompt_text, tuple):
                    prompt_text = "".join(prompt_text)
                p_sha = hashlib.sha256(prompt_text.encode()).hexdigest()
                key = (test_name, p_sha)
                test_id = all_db_tests.get(key)
                if test_id is None:
                    # Try to find by name only (fallback for tests without prompt in entry)
                    row = db.execute("SELECT id FROM tests WHERE name = ? ORDER BY id DESC LIMIT 1", (test_name,)).fetchone()
                    test_id = row[0] if row else None

                if "error" in test_entry:
                    db.execute(
                        "INSERT INTO test_results (model_result_id, test_name, test_id, stop_reason) VALUES (?, ?, ?, ?)",
                        (model_result_id, test_name, test_id, f"ERROR: {test_entry['error']}")
                    )
                    continue

                r = test_entry["result"]
                total_dur = r.get("total_duration_ns", 0) or 0
                load_dur = r.get("load_duration_ns", 0) or 0
                prompt_dur = r.get("prompt_eval_duration_ns", 0) or 0
                eval_dur = r.get("eval_duration_ns", 0) or 0

                response_text = r.get("text", "") or ""
                if len(response_text) > 50000:
                    response_text = response_text[:50000] + "...[truncated]"

                thinking_excerpt = r.get("thinking_text", "") or ""
                if len(thinking_excerpt) > 10000:
                    thinking_excerpt = thinking_excerpt[:10000] + "...[truncated]"

                db.execute(
                    """INSERT INTO test_results (
                        model_result_id, test_name, test_id,
                        input_tokens, output_tokens, total_tokens,
                        thinking_chunks, response_chunks,
                        wall_time, ttft, ttft_response,
                        thinking_duration, response_duration,
                        total_duration, load_duration,
                        prompt_eval_duration, eval_duration,
                        tps, prompt_tps, stop_reason,
                        response_text, thinking_excerpt, quality_rating
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        model_result_id, test_name, test_id,
                        r.get("input_tokens"), r.get("output_tokens"), r.get("total_tokens"),
                        r.get("thinking_chunks"), r.get("response_chunks"),
                        r.get("wall_time"), r.get("ttft"), r.get("ttft_response"),
                        r.get("thinking_duration"), r.get("response_duration"),
                        total_dur / 1e9, load_dur / 1e9,
                        prompt_dur / 1e9, eval_dur / 1e9,
                        r.get("tps"), r.get("prompt_tps"), r.get("done_reason"),
                        response_text, thinking_excerpt, None
                    )
                )
            db.commit()
        db.close()
    return run_id


def save_ratings_to_db(run_id, ratings):
    """ratings = [(model, test_name, score), ...]"""
    with db_lock:
        db = get_db()
        for model_name, test_name, score in ratings:
            db.execute(
                """UPDATE test_results SET quality_rating = ?
                WHERE model_result_id IN (
                    SELECT id FROM model_results WHERE run_id = ? AND model = ?
                ) AND test_name = ?""",
                (score, run_id, model_name, test_name)
            )
        db.commit()
        db.close()


def get_run_data(db, run_id):
    run = db.execute("SELECT id, timestamp, host, gpu_info, system_info, computer_id FROM runs WHERE id = ?", (run_id,)).fetchone()
    if not run:
        return None
    models = db.execute(
        "SELECT id, model, is_thinking FROM model_results WHERE run_id = ?", (run_id,)
    ).fetchall()

    import json
    sys_info = json.loads(run[4]) if run[4] else {}
    result = {"run_id": run[0], "timestamp": run[1], "host": run[2], "gpu_info": run[3] or "",
              "system_info": sys_info, "computer_id": run[5] or "", "models": []}
    for mr_id, model_name, is_thinking in models:
        tests = db.execute(
            """SELECT test_name, input_tokens, output_tokens, total_tokens,
                      thinking_chunks, response_chunks,
                      wall_time, ttft, ttft_response,
                      thinking_duration, response_duration,
                      total_duration, load_duration,
                      prompt_eval_duration, eval_duration,
                      tps, prompt_tps, stop_reason,
                      response_text, thinking_excerpt, quality_rating,
                      judge_model, judge_score, judge_reasoning,
                      judge_dimensions, judge_duration,
                      math_correct, math_answer, math_expected,
                      test_id
               FROM test_results WHERE model_result_id = ?""",
            (mr_id,)
        ).fetchall()

        # Build lookup from test definitions for rating_type
        all_test_defs = load_tests()
        test_rating_type = {t["name"]: t.get("rating_type", "objective" if t.get("expected_answer") else "subjective") for t in all_test_defs}

        model_entry = {"name": model_name, "is_thinking": bool(is_thinking), "tests": []}
        for t in tests:
            tname = t[0]
            model_entry["tests"].append({
                "name": tname, "input_tokens": t[1], "output_tokens": t[2],
                "total_tokens": t[3], "thinking_chunks": t[4], "response_chunks": t[5],
                "wall_time": t[6], "ttft": t[7], "ttft_response": t[8],
                "thinking_duration": t[9], "response_duration": t[10],
                "total_duration": t[11], "load_duration": t[12],
                "prompt_eval_duration": t[13], "eval_duration": t[14],
                "tps": t[15], "prompt_tps": t[16], "stop_reason": t[17],
                "response_text": t[18], "thinking_excerpt": t[19],
                "quality_rating": t[20],
                "judge_model": t[21], "judge_score": t[22],
                "judge_reasoning": t[23], "judge_dimensions": t[24],
                "judge_duration": t[25],
                "math_correct": t[26], "math_answer": t[27], "math_expected": t[28],
                "test_id": t[29],
                "rating_type": test_rating_type.get(tname, "subjective"),
            })
        result["models"].append(model_entry)
    return result

