#!/usr/bin/env python3
"""FastAPI application — routes, lifespan, SSE endpoints."""

import sys
import json
import time
import csv
import io
import shutil
import hashlib
import os
import re
import sqlite3
import asyncio
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from contextlib import asynccontextmanager

import requests as req_lib
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from ollama_bench.config import *
from ollama_bench.state import *
from ollama_bench.ollama_api import *
from ollama_bench.db import *
from ollama_bench.tests import *
from ollama_bench.export import *
from ollama_bench.compare import *
from ollama_bench.runner import *
from ollama_bench.frontend import *
from ollama_bench.suites import *

# ═══════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app):
    global _sse_loop
    init_db()
    seed_tests_to_db()
    backfill_db()
    loop = asyncio.get_running_loop()
    _sse_loop = loop
    # Also set runner's _sse_loop so SSE broadcast works for suite/profile events
    import ollama_bench.runner as _runner_mod
    _runner_mod._sse_loop = loop
    yield

app = FastAPI(lifespan=lifespan)

from ollama_bench.runner import router as runner_router
from ollama_bench.compare import router as compare_router
app.include_router(runner_router)
app.include_router(compare_router)


# ── API Routes ──

@app.get("/api/models")
async def api_models(host: str = Query(default=DEFAULT_OLLAMA_HOST)):
    models = get_models(host)
    return JSONResponse(models)


@app.get("/api/gpu")
async def api_gpu():
    sys_info = detect_system_info()
    sys_info["computer_id"] = compute_computer_id(sys_info) if sys_info.get("gpus") else "unknown-unknown"
    return JSONResponse(sys_info)


@app.get("/api/tests")
async def api_tests():
    tests = load_tests()
    # Include full prompts and DB metadata for the tests tab
    # Also load test_id and prompt_sha256 from DB
    db_tests = {}
    with db_lock:
        db = get_db()
        for row in db.execute("SELECT id, name, prompt_sha256, rating_type FROM tests").fetchall():
            db_tests[row[1] + "|" + row[2]] = {"test_id": row[0], "prompt_sha256": row[2]}
        db.close()
    result = []
    for t in tests:
        p = t.get("prompt", "")
        if isinstance(p, tuple):
            p = "".join(p)
        sha = hashlib.sha256(p.encode()).hexdigest()
        key = t["name"] + "|" + sha
        db_info = db_tests.get(key, {})
        result.append({
            "name": t["name"], "default": t.get("default", False), "prompt": p,
            "expected_answer": t.get("expected_answer"), "rating_type": t.get("rating_type", "subjective"),
            "test_id": db_info.get("test_id"), "prompt_sha256": sha[:12],
        })
    return JSONResponse(result)


@app.post("/api/tests/save")
async def api_tests_save(request: Request):
    """Create or update a custom test. Default tests cannot be edited (only copied as new)."""
    body = await request.json()
    name = body.get("name", "").strip()
    prompt = body.get("prompt", "").strip()
    if not name or not prompt:
        return JSONResponse({"error": "Name and prompt are required"}, status_code=400)
    test_entry = {
        "name": name,
        "prompt": prompt,
        "expected_answer": body.get("expected_answer", "").strip() or None,
        "rating_type": body.get("rating_type", "subjective"),
        "default": False,
    }
    # Load existing custom tests
    custom = []
    if TESTS_FILE.exists():
        try:
            custom = json.loads(TESTS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    # Update existing or append new
    found = False
    for i, t in enumerate(custom):
        if t["name"] == name:
            custom[i] = test_entry
            found = True
            break
    if not found:
        custom.append(test_entry)
    TESTS_FILE.write_text(json.dumps(custom, indent=2, ensure_ascii=False))
    return JSONResponse({"ok": True, "name": name, "action": "updated" if found else "created"})


@app.post("/api/tests/delete")
async def api_tests_delete(request: Request):
    """Delete a custom test by name. Default tests cannot be deleted."""
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Name is required"}, status_code=400)
    # Check it's not a default test
    if any(t["name"] == name for t in DEFAULT_TESTS):
        return JSONResponse({"error": "Cannot delete default tests"}, status_code=400)
    custom = []
    if TESTS_FILE.exists():
        try:
            custom = json.loads(TESTS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    before = len(custom)
    custom = [t for t in custom if t["name"] != name]
    if len(custom) == before:
        return JSONResponse({"error": f"Test '{name}' not found"}, status_code=404)
    TESTS_FILE.write_text(json.dumps(custom, indent=2, ensure_ascii=False))
    return JSONResponse({"ok": True, "deleted": name})


@app.get("/api/compare")
async def api_compare(
    computer_id: list[str] = Query(default=[]),
    model: list[str] = Query(default=[]),
    test: list[str] = Query(default=[])
):
    """Cross-run comparison: get all test_results for matching tests, models, computers.
    Query params:
      computer_id: one or more computer IDs to include (empty = all)
      model: one or more model names to include (empty = all)
      test: one or more test names to include (empty = all tests)
    Returns results grouped by test, with model names, metrics, and run metadata.
    """
    db = get_db()
    conditions = []
    params = []
    if computer_id:
        placeholders = ",".join("?" * len(computer_id))
        conditions.append(f"r.computer_id IN ({placeholders})")
        params.extend(computer_id)
    if model:
        placeholders = ",".join("?" * len(model))
        conditions.append(f"mr.model IN ({placeholders})")
        params.extend(model)
    if test:
        placeholders = ",".join("?" * len(test))
        conditions.append(f"tr.test_name IN ({placeholders})")
        params.extend(test)

    where = " AND ".join(conditions) if conditions else "1=1"

    rows = db.execute(f"""
        SELECT r.id, r.timestamp, r.computer_id, r.gpu_info,
               mr.model, mr.is_thinking,
               tr.test_name, tr.test_id,
               tr.input_tokens, tr.output_tokens, tr.total_tokens,
               tr.thinking_chunks, tr.response_chunks,
               tr.wall_time, tr.ttft, tr.ttft_response,
               tr.thinking_duration, tr.response_duration,
               tr.total_duration, tr.load_duration,
               tr.prompt_eval_duration, tr.eval_duration,
               tr.tps, tr.prompt_tps, tr.stop_reason,
               tr.quality_rating,
               tr.judge_model, tr.judge_score,
               tr.math_correct, tr.math_answer, tr.math_expected
        FROM test_results tr
        JOIN model_results mr ON tr.model_result_id = mr.id
        JOIN runs r ON mr.run_id = r.id
        WHERE {where}
        ORDER BY r.timestamp DESC
    """, params).fetchall()
    db.close()

    tests_map = {}
    for row in rows:
        tname = row[6]
        entry = {
            "run_id": row[0], "timestamp": row[1], "computer_id": row[2] or "", "gpu_info": row[3] or "",
            "model": row[4], "is_thinking": bool(row[5]),
            "test_name": tname, "test_id": row[7],
            "input_tokens": row[8], "output_tokens": row[9], "total_tokens": row[10],
            "thinking_chunks": row[11], "response_chunks": row[12],
            "wall_time": row[13], "ttft": row[14], "ttft_response": row[15],
            "thinking_duration": row[16], "response_duration": row[17],
            "total_duration": row[18], "load_duration": row[19],
            "prompt_eval_duration": row[20], "eval_duration": row[21],
            "tps": row[22], "prompt_tps": row[23], "stop_reason": row[24],
            "quality_rating": row[25],
            "judge_model": row[26], "judge_score": row[27],
            "math_correct": row[28], "math_answer": row[29], "math_expected": row[30],
        }
        if tname not in tests_map:
            tests_map[tname] = []
        tests_map[tname].append(entry)

    db2 = get_db()
    computer_ids = [r[0] for r in db2.execute("SELECT DISTINCT computer_id FROM runs WHERE computer_id IS NOT NULL AND computer_id != '' ORDER BY computer_id").fetchall()]
    all_model_names = [r[0] for r in db2.execute("SELECT DISTINCT model FROM model_results ORDER BY model").fetchall()]
    all_test_names = [r[0] for r in db2.execute("SELECT DISTINCT test_name FROM test_results ORDER BY test_name").fetchall()]
    db2.close()

    return JSONResponse({
        "computer_ids": computer_ids,
        "model_names": all_model_names,
        "test_names": all_test_names,
        "results": tests_map,
    })



@app.get("/api/runs")
async def api_runs():
    db = get_db()
    rows = db.execute(
        """SELECT r.id, r.timestamp, r.host, r.gpu_info, r.system_info, r.computer_id,
                  COUNT(DISTINCT mr.id) as models,
                  COUNT(tr.id) as tests
           FROM runs r
           LEFT JOIN model_results mr ON mr.run_id = r.id
           LEFT JOIN test_results tr ON tr.model_result_id = mr.id
           GROUP BY r.id
           ORDER BY r.id DESC"""
    ).fetchall()
    db.close()
    import json
    return JSONResponse([
        {"id": r[0], "timestamp": r[1], "host": r[2], "gpu_info": r[3] or "",
         "system_info": json.loads(r[4]) if r[4] else {},
         "computer_id": r[5] or "",
         "models": r[6], "tests": r[7]}
        for r in rows
    ])


@app.get("/api/runs/{run_id}")
async def api_run_detail(run_id: int):
    db = get_db()
    data = get_run_data(db, run_id)
    db.close()
    if not data:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    return JSONResponse(data)


@app.delete("/api/runs/{run_id}")
async def api_delete_run(run_id: int):
    with db_lock:
        db = get_db()
        db.execute("DELETE FROM test_results WHERE model_result_id IN (SELECT id FROM model_results WHERE run_id = ?)", (run_id,))
        db.execute("DELETE FROM model_results WHERE run_id = ?", (run_id,))
        db.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        db.commit()
        db.close()
    return JSONResponse({"deleted": run_id})


@app.post("/api/rate/{run_id}")
async def api_rate(run_id: int, request: Request):
    body = await request.json()
    # body = {"ratings": [{"model": "...", "test": "...", "score": 7}, ...]}
    ratings = body.get("ratings", [])
    save_ratings_to_db(run_id, [(r["model"], r["test"], r["score"]) for r in ratings])
    return JSONResponse({"saved": len(ratings)})


@app.post("/api/judge/{run_id}")
async def api_run_judge(run_id: int, request: Request):
    if judge_state.active:
        return JSONResponse({"error": "Judge already running"}, status_code=409)

    body = await request.json()
    ollama_host = body.get("host", DEFAULT_OLLAMA_HOST)
    judge_model = body.get("judge_model", "")
    if not judge_model:
        return JSONResponse({"error": "judge_model required"}, status_code=400)

    db = get_db()
    data = get_run_data(db, run_id)
    db.close()
    if not data:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    total = sum(len(m["tests"]) for m in data["models"])
    judge_state.active = True
    judge_state.progress = 0
    judge_state.total = total
    judge_state.run_id = run_id

    def judge_thread():
        db = get_db()
        progress = 0
        for model_data in data["models"]:
            for test_data in model_data["tests"]:
                judge_state.current_model = model_data["name"]
                judge_state.current_test = test_data["name"]
                progress += 1
                judge_state.progress = progress

                # Notify SSE
                for q in judge_state.event_queues:
                    try:
                        q.put_nowait(json.dumps({"type": "judge_progress", "model": model_data["name"], "test": test_data["name"], "progress": progress, "total": total}))
                    except:
                        pass

                # Only judge tests with response text
                response_text = test_data.get("response_text", "") or ""
                if not response_text or (test_data.get("stop_reason") or "").startswith("ERROR"):
                    continue

                # Find the original prompt and rating_type
                all_tests = load_tests()
                prompt = ""
                rating_type = "subjective"
                for t in all_tests:
                    if t["name"] == test_data["name"]:
                        prompt = t["prompt"]
                        rating_type = t.get("rating_type", "objective" if t.get("expected_answer") else "subjective")
                        break

                if not prompt:
                    continue

                # Skip LLM judging for numerical-only tests (pure speed metrics)
                if rating_type == "numerical":
                    # Still do math verification if applicable
                    expected_answer = None
                    is_correct = False
                    extracted = None
                    for t in all_tests:
                        if t["name"] == test_data["name"]:
                            expected_answer = t.get("expected_answer")
                            break
                    if expected_answer:
                        is_correct, extracted = extract_math_answer(response_text, expected_answer)
                        with db_lock:
                            db.execute(
                                """UPDATE test_results SET math_correct=?, math_answer=?, math_expected=?
                                   WHERE model_result_id IN (
                                       SELECT id FROM model_results WHERE run_id=? AND model=?
                                   ) AND test_name=?""",
                                (1 if is_correct else 0, extracted, expected_answer,
                                 run_id, model_data["name"], test_data["name"])
                            )
                            db.commit()
                    # Notify SSE for progress
                    for q in judge_state.event_queues:
                        try:
                            q.put_nowait(json.dumps({
                                "type": "judge_done",
                                "model": model_data["name"],
                                "test": test_data["name"],
                                "judge_score": None,
                                "math_correct": 1 if expected_answer and is_correct else (0 if expected_answer else None),
                                "math_answer": extracted if expected_answer else None,
                                "math_expected": expected_answer,
                            }))
                        except:
                            pass
                    continue

                # Run judge
                judge_result = judge_response(ollama_host, judge_model, prompt, response_text)

                # Also do math verification if applicable
                expected_answer = None
                for t in all_tests:
                    if t["name"] == test_data["name"]:
                        expected_answer = t.get("expected_answer")
                        break

                math_correct = None
                math_answer_str = None
                if expected_answer:
                    is_correct, extracted = extract_math_answer(response_text, expected_answer)
                    math_correct = 1 if is_correct else 0
                    math_answer_str = extracted

                # Save to DB
                with db_lock:
                    db.execute(
                        """UPDATE test_results SET judge_model=?, judge_score=?, judge_reasoning=?,
                           judge_dimensions=?, judge_duration=?,
                           math_correct=?, math_answer=?, math_expected=?
                           WHERE model_result_id IN (
                               SELECT id FROM model_results WHERE run_id=? AND model=?
                           ) AND test_name=?""",
                        (judge_model, judge_result["score"], judge_result["reasoning"],
                         judge_result["dimensions"], judge_result["duration"],
                         math_correct, math_answer_str, expected_answer,
                         run_id, model_data["name"], test_data["name"])
                    )
                    db.commit()

                # Notify SSE
                for q in judge_state.event_queues:
                    try:
                        q.put_nowait(json.dumps({
                            "type": "judge_done",
                            "model": model_data["name"],
                            "test": test_data["name"],
                            "judge_score": judge_result["score"],
                            "math_correct": math_correct,
                            "math_answer": math_answer_str,
                            "math_expected": expected_answer,
                        }))
                    except:
                        pass

        db.close()
        judge_state.active = False
        judge_state.current_model = ""
        judge_state.current_test = ""

        for q in judge_state.event_queues:
            try:
                q.put_nowait(json.dumps({"type": "judge_complete", "run_id": run_id}))
            except:
                pass

    threading.Thread(target=judge_thread, daemon=True).start()
    return JSONResponse({"started": True, "total": total})


@app.post("/api/verify-math/{run_id}")
async def api_verify_math(run_id: int):
    db = get_db()
    data = get_run_data(db, run_id)
    if not data:
        db.close()
        return JSONResponse({"error": "Run not found"}, status_code=404)

    all_tests = load_tests()
    results = []

    for model_data in data["models"]:
        for test_data in model_data["tests"]:
            expected_answer = None
            for t in all_tests:
                if t["name"] == test_data["name"]:
                    expected_answer = t.get("expected_answer")
                    break

            if not expected_answer:
                continue

            response_text = test_data.get("response_text", "") or ""
            is_correct, extracted = extract_math_answer(response_text, expected_answer)

            math_correct = 1 if is_correct else 0
            math_answer_str = extracted

            with db_lock:
                db.execute(
                    """UPDATE test_results SET math_correct=?, math_answer=?, math_expected=?
                       WHERE model_result_id IN (
                           SELECT id FROM model_results WHERE run_id=? AND model=?
                       ) AND test_name=?""",
                    (math_correct, math_answer_str, expected_answer, run_id, model_data["name"], test_data["name"])
                )

            results.append({
                "model": model_data["name"],
                "test": test_data["name"],
                "correct": is_correct,
                "extracted": math_answer_str,
                "expected": expected_answer,
            })

    db.commit()
    db.close()
    return JSONResponse({"results": results})


@app.get("/api/judge-status")
async def api_judge_status():
    return JSONResponse({
        "active": judge_state.active,
        "current_model": judge_state.current_model,
        "current_test": judge_state.current_test,
        "progress": judge_state.progress,
        "total": judge_state.total,
    })


@app.get("/api/export/csv/{run_id}")
async def api_export_csv(run_id: int):
    content = generate_csv_content(run_id)
    if content is None:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=ollama_bench_run_{run_id}.csv"}
    )


@app.get("/api/export/pdf/{run_id}")
async def api_export_pdf(run_id: int):
    data = generate_pdf_bytes(run_id)
    if data is None:
        if not HAS_FPDF:
            return JSONResponse({"error": "fpdf2 not installed"}, status_code=500)
        return JSONResponse({"error": "Run not found"}, status_code=404)
    return StreamingResponse(
        iter([data]),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=ollama_bench_run_{run_id}.pdf"}
    )


def generate_html_content(run_id):
    """Generate a beautiful self-contained HTML benchmark report."""
    db = get_db()
    data = get_run_data(db, run_id)
    db.close()
    if not data:
        return None

    # Enrich with rating_type from test definitions
    tests_lookup = {t["name"]: t for t in load_tests()}

    si = data.get("system_info", {})
    try:
        dt = datetime.fromisoformat(data["timestamp"])
        ts_display = dt.strftime("%Y-%m-%d %H:%M UTC")
    except:
        ts_display = data["timestamp"][:19]

    # Collect all test names in order
    all_test_names = []
    for m in data["models"]:
        for t in m["tests"]:
            if t["name"] not in all_test_names:
                all_test_names.append(t["name"])

    # Find max tps for bar chart scaling
    max_tps = max((t.get("tps", 0) or 0 for m in data["models"] for t in m["tests"] if t.get("stop_reason", "") and not t["stop_reason"].startswith("ERROR")), default=1)
    if max_tps == 0:
        max_tps = 1

    # Color palette for models
    model_colors = [
        "#4fc3f7", "#81c784", "#ffb74d", "#f06292", "#ba68c8",
        "#4db6ac", "#7986cb", "#a1887f", "#90a4ae", "#e57373",
    ]

    # Build bar chart SVG for throughput comparison
    bar_charts = ""
    for test_name in all_test_names:
        entries = []
        for i, m in enumerate(data["models"]):
            td = {t["name"]: t for t in m["tests"]}
            t = td.get(test_name)
            if t and not (t.get("stop_reason", "") or "").startswith("ERROR") and t.get("tps"):
                entries.append((m["name"], t["tps"], model_colors[i % len(model_colors)]))

        if not entries:
            continue

        bar_height = 28
        label_width = 160
        chart_width = 500
        gap = 4
        chart_height = len(entries) * (bar_height + gap)
        max_val = max(e[1] for e in entries)
        if max_val == 0:
            max_val = 1

        bars_svg = ""
        for j, (name, tps, color) in enumerate(entries):
            y = j * (bar_height + gap)
            bar_w = int((tps / max_val) * (chart_width - label_width - 60))
            short_name = name if len(name) <= 20 else name[:18] + ".."
            bars_svg += f'''<rect x="{label_width}" y="{y + 16}" width="{max(bar_w, 2)}" height="{bar_height - 18}" rx="3" fill="{color}" opacity="0.85"/>
                <text x="8" y="{y + 12}" fill="#ccc" font-size="11" font-family="system-ui,sans-serif">{_html_esc(short_name)}</text>
                <text x="{label_width + max(bar_w, 2) + 6}" y="{y + 24}" fill="#fff" font-size="11" font-family="system-ui,sans-serif" font-weight="600">{tps:.1f} tok/s</text>'''

        bar_charts += f'''<div class="chart-section">
            <h3 style="color:#4fc3f7;margin:16px 0 8px">{_html_esc(test_name)}</h3>
            <svg width="{chart_width}" height="{chart_height}" xmlns="http://www.w3.org/2000/svg" style="background:#1a1a2e;border-radius:8px">
                {bars_svg}
            </svg>
        </div>'''

    # Build quality ratings chart (radar/bar for subjective tests)
    rating_rows = ""
    for m in data["models"]:
        td = {t["name"]: t for t in m["tests"]}
        for tn in all_test_names:
            t = td.get(tn)
            if not t:
                continue
            rt = tests_lookup.get(tn, {}).get("rating_type", "subjective")
            qr = t.get("quality_rating")
            js = t.get("judge_score")
            if rt == "numerical":
                continue  # skip throughput
            if not qr and not js:
                continue
            rating_bar = ""
            if qr:
                pct = ((qr - 4) / 6) * 100  # scale 4-10 to 0-100%
                color_q = "#4fc3f7" if qr >= 7 else ("#ffb74d" if qr >= 5 else "#e57373")
                rating_bar += f'<div style="display:flex;align-items:center;gap:6px"><span style="width:60px;color:#888;font-size:.8em">Human</span><div style="flex:1;height:8px;background:#333;border-radius:4px;overflow:hidden"><div style="width:{pct}%;height:100%;background:{color_q};border-radius:4px"></div></div><span style="color:{color_q};font-size:.85em;font-weight:600">{qr}/10</span></div>'
            if js:
                pct = ((js - 4) / 6) * 100
                color_j = "#81c784" if js >= 7 else ("#ffb74d" if js >= 5 else "#e57373")
                rating_bar += f'<div style="display:flex;align-items:center;gap:6px;margin-top:2px"><span style="width:60px;color:#888;font-size:.8em">Judge</span><div style="flex:1;height:8px;background:#333;border-radius:4px;overflow:hidden"><div style="width:{pct}%;height:100%;background:{color_j};border-radius:4px"></div></div><span style="color:{color_j};font-size:.85em;font-weight:600">{js}/10</span></div>'
            mc = t.get("math_correct")
            math_badge = ""
            if mc is not None:
                math_badge = f' <span style="font-size:.7em;padding:1px 6px;border-radius:4px;background:{"#2e7d32" if mc else "#c62828"};color:#fff">{"CORRECT" if mc else "WRONG"}</span>'

            rating_rows += f'''<div style="margin:6px 0;padding:6px 10px;background:var(--bg-alt);border-radius:6px">
                <div style="font-size:.85em;font-weight:600;color:#ddd">{_html_esc(m["name"])} — {_html_esc(tn)}{math_badge}</div>
                {rating_bar}
            </div>'''

    # Build test details per model
    detail_sections = ""
    for m in data["models"]:
        test_rows = ""
        for t in m["tests"]:
            if (t.get("stop_reason") or "").startswith("ERROR"):
                test_rows += f'''<tr><td>{_html_esc(t["name"])}</td><td colspan="8" style="color:#e57373">ERROR: {_html_esc(t["stop_reason"][6:] if t["stop_reason"].startswith("ERROR:") else t["stop_reason"])}</td></tr>'''
                continue

            rt = tests_lookup.get(t["name"], {}).get("rating_type", "subjective")
            badges = ""
            if rt == "numerical":
                badges += ' <span style="font-size:.65em;padding:1px 5px;border-radius:3px;background:#2196f3;color:#fff">NUM</span>'
            elif rt == "objective":
                badges += ' <span style="font-size:.65em;padding:1px 5px;border-radius:3px;background:#ff9800;color:#fff">OBJ</span>'

            mc = t.get("math_correct")
            if mc is not None:
                badges += f' <span style="font-size:.65em;padding:1px 5px;border-radius:3px;background:{"#2e7d32" if mc else "#c62828"};color:#fff">{"PASS" if mc else "FAIL"}</span>'

            tps = t.get("tps", 0) or 0
            tps_color = "#4fc3f7" if tps >= max_tps * 0.7 else ("#ffb74d" if tps >= max_tps * 0.4 else "#e57373")

            response = t.get("response_text") or ""
            if len(response) > 500:
                response = response[:500] + "..."

            test_rows += f'''<tr>
                <td>{_html_esc(t["name"])}{badges}</td>
                <td style="color:{tps_color};font-weight:600">{tps:.1f}</td>
                <td>{t.get("output_tokens", "?") or "?"}</td>
                <td>{f'{t["wall_time"]:.1f}s' if t.get("wall_time") else "-"}</td>
                <td>{f'{t["ttft"]:.2f}s' if t.get("ttft") else "-"}</td>
                <td>{t.get("quality_rating") or "-"}</td>
                <td>{t.get("judge_score") or "-"}</td>
                <td style="max-width:300px;white-space:pre-wrap;font-size:.8em;color:#aaa">{_html_esc(response)}</td>
            </tr>'''

        detail_sections += f'''<div style="margin-bottom:24px">
            <h3 style="color:#4fc3f7;margin:0 0 8px;display:flex;align-items:center;gap:8px">
                {_html_esc(m["name"])}
                {'<span style="font-size:.7em;padding:2px 8px;border-radius:4px;background:#7c4dff;color:#fff">thinking</span>' if m.get("is_thinking") else ""}
            </h3>
            <table style="width:100%;border-collapse:collapse;font-size:.85em">
                <thead><tr style="border-bottom:2px solid #333">
                    <th style="text-align:left;padding:4px 8px;color:#888">Test</th>
                    <th style="text-align:right;padding:4px 8px;color:#888">tok/s</th>
                    <th style="text-align:right;padding:4px 8px;color:#888">Tokens</th>
                    <th style="text-align:right;padding:4px 8px;color:#888">Time</th>
                    <th style="text-align:right;padding:4px 8px;color:#888">TTFT</th>
                    <th style="text-align:right;padding:4px 8px;color:#888">Rating</th>
                    <th style="text-align:right;padding:4px 8px;color:#888">Judge</th>
                    <th style="text-align:left;padding:4px 8px;color:#888">Response</th>
                </tr></thead>
                <tbody>{test_rows}</tbody>
            </table>
        </div>'''

    # System info badges
    si_badges = ""
    if data.get("gpu_info"):
        si_badges += f'<span class="si-badge"><strong>GPU</strong> {_html_esc(data["gpu_info"])}</span>'
    if si.get("cpu_summary"):
        si_badges += f'<span class="si-badge"><strong>CPU</strong> {_html_esc(si["cpu_summary"])}</span>'
    if si.get("ram_summary"):
        si_badges += f'<span class="si-badge"><strong>RAM</strong> {_html_esc(si["ram_summary"])}</span>'
    if si.get("cuda_version"):
        si_badges += f'<span class="si-badge"><strong>CUDA</strong> {si["cuda_version"]}</span>'
    if si.get("nvidia_driver"):
        si_badges += f'<span class="si-badge"><strong>Driver</strong> {si["nvidia_driver"]}</span>'

    # Build summary table
    summary_rows = ""
    for i, m in enumerate(data["models"]):
        td = {t["name"]: t for t in m["tests"]}
        model_ratings = []
        judge_scores = []
        math_correct = 0
        math_total = 0
        cells = ""
        for tn in all_test_names:
            t = td.get(tn)
            rt = tests_lookup.get(tn, {}).get("rating_type", "subjective")
            if not t:
                cells += '<td style="text-align:center;color:#555">—</td>'
            elif (t.get("stop_reason") or "").startswith("ERROR"):
                cells += '<td style="text-align:center;color:#e57373;font-weight:600">ERR</td>'
            elif rt == "numerical":
                tps = t.get("tps", 0) or 0
                tps_color = "#4fc3f7" if tps >= max_tps * 0.7 else ("#ffb74d" if tps >= max_tps * 0.4 else "#e57373")
                tok_str = f'{t.get("output_tokens", "?") or "?"} tok'
                time_str = f'{t["wall_time"]:.1f}s' if t.get("wall_time") else "?"
                cells += f'<td style="text-align:center"><span style="color:{tps_color};font-weight:700">{tps:.1f}</span><br><span style="color:#666;font-size:.75em">{tok_str}, {time_str}</span></td>'
            else:
                tps = t.get("tps", 0) or 0
                cells += f'<td style="text-align:center"><span style="color:#ddd;font-weight:600">{tps:.1f}</span><br><span style="color:#666;font-size:.75em">{t.get("output_tokens", "?") or "?"} tok</span></td>'
                if t.get("quality_rating"):
                    model_ratings.append(t["quality_rating"])
                if t.get("judge_score"):
                    judge_scores.append(t["judge_score"])
                mc = t.get("math_correct")
                if t.get("math_expected") is not None or mc is not None:
                    math_total += 1
                    if mc:
                        math_correct += 1

        avg_r = f'{sum(model_ratings)/len(model_ratings):.1f}' if model_ratings else '—'
        avg_j = f'{sum(judge_scores)/len(judge_scores):.1f}' if judge_scores else '—'
        math_str = f'{math_correct}/{math_total}' if math_total else '—'
        think = ' <span style="font-size:.65em;padding:1px 5px;border-radius:3px;background:#7c4dff;color:#fff">thinking</span>' if m.get("is_thinking") else ""
        summary_rows += f'''<tr>
            <td style="font-weight:600;color:#ddd">{_html_esc(m["name"])}{think}</td>
            {cells}
            <td style="text-align:center;font-weight:600;color:#4fc3f7">{avg_r}</td>
            <td style="text-align:center;font-weight:600;color:#81c784">{avg_j}</td>
            <td style="text-align:center;font-weight:600;color:#ffb74d">{math_str}</td>
        </tr>'''

    summary_headers = "".join(f'<th style="text-align:center;padding:6px 8px;color:#888;font-size:.85em">{_html_esc(tn)}</th>' for tn in all_test_names)

    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ollama Benchmark #{run_id}</title>
<style>
  :root {{ --bg: #0f0f1a; --bg-alt: #1a1a2e; --text: #ddd; --muted: #888; --accent: #4fc3f7; --border: #333; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--text); font-family: system-ui, -apple-system, sans-serif; padding: 24px; max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #fff; font-size: 1.8em; margin-bottom: 8px; }}
  h2 {{ color: var(--accent); font-size: 1.3em; margin: 24px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  .si-badges {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }}
  .si-badge {{ background: var(--bg-alt); border: 1px solid var(--border); border-radius: 6px; padding: 4px 12px; font-size: .85em; }}
  .si-badge strong {{ color: var(--accent); margin-right: 4px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th, td {{ padding: 6px 8px; border-bottom: 1px solid #222; }}
  thead th {{ border-bottom: 2px solid #444; }}
  tbody tr:hover {{ background: rgba(79,195,247,.05); }}
  .chart-section {{ margin-bottom: 16px; }}
  footer {{ margin-top: 40px; padding-top: 12px; border-top: 1px solid var(--border); color: var(--muted); font-size: .8em; }}
</style>
</head><body>

<h1>Ollama Benchmark Report</h1>
<div style="color:var(--muted);margin-bottom:12px">
  <strong>Run #{data["run_id"]}</strong> — {ts_display} — {_html_esc(data["host"])}
</div>
<div class="si-badges">{si_badges}</div>

<h2>Summary</h2>
<table><thead><tr>
  <th style="text-align:left;padding:6px 8px;color:#888">Model</th>
  {summary_headers}
  <th style="text-align:center;padding:6px 8px;color:#888">Avg Rating</th>
  <th style="text-align:center;padding:6px 8px;color:#888">Judge</th>
  <th style="text-align:center;padding:6px 8px;color:#888">Math</th>
</tr></thead><tbody>{summary_rows}</tbody></table>

<h2>Throughput Comparison</h2>
{bar_charts if bar_charts else '<p style="color:#666">No throughput data.</p>'}

{"<h2>Quality Ratings</h2>" + rating_rows if rating_rows else ""}

<h2>Detailed Results</h2>
{detail_sections}

<footer>Generated by Ollama Benchmark v5.2 &mdash; {_html_esc(ts_display)}</footer>
</body></html>'''
    return html


def _html_esc(s):
    """Escape HTML special characters."""
    if not s:
        return ""
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&#39;"))


@app.get("/api/export/html/{run_id}")
async def api_export_html(run_id: int):
    html = generate_html_content(run_id)
    if html is None:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    return HTMLResponse(content=html, headers={
        "Content-Disposition": f"inline; filename=ollama_bench_run_{run_id}.html"
    })



# Main
# ═══════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ollama Benchmark v4 - Web UI")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST, help="Ollama API host")
    parser.add_argument("--port", type=int, default=8114, help="Web server port")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path")
    parser.add_argument("--web-host", default="0.0.0.0", help="Web server bind host")
    args = parser.parse_args()

    import ollama_bench.db as _db_mod
    _db_mod._cfg["db_path"] = args.db
    init_db()
    seed_tests_to_db()
    backfill_db()

    print(f"  Ollama Benchmark v4 - Web UI")
    print(f"  Ollama host: {args.ollama_host}")
    print(f"  Database:    {args.db}")
    print(f"  Web UI:      http://localhost:{args.port}")
    print()

    import uvicorn
    uvicorn.run(app, host=args.web_host, port=args.port, log_level="info")