#!/usr/bin/env python3
"""Multi-run comparison logic and export endpoints."""

import json
import io
import csv
from datetime import datetime

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

from ollama_bench.db import get_db, get_run_data, detect_system_info, save_results_to_db
from ollama_bench.export import generate_csv_content, _sanitize_pdf
from ollama_bench.config import DEFAULT_OLLAMA_HOST
from ollama_bench.ollama_api import stream_generate_chunks
from ollama_bench.state import bench_state
from ollama_bench.tests import load_tests


from fastapi import Body, APIRouter, Request

router = APIRouter()

@router.post("/api/compare/export/csv")
async def api_compare_export_csv(body: dict = Body(...)):
    results = body.get("results", {})
    if not results:
        return JSONResponse({"error": "No results"}, status_code=400)
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "test_name", "model", "computer_id", "run_id", "timestamp",
        "tps", "prompt_tps", "wall_time", "ttft",
        "quality_rating", "judge_score", "math_correct",
        "input_tokens", "output_tokens", "total_tokens", "stop_reason",
    ])
    for test_name, entries in results.items():
        for e in entries:
            writer.writerow([
                test_name, e.get("model",""), e.get("computer_id",""), e.get("run_id",""), e.get("timestamp",""),
                e.get("tps",""), e.get("prompt_tps",""), e.get("wall_time",""), e.get("ttft",""),
                e.get("quality_rating",""), e.get("judge_score",""), "1" if e.get("math_correct") is True else ("0" if e.get("math_correct") is False else ""),
                e.get("input_tokens",""), e.get("output_tokens",""), e.get("total_tokens",""), e.get("stop_reason",""),
            ])
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ollama_bench_compare.csv"}
    )


@router.post("/api/compare/export/pdf")
async def api_compare_export_pdf(body: dict = Body(...)):
    if not HAS_FPDF:
        return JSONResponse({"error": "fpdf2 not installed"}, status_code=500)
    results = body.get("results", {})
    metric = body.get("metric", "tps")
    if not results:
        return JSONResponse({"error": "No results"}, status_code=400)

    import warnings, io
    try:
        import fpdf as _fpdf_mod
        _ver_str = getattr(_fpdf_mod, '__version__', '0.0.0')
        _fpdf_ver = tuple(int(x) for x in _ver_str.split('.')[:3])
        _fpdf_new = _fpdf_ver >= (2, 7, 8)
    except Exception:
        _fpdf_new = False

    def pdf_cell(pdf, w, h, txt, **kwargs):
        align = kwargs.get('align', '')
        border = kwargs.get('border', 0)
        if _fpdf_new:
            return pdf.cell(w, h, txt, new_x="LMARGIN", new_y="NEXT", align=align, border=border)
        else:
            return pdf.cell(w, h, txt, align=align, border=border, ln=1)

    pdf.set_auto_page_break(auto=True, margin=15)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pdf = FPDF(orientation="L", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf_cell(pdf, 0, 12, "Ollama Benchmark – Comparison", align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf_cell(pdf, 0, 6, f"Metric: {metric}", align="C")
        pdf.ln(5)

        col_w = 36
        row_h = 7
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(col_w, row_h, "Test", border=1)
        pdf.cell(col_w, row_h, "Model", border=1)
        pdf.cell(28, row_h, "Computer", border=1)
        pdf.cell(18, row_h, "Run", border=1, align="C")
        pdf.cell(22, row_h, "tok/s", border=1, align="C")
        pdf.cell(22, row_h, "Wall t", border=1, align="C")
        pdf.cell(22, row_h, "TTFT", border=1, align="C")
        pdf.cell(22, row_h, "Rating", border=1, align="C")
        pdf.cell(22, row_h, "Judge", border=1, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for test_name, entries in results.items():
            for e in entries:
                pdf.cell(col_w, row_h, _sanitize_pdf(test_name[:20]), border=1)
                pdf.cell(col_w, row_h, _sanitize_pdf(e.get("model","")[:20]), border=1)
                pdf.cell(28, row_h, _sanitize_pdf(e.get("computer_id","")[:14]), border=1)
                pdf.cell(18, row_h, str(e.get("run_id","")), border=1, align="C")
                tps = e.get("tps")
                pdf.cell(22, row_h, f"{tps:.1f}" if tps is not None else "-", border=1, align="C")
                wt = e.get("wall_time")
                pdf.cell(22, row_h, f"{wt:.1f}s" if wt is not None else "-", border=1, align="C")
                tt = e.get("ttft")
                pdf.cell(22, row_h, f"{tt:.2f}s" if tt is not None else "-", border=1, align="C")
                qr = e.get("quality_rating")
                pdf.cell(22, row_h, f"{qr}/10" if qr is not None else "-", border=1, align="C")
                js = e.get("judge_score")
                pdf.cell(22, row_h, f"{js}/10" if js is not None else "-", border=1, align="C")
                pdf.ln()

        if _fpdf_new:
            buf = io.BytesIO()
            pdf.output(buf)
            data = buf.getvalue()
        else:
            data = pdf.output(dest='S').encode('latin-1')

    return StreamingResponse(
        iter([data]),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=ollama_bench_compare.pdf"}
    )


@router.post("/api/compare/export/html")
async def api_compare_export_html(body: dict = Body(...)):
    results = body.get("results", {})
    metric = body.get("metric", "tps")
    if not results:
        return JSONResponse({"error": "No results"}, status_code=400)

    colors = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292", "#ba68c8",
              "#4db6ac", "#7986cb", "#a1887f", "#90a4ae", "#e57373"]
    def color_for(cid):
        h = hashlib.md5(str(cid).encode()).hexdigest()
        return colors[int(h, 16) % len(colors)]

    rows_html = ""
    for test_name, entries in results.items():
        for e in entries:
            tps = e.get("tps")
            wt = e.get("wall_time")
            tt = e.get("ttft")
            qr = e.get("quality_rating")
            js = e.get("judge_score")
            mc = e.get("math_correct")
            mc_str = "&#10003;" if mc is True else ("&#10007;" if mc is False else "-")
            thinking_tag = ' <span style="font-size:.75em;color:var(--accent2);">thinking</span>' if e.get('is_thinking') else ''
            tps_str = f"{tps:.1f}" if tps is not None else "-"
            wt_str = f"{wt:.1f}s" if wt is not None else "-"
            tt_str = f"{tt:.2f}s" if tt is not None else "-"
            qr_str = f"{qr}/10" if qr is not None else "-"
            js_str = f"{js}/10" if js is not None else "-"
            rows_html += (
                f"<tr><td>{test_name}</td><td><strong>{e.get('model','')}</strong>{thinking_tag}"
                f"</td><td style='font-family:monospace;font-size:.85em;'>{e.get('computer_id','') or '-'}</td>"
                f"<td>#{e.get('run_id','')}</td>"
                f"<td>{tps_str}</td>"
                f"<td>{wt_str}</td>"
                f"<td>{tt_str}</td>"
                f"<td>{qr_str}</td>"
                f"<td>{js_str}</td>"
                f"<td>{mc_str}</td></tr>"
            )

    chart_svg = ""
    all_entries = []
    for test_name, entries in results.items():
        for e in entries:
            all_entries.append((test_name, e))
    if all_entries:
        tests = list(results.keys())
        bar_w = 16
        gap = 4
        group_gap = 24
        max_val = 0.0001
        for _, e in all_entries:
            v = e.get(metric)
            if v is not None and v > max_val:
                max_val = v
        h = 320
        chart_w = max(600, len(all_entries) * (bar_w + gap) + len(tests) * group_gap + 80)
        chart_h = h + 60
        svg_bars = ""
        x = 50
        y0 = h - 20
        for t in tests:
            entries = results.get(t, [])
            for e in entries:
                v = e.get(metric)
                bh = (v / max_val * (h - 80)) if v is not None else 0
                color = color_for(e.get("computer_id",""))
                svg_bars += f'<rect x="{x}" y="{y0 - bh}" width="{bar_w}" height="{bh}" fill="{color}" rx="2"/>'
                if v is not None:
                    svg_bars += f'<text x="{x + bar_w/2}" y="{y0 - bh - 4}" font-size="9" text-anchor="middle" fill="#222">{v:.1f}</text>'
                x += bar_w + gap
            mid = x - (len(entries) * (bar_w + gap)) / 2 - gap / 2
            svg_bars += f'<text x="{mid}" y="{h}" font-size="11" text-anchor="middle" fill="#444">{t}</text>'
            x += group_gap

        chart_svg = (
            f'<svg viewBox="0 0 {chart_w} {chart_h}" style="width:100%;height:auto;background:#fff;border-radius:8px;border:1px solid #ddd;">'
            f'{svg_bars}</svg>'
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Ollama Benchmark Compare</title>
<style>
:root {{ --bg:#0f1115; --card:#181a20; --border:#222; --text:#e0e0e0; --muted:#888; --accent:#4fc3f7; --accent2:#4ecdc4; --danger:#ff6b6b; --success:#51cf66; }}
body {{ background:#f5f5f5; color:#333; font-family:system-ui,-apple-system,sans-serif; margin:24px; }}
h1 {{ color:var(--accent); }}
table {{ width:100%; border-collapse:collapse; margin-top:16px; background:#fff; border-radius:8px; overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,.1); }}
th {{ background:#222; color:#fff; font-size:.8em; text-transform:uppercase; padding:8px 12px; border-bottom:1px solid #ddd; }}
td {{ padding:8px 12px; font-size:.9em; border-bottom:1px solid #eee; }}
tr:hover {{ background:#fafafa; }}
</style></head><body>
<h1>Ollama Benchmark – Comparison</h1>
<p style="color:#666;">Metric: <strong>{metric}</strong></p>
{chart_svg}
<table><thead><tr><th>Test</th><th>Model</th><th>Computer</th><th>Run</th><th>tok/s</th><th>Wall Time</th><th>TTFT</th><th>Rating</th><th>Judge</th><th>Math</th></tr></thead><tbody>
{rows_html}
</tbody></table></body></html>"""

    return HTMLResponse(content=html, headers={
        "Content-Disposition": "inline; filename=ollama_bench_compare.html"
    })


@router.get("/api/status")
async def api_status():
    return JSONResponse({
        "active": bench_state.active,
        "current_model": bench_state.current_model,
        "current_test": bench_state.current_test,
        "progress": bench_state.progress,
        "total": bench_state.total,
        "run_id": bench_state.run_id,
    })


@router.post("/api/benchmark")
async def api_start_benchmark(request: Request):
    if bench_state.active:
        return JSONResponse({"error": "Benchmark already running"}, status_code=409)

    body = await request.json()
    ollama_host = body.get("host", DEFAULT_OLLAMA_HOST)
    selected_models = body.get("models", [])
    selected_test_names = body.get("tests", [])
    gpu_override = body.get("gpu_override", "")

    all_tests = load_tests()
    selected_tests = [t for t in all_tests if t["name"] in selected_test_names]
    if not selected_tests:
        selected_tests = [t for t in all_tests if t.get("default")]

    bench_state.reset()
    bench_state.active = True
    bench_state.total = len(selected_models) * len(selected_tests)
    bench_state.results = {}

    # Run benchmark in background thread
    def run_thread():
        stopped = False
        for model_name in selected_models:
            if bench_state.stop_requested:
                stopped = True
                break
            bench_state.current_model = model_name
            bench_state.results[model_name] = {
                "is_thinking": False,
                "tests": [],
            }

            # Warmup
            if bench_state.stop_requested:
                stopped = True
                break
            try:
                wr = stream_generate_chunks(ollama_host, model_name, "Hi.", timeout=180, stop_check=lambda: bench_state.stop_requested)
                warmup_metrics = None
                for chunk in wr:
                    if bench_state.stop_requested:
                        stopped = True
                        break
                    if chunk["type"] == "done":
                        warmup_metrics = chunk["data"]
                if stopped:
                    break
                is_thinking = warmup_metrics.get("has_thinking", False) if warmup_metrics else False
                bench_state.results[model_name]["is_thinking"] = is_thinking
            except Exception:
                pass

            for test in selected_tests:
                if bench_state.stop_requested:
                    stopped = True
                    break
                bench_state.current_test = test["name"]
                bench_state.progress += 1

                # Notify SSE subscribers
                for q in bench_state.event_queues:
                    try:
                        q.put_nowait(json.dumps({
                            "type": "progress",
                            "model": model_name,
                            "test": test["name"],
                            "progress": bench_state.progress,
                            "total": bench_state.total,
                        }))
                    except:
                        pass

                # Stream from Ollama
                full_result = {}
                for chunk in stream_generate_chunks(ollama_host, model_name, test["prompt"], stop_check=lambda: bench_state.stop_requested):
                    if bench_state.stop_requested:
                        break
                    if chunk["type"] == "error":
                        bench_state.results[model_name]["tests"].append({
                            "name": test["name"], "error": chunk["data"]
                        })
                        full_result = {"error": chunk["data"]}
                        break
                    elif chunk["type"] == "done":
                        full_result = chunk["data"]
                    # Forward SSE to subscribers
                    # We don't stream individual tokens to SSE here for simplicity
                    # The results are saved to DB and viewable after

                if "error" not in full_result:
                    bench_state.results[model_name]["tests"].append({
                        "name": test["name"], "result": full_result
                    })

                # Notify completion of this test
                for q in bench_state.event_queues:
                    try:
                        summary = {
                            "type": "test_done",
                            "model": model_name,
                            "test": test["name"],
                            "tps": full_result.get("tps", 0),
                            "output_tokens": full_result.get("output_tokens", 0),
                            "wall_time": full_result.get("wall_time", 0),
                        }
                        if "error" in full_result:
                            summary["error"] = full_result["error"]
                        q.put_nowait(json.dumps(summary))
                    except:
                        pass
            if stopped:
                break

        # Save to DB (partial results if stopped)
        if bench_state.results:
            # Remove models with no test results (warmup-only entries)
            for m in list(bench_state.results.keys()):
                if not bench_state.results[m]["tests"]:
                    del bench_state.results[m]
        if bench_state.results:
            sys_info = detect_system_info()
            gpu_info = gpu_override if gpu_override else sys_info.get("gpu_summary", "")
            run_id = save_results_to_db(bench_state.results, ollama_host, gpu_info, sys_info)
            bench_state.run_id = run_id

        bench_state.active = False
        bench_state.stop_requested = False
        bench_state.current_model = ""
        bench_state.current_test = ""

        # Notify done
        for q in bench_state.event_queues:
            try:
                evt = {"type": "done", "run_id": bench_state.run_id}
                if stopped:
                    evt["stopped"] = True
                q.put_nowait(json.dumps(evt))
            except:
                pass

    threading.Thread(target=run_thread, daemon=True).start()
    return JSONResponse({"started": True, "total": bench_state.total})


@router.post("/api/benchmark/stop")
async def api_stop_benchmark():
    """Stop the running benchmark. Saves partial results and allows starting a new one."""
    if not bench_state.active:
        return JSONResponse({"error": "No benchmark running"}, status_code=400)
    bench_state.stop_requested = True
    # Notify SSE subscribers that stop was requested
    for q in bench_state.event_queues:
        try:
            q.put_nowait(json.dumps({"type": "stopping"}))
        except:
            pass
    return JSONResponse({"stopping": True})

