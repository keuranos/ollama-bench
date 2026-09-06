#!/usr/bin/env python3
"""CSV, PDF, HTML report generation."""

import csv
import json
import io
from datetime import datetime

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

from ollama_bench.db import get_db, get_run_data

# ═══════════════════════════════════════════════

def generate_csv_content(run_id):
    db = get_db()
    data = get_run_data(db, run_id)
    db.close()
    if not data:
        return None

    import io
    output = io.StringIO()
    writer = csv.writer(output)
    # Metadata rows
    writer.writerow(["# Run", data["run_id"], "Host", data["host"], "GPU", data.get("gpu_info", ""), "Time", data["timestamp"]])
    si = data.get("system_info", {})
    si_row = ["# System"]
    if si.get("cpu_summary"): si_row.extend(["CPU", si["cpu_summary"]])
    if si.get("ram_summary"): si_row.extend(["RAM", si["ram_summary"]])
    if si.get("cuda_version"): si_row.extend(["CUDA", si["cuda_version"]])
    if si.get("nvidia_driver"): si_row.extend(["Driver", si["nvidia_driver"]])
    if len(si_row) > 1: writer.writerow(si_row)
    writer.writerow([
        "model", "is_thinking", "test",
        "input_tokens", "output_tokens", "total_tokens",
        "wall_time_s", "ttft_s", "ttft_response_s",
        "thinking_duration_s", "response_duration_s",
        "total_duration_s", "load_duration_s",
        "prompt_eval_duration_s", "eval_duration_s",
        "tps", "prompt_tps", "stop_reason",
        "quality_rating_4_10",
        "judge_model", "judge_score", "judge_reasoning", "judge_dimensions", "judge_duration",
        "math_correct", "math_answer", "math_expected",
    ])
    for model in data["models"]:
        for t in model["tests"]:
            writer.writerow([
                model["name"], model["is_thinking"], t["name"],
                t["input_tokens"], t["output_tokens"], t["total_tokens"],
                f"{t['wall_time']:.2f}" if t["wall_time"] else "",
                f"{t['ttft']:.2f}" if t["ttft"] else "",
                f"{t['ttft_response']:.2f}" if t["ttft_response"] else "",
                f"{t['thinking_duration']:.2f}" if t["thinking_duration"] else "",
                f"{t['response_duration']:.2f}" if t["response_duration"] else "",
                f"{t['total_duration']:.2f}" if t["total_duration"] else "",
                f"{t['load_duration']:.2f}" if t["load_duration"] else "",
                f"{t['prompt_eval_duration']:.2f}" if t["prompt_eval_duration"] else "",
                f"{t['eval_duration']:.2f}" if t["eval_duration"] else "",
                t["tps"], t["prompt_tps"], t["stop_reason"],
                t["quality_rating"] if t["quality_rating"] else "",
                t.get("judge_model", ""), t.get("judge_score", ""),
                t.get("judge_reasoning", ""), t.get("judge_dimensions", ""),
                t.get("judge_duration", ""),
                t.get("math_correct", ""), t.get("math_answer", ""),
                t.get("math_expected", ""),
            ])
    return output.getvalue()


def _sanitize_pdf(text):
    if text is None:
        return ""
    replacements = {
        "\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2605": "*",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_pdf_bytes(run_id):
    if not HAS_FPDF:
        return None
    db = get_db()
    data = get_run_data(db, run_id)
    db.close()
    if not data:
        return None

    import warnings, io
    # fpdf2 v1.x uses ln=1, v2.7.8+ uses new_x/new_y
    try:
        import fpdf as _fpdf_mod
        _ver_str = getattr(_fpdf_mod, '__version__', '0.0.0')
        _fpdf_ver = tuple(int(x) for x in _ver_str.split('.')[:3])
        _fpdf_new = _fpdf_ver >= (2, 7, 8)
    except Exception:
        _fpdf_new = False

    def pdf_cell(pdf, w, h, txt, **kwargs):
        """Compatibility wrapper: uses new_x/new_y on fpdf2 >=2.7.8, ln on older."""
        align = kwargs.get('align', '')
        border = kwargs.get('border', 0)
        center = kwargs.get('center', False)
        if _fpdf_new:
            return pdf.cell(w, h, txt, new_x="LMARGIN", new_y="NEXT", align=align, border=border)
        else:
            return pdf.cell(w, h, txt, align=align, border=border, ln=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pdf = FPDF(orientation="L", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf_cell(pdf, 0, 12, "Ollama Model Benchmark", align="C")
        pdf.set_font("Helvetica", "", 10)
        try:
            dt = datetime.fromisoformat(data["timestamp"])
            ts_display = dt.strftime("%Y-%m-%d %H:%M UTC")
        except:
            ts_display = data["timestamp"][:19]
        pdf_cell(pdf, 0, 6, _sanitize_pdf(f"Run #{data['run_id']} -- {ts_display} -- {data['host']}"), align="C")
        if data.get("computer_id"):
            pdf_cell(pdf, 0, 6, _sanitize_pdf(f"Computer: {data['computer_id']}"), align="C")
        if data.get("gpu_info"):
            pdf_cell(pdf, 0, 6, _sanitize_pdf(f"GPU: {data['gpu_info']}"), align="C")
        si = data.get("system_info", {})
        if si:
            si_parts = []
            if si.get("cpu_summary"): si_parts.append(f"CPU: {si['cpu_summary']}")
            if si.get("ram_summary"): si_parts.append(f"RAM: {si['ram_summary']}")
            if si.get("cuda_version"): si_parts.append(f"CUDA: {si['cuda_version']}")
            if si.get("nvidia_driver"): si_parts.append(f"Driver: {si['nvidia_driver']}")
            if si_parts:
                pdf_cell(pdf, 0, 6, _sanitize_pdf(" | ".join(si_parts)), align="C")
        pdf.ln(5)

        all_test_names = []
        for model in data["models"]:
            for t in model["tests"]:
                if t["name"] not in all_test_names:
                    all_test_names.append(t["name"])

        model_w, test_w, rating_w, row_h = 45, 24, 18, 7
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(model_w, row_h, "Model", border=1)
        for tn in all_test_names:
            pdf.cell(test_w, row_h, tn[:8], border=1, align="C")
        pdf.cell(rating_w, row_h, "AvgQual", border=1, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for model in data["models"]:
            name = model["name"][:22] if len(model["name"]) > 22 else model["name"]
            think_tag = " *" if model["is_thinking"] else ""
            pdf.cell(model_w, row_h, _sanitize_pdf(name + think_tag), border=1)
            test_dict = {t["name"]: t for t in model["tests"]}
            ratings = []
            for tn in all_test_names:
                t = test_dict.get(tn)
                if t is None:
                    pdf.cell(test_w, row_h, "--", border=1, align="C")
                elif t.get("stop_reason", "").startswith("ERROR"):
                    pdf.cell(test_w, row_h, "ERR", border=1, align="C")
                else:
                    pdf.cell(test_w, row_h, f"{t['tps']:.1f}", border=1, align="C")
                    if t.get("quality_rating"):
                        ratings.append(t["quality_rating"])
            avg_q = sum(ratings) / len(ratings) if ratings else 0
            q_str = f"{avg_q:.1f}" if ratings else "--"
            pdf.cell(rating_w, row_h, _sanitize_pdf(q_str), border=1, align="C")
            pdf.ln()

        # Quality ratings detail
        rated = [(m["name"], t["name"], t["quality_rating"])
                 for m in data["models"] for t in m["tests"]
                 if t.get("quality_rating")]
        if rated:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf_cell(pdf, 0, 10, "Quality Ratings (4-10)")
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(50, 7, "Model", border=1)
            pdf.cell(30, 7, "Test", border=1, align="C")
            pdf.cell(25, 7, "Rating", border=1, align="C")
            pdf.ln()
            pdf.set_font("Helvetica", "", 9)
            for model_name, test_name, rating in rated:
                pdf.cell(50, 6, _sanitize_pdf(model_name[:30]), border=1)
                pdf.cell(30, 6, _sanitize_pdf(test_name), border=1, align="C")
                pdf.cell(25, 6, str(rating), border=1, align="C")
                pdf.ln()

        # fpdf2 v1.x: output(dest='S') returns bytes; v2.x: output(filelike) works
        if _fpdf_new:
            buf = io.BytesIO()
            pdf.output(buf)
            return buf.getvalue()
        else:
            return pdf.output(dest='S').encode('latin-1')

