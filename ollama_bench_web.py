#!/usr/bin/env python3
"""
Ollama Model Benchmark v4 — Web UI
  - FastAPI + uvicorn web server
  - Streaming model responses via SSE
  - Interactive quality rating (4-10 slider) after reading full output
  - Real-time benchmark progress
  - CSV / PDF export
  - SQLite persistence (same DB as v3 CLI)

Usage:
  python3 ollama_bench_web.py [--host OLLAMA_HOST] [--port WEB_PORT] [--db DB_PATH]
"""

import sys
from pathlib import Path

# Auto-detect venv if not installed system-wide
_VENV = Path(__file__).parent / "ollama-bench-venv"
if _VENV.exists():
    for _site in sorted((_VENV / "lib").glob("python*/site-packages")):
        sys.path.insert(0, str(_site))

import json
import time
import csv
import shutil
import hashlib
import os
import re
import sqlite3
import asyncio
import threading
from datetime import datetime, timezone
from typing import Optional
from contextlib import asynccontextmanager

import requests as req_lib
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# ═══════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
SCRIPT_DIR = Path(__file__).parent
DEFAULT_DB = SCRIPT_DIR / "ollama_bench.db"
TESTS_FILE = SCRIPT_DIR / "ollama_bench_tests.json"
TESTPACKS_DIR = SCRIPT_DIR / "testpacks"
PACK_VERSION = 1
HIGH_NUM_PREDICT = 32768
DEFAULT_TIMEOUT = 900

# ═══════════════════════════════════════════════
# Benchmark Suites (MMLU-Pro, IFEval)
# ═══════════════════════════════════════════════

BENCHMARK_SUITES = {
    "mmlu_pro": {
        "name": "MMLU-Pro",
        "description": "Massive Multitask Language Understanding — Professional (12K MCQ, 14 categories)",
        "hf_dataset": "TIGER-Lab/MMLU-Pro",
        "hf_split": "test",
        "sample_size": 20,   # questions per category
        "answer_type": "mcq", # multiple choice question
        "num_options": 10,   # A-J
        "categories": ["business","chemistry","economics","engineering","health","history","law","math","other","philosophy","physics","psychology","biology","computer science"],
    },
    "ifeval": {
        "name": "IFEval",
        "description": "Instruction Following Evaluation (541 prompts with verifiable constraints)",
        "hf_dataset": "google/IFEval",
        "hf_split": "train",
        "sample_size": 0,    # 0 = all questions
        "answer_type": "instruction",  # constraint-based checking
    },
    "bfcl_v3": {
        "name": "BFCL v3",
        "description": "Berkeley Function Calling Leaderboard v3 — tool/function calling (2,485 questions, 6 categories)",
        "hf_dataset": "teddyyyy123/bfcl_v3",
        "hf_split": "train",
        "sample_size": 3,    # per category (3 × 6 cats = 18 questions default)
        "answer_type": "function_call",  # structured JSON function call matching
        "categories": ["simple", "multiple", "parallel", "parallel_multiple", "java", "javascript"],
    },
}

def load_benchmark_dataset(suite_key):
    """Load a benchmark dataset from HuggingFace, cache as parquet locally."""
    suite = BENCHMARK_SUITES[suite_key]
    cache_dir = TESTPACKS_DIR / suite_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "data.parquet"

    if cache_file.exists():
        import pandas as pd
        df = pd.read_parquet(cache_file)
        records = df.to_dict("records")
    else:
        # Download from HuggingFace
        try:
            from datasets import load_dataset
            ds = load_dataset(suite["hf_dataset"], split=suite["hf_split"])
            df = ds.to_pandas()
            df.to_parquet(cache_file)
            records = df.to_dict("records")
        except ImportError:
            raise RuntimeError("Install `datasets` and `pyarrow`: pip install datasets pyarrow")

    # Post-load: derive category column for BFCL from id prefix
    if suite_key == "bfcl_v3":
        import re
        for row in records:
            if "category" not in row or not row["category"]:
                row_id = str(row.get("id", ""))
                # Remove trailing _N-N-N or _N patterns to get category prefix
                m = re.match(r"^(.+?)_\d", row_id)
                raw_cat = m.group(1) if m else row_id
                # Normalize: strip "live_" prefix so live_simple→simple, live_multiple→multiple, etc.
                if raw_cat.startswith("live_"):
                    raw_cat = raw_cat[5:]  # remove "live_" prefix
                row["category"] = raw_cat

    return records

def format_mmlu_question(row):
    """Format an MMLU-Pro row into a prompt string + correct answer letter."""
    question = row["question"]
    raw_options = row["options"]
    if isinstance(raw_options, list):
        options = raw_options
    elif hasattr(raw_options, "tolist"):
        # numpy array or pandas series from parquet
        options = raw_options.tolist()
    elif isinstance(raw_options, str):
        options = json.loads(raw_options)
    else:
        options = list(raw_options)
    answer_index = int(row.get("answer_index", 0))
    # MMLU-Pro answer_index is 0-based, options are A=0, B=1, etc.
    # But answer_index sometimes exceeds len(options) — fallback to first char of answer
    answer_letter = str(row.get("answer", ""))
    if answer_letter and answer_letter.isalpha():
        correct = answer_letter.upper()
    else:
        correct = chr(65 + answer_index) if answer_index < len(options) else "A"

    option_strs = []
    for i, opt in enumerate(options):
        letter = chr(65 + i)
        option_strs.append(f"({letter}) {opt}")

    prompt = f"{question}\n\n" + "\n".join(option_strs) + "\n\nAnswer with just the letter (A-J):"
    return prompt, correct

def format_ifeval_prompt(row):
    """Format an IFEval row into (prompt, instruction_ids, kwargs)."""
    return row["prompt"], row.get("instruction_id_list", []), row.get("kwargs", [])


# ─── BFCL v3 functions ──────────────────────────────────────────────────────

BFCL_CATEGORIES = ["simple", "multiple", "parallel", "parallel_multiple", "java", "javascript"]

def _sample_bfcl(all_data, per_category=3):
    """Sample N questions per BFCL category for balanced coverage."""
    import random
    random.seed(42)
    by_cat = {}
    for row in all_data:
        cat = row.get("category", "simple")
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(row)
    sampled = []
    for cat in BFCL_CATEGORIES:
        cat_data = by_cat.get(cat, [])
        n = min(per_category, len(cat_data))
        if n > 0:
            sampled.extend(random.sample(cat_data, n))
    return sampled


def _normalize_bfcl_value(val):
    """Normalize a BFCL ground truth value.

    Rules:
    - ["Paris"] → "Paris" (unwrap single-element lists)
    - ["", "celsius"] → "celsius" (optional with default)
    - ["", null] → None (don't care / optional)
    - ["Paris", "Paris, France"] → ["Paris", "Paris, France"] (any match)
    - "" → None (empty = don't care)
    - null → None
    - scalar → scalar (pass through)
    """
    if val is None:
        return None
    if isinstance(val, list):
        if len(val) == 0:
            return None
        if len(val) == 1:
            return _normalize_bfcl_value(val[0])
        # Two-element lists
        if len(val) == 2:
            first, second = val[0], val[1]
            # ["", null] = don't care
            if first == "" and second is None:
                return None
            # ["", value] = optional with default
            if first == "":
                return _normalize_bfcl_value(second)
            # [value, ""] = value with optional default
            if second == "":
                return _normalize_bfcl_value(first)
            # Both non-empty = multiple acceptable values
            return [_normalize_bfcl_value(v) for v in val]
        # Long lists: multiple acceptable values
        return [_normalize_bfcl_value(v) for v in val]
    if isinstance(val, str):
        if val == "":
            return None
        return val
    if isinstance(val, (int, float, bool)):
        return val
    return val


def _args_match(model_args, gt_args):
    """Check if model's function call args match ground truth args.

    - Model may omit optional GT params (those normalizing to None)
    - Model must NOT include extra params not in GT
    - Values compared with exact, case-insensitive, substring, numeric tolerance
    """
    if not isinstance(model_args, dict) or not isinstance(gt_args, dict):
        return False

    # Normalize GT values
    normalized_gt = {}
    for k, v in gt_args.items():
        normalized_gt[k] = _normalize_bfcl_value(v)

    # Check all model params are in GT (no extra params)
    for k in model_args:
        if k not in normalized_gt:
            return False

    # Check required GT params are present and match
    for k, gt_val in normalized_gt.items():
        if gt_val is None:
            # Optional / don't care — model can omit or include any value
            continue
        if k not in model_args:
            # Required param missing
            return False
        model_val = model_args[k]

        # Compare values
        if isinstance(gt_val, list):
            # Multiple acceptable values — model matches ANY
            if not any(_values_match(model_val, acceptable) for acceptable in gt_val):
                return False
        else:
            if not _values_match(model_val, gt_val):
                return False

    return True


def _values_match(model_val, gt_val):
    """Compare a model value against a ground truth value with flexible matching."""
    if model_val == gt_val:
        return True
    # Case-insensitive string match
    if isinstance(model_val, str) and isinstance(gt_val, str):
        if model_val.lower() == gt_val.lower():
            return True
        # Substring match (model "Paris" matches GT "Paris, France")
        if gt_val.lower() in model_val.lower() or model_val.lower() in gt_val.lower():
            return True
    # Numeric tolerance
    if isinstance(model_val, (int, float)) and isinstance(gt_val, (int, float)):
        if abs(float(model_val) - float(gt_val)) < 0.01 + 0.01 * abs(float(gt_val)):
            return True
    # String representation of numbers
    if isinstance(model_val, str) and isinstance(gt_val, (int, float)):
        try:
            if abs(float(model_val) - float(gt_val)) < 0.01 + 0.01 * abs(float(gt_val)):
                return True
        except (ValueError, TypeError):
            pass
    if isinstance(gt_val, str) and isinstance(model_val, (int, float)):
        try:
            if abs(float(gt_val) - float(model_val)) < 0.01 + 0.01 * abs(float(model_val)):
                return True
        except (ValueError, TypeError):
            pass
    return False


def check_bfcl_answer(response_text, ground_truth):
    """Check a BFCL function call answer against ground truth.

    Uses 3 parsing strategies: direct JSON parse, code block extraction, bracket match.
    Normalizes both model output and ground truth before comparison.

    Returns (correct: bool, model_answer: str, details: dict)
    """
    # Parse the model's response — try multiple strategies
    model_calls = _parse_bfcl_response(response_text)

    # Parse ground truth if it's a string
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            return False, response_text[:200], {"error": "cannot parse ground truth"}

    # Ground truth can be a list of expected call sets (multiple valid answers)
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]

    # Each element in ground_truth is a valid answer set
    # Normalize GT calls
    for i, gt_set in enumerate(ground_truth):
        if isinstance(gt_set, list):
            # List of function calls
            if _check_calls_against_gt(model_calls, gt_set):
                return True, response_text[:200], {"matched_gt_set": i}
        elif isinstance(gt_set, dict):
            # Single function call
            if _check_calls_against_gt(model_calls, [gt_set]):
                return True, response_text[:200], {"matched_gt_set": i}
        else:
            # Might be a string-keyed dict like {"func_name": {args}}
            pass

    return False, response_text[:200], {"model_calls": len(model_calls), "gt_sets": len(ground_truth)}


def _parse_python_function_calls(text):
    """Parse Python-style function calls like func_name(param="val", param2=123).
    
    Handles dotted names like hospital.locate(), multiple calls in [],
    dict/list values, and mixed quoting styles.
    Returns list of {"func_name": {"param": value}} dicts.
    """
    results = []
    # Match: word.word(key="val", ...) or word(key="val") — dots allowed in func name
    pattern = r'([\w.]+)\(([^)]*)\)'
    for m in re.finditer(pattern, text):
        func_name = m.group(1)
        args_str = m.group(2).strip()
        if not args_str:
            results.append({func_name: {}})
            continue
        args = {}
        parts = []
        current = ""
        in_quote = None
        paren_depth = 0
        bracket_depth = 0
        for ch in args_str:
            if ch in ('"', "'") and in_quote is None:
                in_quote = ch
                current += ch
            elif ch == in_quote:
                in_quote = None
                current += ch
            elif ch == '(' and not in_quote:
                paren_depth += 1
                current += ch
            elif ch == ')' and not in_quote:
                paren_depth -= 1
                current += ch
            elif ch == '{' and not in_quote:
                bracket_depth += 1
                current += ch
            elif ch == '}' and not in_quote:
                bracket_depth -= 1
                current += ch
            elif ch == ',' and not in_quote and paren_depth == 0 and bracket_depth == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            parts.append(current.strip())

        for part in parts:
            eq = part.find('=')
            if eq == -1:
                continue
            key = part[:eq].strip()
            val_str = part[eq+1:].strip()
            # Parse the value
            if (val_str.startswith('"') and val_str.endswith('"')) or \
               (val_str.startswith("'") and val_str.endswith("'")):
                val = val_str[1:-1]
            elif val_str.lower() == 'true':
                val = True
            elif val_str.lower() == 'false':
                val = False
            elif val_str.lower() == 'none':
                val = None
            else:
                try:
                    if '.' in val_str:
                        val = float(val_str)
                    else:
                        val = int(val_str)
                except ValueError:
                    # Dict/list values: try JSON parse (replace single quotes)
                    if val_str.startswith('{') or val_str.startswith('['):
                        try:
                            val = json.loads(val_str.replace("'", '"'))
                        except (json.JSONDecodeError, TypeError):
                            val = val_str
                    else:
                        val = val_str
            args[key] = val
        if args:
            results.append({func_name: args})
    return results


def _parse_bfcl_response(text):
    """Parse model response into a list of function call dicts.

    Strategies:
    1. Direct JSON parse
    2. Extract from code block (```json ... ```)
    3. Find first [ ... ] or { ... } bracket pair
    4. Parse Python-style function calls like func(param="val")
    """
    text = text.strip()

    # Strategy 1: Direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: Code block extraction
    code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if code_block:
        try:
            parsed = json.loads(code_block.group(1).strip())
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 3: Bracket matching — find first [ ... ] or { ... }
    for opening, closing in [('[', ']'), ('{', '}')]:
        start = text.find(opening)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opening:
                depth += 1
            elif text[i] == closing:
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:i+1])
                        if isinstance(parsed, list):
                            return parsed
                        if isinstance(parsed, dict):
                            return [parsed]
                    except (json.JSONDecodeError, TypeError):
                        break

    # Strategy 4: Python-style function calls
    python_calls = _parse_python_function_calls(text)
    if python_calls:
        return python_calls

    return []


def _check_calls_against_gt(model_calls, gt_calls):
    """Check if model_calls matches gt_calls element-by-element.

    Both are lists of function call dicts like {"func_name": {"param": "val"}}.
    """
    if len(model_calls) != len(gt_calls):
        return False

    for model_call, gt_call in zip(model_calls, gt_calls):
        if not isinstance(model_call, dict) or not isinstance(gt_call, dict):
            continue

        # Each call is {"function_name": {"param1": "val1", ...}}
        if len(model_call) != 1 or len(gt_call) != 1:
            # Multiple keys or unexpected format — try partial match
            all_keys_match = True
            for func_name, func_args in gt_call.items():
                if func_name not in model_call:
                    all_keys_match = False
                    break
                m_args = model_call[func_name]
                g_args = func_args if isinstance(func_args, dict) else {}
                m_args_dict = m_args if isinstance(m_args, dict) else {}
                if not _args_match(m_args_dict, g_args):
                    all_keys_match = False
                    break
            if not all_keys_match:
                return False
            continue

        # Standard format: {"func_name": {"param": "val"}}
        gt_func_name = list(gt_call.keys())[0]
        model_func_name = list(model_call.keys())[0]

        # Function name must match exactly (case-sensitive)
        if model_func_name != gt_func_name:
            return False

        gt_args = gt_call[gt_func_name]
        model_args = model_call[model_func_name]

        if not isinstance(gt_args, dict):
            gt_args = {}
        if not isinstance(model_args, dict):
            model_args = {}

        if not _args_match(model_args, gt_args):
            return False

    return True

def check_ifeval_constraints(response, instruction_ids, kwargs_list):
    """Check if the model response satisfies IFEval constraints.
    Returns list of (instruction_id, passed: bool).
    Implements the most common constraint checks.
    """
    import re as _re
    results = []
    for i, inst_id in enumerate(instruction_ids):
        kw = kwargs_list[i] if i < len(kwargs_list) else {}
        passed = _check_single_constraint(response, inst_id, kw)
        results.append((inst_id, passed))
    return results

def _check_single_constraint(text, inst_id, kw):
    """Check a single IFEval constraint. Returns True if satisfied."""
    import re as _re
    text_lower = text.lower()

    if inst_id == "punctuation:no_comma":
        return "," not in text
    elif inst_id == "punctuation:no_period":
        return "." not in text.split("\n")[-1] if text.split("\n")[-1] else "." not in text
    elif inst_id == "punctuation:no_question_mark":
        return "?" not in text
    elif inst_id == "punctuation:no_exclamation":
        return "!" not in text
    elif inst_id.startswith("length_constraints:number_words"):
        num_words = kw.get("num_words")
        relation = kw.get("relation")
        word_count = len(text.split())
        return _check_relation(word_count, relation, num_words)
    elif inst_id.startswith("length_constraints:number_sentences"):
        num_sentences = kw.get("num_sentences")
        relation = kw.get("relation")
        sentence_count = len([s for s in _re.split(r'[.!?]+', text) if s.strip()])
        return _check_relation(sentence_count, relation, num_sentences)
    elif inst_id.startswith("length_constraints:number_paragraphs"):
        num_paragraphs = kw.get("num_paragraphs")
        relation = kw.get("relation")
        para_count = len([p for p in text.split("\n\n") if p.strip()])
        return _check_relation(para_count, relation, num_paragraphs)
    elif inst_id.startswith("detectable_format:number_highlighted_sections"):
        num_highlights = kw.get("num_highlights")
        relation = kw.get("relation")
        highlight_count = len(_re.findall(r'\*[^*]+\*', text))
        return _check_relation(highlight_count, relation, num_highlights)
    elif inst_id.startswith("detectable_format:number_bullet_lists"):
        num_bullets = kw.get("num_bullets")
        relation = kw.get("relation")
        bullet_count = len(_re.findall(r'^\s*[-*•]\s', text, _re.MULTILINE))
        return _check_relation(bullet_count, relation, num_bullets)
    elif inst_id.startswith("detectable_format:title"):
        # Check for a title wrapped in double angle brackets
        return bool(_re.search(r'<<.+>>', text))
    elif inst_id.startswith("detectable_format:json_format"):
        try:
            import json as _json
            _json.loads(text.strip())
            return True
        except:
            return False
    elif inst_id.startswith("detectable_format:multiple_sections"):
        section_splitter = kw.get("section_splitter", "SECTION")
        sections = text.count(section_splitter)
        return sections >= 1
    elif inst_id.startswith("detectable_content:number_placeholders"):
        num_placeholders = kw.get("num_placeholders")
        relation = kw.get("relation")
        placeholder_count = len(_re.findall(r'\[.+?\]', text))
        return _check_relation(placeholder_count, relation, num_placeholders)
    elif inst_id.startswith("detectable_content:postscript"):
        ps_keyword = kw.get("postscript_marker", "P.S.")
        return ps_keyword.lower() in text_lower
    elif inst_id.startswith("keywords:forbidden_words"):
        forbidden = kw.get("forbidden_words", [])
        return not any(fw.lower() in text_lower for fw in forbidden)
    elif inst_id.startswith("keywords:letter_frequency"):
        # Check if certain letters appear with required frequency
        letter = kw.get("letter", "").lower()
        let_relation = kw.get("let_relation", "at least")
        let_freq = kw.get("let_frequency", 0)
        if not letter:
            return True
        actual = text_lower.count(letter)
        return _check_relation(actual, let_relation, let_freq)
    elif inst_id.startswith("keywords:sentence_frequency"):
        # Check if a keyword appears in a certain number of sentences
        keyword = kw.get("keyword", "").lower()
        sentence_relation = kw.get("relation", "at least")
        sentence_freq = kw.get("sentence_frequency", 0)
        if not keyword:
            return True
        sentences_with_kw = sum(1 for s in text.split(".") if keyword in s.lower())
        return _check_relation(sentences_with_kw, sentence_relation, sentence_freq)
    elif inst_id.startswith("change_case:"):
        if "capital" in inst_id or "uppercase" in inst_id:
            return text == text.upper()
        elif "lowercase" in inst_id:
            return text == text.lower()
        return True
    elif inst_id.startswith("startend:"):
        if "start_with" in inst_id:
            start_word = kw.get("start_word", "")
            return text.strip().startswith(start_word)
        elif "end_with" in inst_id:
            end_word = kw.get("end_word", "")
            return text.strip().endswith(end_word)
        return True
    # Default: unknown constraint — don't penalize
    return True

def _check_relation(actual, relation, target):
    """Check numeric relation constraint."""
    if target is None or actual is None:
        return True
    target = int(target) if isinstance(target, str) else target
    if relation in ("at least", "more than", "greater than", ">="):
        return actual >= target
    elif relation in ("at most", "less than", "fewer than", "<="):
        return actual <= target
    elif relation in ("exactly", "equal to", "=="):
        return actual == target
    return actual >= target  # default: at least

def extract_mcq_answer(response_text, num_options=10):
    """Extract a multiple choice answer letter from model response.
    Returns letter (A-J) or None.
    """
    import re as _re
    # Try to find a standalone letter answer like "A" or "(A)" at start or after "Answer:"
    # First, try to find the last explicit letter reference
    # Pattern: answer is the first capital letter at the start, or after "answer" keyword
    patterns = [
        r'[Aa]nswer\s*(?:is)?\s*[:\-)]?\s*\(?([A-Ja-j])\)?',
        r'(?:I choose|I select|choose|select)\s*\(?([A-Ja-j])\)?',
        r'^\s*\(?([A-Ja-j])\)?\s*[\.\,\:]',  # Letter at start of response
        r'^\s*\(?([A-Ja-j])\)?\s*$',           # Just the letter alone
        r'\(([A-Ja-j])\)',                     # (A) style anywhere
    ]
    for pat in patterns:
        matches = _re.findall(pat, response_text[:300])
        if matches:
            return matches[-1].upper()

    # Fallback: find any standalone capital letter A-J that's not part of a word
    standalone = _re.findall(r'(?<![A-Za-z])([A-J])(?![A-Za-z])', response_text[:200])
    if standalone:
        return standalone[-1]
    return None

DEFAULT_TESTS = [
    {
        "name": "Throughput",
        "prompt": (
            "Write a detailed essay about the history of computing from the abacus "
            "to modern AI. Include at least 5 major milestones and explain each one's "
            "significance in depth."
        ),
        "default": True,
        "rating_type": "numerical",  # Pure speed metric, no quality rating needed
    },
    {
        "name": "Reasoning",
        "prompt": (
            "A farmer has a fox, a chicken, and a bag of corn. He needs to cross a "
            "river in a boat that can carry only him and one item at a time. If he "
            "leaves the fox with the chicken, the fox will eat the chicken. If he "
            "leaves the chicken with the corn, the chicken will eat the corn. How "
            "can he get everything across safely? Explain your reasoning step by step."
        ),
        "default": True,
    },
    {
        "name": "Coding",
        "prompt": (
            "Write a Python function called 'merge_sorted_lists' that takes two sorted "
            "lists of integers and merges them into a single sorted list. Include type "
            "hints and a docstring. Do not use the built-in sorted() function -- "
            "implement the merge algorithm."
        ),
        "default": True,
    },
    {
        "name": "Math",
        "prompt": (
            "Solve step by step: A train travels from city A to city B at 80 km/h. "
            "On the return trip, it travels at 120 km/h. The total round trip takes "
            "5 hours. What is the distance between city A and city B?"
        ),
        "expected_answer": "240",
        "default": True,
        "rating_type": "objective",  # Has known correct answer, verified automatically
    },
    {
        "name": "Finnish",
        "prompt": "Kirjoita runo tietokoneena olemisen vaikeudesta.",
        "default": True,
    },
]

# ═══════════════════════════════════════════════
# Database (shared with v3 CLI)
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


# ═══════════════════════════════════════════════
# Ollama API
# ═══════════════════════════════════════════════

def get_models(host):
    try:
        r = req_lib.get(f"{host}/api/tags", timeout=10)
        r.raise_for_status()
        models = r.json().get("models", [])
        embed_prefixes = ("nomic-embed-text", "mxbai-embed-large")
        return [{"name": m["name"], "size": m.get("size", 0)} for m in models
                if not any(m["name"].startswith(p) for p in embed_prefixes)]
    except Exception as e:
        return []


def stream_generate_chunks(host, model, prompt, timeout=DEFAULT_TIMEOUT, num_predict=None, temperature=0.4, stop_check=None, extra_options=None, stop_sequences=None):
    """Yield SSE-style dicts: {type, data} for streaming to frontend.
    
    stop_check: optional callable returning bool — if True, abort generation and close stream.
    extra_options: optional dict merged into Ollama options (e.g. {"num_ctx": 4096}).
    stop_sequences: optional list of stop strings (e.g. ["\n\n\n", "\nQuestion"]).
    """
    url = f"{host}/api/generate"
    np = num_predict if num_predict else HIGH_NUM_PREDICT
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": np, "temperature": temperature},
    }
    if extra_options:
        payload["options"].update(extra_options)
    if stop_sequences:
        payload["stop"] = stop_sequences

    start_time = time.time()
    first_chunk_time = None
    first_thinking_time = None
    last_thinking_time = None
    first_response_time = None
    last_response_time = None

    thinking_chunks = 0
    response_chunks = 0
    thinking_text = ""
    response_text = ""

    eval_count = None
    prompt_eval_count = None
    total_duration_ns = None
    load_duration_ns = None
    prompt_eval_duration_ns = None
    eval_duration_ns = None
    done_reason = None

    try:
        with req_lib.post(url, json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                # Check stop_check before processing each line — allows skip/cancel mid-stream
                if stop_check and stop_check():
                    break
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                now = time.time()
                if first_chunk_time is None:
                    first_chunk_time = now

                if chunk.get("done"):
                    eval_count = chunk.get("eval_count")
                    prompt_eval_count = chunk.get("prompt_eval_count")
                    total_duration_ns = chunk.get("total_duration")
                    load_duration_ns = chunk.get("load_duration")
                    prompt_eval_duration_ns = chunk.get("prompt_eval_duration")
                    eval_duration_ns = chunk.get("eval_duration")
                    done_reason = chunk.get("done_reason", "")
                    final_text = chunk.get("response", "")
                    if final_text:
                        response_text += final_text
                        if first_response_time is None:
                            first_response_time = now
                        last_response_time = now
                        response_chunks += 1
                    break

                thinking = chunk.get("thinking", "")
                token_text = chunk.get("response", "")

                if thinking:
                    if first_thinking_time is None:
                        first_thinking_time = now
                    last_thinking_time = now
                    thinking_text += thinking
                    thinking_chunks += 1
                    yield {"type": "thinking", "data": thinking}

                if token_text:
                    if first_response_time is None:
                        first_response_time = now
                    last_response_time = now
                    response_text += token_text
                    response_chunks += 1
                    yield {"type": "token", "data": token_text}

    except req_lib.exceptions.Timeout:
        yield {"type": "error", "data": "TIMEOUT"}
        return
    except req_lib.exceptions.ConnectionError:
        yield {"type": "error", "data": "CONNECTION_ERROR"}
        return
    except Exception as e:
        yield {"type": "error", "data": str(e)}
        return

    end_time = time.time()
    wall_time = end_time - start_time
    has_thinking = thinking_chunks > 0

    if has_thinking and first_thinking_time:
        ttft = first_thinking_time - start_time
    elif first_response_time:
        ttft = first_response_time - start_time
    elif first_chunk_time:
        ttft = first_chunk_time - start_time
    else:
        ttft = None

    thinking_duration = (last_thinking_time - first_thinking_time) if (first_thinking_time and last_thinking_time) else None
    ttft_response = (first_response_time - start_time) if first_response_time else None
    response_duration = (last_response_time - first_response_time) if (first_response_time and last_response_time) else None

    eval_duration_s = eval_duration_ns / 1e9 if eval_duration_ns else None
    if eval_duration_s and eval_duration_s > 0 and eval_count:
        tps = eval_count / eval_duration_s
    elif wall_time > 0 and eval_count:
        tps = eval_count / wall_time
    else:
        tps = 0

    prompt_eval_s = prompt_eval_duration_ns / 1e9 if prompt_eval_duration_ns else None
    if prompt_eval_s and prompt_eval_s > 0 and prompt_eval_count:
        prompt_tps = prompt_eval_count / prompt_eval_s
    else:
        prompt_tps = 0

    total_tokens = (prompt_eval_count or 0) + (eval_count or 0)

    metrics = {
        "text": response_text.strip(),
        "thinking_text": thinking_text.strip(),
        "has_thinking": has_thinking,
        "input_tokens": prompt_eval_count,
        "output_tokens": eval_count,
        "total_tokens": total_tokens,
        "thinking_chunks": thinking_chunks,
        "response_chunks": response_chunks,
        "total_duration_ns": total_duration_ns,
        "load_duration_ns": load_duration_ns,
        "prompt_eval_duration_ns": prompt_eval_duration_ns,
        "eval_duration_ns": eval_duration_ns,
        "wall_time": round(wall_time, 2),
        "ttft": round(ttft, 2) if ttft else None,
        "ttft_response": round(ttft_response, 2) if ttft_response else None,
        "thinking_duration": round(thinking_duration, 2) if thinking_duration else None,
        "response_duration": round(response_duration, 2) if response_duration else None,
        "tps": round(tps, 1),
        "prompt_tps": round(prompt_tps, 1),
        "done_reason": done_reason or "",
    }
    yield {"type": "done", "data": metrics}


# ═══════════════════════════════════════════════
# Test management
# ═══════════════════════════════════════════════

def format_bfcl_messages(row):
    """Convert a BFCL v3 row into Ollama chat messages for /api/chat.

    Returns a list of {role, content} dicts suitable for Ollama's chat API.
    The BFCL dataset's chat_completion_input already contains properly
    formatted system+user messages with function definitions.
    """
    chat_input = row.get("chat_completion_input", [])
    functions = row.get("function", [])

    # Parse from JSON strings if needed
    if isinstance(chat_input, str):
        try:
            chat_input = json.loads(chat_input)
        except (json.JSONDecodeError, TypeError):
            pass
    if isinstance(functions, str):
        try:
            functions = json.loads(functions)
        except (json.JSONDecodeError, TypeError):
            pass

    messages = []
    if isinstance(chat_input, list) and len(chat_input) > 0:
        for msg in chat_input:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Ollama chat API uses "system", "user", "assistant"
            if role in ("system", "user", "assistant"):
                messages.append({"role": role, "content": content})
            else:
                messages.append({"role": "user", "content": content})
    else:
        # Fallback: build messages from scratch
        if functions:
            func_str = json.dumps(functions, indent=2) if isinstance(functions, list) else str(functions)
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant with access to the following functions. Use them if required:\n\n"
                           "<functions>\n" + func_str + "\n</functions>\n\n"
                           "Output ONLY a JSON array of function calls."
            })

    return messages


def format_bfcl_prompt(row):
    """Format a BFCL v3 row into a prompt string for /api/generate fallback.

    Uses Ollama's chat template format (<|im_start|>/<|im_end|>) so the
    model properly understands multi-turn messages even via generate API.
    """
    messages = format_bfcl_messages(row)
    if not messages:
        return "", row.get("ground_truth", "")

    # Build prompt using chat.ml template format
    prompt_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    # Add assistant start for the model to continue
    prompt_parts.append("<|im_start|>assistant\n")
    prompt = "\n".join(prompt_parts)

    ground_truth = row.get("ground_truth", row.get("expected", ""))
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            pass

    return prompt, ground_truth


def stream_chat_chunks(host, model, messages, timeout=DEFAULT_TIMEOUT, num_predict=None, temperature=0.0, stop_check=None, extra_options=None, stop_sequences=None):
    """Yield SSE-style dicts using Ollama's /api/chat endpoint.
    
    Same output format as stream_generate_chunks but uses the chat API,
    which properly applies the model's native chat template.
    Handles thinking tokens from reasoning models.
    """
    url = f"{host}/api/chat"
    np = num_predict if num_predict else HIGH_NUM_PREDICT
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"num_predict": np, "temperature": temperature},
    }
    if extra_options:
        payload["options"].update(extra_options)
    if stop_sequences:
        payload["stop"] = stop_sequences

    start_time = time.time()
    first_chunk_time = None
    response_text = ""
    thinking_text = ""
    thinking_chunks = 0
    response_chunks = 0
    eval_count = None
    prompt_eval_count = None
    total_duration_ns = None
    prompt_eval_duration_ns = None
    eval_duration_ns = None
    done_reason = None

    try:
        with req_lib.post(url, json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if stop_check and stop_check():
                    break
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if data.get("done"):
                    eval_count = data.get("eval_count", eval_count)
                    prompt_eval_count = data.get("prompt_eval_count", prompt_eval_count)
                    total_duration_ns = data.get("total_duration", total_duration_ns)
                    prompt_eval_duration_ns = data.get("prompt_eval_duration", prompt_eval_duration_ns)
                    eval_duration_ns = data.get("eval_duration", eval_duration_ns)
                    done_reason = data.get("done_reason", "stop")
                    break

                msg = data.get("message", {})
                content = msg.get("content", "")
                thinking = msg.get("thinking", "")

                if thinking:
                    thinking_chunks += 1
                    thinking_text += thinking
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                    yield {"type": "thinking", "data": thinking}

                if content:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                    response_chunks += 1
                    response_text += content
                    yield {"type": "token", "data": content}

    except Exception as e:
        error_msg = f"Chat API error: {e}"
        yield {"type": "error", "data": error_msg}
        return

    # Yield result (same format as stream_generate_chunks)
    wall_time = round(time.time() - start_time, 2)
    eval_duration_s = eval_duration_ns / 1e9 if eval_duration_ns else None
    if eval_duration_s and eval_duration_s > 0 and eval_count:
        tps = round(eval_count / eval_duration_s, 1)
    elif wall_time > 0 and eval_count:
        tps = round(eval_count / wall_time, 1)
    else:
        tps = 0
    prompt_eval_s = prompt_eval_duration_ns / 1e9 if prompt_eval_duration_ns else None
    if prompt_eval_s and prompt_eval_s > 0 and prompt_eval_count:
        prompt_tps = round(prompt_eval_count / prompt_eval_s, 1)
    else:
        prompt_tps = 0
    total_tokens = (prompt_eval_count or 0) + (eval_count or 0)

    result = {
        "done": True,
        "text": response_text.strip(),
        "response": response_text.strip(),
        "has_thinking": bool(thinking_text.strip()),
        "input_tokens": prompt_eval_count,
        "output_tokens": eval_count,
        "total_tokens": total_tokens,
        "thinking_chunks": thinking_chunks,
        "response_chunks": response_chunks,
        "total_duration_ns": total_duration_ns,
        "prompt_eval_duration_ns": prompt_eval_duration_ns,
        "eval_duration_ns": eval_duration_ns,
        "wall_time": wall_time,
        "tps": tps,
        "prompt_tps": prompt_tps,
        "done_reason": done_reason or "",
        "thinking_text": thinking_text.strip(),
    }
    yield {"type": "done", "data": result}


def load_tests():
    tests = list(DEFAULT_TESTS)
    if TESTS_FILE.exists():
        try:
            custom = json.loads(TESTS_FILE.read_text())
            for t in custom:
                t["default"] = False
            tests.extend(custom)
        except (json.JSONDecodeError, OSError):
            pass
    return tests


def extract_math_answer(response_text, expected_answer, answer_pattern=None):
    """Extract the numeric answer from a model response and compare to expected.
    Returns (is_correct: bool|None, extracted: str|None)
    """
    if not expected_answer or not response_text:
        return None, None

    import re as _re
    if answer_pattern:
        match = _re.search(answer_pattern, response_text)
        extracted = match.group(1) if match else None
    else:
        # Find all numbers (including decimals) in the response
        numbers = _re.findall(r'[\d]+(?:\.\d+)?', response_text)
        extracted = numbers[-1] if numbers else None

    if extracted is None:
        return None, None

    # Compare: tolerate float formatting and percentage tolerance
    try:
        expected_float = float(expected_answer)
        extracted_float = float(extracted)
        is_correct = abs(extracted_float - expected_float) < 0.01 * abs(expected_float) + 0.5
    except (ValueError, ZeroDivisionError):
        is_correct = extracted.strip() == expected_answer.strip()

    return is_correct, extracted


def judge_response(host, judge_model, prompt, response, timeout=300):
    """Send a structured evaluation request to the judge model.
    Returns dict: {score, reasoning, dimensions, duration}
    """
    judge_prompt = f"""You are an expert evaluator. Rate the quality of the following AI response on a 4-10 scale.

Original prompt:
---
{prompt}
---

AI response to evaluate:
---
{response[:4000]}
---

Rate on four dimensions (each 1-10):
- Correctness: Is the information accurate?
- Completeness: Does it fully address the prompt?
- Clarity: Is it well-structured and easy to understand?
- Relevance: Does it stay on-topic?

Then provide a composite score (4-10, where 4=poor, 7=good, 10=excellent).

Respond with ONLY this JSON format, no other text:
{{"score": <int 4-10>, "reasoning": "<brief explanation>", "dimensions": {{"correctness": <int>, "completeness": <int>, "clarity": <int>, "relevance": <int>}}}}"""

    start = time.time()
    result_list = list(stream_generate_chunks(host, judge_model, judge_prompt, timeout=timeout))
    duration = time.time() - start

    # Collect the full judge response
    judge_text = ""
    judge_error = None
    for chunk in result_list:
        if chunk["type"] == "token":
            judge_text += chunk["data"]
        elif chunk["type"] == "thinking":
            pass  # Skip thinking from judge
        elif chunk["type"] == "error":
            judge_error = chunk["data"]
        elif chunk["type"] == "done":
            judge_text += chunk["data"].get("text", "")

    if judge_error:
        return {"score": None, "reasoning": f"Judge error: {judge_error}", "dimensions": None, "duration": round(duration, 2)}

    # Parse JSON from judge response
    import re as _re
    json_match = _re.search(r'\{[^{}]*"score"[^{}]*\}', judge_text, _re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            score = int(parsed.get("score", 0))
            if score < 4: score = 4
            if score > 10: score = 10
            return {
                "score": score,
                "reasoning": parsed.get("reasoning", ""),
                "dimensions": json.dumps(parsed.get("dimensions", {})),
                "duration": round(duration, 2),
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: extract any number 4-10 from the response
    numbers = _re.findall(r'\b(\d+)\b', judge_text)
    valid = [int(n) for n in numbers if 4 <= int(n) <= 10]
    if valid:
        return {"score": valid[0], "reasoning": judge_text[:200], "dimensions": None, "duration": round(duration, 2)}

    return {"score": None, "reasoning": judge_text[:200], "dimensions": None, "duration": round(duration, 2)}


# ═══════════════════════════════════════════════
# CSV / PDF Export (shared logic)
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


# ═══════════════════════════════════════════════
# Benchmark run state (in-memory for active runs)
# ═══════════════════════════════════════════════

class BenchmarkState:
    def __init__(self):
        self.active = False
        self.stop_requested = False
        self.current_model = ""
        self.current_test = ""
        self.progress = 0
        self.total = 0
        self.run_id = None
        self.results = {}  # model_name -> {is_thinking, tests: [...]}
        self.event_queues = []  # list of asyncio.Queues for SSE subscribers

    def reset(self):
        self.active = False
        self.stop_requested = False
        self.current_model = ""
        self.current_test = ""
        self.progress = 0
        self.total = 0
        self.run_id = None
        self.results = {}
        self.event_queues = []

bench_state = BenchmarkState()


class JudgeState:
    def __init__(self):
        self.active = False
        self.current_model = ""
        self.current_test = ""
        self.progress = 0
        self.total = 0
        self.run_id = None
        self.event_queues = []

judge_state = JudgeState()


# ═══════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app):
    global _sse_loop
    init_db()
    seed_tests_to_db()
    backfill_db()
    _sse_loop = asyncio.get_running_loop()
    yield

app = FastAPI(lifespan=lifespan)


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


@app.post("/api/compare/export/csv")
async def api_compare_export_csv(request: Request):
    """Export comparison data as CSV. Accepts same JSON body as /api/compare response."""
    data = await request.json()
    results = data.get("results", {})
    import io
    buf = io.StringIO()
    buf.write("Test,Model,Computer,Run,Timestamp,Is Thinking,Input Tokens,Output Tokens,Total Tokens,Wall Time (s),TTFT (s),tok/s,Prompt tok/s,Quality Rating,Judge Score,Math Correct\n")
    for test_name, entries in results.items():
        for e in entries:
            buf.write(f'"{test_name}","{e.get("model","")}",{e.get("computer_id","")},{e.get("run_id","")},{e.get("timestamp","")},{e.get("is_thinking",False)},{e.get("input_tokens","")},{e.get("output_tokens","")},{e.get("total_tokens","")},{e.get("wall_time","")},{e.get("ttft","")},{e.get("tps","")},{e.get("prompt_tps","")},{e.get("quality_rating","")},{e.get("judge_score","")},{e.get("math_correct","")}\n')
    buf.seek(0)
    return StreamingResponse(
        iter([buf.read()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ollama_bench_compare.csv"}
    )

@app.post("/api/compare/export/html")
async def api_compare_export_html(request: Request):
    """Export comparison data as a self-contained HTML report."""
    data = await request.json()
    results = data.get("results", {})
    computer_ids = data.get("computer_ids", [])
    model_names = data.get("model_names", [])
    test_names = data.get("test_names", [])
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Ollama Benchmark Comparison</title>
<style>
@keyframes blink-cursor {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0; }} }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #7c3aed; }} h2 {{ color: #a78bfa; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
th, td {{ border: 1px solid #333; padding: 6px 10px; text-align: left; font-size: .9em; }}
th {{ background: #2a2a4a; color: #a78bfa; }}
tr:nth-child(even) {{ background: #222240; }}
.badge {{ padding: 2px 6px; border-radius: 4px; font-size: .75em; background: #7c3aed38; color: #a78bfa; }}
.meta {{ color: #888; font-size: .85em; margin: 8px 0; }}
</style></head><body>
<h1>Ollama Benchmark Comparison</h1>
<p class="meta">Computers: {', '.join(computer_ids) if computer_ids else 'All'} &nbsp;|&nbsp; Tests: {', '.join(test_names) if test_names else 'All'}</p>
"""
    for test_name, entries in results.items():
        html += f'<h2>{test_name}</h2>\n<table><thead><tr><th>Model</th><th>Computer</th><th>Run</th><th>Time</th><th>tok/s</th><th>Wall Time</th><th>TTFT</th><th>Rating</th><th>Judge</th><th>Math</th></tr></thead><tbody>\n'
        # Sort by tps desc
        for e in sorted(entries, key=lambda x: float(x.get("tps") or 0), reverse=True):
            rating = f'{e.get("quality_rating","")}/10' if e.get("quality_rating") else "-"
            judge = f'{e.get("judge_score","")}/10' if e.get("judge_score") else "-"
            math_str = "&#10003;" if e.get("math_correct") else ("&#10007;" if e.get("math_correct") is not None else "-")
            think = ' <span class="badge">thinking</span>' if e.get("is_thinking") else ""
            html += f'<tr><td><strong>{e.get("model","")}</strong>{think}</td><td style="font-family:monospace">{e.get("computer_id","-")}</td><td>#{e.get("run_id","")}</td><td>{e.get("timestamp","")[:16]}</td><td><strong>{e.get("tps","-")}</strong></td><td>{e.get("wall_time","-")}s</td><td>{e.get("ttft","-")}s</td><td>{rating}</td><td>{judge}</td><td>{math_str}</td></tr>\n'
        html += '</tbody></table>\n'

    html += '</body></html>'
    return StreamingResponse(
        iter([html]),
        media_type="text/html",
        headers={"Content-Disposition": "attachment; filename=ollama_bench_compare.html"}
    )

@app.post("/api/compare/export/pdf")
async def api_compare_export_pdf(request: Request):
    """Export comparison data as PDF. Uses fpdf2 (v1.7.2 compatible)."""
    if not HAS_FPDF:
        return JSONResponse({"error": "fpdf2 not installed"}, status_code=500)
    data = await request.json()
    results = data.get("results", {})
    
    import io as _io, warnings
    # fpdf2 v1.x compat
    try:
        import fpdf as _fpdf_mod
        _ver_str = getattr(_fpdf_mod, '__version__', '0.0.0')
        _fpdf_ver = tuple(int(x) for x in _ver_str.split('.')[:3])
        _fpdf_new = _fpdf_ver >= (2, 7, 8)
    except Exception:
        _fpdf_new = False

    def pc(pdf, w, h, txt, **kw):
        a = kw.get('align', ''); b = kw.get('border', 0)
        if _fpdf_new:
            return pdf.cell(w, h, txt, new_x="LMARGIN", new_y="NEXT", align=a, border=b)
        else:
            return pdf.cell(w, h, txt, align=a, border=b, ln=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pdf = FPDF(orientation="L", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pc(pdf, 0, 10, "Ollama Benchmark Comparison")
        pdf.set_font("Helvetica", "", 9)
        pdf.ln(2)

        model_w, comp_w, run_w, tps_w, time_w, ttft_w, rate_w, judge_w, math_w = 42, 24, 12, 18, 18, 18, 16, 16, 14
        row_h = 6

        for test_name, entries in results.items():
            # Check if we need a new page
            if pdf.get_y() > 260:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 11)
            pc(pdf, 0, 8, _sanitize_pdf(test_name), align="L")
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 7)
            pdf.cell(model_w, row_h, "Model", border=1)
            pdf.cell(comp_w, row_h, "Computer", border=1, align="C")
            pdf.cell(run_w, row_h, "Run", border=1, align="C")
            pdf.cell(tps_w, row_h, "tok/s", border=1, align="C")
            pdf.cell(time_w, row_h, "Wall(s)", border=1, align="C")
            pdf.cell(ttft_w, row_h, "TTFT(s)", border=1, align="C")
            pdf.cell(rate_w, row_h, "Rating", border=1, align="C")
            pdf.cell(judge_w, row_h, "Judge", border=1, align="C")
            pdf.cell(math_w, row_h, "Math", border=1, align="C")
            pdf.ln()

            pdf.set_font("Helvetica", "", 7)
            for e in sorted(entries, key=lambda x: float(x.get("tps") or 0), reverse=True):
                name = e.get("model", "")[:25]
                if e.get("is_thinking"): name += " *"
                cid = e.get("computer_id", "-")[:12]
                tps = f"{e.get('tps', 0):.1f}" if e.get("tps") else "-"
                wt = f"{e.get('wall_time', 0):.1f}" if e.get("wall_time") else "-"
                ttft = f"{e.get('ttft', 0):.2f}" if e.get("ttft") else "-"
                rate = f"{e.get('quality_rating')}/10" if e.get("quality_rating") else "-"
                judge = f"{e.get('judge_score')}/10" if e.get("judge_score") else "-"
                math_s = "Y" if e.get("math_correct") else ("N" if e.get("math_correct") is not None else "-")

                if pdf.get_y() > 270:
                    pdf.add_page()
                pdf.cell(model_w, row_h, _sanitize_pdf(name), border=1)
                pdf.cell(comp_w, row_h, _sanitize_pdf(cid), border=1, align="C")
                pdf.cell(run_w, row_h, str(e.get("run_id", "")), border=1, align="C")
                pdf.cell(tps_w, row_h, tps, border=1, align="C")
                pdf.cell(time_w, row_h, wt, border=1, align="C")
                pdf.cell(ttft_w, row_h, ttft, border=1, align="C")
                pdf.cell(rate_w, row_h, rate, border=1, align="C")
                pdf.cell(judge_w, row_h, judge, border=1, align="C")
                pdf.cell(math_w, row_h, math_s, border=1, align="C")
                pdf.ln()
            pdf.ln(4)

        if _fpdf_new:
            buf = _io.BytesIO()
            pdf.output(buf)
            pdf_bytes = buf.getvalue()
        else:
            pdf_bytes = pdf.output(dest='S').encode('latin-1')

    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=ollama_bench_compare.pdf"}
    )


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


# ═══════════════════════════════════════════════
# Compare export endpoints
# ═══════════════════════════════════════════════

from fastapi import Body

@app.post("/api/compare/export/csv")
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


@app.post("/api/compare/export/pdf")
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


@app.post("/api/compare/export/html")
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


@app.get("/api/status")
async def api_status():
    return JSONResponse({
        "active": bench_state.active,
        "current_model": bench_state.current_model,
        "current_test": bench_state.current_test,
        "progress": bench_state.progress,
        "total": bench_state.total,
        "run_id": bench_state.run_id,
    })


@app.post("/api/benchmark")
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


@app.post("/api/benchmark/stop")
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


# ═══════════════════════════════════════════════
# Benchmark Suite Runner (MMLU-Pro, IFEval)
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
        self.max_parallel = 2  # how many jobs can run concurrently
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
_sse_loop = None  # set once at startup from @app.on_event("startup")

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


@app.get("/api/profile/results")
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


@app.post("/api/profile/run")
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


@app.get("/api/profile/status")
async def api_profile_status():
    """Return current profiling status."""
    return {
        "active": profile_state.active,
        "current_model": profile_state.current_model,
        "progress": profile_state.progress,
        "queue_length": len(profile_state.queue),
        "queue": profile_state.queue,
    }


@app.delete("/api/profile/queue")
async def api_profile_clear():
    """Clear the profile queue (does not stop an in-progress profile)."""
    with profile_state._lock:
        cleared = len(profile_state.queue)
        profile_state.queue.clear()
    return {"cleared": cleared}


@app.get("/api/suites")
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


@app.post("/api/suites/download/{suite_key}")
async def api_download_suite(suite_key: str):
    """Download/cache a benchmark suite dataset from HuggingFace."""
    if suite_key not in BENCHMARK_SUITES:
        return JSONResponse({"error": f"Unknown suite: {suite_key}"}, status_code=404)
    try:
        data = load_benchmark_dataset(suite_key)
        return JSONResponse({"status": "ok", "rows": len(data)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/suites/run")
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

    # Enqueue jobs (one per model)
    queued = []
    for model in models:
        job = {"host": ollama_host, "model": model, "suite_key": suite_key, "sample_size": sample_size}
        suite_state.suite_queue.append(job)
        queued.append({"model": model, "suite_key": suite_key})

    # Start processor if not already running
    suite_state.stop_requested = False  # reset any stale stop flag
    if not suite_state._processor_running:
        _start_suite_processor()

    return JSONResponse({"queued": queued, "queue_length": len(suite_state.suite_queue)})


def _get_model_vram_mb(host, model):
    """Estimate VRAM needed for a model.
    
    Strategy (in order of accuracy):
    1. Stored profile (cold-measured VRAM) — best estimate.
    2. If model is currently loaded (/api/ps), use actual SIZE — ground truth.
    3. Estimate from model file size × 1.5 (weights + KV cache overhead).
    4. Estimate from parameter_size × quant × MoE ratio × 1.5.
    5. Fallback: full GPU VRAM (forces solo run).
    """
    # 0. Check stored profile first — previous cold measurement
    try:
        sys_info = detect_system_info()
        computer_id = compute_computer_id(sys_info)
        with db_lock:
            db = get_db()
            row = db.execute(
                "SELECT vram_mib FROM model_profiles WHERE computer_id=? AND model=?",
                (computer_id, model)
            ).fetchone()
            db.close()
        if row and row[0] and row[0] > 0:
            return row[0]
    except Exception:
        pass

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
    return _get_total_vram_mb()


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

        # Load dataset
        try:
            all_data = load_benchmark_dataset(suite_key)
        except Exception:
            job.active = False
            return False

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
        elif suite_key == "bfcl_v3":
            bfcl_sample = sample_size or BENCHMARK_SUITES["bfcl_v3"]["sample_size"] or 3
            questions = _sample_bfcl(all_data, per_category=bfcl_sample)
        else:
            questions = all_data

        suite = BENCHMARK_SUITES[suite_key]
        job.total_questions = len(questions)

        # Notify: job started
        _sse_broadcast({
            "type": "job_start",
            "job_id": job.job_id,
            "model": model,
            "suite": suite_key,
            "total_questions": len(questions)
        })

        results = []

        try:

          for i, q in enumerate(questions):
            if suite_state.stop_requested:
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

            def _should_stop():
                """Check both per-job skip and global stop — allows interrupting streaming generation."""
                if job.skip_requested:
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
                for chunk in stream_iter:
                    if job.skip_requested:
                        _interrupted[0] = "skip"
                        # Send skip event to SSE
                        _sse_broadcast({
                            "type": "question_skipped",
                            "job_id": job.job_id,
                            "question": i + 1,
                            "model": model,
                            "suite": suite_key
                        })
                        return
                    if suite_state.stop_requested:
                        _interrupted[0] = "stop"
                        return
                    if chunk["type"] == "error":
                        full_result["error"] = chunk["data"]
                        # Send error event to SSE
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
                        _stream_tokens[0] += 1
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
                        _stream_tokens[0] += 1
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
                # If the generator was aborted by stop_check (skip/stop), mark it and send SSE event
                if not got_result.get("done") and not full_result.get("error") and (job.skip_requested or suite_state.stop_requested):
                    if job.skip_requested:
                        _interrupted[0] = "skip"
                        _sse_broadcast({
                            "type": "question_skipped",
                            "job_id": job.job_id,
                            "question": i + 1,
                            "model": model,
                            "suite": suite_key
                        })
                    else:
                        _interrupted[0] = "stop"
                        _sse_broadcast({
                            "type": "stream_error",
                            "job_id": job.job_id,
                            "question": i + 1,
                            "model": model,
                            "suite": suite_key,
                            "error": "Stopped by user"
                        })
                else:
                    got_result["done"] = True

            q_thread = threading.Thread(target=_run_question, daemon=True)
            q_thread.start()
            q_thread.join(timeout=SUITE_QUESTION_TIMEOUT)

            # Handle skip (user pressed Skip button) — stream was interrupted mid-generation
            if _interrupted[0] == "skip":
                results.append({
                    "question_idx": i,
                    "correct": False,
                    "model_answer": "SKIPPED",
                    "expected": json.dumps(expected, ensure_ascii=False)[:500] if expected and not isinstance(expected, str) else (expected or ""),
                    "tps": 0,
                    "wall_time": 0,
                    "wait_count": _wait_count[0],
                    "error": "Skipped by user",
                })
                job.skip_requested = False  # reset for next question
                continue

            # Handle stop — stream was interrupted, stop processing remaining questions
            if _interrupted[0] == "stop" or suite_state.stop_requested:
                results.append({
                    "question_idx": i,
                    "correct": False,
                    "model_answer": "STOPPED",
                    "expected": json.dumps(expected, ensure_ascii=False)[:500] if expected and not isinstance(expected, str) else (expected or ""),
                    "tps": 0,
                    "wall_time": 0,
                    "wait_count": _wait_count[0],
                    "error": "Stopped by user",
                })
                break  # don't continue to next question

            if q_thread.is_alive():
                # Question timed out — skip it
                results.append({
                    "question_idx": i,
                    "correct": False,
                    "model_answer": "TIMEOUT",
                    "expected": json.dumps(expected, ensure_ascii=False)[:500] if expected and not isinstance(expected, str) else (expected or ""),
                    "tps": 0,
                    "wall_time": SUITE_QUESTION_TIMEOUT,
                    "wait_count": _wait_count[0],
                    "error": f"Question timed out after {SUITE_QUESTION_TIMEOUT}s",
                })
                # Unload model to prevent VRAM hang from stale state
                try:
                    req_lib.post(f"{ollama_host}/api/generate", json={"model": model, "prompt": "", "keep_alive": 0}, timeout=10)
                except Exception:
                    pass
                continue

            response_text = got_result.get("text", "")

            # Evaluate
            if "error" in full_result:
                results.append({
                    "question_idx": i,
                    "correct": False,
                    "model_answer": "ERROR",
                    "expected": json.dumps(expected, ensure_ascii=False)[:500] if expected and not isinstance(expected, str) else (expected or ""),
                    "tps": 0,
                    "wall_time": 0,
                    "wait_count": _wait_count[0],
                    "error": full_result["error"],
                })
                continue

            if suite["answer_type"] == "mcq":
                model_answer = extract_mcq_answer(response_text, suite.get("num_options", 10))
                is_correct = (model_answer == expected) if model_answer and expected else False
                results.append({
                    "question_idx": i,
                    "correct": is_correct,
                    "model_answer": model_answer,
                    "expected": expected,
                    "tps": full_result.get("tps", 0),
                    "wall_time": full_result.get("wall_time", 0),
                    "wait_count": _wait_count[0],
                    "category": q.get("category", ""),
                })
            elif suite["answer_type"] == "function_call":
                is_correct, model_answer, details = check_bfcl_answer(response_text, expected)
                results.append({
                    "question_idx": i,
                    "correct": is_correct,
                    "model_answer": model_answer,
                    "expected": json.dumps(expected, ensure_ascii=False)[:500] if expected else "",
                    "tps": full_result.get("tps", 0),
                    "wall_time": full_result.get("wall_time", 0),
                    "wait_count": _wait_count[0],
                    "category": q.get("category", ""),
                    "bfcl_details": details,
                })
            elif suite["answer_type"] == "instruction":
                inst_ids, kwargs_list = q.get("instruction_id_list", []), q.get("kwargs", [])
                constraint_results = check_ifeval_constraints(response_text, inst_ids, kwargs_list)
                constraints_total = len(constraint_results)
                constraints_passed = sum(1 for _, p in constraint_results if p)
                results.append({
                    "question_idx": i,
                    "correct": constraints_passed == constraints_total,
                    "constraints_total": constraints_total,
                    "constraints_passed": constraints_passed,
                    "tps": full_result.get("tps", 0),
                    "wall_time": full_result.get("wall_time", 0),
                    "wait_count": _wait_count[0],
                    "constraint_details": [(cid, p) for cid, p in constraint_results],
                })

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
        avg_tps = sum(r.get("tps", 0) for r in results) / max(total_q, 1)
        total_wall = sum(r.get("wall_time", 0) for r in results)

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
            "total": total_q
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

            # Wait for a job to complete before trying again
            completion_event.clear()
            completion_event.wait(timeout=5)

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


@app.get("/api/suites/stream")
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


@app.get("/api/suites/status")
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


@app.get("/api/suites/queue")
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


@app.delete("/api/suites/queue")
async def api_suite_queue_clear():
    """Clear all pending jobs from the queue (does not stop running jobs)."""
    cleared = len(suite_state.suite_queue)
    suite_state.suite_queue.clear()
    return JSONResponse({"cleared": cleared})


@app.post("/api/suites/stop")
async def api_suite_stop():
    """Stop all suite jobs: clears queue, signals running jobs to stop, saves partial results."""
    cleared = len(suite_state.suite_queue)
    suite_state.suite_queue.clear()
    if suite_state.active:
        suite_state.stop_requested = True
        # Notify SSE subscribers
        _sse_broadcast({"type": "stopping", "cleared": cleared})
    return JSONResponse({"stopping": suite_state.active, "cleared_queue": cleared})


@app.post("/api/suites/skip/{job_id}")
async def api_suite_skip_question(job_id: int):
    """Skip the current question for a running suite job. The model continues to the next question."""
    for job in suite_state.active_jobs:
        if job.job_id == job_id and job.active:
            job.skip_requested = True
            return JSONResponse({"skipped": True, "job_id": job_id, "model": job.model, "suite": job.suite_key, "question": job.current_question})
    return JSONResponse({"error": f"No active job with id {job_id}", "skipped": False}, status_code=404)


@app.post("/api/suites/max-parallel")
async def api_set_max_parallel(request: Request):
    """Set max parallel jobs. Body: {max_parallel: N}. Only affects future dispatches."""
    body = await request.json()
    n = int(body.get("max_parallel", 2))
    n = max(1, min(n, 8))  # clamp 1-8
    suite_state.max_parallel = n
    return JSONResponse({"max_parallel": suite_state.max_parallel})


@app.get("/api/suites/results")
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
            "timestamp": row[12],
            "computer_id": row[13] or "",
        })
    return JSONResponse(results)


@app.delete("/api/suites/results/{result_id}")
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


@app.delete("/api/suites/results")
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


@app.get("/api/stream")
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


@app.get("/api/judge-stream")
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


# ═══════════════════════════════════════════════
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
          <label style="font-size:.85em;color:var(--muted);display:block;margin-bottom:4px;">Questions per category:</label>
          <input id="bm-sample" type="number" value="20" min="1" max="200" style="width:100%;padding:6px 10px;border-radius:6px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.95em;">
        </div>
        <div style="min-width:160px;">
          <label style="font-size:.85em;color:var(--muted);display:block;margin-bottom:4px;">Max parallel jobs:</label>
          <input id="bm-parallel" type="number" value="2" min="1" max="8" style="width:100%;padding:6px 10px;border-radius:6px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.95em;" onchange="setBmParallel(this.value)">
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
let bmStreamStates = new Map();  // Map<job_id, {model, suite, question, text, thinking, tokens, tps, answers, waitCount}>
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
    const div = document.createElement('div');
    div.className = 'select-item';
    div.dataset.key = key;
    div.innerHTML = `<input type="checkbox" id="bm-suite-${key}" style="margin-right:6px;">` +
      `<label for="bm-suite-${key}" style="cursor:pointer;"><strong>${s.name}</strong><br><span style="font-size:.8em;color:var(--muted);">${s.description}</span></label>` +
      `<span style="margin-left:auto;font-size:.75em;color:${s.cached ? '#4caf50' : 'var(--muted)'}">${s.cached ? '✓ cached' : 'needs download'}</span>`;
    div.querySelector('input').addEventListener('change', () => {
      bmSelectedSuites = [...document.querySelectorAll('#bm-suite-list input:checked')].map(i => i.closest('.select-item').dataset.key);
    });
    // Default: mmlu_pro checked
    if (key === 'mmlu_pro') {
      div.querySelector('input').checked = true;
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
      for (const job of data.queue) {
        items.push(`<div style="padding:6px 12px;border-bottom:1px solid rgba(108,140,255,.08);font-size:.85em;display:flex;justify-content:space-between;font-family:monospace;color:#556;">
          <span>${job.model} — ${job.suite_key}</span>
          <span style="color:#556;">queued</span>
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
        document.getElementById('btn-bm-run').style.display = 'none';
        document.getElementById('btn-bm-stop').style.display = '';
        for (const job of st.active_jobs) {
          _createStreamPanel(job.job_id, job.model, job.suite_key);
          bmStreamStates.set(job.job_id, {model: job.model, suite: job.suite_key, question: 0, text: '', thinking: false, tokens: 0, tps: 0, answers: [], waitCount: 0, correctCount: 0, totalCount: 0});
        }
      }
    }
  } catch(e) {}
}

function connectBmSSE() {
  if (bmSuiteSource) { bmSuiteSource.close(); bmSuiteSource = null; }
  if (bmPollInterval) { clearInterval(bmPollInterval); bmPollInterval = null; }

  document.getElementById('bm-progress').style.display = 'block';

  bmSuiteSource = new EventSource('/api/suites/stream');
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
        document.getElementById('btn-bm-run').style.display = 'none';
        document.getElementById('btn-bm-stop').style.display = '';
        // Create streaming panel for this job
        _createStreamPanel(jobId, d.model, d.suite);
        bmStreamStates.set(jobId, {model: d.model, suite: d.suite, question: 0, text: '', thinking: false, tokens: 0, tps: 0, answers: [], waitCount: 0, correctCount: 0, totalCount: 0});
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
        _startTimer(jobId);
        _resetBarGraph(jobId);
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
      } else if (d.type === 'job_done') {
        toast(`${d.model} on ${d.suite}: ${d.accuracy.toFixed(1)}% (${d.correct}/${d.total})`);
        // Show final summary overlay in the panel
        _stopTimer(jobId);
        const panel = document.getElementById('bm-stream-' + jobId);
        if (panel) {
          const acc = d.accuracy.toFixed(1);
          const accColor = acc >= 80 ? '#4ade80' : acc >= 50 ? '#facc15' : '#f87171';
          const overlay = document.createElement('div');
          overlay.style.cssText = 'position:absolute;top:0;left:0;right:0;bottom:0;display:flex;flex-direction:column;align-items:center;justify-content:center;background:rgba(8,10,16,.88);z-index:10;animation:panel-appear .4s ease-out;';
          overlay.innerHTML = `<div style="font-family:monospace;font-size:2.2em;font-weight:700;color:${accColor};letter-spacing:1px;">${acc}%</div><div style="font-family:monospace;font-size:.85em;color:#6c8cff;margin-top:6px;">${d.correct}/${d.total} correct</div><div style="font-family:monospace;font-size:.75em;color:#556;margin-top:4px;">${d.model}</div>`;
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

async function runBenchmarkSuite() {
  if (!bmSelectedSuites.length) return alert('Select at least one suite');
  if (!bmSelectedModels.length) return alert('Add at least one model');
  const sampleSize = parseInt(document.getElementById('bm-sample').value) || 20;
  const host = document.getElementById('ollama-host').value;

  // Queue all combinations: each suite × each model
  let totalQueued = 0;
  for (const suiteKey of bmSelectedSuites) {
    try {
      const res = await fetch('/api/suites/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({host, models: bmSelectedModels, suite_key: suiteKey, sample_size: sampleSize})
      });
      const data = await res.json();
      if (data.error) { toast(`Error for ${suiteKey}: ${data.error}`); continue; }
      totalQueued += (data.queued || []).length;
    } catch(e) { toast(`Failed to queue ${suiteKey}: ${e.message}`); }
  }

  if (!totalQueued) return;

  toast(`Queued ${totalQueued} job${totalQueued > 1 ? 's' : ''}`);
  connectBmSSE();
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
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ollama Benchmark v4 - Web UI")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST, help="Ollama API host")
    parser.add_argument("--port", type=int, default=8114, help="Web server port")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path")
    parser.add_argument("--web-host", default="0.0.0.0", help="Web server bind host")
    args = parser.parse_args()

    _cfg["db_path"] = args.db
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