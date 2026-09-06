#!/usr/bin/env python3
"""Benchmark suite definitions — MMLU-Pro, IFEval, BFCL loading, formatting, scoring."""

import json
import re
import random
from pathlib import Path

from ollama_bench.config import TESTPACKS_DIR, HIGH_NUM_PREDICT, DEFAULT_TIMEOUT


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
        "sample_size": 3,    # per category (3 x 6 cats = 18 questions default)
        "answer_type": "function_call",  # structured JSON function call matching
        "categories": ["simple", "multiple", "parallel", "parallel_multiple", "java", "javascript"],
    },
    "bfcl_v4": {
        "name": "BFCL v4",
        "description": "Berkeley Function Calling Leaderboard v4 — tool/function calling (12 single-turn categories)",
        "hf_dataset": "jeypiii/bfcl_v4_single_turn",
        "hf_split": None,  # loaded per-category
        "sample_size": 3,
        "answer_type": "function_call",
        "categories": [
            "simple_python", "simple_java", "simple_javascript",
            "multiple", "parallel", "parallel_multiple",
            "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
            "irrelevance", "live_irrelevance",
        ],
        "num_predict": 4096,
        "num_ctx": 8192,
        "temperature": 0.0,
    },
    "bfcl_v4_web_search": {
        "name": "BFCL v4 — Web Search",
        "description": "BFCL v4 agentic web search — multi-hop reasoning with real web searches",
        "hf_dataset": "arcee-ai/bfcl_v4_web_search",
        "hf_split": "train",
        "sample_size": 5,
        "answer_type": "web_search",
        "categories": ["web_search_base"],
        "num_predict": 4096,
        "num_ctx": 8192,
        "temperature": 0.0,
        "requires_search": True,
    },
    "bfcl_v4_memory_kv": {
        "name": "BFCL v4 — Memory (KV)",
        "description": "BFCL v4 agentic memory — key-value store interactions",
        "data_source": "gorilla_github",
        "sample_size": 5,
        "answer_type": "memory",
        "memory_type": "kv",
        "categories": ["memory_kv"],
        "num_predict": 4096, "num_ctx": 8192, "temperature": 0.0,
    },
    "bfcl_v4_memory_vector": {
        "name": "BFCL v4 — Memory (Vector)",
        "description": "BFCL v4 agentic memory — vector database interactions",
        "data_source": "gorilla_github",
        "sample_size": 5,
        "answer_type": "memory",
        "memory_type": "vector",
        "categories": ["memory_vector"],
        "num_predict": 4096, "num_ctx": 8192, "temperature": 0.0,
    },
    "bfcl_v4_memory_rec_sum": {
        "name": "BFCL v4 — Memory (RecSum)",
        "description": "BFCL v4 agentic memory — recursive summarization",
        "data_source": "gorilla_github",
        "sample_size": 5,
        "answer_type": "memory",
        "memory_type": "rec_sum",
        "categories": ["memory_rec_sum"],
        "num_predict": 4096, "num_ctx": 8192, "temperature": 0.0,
    },
}

def load_benchmark_dataset(suite_key):
    """Load a benchmark dataset from HuggingFace, cache as parquet locally."""
    suite = BENCHMARK_SUITES[suite_key]
    cache_dir = TESTPACKS_DIR / suite_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "data.parquet"

    if suite_key == "bfcl_v4":
        import pandas as pd
        all_dfs = []
        for cat in suite["categories"]:
            cat_cache = cache_dir / f"{cat}.parquet"
            if cat_cache.exists():
                cat_df = pd.read_parquet(cat_cache)
            else:
                from datasets import load_dataset
                ds = load_dataset(suite["hf_dataset"], cat, split="test")
                cat_df = ds.to_pandas()
                cat_df.to_parquet(cat_cache)
            if "category" not in cat_df.columns or cat_df["category"].isna().any():
                cat_df["category"] = cat
            all_dfs.append(cat_df)
        df = pd.concat(all_dfs, ignore_index=True)
        df.to_parquet(cache_file)
        records = df.to_dict("records")
        for row in records:
            row["category"] = row.get("category", cat)
        return records
    elif suite_key == "bfcl_v4_web_search":
        import pandas as pd
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
        else:
            from datasets import load_dataset
            ds = load_dataset(suite["hf_dataset"], split=suite.get("hf_split", "train"))
            df = ds.to_pandas()
            df.to_parquet(cache_file)
        records = df.to_dict("records")
        for row in records:
            row["category"] = "web_search_base"
        return records
    elif suite_key.startswith("bfcl_v4_memory"):
        import pandas as pd
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            # Validate cache — reject stale/wrong-format caches (e.g. synthetic data
            # with "prompt"/"initial_config" columns instead of gorilla format).
            # Gorilla data has "prerequisites" column; stale synthetic data does not.
            if "prerequisites" not in df.columns or len(df) < 50:
                print(f"[cache] Invalid {suite_key} cache ({len(df)} rows, missing 'prerequisites') — regenerating from gorilla")
                cache_file.unlink(missing_ok=True)
            else:
                records = df.to_dict("records")
                return records
        # Load from gorilla GitHub repo (official BFCL v4 source)
        records = _load_bfcl_memory_from_gorilla(suite, cache_dir)
        if records:
            df = pd.DataFrame(records)
            df.to_parquet(cache_file)
        return records

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
    if suite_key in ("bfcl_v3", "bfcl_v4"):
        import re
        for row in records:
            if "category" not in row or not row["category"]:
                row_id = str(row.get("id", ""))
                matched = False
                for cat in BENCHMARK_SUITES[suite_key]["categories"]:
                    if row_id.startswith(cat):
                        row["category"] = cat
                        matched = True
                        break
                if not matched:
                    m = re.match(r"^(.+?)_\d", row_id)
                    row["category"] = m.group(1) if m else row_id
    return records


# ─── BFCL v4 Memory Dataset Loader ──────────────────────────────────────
GORILLA_RAW_BASE = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data"
MEMORY_SCENARIOS = ["customer", "finance", "healthcare", "notetaker", "student"]


def _load_bfcl_memory_from_gorilla(suite, cache_dir):
    """Load BFCL v4 memory test data from the official gorilla GitHub repo.
    
    Downloads:
    - BFCL_v4_memory.json: 155 test entries (id, question, involved_classes, scenario)
    - possible_answer/BFCL_v4_memory.json: ground truth answers (id, ground_truth, source)
    - memory_prereq_conversation/memory_{scenario}.json: prerequisite conversations
    
    Each question runs prerequisite conversations through the model first (so it stores
    facts in memory), then asks the actual question. The model's text answer is compared
    against ground_truth (list of acceptable strings).
    
    Returns a list of dicts, one per test entry, with:
    - id, question, involved_classes, scenario
    - ground_truth: list of acceptable answer strings
    - source: context text about the person/entity
    - prerequisites: list of prerequisite conversation dicts
    - category: memory_kv / memory_vector / memory_rec_sum
    """
    import requests as _req
    
    test_entries = []
    answers_by_id = {}
    prereqs_by_id = {}
    
    # 1. Load main test entries (JSONL)
    try:
        url = f"{GORILLA_RAW_BASE}/BFCL_v4_memory.json"
        r = _req.get(url, timeout=30)
        r.raise_for_status()
        for line in r.text.strip().split("\n"):
            if line.strip():
                test_entries.append(json.loads(line))
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to download BFCL_v4_memory.json: {e}")
        return []
    
    # 2. Load possible answers (JSONL)
    try:
        url = f"{GORILLA_RAW_BASE}/possible_answer/BFCL_v4_memory.json"
        r = _req.get(url, timeout=30)
        r.raise_for_status()
        for line in r.text.strip().split("\n"):
            if line.strip():
                ans = json.loads(line)
                answers_by_id[ans["id"]] = ans
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to download BFCL_v4_memory answers: {e}")
    
    # 3. Load prerequisite conversations per scenario (JSONL)
    for scenario in MEMORY_SCENARIOS:
        try:
            url = f"{GORILLA_RAW_BASE}/memory_prereq_conversation/memory_{scenario}.json"
            r = _req.get(url, timeout=30)
            r.raise_for_status()
            for line in r.text.strip().split("\n"):
                if line.strip():
                    prereq = json.loads(line)
                    prereq_id = prereq.get("id", "")
                    prereqs_by_id[prereq_id] = prereq
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to download memory prereqs for {scenario}: {e}")
    
    # 4. Merge data into records
    # Determine the memory backend type from involved_classes
    memory_type = suite.get("memory_type", "kv")
    backend_class = {"kv": "MemoryAPI_kv", "vector": "MemoryAPI_vector", "rec_sum": "MemoryAPI_rec_sum"}[memory_type]
    
    records = []
    for entry in test_entries:
        entry_id = entry.get("id", "")
        
        # Match prerequisite conversations to this test entry
        # All prerequisite conversations from the same scenario belong to ALL test
        # entries in that scenario — the model should store facts across multiple
        # conversations, then each question tests recall of a specific fact.
        entry_prereqs = []
        scenario = entry.get("scenario", "")
        if scenario:
            for pid, prereq in prereqs_by_id.items():
                # Match by scenario: prereq has a "scenario" field from the data
                # and also the scenario name appears in the ID
                prereq_scenario = prereq.get("scenario", "")
                if prereq_scenario == scenario or (scenario in pid):
                    entry_prereqs.append(prereq)
        
        # Get ground truth answer
        ans = answers_by_id.get(entry_id, {})
        ground_truth = ans.get("ground_truth", [])
        source = ans.get("source", "")
        
        # Build the record
        record = {
            "id": entry_id,
            "question": entry.get("question", []),
            "involved_classes": [backend_class],
            "scenario": scenario,
            "ground_truth": ground_truth,
            "source": source,
            "prerequisites": entry_prereqs,
            "category": f"memory_{memory_type}",
        }
        records.append(record)
    
    # Save prerequisite data as separate cache for the runner
    if prereqs_by_id:
        prereq_cache = cache_dir / "prereqs.json"
        with open(prereq_cache, "w") as f:
            json.dump(prereqs_by_id, f, ensure_ascii=False)
    
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

def _sample_bfcl(all_data, per_category=3, suite_key="bfcl_v3"):
    """Sample N questions per BFCL category for balanced coverage."""
    import random
    random.seed(42)
    categories = BENCHMARK_SUITES[suite_key]["categories"]
    by_cat = {}
    for row in all_data:
        cat = row.get("category", categories[0])
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(row)
    sampled = []
    for cat in categories:
        cat_data = by_cat.get(cat, [])
        n = min(per_category, len(cat_data))
        if n > 0:
            sampled.extend(random.sample(cat_data, n))
    return sampled


def check_web_search_answer(model_answer, ground_truth_answers):
    """Check if model answer matches any acceptable answer from web search ground truth."""
    if not ground_truth_answers:
        return False, ""
    model_lower = model_answer.strip().lower()
    for gt in ground_truth_answers:
        gt_lower = str(gt).strip().lower()
        if model_lower == gt_lower:
            return True, gt
        if gt_lower in model_lower:
            return True, gt
        try:
            gt_num = float(gt_lower)
            nums = re.findall(r'[\d.]+', model_lower)
            for n in nums:
                if abs(float(n) - gt_num) < 0.01:
                    return True, gt
        except (ValueError, IndexError):
            pass
    return False, ""




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
    # Handle irrelevance cases (no expected function call)
    if not ground_truth or (isinstance(ground_truth, list) and len(ground_truth) == 0):
        model_calls = _parse_bfcl_response(response_text)
        if not model_calls:
            return True, "no_call_correct", {"irrelevance": True}
        return False, "irrelevance_false_positive", {"irrelevance": True, "model_calls": len(model_calls)}

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
