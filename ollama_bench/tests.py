#!/usr/bin/env python3
"""Test management — CRUD, seed, backfill."""

import json
import time
from pathlib import Path

import requests as req_lib

from ollama_bench.config import TESTS_FILE, TESTPACKS_DIR, PACK_VERSION, DEFAULT_TIMEOUT, HIGH_NUM_PREDICT
from ollama_bench.ollama_api import stream_generate_chunks
from ollama_bench.suites import DEFAULT_TESTS

# ═══════════════════════════════════════════════

def format_bfcl_messages(row):
    # v4: question field contains the messages
    # v3: chat_completion_input field contains the messages
    chat_input = row.get("chat_completion_input") or row.get("question", [])
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

    # v4 question field is a numpy array of message dicts — convert to list
    if hasattr(chat_input, 'tolist'):
        chat_input = chat_input.tolist()

    messages = []
    if isinstance(chat_input, list) and len(chat_input) > 0:
        for msg in chat_input:
            # Each element might be {role, content} or a nested list containing that
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            elif isinstance(msg, list) and len(msg) > 0 and isinstance(msg[0], dict):
                # Nested list of dicts
                for submsg in msg:
                    if isinstance(submsg, dict):
                        role = submsg.get("role", "user")
                        content = submsg.get("content", "")
                        if role in ("system", "user", "assistant"):
                            messages.append({"role": role, "content": content})
                        else:
                            messages.append({"role": "user", "content": content})
                continue
            else:
                continue
            if role in ("system", "user", "assistant"):
                messages.append({"role": role, "content": content})
            else:
                messages.append({"role": "user", "content": content})
    else:
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


def stream_chat_chunks(host, model, messages, timeout=DEFAULT_TIMEOUT, num_predict=None, temperature=0.0, stop_check=None, extra_options=None, stop_sequences=None, tools=None):
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
    if tools:
        payload["tools"] = tools

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
    # Accumulate tool_calls from Ollama's streaming chat response
    # When a model uses native tool calling, Ollama sends message.tool_calls
    # in the final chunk(s). We collect them so the agentic loop can use them.
    collected_tool_calls = []

    resp_ref = [None]  # mutable ref so caller can close the connection externally

    try:
        connect_timeout = min(30, timeout) if isinstance(timeout, (int, float)) else 30
        read_timeout = timeout if isinstance(timeout, (int, float)) else 900
        with req_lib.post(url, json=payload, stream=True, timeout=(connect_timeout, read_timeout)) as resp:
            resp.raise_for_status()
            resp_ref[0] = resp  # store for external abort
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

                # Collect native tool calls from Ollama's streaming response
                # Ollama sends tool_calls in message chunks when model uses tools
                tc_list = msg.get("tool_calls")
                if tc_list:
                    collected_tool_calls.extend(tc_list)

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

    except req_lib.exceptions.ReadTimeout:
        if stop_check and stop_check():
            yield {"type": "error", "data": "ABORTED"}
        else:
            yield {"type": "error", "data": "READ_TIMEOUT"}
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
        "tool_calls": collected_tool_calls if collected_tool_calls else None,
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

