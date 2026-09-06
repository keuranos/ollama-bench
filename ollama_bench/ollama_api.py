#!/usr/bin/env python3
"""Ollama API wrappers — streaming generate and chat."""

import json
import time

import requests as req_lib

from ollama_bench.config import DEFAULT_TIMEOUT, HIGH_NUM_PREDICT

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

    resp_ref = [None]  # mutable ref so caller can close the connection externally
    try:
        # Use tuple timeout: (connect_timeout, read_timeout).
        # Read timeout ensures iter_lines() can't block forever.
        # Must be >= question timeout so the question timeout thread handles cancellation,
        # not HTTP timeout. When two models share GPUs, one can sit idle for 60+ seconds
        # waiting for GPU time — a 60s read timeout kills it before it can answer.
        connect_timeout = min(30, timeout) if isinstance(timeout, (int, float)) else 30
        read_timeout = timeout if isinstance(timeout, (int, float)) else 900
        with req_lib.post(url, json=payload, stream=True, timeout=(connect_timeout, read_timeout)) as resp:
            resp.raise_for_status()
            resp_ref[0] = resp  # store for external abort
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

    except req_lib.exceptions.ReadTimeout:
        # Read timeout — likely the model was processing and no data was sent within
        # the read_timeout window. If stop_check says abort, yield error; otherwise reconnect.
        if stop_check and stop_check():
            yield {"type": "error", "data": "ABORTED"}
            return
        yield {"type": "error", "data": "READ_TIMEOUT"}
        return
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

