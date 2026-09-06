#!/usr/bin/env python3
"""Agentic evaluation loop, web search backend, and memory backends for BFCL v4."""

import json
import time
import re
import os
from abc import ABC, abstractmethod
from typing import Optional, Callable

import requests as req_lib

from ollama_bench.config import DEFAULT_TIMEOUT, HIGH_NUM_PREDICT
from ollama_bench.ollama_api import stream_generate_chunks
from ollama_bench.tests import stream_chat_chunks
from ollama_bench.suites import _parse_bfcl_response


# ─── Tool Definitions ───────────────────────────────────────────────────────
# Aligned with official BFCL v4 gorilla repo definitions:
# github.com/ShishirPatil/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/

WEB_SEARCH_TOOLS = [
    {
        "type": "function", "function": {
            "name": "search_engine_query",
            "description": "Query the search engine for keywords and region.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "The keywords to search for."},
                    "max_results": {"type": "integer", "description": "Max results to return.", "default": 10},
                    "region": {"type": "string", "description": "Region code.", "default": "wt-wt"}
                },
                "required": ["keywords"]
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "fetch_url_content",
            "description": "Fetch and process content from a URL. Modes: raw, markdown, truncate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."},
                    "mode": {"type": "string", "enum": ["raw", "markdown", "truncate"], "default": "truncate"}
                },
                "required": ["url"]
            }
        }
    },
]

MEMORY_KV_TOOLS = [
    {"type": "function", "function": {"name": "archival_memory_add", "description": "This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system. Tool description: Add a key-value pair to the long-term memory. Make sure to use meaningful keys for easy retrieval later.", "parameters": {"type": "object", "properties": {"key": {"type": "string", "description": "The key to add to the long-term memory."}, "value": {"type": "string", "description": "The value to associate with the key in the long-term memory."}}, "required": ["key", "value"]}}},
    {"type": "function", "function": {"name": "archival_memory_clear", "description": "This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system. Tool description: Clear all key-value pairs from the long-term memory, including those from previous interactions. This operation is irreversible.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "archival_memory_key_search", "description": "This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system. Tool description: Search for key names in the long-term memory that are similar to the query using BM25+ algorithm.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The query to search for similar key names in the long-term memory."}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "archival_memory_list_keys", "description": "This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system. Tool description: List all keys currently in the long-term memory.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "archival_memory_remove", "description": "This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system. Tool description: Remove a key-value pair from the long-term memory.", "parameters": {"type": "object", "properties": {"key": {"type": "string", "description": "The key to remove from the long-term memory."}}, "required": ["key"]}}},
    {"type": "function", "function": {"name": "archival_memory_replace", "description": "This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system. Tool description: Replace a key-value pair in the long-term memory with a new value.", "parameters": {"type": "object", "properties": {"key": {"type": "string", "description": "The key to replace in the long-term memory."}, "old_value": {"type": "string", "description": "The old value to replace."}, "new_value": {"type": "string", "description": "The new value to set."}}, "required": ["key", "old_value", "new_value"]}}},
    {"type": "function", "function": {"name": "archival_memory_retrieve", "description": "This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system. Tool description: Retrieve the value associated with a key from the long-term memory. This function does not support partial key matching or similarity search.", "parameters": {"type": "object", "properties": {"key": {"type": "string", "description": "The key to retrieve from the long-term memory."}}, "required": ["key"]}}},
    {"type": "function", "function": {"name": "core_memory_add", "description": "This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system. Tool description: Add a key-value pair to the short-term memory. Make sure to use meaningful keys for easy retrieval later.", "parameters": {"type": "object", "properties": {"key": {"type": "string", "description": "The key to add to the short-term memory."}, "value": {"type": "string", "description": "The value to associate with the key in the short-term memory."}}, "required": ["key", "value"]}}},
    {"type": "function", "function": {"name": "core_memory_clear", "description": "This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system. Tool description: Clear all key-value pairs from the short-term memory, including those from previous interactions. This operation is irreversible.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "core_memory_key_search", "description": "This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system. Tool description: Search for key names in the short-term memory that are similar to the query using BM25+ algorithm.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The query to search for similar key names in the short-term memory."}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "core_memory_list_keys", "description": "This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system. Tool description: List all keys currently in the short-term memory.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "core_memory_remove", "description": "This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system. Tool description: Remove a key-value pair from the short-term memory.", "parameters": {"type": "object", "properties": {"key": {"type": "string", "description": "The key to remove from the short-term memory."}}, "required": ["key"]}}},
    {"type": "function", "function": {"name": "core_memory_replace", "description": "This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system. Tool description: Replace a key-value pair in the short-term memory with a new value.", "parameters": {"type": "object", "properties": {"key": {"type": "string", "description": "The key to replace in the short-term memory."}, "old_value": {"type": "string", "description": "The old value to replace."}, "new_value": {"type": "string", "description": "The new value to set."}}, "required": ["key", "old_value", "new_value"]}}},
    {"type": "function", "function": {"name": "core_memory_retrieve", "description": "This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system. Tool description: Retrieve the value associated with a key from the short-term memory. This function does not support partial key matching or similarity search.", "parameters": {"type": "object", "properties": {"key": {"type": "string", "description": "The key to retrieve from the short-term memory."}}, "required": ["key"]}}},
    {"type": "function", "function": {"name": "core_memory_retrieve_all", "description": "This tool belongs to the memory suite, which provides APIs to interact with a key-value based memory system. Tool description: Retrieve all key-value pairs from the short-term memory.", "parameters": {"type": "object", "properties": {}, "required": []}}},
]

MEMORY_VECTOR_TOOLS = [
    {"type": "function", "function": {"name": "archival_memory_add", "description": "This tool belongs to the memory suite, which provides APIs to interact with a vector database based memory system. Tool description: Add a text entry to the archival (long-term) memory.", "parameters": {"type": "object", "properties": {"content": {"type": "string", "description": "The text content to add to the archival memory."}}, "required": ["content"]}}},
    {"type": "function", "function": {"name": "archival_memory_clear", "description": "This tool belongs to the memory suite, which provides APIs to interact with a vector database based memory system. Tool description: Clear all entries from the archival (long-term) memory.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "archival_memory_remove", "description": "This tool belongs to the memory suite, which provides APIs to interact with a vector database based memory system. Tool description: Remove a text entry from the archival (long-term) memory by matching exact content.", "parameters": {"type": "object", "properties": {"content": {"type": "string", "description": "The exact text content to remove from the archival memory."}}, "required": ["content"]}}},
    {"type": "function", "function": {"name": "archival_memory_retrieve", "description": "This tool belongs to the memory suite, which provides APIs to interact with a vector database based memory system. Tool description: Retrieve entries from the archival (long-term) memory by semantic similarity search.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The query to search for semantically similar entries in the archival memory."}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "archival_memory_retrieve_all", "description": "This tool belongs to the memory suite, which provides APIs to interact with a vector database based memory system. Tool description: Retrieve all entries from the archival (long-term) memory.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "archival_memory_update", "description": "This tool belongs to the memory suite, which provides APIs to interact with a vector database based memory system. Tool description: Update a text entry in the archival (long-term) memory by replacing old content with new content.", "parameters": {"type": "object", "properties": {"old_content": {"type": "string", "description": "The old text content to find and replace."}, "new_content": {"type": "string", "description": "The new text content to replace with."}}, "required": ["old_content", "new_content"]}}},
    {"type": "function", "function": {"name": "core_memory_add", "description": "This tool belongs to the memory suite, which provides APIs to interact with a vector database based memory system. Tool description: Add a text entry to the core (short-term) memory.", "parameters": {"type": "object", "properties": {"content": {"type": "string", "description": "The text content to add to the core memory."}}, "required": ["content"]}}},
    {"type": "function", "function": {"name": "core_memory_clear", "description": "This tool belongs to the memory suite, which provides APIs to interact with a vector database based memory system. Tool description: Clear all entries from the core (short-term) memory.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "core_memory_remove", "description": "This tool belongs to the memory suite, which provides APIs to interact with a vector database based memory system. Tool description: Remove a text entry from the core (short-term) memory by matching exact content.", "parameters": {"type": "object", "properties": {"content": {"type": "string", "description": "The exact text content to remove from the core memory."}}, "required": ["content"]}}},
    {"type": "function", "function": {"name": "core_memory_retrieve", "description": "This tool belongs to the memory suite, which provides APIs to interact with a vector database based memory system. Tool description: Retrieve entries from the core (short-term) memory by semantic similarity search.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The query to search for semantically similar entries in the core memory."}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "core_memory_retrieve_all", "description": "This tool belongs to the memory suite, which provides APIs to interact with a vector database based memory system. Tool description: Retrieve all entries from the core (short-term) memory.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "core_memory_update", "description": "This tool belongs to the memory suite, which provides APIs to interact with a vector database based memory system. Tool description: Update a text entry in the core (short-term) memory by replacing old content with new content.", "parameters": {"type": "object", "properties": {"old_content": {"type": "string", "description": "The old text content to find and replace."}, "new_content": {"type": "string", "description": "The new text content to replace with."}}, "required": ["old_content", "new_content"]}}},
]

MEMORY_RECSUM_TOOLS = [
    {"type": "function", "function": {"name": "memory_append", "description": "This tool belongs to the memory suite, which provides APIs to interact with a recursive summarization based memory system. Tool description: Append text to the memory.", "parameters": {"type": "object", "properties": {"content": {"type": "string", "description": "The text content to append to memory."}}, "required": ["content"]}}},
    {"type": "function", "function": {"name": "memory_clear", "description": "This tool belongs to the memory suite, which provides APIs to interact with a recursive summarization based memory system. Tool description: Clear all content from the memory.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "memory_replace", "description": "This tool belongs to the memory suite, which provides APIs to interact with a recursive summarization based memory system. Tool description: Replace text in the memory by finding old content and substituting with new content.", "parameters": {"type": "object", "properties": {"old_content": {"type": "string", "description": "The old text content to find and replace."}, "new_content": {"type": "string", "description": "The new text content to replace with."}}, "required": ["old_content", "new_content"]}}},
    {"type": "function", "function": {"name": "memory_retrieve", "description": "This tool belongs to the memory suite, which provides APIs to interact with a recursive summarization based memory system. Tool description: Retrieve the entire memory content.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "memory_update", "description": "This tool belongs to the memory suite, which provides APIs to interact with a recursive summarization based memory system. Tool description: Update the memory by providing a complete new content to overwrite existing memory.", "parameters": {"type": "object", "properties": {"content": {"type": "string", "description": "The complete new content to write to memory."}}, "required": ["content"]}}},
]


# ─── Agentic Eval Loop ──────────────────────────────────────────────────────

def parse_tool_calls(response_text, ollama_tool_calls=None):
    """Extract tool calls from model output.
    
    Priority:
    1. Ollama native tool_calls from chat API response
    2. Text parsing via _parse_bfcl_response
    """
    if ollama_tool_calls:
        result = []
        for tc in ollama_tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except (json.JSONDecodeError, TypeError):
                args = {}
            result.append({"name": name, "arguments": args})
        return result
    
    # Text fallback: use existing BFCL parser
    bfcl_calls = _parse_bfcl_response(response_text)
    result = []
    for call in bfcl_calls:
        if isinstance(call, dict):
            if len(call) == 1:
                name = list(call.keys())[0]
                args = call[name] if isinstance(call[name], dict) else {}
                result.append({"name": name, "arguments": args})
            else:
                # Might already be {name, arguments} format
                name = call.get("name", call.get("function", ""))
                args = call.get("arguments", call.get("parameters", {}))
                if name:
                    result.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
    return result


def run_agentic_eval(host, model, messages, tool_definitions, tool_executor,
                     max_turns=10, timeout_per_turn=60, num_predict=4096,
                     num_ctx=8192, temperature=0.0,
                     stop_check=None, extra_options=None,
                     sse_callback=None):
    """Run a multi-turn agentic evaluation loop.
    
    1. Send messages + tool definitions to model via /api/chat
    2. If model outputs tool_calls, execute them via tool_executor
    3. Append tool results, send back to model
    4. Repeat until model gives text response or max_turns
    
    Args:
        host: Ollama host URL
        model: Model name
        messages: Initial conversation messages
        tool_definitions: Tool schema list for Ollama tools param
        tool_executor: Callable(func_name, args) -> result_dict
        max_turns: Max agentic turns (default 10)
        num_predict: Max output tokens per turn
        sse_callback: Optional callable(dict) to broadcast SSE events
    
    Returns dict with: answer, tool_calls, tool_results, turns, tps, wall_time
    """
    conversation = list(messages)  # copy
    all_tool_calls = []
    all_tool_results = []
    total_output_tokens = 0
    start_time = time.time()
    
    for turn in range(max_turns):
        # Stream chat with tools
        full_result = {}
        got_result = {"text": "", "done": False}
        tool_calls_from_ollama = None
        response_text = ""
        
        options = {"num_predict": num_predict, "num_ctx": num_ctx, "temperature": temperature}
        if extra_options:
            options.update(extra_options)
        
        for chunk in stream_chat_chunks(host, model, conversation,
                                         num_predict=num_predict,
                                         temperature=temperature,
                                         stop_check=stop_check,
                                         extra_options={"num_ctx": num_ctx},
                                         tools=tool_definitions):
            if chunk["type"] == "error":
                return {"answer": "", "tool_calls": all_tool_calls, "tool_results": all_tool_results,
                        "turns": turn, "tps": 0, "wall_time": time.time() - start_time, "error": chunk["data"]}
            elif chunk["type"] == "done":
                full_result.update(chunk["data"])
                got_result["text"] = chunk["data"].get("text", "")
                got_result["done"] = True
                tool_calls_from_ollama = chunk["data"].get("tool_calls")  # if Ollama returns them
                total_output_tokens += chunk["data"].get("output_tokens", 0) or 0
                break
            elif chunk["type"] == "token":
                response_text += chunk["data"]
            elif chunk["type"] == "thinking":
                pass  # skip thinking in agentic mode
        
        if not got_result["done"]:
            # Timed out or interrupted
            break
        
        # Parse tool calls from response
        current_text = got_result["text"] or response_text
        tool_calls = parse_tool_calls(current_text, tool_calls_from_ollama)
        
        if not tool_calls:
            # Model gave final text answer — done
            wall_time = time.time() - start_time
            avg_tps = total_output_tokens / wall_time if wall_time > 0 else 0
            return {
                "answer": current_text.strip(),
                "tool_calls": all_tool_calls,
                "tool_results": all_tool_results,
                "turns": turn + 1,
                "tps": round(avg_tps, 1),
                "wall_time": round(wall_time, 2),
            }
        
        # Execute tool calls and append to conversation
        # When Ollama returns native tool_calls, include them in the assistant
        # message so the conversation context is correct for the next turn
        assistant_msg = {"role": "assistant", "content": current_text or ""}
        if tool_calls_from_ollama:
            assistant_msg["tool_calls"] = tool_calls_from_ollama
        conversation.append(assistant_msg)
        
        for tc in tool_calls:
            func_name = tc.get("name", "")
            func_args = tc.get("arguments", {})
            
            # Execute the tool
            try:
                result = tool_executor(func_name, func_args)
            except Exception as e:
                result = {"error": str(e)}
            
            all_tool_calls.append({"name": func_name, "arguments": func_args})
            all_tool_results.append(result)
            
            # Add tool result to conversation
            # Include tool_call_id if present (Ollama native tool calling uses this)
            tool_msg = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
            tool_result_msg = {"role": "tool", "content": tool_msg}
            if tc.get("id"):
                tool_result_msg["tool_call_id"] = tc["id"]
            tool_result_msg["name"] = func_name
            conversation.append(tool_result_msg)
            
            if sse_callback:
                sse_callback({
                    "type": "agentic_tool_call",
                    "name": func_name,
                    "args": func_args,
                    "result_preview": str(result)[:200] if result else "",
                    "turn": turn + 1,
                })
    
    # Max turns reached
    wall_time = time.time() - start_time
    return {
        "answer": current_text.strip() if current_text else "",
        "tool_calls": all_tool_calls,
        "tool_results": all_tool_results,
        "turns": max_turns,
        "tps": round(total_output_tokens / wall_time, 1) if wall_time > 0 else 0,
        "wall_time": round(wall_time, 2),
        "max_turns_reached": True,
    }


# ─── Web Search Backend ─────────────────────────────────────────────────────

class WebSearchBackend:
    """Search backend for BFCL v4 web search evaluation.
    Supports SearXNG (self-hosted, free) and SerpAPI (cloud, paid).
    """
    
    def __init__(self, mode="searxng", searxng_url="http://localhost:8888",
                 serpapi_key=None, show_snippet=True):
        self.mode = mode
        self.searxng_url = searxng_url
        self.serpapi_key = serpapi_key
        self.show_snippet = show_snippet
    
    def __call__(self, func_name, args):
        """Route tool calls to backend methods."""
        if isinstance(args, dict):
            if func_name == "search_engine_query":
                return self.search_engine_query(**args)
            elif func_name == "fetch_url_content":
                return self.fetch_url_content(**args)
        return {"error": f"Unknown function: {func_name}"}
    
    def search_engine_query(self, keywords, max_results=10, region="wt-wt"):
        """Search the web. Returns list of {title, href, body} dicts."""
        if self.mode == "searxng":
            return self._searxng_search(keywords, max_results, region)
        elif self.mode == "serpapi":
            return self._serpapi_search(keywords, max_results, region)
        return {"error": f"Unknown search mode: {self.mode}"}
    
    def _searxng_search(self, keywords, max_results, region):
        """Search via self-hosted SearXNG instance."""
        try:
            resp = req_lib.get(f"{self.searxng_url}/search",
                params={"q": keywords, "format": "json"},
                timeout=10)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            formatted = []
            for r in results[:max_results]:
                entry = {"title": r.get("title", ""), "href": r.get("url", "")}
                if self.show_snippet:
                    entry["body"] = r.get("content", "")
                formatted.append(entry)
            return formatted
        except Exception as e:
            return {"error": f"SearXNG search error: {e}"}
    
    def _serpapi_search(self, keywords, max_results, region):
        """Search via SerpAPI (DuckDuckGo engine) — official BFCL method."""
        try:
            from serpapi import GoogleSearch
            params = {
                "engine": "duckduckgo",
                "q": keywords,
                "kl": region,
                "api_key": self.serpapi_key or os.getenv("SERPAPI_API_KEY"),
            }
            search = GoogleSearch(params)
            search_results = search.get_dict()
            if "organic_results" not in search_results:
                return {"error": "No results from SerpAPI"}
            results = search_results["organic_results"]
            formatted = []
            for r in results[:max_results]:
                entry = {"title": r.get("title", ""), "href": r.get("link", "")}
                if self.show_snippet:
                    entry["body"] = r.get("snippet", "")
                formatted.append(entry)
            return formatted
        except Exception as e:
            return {"error": f"SerpAPI error: {e}"}
    
    def fetch_url_content(self, url, mode="truncate"):
        """Fetch and process URL content."""
        if not url.startswith(("http://", "https://")):
            return {"error": f"Invalid URL: {url}"}
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            resp = req_lib.get(url, headers=headers, timeout=20, allow_redirects=True)
            resp.raise_for_status()
            
            if mode == "raw":
                return {"content": resp.text}
            elif mode == "markdown":
                try:
                    import html2text
                    converter = html2text.HTML2Text()
                    markdown = converter.handle(resp.text)
                    return {"content": markdown}
                except ImportError:
                    return {"content": resp.text}
            elif mode == "truncate":
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for tag in soup(["script", "style"]):
                        tag.extract()
                    text = soup.get_text(separator="\n", strip=True)
                    return {"content": text[:5000]}  # truncate to prevent token overflow
                except ImportError:
                    return {"content": resp.text[:5000]}
            else:
                return {"error": f"Unknown mode: {mode}"}
        except Exception as e:
            return {"error": f"Fetch error for {url}: {e}"}


# ─── Memory Backends ────────────────────────────────────────────────────────

# Constants from official gorilla repo
MAX_CORE_MEMORY_SIZE = 7
MAX_CORE_MEMORY_ENTRY_LENGTH = 300
MAX_ARCHIVAL_MEMORY_SIZE = 50
MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH = 2000


class MemoryBackendBase(ABC):
    """Base class for BFCL v4 memory evaluation backends."""
    
    @abstractmethod
    def execute_function(self, name, args):
        """Execute a memory function call. Returns result dict."""
        pass
    
    def __call__(self, func_name, args):
        return self.execute_function(func_name, args)
    
    @abstractmethod
    def load_scenario(self, initial_config):
        """Load test scenario's initial memory state."""
        pass
    
    @abstractmethod
    def get_state(self):
        """Get current memory state for comparison."""
        pass


class MemoryKVBackend(MemoryBackendBase):
    """Key-value memory backend for BFCL v4 memory_kv category.
    
    Core memory: small, limited entries (max 7, 300 chars each)
    Archival memory: larger storage (max 50, 2000 chars each) with BM25+ search
    
    API aligned with official gorilla repo: 15 functions total.
    """
    
    def __init__(self):
        self.core_memory = {}  # key -> value
        self.archival_memory = {}  # key -> value
    
    def load_scenario(self, initial_config):
        """Load initial memory state from test entry."""
        if isinstance(initial_config, dict):
            self.core_memory = dict(initial_config.get("core_memory", {}))
            self.archival_memory = dict(initial_config.get("archival_memory", {}))
    
    def get_state(self):
        return {
            "core_memory": dict(self.core_memory),
            "archival_memory": dict(self.archival_memory),
        }
    
    @staticmethod
    def _is_valid_key_format(s):
        """Check if the key is in snake_case format."""
        return bool(re.match(r"^[a-z]+(_[a-z0-9]+)*$", s))
    
    @staticmethod
    def _bm25_search(query, keys, k=5):
        """BM25+ search over key names."""
        try:
            from rank_bm25 import BM25Plus
            if not keys:
                return []
            corpus = list(keys)
            tokenized = [k.replace("_", " ").lower().split() for k in corpus]
            bm25 = BM25Plus(tokenized)
            q_tokens = query.replace("_", " ").lower().split()
            scores = bm25.get_scores(q_tokens)
            ranked = sorted(zip(scores, corpus), key=lambda x: x[0], reverse=True)
            return [{"key": k, "score": float(s)} for s, k in ranked[:k] if s > 0]
        except ImportError:
            # Fallback: simple substring search
            results = []
            for key in keys:
                if query.lower() in key.lower():
                    results.append({"key": key, "score": 1.0})
            return results[:k]
    
    def execute_function(self, name, args):
        if not isinstance(args, dict):
            args = {}
        
        # ─── Core memory functions ───────────────────────────────────
        if name == "core_memory_add":
            key = args.get("key", "")
            value = args.get("value", "")
            if not key:
                return {"error": "key is required"}
            if key in self.core_memory:
                return {"error": f"Key '{key}' already exists in core memory. Use core_memory_replace to update."}
            if len(self.core_memory) >= MAX_CORE_MEMORY_SIZE:
                return {"error": f"core memory full (max {MAX_CORE_MEMORY_SIZE} entries)"}
            if len(value) > MAX_CORE_MEMORY_ENTRY_LENGTH:
                return {"error": f"Value exceeds max length of {MAX_CORE_MEMORY_ENTRY_LENGTH} characters"}
            self.core_memory[key] = value
            return {"status": "ok", "key": key, "message": f"Added key '{key}' to core memory."}
        
        elif name == "core_memory_replace":
            key = args.get("key", "")
            old_value = args.get("old_value", "")
            new_value = args.get("new_value", "")
            if key not in self.core_memory:
                return {"error": f"Key '{key}' not found in core memory"}
            current = self.core_memory[key]
            if old_value not in current:
                return {"error": f"old_value not found in key '{key}'"}
            if len(new_value) > MAX_CORE_MEMORY_ENTRY_LENGTH:
                return {"error": f"New value exceeds max length of {MAX_CORE_MEMORY_ENTRY_LENGTH} characters"}
            self.core_memory[key] = current.replace(old_value, new_value)
            return {"status": "ok", "key": key, "message": f"Replaced value for key '{key}' in core memory."}
        
        elif name == "core_memory_remove":
            key = args.get("key", "")
            if key not in self.core_memory:
                return {"error": f"Key '{key}' not found in core memory"}
            del self.core_memory[key]
            return {"status": "ok", "key": key, "message": f"Removed key '{key}' from core memory."}
        
        elif name == "core_memory_retrieve":
            key = args.get("key", "")
            if key not in self.core_memory:
                return {"error": f"Key '{key}' not found in core memory"}
            return {"key": key, "value": self.core_memory[key]}
        
        elif name == "core_memory_retrieve_all":
            return {"key": list(self.core_memory.keys()), "value": list(self.core_memory.values())}
        
        elif name == "core_memory_list_keys":
            return {"keys": list(self.core_memory.keys())}
        
        elif name == "core_memory_key_search":
            query = args.get("query", "")
            results = self._bm25_search(query, list(self.core_memory.keys()))
            return {"ranked_results": results}
        
        elif name == "core_memory_clear":
            self.core_memory.clear()
            return {"status": "ok", "message": "Core memory cleared."}
        
        # ─── Archival memory functions ──────────────────────────────
        elif name == "archival_memory_add":
            key = args.get("key", "")
            value = args.get("value", "")
            if not key:
                return {"error": "key is required"}
            if key in self.archival_memory:
                return {"error": f"Key '{key}' already exists in archival memory. Use archival_memory_replace to update."}
            if len(self.archival_memory) >= MAX_ARCHIVAL_MEMORY_SIZE:
                return {"error": f"archival memory full (max {MAX_ARCHIVAL_MEMORY_SIZE} entries)"}
            if len(value) > MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH:
                return {"error": f"Value exceeds max length of {MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH} characters"}
            self.archival_memory[key] = value
            return {"status": "ok", "key": key, "message": f"Added key '{key}' to archival memory."}
        
        elif name == "archival_memory_replace":
            key = args.get("key", "")
            old_value = args.get("old_value", "")
            new_value = args.get("new_value", "")
            if key not in self.archival_memory:
                return {"error": f"Key '{key}' not found in archival memory"}
            current = self.archival_memory[key]
            if old_value not in current:
                return {"error": f"old_value not found in key '{key}'"}
            if len(new_value) > MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH:
                return {"error": f"New value exceeds max length of {MAX_ARCHIVAL_MEMORY_ENTRY_LENGTH} characters"}
            self.archival_memory[key] = current.replace(old_value, new_value)
            return {"status": "ok", "key": key, "message": f"Replaced value for key '{key}' in archival memory."}
        
        elif name == "archival_memory_remove":
            key = args.get("key", "")
            if key not in self.archival_memory:
                return {"error": f"Key '{key}' not found in archival memory"}
            del self.archival_memory[key]
            return {"status": "ok", "key": key, "message": f"Removed key '{key}' from archival memory."}
        
        elif name == "archival_memory_retrieve":
            key = args.get("key", "")
            if key not in self.archival_memory:
                return {"error": f"Key '{key}' not found in archival memory"}
            return {"key": key, "value": self.archival_memory[key]}
        
        elif name == "archival_memory_list_keys":
            return {"keys": list(self.archival_memory.keys())}
        
        elif name == "archival_memory_key_search":
            query = args.get("query", "")
            results = self._bm25_search(query, list(self.archival_memory.keys()))
            return {"ranked_results": results}
        
        elif name == "archival_memory_clear":
            self.archival_memory.clear()
            return {"status": "ok", "message": "Archival memory cleared."}
        
        return {"error": f"Unknown function: {name}"}


class MemoryRecSumBackend(MemoryBackendBase):
    """Recursive summarization memory backend for BFCL v4.
    
    Simplest backend: single string memory with append/replace/clear/retrieve/update.
    API aligned with official gorilla repo: 5 functions total.
    """
    
    def __init__(self):
        self.memory = ""
        self.max_chars = 10000
    
    def load_scenario(self, initial_config):
        if isinstance(initial_config, dict):
            # RecSum uses "summary" key, but also accept "memory" for compatibility
            self.memory = initial_config.get("summary", initial_config.get("memory", ""))
    
    def get_state(self):
        return {"summary": self.memory, "memory": self.memory}
    
    def execute_function(self, name, args):
        if not isinstance(args, dict):
            args = {}
        
        if name == "memory_retrieve":
            return {"content": self.memory}
        
        elif name == "memory_append":
            content = args.get("content", "")
            if len(self.memory) + len(content) > self.max_chars:
                return {"error": "memory full"}
            self.memory += content
            return {"status": "ok", "message": "Content appended to memory."}
        
        elif name == "memory_replace":
            old_content = args.get("old_content", "")
            new_content = args.get("new_content", "")
            if old_content not in self.memory:
                return {"error": "old_content not found in memory"}
            self.memory = self.memory.replace(old_content, new_content)
            return {"status": "ok", "message": "Content replaced in memory."}
        
        elif name == "memory_clear":
            self.memory = ""
            return {"status": "ok", "message": "Memory cleared."}
        
        elif name == "memory_update":
            content = args.get("content", "")
            if len(content) > self.max_chars:
                return {"error": f"Content exceeds max length of {self.max_chars} characters"}
            self.memory = content
            return {"status": "ok", "message": "Memory updated with new content."}
        
        return {"error": f"Unknown function: {name}"}


class MemoryVectorBackend(MemoryBackendBase):
    """Vector database memory backend for BFCL v4.
    
    Uses FAISS + sentence-transformers for semantic search.
    Lazy-loads heavy dependencies only when instantiated.
    API aligned with official gorilla repo: 12 functions total.
    """
    
    def __init__(self):
        self.core_entries = []   # list of text strings
        self.archival_entries = []  # list of text strings
        self._encoder = None
        self._core_index = None
        self._archival_index = None
    
    def _ensure_encoder(self):
        """Lazy-load sentence-transformers encoder."""
        if self._encoder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            raise RuntimeError("Memory vector backend requires: pip install sentence-transformers faiss-cpu")
    
    def _build_index(self, entries):
        """Build FAISS index from entries."""
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise RuntimeError("Memory vector backend requires: pip install faiss-cpu")
        self._ensure_encoder()
        if not entries:
            return None
        embeddings = self._encoder.encode(entries)
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index
    
    def _rebuild_core_index(self):
        self._core_index = self._build_index(self.core_entries)
    
    def _rebuild_archival_index(self):
        self._archival_index = self._build_index(self.archival_entries)
    
    def _search_index(self, index, entries, query, k=5):
        """Search a FAISS index."""
        if index is None or not entries:
            return {"results": []}
        try:
            import numpy as np
            import faiss
            self._ensure_encoder()
            q_emb = self._encoder.encode([query]).astype('float32')
            faiss.normalize_L2(q_emb)
            k = min(k, len(entries))
            scores, indices = index.search(q_emb, k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0:
                    results.append({"content": entries[idx], "score": float(scores[0][i])})
            return {"results": results}
        except Exception as e:
            return {"error": str(e)}
    
    def load_scenario(self, initial_config):
        if isinstance(initial_config, dict):
            self.core_entries = list(initial_config.get("core_entries", []))
            self.archival_entries = list(initial_config.get("archival_entries", []))
            if self.core_entries:
                self._ensure_encoder()
                self._rebuild_core_index()
            if self.archival_entries:
                self._ensure_encoder()
                self._rebuild_archival_index()
    
    def get_state(self):
        return {"core_entries": list(self.core_entries), "archival_entries": list(self.archival_entries)}
    
    def execute_function(self, name, args):
        if not isinstance(args, dict):
            args = {}
        
        # ─── Core memory functions ───────────────────────────────────
        if name == "core_memory_add":
            content = args.get("content", "")
            self.core_entries.append(content)
            self._rebuild_core_index()
            return {"status": "ok", "message": "Content added to core memory."}
        
        elif name == "core_memory_remove":
            content = args.get("content", "")
            if content in self.core_entries:
                self.core_entries.remove(content)
                self._rebuild_core_index()
                return {"status": "ok", "message": "Content removed from core memory."}
            return {"error": "Content not found in core memory"}
        
        elif name == "core_memory_update":
            old_content = args.get("old_content", "")
            new_content = args.get("new_content", "")
            if old_content in self.core_entries:
                idx = self.core_entries.index(old_content)
                self.core_entries[idx] = new_content
                self._rebuild_core_index()
                return {"status": "ok", "message": "Content updated in core memory."}
            return {"error": "old_content not found in core memory"}
        
        elif name == "core_memory_retrieve":
            query = args.get("query", "")
            return self._search_index(self._core_index, self.core_entries, query)
        
        elif name == "core_memory_retrieve_all":
            return {"content": "\n---\n".join(self.core_entries)}
        
        elif name == "core_memory_clear":
            self.core_entries.clear()
            self._core_index = None
            return {"status": "ok", "message": "Core memory cleared."}
        
        # ─── Archival memory functions ──────────────────────────────
        elif name == "archival_memory_add":
            content = args.get("content", "")
            self.archival_entries.append(content)
            self._rebuild_archival_index()
            return {"status": "ok", "message": "Content added to archival memory."}
        
        elif name == "archival_memory_remove":
            content = args.get("content", "")
            if content in self.archival_entries:
                self.archival_entries.remove(content)
                self._rebuild_archival_index()
                return {"status": "ok", "message": "Content removed from archival memory."}
            return {"error": "Content not found in archival memory"}
        
        elif name == "archival_memory_update":
            old_content = args.get("old_content", "")
            new_content = args.get("new_content", "")
            if old_content in self.archival_entries:
                idx = self.archival_entries.index(old_content)
                self.archival_entries[idx] = new_content
                self._rebuild_archival_index()
                return {"status": "ok", "message": "Content updated in archival memory."}
            return {"error": "old_content not found in archival memory"}
        
        elif name == "archival_memory_retrieve":
            query = args.get("query", "")
            return self._search_index(self._archival_index, self.archival_entries, query)
        
        elif name == "archival_memory_retrieve_all":
            return {"content": "\n---\n".join(self.archival_entries)}
        
        elif name == "archival_memory_clear":
            self.archival_entries.clear()
            self._archival_index = None
            return {"status": "ok", "message": "Archival memory cleared."}
        
        return {"error": f"Unknown function: {name}"}


# ─── Backend Factory ────────────────────────────────────────────────────────

def create_memory_backend(memory_type, initial_config=None):
    """Create a memory backend of the specified type."""
    if memory_type == "kv":
        backend = MemoryKVBackend()
    elif memory_type == "vector":
        backend = MemoryVectorBackend()
    elif memory_type == "rec_sum":
        backend = MemoryRecSumBackend()
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")
    if initial_config:
        backend.load_scenario(initial_config)
    return backend


def create_search_backend(mode="searxng", searxng_url="http://localhost:8888",
                          serpapi_key=None, show_snippet=True):
    """Create a web search backend."""
    return WebSearchBackend(mode=mode, searxng_url=searxng_url,
                            serpapi_key=serpapi_key, show_snippet=show_snippet)


def get_tools_for_suite(suite_key):
    """Get tool definitions for an agentic suite."""
    if "web_search" in suite_key:
        return WEB_SEARCH_TOOLS
    elif "memory_kv" in suite_key:
        return MEMORY_KV_TOOLS
    elif "memory_vector" in suite_key:
        return MEMORY_VECTOR_TOOLS
    elif "memory_rec_sum" in suite_key:
        return MEMORY_RECSUM_TOOLS
    return []