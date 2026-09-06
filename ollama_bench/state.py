#!/usr/bin/env python3
"""In-memory state for active benchmark/suite/judge/profile runs."""

import threading
import time

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
        self.search_mode = "searxng"
        self.search_url = "http://localhost:8888"

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

