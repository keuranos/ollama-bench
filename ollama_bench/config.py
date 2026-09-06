#!/usr/bin/env python3
"""Configuration constants and benchmark suite definitions."""

import json
from pathlib import Path

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
