# ollama-bench

Benchmark harness for locally-served Ollama models: standardized quality
suites (MMLU-Pro, IFEval, BFCL v3/v4 + memory extensions), throughput
measurement, interactive human rating, side-by-side comparison, and
CSV/PDF export — driven by a FastAPI web UI.

Runs entirely on your own hardware against your own Ollama server.
Developed on 2013-era Xeon + Tesla V100 GPUs; no cloud anything.

## Features

- **Quality suites** — MMLU-Pro (12K MCQ, 14 categories), IFEval (541
  verifiable-constraint prompts), BFCL v3 function calling, BFCL v4
  (incl. agentic memory: KV / recursive-summarization / vector variants,
  and a web-search extension)
- **Throughput + timing** — tokens/s, TTFT, per-question latency, GPU
  utilization sampled during runs (`gpu_sampler.py`, NVIDIA)
- **Interactive rating** — after each answer, rate quality 4–10 having
  read the full output; stored alongside automated scores
- **Comparison** — head-to-head model vs model on any suite, win rates
  and deltas
- **Export** — CSV and PDF reports; SQLite persistence for every run
- **Agentic extensions** (`agentic.py`) — optional SerpAPI-backed
  search-augmented evaluation (key via `SERPAPI_API_KEY` env, never
  hardcoded)
- **Single-file or package** — run the self-contained monolith, or the
  modular `ollama_bench` package (v5 refactor), same DB schema

## Quick start

```bash
pip install fastapi uvicorn sse-starlette requests          # + fpdf2 for PDF export
python3 ollama_bench_web.py --host http://localhost:11434 --port 8501
# open http://localhost:8501/
```

Or the modular variant:

```bash
python3 -m ollama_bench.app --port 8501
```

Point it at any Ollama server. Models are pulled live from `/api/tags`.

## Suites

| Suite | What it measures | Size |
|---|---|---|
| MMLU-Pro | knowledge + reasoning (MCQ, 10 options) | 20 q/category sampled |
| IFEval | instruction following with verifiable constraints | 541 prompts |
| BFCL v3 | tool/function calling | 2,485 questions, 6 categories |
| BFCL v4 | next-gen function calling + multi-turn | sampled per category |
| BFCL v4 memory | agentic memory: KV, rec-sum, vector | sampled |
| BFCL v4 web search | search-augmented tool use | sampled |

Sample sizes are configurable in `ollama_bench/suites.py`; datasets load
from HuggingFace on first use and are cached.

## Configuration

All defaults live in `ollama_bench/config.py`:
`DEFAULT_OLLAMA_HOST` (default `http://localhost:11434`), DB path
(`ollama_bench.db` next to the code), timeouts, and `HIGH_NUM_PREDICT`
(32k token ceiling for long generations).

## Repository layout

```
ollama_bench_web.py     # self-contained monolith (v4, everything in one file)
ollama_bench/           # modular package (v5 refactor)
├── app.py              # FastAPI routes + SSE endpoints
├── suites.py           # suite definitions, dataset loading, scoring
├── tests.py            # test management
├── runner.py           # benchmark execution
├── ollama_api.py       # Ollama HTTP client
├── db.py               # SQLite persistence
├── gpu_sampler.py      # NVIDIA utilization sampling during runs
├── agentic.py          # search-augmented agentic eval extensions
├── compare.py          # model-vs-model comparison
├── export.py           # CSV / PDF export
└── frontend.py         # the web UI
```

## Notes

- Benchmark datasets load from HuggingFace (`TIGER-Lab/MMLU-Pro`,
  `google/IFEval`, BFCL). First run downloads and caches them.
- Results accumulate in SQLite; every run is reproducible from the DB
  (model, suite, seed, timestamps, per-question rows).
- The interactive rating slider is deliberately manual: automated scores
  and human judgment are stored as separate columns, never merged.

## License

MIT
