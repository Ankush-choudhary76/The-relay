# The Relay — Research Paper Assistant & Fault-Tolerant RAG Pipeline

Strictly grounded question answering over three target arXiv research papers with automated citation verification, calibrated stand-in simulation, genuine dual-provider switching (Stand-in vs Real Groq API), and clean mathematical notation formatting.

**Stack:** Python 3.11, FastAPI, Streamlit, BM25, Groq API, asyncio, Typer, Rich  
**Default Model:** `openai/gpt-oss-20b` (Groq free tier)  
**Target Research Papers:**
1. *Attention Is All You Need* (`arXiv:1706.03762v7`, 15 pages)
2. *Moshi: a speech-text foundation model for real-time dialogue* (`arXiv:2410.00037v1`, 66 pages)
3. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (`arXiv:2507.07955v2`, 39 pages)

---

## Key Features

- **Genuine Dual Model Provider Switching:**
  - **Stand-in (Safe Simulator):** In-process calibrated simulator matching Groq's LPU latency, 8,000 TPM limit, and failure scenarios without consuming real quota.
  - **Real Groq Hosted API:** Routes live to Groq's official API (`openai/gpt-oss-20b`), streaming tokens, capturing true TTFT and request telemetry.
  - **Zero Silent Fallback:** Server-side request validation rejects unknown providers and reports genuine errors rather than quietly falling back.
- **Mathematical Formatting Engine:**
  - Post-processing converts raw LaTeX markup into clean Unicode notation.
  - Square roots (`√dₖ`), matrix transpose (`QKᵀ`), superscript exponents (`x²`), subscripts (`dₖ`, `dᵥ`), Greek characters (`α`, `β`, `λ`, `θ`), and clean division fractions (`QKᵀ / √dₖ`).
  - Prominent equation blockquotes (`> **Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V**`).
- **Interactive Streamlit Web Dashboard:**
  - Runtime provider toggle (Stand-in vs Real Groq).
  - Retrieval parameter selection (Settings A, B, C).
  - Live metric telemetry: Status, Provider Mode (`Simulated` vs `Real API`), TTFT, Total Latency, and Token counts.
  - Expandable Verified Citation cards detailing arXiv ID, page, section, and supporting passages.
  - 7-Stage pipeline timing waterfall.
- **Fault-Tolerant Resilience:**
  - Token-bucket sliding window rate limiting (30 RPM, 8K TPM, 1,000 RPD).
  - Bounded exponential backoff with jitter and `Retry-After` header adherence.
  - Circuit breaker to prevent cascading failures.
  - Overall request deadline enforcement.
- **Comprehensive Test Suite:**
  - **71 passing unit and integration tests** covering pipeline stages, provider switching, math formatting, retrieval, and resilience.

---

## Quick Start

### Prerequisites

- Python 3.11 (`python3.11 -V`)
- A Groq account and API key (optional for Stand-in mode, required for Real Groq mode)

### 1. Create virtual environment and install dependencies

```bash
cd the-relay
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and enter your GROQ_API_KEY if testing live Groq API
```

Core `.env` configuration:

```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
PROVIDER_TYPE=standin
PROVIDER_TIMEOUT=30.0
MAX_RETRIES=3
MAX_CONCURRENT_REQUESTS=5
REQUEST_DEADLINE=60.0
CLIENT_RPM_LIMIT=30
CLIENT_TPM_LIMIT=8000
CLIENT_RPD_LIMIT=1000
```

### 3. Verify / Ingest Research Papers

Ensure the target papers are in `data/papers/`:
- `data/papers/1706.03762v7.pdf`
- `data/papers/2410.00037v1.pdf`
- `data/papers/2507.07955v2.pdf`

To download or verify from arXiv:
```bash
PYTHONPATH=src:. .venv/bin/python scripts/download_papers.py
```

### 4. Run Test Suite (71 Tests)

```bash
PYTHONPATH=src:. .venv/bin/pytest tests -v
```

All 71 tests pass:
- `tests/test_provider_switching.py` (7 tests: Stand-in routing, Real Groq routing, runtime toggle, missing key refusal, streaming, concurrency, invalid provider rejection)
- `tests/test_math_format.py` (32 tests: Unicode math formatting, square roots, transpose, fractions, subscripts)
- `tests/test_pipeline.py` (9 tests: End-to-end execution, validation, refusal, error states)
- `tests/test_citation_verifier.py`, `tests/test_retrieval.py`, `tests/test_rate_limiter_and_queue.py`, `tests/test_retry_and_resilience.py`, `tests/test_standin_service.py`, `tests/test_evidence_and_intent.py`, `tests/test_papers_and_chunking.py`, `tests/test_metrics_and_csv.py`

---

## Running The Relay

### Option A: Interactive Streamlit Web UI (Recommended)

Start the dashboard:
```bash
PYTHONPATH=src:. .venv/bin/streamlit run streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.
- Switch between **Stand-in (Safe Simulator)** and **Real Groq API** directly from the sidebar.
- Click any quick prompt or enter a custom engineering question.
- Inspect real-time TTFT, latency, token counts, and verified citations.

### Option B: Command-Line Interface (CLI)

Ask a question using the **Stand-in Simulator**:
```bash
PYTHONPATH=src:. .venv/bin/python src/relay/cli.py ask "What is Scaled Dot-Product Attention in the Transformer?" --provider standin
```

Ask a question using the **Real Groq API**:
```bash
PYTHONPATH=src:. .venv/bin/python src/relay/cli.py ask "What is Scaled Dot-Product Attention in the Transformer?" --provider groq
```

### Option C: FastAPI REST Server

Launch the REST server:
```bash
PYTHONPATH=src:. .venv/bin/uvicorn relay.server:app --host 127.0.0.1 --port 8000
```

Submit a request via `curl`:
```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the frame rate of the Mimi neural audio codec?",
    "provider": "groq",
    "retrieval_setting": "setting_b"
  }'
```

Stream tokens via Server-Sent Events (SSE):
```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Scaled Dot-Product Attention in the Transformer?",
    "provider": "groq"
  }'
```

---

## Stand-in Service Simulation & Load Testing

The stand-in simulator accurately reproduces Groq's hardware characteristics, latency curves, and quota thresholds locally without burning live API budget.

### Run Retrieval Evaluation (16 Questions × 3 Settings)

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_evaluation.py
```
Evaluates Settings A, B, and C across all benchmark questions and appends results to `results.csv`.

### Run Simulated Overload Scenarios

```bash
# Available scenarios: steady | burst | sustained_overload | rpm_halved | daily_exhausted | all
PYTHONPATH=src:. .venv/bin/python scripts/run_load_test.py burst
```

### Real Groq Baseline Measurement

```bash
PYTHONPATH=src:. .venv/bin/python scripts/measure_real_groq.py
```

---

## Project Structure

```
the-relay/
├── .env                         # Environment variables and API keys (gitignored)
├── .env.example                 # Configuration template
├── requirements.txt             # Project dependencies
├── results.csv                  # Benchmark and evaluation measurements
├── streamlit_app.py             # Streamlit interactive dashboard
├── data/
│   ├── papers/                  # Target PDFs (1706.03762v7, 2410.00037v1, 2507.07955v2)
│   ├── measurements/            # real_service.json, retrieval_comparison.json
│   ├── processed/               # Extracted and structured paper text
│   └── questions.jsonl          # 16 standard evaluation questions
├── src/relay/                   # Core pipeline package
│   ├── chunking.py              # Token-aware sliding window chunking
│   ├── citation_verifier.py     # Deterministic citation extractor & ground-truth verifier
│   ├── cli.py                   # Typer + Rich command-line interface
│   ├── config.py                # Pydantic BaseSettings configuration
│   ├── evidence.py              # Evidence threshold & sufficiency evaluator
│   ├── generation.py            # Grounded prompt builder & AnswerGenerator
│   ├── groq_provider.py         # Official Groq API HTTP client & token streamer
│   ├── intent.py                # Query intent classifier & refusal logic
│   ├── math_format.py           # Unicode mathematical formatting post-processor
│   ├── papers.py                # PyMuPDF PDF extractor
│   ├── pipeline.py              # 7-stage RelayPipeline with runtime provider routing
│   ├── provider_interface.py    # Common ModelProvider abstract interface
│   ├── queue_manager.py         # Concurrency slots & priority queue
│   ├── rate_limiter.py          # Sliding-window RPM, TPM, RPD rate limiter
│   ├── retrieval.py             # BM25 Okapi retrieval & phrase-boost reranker
│   ├── retry.py                 # Exponential backoff, jitter, and circuit breaker
│   ├── schemas.py               # Pydantic request/response/telemetry models
│   ├── server.py                # FastAPI REST & SSE endpoints
│   └── standin_provider.py      # In-process calibrated simulator provider
├── standin/                     # Stand-in simulator service & scenarios
│   ├── app.py                   # Standalone FastAPI simulation server
│   ├── quota.py                 # Sliding-window token tracking
│   ├── scenarios.py             # Pre-configured load & failure scenarios
│   └── simulator.py             # Calibrated responses with structured math
├── scripts/                     # Evaluation and benchmark runner scripts
├── tests/                       # Complete pytest suite (71 tests)
│   ├── test_citation_verifier.py
│   ├── test_evidence_and_intent.py
│   ├── test_math_format.py
│   ├── test_metrics_and_csv.py
│   ├── test_papers_and_chunking.py
│   ├── test_pipeline.py
│   ├── test_provider_switching.py
│   ├── test_rate_limiter_and_queue.py
│   ├── test_retrieval.py
│   ├── test_retry_and_resilience.py
│   └── test_standin_service.py
└── reports/                     # Technical research and analysis reports
```

---

## Configuration Reference

| Environment Variable | Default | Description |
|:---------------------|:--------|:------------|
| `GROQ_API_KEY` | `""` | Official Groq API key (never committed) |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Model identifier used for Groq generation |
| `PROVIDER_TYPE` | `standin` | Default startup provider (`standin` or `real`) |
| `PROVIDER_TIMEOUT` | `30.0` | Individual model provider request timeout (seconds) |
| `MAX_RETRIES` | `3` | Maximum retry attempts upon transient HTTP failures |
| `MAX_CONCURRENT_REQUESTS` | `5` | Maximum active pipeline generation tasks |
| `REQUEST_DEADLINE` | `60.0` | Maximum overall request end-to-end deadline |
| `CLIENT_RPM_LIMIT` | `30` | Client-side Requests Per Minute ceiling |
| `CLIENT_TPM_LIMIT` | `8000` | Client-side Tokens Per Minute ceiling (matches free quota) |
| `CLIENT_RPD_LIMIT` | `1000` | Client-side Requests Per Day ceiling |
| `CHUNK_SIZE_TOKENS` | `600` | Default chunk size (Setting B) |
| `RETRIEVAL_TOP_K` | `5` | Default number of retrieved passages passed to generation |
| `EVIDENCE_THRESHOLD` | `0.05` | Minimum BM25 relevance score required to prevent refusal |
