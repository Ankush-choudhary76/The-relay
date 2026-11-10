# Architecture — The Relay

## Pipeline Stages

```
                    ┌──────────────────────────────────────┐
                    │           USER QUESTION               │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  STAGE 1: REQUEST VALIDATION          │
                    │  - Empty / oversized / invalid chars  │
                    │  - Assign unique request_id           │
                    │  - Start deadline clock               │
                    └──────────────┬───────────────────────┘
                                   │ valid
                    ┌──────────────▼───────────────────────┐
                    │  STAGE 2: INTENT ANALYSIS             │
                    │  - Rule-based classifier              │
                    │  - Categories: factual, architectural,│
                    │    comparative, definition, multipart,│
                    │    unanswerable                       │
                    │  - Fast exit for unanswerable intent  │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  STAGE 3: BM25 RETRIEVAL (TOOL)       │
                    │  - BM25Okapi over 3 paper corpus      │
                    │  - Stopword-filtered query tokens     │
                    │  - Configurable top-k (3 / 5 / 8)    │
                    │  - Optional deterministic reranking   │
                    │  - Returns: passage, paper_id, page,  │
                    │    section, bm25_score, final_score   │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  STAGE 4: EVIDENCE EVALUATION         │
                    │  - Score threshold check              │
                    │  - Duplicate passage deduplication    │
                    │  - Multi-part coverage check          │
                    └──────┬───────────────────────────────┘
                           │                     │
                    sufficient             insufficient
                           │                     │
                           │       ┌─────────────▼──────────────┐
                           │       │  QUERY REFINEMENT           │
                           │       │  - Expand/rephrase query    │
                           │       │  - Re-retrieve (once only)  │
                           │       └─────────────┬──────────────┘
                           │                     │ still insufficient
                           │                     ▼
                           │             REFUSE (no fabrication)
                           │
                    ┌──────▼───────────────────────────────┐
                    │  STAGE 5: ANSWER GENERATION (Groq)    │
                    │  - Strict grounded-only prompt        │
                    │  - Use only retrieved passages        │
                    │  - Cite paper + page + section        │
                    │  - Streaming supported                │
                    │  - Wrapped by: rate limiter, queue,   │
                    │    retry manager, circuit breaker     │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  STAGE 6: CITATION VERIFICATION       │
                    │  - Extract citations from answer text │
                    │  - Validate paper_id exists in corpus │
                    │  - Validate page number in range      │
                    │  - Check supporting passage in text   │
                    │  - Invalid citations → refuse answer  │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  STREAM RESPONSE TO USER              │
                    │  - answer, citations, timings, status │
                    └──────────────────────────────────────┘
```

## Resilience Layers

```
         REQUEST
            │
   ┌────────▼─────────┐
   │  ClientRateLimiter│  ← sliding window TPM + RPM + RPD + TPD
   └────────┬─────────┘
            │ within limit
   ┌────────▼─────────┐
   │   QueueManager   │  ← asyncio.Semaphore, max_concurrent, max_queue_size
   └────────┬─────────┘
            │ slot acquired
   ┌────────▼─────────┐
   │   RetryManager   │  ← exponential backoff + jitter
   │  CircuitBreaker  │  ← trips after 3 consecutive failures
   └────────┬─────────┘
            │
   ┌────────▼─────────┐
   │  Model Provider  │  ← Groq API (real) or Stand-in (ASGI)
   └──────────────────┘
```

## Module Map

```
the-relay/
├── src/relay/
│   ├── config.py          # Pydantic-settings config (all from .env)
│   ├── schemas.py         # Pydantic models for all pipeline I/O
│   ├── papers.py          # PyMuPDF PDF extractor → page text dict
│   ├── chunking.py        # Token-count chunker with overlap
│   ├── retrieval.py       # BM25Okapi + deterministic reranker
│   ├── intent.py          # Rule-based intent classifier
│   ├── evidence.py        # Evidence evaluator + query refinement
│   ├── generation.py      # AnswerGenerator: grounded prompt builder
│   ├── citation_verifier.py  # Citation extraction + validation
│   ├── provider_interface.py # Abstract ModelProvider base class
│   ├── groq_provider.py   # Real Groq streaming client (httpx)
│   ├── standin_provider.py   # Stand-in ASGI transport client
│   ├── rate_limiter.py    # Sliding-window rate limiter
│   ├── queue_manager.py   # Bounded semaphore + queue
│   ├── retry.py           # Exponential backoff + circuit breaker
│   ├── pipeline.py        # 6-stage pipeline orchestration
│   ├── metrics.py         # MetricsRecorder + CSV summary
│   ├── evaluation.py      # Evaluation harness + grader
│   ├── server.py          # FastAPI HTTP server
│   └── cli.py             # Typer CLI (ask, evaluate, load-test)
│
├── standin/
│   ├── app.py             # FastAPI stand-in service
│   ├── simulator.py       # Keyword-matched answer generator
│   ├── quota.py           # In-memory quota tracker (RPM/TPM/RPD)
│   └── scenarios.py       # 5 load scenario configurations
│
├── scripts/
│   ├── download_papers.py    # Download 3 arXiv PDFs (requires network)
│   ├── ingest_papers.py      # Extract text, save to data/processed/
│   ├── measure_real_groq.py  # 5-request real Groq baseline (uses quota)
│   ├── run_evaluation.py     # Run 16 questions × 3 retrieval settings
│   ├── run_load_test.py      # Run 1 or all 5 load scenarios
│   └── generate_reports.py   # Print Table 2a, 2b, 3 from results.csv
│
├── tests/                    # pytest test suite (32 tests)
├── data/papers/              # 3 PDF files + metadata.json
├── data/measurements/        # real_service.json, retrieval_comparison.json
├── results.csv               # All experiment results (153 rows)
└── reports/                  # groq_limits.md, standin_calibration.md,
                              #   retrieval_analysis.md, writeup.md
```

## Data Flow — results.csv

Every pipeline execution writes one row to `results.csv`. Columns:

| Column | Description |
|--------|-------------|
| `request_id` | Unique request identifier |
| `scenario` | Load scenario or `retrieval_eval` or `steady_baseline` |
| `source` | `real` or `stand-in` |
| `start_time` | ISO 8601 timestamp |
| `first_token_ms` | Time from request start to first output token |
| `total_ms` | Total pipeline wall-clock time |
| `status` | `ok`, `rate_limited`, `timeout`, `error` |
| `retries` | Number of retry attempts made |
| `tokens_in` | Input tokens reported by provider |
| `tokens_out` | Output tokens reported by provider |
| `retrieval_setting` | `setting_a`, `setting_b`, or `setting_c` |
| `answer_grade` | `correct`, `wrong`, `refused_correctly`, `refused_wrongly`, or `""` |
| `queue_wait_ms` | Time spent waiting for a queue slot |
| `provider_working_ms` | Time the provider was actually generating |
| `retry_wait_ms` | Total time spent in retry backoff |
| `peak_queue_len` | Maximum observed queue depth during request |

