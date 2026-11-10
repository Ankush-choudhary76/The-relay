# Stand-in Service Calibration

**Document purpose:** Explain how the stand-in service was calibrated against real Groq measurements, what it accurately simulates, and where it diverges from real behavior.

---

## 1. What the Stand-in Service Is

The stand-in service (`standin/`) is a local FastAPI application that exposes the same `/openai/v1/chat/completions` endpoint as Groq. It is used for load testing and reproducible experimentation without consuming real API quota.

The stand-in is **not** a language model. It generates answers by keyword pattern-matching the user's question text against known categories, then returns pre-written synthetic answers with injected delays.

The pipeline communicates with the stand-in via `httpx.ASGITransport` (in-process transport), not an external HTTP connection. This avoids network overhead and port conflicts in sandboxed environments.

---

## 2. Calibration Parameters

All timing parameters in `standin/scenarios.py` were calibrated against the 5-request real Groq baseline collected on 2026-11-10 (`data/measurements/real_service.json`).

### 2.1 Base Latency

| Parameter | Value | Derivation |
|-----------|-------|-----------|
| `base_ttft_ms` | 680 ms | Real TTFT p50 ≈ 758 ms, minus ~80 ms for in-process overhead |
| `jitter_ms` | ±120 ms | Covers observed range 626 ms – 1,719 ms |
| `tokens_per_ms` | ~2.0 tok/ms | Approximated from real response timing |

The stand-in models TTFT with:  
`simulated_ttft = base_ttft_ms + uniform(-jitter_ms, +jitter_ms)`

For output length, total time is extended proportionally:  
`simulated_total = simulated_ttft + (output_tokens / tokens_per_ms)`

### 2.2 Quota Limits

| Parameter | Stand-in Value | Real Groq Value | Source |
|-----------|---------------|-----------------|--------|
| Requests per day (RPD) | 1,000 | 1,000 | Confirmed from `x-ratelimit-limit-requests` header |
| Tokens per minute (TPM) | 8,000 | 8,000 | Confirmed from `x-ratelimit-limit-tokens` header |
| Requests per minute (RPM) | 30 | Not documented | Inferred from load test behavior |
| Tokens per day (TPD) | Not enforced | Not documented | — |

### 2.3 Load Slowdown Factor

Under concurrent load, the stand-in adds:  
`extra_delay_ms = concurrent_requests * 50`

This models increasing provider latency under load, though it is a rough approximation. Real Groq LPU latency behavior under concurrent requests was not measured (only 5 sequential requests were used in the baseline).

---

## 3. The Five Load Scenarios

| Scenario | Requests | Rate / Concurrency | Intent |
|----------|----------|-------------------|--------|
| `steady` | 15 | 0.5 req/s | Baseline, within quota |
| `burst` | 20 | 20 concurrent, 0.06s | Instantaneous burst — exhausts queue |
| `sustained_overload` | 35 | 90 RPM effective | Sustained above RPM limit |
| `rpm_halved` | 20 | 1 req/2s, RPM cap=15 | Quota restriction mid-experiment |
| `daily_exhausted` | 10 | 10 concurrent | Simulates day-quota already spent |

For `burst` and `daily_exhausted`, the stand-in immediately returns errors (HTTP 429 or 503) without generating an answer. For `rpm_halved` and `sustained_overload`, the stand-in rate-limits intermittently — some requests succeed, most are rejected or queued.

---

## 4. Answer Generation

The stand-in does **not** run a language model. It uses keyword pattern matching on the question text (extracted after the `QUESTION:` marker in the prompt) to select a canned response:

- Questions about "attention", "transformer", "query key value" → Attention Is All You Need canned answer
- Questions about "moshi", "mimi", "audio", "speech", "duplex" → Moshi canned answer
- Questions about "dynamic chunk", "hierarchical", "sequence modeling" → Dynamic Chunking canned answer
- Questions triggering "unanswerable" patterns → Explicit refusal

Canned answers always include properly formatted citations matching the paper IDs in `data/papers/metadata.json`. This is intentional — the goal of stand-in is to exercise the pipeline infrastructure (queuing, rate limiting, retries, citation verification), not to test answer quality.

**Consequence for evaluation:** The 75% correct / 18.8% refused-correctly results in `data/measurements/retrieval_comparison.json` reflect the stand-in's keyword matching accuracy, not retrieval quality. The retrieval settings (A/B/C) affect what passages are passed to the generator, but the stand-in generator ignores the passages and matches on the question text only. This is a known limitation; the stand-in is not intended to substitute for a real model in quality evaluation.

---

## 5. Key Differences from Real Groq

| Aspect | Stand-in | Real Groq |
|--------|----------|-----------|
| Token inference | None — keyword pattern match | Actual LLM inference on LPU hardware |
| Answer quality | Fixed canned responses | Variable; depends on model, temperature, context |
| Latency source | `asyncio.sleep()` with jitter | Hardware + network RTT |
| Token count | Estimated from canned answer length | Actual BPE tokenization |
| Rate limit enforcement | In-memory counter | Server-side rate limiter with rolling window |
| 429 behavior | Immediate rejection | May include `Retry-After` header with actual wait time |
| Streaming | Simulated (chunked sleep) | Real chunk-by-chunk token streaming |

---

## 6. How to Run the Stand-in

**External HTTP mode** (for testing with curl or separate processes):
```bash
PYTHONPATH=the-relay/src:the-relay the-relay/.venv/bin/uvicorn standin.app:app --host 127.0.0.1 --port 8001
```

**In-process ASGI mode** (used by the pipeline and load tests):
Set `PROVIDER_TYPE=standin` in `.env`. The `StandinProvider` uses `httpx.ASGITransport(app=standin_app)` to route requests in-process.

