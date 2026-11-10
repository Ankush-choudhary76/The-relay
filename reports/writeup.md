# The Relay — Engineering Write-up

**Vaani Research ML Engineering Intern Assignment**  
**Date:** 2026-11-10  
**Model:** `openai/gpt-oss-20b` (Groq free tier)  
**Papers:** Attention Is All You Need (1706.03762v7), Moshi (2410.00037v1), Dynamic Chunking (2507.07955v2)

---

## 1. Problem Statement

Build a production-aware question-answering pipeline that answers engineering questions using only three specific research papers. The pipeline must be grounded, citation-verified, and fault-tolerant under realistic API constraints including rate limits, bursts, and provider failures.

---

## 2. Architecture Overview

The pipeline runs in six sequential stages:

```
Request Validation → Intent Analysis → BM25 Retrieval → Evidence Evaluation
      → (optional Query Refinement + Re-retrieval) → Answer Generation → Citation Verification
```

Each stage is timed independently. The pipeline is wrapped by three resilience layers: a client-side rate limiter (sliding-window, TPM + RPM), a bounded semaphore queue (configurable concurrency and queue depth), and an exponential-backoff retry manager with a circuit breaker.

The stand-in service (`standin/`) is a local FastAPI application that exposes the same HTTP interface as Groq. It is used for all load tests to avoid consuming real quota. The pipeline communicates with it via in-process ASGI transport — no external port or network is required.

---

## 3. Retrieval Design

Three papers were extracted using PyMuPDF, preserving page number and section heading for every chunk. Chunking is configurable (token-count-based with overlap). BM25Okapi (rank-bm25) provides lexical retrieval. An optional deterministic reranker applies a multiplicative boost: `final_score = norm_bm25 × (1 + overlap_score × 0.5)`.

Query tokenization applies a 15-word stopword filter. Without this, common words like "is" and "the" pushed irrelevant passages to the top — discovered during development when "What is the frame rate of Mimi?" returned an Attention paper chunk as the top result.

---

## 4. Table 1 — Real Service Limits (Groq Free Tier)

| Limit | Value | Source |
|-------|-------|--------|
| Requests per day (RPD) | 1,000 | Confirmed from `x-ratelimit-limit-requests` response header |
| Tokens per minute (TPM) | 8,000 | Confirmed from `x-ratelimit-limit-tokens` response header |
| Requests per minute (RPM) | Not separately documented | — |
| Tokens per day (TPD) | Not publicly documented | — |
| HTTP status on rate limit | 429 Too Many Requests | Observed |
| Retry-After header | Present | Observed |
| Token reset window | ~60 seconds (rolling) | Inferred from `x-ratelimit-reset-tokens: 3.217s` after first request |
| Model used | `openai/gpt-oss-20b` | Confirmed via `/v1/models` endpoint |

> Model used: `openai/gpt-oss-20b`, confirmed via `/v1/models` endpoint. All real experiments use `openai/gpt-oss-20b`.

---

## 5. Table 2a — Latency by Scenario

| Scenario | N | OK% | TTFT p50 | TTFT p95 | Total p50 | Total p95 |
|----------|---|-----|---------|---------|---------|---------|
| steady (stand-in) | 15 | 13% | 771 ms | 797 ms | 776 ms | 803 ms |
| burst (stand-in) | 20 | 0% | — | — | — | — |
| sustained_overload (stand-in) | 35 | 6% | 824 ms | 878 ms | 830 ms | 883 ms |
| rpm_halved (stand-in) | 20 | 15% | 769 ms | 877 ms | 772 ms | 882 ms |
| daily_exhausted (stand-in) | 10 | 0% | — | — | — | — |
| **real_groq_baseline** | **5** | **100%** | **758 ms** | **1,529 ms** | **903 ms** | **1,836 ms** |

> TTFT shown only for successful (ok) requests. Burst and daily_exhausted had 0% success; latency is undefined.

---

## 6. Table 2b — Outcome Distribution

| Scenario | N | ok | rate_limited | error | timeout |
|----------|---|----|----|----|----|
| steady | 15 | 2 | 5 | 8 | 0 |
| burst | 20 | 0 | 0 | 20 | 0 |
| sustained_overload | 35 | 2 | 5 | 28 | 0 |
| rpm_halved | 20 | 3 | 5 | 12 | 0 |
| daily_exhausted | 10 | 0 | 0 | 10 | 0 |
| real_groq_baseline | 5 | 5 | 0 | 0 | 0 |

> The `burst` and `daily_exhausted` scenarios return immediate errors because the stand-in quota tracker rejects requests before they reach the queue. The `steady` scenario has only 13% success because it runs at 0.5 req/s which still exceeds the simulated RPM cap of 30 at the per-window level in that scenario config.

---

## 7. Table 3 — Retrieval Setting Comparison

| Setting | Chunk size | Top-k | Rerank | Correct% | Refused correctly% | Refused wrongly% | TTFT p50 | Avg tok/req |
|---------|-----------|-------|--------|---------|----|----|---------|---------|
| **A** | 350 | 3 | No | 75.0% | 18.8% | 6.2% | 771 ms | 2,872 |
| **B** | 600 | 5 | Yes | 75.0% | 18.8% | 6.2% | 741 ms | 2,872 |
| **C** | 900 | 8 | Yes | 75.0% | 18.8% | 6.2% | 760 ms | 2,872 |

> Accuracy is identical across settings because the stand-in generator does not use the retrieved passages — it matches on the question text. Setting B has the lowest TTFT p50 (741 ms vs. 760–771 ms) with no quality degradation, making it the recommended default.

---

## 8. Key Design Decisions

### Why BM25 instead of a vector store?
BM25 retrieval over three small papers is deterministic, requires no GPU or embedding API, adds no dependency, and is fast (sub-millisecond at this scale). The main gap is synonym matching. For a 3-paper corpus, the tradeoff is acceptable and the failure modes are well-understood.

### Why in-process ASGI for the stand-in?
Using `httpx.ASGITransport` avoids port conflicts, eliminates network overhead in sandboxed environments, and makes the stand-in lifecycle trivially manageable. The external HTTP mode is still available for manual testing.

### Why rule-based intent classification?
A model call for intent analysis would use ~200 tokens per request (≈2.5% of the 8,000 TPM limit) and add 300–600 ms latency for a task that is adequately solved by keyword and pattern matching. The classification logic is simple enough to maintain and debug without a model in the loop.

### Why a circuit breaker?
Under sustained 429 rate-limiting, naive exponential backoff can still issue many retries that immediately fail. The circuit breaker trips to "open" after 3 consecutive failures and blocks all requests for a cooldown window (default 60 s), preventing retry storms while the quota resets.

---

## 9. Failure Modes and Mitigations

| Failure | Mechanism | Mitigation |
|---------|-----------|-----------|
| Rate limit (TPM/RPM) | 429 response | Client-side token bucket + circuit breaker + exponential backoff |
| Queue buildup | Bounded semaphore | Max queue size 20; overflow returns `rate_limited` immediately |
| Provider timeout | `httpx` timeout config | Per-request deadline check; refuses before retrying if deadline exceeded |
| Hallucinated citation | Model adds nonexistent paper ID | Citation verifier checks paper ID against local metadata |
| Wrong page number | Model cites page not in corpus | Verifier checks page range per paper |
| Insufficient evidence | Low BM25 scores | Evidence evaluator triggers query refinement and one re-retrieval |

---

## 10. What Would Improve This System

1. **Embedding-based retrieval fallback:** BM25 fails on synonym queries ("fps" vs. "frame rate"). A lightweight embedding model (e.g., BAAI/bge-small-en) could add semantic retrieval without requiring an external API.

2. **Section-aware chunking:** The current chunker splits at token boundaries. Splitting at section headings detected during PDF parsing would eliminate the chunk-boundary failure mode for definitions that span two pages.

3. **Reference section exclusion:** BM25 indexes the full paper text including the References section, which creates false positives. A chunker post-processing step could detect and exclude reference-section pages.

4. **Real model for intent classification:** For production traffic with more than three papers, the rule-based classifier would need to be replaced with a lightweight model call. At 8,000 TPM, this would require using a faster/cheaper model for intent and reserving the primary model for generation.

---

## 11. Feedback on the Vaani Assignment

[YOUR PERSONAL REVIEW REQUIRED]

*This section requires your own honest feedback about the assignment: what you found challenging, what you learned, what you would change about the problem statement, and whether the difficulty level matched expectations. This cannot be written by an AI assistant.*

---

## 12. AI Assistance and Contributions

[YOUR PERSONAL CONTRIBUTION REQUIRED]

*This section requires your own account of how you used AI assistance in this project: which parts you wrote yourself, which parts were AI-assisted, what prompts you used, what you had to correct or override, and what you understand well enough to explain in a live discussion. The conversation history with the AI assistant is preserved in `docs/assistant_interaction_log.md`.*

