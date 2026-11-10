# Design Decisions Log

This log records the significant technical decisions made during development of The Relay, with the rationale for each choice.

---

## D1: BM25 for retrieval (not vector embeddings)

**Decision:** Use BM25Okapi (rank-bm25) as the sole retrieval engine.

**Rationale:**
- The corpus is small (3 papers, ~120 total pages). Dense retrieval would not provide a meaningful accuracy advantage at this scale.
- BM25 is deterministic — the same query always produces the same ranked list, making experiments reproducible without an embedding API.
- No external API, GPU, or additional dependency is required.
- The main limitation (no synonym matching) is acceptable for a technical paper QA system where precise terminology is the norm.

**Rejected alternatives:** FAISS with sentence-transformers — adds 500 MB model download, GPU dependency, and embedding API latency for minimal gain over 3 papers.

---

## D2: Rule-based intent classifier (not an LLM call)

**Decision:** Classify question intent using keyword and pattern matching rather than a Groq model call.

**Rationale:**
- The free-tier TPM limit is 8,000 tokens/min. An intent classification call would cost ~200 tokens (2.5% of the limit) and add 300–600 ms latency per request.
- The 6 intent categories can be reliably detected by pattern matching for the domain (technical paper QA). "What is X?" → definition; "Compare X and Y" → comparative; "How does X work?" → architectural, etc.
- If the classifier is wrong, the pipeline recovers gracefully — the evidence evaluator catches irrelevant retrieval results.

**Trade-off accepted:** Rule-based classification fails on ambiguous questions. For this 3-paper corpus, the failure rate is low enough that the token savings justify it.

---

## D3: In-process ASGI transport for stand-in

**Decision:** Route pipeline requests to the stand-in via `httpx.ASGITransport(app=standin_app)` rather than binding to `127.0.0.1:8001`.

**Rationale:**
- In sandboxed development environments, external HTTP calls to localhost may be blocked or unreliable.
- In-process transport eliminates port conflicts when running multiple test scenarios.
- Latency difference is negligible (both are essentially function calls; ASGI transport has ~0.1 ms overhead vs. real network).
- The external HTTP mode is still available for manual testing by running `uvicorn standin.app:app`.

**Trade-off accepted:** In-process transport means the stand-in cannot be observed with external tools (curl, Postman) during automated tests. This is a development-only trade-off; the external mode is used in the README instructions.

---

## D4: Multiplicative reranking formula

**Decision:** Reranking score formula is `final_score = norm_bm25 × (1 + overlap_score × 0.5)`.

**Rationale:**
- The initial formula `0.6 × bm25 + 0.4 × rerank` caused a test regression: a chunk with perfect BM25 score (1.0) and no rerank boost (0.0) received `0.6`, while a chunk with BM25=0.8 and rerank=0.5 received `0.68`. The reranker was reducing the score of the top BM25 result.
- The multiplicative formula ensures that reranking can only boost a chunk, never demote the BM25-dominant result.
- The 0.5 boost factor was chosen to limit the maximum reranking effect to 50% above the BM25 score, keeping BM25 the primary signal.

---

## D5: Sliding-window rate limiter over token bucket

**Decision:** Implement client-side rate limiting with a sliding window (deque of timestamps) for requests, and a time-decay counter for tokens.

**Rationale:**
- Groq's reset headers (`x-ratelimit-reset-tokens: 3.217s`) indicate a rolling window, not a fixed-interval bucket. A sliding window is a more accurate model.
- Token bucket implementations can allow burst usage that exceeds the actual sliding-window limit.
- The sliding window implementation is straightforward to reason about and test.

---

## D6: Circuit breaker threshold = 3 consecutive failures

**Decision:** Circuit breaker trips to "open" after 3 consecutive errors, with a 60-second cooldown.

**Rationale:**
- With 1,000 RPD, each failed request consumes quota even on 429 responses. Tripping early reduces wasted quota.
- 3 is conservative — a single transient failure doesn't trip the breaker, but a sustained failure pattern does.
- 60-second cooldown aligns approximately with the TPM reset window observed in Groq response headers (~60s for the token counter).

---

## D7: Groq model selection — `openai/gpt-oss-20b`

**Decision:** Use `openai/gpt-oss-20b` verified as active on the account.

**Rationale:**
- The assignment did not mandate a specific model — it said "a model currently available on the free tier, verified through official Groq documentation."
- `openai/gpt-oss-20b` was confirmed available via `GET /v1/models` with the project API key.
- The model produces coherent, citation-formatted answers at the low temperatures (0.2) used by the generation stage.
- All reported measurements are explicitly for this model.

---

## D8: Conservative citation verification

**Decision:** Citation verification is conservative — it refuses an answer if extracted citations contain paper IDs not in the local metadata, or if page numbers exceed the paper's known page count.

**Rationale:**
- The alternative (lenient verification that only checks syntactic form) would allow the model to fabricate plausible-looking citations. The assignment explicitly prohibits guessing.
- The limitation of this approach is documented: verification checks paper ID and page range, but cannot semantically verify that the cited passage actually supports the claim. Full semantic verification would require a second model call.
- When verification fails, the pipeline returns the standard refusal message rather than the unverified answer.

---

## D9: 15-word stopword list for BM25 query tokenization

**Decision:** Filter a fixed 15-word stopword list from query tokens before BM25 scoring.

**Rationale:**
- Discovered during development: the query "What is the frame rate of Mimi?" returned an Attention paper chunk as the top result because "is", "the", and "of" scored highly on a dense, generic passage from the Attention paper.
- Filtering stopwords from the query (not from the corpus) is the minimal fix — it doesn't change the indexed corpus, only what the query asks for.
- The 15-word list was chosen as the minimum necessary to fix observed regressions without over-filtering technical terms.
- Corpus-side stopword filtering was rejected because it would affect IDF calculation and could lower scores for legitimate technical passages.

