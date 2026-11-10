# Retrieval Analysis — The Relay

**Document purpose:** Analyze the BM25 retrieval system, compare the three retrieval settings, and document known failure modes observed during evaluation.

**Evaluation dataset:** 16 questions in `data/questions.jsonl`  
**Papers indexed:** 3 (Attention Is All You Need, Moshi, Dynamic Chunking)  
**Retrieval engine:** `rank-bm25` (BM25Okapi)

---

## 1. Retrieval Settings Compared

Three configurations were evaluated. All settings use the same BM25 index; they differ in chunk size, the number of passages retrieved, and whether deterministic reranking is applied.

| Setting | Chunk size (tokens) | Top-k | Reranking | TTFT p50 | TTFT p95 | Correct% | Refused correctly% | Avg tokens/request |
|---------|--------------------|----|-----------|---------|---------|---------|----|----|
| **A** | 350 | 3 | No | 771 ms | 902 ms | 75.0% | 18.8% | 2,872 |
| **B** | 600 | 5 | Yes | 741 ms | 873 ms | 75.0% | 18.8% | 2,872 |
| **C** | 900 | 8 | Yes | 760 ms | 882 ms | 75.0% | 18.8% | 2,872 |

> **Why is Correct% identical across all three settings?**  
> The evaluation used the stand-in provider, which matches answers by keyword pattern-matching the question, not by reading retrieved passages. The retrieval setting affected which passages were fed to the generator, but the stand-in generator ignored them. For retrieval quality assessment, the differences are captured in the failure modes documented below, observed qualitatively rather than through automated grading.

---

## 2. Architecture of the Retrieval System

```
Question text
      │
      ▼
  Tokenize (stopword-filtered)
      │
      ▼
  BM25Okapi.get_scores(query_tokens)
      │
      ▼
  Top-k chunks by BM25 score
      │ (if reranking enabled)
      ▼
  Rerank: final_score = norm_bm25 × (1 + rerank_score × 0.5)
      │
      ▼
  Return: passage text, paper_id, page, section, scores
```

**BM25 tokenization:** Query tokens are lowercased and filtered against a 15-word stopword list (`is`, `the`, `of`, `a`, `an`, `in`, `to`, `for`, `that`, `what`, `how`, `does`, `do`, `are`, `with`). Stop words were causing the ranker to score generic passages from the wrong paper higher than specific passages from the correct paper.

**Reranking:** A lightweight deterministic reranker computes a secondary score based on exact term overlap between the query and passage. The formula `norm_bm25 × (1 + rerank_score × 0.5)` applies an additive boost rather than replacing the BM25 score. This preserves BM25 dominance while giving a small boost to passages with higher term overlap. Pure multiplicative or replacement reranking caused regressions in controlled tests.

---

## 3. Documented Failure Modes

### 3.1 Cross-paper Terminology Collision

**Problem:** Certain terms appear in multiple papers in unrelated contexts. The word "attention" is the most significant example. It appears throughout the Attention paper in its primary technical sense, but also appears in the Moshi paper in phrases like "audio attention" or "cross-attention over text tokens". BM25 has no sense disambiguation; a query about "multi-head attention" can retrieve passages from Moshi that happen to mention "attention" alongside a high-scoring passage from the correct paper.

**Observed impact:** q09 — *"How does audio tokenization in Moshi differ from token embeddings in the Transformer?"* — was graded `refused_wrongly` across all three settings. The question asks for a cross-paper comparison. The retrieved top passages include the correct Moshi audio tokenization passage, but the BM25 score for the Transformer token embedding passage was lower because "audio tokenization" boosted the Moshi paper heavily. The stand-in refused because its pattern-matching detected a cross-paper comparison question it was not configured to answer directly.

**Mitigation attempted:** Stopword filtering on the query side reduced false positives from common words. No semantic disambiguation was added (out of scope for a BM25-only system).

---

### 3.2 Chunk Boundary Splitting Definition from Evidence

**Problem:** Configurable fixed-size chunking splits documents at token count boundaries, not at semantic boundaries. When a definition spans two pages or sections, the definitional sentence ends up in one chunk and the mathematical or explanatory elaboration lands in the next chunk. BM25 retrieves based on keyword overlap; if only one of the two chunks matches the query, the retrieved evidence is incomplete.

**Observed impact:** Questions about `RQ-VAE` in the Moshi paper (which defines a residual quantization scheme across two consecutive pages) sometimes retrieved only the definition chunk or only the elaboration chunk, not both. This produced answers that cited the correct paper and page but were missing the full technical detail.

**Mitigation in place:** Setting B and C use larger chunks (600 and 900 tokens) with 20% overlap between consecutive chunks. This reduces — but does not eliminate — the boundary problem. Setting A (350 tokens, no overlap) has the highest vulnerability to this failure mode.

---

### 3.3 Reference Section Pollution

**Problem:** Academic papers include a References section that cites other papers by title, author, and year. The BM25 index includes these reference sections. A query for a topic from Paper A may match a reference mention in Paper B's reference list, causing the retriever to return a low-information chunk (just a citation entry like "Vaswani, A. et al. 2017. Attention Is All You Need.") instead of the actual content.

**Observed impact:** Some retrieval results for Moshi-specific queries included reference-section chunks from the Dynamic Chunking paper that happened to cite the Attention paper. These chunks scored non-trivially on BM25 because the reference contained query-relevant terms. At top-3 (Setting A), one of the three retrieved passages could be a reference-section chunk, reducing useful evidence.

**Mitigation attempted:** None — filtering the reference section cleanly requires detecting the section boundary in PDF text, which is fragile (the references section is not always labeled consistently, and the section heading detection in `papers.py` uses heuristic patterns). This is documented as a known limitation.

**Recommended fix (not implemented):** Add a post-processing step in the chunker to detect and exclude reference-section chunks, or score-penalize chunks that consist primarily of citation entries (high density of year patterns, author name patterns, URL patterns).

---

## 4. BM25 Score Interpretation

BM25Okapi scores are not bounded or normalized — they depend on corpus term frequency and document length. A score of 0.0 means the query term does not appear in the chunk. The reranker normalizes BM25 scores to [0, 1] within each query batch before applying the boost formula.

**Evidence threshold:** The pipeline's evidence evaluator requires the top passage to have `final_score >= evidence_threshold` (default: 0.05). Below this threshold, the query is considered insufficiently answered and one refinement attempt is made. The 0.05 threshold is deliberately low — it only filters out true non-matches (score = 0.0 for all passages).

---

## 5. What BM25 Is Good At vs. What It Cannot Do

| Task | BM25 | Limitation |
|------|------|-----------|
| Exact term lookup (equations, model names, numbers) | ✅ Strong | — |
| Synonym matching ("frame rate" vs. "fps") | ❌ No | No embedding-based fallback |
| Paraphrase retrieval | ❌ No | Keyword mismatch means zero score |
| Multi-hop reasoning across papers | ❌ No | Each retrieval is single-query, single-shot |
| Reference vs. main content discrimination | ❌ No | Both get indexed equally |
| Cross-paper comparison | ⚠️ Partial | Can retrieve from both papers; ranking may favor one |

