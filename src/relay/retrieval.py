"""Retrieval engine implementing BM25 lexical search and deterministic reranking."""

import math
import re
from typing import List, Optional, Set, Tuple
from rank_bm25 import BM25Okapi

from relay.chunking import Chunk
from relay.schemas import RetrievedPassage

STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "tell", "please", "can",
}


def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """Alphanumeric tokenization with optional stopword removal."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
    if remove_stopwords:
        filtered = [t for t in tokens if t not in STOP_WORDS]
        return filtered if filtered else tokens
    return tokens


class DocumentRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.corpus_tokens = [tokenize(c.text, remove_stopwords=False) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        enable_reranking: bool = True,
        reduced_budget_mode: bool = False,
    ) -> List[RetrievedPassage]:
        """Retrieves top-k passages matching query using BM25 and deterministic reranking."""
        if not self.chunks or not self.bm25:
            return []

        # If running in reduced budget mode, throttle top-k to conserve prompt tokens
        effective_top_k = min(top_k, 2) if reduced_budget_mode else top_k

        # For search, filter stop words from query to avoid common word noise
        query_search_tokens = tokenize(query, remove_stopwords=True)
        if not query_search_tokens:
            query_search_tokens = tokenize(query, remove_stopwords=False)

        if not query_search_tokens:
            return []

        bm25_scores = self.bm25.get_scores(query_search_tokens)
        max_score = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0

        scored_candidates: List[Tuple[float, Chunk, float]] = []
        for idx, chunk in enumerate(self.chunks):
            raw_bm25 = float(bm25_scores[idx])
            normalized_bm25 = raw_bm25 / max_score if max_score > 0 else 0.0

            if raw_bm25 > 0:
                scored_candidates.append((normalized_bm25, chunk, raw_bm25))

        # Sort by raw BM25 first
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Take candidate pool (up to 3x effective_top_k for reranking)
        pool = scored_candidates[: max(effective_top_k * 3, 10)]

        passages: List[RetrievedPassage] = []
        for norm_bm25, chunk, raw_bm25 in pool:
            rerank_score = 0.0
            if enable_reranking:
                rerank_score = self._compute_rerank_boost(query, query_search_tokens, chunk)

            # Reranking boost is additive bonus on top of normalized BM25
            final_score = (norm_bm25 * (1.0 + rerank_score * 0.5)) if enable_reranking else norm_bm25

            passages.append(
                RetrievedPassage(
                    passage_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    paper_title=chunk.paper_title,
                    arxiv_id=chunk.arxiv_id,
                    version=chunk.version,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    text=chunk.text,
                    bm25_score=round(raw_bm25, 4),
                    rerank_score=round(rerank_score, 4),
                    final_score=round(final_score, 4),
                )
            )

        passages.sort(key=lambda p: p.final_score, reverse=True)
        return passages[:effective_top_k]

    def _compute_rerank_boost(
        self, query: str, query_tokens: List[str], chunk: Chunk
    ) -> float:
        """Deterministic reranking features:
        1. Exact phrase occurrence boost (+0.35)
        2. Query term coverage ratio (+0.35)
        3. Section header keyword relevance (+0.15)
        4. Paper title keyword match (+0.15)
        """
        chunk_text_lower = chunk.text.lower()
        score = 0.0

        # 1. Exact query phrase or n-gram match
        clean_q = query.strip().lower()
        if len(clean_q) > 10 and clean_q in chunk_text_lower:
            score += 0.35
        else:
            # Check for 2-word phrase matches
            bigrams = [f"{query_tokens[i]} {query_tokens[i+1]}" for i in range(len(query_tokens) - 1)]
            if bigrams and any(bg in chunk_text_lower for bg in bigrams):
                score += 0.20

        # 2. Query term coverage (what fraction of query terms appear in chunk)
        chunk_words = set(tokenize(chunk.text, remove_stopwords=False))
        matched_terms = [t for t in query_tokens if t in chunk_words]
        coverage = len(matched_terms) / len(query_tokens) if query_tokens else 0.0
        score += coverage * 0.35

        # 3. Section header boost
        if chunk.section:
            section_lower = chunk.section.lower()
            if any(t in section_lower for t in query_tokens if len(t) > 3):
                score += 0.15

        # 4. Title match boost
        title_lower = chunk.paper_title.lower()
        if any(t in title_lower for t in query_tokens if len(t) > 4):
            score += 0.15

        return min(1.0, score)
