"""Lightweight intent classification for questions."""

import re
from typing import List, Optional
from relay.schemas import IntentCategory, IntentResult

# Known keywords for the three papers
PAPER_TOPICS = {
    "transformer", "attention", "self-attention", "multi-head", "scaled dot-product",
    "encoder", "decoder", "vaswani", "bleu", "positional encoding", "feed-forward",
    "moshi", "mimi", "helium", "speech-text", "dialogue", "full-duplex", "rq-vae",
    "quantization", "audio codec", "kyutai", "latency", "streaming", "interleaving",
    "dynamic chunking", "hierarchical sequence", "acoustic boundary", "chunk boundary",
    "boundary predictor", "hierarchical modeling",
}

OUT_OF_SCOPE_MARKERS = {
    "weather", "president", "recipe", "baking", "cake", "stock market", "crypto",
    "bitcoin", "sports", "football", "cricket", "nba", "movie", "celebrity",
    "quantum computing", "dna sequencing", "astronomy", "mars rover",
    "capital of", "population", "geography", "country", "continent",
}


class IntentClassifier:
    def classify(self, question: str) -> IntentResult:
        """Classifies incoming question into intent categories while conserving tokens."""
        q_lower = question.lower().strip()

        # 1. Check for unanswerable / completely unrelated topics
        for marker in OUT_OF_SCOPE_MARKERS:
            if re.search(rf"\b{marker}\b", q_lower):
                return IntentResult(
                    category=IntentCategory.UNANSWERABLE,
                    confidence=0.95,
                    sub_queries=[],
                    reasoning=f"Question touches out-of-scope domain '{marker}' not in the 3 research papers.",
                )

        # 2. Check for comparative queries
        if any(term in q_lower for term in ["compare", "difference between", "versus", "vs", "how does", "differ from"]):
            return IntentResult(
                category=IntentCategory.COMPARATIVE,
                confidence=0.88,
                sub_queries=self._extract_comparison_aspects(question),
                reasoning="Comparative query across mechanisms or architectures.",
            )

        # 3. Check for architectural queries
        if any(term in q_lower for term in ["architecture", "layer", "encoder", "decoder", "sub-layer", "parameter", "loss", "residual", "quantizer"]):
            return IntentResult(
                category=IntentCategory.ARCHITECTURAL_DETAILS,
                confidence=0.85,
                sub_queries=[question],
                reasoning="Inquiry regarding internal model structure or architecture.",
            )

        # 4. Check for definitions
        if q_lower.startswith(("what is", "define", "what are", "what does")):
            return IntentResult(
                category=IntentCategory.DEFINITION,
                confidence=0.82,
                sub_queries=[question],
                reasoning="Definition or conceptual query.",
            )

        # 5. Check for multi-part questions
        if " and " in q_lower and ("?" in question[:-1] or ";" in question or "," in question):
            parts = [p.strip() for p in re.split(r"[?,;]|\band\b", question) if len(p.strip()) > 5]
            if len(parts) > 1:
                return IntentResult(
                    category=IntentCategory.MULTIPART,
                    confidence=0.80,
                    sub_queries=parts,
                    reasoning="Multi-part question composed of distinct sub-inquiries.",
                )

        # Default: factual lookup
        return IntentResult(
            category=IntentCategory.FACTUAL_LOOKUP,
            confidence=0.75,
            sub_queries=[question],
            reasoning="Direct factual inquiry grounded in paper text.",
        )

    def _extract_comparison_aspects(self, question: str) -> List[str]:
        parts = re.split(r"\b(?:versus|vs\.?|compared to|difference between|and)\b", question, flags=re.IGNORECASE)
        return [p.strip() for p in parts if len(p.strip()) > 3]

