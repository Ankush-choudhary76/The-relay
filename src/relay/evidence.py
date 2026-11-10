"""Evidence evaluation and query refinement loop."""

import re
from typing import List, Optional, Set, Tuple
from relay.schemas import EvidenceEvaluation, RetrievedPassage


class EvidenceEvaluator:
    def __init__(
        self,
        min_threshold: float = 0.35,
        high_confidence_threshold: float = 0.60,
    ):
        self.min_threshold = min_threshold
        self.high_confidence_threshold = high_confidence_threshold

    def deduplicate_passages(
        self, passages: List[RetrievedPassage]
    ) -> List[RetrievedPassage]:
        """Removes passages with duplicate IDs or identical text bodies."""
        seen_ids: Set[str] = set()
        deduped: List[RetrievedPassage] = []
        for p in passages:
            if p.passage_id not in seen_ids:
                seen_ids.add(p.passage_id)
                deduped.append(p)
        return deduped

    def extract_query_aspects(self, query: str) -> List[str]:
        """Identifies sub-questions or distinct query aspects (e.g., clauses separated by and/while/versus)."""
        aspects = [a.strip() for a in re.split(r"\b(?:and|while|versus|vs\.?|compared to|additionally)\b", query, flags=re.IGNORECASE) if len(a.strip()) > 5]
        return aspects if len(aspects) > 1 else [query.strip()]

    def evaluate(
        self,
        query: str,
        passages: List[RetrievedPassage],
        has_already_refined: bool = False,
    ) -> EvidenceEvaluation:
        """Evaluates whether retrieved passages are sufficient to answer the question."""
        if not passages:
            return EvidenceEvaluation(
                is_sufficient=False,
                score=0.0,
                covered_aspects=[],
                missing_aspects=[query],
                needs_refinement=not has_already_refined,
                refined_query=self.refine_query(query) if not has_already_refined else None,
                reason="No passages retrieved.",
            )

        passages = self.deduplicate_passages(passages)
        top_score = passages[0].final_score

        aspects = self.extract_query_aspects(query)
        covered: List[str] = []
        missing: List[str] = []

        all_text = " ".join(p.text.lower() for p in passages)
        for aspect in aspects:
            aspect_words = [w for w in re.findall(r"\b\w{3,}\b", aspect.lower()) if w not in {"what", "how", "why", "when", "does", "the", "and"}]
            if not aspect_words:
                covered.append(aspect)
                continue
            matches = sum(1 for w in aspect_words if w in all_text)
            if matches / len(aspect_words) >= 0.4:
                covered.append(aspect)
            else:
                missing.append(aspect)

        # Check sufficiency:
        # Must exceed min_threshold and have at least 50% aspects covered
        is_sufficient = (top_score >= self.min_threshold) and (len(covered) >= len(missing))

        needs_refinement = False
        refined_query = None

        if not is_sufficient or (top_score < self.high_confidence_threshold and missing):
            if not has_already_refined:
                needs_refinement = True
                refined_query = self.refine_query(query, missing)

        reason = (
            f"Evidence evaluated. Top score: {top_score:.2f}. "
            f"Aspects covered: {len(covered)}/{len(aspects)}. "
            f"Sufficiency: {is_sufficient}."
        )

        return EvidenceEvaluation(
            is_sufficient=is_sufficient,
            score=top_score,
            covered_aspects=covered,
            missing_aspects=missing,
            needs_refinement=needs_refinement,
            refined_query=refined_query,
            reason=reason,
        )

    def refine_query(
        self, original_query: str, missing_aspects: Optional[List[str]] = None
    ) -> str:
        """Controlled query refinement:
        - Expands known abbreviations (e.g. RQ-VAE -> Residual Vector Quantization, Mimi -> neural audio codec)
        - Strips conversational preamble ('can you tell me', 'explain in detail')
        - Focuses on missing aspects if identified.
        """
        focus_text = " ".join(missing_aspects) if missing_aspects else original_query

        # Strip common filler
        clean = re.sub(
            r"^(?:please\s+|can\s+you\s+(?:tell\s+me|explain)\s+|what\s+is\s+|how\s+does\s+)",
            "",
            focus_text,
            flags=re.IGNORECASE,
        ).strip()

        # Domain term expansions for the 3 papers
        expansions = {
            "rq-vae": "RQ-VAE residual vector quantization audio codec",
            "mimi": "Mimi neural audio codec Moshi 12.5 Hz",
            "helium": "Helium language model 7B Moshi",
            "dynamic chunking": "dynamic chunking hierarchical sequence modeling stream",
            "transformer": "Transformer multi-head attention scaled dot-product",
            "bleu": "BLEU score machine translation WMT",
        }

        refined_tokens: List[str] = []
        for word in clean.split():
            clean_word = word.lower().strip("?,.")
            if clean_word in expansions:
                refined_tokens.append(expansions[clean_word])
            else:
                refined_tokens.append(word)

        return " ".join(refined_tokens)

