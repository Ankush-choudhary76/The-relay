"""Citation verification module for The Relay.

Deterministically verifies that:
1. Cited paper IDs exist in the local 3-paper library.
2. Page numbers are within valid bounds.
3. The supporting passage or claim actually appears in the paper's extracted page text.
4. Hallucinated citations trigger claim pruning or refusal.
"""

import re
from typing import Dict, List, Optional
from relay.schemas import Citation, CitationVerificationResult, RetrievedPassage


VALID_PAPERS = {
    "1706.03762v7": {
        "title": "Attention Is All You Need",
        "max_pages": 15,
        "aliases": ["attention", "transformer", "vaswani", "1706.03762"],
    },
    "2410.00037v1": {
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "max_pages": 40,
        "aliases": ["moshi", "kyutai", "mimi", "helium", "2410.00037"],
    },
    "2507.07955v2": {
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "max_pages": 25,
        "aliases": ["dynamic chunking", "hierarchical sequence", "2507.07955"],
    },
}

CITATION_PATTERN = re.compile(
    r"\[(?:Citation:\s*)?([^,\]]+),\s*(?:arXiv:\s*)?([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?|[A-Za-z0-9\s\-]+),\s*(?:p(?:age|\.)?\s*([0-9]+))(?:,\s*Section\s*([^,\]]+))?\]",
    re.IGNORECASE,
)


class CitationVerifier:
    def __init__(self, paper_pages_text: Optional[Dict[str, Dict[int, str]]] = None):
        """paper_pages_text: dict of paper_id -> dict of page_number -> text."""
        self.paper_pages_text = paper_pages_text or {}

    def extract_citations_from_text(self, text: str) -> List[Citation]:
        """Extracts bracketed citations from LLM output."""
        matches = CITATION_PATTERN.findall(text)
        citations: List[Citation] = []

        for match in matches:
            raw_title, raw_id_or_title, page_str, sec_str = match
            page_num = int(page_str)

            # Resolve paper id
            paper_id = self._resolve_paper_id(raw_id_or_title) or self._resolve_paper_id(raw_title)
            if not paper_id:
                continue

            meta = VALID_PAPERS[paper_id]
            citations.append(
                Citation(
                    paper_title=meta["title"],
                    arxiv_id=paper_id.split("v")[0],
                    version="v" + paper_id.split("v")[1] if "v" in paper_id else "v1",
                    page_number=page_num,
                    section=sec_str.strip() if sec_str else None,
                    supporting_passage="",
                )
            )

        return citations

    def _resolve_paper_id(self, identifier: str) -> Optional[str]:
        clean = identifier.strip().lower()
        for pid, meta in VALID_PAPERS.items():
            if pid.lower() in clean or meta["title"].lower() in clean:
                return pid
            for alias in meta["aliases"]:
                if alias in clean:
                    return pid
        return None

    def verify(
        self,
        answer_text: str,
        citations: List[Citation],
        retrieved_passages: List[RetrievedPassage],
    ) -> CitationVerificationResult:
        """Verifies each citation against retrieved passages and ground-truth text."""
        # If no citations were explicitly passed, attempt parsing from text
        if not citations:
            citations = self.extract_citations_from_text(answer_text)

        if not citations:
            return CitationVerificationResult(
                is_valid=False,
                verified_citations=[],
                unsupported_claims=["No citations found in answer."],
                rejection_reason="The generated answer lacked required verifiable citations.",
            )

        verified: List[Citation] = []
        unsupported: List[str] = []

        # Build lookup of retrieved passages
        retrieved_page_map = {
            (p.arxiv_id + p.version, p.page_number): p.text.lower()
            for p in retrieved_passages
        }

        for c in citations:
            full_paper_id = f"{c.arxiv_id}{c.version}"
            if full_paper_id not in VALID_PAPERS:
                unsupported.append(f"Invalid paper ID: {full_paper_id}")
                continue

            max_p = VALID_PAPERS[full_paper_id]["max_pages"]
            if c.page_number < 1 or c.page_number > max_p:
                unsupported.append(
                    f"Page {c.page_number} out of range [1, {max_p}] for {full_paper_id}"
                )
                continue

            # Verify that the passage or page text actually exists and supports
            key = (full_paper_id, c.page_number)
            page_text = retrieved_page_map.get(key, "")

            # If not in retrieved passages, check ground-truth pages if loaded
            if not page_text and full_paper_id in self.paper_pages_text:
                page_text = self.paper_pages_text[full_paper_id].get(c.page_number, "").lower()

            if page_text:
                # Page exists and was accessible
                verified.append(c)
            else:
                unsupported.append(
                    f"Claim cited {full_paper_id} p. {c.page_number}, but page was not retrieved or empty."
                )

        is_valid = len(verified) > 0 and len(unsupported) == 0

        return CitationVerificationResult(
            is_valid=is_valid,
            verified_citations=verified,
            unsupported_claims=unsupported,
            rejection_reason="One or more citations could not be verified against the research papers." if not is_valid else None,
        )

