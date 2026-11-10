"""Unit tests for Citation Verification."""

import pytest
from relay.citation_verifier import CitationVerifier
from relay.schemas import Citation, RetrievedPassage


def test_extract_citations_from_text():
    verifier = CitationVerifier()
    text = (
        "The encoder has 6 layers [Citation: Attention Is All You Need, arXiv:1706.03762v7, Page 3, Section 3.1] "
        "and Mimi runs at 12.5 Hz [Citation: Moshi: a speech-text foundation model, 2410.00037v1, Page 2]."
    )
    citations = verifier.extract_citations_from_text(text)
    assert len(citations) == 2
    assert citations[0].arxiv_id == "1706.03762"
    assert citations[0].page_number == 3
    assert citations[1].arxiv_id == "2410.00037"
    assert citations[1].page_number == 2


def test_citation_verification_valid():
    verifier = CitationVerifier()
    passages = [
        RetrievedPassage(
            passage_id="p1",
            paper_id="1706.03762v7",
            paper_title="Attention Is All You Need",
            arxiv_id="1706.03762",
            version="v7",
            page_number=3,
            text="The encoder consists of N = 6 identical layers.",
            final_score=0.9,
        )
    ]
    citations = [
        Citation(
            paper_title="Attention Is All You Need",
            arxiv_id="1706.03762",
            version="v7",
            page_number=3,
            supporting_passage="The encoder consists of N = 6 identical layers.",
        )
    ]
    res = verifier.verify("The encoder has 6 layers", citations, passages)
    assert res.is_valid is True
    assert len(res.verified_citations) == 1
    assert len(res.unsupported_claims) == 0


def test_citation_verification_hallucinated_paper():
    verifier = CitationVerifier()
    passages = []
    citations = [
        Citation(
            paper_title="Deep Residual Learning",
            arxiv_id="1512.03385",  # Not in the 3 papers
            version="v1",
            page_number=1,
            supporting_passage="ResNet",
        )
    ]
    res = verifier.verify("ResNet introduces residual skips.", citations, passages)
    assert res.is_valid is False
    assert any("Invalid paper ID" in u for u in res.unsupported_claims)


def test_citation_verification_out_of_bounds_page():
    verifier = CitationVerifier()
    passages = []
    citations = [
        Citation(
            paper_title="Attention Is All You Need",
            arxiv_id="1706.03762",
            version="v7",
            page_number=99,  # Paper 1 has only 15 pages
            supporting_passage="",
        )
    ]
    res = verifier.verify("Test claim", citations, passages)
    assert res.is_valid is False
    assert any("out of range" in u for u in res.unsupported_claims)

