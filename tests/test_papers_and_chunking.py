"""Unit tests for PDF extraction and chunking with metadata preservation."""

import pytest
from pathlib import Path
from relay.chunking import Chunk, chunk_page, chunk_papers, estimate_tokens
from relay.papers import ExtractedPage, ExtractedPaper, extract_all_papers


def test_estimate_tokens():
    text = "The Transformer architecture replaces recurrent neural networks with multi-head self-attention mechanisms."
    tokens = estimate_tokens(text)
    assert tokens > 10
    assert tokens < 30


def test_chunk_page_metadata_preservation():
    paper = ExtractedPaper(
        paper_id="1706.03762v7",
        arxiv_id="1706.03762",
        version="v7",
        title="Attention Is All You Need",
        file_name="1706.03762v7.pdf",
        total_pages=15,
        pages=[],
    )
    page = ExtractedPage(
        page_number=3,
        text="3 Model Architecture\n\nThe Transformer follows this overall architecture.\n\n3.1 Encoder and Decoder Stacks\n\nThe encoder is composed of a stack of N = 6 identical layers.",
        section_headers=["3 Model Architecture", "3.1 Encoder and Decoder Stacks"],
    )

    chunks = chunk_page(page, paper, target_tokens=350, overlap_tokens=50)
    assert len(chunks) >= 1
    c = chunks[0]
    assert c.paper_id == "1706.03762v7"
    assert c.page_number == 3
    assert c.arxiv_id == "1706.03762"
    assert c.version == "v7"
    assert "Encoder" in c.text or "Model Architecture" in c.text


def test_extract_all_papers_loads_exact_versions():
    papers = extract_all_papers()
    assert len(papers) == 3

    paper_map = {p.arxiv_id: p for p in papers}
    assert "1706.03762" in paper_map
    assert paper_map["1706.03762"].version == "v7"
    assert paper_map["1706.03762"].total_pages == 15

    assert "2410.00037" in paper_map
    assert paper_map["2410.00037"].version == "v1"

    assert "2507.07955" in paper_map
    assert paper_map["2507.07955"].version == "v2"

