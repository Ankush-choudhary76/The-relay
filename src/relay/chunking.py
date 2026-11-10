"""Chunking strategies for research paper texts with metadata preservation."""

import re
from typing import List, Optional
from pydantic import BaseModel, Field

from relay.papers import ExtractedPaper, ExtractedPage
from relay.schemas import RetrievedPassage


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str
    arxiv_id: str
    version: str
    page_number: int
    section: Optional[str] = None
    text: str
    approx_token_count: int


def estimate_tokens(text: str) -> int:
    """Estimates tokens: ~0.75 words per token or ~4 chars per token."""
    words = len(text.split())
    chars = len(text)
    return max(1, int((words * 1.3 + chars / 4.0) / 2))


def chunk_page(
    page: ExtractedPage,
    paper: ExtractedPaper,
    target_tokens: int = 600,
    overlap_tokens: int = 90,
) -> List[Chunk]:
    """Splits a single page into overlapping chunks respecting paragraph boundaries."""
    paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
    if not paragraphs:
        # Fallback to single text if no blank lines
        paragraphs = [line.strip() for line in page.text.splitlines() if line.strip()]

    chunks: List[Chunk] = []
    current_tokens = 0
    current_paragraphs: List[str] = []
    chunk_idx = 1
    current_section = page.section_headers[0] if page.section_headers else "General"

    for para in paragraphs:
        # Check if this paragraph itself is a section header
        for h in page.section_headers:
            if h in para:
                current_section = h

        para_tokens = estimate_tokens(para)

        if current_tokens + para_tokens > target_tokens and current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.paper_id}_p{page.page_number}_c{chunk_idx}",
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    arxiv_id=paper.arxiv_id,
                    version=paper.version,
                    page_number=page.page_number,
                    section=current_section,
                    text=chunk_text,
                    approx_token_count=estimate_tokens(chunk_text),
                )
            )
            chunk_idx += 1

            # Keep overlap paragraphs from the tail
            overlap_accum = 0
            overlap_paras: List[str] = []
            for p in reversed(current_paragraphs):
                p_tok = estimate_tokens(p)
                if overlap_accum + p_tok <= overlap_tokens:
                    overlap_paras.insert(0, p)
                    overlap_accum += p_tok
                else:
                    break
            current_paragraphs = overlap_paras
            current_tokens = overlap_accum

        current_paragraphs.append(para)
        current_tokens += para_tokens

    if current_paragraphs:
        chunk_text = "\n\n".join(current_paragraphs)
        chunks.append(
            Chunk(
                chunk_id=f"{paper.paper_id}_p{page.page_number}_c{chunk_idx}",
                paper_id=paper.paper_id,
                paper_title=paper.title,
                arxiv_id=paper.arxiv_id,
                version=paper.version,
                page_number=page.page_number,
                section=current_section,
                text=chunk_text,
                approx_token_count=estimate_tokens(chunk_text),
            )
        )

    return chunks


def chunk_papers(
    papers: List[ExtractedPaper],
    chunk_size_tokens: int = 600,
    overlap_tokens: int = 90,
) -> List[Chunk]:
    """Generates chunks across all extracted research papers."""
    all_chunks: List[Chunk] = []
    for paper in papers:
        for page in paper.pages:
            page_chunks = chunk_page(
                page=page,
                paper=paper,
                target_tokens=chunk_size_tokens,
                overlap_tokens=overlap_tokens,
            )
            all_chunks.extend(page_chunks)
    return all_chunks

