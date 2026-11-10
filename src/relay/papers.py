"""PDF parsing and text extraction module for The Relay.

Uses PyMuPDF (fitz) to extract text while strictly preserving:
- Paper title, arXiv ID, version
- PDF page numbers (1-indexed)
- Section headings and sub-headings
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import fitz  # PyMuPDF
from pydantic import BaseModel, Field

from relay.config import settings


class ExtractedPage(BaseModel):
    page_number: int  # 1-indexed
    text: str
    section_headers: List[str] = Field(default_factory=list)


class ExtractedPaper(BaseModel):
    paper_id: str
    arxiv_id: str
    version: str
    title: str
    file_name: str
    total_pages: int
    pages: List[ExtractedPage] = Field(default_factory=list)


SECTION_HEADER_REGEX = re.compile(
    r"^(?:(?:[0-9]+|[A-Z])\s+[A-Z][A-Za-z0-9\s\-_,:]{2,50}|(?:Abstract|Introduction|Related Work|Background|Architecture|Experiments|Conclusion|References|Appendix)\b)",
    re.MULTILINE,
)


def extract_paper(pdf_path: Path, paper_metadata: Dict[str, Any]) -> ExtractedPaper:
    """Extracts text and section headings from a PDF file."""
    doc = fitz.open(pdf_path)
    pages: List[ExtractedPage] = []

    current_section = "Introduction"

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        page_text = page.get_text("text")

        # Detect headers on this page
        headers: List[str] = []
        for line in page_text.splitlines():
            clean_line = line.strip()
            if not clean_line:
                continue
            if SECTION_HEADER_REGEX.match(clean_line):
                headers.append(clean_line)
                current_section = clean_line

        pages.append(
            ExtractedPage(
                page_number=page_num,
                text=page_text,
                section_headers=headers,
            )
        )

    doc.close()

    return ExtractedPaper(
        paper_id=paper_metadata["paper_id"],
        arxiv_id=paper_metadata["arxiv_id"],
        version=paper_metadata["version"],
        title=paper_metadata["title"],
        file_name=paper_metadata["file_name"],
        total_pages=len(pages),
        pages=pages,
    )


def extract_all_papers(
    papers_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> List[ExtractedPaper]:
    """Reads metadata.json, extracts all PDFs, and saves to processed_dir."""
    p_dir = papers_dir or settings.papers_dir
    o_dir = output_dir or settings.processed_dir
    o_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = p_dir / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file missing at: {metadata_file}")

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    extracted_papers: List[ExtractedPaper] = []
    for meta in metadata_list:
        pdf_path = p_dir / meta["file_name"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF missing: {pdf_path}")
        paper = extract_paper(pdf_path, meta)
        extracted_papers.append(paper)

    # Save to json
    out_file = o_dir / "papers_extracted.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in extracted_papers], f, indent=2)

    return extracted_papers

