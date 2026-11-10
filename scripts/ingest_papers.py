"""Simple script to ingest all three PDFs and save extracted text to data/processed/."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from relay.papers import extract_all_papers


def main() -> None:
    metadata_path = PROJECT_ROOT / "data" / "papers" / "metadata.json"
    if not metadata_path.exists():
        print("ERROR: data/papers/metadata.json not found. Run scripts/download_papers.py first.")
        sys.exit(1)

    print("Extracting all 3 papers using PyMuPDF...")
    extracted = extract_all_papers()
    print(f"\nExtracted {len(extracted)} papers successfully:")
    for ep in extracted:
        total_chars = sum(len(p.text) for p in ep.pages)
        print(f"  • {ep.paper_id} ({ep.title}): {ep.total_pages} pages, {total_chars:,} chars")

    output_path = PROJECT_ROOT / "data" / "processed" / "papers_extracted.json"
    print(f"\nProcessed JSON saved to: {output_path}")


if __name__ == "__main__":
    main()

