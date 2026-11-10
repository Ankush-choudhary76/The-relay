"""
Download and verify the three exact arXiv research papers required for The Relay.

Papers:
1. "Attention Is All You Need", arXiv:1706.03762, v7
2. "Moshi: a speech-text foundation model for real-time dialogue", arXiv:2410.00037, v1
3. "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", arXiv:2507.07955, v2
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

PAPERS_TO_DOWNLOAD = [
    {
        "paper_id": "1706.03762v7",
        "arxiv_id": "1706.03762",
        "version": "v7",
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762v7",
        "filename": "1706.03762v7.pdf",
    },
    {
        "paper_id": "2410.00037v1",
        "arxiv_id": "2410.00037",
        "version": "v1",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "url": "https://arxiv.org/pdf/2410.00037v1",
        "filename": "2410.00037v1.pdf",
    },
    {
        "paper_id": "2507.07955v2",
        "arxiv_id": "2507.07955",
        "version": "v2",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "url": "https://arxiv.org/pdf/2507.07955v2",
        "filename": "2507.07955v2.pdf",
    },
]


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def download_paper(paper_info: dict, dest_dir: Path) -> dict:
    dest_path = dest_dir / paper_info["filename"]
    print(f"\nFetching {paper_info['title']} ({paper_info['paper_id']})...")
    print(f"URL: {paper_info['url']}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
            "TheRelayAssignment/1.0"
        )
    }

    req = urllib.request.Request(paper_info["url"], headers=headers)
    max_retries = 3
    delay = 2.0

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Unexpected status code: {resp.status}")
                data = resp.read()
                if not data.startswith(b"%PDF-"):
                    raise ValueError("Downloaded file does not start with %PDF- header!")
                
                with open(dest_path, "wb") as f:
                    f.write(data)
                break
        except Exception as e:
            print(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay *= 2

    sha256 = compute_sha256(dest_path)
    file_size = dest_path.stat().st_size
    print(f"Downloaded successfully: {dest_path.name} ({file_size / 1024:.1f} KB, SHA256: {sha256[:12]}...)")

    return {
        "paper_id": paper_info["paper_id"],
        "arxiv_id": paper_info["arxiv_id"],
        "version": paper_info["version"],
        "title": paper_info["title"],
        "download_url": paper_info["url"],
        "download_date": datetime.now(timezone.utc).isoformat(),
        "file_name": paper_info["filename"],
        "file_size_bytes": file_size,
        "sha256": sha256,
        "file_path": str(dest_path.resolve()),
    }


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_papers_dir = project_root / "data" / "papers"
    data_papers_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = data_papers_dir / "metadata.json"
    results = []

    for paper in PAPERS_TO_DOWNLOAD:
        record = download_paper(paper, data_papers_dir)
        results.append(record)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll 3 papers downloaded and metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()

