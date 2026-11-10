"""Controlled real-service measurement script for Groq API."""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from relay.config import settings
from relay.groq_provider import GroqProvider
from relay.metrics import MetricsRecorder
from relay.schemas import ResultCSVRow

BENCHMARK_PROMPTS = [
    "Explain the self-attention mechanism in the Transformer architecture in 3 sentences.",
    "What are the main components of the Mimi neural audio codec described in Moshi?",
    "Summarize how dynamic chunking reduces quadratic attention complexity in hierarchical sequence modeling.",
    "State the exact mathematical formula for Scaled Dot-Product Attention.",
    "What is the difference between autoregressive audio modeling and full-duplex conversational modeling?",
]


async def run_real_groq_measurements(
    max_requests: int = 5,
    concurrency_cap: int = 2,
    output_path: Optional[Path] = None,
):
    api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("\n" + "=" * 60)
        print("NOTICE: GROQ_API_KEY is not configured!")
        print("To run real measurements against your free Groq account:")
        print("1. Copy .env.example to .env: cp .env.example .env")
        print("2. Insert your key: GROQ_API_KEY=gsk_...")
        print("3. Re-run: the-relay/.venv/bin/python the-relay/scripts/measure_real_groq.py")
        print("=" * 60 + "\n")
        return None

    out_file = output_path or (settings.measurements_dir / "real_service.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    provider = GroqProvider(api_key=api_key, model=settings.groq_model)
    recorder = MetricsRecorder()

    semaphore = asyncio.Semaphore(concurrency_cap)
    measurements: List[Dict] = []

    print(f"\nStarting controlled real Groq measurement ({max_requests} requests, max concurrency {concurrency_cap})...")
    print(f"Target Model: {settings.groq_model}")

    async def measure_one(idx: int, prompt: str):
        async with semaphore:
            req_id = f"groq_real_{idx+1:03d}"
            start_ts = time.time()
            start_iso = datetime.now(timezone.utc).isoformat()

            messages = [
                {"role": "system", "content": "You are a precise technical assistant. Give concise answers."},
                {"role": "user", "content": prompt},
            ]

            # Execute streaming to measure TTFT
            t0 = time.perf_counter()
            first_token_ms = 0.0
            total_ms = 0.0
            accumulated = []
            final_resp = None

            try:
                async for delta, resp in provider.generate_stream(messages):
                    if delta:
                        accumulated.append(delta)
                    if resp:
                        final_resp = resp

                if final_resp:
                    first_token_ms = final_resp.first_token_ms
                    total_ms = final_resp.total_ms
                    tokens_in = final_resp.tokens_in
                    tokens_out = final_resp.tokens_out
                    status_code = final_resp.status_code
                    headers = final_resp.headers
                    is_err = final_resp.is_error
                    err_msg = final_resp.error_message
                else:
                    total_ms = (time.perf_counter() - t0) * 1000.0
                    tokens_in = len(prompt.split()) * 2
                    tokens_out = len("".join(accumulated).split())
                    status_code = 200
                    headers = {}
                    is_err = False
                    err_msg = None

            except Exception as e:
                total_ms = (time.perf_counter() - t0) * 1000.0
                first_token_ms = total_ms
                tokens_in = len(prompt.split()) * 2
                tokens_out = 0
                status_code = 500
                headers = {}
                is_err = True
                err_msg = str(e)

            status_val = "ok" if not is_err else ("rate_limited" if status_code == 429 else "error")

            record = {
                "request_id": req_id,
                "prompt": prompt,
                "start_time": start_iso,
                "first_token_ms": round(first_token_ms, 2),
                "total_ms": round(total_ms, 2),
                "status_code": status_code,
                "status": status_val,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "rate_limit_headers": headers,
                "error_message": err_msg,
                "answer_preview": "".join(accumulated)[:120],
            }
            measurements.append(record)

            # Record in results.csv
            csv_row = ResultCSVRow(
                request_id=req_id,
                scenario="real_baseline",
                source="real",
                start_time=start_iso,
                first_token_ms=round(first_token_ms, 1),
                total_ms=round(total_ms, 1),
                status=status_val,
                retries=0,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                retrieval_setting="setting_b",
                answer_grade="",
            )
            recorder.record_row(csv_row)
            print(f"[{req_id}] status={status_code} TTFT={first_token_ms:.1f}ms total={total_ms:.1f}ms in={tokens_in} out={tokens_out}")

    tasks = [
        measure_one(i, BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)])
        for i in range(max_requests)
    ]
    await asyncio.gather(*tasks)

    # Save to JSON
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(measurements, f, indent=2)

    print(f"\nReal Groq measurements saved to: {out_file}")
    return measurements


def main():
    asyncio.run(run_real_groq_measurements())


if __name__ == "__main__":
    main()

