"""Metrics recording, results.csv maintenance, and statistical performance analysis."""

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from relay.config import settings
from relay.schemas import RelayResponse, ResultCSVRow

REQUIRED_CSV_COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
]

DIAGNOSTIC_COLUMNS = [
    "queue_wait_ms",
    "provider_working_ms",
    "retry_wait_ms",
    "peak_queue_len",
]


class MetricsRecorder:
    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = csv_path or settings.results_csv_path
        self._ensure_csv_headers()

    def _ensure_csv_headers(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(REQUIRED_CSV_COLUMNS + DIAGNOSTIC_COLUMNS)

    def record_row(self, row: ResultCSVRow):
        """Appends a single verified metric row into results.csv."""
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    row.request_id,
                    row.scenario,
                    row.source,
                    row.start_time,
                    f"{row.first_token_ms:.1f}",
                    f"{row.total_ms:.1f}",
                    row.status,
                    row.retries,
                    row.tokens_in,
                    row.tokens_out,
                    row.retrieval_setting,
                    row.answer_grade or "",
                    f"{row.queue_wait_ms:.1f}" if row.queue_wait_ms is not None else "",
                    f"{row.provider_working_ms:.1f}" if row.provider_working_ms is not None else "",
                    f"{row.retry_wait_ms:.1f}" if row.retry_wait_ms is not None else "",
                    row.peak_queue_len if row.peak_queue_len is not None else "",
                ]
            )

    @classmethod
    def compute_summary_table(cls, csv_path: Optional[Path] = None) -> pd.DataFrame:
        """Reads results.csv and computes percentile latency and outcomes per scenario."""
        target_path = csv_path or settings.results_csv_path
        if not target_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(target_path)
        if df.empty:
            return pd.DataFrame()

        summary_rows = []
        for (scenario, source), group in df.groupby(["scenario", "source"]):
            total_reqs = len(group)
            ok_reqs = len(group[group["status"] == "ok"])
            rl_reqs = len(group[group["status"] == "rate_limited"])
            timeout_reqs = len(group[group["status"] == "timeout"])
            error_reqs = len(group[group["status"] == "error"])

            succeeded_pct = (ok_reqs / total_reqs) * 100.0 if total_reqs > 0 else 0.0
            rl_pct = (rl_reqs / total_reqs) * 100.0 if total_reqs > 0 else 0.0
            timeout_pct = ((timeout_reqs + error_reqs) / total_reqs) * 100.0 if total_reqs > 0 else 0.0

            # Latency percentiles on successful requests (or all where measured)
            valid_ttft = group[group["first_token_ms"] > 0]["first_token_ms"]
            valid_total = group[group["total_ms"] > 0]["total_ms"]

            ttft_p50 = np.percentile(valid_ttft, 50) if not valid_ttft.empty else 0.0
            ttft_p95 = np.percentile(valid_ttft, 95) if not valid_ttft.empty else 0.0
            ttft_p99 = np.percentile(valid_ttft, 99) if not valid_ttft.empty else 0.0

            total_p50 = np.percentile(valid_total, 50) if not valid_total.empty else 0.0
            total_p95 = np.percentile(valid_total, 95) if not valid_total.empty else 0.0
            total_p99 = np.percentile(valid_total, 99) if not valid_total.empty else 0.0

            avg_retries = group["retries"].mean() if not group.empty else 0.0
            avg_tokens = (group["tokens_in"] + group["tokens_out"]).mean() if not group.empty else 0.0

            peak_q = group["peak_queue_len"].max() if "peak_queue_len" in group and not group["peak_queue_len"].isna().all() else 0

            # Cost estimation: list price for Llama 3.3 70B (Input: $0.59 / 1M, Output: $0.79 / 1M)
            avg_in = group["tokens_in"].mean() if not group.empty else 0.0
            avg_out = group["tokens_out"].mean() if not group.empty else 0.0
            cost_per_1000 = ((avg_in * 0.59 / 1_000_000) + (avg_out * 0.79 / 1_000_000)) * 1000.0

            summary_rows.append(
                {
                    "scenario": scenario,
                    "source": source,
                    "requests_sent": total_reqs,
                    "succeeded_pct": round(succeeded_pct, 1),
                    "rate_limited_pct": round(rl_pct, 1),
                    "timeout_pct": round(timeout_pct, 1),
                    "retries_per_req": round(avg_retries, 2),
                    "peak_queue": int(peak_q) if not pd.isna(peak_q) else 0,
                    "tokens_per_ans": round(avg_tokens, 1),
                    "cost_per_1000": f"${cost_per_1000:.4f}",
                    "ttft_p50": round(ttft_p50, 1),
                    "ttft_p95": round(ttft_p95, 1),
                    "ttft_p99": round(ttft_p99, 1),
                    "total_p50": round(total_p50, 1),
                    "total_p95": round(total_p95, 1),
                    "total_p99": round(total_p99, 1),
                }
            )

        return pd.DataFrame(summary_rows)

