"""Unit tests for metrics and results.csv schema conformity."""

import csv
import tempfile
from pathlib import Path
import pytest
from relay.metrics import MetricsRecorder, REQUIRED_CSV_COLUMNS
from relay.schemas import ResultCSVRow


def test_results_csv_schema_conformity():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "results.csv"
        recorder = MetricsRecorder(csv_path=csv_path)

        # Write test row
        row = ResultCSVRow(
            request_id="req_test_01",
            scenario="steady",
            source="stand-in",
            start_time="2026-11-10T12:00:00Z",
            first_token_ms=185.4,
            total_ms=450.2,
            status="ok",
            retries=0,
            tokens_in=240,
            tokens_out=45,
            retrieval_setting="setting_b",
            answer_grade="correct",
            queue_wait_ms=1.2,
            provider_working_ms=440.0,
            retry_wait_ms=0.0,
            peak_queue_len=1,
        )
        recorder.record_row(row)

        # Verify CSV contents and column order
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            # Verify first 12 columns match REQUIRED_CSV_COLUMNS exactly
            assert header[:12] == REQUIRED_CSV_COLUMNS

            data_row = next(reader)
            assert data_row[0] == "req_test_01"
            assert data_row[1] == "steady"
            assert data_row[2] == "stand-in"
            assert data_row[3] == "2026-11-10T12:00:00Z"
            assert data_row[4] == "185.4"
            assert data_row[5] == "450.2"
            assert data_row[6] == "ok"
            assert data_row[7] == "0"
            assert data_row[8] == "240"
            assert data_row[9] == "45"
            assert data_row[10] == "setting_b"
            assert data_row[11] == "correct"

