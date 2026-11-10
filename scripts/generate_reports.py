"""
Generate human-readable summary tables from results.csv.

Usage:
    PYTHONPATH=the-relay/src:the-relay the-relay/.venv/bin/python the-relay/scripts/generate_reports.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np


RESULTS_PATH = PROJECT_ROOT / "results.csv"


def fmt_ms(val) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:.0f} ms"


def fmt_pct(val) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:.1f}%"


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    return df


def table_2a_latency(df: pd.DataFrame) -> None:
    """Table 2a — Latency by scenario (stand-in only, excluding retrieval_eval)."""
    print("\n## Table 2a: Latency by Scenario (stand-in provider)")
    print("-" * 70)

    sub = df[(df["source"] == "stand-in") & (df["scenario"] != "retrieval_eval")].copy()
    sub["first_token_ms"] = pd.to_numeric(sub["first_token_ms"], errors="coerce")
    sub["total_ms"] = pd.to_numeric(sub["total_ms"], errors="coerce")

    header = f"{'Scenario':<22} {'N':>4} {'OK%':>6} {'TTFT p50':>10} {'TTFT p95':>10} {'Total p50':>10} {'Total p95':>10}"
    print(header)
    print("-" * 70)

    for scenario in ["steady", "burst", "sustained_overload", "rpm_halved", "daily_exhausted"]:
        grp = sub[sub["scenario"] == scenario]
        if grp.empty:
            continue
        ok_pct = (grp["status"] == "ok").mean() * 100
        ok_rows = grp[grp["first_token_ms"] > 0]
        ttft_p50 = ok_rows["first_token_ms"].quantile(0.50) if not ok_rows.empty else float("nan")
        ttft_p95 = ok_rows["first_token_ms"].quantile(0.95) if not ok_rows.empty else float("nan")
        total_p50 = ok_rows["total_ms"].quantile(0.50) if not ok_rows.empty else float("nan")
        total_p95 = ok_rows["total_ms"].quantile(0.95) if not ok_rows.empty else float("nan")
        print(
            f"{scenario:<22} {len(grp):>4} {fmt_pct(ok_pct):>6} "
            f"{fmt_ms(ttft_p50):>10} {fmt_ms(ttft_p95):>10} "
            f"{fmt_ms(total_p50):>10} {fmt_ms(total_p95):>10}"
        )

    # Real baseline
    real = df[df["source"] == "real"].copy()
    if not real.empty:
        ok_pct = (real["status"] == "ok").mean() * 100
        ttft_p50 = real["first_token_ms"].quantile(0.50)
        ttft_p95 = real["first_token_ms"].quantile(0.95)
        total_p50 = real["total_ms"].quantile(0.50)
        total_p95 = real["total_ms"].quantile(0.95)
        print(
            f"{'real_groq_baseline':<22} {len(real):>4} {fmt_pct(ok_pct):>6} "
            f"{fmt_ms(ttft_p50):>10} {fmt_ms(ttft_p95):>10} "
            f"{fmt_ms(total_p50):>10} {fmt_ms(total_p95):>10}"
        )


def table_2b_outcomes(df: pd.DataFrame) -> None:
    """Table 2b — Outcome distribution by scenario."""
    print("\n## Table 2b: Outcome Distribution by Scenario")
    print("-" * 60)

    sub = df[df["scenario"] != "retrieval_eval"].copy()

    header = f"{'Scenario':<22} {'N':>4} {'ok':>6} {'rate_lim':>10} {'error':>8} {'timeout':>8}"
    print(header)
    print("-" * 60)

    all_scenarios = ["steady", "burst", "sustained_overload", "rpm_halved", "daily_exhausted", "steady_baseline"]
    for scenario in all_scenarios:
        grp = sub[sub["scenario"] == scenario]
        if grp.empty:
            continue
        total = len(grp)
        ok = (grp["status"] == "ok").sum()
        rl = (grp["status"] == "rate_limited").sum()
        err = (grp["status"] == "error").sum()
        to = (grp["status"] == "timeout").sum()
        print(f"{scenario:<22} {total:>4} {ok:>6} {rl:>10} {err:>8} {to:>8}")


def table_3_retrieval(df: pd.DataFrame) -> None:
    """Table 3 — Retrieval setting comparison."""
    print("\n## Table 3: Retrieval Setting Comparison (stand-in, retrieval_eval scenario)")
    print("-" * 90)

    sub = df[(df["scenario"] == "retrieval_eval") & (df["source"] == "stand-in")].copy()
    if sub.empty:
        print("  No retrieval_eval rows found.")
        return

    header = (
        f"{'Setting':<12} {'N':>4} {'Correct%':>10} {'Refused✓%':>10} "
        f"{'Refused✗%':>10} {'TTFT p50':>10} {'TTFT p95':>10} {'Avg tok_in':>12}"
    )
    print(header)
    print("-" * 90)

    for setting in ["setting_a", "setting_b", "setting_c"]:
        grp = sub[sub["retrieval_setting"] == setting]
        if grp.empty:
            continue
        total = len(grp)
        correct_pct = (grp["answer_grade"] == "correct").mean() * 100
        ref_ok_pct = (grp["answer_grade"] == "refused_correctly").mean() * 100
        ref_bad_pct = (grp["answer_grade"] == "refused_wrongly").mean() * 100
        ok_rows = grp[grp["first_token_ms"] > 0]
        ttft_p50 = ok_rows["first_token_ms"].quantile(0.50) if not ok_rows.empty else float("nan")
        ttft_p95 = ok_rows["first_token_ms"].quantile(0.95) if not ok_rows.empty else float("nan")
        avg_tok = grp["tokens_in"].mean()
        print(
            f"{setting:<12} {total:>4} {fmt_pct(correct_pct):>10} {fmt_pct(ref_ok_pct):>10} "
            f"{fmt_pct(ref_bad_pct):>10} {fmt_ms(ttft_p50):>10} {fmt_ms(ttft_p95):>10} {avg_tok:>12.0f}"
        )


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Run evaluation and load tests first.")
        sys.exit(1)

    df = load()
    print(f"Loaded {len(df)} rows from results.csv")
    print(f"Scenarios: {sorted(df['scenario'].unique())}")
    print(f"Sources: {sorted(df['source'].unique())}")

    table_2a_latency(df)
    table_2b_outcomes(df)
    table_3_retrieval(df)

    print("\nDone.")


if __name__ == "__main__":
    main()

