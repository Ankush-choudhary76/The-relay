"""Scenario simulation profiles and parameters for the Stand-in model service."""

from pydantic import BaseModel, Field


class StandinConfig(BaseModel):
    # Latency calibration (empirically calibrated against real Groq openai/gpt-oss-20b measurements)
    base_ttft_ms: float = 680.0  # Observed median TTFT: ~650-750ms
    ttft_stddev_ms: float = 80.0
    inter_token_delay_ms: float = 1.0  # Calibrated to Groq's LPU streaming token rate
    load_concurrency_factor_ms: float = 50.0  # Latency added per active in-flight request

    # Error injection
    error_500_prob: float = 0.0
    error_503_prob: float = 0.0
    slow_request_prob: float = 0.0
    slow_request_delay_ms: float = 4000.0
    timeout_prob: float = 0.0

    # Quota constraints (matching official Groq Free Tier headers: 8000 TPM, 1000 RPD)
    rpm_limit: int = 30
    tpm_limit: int = 8000
    rpd_limit: int = 1000
    tpd_limit: int = 200000

    # Scenarios shortcuts
    scenario_name: str = "baseline"


# Preconfigured scenarios for mandatory experiments
SCENARIO_CONFIGS = {
    "steady": StandinConfig(
        scenario_name="steady",
        base_ttft_ms=680.0,
        load_concurrency_factor_ms=30.0,
        rpm_limit=30,
        tpm_limit=8000,
    ),
    "burst": StandinConfig(
        scenario_name="burst",
        base_ttft_ms=680.0,
        load_concurrency_factor_ms=60.0,
        rpm_limit=30,
        tpm_limit=8000,
    ),
    "sustained_overload": StandinConfig(
        scenario_name="sustained_overload",
        base_ttft_ms=720.0,
        load_concurrency_factor_ms=70.0,
        rpm_limit=30,
        tpm_limit=8000,
    ),
    "rpm_halved": StandinConfig(
        scenario_name="rpm_halved",
        base_ttft_ms=680.0,
        load_concurrency_factor_ms=35.0,
        rpm_limit=15,  # Exactly halved
        tpm_limit=8000,
    ),
    "daily_exhausted": StandinConfig(
        scenario_name="daily_exhausted",
        base_ttft_ms=680.0,
        rpd_limit=10,  # Tiny budget that immediately exhausts
        tpd_limit=2000,
    ),
}

