"""Unit tests for Stand-in model service and simulator logic."""

import pytest
from standin.quota import StandinQuotaTracker
from standin.scenarios import StandinConfig
from standin.simulator import StandinSimulator


def test_standin_quota_minute_limits():
    tracker = StandinQuotaTracker(rpm_limit=2, tpm_limit=500)

    # 1st and 2nd allowed
    ok1, _, _ = tracker.check_and_consume(100)
    ok2, _, _ = tracker.check_and_consume(100)
    assert ok1 is True
    assert ok2 is True

    # 3rd rejected by RPM
    ok3, msg, headers = tracker.check_and_consume(100)
    assert ok3 is False
    assert "RPM" in msg
    assert "retry-after" in headers


def test_standin_daily_budget_exhaustion():
    tracker = StandinQuotaTracker(rpd_limit=5)
    tracker.force_exhaust_daily_budget()

    ok, msg, headers = tracker.check_and_consume(50)
    assert ok is False
    assert "Daily" in msg
    assert headers["retry-after"] == "86400"


@pytest.mark.asyncio
async def test_standin_load_dependent_slowdown():
    cfg = StandinConfig(base_ttft_ms=100.0, ttft_stddev_ms=0.0, load_concurrency_factor_ms=50.0)
    sim = StandinSimulator(cfg)

    # 0 active requests
    ttft_idle = sim.calculate_ttft()

    # Increase active in-flight requests to 4
    for _ in range(4):
        await sim.increment_active()

    ttft_loaded = sim.calculate_ttft()
    assert ttft_loaded >= ttft_idle + (4 * 50.0) - 5.0

