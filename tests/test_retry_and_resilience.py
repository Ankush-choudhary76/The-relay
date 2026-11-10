"""Unit tests for Retry Manager, Jitter Backoff, and Circuit Breaker."""

import pytest
from relay.retry import CircuitBreaker, RetryManager
from relay.schemas import ProviderResponse


def test_retry_classification():
    rm = RetryManager(max_retries=3)

    # 500/503 is retryable
    resp_503 = ProviderResponse(is_error=True, status_code=503, error_message="Service unavailable")
    assert rm.is_retryable(resp_503)[0] is True

    # 429 transient is retryable
    resp_429 = ProviderResponse(is_error=True, status_code=429, headers={"retry-after": "3"})
    assert rm.is_retryable(resp_429)[0] is True

    # 400 Bad Request is NOT retryable
    resp_400 = ProviderResponse(is_error=True, status_code=400, error_message="Invalid payload")
    assert rm.is_retryable(resp_400)[0] is False

    # Permanent daily quota is NOT retryable
    resp_daily = ProviderResponse(is_error=True, status_code=429, error_message="Daily quota exhausted")
    assert rm.is_retryable(resp_daily)[0] is False


def test_retry_backoff_honors_retry_after():
    rm = RetryManager()
    resp = ProviderResponse(headers={"retry-after": "5"})
    backoff = rm.compute_backoff(1, resp)
    assert backoff >= 5.0
    assert backoff <= 5.6


def test_circuit_breaker_trips():
    cb = CircuitBreaker(failure_threshold=3, recovery_time_s=10.0)
    assert cb.can_attempt() is True

    cb.record_failure()
    cb.record_failure()
    assert cb.state == "CLOSED"
    assert cb.can_attempt() is True

    cb.record_failure()
    assert cb.state == "OPEN"
    assert cb.can_attempt() is False

    # Success resets breaker
    cb.record_success()
    assert cb.state == "CLOSED"
    assert cb.can_attempt() is True

