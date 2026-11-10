"""Resilience module: bounded retries with jitter, deadline enforcement, and circuit breaker."""

import asyncio
import random
import time
from typing import Callable, Optional, Tuple
from relay.config import settings
from relay.schemas import ProviderResponse


class CircuitBreakerOpenException(Exception):
    pass


class RequestDeadlineExceededException(Exception):
    pass


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 4,
        recovery_time_s: float = 15.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_time_s = recovery_time_s

        self.consecutive_failures = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = 0.0

    def record_success(self):
        self.consecutive_failures = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        if self.consecutive_failures >= self.failure_threshold:
            self.state = "OPEN"

    def can_attempt(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.recovery_time_s:
                self.state = "HALF_OPEN"
                return True
            return False
        if self.state == "HALF_OPEN":
            return True
        return True


class RetryManager:
    def __init__(
        self,
        max_retries: Optional[int] = None,
        base_delay_s: Optional[float] = None,
        max_delay_s: Optional[float] = None,
    ):
        self.max_retries = max_retries or settings.max_retries
        self.base_delay_s = base_delay_s or settings.retry_base_delay
        self.max_delay_s = max_delay_s or settings.retry_max_delay
        self.circuit_breaker = CircuitBreaker()

    def is_retryable(self, response: ProviderResponse) -> Tuple[bool, str]:
        """Classifies if error is transient and retryable."""
        if not response.is_error:
            return (False, "Success")

        status = response.status_code

        # Non-retryable client errors
        if status in {400, 401, 403, 404, 422}:
            return (False, f"Non-retryable client error HTTP {status}")

        # Daily budget exhaustion is non-retryable
        msg = (response.error_message or "").lower()
        if "daily" in msg or "budget" in msg or "quota" in msg:
            return (False, "Permanent daily quota exhaustion")

        # Retry-After check for 429
        if status == 429:
            retry_after_str = response.headers.get("retry-after")
            if retry_after_str:
                try:
                    retry_sec = float(retry_after_str)
                    if retry_sec > 30.0:
                        return (False, f"Retry-After {retry_sec}s exceeds reasonable retry window")
                except ValueError:
                    pass
            return (True, "Transient rate limit (HTTP 429)")

        # Transient server errors
        if status in {500, 502, 503, 504}:
            return (True, f"Transient server error HTTP {status}")

        return (False, f"Unhandled HTTP {status}")

    def compute_backoff(self, attempt: int, response: Optional[ProviderResponse] = None) -> float:
        """Calculates backoff delay: honors Retry-After or uses exponential backoff with full jitter."""
        if response and response.headers.get("retry-after"):
            try:
                retry_after_val = float(response.headers["retry-after"])
                # Add small jitter to avoid thundering herd on simultaneous reset
                return retry_after_val + random.uniform(0.1, 0.5)
            except ValueError:
                pass

        # Exponential backoff: base * 2^attempt
        exp_delay = min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
        # Full jitter: Uniform(0.5, 1.5) * exp_delay
        return max(0.1, exp_delay * random.uniform(0.5, 1.5))

