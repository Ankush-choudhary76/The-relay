"""Client-side sliding window rate limiter for RPM and TPM management."""

import time
from typing import Optional, Tuple
from relay.config import settings


class ClientRateLimiter:
    def __init__(
        self,
        rpm_limit: Optional[int] = None,
        tpm_limit: Optional[int] = None,
        rpd_limit: Optional[int] = None,
        tpd_limit: Optional[int] = None,
    ):
        self.rpm_limit = rpm_limit or settings.client_rpm_limit
        self.tpm_limit = tpm_limit or settings.client_tpm_limit
        self.rpd_limit = rpd_limit or settings.client_rpd_limit
        self.tpd_limit = tpd_limit or settings.client_tpd_limit

        self.minute_window = []  # list of (timestamp, token_count)
        self.daily_requests_used = 0
        self.daily_tokens_used = 0

    def can_proceed(self, estimated_tokens: int = 200) -> Tuple[bool, Optional[float], str]:
        """Checks if request can be scheduled without violating local rate limits.
        Returns: (can_proceed, wait_seconds_needed, reason)
        """
        now = time.time()
        self.minute_window = [
            (ts, tok) for (ts, tok) in self.minute_window if now - ts < 60.0
        ]

        # 1. Daily limit check
        if self.daily_requests_used >= self.rpd_limit:
            return (False, 86400.0, "Daily request budget exhausted.")
        if self.daily_tokens_used >= self.tpd_limit:
            return (False, 86400.0, "Daily token budget exhausted.")

        # 2. Per-minute limit check
        current_rpm = len(self.minute_window)
        current_tpm = sum(tok for (_, tok) in self.minute_window)

        if current_rpm >= self.rpm_limit:
            oldest_ts = self.minute_window[0][0]
            wait_time = max(0.1, 60.0 - (now - oldest_ts))
            return (False, wait_time, f"Local RPM limit ({self.rpm_limit}) reached.")

        if current_tpm + estimated_tokens > self.tpm_limit:
            oldest_ts = self.minute_window[0][0]
            wait_time = max(0.1, 60.0 - (now - oldest_ts))
            return (False, wait_time, f"Local TPM limit ({self.tpm_limit}) reached.")

        return (True, None, "Allowed")

    def record_usage(self, tokens: int = 100):
        now = time.time()
        self.minute_window.append((now, tokens))
        self.daily_requests_used += 1
        self.daily_tokens_used += tokens

    def is_daily_exhausted(self) -> bool:
        return (self.daily_requests_used >= self.rpd_limit) or (
            self.daily_tokens_used >= self.tpd_limit
        )

    def is_near_budget_exhaustion(self, safety_margin: float = 0.85) -> bool:
        """Returns True if consumed quota exceeds safety margin (e.g. 85%)."""
        rpm_ratio = len(self.minute_window) / max(1, self.rpm_limit)
        rpd_ratio = self.daily_requests_used / max(1, self.rpd_limit)
        return (rpm_ratio >= safety_margin) or (rpd_ratio >= safety_margin)

