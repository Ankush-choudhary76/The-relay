"""Quota and Rate Limit tracker for the Stand-in model service."""

import time
from typing import Dict, Optional, Tuple


class StandinQuotaTracker:
    def __init__(
        self,
        rpm_limit: int = 30,
        tpm_limit: int = 12000,
        rpd_limit: int = 1000,
        tpd_limit: int = 100000,
    ):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        self.tpd_limit = tpd_limit

        # Sliding window requests: list of (timestamp, token_count)
        self.minute_window = []

        # Daily usage
        self.daily_requests_used = 0
        self.daily_tokens_used = 0
        self.day_start_ts = time.time()
        self.forced_daily_exhaustion = False

    def reset(self):
        """Resets all usage counters."""
        self.minute_window = []
        self.daily_requests_used = 0
        self.daily_tokens_used = 0
        self.day_start_ts = time.time()
        self.forced_daily_exhaustion = False

    def force_exhaust_daily_budget(self):
        """Immediately sets daily budget to exhausted for Scenario 5."""
        self.forced_daily_exhaustion = True
        self.daily_requests_used = self.rpd_limit
        self.daily_tokens_used = self.tpd_limit

    def check_and_consume(
        self, estimated_tokens: int = 100
    ) -> Tuple[bool, str, Dict[str, str]]:
        """Checks if request is allowed under RPM, TPM, RPD, and TPD.
        Returns: (is_allowed, error_message, headers_dict)
        """
        now = time.time()

        # Prune minute window older than 60s
        self.minute_window = [
            (ts, tok) for (ts, tok) in self.minute_window if now - ts < 60.0
        ]

        current_rpm = len(self.minute_window)
        current_tpm = sum(tok for (_, tok) in self.minute_window)

        # Calculate remaining
        rem_rpm = max(0, self.rpm_limit - current_rpm)
        rem_tpm = max(0, self.tpm_limit - current_tpm)
        rem_rpd = max(0, self.rpd_limit - self.daily_requests_used)
        rem_tpd = max(0, self.tpd_limit - self.daily_tokens_used)

        headers = {
            "x-ratelimit-limit-requests": str(self.rpd_limit),
            "x-ratelimit-limit-tokens": str(self.tpm_limit),
            "x-ratelimit-remaining-requests": str(rem_rpd),
            "x-ratelimit-remaining-tokens": str(rem_tpm),
            "x-ratelimit-reset-requests": "24h00m00s",
            "x-ratelimit-reset-tokens": f"{max(1, int(60 - (now - (self.minute_window[0][0] if self.minute_window else now))))}s",
        }

        # 1. Check daily quota exhaustion
        if self.forced_daily_exhaustion or self.daily_requests_used >= self.rpd_limit:
            headers["retry-after"] = "86400"
            return (
                False,
                f"Daily request quota of {self.rpd_limit} requests exhausted. Quota resets at 00:00 UTC.",
                headers,
            )

        if self.daily_tokens_used >= self.tpd_limit:
            headers["retry-after"] = "86400"
            return (
                False,
                f"Daily token quota of {self.tpd_limit} tokens exhausted. Quota resets at 00:00 UTC.",
                headers,
            )

        # 2. Check per-minute limits
        if current_rpm >= self.rpm_limit:
            wait_time = int(60.0 - (now - self.minute_window[0][0])) + 1
            headers["retry-after"] = str(max(1, wait_time))
            return (
                False,
                f"Rate limit reached for requests per minute (RPM={self.rpm_limit}). Please wait {wait_time}s.",
                headers,
            )

        if current_tpm + estimated_tokens > self.tpm_limit:
            wait_time = int(60.0 - (now - self.minute_window[0][0])) + 1
            headers["retry-after"] = str(max(1, wait_time))
            return (
                False,
                f"Rate limit reached for tokens per minute (TPM={self.tpm_limit}). Please wait {wait_time}s.",
                headers,
            )

        # Allowed: record consumption
        self.minute_window.append((now, estimated_tokens))
        self.daily_requests_used += 1
        self.daily_tokens_used += estimated_tokens

        return (True, "", headers)

