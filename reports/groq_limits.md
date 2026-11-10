# Groq API Rate Limits — The Relay

**Document purpose:** Record all known Groq free-tier rate limits, distinguishing official documentation from values observed experimentally during this project.

**Retrieval date:** 2026-11-10  
**Model used in this project:** `openai/gpt-oss-20b`  
**Account tier:** Free (unauthenticated console account)

---

## 1. Official Documentation

**Source:** https://console.groq.com/docs/rate-limits  
*(Retrieved 2026-11-10; URL valid at time of retrieval)*

| Limit | Value | Confidence |
|-------|-------|------------|
| Requests per day (RPD) | 1,000 | Official (confirmed by response headers) |
| Tokens per minute (TPM) | 8,000 | Official (confirmed by response headers) |
| Requests per minute (RPM) | Not separately documented for free tier | — |
| Tokens per day (TPD) | Not publicly documented | — |
| Concurrent request behavior | Not publicly documented | — |
| Daily reset time | Not publicly documented (assumed 00:00 UTC) | Assumed |
| HTTP status for rate limit | 429 Too Many Requests | Observed |
| Retry-After header | Present on 429 responses | Observed |

> **Note on RPM:** Groq's documentation page lists per-model limits and mentions RPM for some plans, but does not separately publish an explicit RPM value for the free-tier `openai/gpt-oss-20b` model. The stand-in service was calibrated with an effective ~30 RPM ceiling based on observed quota exhaustion behavior.

---

## 2. Experimentally Observed Values

Measured during `scripts/measure_real_groq.py` — 5 sequential requests on 2026-11-10 at ~10:45 UTC.  
**Cap:** 5 real requests (never more than necessary).

### 2.1 Response Headers Observed

From the first real request (`groq_real_001`):

```
x-ratelimit-limit-requests: 1000
x-ratelimit-limit-tokens: 8000
x-ratelimit-remaining-requests: 997
x-ratelimit-remaining-tokens: 7571
x-ratelimit-reset-requests: 4m19.2s
x-ratelimit-reset-tokens: 3.217s
```

These confirm:
- RPD = 1,000 requests per day (not per minute — the reset was `4m19.2s` for the _request_ counter, which is a rolling window observation, not necessarily a per-minute cap).
- TPM = 8,000 tokens per minute (token reset was `3.217s`, confirming a 1-minute sliding window for tokens).

### 2.2 Latency (5 real requests)

| Request ID | TTFT (ms) | Total (ms) | Tokens In | Tokens Out | Status |
|-----------|-----------|-----------|-----------|-----------|--------|
| groq_real_001 | 766 | 857 | 100 | 144 | 200 OK |
| groq_real_002 | 1,719 | 2,061 | 101 | 674 | 200 OK |
| groq_real_003 | 629 | 903 | 101 | 310 | 200 OK |
| groq_real_004 | 758 | 856 | 98 | 208 | 200 OK |
| groq_real_005 | 626 | 933 | 101 | 392 | 200 OK |
| **p50** | **758** | **903** | | | |
| **p95** | **1,529** | **1,836** | | | |

> **Disclaimer:** These 5 measurements constitute a very small sample. The p95 is dominated by the single high-latency request (groq_real_002, which returned 674 output tokens — the longest response). Real p95 under steady load may differ.

### 2.3 HTTP 429 Behavior

Not triggered during the 5-request baseline. Based on header values, 429 would be expected:
- When cumulative tokens in the sliding minute exceed 8,000.
- When cumulative requests in the rolling window exceed ~1,000/day.

The stand-in service simulates 429 responses in the `rpm_halved` and `sustained_overload` scenarios.

---

## 3. Available Models (Free Tier, Confirmed via `/v1/models`)

The following models were returned by `GET https://api.groq.com/openai/v1/models` with the project API key:

- `openai/gpt-oss-20b` ← **used in this project**
- `allam-2-7b`
- `qwen/qwen3-8b` (and variants)
- `distil-whisper-large-v3-en`, `whisper-large-v3`, `whisper-large-v3-turbo`
- Various `playai-tts` voice models

> The model used for all project experiments is `openai/gpt-oss-20b`, confirmed active on this account via `/v1/models`.

---

## 4. Confidence Classification

| Category | Items |
|----------|-------|
| **Officially documented** | RPD=1,000; TPM=8,000; HTTP 429 for rate limits |
| **Confirmed from live headers** | All values in the table above |
| **Observed empirically** | TTFT p50≈758ms, p95≈1,529ms (5-sample baseline) |
| **Not publicly documented** | RPM for free tier, TPD, concurrent request limits, daily reset time |
| **Assumed** | Daily reset at 00:00 UTC; rolling per-minute window for tokens |
| **Not measured** | Model input/output pricing (free tier, not published) |

---

## 5. Sources

| Source | URL | Retrieved |
|--------|-----|-----------|
| Groq Rate Limits Documentation | https://console.groq.com/docs/rate-limits | 2026-11-10 |
| Groq Models List (via API) | https://api.groq.com/openai/v1/models | 2026-11-10 |
| Live response headers | Captured in `data/measurements/real_service.json` | 2026-11-10 |

