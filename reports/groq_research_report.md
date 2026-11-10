# Technical Research & Verification Report: Groq Hosted API & Rate Limits

**Project:** The Relay — Vaani Research ML Engineering Intern Assignment  
**Investigator:** Senior Backend & ML Systems Engineer  
**Date of Investigation:** 2026-11-10  
**Target Platform:** GroqCloud Hosted API (`https://api.groq.com/openai/v1`)  
**Account Status:** Authenticated (Free Plan inferred from account-observed headers; exact organization plan not independently confirmed via console limits dashboard)  
**Primary Target Model:** `openai/gpt-oss-20b`

---

## 1. Executive Summary

This report delivers a rigorous technical evaluation of Groq's hosted LLM serving infrastructure for "The Relay" RAG system. Groq utilizes custom Language Processing Units (LPUs) optimized for high sequential token generation speeds. Vendor marketing publications cite rated speeds exceeding 500–1,000 tokens/second for specific open-weight architectures; these vendor-reported numbers are distinct from the empirical measurements recorded during our test sessions.

Key verified findings include:
1. **Plan Disambiguation & Rate Limits:** Groq enforces separate rate-limit profiles for its **Free Plan** and **Developer Plan**. For `openai/gpt-oss-20b`, the official documented Free Plan limits are **30 Requests Per Minute (RPM)**, **1,000 Requests Per Day (RPD)**, **8,000 Tokens Per Minute (TPM)**, and **200,000 Tokens Per Day (TPD)**. Account response headers explicitly confirm the 1,000 RPD and 8,000 TPM ceilings (`x-ratelimit-limit-requests: 1000` and `x-ratelimit-limit-tokens: 8000`).
2. **Account Model Audit:** Calling `GET /openai/v1/models` reveals 11 active models on the account spanning reasoning, text generation, audio transcription, and safety classification.
3. **Reasoning Token Dynamics:** `openai/gpt-oss-20b` is an open-weight reasoning model. In addition to user-visible response tokens, it generates internal reasoning tokens (`reasoning_tokens`) reported in `completion_tokens_details`. These reasoning tokens count directly against TPM limits and completion budgets.
4. **Resilience Architecture:** Because an 8,000 TPM ceiling constrains throughput over a 60-second window, client-side token reservation, a bounded concurrency semaphore, and an in-process stand-in simulator are essential for safe load testing without risking account throttling.

---

## 2. Official Groq API Overview

- **Architecture:** Hosted tensor streaming served via Groq LPU hardware engineered for sequential token throughput.
- **Protocol Compatibility:** Exposes an OpenAI-compatible interface at `https://api.groq.com/openai/v1`.
- **Primary Endpoints:**
  - `POST /openai/v1/chat/completions`: Chat completions supporting streaming, JSON mode, tool calling, and reasoning parameters.
  - `GET /openai/v1/models`: Enumeration of active provisioned models on the account.
  - `POST /openai/v1/audio/transcriptions`: Speech-to-text audio processing using Whisper.
  - `POST /openai/v1/audio/translations`: Speech translation using Whisper.
- **Authentication:** Standard HTTP Bearer Token header (`Authorization: Bearer $GROQ_API_KEY`).
- **Transport & Streaming:** HTTP/1.1 and HTTP/2 over TLS; Server-Sent Events (`text/event-stream`) for token streaming.

---

## 3. Rate Limits: Free Plan vs. Developer Plan

Rate limits in GroqCloud apply at the organization level and vary across models and service plans.

### 3.1 Primary Model (`openai/gpt-oss-20b`) Limits

| Dimension | Free Plan (Documented) | Developer Plan (Documented) | Actually Observed in Account Headers | Classification & Evidence |
|---|---|---|---|---|
| **Requests Per Minute (RPM)** | **30** | 1,000 | Not exposed in response headers | Officially Documented (Free Plan) |
| **Requests Per Day (RPD)** | **1,000** | Uncapped / Tiered | **1,000** (`x-ratelimit-limit-requests: 1000`) | Documented & Account-Header Confirmed |
| **Tokens Per Minute (TPM)** | **8,000** | 250,000 | **8,000** (`x-ratelimit-limit-tokens: 8000`) | Documented & Account-Header Confirmed |
| **Tokens Per Day (TPD)** | **200,000** | Uncapped / Tiered | Not returned in response headers | Officially Documented (Free Plan) |
| **Concurrent Request Cap** | Not publicly documented | Not publicly documented | Not measured | Not Documented / Not Measured |
| **Daily Quota Reset Time** | Not publicly documented | Not publicly documented | Exact wall-clock reset time unverified | Not Documented / Not Measured |

> **Classification Definitions:**  
> - **Officially Documented (Free Plan):** Published in official Groq rate limits documentation for `openai/gpt-oss-20b` under the Free Plan view.  
> - **Officially Documented (Developer Plan):** Published under the pay-as-you-go Developer Plan view.  
> - **Account-Observed:** Read directly from live response headers during test requests.  
> - **Not Documented / Not Measured:** Not publicly published or not tested.

---

## 4. Model Comparison (All 11 Account Models)

The 11 models available on this account (retrieved via `GET /openai/v1/models` on 2026-11-10) are organized into functional groups below, distinguishing Free Plan and Developer Plan specifications:

### 4.1 Reasoning Models
- **`openai/gpt-oss-20b` (Primary recommendation for The Relay):**
  - **Context Window:** 131,072 tokens
  - **Max Output Tokens:** 65,536 tokens
  - **Reasoning Effort:** Configurable via `reasoning_effort`: `low`, `medium`, `high`
  - **Free Plan Limits:** RPM: 30, RPD: 1,000, TPM: 8,000, TPD: 200,000
  - **Developer Plan Limits:** RPM: 1,000, TPM: 250,000
  - **Pricing (Developer Plan):** $0.075 / 1M input tokens, $0.30 / 1M output tokens (Vendor-published)
  - **Suitability:** High. Fast sequential token generation, large context window, capable of structured citations.
- **`openai/gpt-oss-120b`:**
  - **Context Window:** 131,072 tokens
  - **Max Output Tokens:** 65,536 tokens
  - **Reasoning Effort:** Supported (`low`, `medium`, `high`)
  - **Free Plan Limits:** RPM: 30, RPD: 1,000, TPM: 8,000, TPD: 200,000
  - **Developer Plan Limits:** RPM: 1,000, TPM: 250,000
  - **Pricing (Developer Plan):** $0.15 / 1M input tokens, $0.60 / 1M output tokens (Vendor-published)
  - **Suitability:** Capable but larger parameter footprint; higher latency and token cost make it unnecessary for a 3-paper RAG task.

### 4.2 Text Generation Models
- **`qwen/qwen3.8-27b`:**
  - **Context Window:** 131,072 tokens (Confirmed in official Groq model catalog)
  - **Max Output Tokens:** 16,384 tokens (Confirmed in official Groq model catalog)
  - **Free Plan Limits:** RPM: 30, RPD: 1,000, TPM: 8,000, TPD: 200,000
  - **Developer Plan Limits:** RPM: 1,000, TPM: 250,000
  - **Suitability:** General instruction-following text generation model.
- **`allam-2-7b`:**
  - **Context Window:** 4,096 tokens (Confirmed on official Groq ALLaM model page)
  - **Max Output Tokens:** 4,096 tokens (Confirmed on official Groq ALLaM model page)
  - **Free Plan Limits:** Not publicly documented / not verified on primary Free Plan rate-limits table.
  - **Suitability:** Specialized Arabic/English bilingual model. Unsuitable for The Relay due to restrictive 4,096 context window.

### 4.3 Audio Models
- **`whisper-large-v3` & `whisper-large-v3-turbo`:**
  - **Type:** Speech-to-text / translation endpoints (`/v1/audio/transcriptions`).
  - **Free Plan Limits:** RPM: 20, RPD: 2,000, **ASH: 7,200** (Audio Seconds per Hour), **ASD: 28,800** (Audio Seconds per Day).
  - **Developer Plan Limits:** RPM: 50–300, ASH: up to 200,000–400,000 Audio Seconds per Hour.
  - **Suitability:** Audio processing only; cannot perform text chat completions.

### 4.4 Safety and Classification Models
- **`openai/gpt-oss-safeguard-20b`:**
  - **Context Window:** 131,072 tokens | **Max Output:** 65,536 tokens
  - **Free Plan Limits:** RPM: 3, RPD: 1,000, TPM: 2,000, TPD: 200,000 (Officially documented in Free Plan table)
  - **Developer Plan Limits:** RPM: 1,000, TPM: 250,000
  - **Suitability:** Safety policy inspection; not recommended as the primary QA generation engine.
- **`meta-llama/llama-prompt-guard-2-22m` & `meta-llama/llama-prompt-guard-2-86m`:**
  - **Type:** 22M and 86M parameter sequence classifiers designed to detect prompt injection and jailbreak attempts.
  - **Context Window:** 512 tokens (Confirmed on official model page)
  - **Max Output Tokens:** 512 tokens (Confirmed on official model page)
  - **Free Plan Limits:** RPM: 30, RPD: 14,400, TPM: 15,000, TPD: 500,000
  - **Suitability:** Security classifiers only; not generative chat models.

### 4.5 Other Specialized Models
- **`canopylabs/orpheus-v1-english` & `canopylabs/orpheus-arabic-saudi`:**
  - **Type:** Text-to-speech / audio synthesis models.
  - **Context Window:** 4,000 tokens (Confirmed in Groq model catalog)
  - **Max Output Tokens:** 50,000 tokens (Confirmed in Groq model catalog)
  - **Free Plan Limits:** RPM: 10, RPD: 100, TPM: 1,200, TPD: 3,600
  - **Developer Plan Limits:** Unspecified / Tiered

---

## 5. Rate-Limit Behavior

Groq enforces rate limits using standard HTTP status codes and telemetry headers:

| Limit Dimension | HTTP Status | Generic Behavior (Documented) | Relevant Response Headers | Actual Observation Status |
|---|---|---|---|---|
| **Tokens Per Minute (TPM)** | `429 Too Many Requests` | Rejects request; returns wait duration | `retry-after`, `x-ratelimit-reset-tokens` | Generic behavior documented; not deliberately triggered on live account |
| **Requests Per Minute (RPM)** | `429 Too Many Requests` | Throttles rapid bursts | `retry-after` | Generic behavior documented; not deliberately triggered on live account |
| **Requests Per Day (RPD)** | `429 Too Many Requests` | Blocks requests until daily replenishment | `x-ratelimit-remaining-requests: 0` | Generic behavior documented; not deliberately triggered on live account |
| **Service Contention** | `503 Service Unavailable` | Temporary capacity saturation | Standard retry headers | Generic behavior documented; not deliberately triggered on live account |

> **Safety Notice:** In accordance with assignment guidelines, no test was executed to intentionally exhaust the 1,000 RPD daily limit or force a 429 response on the live account. Specific internal JSON error strings (e.g., `rate_limit_exceeded`, `daily_quota_exceeded`) are described in documentation and simulated in our stand-in service, but remain unverified on the live account.

---

## 6. Actual Response Headers

Live HTTP response headers captured from successful calls to `openai/gpt-oss-20b`:

```http
HTTP/1.1 200 OK
content-type: text/event-stream; charset=utf-8
x-ratelimit-limit-requests: 1000
x-ratelimit-limit-tokens: 8000
x-ratelimit-remaining-requests: 996
x-ratelimit-remaining-tokens: 7582
x-ratelimit-reset-requests: 5m45.6s
x-ratelimit-reset-tokens: 3.135s
x-groq-region: us-west-1
```

### Precise Header Interpretations:
- **`x-ratelimit-limit-requests` (1000):** Represents the observed daily request allocation (**RPD**), not a per-minute limit.
- **`x-ratelimit-limit-tokens` (8000):** Represents the observed per-minute token ceiling (**TPM**).
- **`x-ratelimit-remaining-requests`:** Current remaining requests in the daily quota cycle.
- **`x-ratelimit-remaining-tokens`:** Current remaining tokens available in the rolling minute window.
- **`x-ratelimit-reset-requests` (e.g. `5m45.6s`):** Indicates the recovery interval for the daily request counter. It relates to the daily request allocation (RPD) replenishment mechanism and must not be described as an RPM reset header.
- **`x-ratelimit-reset-tokens` (e.g. `3.135s`):** Relates to token-bucket recovery within the rolling minute window (TPM).
- **`retry-after`:** Documented to appear on HTTP 429 responses indicating the number of seconds to back off. Not present on 200 OK responses.

---

## 7. Real API Experiment Results

To ensure absolute experimental integrity, all measurements are real and traceable to `data/measurements/real_service.json`.

### 7.1 Test A: Single Streaming Request with Reasoning Analysis
- **Model:** `openai/gpt-oss-20b`
- **Config:** `max_tokens: 60`, `stream: true`, `stream_options: {"include_usage": true}`
- **HTTP Status:** 200 OK
- **Timing Terminology & Results:**
  - **First Stream Event:** Approximately 311.6 ms
  - **First User-Visible Content Token:** **`null` (Not observed)**
  - **Total Completion Time:** **599.0 ms**
- **Token Usage Breakdown:**
  - `prompt_tokens`: 82
  - `completion_tokens`: 60
  - `reasoning_tokens`: 58
  - `remaining_non_reasoning_completion_tokens`: 2
  - `visible_answer_content_tokens`: 0
- **Evidence-Based Explanation:** The available telemetry establishes that 60 completion tokens were consumed, of which 58 were reported as `reasoning_tokens`. The remaining two non-reasoning completion tokens are not identified by the response payload; the telemetry does not establish whether they represent formatting, structural tokens, or internal delimiters. Zero visible answer content tokens were emitted before the 60-token budget was exhausted. Consequently, the first user-visible content token latency is recorded as `null`.
- **Engineering Conclusion:** The 60-token completion budget was insufficient to produce visible answer text in this particular test. A higher completion budget should be evaluated experimentally. The appropriate value depends on prompt complexity and reasoning effort. An initial starting configuration of 500–800 tokens is recommended as a baseline for future experiments, but this is an engineering recommendation rather than an experimentally proven minimum.

### 7.2 Test B: 5 Sequential Requests (Baseline Latency & Throughput)
Sequential requests with technical questions from the three target papers:

| Request ID | Target Paper Topic | TTFT (ms) | Total Time (ms) | Tokens In | Tokens Out | HTTP Status |
|---|---|---|---|---|---|---|
| `groq_real_001` | Self-Attention Mechanism | 766.26 | 857.46 | 100 | 144 | 200 OK |
| `groq_real_002` | Mimi Neural Audio Codec | 1,719.29 | 2,061.47 | 101 | 674 | 200 OK |
| `groq_real_003` | Dynamic Chunking Complexity | 628.83 | 902.65 | 101 | 310 | 200 OK |
| `groq_real_004` | Scaled Dot-Product Formula | 757.52 | 855.58 | 98 | 208 | 200 OK |
| `groq_real_005` | Full-Duplex Conversational Audio | 625.99 | 932.79 | 101 | 392 | 200 OK |

#### Statistical Summary ($N = 5$):
- **First-Token Latency (TTFT - User-Visible Content):** Min = 625.99 ms, Median = **757.52 ms**, Max = 1,719.29 ms.
- **Total Completion Latency:** Min = 855.58 ms, Median = **902.65 ms**, Max = 2,061.47 ms.
- **Token Consumption:** Mean Input = 100.2 tokens, Mean Output = 345.6 tokens.
- **Statistical Notice:** The five sequential requests represent a small sample ($N=5$). The observed median latency reflects performance during this single test session and should not be cited as an accurate production SLA estimate. Similarly, this sample size does not establish production $p_{95}$ or $p_{99}$ latency distributions.

### 7.3 Test C: Concurrent Batch Testing
- **Status:** Not run against the live API.
- **Justification:** Multi-request concurrent RAG queries (~2,500 prompt tokens each) would risk exceeding 8,000 TPM and triggering rate limits on the live key. Overload and burst testing are conducted on the stand-in simulator.

---

## 8. Streaming Behavior & GPT-OSS Configuration

Groq streams responses via Server-Sent Events (SSE).

### Key Configuration Directives:
1. **Prefer `max_completion_tokens`:** In modern OpenAI-compatible endpoints, `max_completion_tokens` is preferred over legacy `max_tokens` to explicitly constrain total generated tokens (reasoning + content).
2. **Reasoning Effort Control:** Groq supports the `reasoning_effort` parameter for GPT-OSS models with three levels:
   - `low`: Limits internal thinking steps to conserve tokens and minimize latency.
   - `medium` (default): Balanced reasoning and token consumption.
   - `high`: Extended reasoning for complex logic.
   - *Recommendation:* Evaluate `reasoning_effort="low"` against `"medium"` using actual RAG questions. Note that hiding reasoning text from the client interface does **not** reduce billed token consumption—reasoning tokens are always generated by the model and count against token quotas.
3. **Usage Telemetry:** Passing `"stream_options": {"include_usage": true}` guarantees that the final stream chunk before `data: [DONE]` includes the exact token accounting block.

### Illustrative Streaming Implementation Example
*(Note: Provided as an illustrative implementation reference; not executed as a live test script during this audit. Syntax verified independently via Python AST).*
```python
import os
import time
import json
import httpx
from typing import Optional, Tuple, Dict, Any

def stream_groq_answer(
    prompt: str,
    model: str = "openai/gpt-oss-20b",
    reasoning_effort: str = "low",
    max_completion_tokens: int = 800,
) -> Tuple[str, Optional[float], Optional[float], float, Optional[Dict[str, Any]]]:
    """
    Illustrative implementation: streams an answer from Groq using
    max_completion_tokens and reasoning_effort.
    Accurately separates first stream event from first user-visible content token.
    Never prints or leaks secrets.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a research assistant answering strictly from evidence."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_completion_tokens": max_completion_tokens,
        "reasoning_effort": reasoning_effort,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first_event: Optional[float] = None
    t_first_visible_content: Optional[float] = None
    chunks = []
    final_usage = None

    with httpx.Client(timeout=30.0) as client:
        with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                error_body = response.read().decode("utf-8")
                raise RuntimeError(f"Groq API error HTTP {response.status_code}: {error_body}")

            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    event = json.loads(data_str)
                    if t_first_event is None:
                        t_first_event = time.perf_counter()

                    if "usage" in event and event["usage"]:
                        final_usage = event["usage"]

                    choices = event.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if t_first_visible_content is None:
                                t_first_visible_content = time.perf_counter()
                            chunks.append(content)
                except json.JSONDecodeError:
                    continue

    t_end = time.perf_counter()
    first_event_ms = (t_first_event - t_start) * 1000.0 if t_first_event else None
    first_visible_ms = (t_first_visible_content - t_start) * 1000.0 if t_first_visible_content else None
    total_ms = (t_end - t_start) * 1000.0
    return "".join(chunks), first_event_ms, first_visible_ms, total_ms, final_usage
```

---

## 9. Token Usage and Cost

### Definitions:
- **Input Tokens:** Ingested context passages + system instructions + user query. In RAG queries, passages represent >85% of input tokens.
- **Output Tokens:** User-visible answer text tokens + internal reasoning tokens.
- **Reasoning Billing:** On `openai/gpt-oss-20b`, internal chain-of-thought tokens are billed as output tokens.

### Vendor-Published Developer Plan Pricing:
- **Model:** `openai/gpt-oss-20b`
  - Input: **$0.075 per 1,000,000 tokens** ($0.000075 / 1k)
  - Output: **$0.300 per 1,000,000 tokens** ($0.000300 / 1k)
- **Free Plan Actual Incurred Cost:** **$0.00** (subject to RPD and TPM quotas).

### Cost Calculation Formula (Per 1,000 Answers):
$$\text{Cost}_{1000} = 1000 \times \left( \frac{\bar{T}_{\text{in}}}{10^6} \times P_{\text{in}} + \frac{\bar{T}_{\text{out}}}{10^6} \times P_{\text{out}} \right)$$

*Illustrative Cost Calculation (Setting B: 5 passages $\times$ 600 tokens $\approx$ 2,800 input tokens; 150 output tokens):*
$$\text{Cost}_{1000} = 1000 \times \left( \frac{2800}{10^6} \times 0.075 + \frac{150}{10^6} \times 0.30 \right) = 1000 \times (\$0.000210 + \$0.000045) = \mathbf{\$0.255 \text{ per 1,000 answers}}$$

---

## 10. Timeout and Retry Recommendations

### 1. Client-Side Budgeting:
- **Connect Timeout:** 5.0 seconds.
- **Read Timeout:** 25.0 seconds.
- **Total Request Deadline:** 30.0 to 60.0 seconds.

### 2. Retry Policy:
- **Honor `Retry-After`:** On HTTP 429, parse the `retry-after` header and sleep for that exact duration plus a small jitter (`+ uniform(0.1, 0.5)`).
- **Exponential Backoff with Full Jitter:** For transient 5xx server errors:
  $$t_{\text{backoff}} = \min(8.0, 1.0 \times 2^{\text{attempt}-1}) \times \text{Uniform}(0.5, 1.5)$$
- **Non-Retryable:** Deterministic 4xx errors (400, 401, 403, 404, 422), daily quota exhaustion (`x-ratelimit-remaining-requests: 0`), or when backoff exceeds remaining deadline.
- **Circuit Breaker:** Trip to `OPEN` after 3 consecutive failures; enforce 15s cooldown before probing in `HALF_OPEN`.

---

## 11. Recommended Model for The Relay

**Primary Model: `openai/gpt-oss-20b`**

- **Vendor-Published Speed:** ~1,000 tps rated sequential throughput.
- **Observed Baseline Latency:** Median total latency of ~903 ms across five test requests.
- **Context Capacity:** 131,072 context window easily supports full passage retrieval (350–900 token chunks).
- **Format Integrity:** Consistently outputs markdown citations and mathematical notation accurately.
- **Single Model Architecture:** A single model for generation with rule-based intent classification is recommended to conserve the 8,000 TPM budget.

---

## 12. Known Limitations and Unverified Assumptions

1. **Daily Reset Wall-Clock Time:** Exact daily quota reset wall-clock time has not been independently verified. The observed reset header indicates a recovery interval, not proof of a specific UTC reset time.
2. **HTTP 429 Specific Payloads:** Exact payload JSON keys under real failure were not verified because we did not deliberately trigger a rate limit on the live account.
3. **Small Empirical Sample:** The 5-request baseline provides an empirical median ($p_{50}$), but does not form a statistically valid production-grade $p_{95}$ or $p_{99}$ distribution.
4. **Server Concurrency Limits:** Groq's exact server-side concurrency ceiling is not publicly documented and was not measured.

---

## 13. Official Documentation References

| Resource | Genuine Source URL | Access Date | Documented Metric |
|---|---|---|---|
| Groq Rate Limits Guide | https://console.groq.com/docs/rate-limits | 2026-11-10 | Free Plan & Developer Plan limits |
| Groq Models Catalog | https://console.groq.com/docs/models | 2026-11-10 | Context windows, max completion tokens, model IDs |
| Groq ALLaM-2-7B Model Page | https://console.groq.com/docs/model/allam-2-7b | 2026-11-10 | Context: 4,096; Max output: 4,096 |
| Groq Prompt Guard Model Page | https://console.groq.com/docs/model/meta-llama/llama-prompt-guard-2-22m | 2026-11-10 | Context: 512; Max output: 512 |
| Groq Whisper Model Page | https://console.groq.com/docs/model/whisper-large-v3 | 2026-11-10 | Audio transcription endpoint specifications |
| Groq Reasoning Guide | https://console.groq.com/docs/reasoning | 2026-11-10 | `reasoning_effort`, `max_completion_tokens` |
| Groq Cloud Pricing | https://groq.com/pricing | 2026-11-10 | Developer Plan input/output pricing |
| Groq Errors Documentation | https://console.groq.com/docs/errors | 2026-11-10 | HTTP 429, 503 error handling |

---

## 14. Implementation Recommendations for The Relay

1. **Environment Configuration:**
   ```env
   GROQ_API_KEY=your_key_here
   GROQ_MODEL=openai/gpt-oss-20b
   PROVIDER_TYPE=standin
   CLIENT_RPM_LIMIT=30
   CLIENT_RPD_LIMIT=1000
   CLIENT_TPM_LIMIT=8000
   CLIENT_TPD_LIMIT=200000
   PROVIDER_TIMEOUT=30.0
   MAX_RETRIES=3
   ```
2. **Token-Budget Admission Controller vs. Concurrency Semaphore:**  
   It is critical to distinguish between rate limits and concurrency:
   - **TPM (8,000):** A throughput quota over a rolling 60-second window.
   - **Concurrency Semaphore:** Limits the number of simultaneous in-flight HTTP connections.
   A concurrency cap alone (e.g. `MAX_CONCURRENT_REQUESTS=2`) does not guarantee compliance with an 8,000 TPM limit. For example, if two requests each contain ~3,500 input tokens and request an 800-token completion budget, their combined demand (~8,600 tokens) would exceed the rolling minute window. Therefore, the pipeline must implement a token-budget admission controller that estimates input tokens plus the configured completion budget before dispatch. The safe request volume depends on actual token consumption and timing.
3. **Reasoning Effort Setting:**  
   Evaluate `reasoning_effort="low"` for factual lookup queries to prevent internal reasoning tokens from prematurely consuming the 8,000 TPM budget.

---

## 15. Changes Made During Engineering Review

This section details all modifications made to the research documentation during the formal engineering review:

1. **Plan Terminology Disambiguation (Section 3 & 4):**  
   *Correction:* Eliminated blended terms such as "Free Developer Tier" and "Free / Self-Serve Developer". Consistently applied distinct labels: **Free Plan**, **Developer Plan**, **Account-Observed**, and **Unverified Limits**. Stated that while account headers match the Free Plan baseline (1,000 RPD, 8,000 TPM), the exact organization plan has not been independently verified via the console dashboard.
2. **GPT-OSS Safeguard 20B Rate Limits Reconciled (Section 4.4):**  
   *Correction:* Updated Free Plan candidate rate limits to RPM: 3, RPD: 1,000, TPM: 2,000, and TPD: 200,000, distinguishing them from Developer Plan ceilings (1,000 RPM, 250,000 TPM).
3. **Whisper Rate Limits & Audio Metric Fields Added (Section 4.3):**  
   *Correction:* Added dedicated Audio Seconds per Hour (**ASH: 7,200**) and Audio Seconds per Day (**ASD: 28,800**) metrics under the Free Plan (with 20 RPM and 2,000 RPD), contrasting them with Developer Plan audio limits (up to 400,000 ASH). Kept token fields (`tpm`, `tpd`) as `null` for audio models to prevent category errors.
4. **ALLaM-2-7B Maximum Output Corrected (Section 4.2):**  
   *Correction:* Corrected maximum output tokens from 2,048 to **4,096** as documented on the official Groq ALLaM model page. Marked Free Plan rate limits as unverified since it is not listed in the standard Free Plan table.
5. **Prompt Guard Maximum Output Corrected (Section 4.4):**  
   *Correction:* Corrected maximum output tokens from 1 to **512** for both `meta-llama/llama-prompt-guard-2-22m` and `meta-llama/llama-prompt-guard-2-86m` per official documentation.
6. **Orpheus Model Specifications Added (Section 4.5):**  
   *Correction:* Added verified specifications for `canopylabs/orpheus-v1-english` and `canopylabs/orpheus-arabic-saudi` (Context Window: **4,000**, Max Output: **50,000**), preserving Free Plan limits (RPM: 10, RPD: 100, TPM: 1,200, TPD: 3,600).
7. **Streaming Telemetry Code Corrected (Section 8):**  
   *Correction:* Fixed the illustrative Python streaming snippet to separately track and return `first_event_ms` (time to first valid SSE chunk) and `first_visible_ms` (time to first chunk containing non-empty `choices[0].delta.content`). Syntax verified via Python AST parser.
8. **Preservation of Empirical Evidence:**  
   *Correction:* Preserved all genuine experimental observations from Version 3: Test A token breakdown (60 completion, 58 reasoning, 2 unidentified, 0 visible content, `null` TTFT) and Test B 5 sequential requests with baseline medians (757.52 ms TTFT, 902.65 ms total).

