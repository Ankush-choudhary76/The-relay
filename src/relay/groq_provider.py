"""Groq hosted API provider implementation with streaming and rate-limit telemetry."""

import json
import time
from typing import AsyncGenerator, Dict, List, Optional, Tuple
import httpx

from relay.config import settings
from relay.provider_interface import ModelProvider
from relay.schemas import ProviderResponse


class GroqProvider(ModelProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = settings.groq_api_key if api_key is None else api_key
        self.model = model or settings.groq_model
        self.base_url = (base_url or settings.groq_base_url).rstrip("/")
        self.timeout = timeout or settings.provider_timeout

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def is_simulated(self) -> bool:
        return False

    def _sanitize_error(self, message: str) -> str:
        """Ensures any accidental token/key leak in error responses is redacted."""
        if self.api_key and self.api_key in message:
            return message.replace(self.api_key, "[REDACTED_API_KEY]")
        return message

    def _get_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please set it in .env or configure backend environment."
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "TheRelay-Benchmark/1.0",
        }

    def _extract_rate_limit_headers(self, headers: httpx.Headers) -> Dict[str, str]:
        """Extracts x-ratelimit-* and retry-after headers."""
        extracted = {}
        for key in [
            "x-ratelimit-limit-requests",
            "x-ratelimit-limit-tokens",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-remaining-tokens",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
            "retry-after",
        ]:
            if key in headers:
                extracted[key] = headers[key]
        return extracted

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        **kwargs,
    ) -> ProviderResponse:
        """Executes a non-streaming chat completion to Groq."""
        target_model = model or self.model

        # Check API key before making any call
        if not self.api_key:
            return ProviderResponse(
                text="",
                tokens_in=0,
                tokens_out=0,
                first_token_ms=0.0,
                total_ms=0.0,
                status_code=401,
                is_error=True,
                error_message="GROQ_API_KEY is not configured in the backend environment. Please set GROQ_API_KEY to use Real Groq API.",
                provider=self.provider_name,
                model=target_model,
                is_simulated=False,
            )

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        t_start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, headers=self._get_headers(), json=payload)
                t_end = time.perf_counter()
                total_ms = (t_end - t_start) * 1000.0

                rl_headers = self._extract_rate_limit_headers(resp.headers)

                if resp.status_code != 200:
                    err_text = self._sanitize_error(resp.text)
                    return ProviderResponse(
                        text="",
                        tokens_in=0,
                        tokens_out=0,
                        first_token_ms=total_ms,
                        total_ms=total_ms,
                        status_code=resp.status_code,
                        headers=rl_headers,
                        is_error=True,
                        error_message=f"Groq API HTTP {resp.status_code}: {err_text[:200]}",
                        provider=self.provider_name,
                        model=target_model,
                        is_simulated=False,
                    )

                data = resp.json()
                choice = data.get("choices", [{}])[0]
                text = choice.get("message", {}).get("content", "")
                usage = data.get("usage", {})
                tokens_in = usage.get("prompt_tokens", 0)
                tokens_out = usage.get("completion_tokens", 0)

                return ProviderResponse(
                    text=text,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    first_token_ms=total_ms,
                    total_ms=total_ms,
                    status_code=resp.status_code,
                    headers=rl_headers,
                    is_error=False,
                    provider=self.provider_name,
                    model=target_model,
                    is_simulated=False,
                )
        except httpx.TimeoutException:
            t_end = time.perf_counter()
            total_ms = (t_end - t_start) * 1000.0
            return ProviderResponse(
                text="",
                first_token_ms=total_ms,
                total_ms=total_ms,
                status_code=504,
                is_error=True,
                error_message="Groq API request timed out.",
                provider=self.provider_name,
                model=target_model,
                is_simulated=False,
            )
        except Exception as e:
            t_end = time.perf_counter()
            total_ms = (t_end - t_start) * 1000.0
            err_msg = self._sanitize_error(str(e))
            return ProviderResponse(
                text="",
                first_token_ms=total_ms,
                total_ms=total_ms,
                status_code=500,
                is_error=True,
                error_message=f"Groq client exception: {err_msg}",
                provider=self.provider_name,
                model=target_model,
                is_simulated=False,
            )

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        **kwargs,
    ) -> AsyncGenerator[Tuple[str, Optional[ProviderResponse]], None]:
        """Executes a streaming chat completion to Groq, measuring TTFT accurately."""
        target_model = model or self.model

        if not self.api_key:
            res = ProviderResponse(
                text="",
                tokens_in=0,
                tokens_out=0,
                first_token_ms=0.0,
                total_ms=0.0,
                status_code=401,
                is_error=True,
                error_message="GROQ_API_KEY is not configured in the backend environment. Please set GROQ_API_KEY to use Real Groq API.",
                provider=self.provider_name,
                model=target_model,
                is_simulated=False,
            )
            yield ("", res)
            return

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        t_start = time.perf_counter()
        first_token_ms = 0.0
        accumulated_text: List[str] = []
        tokens_in = 0
        tokens_out = 0
        rl_headers: Dict[str, str] = {}
        status_code = 200

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", url, headers=self._get_headers(), json=payload
                ) as response:
                    status_code = response.status_code
                    rl_headers = self._extract_rate_limit_headers(response.headers)

                    if status_code != 200:
                        err_bytes = await response.aread()
                        err_text = self._sanitize_error(err_bytes.decode('utf-8', errors='ignore'))
                        total_ms = (time.perf_counter() - t_start) * 1000.0
                        res = ProviderResponse(
                            text="",
                            first_token_ms=total_ms,
                            total_ms=total_ms,
                            status_code=status_code,
                            headers=rl_headers,
                            is_error=True,
                            error_message=f"Groq stream HTTP {status_code}: {err_text[:200]}",
                            provider=self.provider_name,
                            model=target_model,
                            is_simulated=False,
                        )
                        yield ("", res)
                        return

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            usage = chunk.get("usage")
                            if usage:
                                tokens_in = usage.get("prompt_tokens", tokens_in)
                                tokens_out = usage.get("completion_tokens", tokens_out)

                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {}).get("content", "")
                                if delta:
                                    if first_token_ms == 0.0:
                                        first_token_ms = (time.perf_counter() - t_start) * 1000.0
                                    accumulated_text.append(delta)
                                    yield (delta, None)
                        except json.JSONDecodeError:
                            continue

            total_ms = (time.perf_counter() - t_start) * 1000.0
            if first_token_ms == 0.0:
                first_token_ms = total_ms

            full_text = "".join(accumulated_text)
            if tokens_out == 0:
                tokens_out = max(1, len(full_text.split()))

            final_response = ProviderResponse(
                text=full_text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                first_token_ms=round(first_token_ms, 2),
                total_ms=round(total_ms, 2),
                status_code=status_code,
                headers=rl_headers,
                is_error=False,
                provider=self.provider_name,
                model=target_model,
                is_simulated=False,
            )
            yield ("", final_response)

        except Exception as e:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            err_msg = self._sanitize_error(str(e))
            res = ProviderResponse(
                text="".join(accumulated_text),
                first_token_ms=round(first_token_ms or total_ms, 2),
                total_ms=round(total_ms, 2),
                status_code=500,
                headers=rl_headers,
                is_error=True,
                error_message=f"Groq stream error: {err_msg}",
                provider=self.provider_name,
                model=target_model,
                is_simulated=False,
            )
            yield ("", res)
