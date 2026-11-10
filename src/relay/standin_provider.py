"""Provider client adapter for the Stand-in model service."""

import json
import time
from typing import AsyncGenerator, Dict, List, Optional, Tuple
import httpx

from relay.config import settings
from relay.provider_interface import ModelProvider
from relay.schemas import ProviderResponse
from standin.app import app as standin_app


class StandinProvider(ModelProvider):
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        use_asgi: bool = True,
        model: Optional[str] = "standin-calibrated-lpu",
    ):
        self.base_url = (base_url or settings.standin_base_url).rstrip("/")
        self.timeout = timeout or settings.provider_timeout
        self.use_asgi = use_asgi
        self._model = model or "standin-calibrated-lpu"

    @property
    def provider_name(self) -> str:
        return "standin"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def is_simulated(self) -> bool:
        return True

    def _get_client(self) -> httpx.AsyncClient:
        if self.use_asgi:
            return httpx.AsyncClient(
                transport=httpx.ASGITransport(app=standin_app),
                base_url="http://standin",
                timeout=self.timeout,
            )
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    def _extract_headers(self, headers: httpx.Headers) -> Dict[str, str]:
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
        target_model = model or self._model
        url = "/v1/chat/completions" if self.use_asgi else f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        t_start = time.perf_counter()
        try:
            async with self._get_client() as client:
                resp = await client.post(url, json=payload)
                t_end = time.perf_counter()
                total_ms = (t_end - t_start) * 1000.0

                headers = self._extract_headers(resp.headers)

                if resp.status_code != 200:
                    return ProviderResponse(
                        text="",
                        first_token_ms=total_ms,
                        total_ms=total_ms,
                        status_code=resp.status_code,
                        headers=headers,
                        is_error=True,
                        error_message=f"Stand-in HTTP {resp.status_code}: {resp.text[:200]}",
                        provider=self.provider_name,
                        model=target_model,
                        is_simulated=True,
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
                    headers=headers,
                    is_error=False,
                    provider=self.provider_name,
                    model=target_model,
                    is_simulated=True,
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
                error_message="Stand-in provider request timed out.",
                provider=self.provider_name,
                model=target_model,
                is_simulated=True,
            )
        except Exception as e:
            t_end = time.perf_counter()
            total_ms = (t_end - t_start) * 1000.0
            return ProviderResponse(
                text="",
                first_token_ms=total_ms,
                total_ms=total_ms,
                status_code=500,
                is_error=True,
                error_message=f"Stand-in client error: {str(e)}",
                provider=self.provider_name,
                model=target_model,
                is_simulated=True,
            )

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        **kwargs,
    ) -> AsyncGenerator[Tuple[str, Optional[ProviderResponse]], None]:
        target_model = model or self._model
        url = "/v1/chat/completions" if self.use_asgi else f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        t_start = time.perf_counter()
        first_token_ms = 0.0
        accumulated_text: List[str] = []
        tokens_in = 0
        tokens_out = 0
        headers_dict: Dict[str, str] = {}
        status_code = 200

        try:
            async with self._get_client() as client:
                async with client.stream("POST", url, json=payload) as response:
                    status_code = response.status_code
                    headers_dict = self._extract_headers(response.headers)

                    if status_code != 200:
                        err_bytes = await response.aread()
                        total_ms = (time.perf_counter() - t_start) * 1000.0
                        res = ProviderResponse(
                            text="",
                            first_token_ms=total_ms,
                            total_ms=total_ms,
                            status_code=status_code,
                            headers=headers_dict,
                            is_error=True,
                            error_message=f"Stand-in stream HTTP {status_code}: {err_bytes.decode('utf-8', errors='ignore')[:200]}",
                            provider=self.provider_name,
                            model=target_model,
                            is_simulated=True,
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

            final_res = ProviderResponse(
                text=full_text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                first_token_ms=round(first_token_ms, 2),
                total_ms=round(total_ms, 2),
                status_code=status_code,
                headers=headers_dict,
                is_error=False,
                provider=self.provider_name,
                model=target_model,
                is_simulated=True,
            )
            yield ("", final_res)

        except Exception as e:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            res = ProviderResponse(
                text="".join(accumulated_text),
                first_token_ms=round(first_token_ms or total_ms, 2),
                total_ms=round(total_ms, 2),
                status_code=500,
                headers=headers_dict,
                is_error=True,
                error_message=f"Stand-in stream exception: {str(e)}",
                provider=self.provider_name,
                model=target_model,
                is_simulated=True,
            )
            yield ("", res)
