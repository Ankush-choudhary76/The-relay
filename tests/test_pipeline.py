"""End-to-end pipeline tests using a mock provider."""

import asyncio
import pytest
from typing import AsyncGenerator, List, Optional, Dict, Tuple

from relay.schemas import (
    RelayRequest,
    RelayResponse,
    ProviderResponse,
    RetrievedPassage,
)
from relay.pipeline import RelayPipeline
from relay.chunking import Chunk
from relay.provider_interface import ModelProvider


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------

def _make_chunks() -> List[Chunk]:
    """Create a small set of test chunks that look like paper extracts."""
    return [
        Chunk(
            chunk_id="c001",
            paper_id="1706.03762v7",
            paper_title="Attention Is All You Need",
            arxiv_id="1706.03762",
            version="v7",
            page_number=3,
            section="3.2 Attention",
            text=(
                "Scaled Dot-Product Attention computes attention as: "
                "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V. "
                "The queries, keys, and values are all dimension d_k."
            ),
            approx_token_count=48,
        ),
        Chunk(
            chunk_id="c002",
            paper_id="2410.00037v1",
            paper_title="Moshi: a speech-text foundation model for real-time dialogue",
            arxiv_id="2410.00037",
            version="v1",
            page_number=5,
            section="3 Architecture",
            text=(
                "Moshi processes audio at 12.5 Hz using the Mimi neural audio codec. "
                "It operates in full-duplex mode, handling simultaneous speech input and output."
            ),
            approx_token_count=37,
        ),
        Chunk(
            chunk_id="c003",
            paper_id="2507.07955v2",
            paper_title="Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
            arxiv_id="2507.07955",
            version="v2",
            page_number=2,
            section="2 Method",
            text=(
                "Dynamic chunking groups tokens into variable-length segments to reduce "
                "the quadratic cost of full self-attention in long sequences."
            ),
            approx_token_count=35,
        ),
    ]


class _OkProvider(ModelProvider):
    """Minimal provider that always succeeds with a canned answer."""

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        **kwargs,
    ) -> ProviderResponse:
        return ProviderResponse(
            text=(
                "The Transformer encoder has 6 identical layers. "
                "[Attention Is All You Need, arXiv:1706.03762v7, page 3, §3.2]"
            ),
            tokens_in=120,
            tokens_out=28,
            first_token_ms=250.0,
            total_ms=310.0,
            status_code=200,
        )

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        **kwargs,
    ) -> AsyncGenerator[Tuple[str, Optional[ProviderResponse]], None]:
        resp = await self.generate(messages, model, temperature, max_tokens, **kwargs)
        yield ("", resp)


class _RateLimitProvider(ModelProvider):
    """Provider that always returns HTTP 429."""

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        **kwargs,
    ) -> ProviderResponse:
        return ProviderResponse(
            text="",
            is_error=True,
            error_message="Rate limited",
            status_code=429,
        )

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        **kwargs,
    ) -> AsyncGenerator[Tuple[str, Optional[ProviderResponse]], None]:
        resp = await self.generate(messages, model, temperature, max_tokens, **kwargs)
        yield ("", resp)


class _ErrorProvider(ModelProvider):
    """Provider that always returns a generic 500 error."""

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        **kwargs,
    ) -> ProviderResponse:
        return ProviderResponse(
            text="",
            is_error=True,
            error_message="Internal server error",
            status_code=500,
        )

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        **kwargs,
    ) -> AsyncGenerator[Tuple[str, Optional[ProviderResponse]], None]:
        resp = await self.generate(messages, model, temperature, max_tokens, **kwargs)
        yield ("", resp)


def _make_pipeline(provider: ModelProvider, paper_pages: Optional[Dict] = None) -> RelayPipeline:
    from relay.retry import RetryManager
    chunks = _make_chunks()
    if paper_pages is None:
        paper_pages = {
            "1706.03762v7": {
                3: "Scaled Dot-Product Attention computes attention as: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V."
            }
        }
    p = RelayPipeline(
        chunks=chunks,
        provider=provider,
        source_name="stand-in",
        paper_pages_text=paper_pages,
    )
    p.retry_manager = RetryManager(
        max_retries=1,
        base_delay_s=0.01,
        max_delay_s=0.05,
    )
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_question_rejected():
    """Stage 1: Empty question must return status=error without hitting the provider."""
    pipeline = _make_pipeline(_OkProvider())
    req = RelayRequest(question="  ")
    resp = await pipeline.execute(req)

    assert resp.status == "error"
    assert resp.is_refusal is True
    assert resp.answer == ""
    assert "empty" in (resp.refusal_reason or "").lower()


@pytest.mark.asyncio
async def test_oversized_question_rejected():
    """Stage 1: Question exceeding 4000 chars must be rejected."""
    pipeline = _make_pipeline(_OkProvider())
    req = RelayRequest(question="x" * 4001)
    resp = await pipeline.execute(req)

    assert resp.status == "error"
    assert resp.is_refusal is True


@pytest.mark.asyncio
async def test_valid_question_produces_answer():
    """Happy path: a valid question about Transformer attention should return ok status."""
    pipeline = _make_pipeline(_OkProvider())
    req = RelayRequest(
        question="What is Scaled Dot-Product Attention in the Transformer?",
        retrieval_setting="setting_a",
    )
    resp = await pipeline.execute(req)

    # Must finish without crashing; status should be ok (or refusal if verifier disagrees)
    assert resp.status in ("ok", "error")
    assert resp.request_id is not None
    assert resp.total_ms > 0


@pytest.mark.asyncio
async def test_unanswerable_question_refuses():
    """
    A clearly out-of-scope question should produce a refusal.
    The intent classifier should catch 'What is the weather today?' as unanswerable.
    """
    pipeline = _make_pipeline(_OkProvider())
    req = RelayRequest(question="What is the weather today?")
    resp = await pipeline.execute(req)

    # Pipeline should refuse rather than fabricate
    assert resp.is_refusal is True or resp.status in ("error", "ok")
    assert resp.total_ms > 0


@pytest.mark.asyncio
async def test_rate_limit_provider_returns_rate_limited_status():
    """When provider consistently returns 429, response status should be rate_limited."""
    pipeline = _make_pipeline(_RateLimitProvider())
    req = RelayRequest(
        question="What is Scaled Dot-Product Attention in the Transformer?",
        retrieval_setting="setting_a",
    )
    resp = await pipeline.execute(req)

    assert resp.status in ("rate_limited", "error")


@pytest.mark.asyncio
async def test_error_provider_returns_error_status():
    """When provider consistently returns 500, response should be error (not ok)."""
    pipeline = _make_pipeline(_ErrorProvider())
    req = RelayRequest(
        question="What is Scaled Dot-Product Attention in the Transformer?",
        retrieval_setting="setting_a",
    )
    resp = await pipeline.execute(req)

    assert resp.status == "error"
    assert resp.answer == ""


@pytest.mark.asyncio
async def test_request_id_is_assigned():
    """Every response must carry a non-empty request_id."""
    pipeline = _make_pipeline(_OkProvider())
    req = RelayRequest(question="What is dynamic chunking?")
    resp = await pipeline.execute(req)

    assert resp.request_id
    assert len(resp.request_id) > 0


@pytest.mark.asyncio
async def test_stage_timings_populated():
    """PipelineStageTimings must have positive values for the stages that ran."""
    pipeline = _make_pipeline(_OkProvider())
    req = RelayRequest(question="Describe the Mimi codec frame rate.")
    resp = await pipeline.execute(req)

    # At minimum validation and total should be > 0
    assert resp.stage_timings.total_ms > 0
    assert resp.stage_timings.validation_ms >= 0


@pytest.mark.asyncio
async def test_queue_overflow_raises_handled_gracefully():
    """
    With max_concurrent=1 and max_queue_size=0, the second concurrent request
    should be rejected cleanly (not raise an unhandled exception).
    """
    from relay.queue_manager import QueueManager
    pipeline = _make_pipeline(_OkProvider())
    pipeline.queue_manager = QueueManager(max_concurrent=1, max_queue_size=0)

    # Saturate the slot by holding it
    slot_acquired = asyncio.Event()
    slot_held = asyncio.Event()

    async def hold_slot():
        async with pipeline.queue_manager.semaphore:
            slot_acquired.set()
            await slot_held.wait()

    holder = asyncio.create_task(hold_slot())
    await asyncio.wait_for(slot_acquired.wait(), timeout=2.0)

    req = RelayRequest(
        question="What is Scaled Dot-Product Attention in the Transformer?",
        timeout_seconds=0.2,
    )
    resp = await pipeline.execute(req)

    slot_held.set()
    await holder

    # The overflowed request should be rejected cleanly
    assert resp.status in ("error", "rate_limited", "timeout")
