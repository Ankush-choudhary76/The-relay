"""Comprehensive tests for backend model provider switching (Stand-in vs. Real Groq).

Tests A through G verify:
- Test A: Stand-in selection invokes simulator without calling Groq.
- Test B: Real Groq selection invokes Groq adapter with configured model.
- Test C: Runtime switching back and forth on the same pipeline instance.
- Test D: Missing API key returns clear backend error without silent fallback.
- Test E: Streaming generation works independently for both providers.
- Test F: Concurrent requests maintain strict provider isolation.
- Test G: Unsupported provider values are rejected by validation.
"""

import asyncio
from unittest.mock import AsyncMock, patch
import pytest

from relay.chunking import chunk_papers
from relay.groq_provider import GroqProvider
from relay.papers import extract_all_papers
from relay.pipeline import RelayPipeline
from relay.schemas import ProviderResponse, RelayRequest, RelayResponse
from relay.standin_provider import StandinProvider
from standin.app import quota_tracker


# ---------------------------------------------------------------------------
# Test Fixtures (cached at module level for speed)
# ---------------------------------------------------------------------------

_CACHED_CHUNKS = None
_CACHED_PAGES = None


def _get_test_engine():
    global _CACHED_CHUNKS, _CACHED_PAGES
    if _CACHED_CHUNKS is None:
        extracted = extract_all_papers()
        _CACHED_CHUNKS = chunk_papers(extracted, chunk_size_tokens=600, overlap_tokens=90)
        _CACHED_PAGES = {
            p.paper_id: {page.page_number: page.text for page in p.pages}
            for p in extracted
        }
    return _CACHED_CHUNKS, _CACHED_PAGES


def _create_pipeline(mock_groq=None, mock_standin=None):
    chunks, paper_pages = _get_test_engine()
    standin = mock_standin or StandinProvider()
    groq = mock_groq or GroqProvider(api_key="gsk_test_mock_key_for_routing_only")
    return RelayPipeline(
        chunks=chunks,
        paper_pages_text=paper_pages,
        standin_provider=standin,
        groq_provider=groq,
    )


# ---------------------------------------------------------------------------
# Test A: Stand-in Selection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_standin_selection():
    """Verify selecting stand-in invokes simulator, marks is_simulated=True, and never calls Groq."""
    quota_tracker.reset()
    mock_groq = GroqProvider(api_key="gsk_test_dummy_key")
    mock_groq.generate = AsyncMock()

    pipeline = _create_pipeline(mock_groq=mock_groq)

    req = RelayRequest(
        question="What is Scaled Dot-Product Attention in the Transformer?",
        provider="standin",
    )
    resp = await pipeline.execute(req)

    assert resp.status == "ok"
    assert resp.provider == "standin"
    assert resp.is_simulated is True
    assert resp.source == "stand-in"
    assert "Attention" in resp.answer
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
    assert resp.latency_ms > 0

    # Ensure Groq was NOT called
    mock_groq.generate.assert_not_called()


# ---------------------------------------------------------------------------
# Test B: Real Groq Selection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_b_real_groq_selection():
    """Verify selecting real Groq calls Groq adapter with configured model and is_simulated=False."""
    mock_groq = GroqProvider(api_key="gsk_test_dummy_key", model="openai/gpt-oss-20b")
    canned_answer = (
        "Scaled Dot-Product Attention is computed as Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V "
        "[Citation: Attention Is All You Need, arXiv:1706.03762v7, Page 4, Section 3.2.1]."
    )
    mock_groq.generate = AsyncMock(
        return_value=ProviderResponse(
            text=canned_answer,
            tokens_in=350,
            tokens_out=42,
            first_token_ms=450.0,
            total_ms=520.0,
            status_code=200,
            provider="groq",
            model="openai/gpt-oss-20b",
            is_simulated=False,
        )
    )

    pipeline = _create_pipeline(mock_groq=mock_groq)

    req = RelayRequest(
        question="What is Scaled Dot-Product Attention in the Transformer?",
        provider="groq",
    )
    resp = await pipeline.execute(req)

    assert resp.status == "ok"
    assert resp.provider == "groq"
    assert resp.is_simulated is False
    assert resp.source == "real"
    assert resp.model == "openai/gpt-oss-20b"
    assert resp.input_tokens == 350
    assert resp.output_tokens == 42
    assert resp.total_tokens == 392
    assert len(resp.citations) == 1

    # Verify Groq was indeed called once
    assert mock_groq.generate.call_count == 1


# ---------------------------------------------------------------------------
# Test C: Runtime Provider Switching
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_c_runtime_switching():
    """Verify that a single pipeline instance dynamically switches between Stand-in and Groq."""
    quota_tracker.reset()
    mock_groq = GroqProvider(api_key="gsk_test_dummy_key", model="openai/gpt-oss-20b")
    mock_groq.generate = AsyncMock(
        return_value=ProviderResponse(
            text="Groq generated answer [Citation: Attention Is All You Need, arXiv:1706.03762v7, Page 4, Section 3.2.1].",
            tokens_in=200,
            tokens_out=25,
            first_token_ms=300.0,
            total_ms=350.0,
            status_code=200,
            provider="groq",
            model="openai/gpt-oss-20b",
            is_simulated=False,
        )
    )

    pipeline = _create_pipeline(mock_groq=mock_groq)

    # 1. Request with Stand-in
    req1 = RelayRequest(
        question="What is Scaled Dot-Product Attention in the Transformer?",
        provider="standin",
    )
    resp1 = await pipeline.execute(req1)
    assert resp1.provider == "standin"
    assert resp1.is_simulated is True
    assert mock_groq.generate.call_count == 0

    # 2. Switch to Real Groq on the SAME pipeline instance
    req2 = RelayRequest(
        question="What is Scaled Dot-Product Attention in the Transformer?",
        provider="groq",
    )
    resp2 = await pipeline.execute(req2)
    assert resp2.provider == "groq"
    assert resp2.is_simulated is False
    assert mock_groq.generate.call_count == 1

    # 3. Switch back to Stand-in
    quota_tracker.reset()
    req3 = RelayRequest(
        question="What is the frame rate of the Mimi neural audio codec?",
        provider="standin",
    )
    resp3 = await pipeline.execute(req3)
    assert resp3.provider == "standin"
    assert resp3.is_simulated is True
    assert mock_groq.generate.call_count == 1  # No additional Groq call


# ---------------------------------------------------------------------------
# Test D: Missing API Key Error (No Silent Fallback)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d_missing_api_key_no_fallback():
    """Verify missing GROQ_API_KEY returns a clear error and never silently falls back to standin."""
    groq_without_key = GroqProvider(api_key="")
    pipeline = _create_pipeline(mock_groq=groq_without_key)

    req = RelayRequest(
        question="What is Scaled Dot-Product Attention in the Transformer?",
        provider="groq",
    )
    resp = await pipeline.execute(req)

    # Must be error status
    assert resp.status == "error"
    assert resp.provider == "groq"
    assert resp.is_simulated is False
    assert resp.refusal_reason is not None
    assert "GROQ_API_KEY" in resp.refusal_reason
    # Answer must NOT contain simulator text
    assert resp.answer == ""


# ---------------------------------------------------------------------------
# Test E: Streaming Support for Both Providers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e_streaming_both_providers():
    """Verify execute_stream yields token deltas and final RelayResponse for both providers."""
    quota_tracker.reset()
    pipeline = _create_pipeline()
    req_standin = RelayRequest(
        question="What is Scaled Dot-Product Attention in the Transformer?",
        provider="standin",
        stream=True,
    )
    deltas_standin = []
    final_standin = None

    async for delta, resp in pipeline.execute_stream(req_standin):
        if delta:
            deltas_standin.append(delta)
        if resp:
            final_standin = resp

    assert len(deltas_standin) > 0
    assert final_standin is not None
    assert final_standin.provider == "standin"
    assert final_standin.is_simulated is True
    assert final_standin.status == "ok"

    # 2. Groq stream
    mock_groq = GroqProvider(api_key="gsk_test_dummy_key")

    async def mock_stream_gen(messages, **kwargs):
        tokens = [
            "Attention", " mechanism", " operates", " efficiently",
            " [Citation: Attention Is All You Need, arXiv:1706.03762v7, Page 4, Section 3.2.1]."
        ]
        for t in tokens:
            yield (t, None)
        final = ProviderResponse(
            text="".join(tokens),
            tokens_in=250,
            tokens_out=15,
            first_token_ms=320.0,
            total_ms=400.0,
            status_code=200,
            provider="groq",
            model="openai/gpt-oss-20b",
            is_simulated=False,
        )
        yield ("", final)

    mock_groq.generate_stream = mock_stream_gen
    pipeline_groq = _create_pipeline(mock_groq=mock_groq)

    req_groq = RelayRequest(
        question="What is Scaled Dot-Product Attention in the Transformer?",
        provider="groq",
        stream=True,
    )
    deltas_groq = []
    final_groq = None

    async for delta, resp in pipeline_groq.execute_stream(req_groq):
        if delta:
            deltas_groq.append(delta)
        if resp:
            final_groq = resp

    assert len(deltas_groq) == 5
    assert final_groq is not None
    assert final_groq.provider == "groq"
    assert final_groq.is_simulated is False
    assert final_groq.status == "ok"


# ---------------------------------------------------------------------------
# Test F: Concurrent Request Isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_f_concurrent_requests_isolation():
    """Verify concurrent requests with different providers maintain isolation without state contamination."""
    quota_tracker.reset()
    mock_groq = GroqProvider(api_key="gsk_test_dummy_key")

    async def mock_groq_gen(messages, **kwargs):
        await asyncio.sleep(0.05)  # Simulate network latency
        return ProviderResponse(
            text="Groq response [Citation: Attention Is All You Need, arXiv:1706.03762v7, Page 4, Section 3.2.1].",
            tokens_in=180,
            tokens_out=20,
            first_token_ms=280.0,
            total_ms=310.0,
            status_code=200,
            provider="groq",
            model="openai/gpt-oss-20b",
            is_simulated=False,
        )

    mock_groq.generate = mock_groq_gen
    pipeline = _create_pipeline(mock_groq=mock_groq)

    # Create 6 interleaved requests
    tasks = []
    expected_providers = []
    for i in range(6):
        prov = "standin" if (i % 2 == 0) else "groq"
        expected_providers.append(prov)
        req = RelayRequest(
            question="What is Scaled Dot-Product Attention in the Transformer?",
            provider=prov,
        )
        tasks.append(pipeline.execute(req))

    results = await asyncio.gather(*tasks)

    for i, res in enumerate(results):
        exp = expected_providers[i]
        assert res.provider == exp, f"Request {i} expected provider {exp}, got {res.provider}"
        if exp == "standin":
            assert res.is_simulated is True
            assert res.source == "stand-in"
        else:
            assert res.is_simulated is False
            assert res.source == "real"


# ---------------------------------------------------------------------------
# Test G: Unsupported Provider Rejection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_g_invalid_provider_rejected():
    """Verify unsupported provider strings are strictly rejected by validation."""
    # 1. Pydantic validation rejection on schema instantiation
    with pytest.raises(Exception) as exc_info:
        RelayRequest(
            question="What is Scaled Dot-Product Attention in the Transformer?",
            provider="claude-opus-fake",
        )
    assert "Invalid model provider" in str(exc_info.value)

    # 2. Pipeline get_provider validation
    pipeline = _create_pipeline()
    with pytest.raises(ValueError) as val_err:
        pipeline.get_provider("unsupported-provider-xyz")
    assert "Unsupported model provider" in str(val_err.value)
