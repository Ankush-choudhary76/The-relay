"""The Relay Pipeline: Multi-stage fault-tolerant question answering."""

import asyncio
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from relay.chunking import Chunk
from relay.citation_verifier import CitationVerifier
from relay.config import settings
from relay.evidence import EvidenceEvaluator
from relay.generation import AnswerGenerator, REFUSAL_MESSAGE
from relay.intent import IntentClassifier
from relay.math_format import format_math_unicode
from relay.provider_interface import ModelProvider
from relay.queue_manager import QueueFullException, QueueManager
from relay.rate_limiter import ClientRateLimiter
from relay.retrieval import DocumentRetriever
from relay.retry import CircuitBreakerOpenException, RequestDeadlineExceededException, RetryManager
from relay.schemas import (
    Citation,
    PipelineStageTimings,
    ProviderResponse,
    RelayRequest,
    RelayResponse,
    RetrievedPassage,
)


class RelayPipeline:
    def __init__(
        self,
        chunks: List[Chunk],
        provider: Optional[ModelProvider] = None,
        source_name: str = "stand-in",
        paper_pages_text: Optional[Dict[str, Dict[int, str]]] = None,
        standin_provider: Optional[ModelProvider] = None,
        groq_provider: Optional[ModelProvider] = None,
    ):
        self.chunks = chunks
        self.paper_pages_text = paper_pages_text or {}

        # Lazy import default provider implementations to prevent circular imports
        from relay.standin_provider import StandinProvider
        from relay.groq_provider import GroqProvider

        self.standin_provider = standin_provider or StandinProvider()
        self.groq_provider = groq_provider or GroqProvider()

        # If a single custom provider was passed directly (e.g. in test fixtures), prioritize it
        self.default_provider = provider or self.standin_provider
        self.source_name = source_name

        self.retriever = DocumentRetriever(chunks)
        self.intent_classifier = IntentClassifier()
        self.evidence_evaluator = EvidenceEvaluator(
            min_threshold=settings.evidence_threshold,
            high_confidence_threshold=settings.high_confidence_threshold,
        )
        self.generator = AnswerGenerator(self.default_provider)
        self.citation_verifier = CitationVerifier(paper_pages_text)

        # Resilience components
        self.queue_manager = QueueManager(
            max_concurrent=settings.max_concurrent_requests,
            max_queue_size=settings.queue_max_size,
        )
        self.rate_limiter = ClientRateLimiter(
            rpm_limit=settings.client_rpm_limit,
            tpm_limit=settings.client_tpm_limit,
            rpd_limit=settings.client_rpd_limit,
            tpd_limit=settings.client_tpd_limit,
        )
        self.retry_manager = RetryManager(
            max_retries=settings.max_retries,
            base_delay_s=settings.retry_base_delay,
            max_delay_s=settings.retry_max_delay,
        )

    def get_provider(
        self, provider_choice: Optional[str] = None
    ) -> Tuple[ModelProvider, str, bool]:
        """Resolves the model provider for a specific request.
        Returns: (provider_instance, normalized_provider_name, is_simulated).
        Validates choice: only 'standin' and 'groq' are permitted.
        """
        if not provider_choice:
            prov = self.default_provider
            prov_name = getattr(prov, "provider_name", "standin")
            is_sim = getattr(prov, "is_simulated", True)
            return prov, prov_name, is_sim

        choice = str(provider_choice).lower().strip()
        if choice in {"standin", "stand-in", "simulator"}:
            return self.standin_provider, "standin", True
        elif choice in {"groq", "real"}:
            return self.groq_provider, "groq", False
        else:
            raise ValueError(
                f"Unsupported model provider '{provider_choice}'. Allowed providers are 'standin' and 'groq'."
            )

    def validate_request(self, req: RelayRequest) -> Tuple[bool, Optional[str]]:
        """Stage 1: Request Validation."""
        q = req.question.strip() if req.question else ""
        if not q:
            return False, "Question cannot be empty or purely whitespace."
        if len(q) > 4000:
            return False, "Question exceeds maximum allowed character length (4,000 chars)."
        # Filter suspicious non-printable payloads
        if any(ord(char) < 32 and char not in "\n\r\t" for char in q):
            return False, "Question contains invalid or unprintable characters."

        # Validate provider if present on request
        if getattr(req, "provider", None):
            norm_prov = str(req.provider).lower().strip()
            if norm_prov not in {"standin", "stand-in", "simulator", "groq", "real"}:
                return False, f"Unsupported model provider '{req.provider}'. Allowed providers are 'standin' and 'groq'."

        return True, None

    async def execute(self, req: RelayRequest) -> RelayResponse:
        """Executes the full Relay pipeline non-streaming."""
        t_start = time.perf_counter()
        req_id = req.request_id or f"req_{uuid.uuid4().hex[:8]}"
        deadline_s = req.timeout_seconds or settings.request_deadline
        deadline_ts = t_start + deadline_s

        timings = PipelineStageTimings()

        # Resolve provider for this request
        try:
            active_provider, provider_name, is_simulated = self.get_provider(req.provider)
        except ValueError as e:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            timings.total_ms = total_ms
            return RelayResponse(
                request_id=req_id,
                question=req.question,
                answer="",
                is_refusal=True,
                refusal_reason=str(e),
                status="error",
                provider="standin",
                model="",
                is_simulated=True,
                source="stand-in",
                total_ms=round(total_ms, 2),
                latency_ms=round(total_ms, 2),
                stage_timings=timings,
            )

        target_model = getattr(active_provider, "model_name", "standin-calibrated-lpu")
        source_val = "stand-in" if is_simulated else "real"

        # -------------------------------------------------------------
        # STAGE 1: Request Validation
        # -------------------------------------------------------------
        t_val_start = time.perf_counter()
        is_valid, val_err = self.validate_request(req)
        timings.validation_ms = (time.perf_counter() - t_val_start) * 1000.0

        if not is_valid:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            timings.total_ms = total_ms
            return RelayResponse(
                request_id=req_id,
                question=req.question,
                answer="",
                is_refusal=True,
                refusal_reason=val_err,
                status="error",
                provider=provider_name,  # type: ignore
                model=target_model,
                is_simulated=is_simulated,
                source=source_val,
                total_ms=round(total_ms, 2),
                latency_ms=round(total_ms, 2),
                stage_timings=timings,
            )

        # -------------------------------------------------------------
        # STAGE 2: Intent Analysis
        # -------------------------------------------------------------
        t_intent_start = time.perf_counter()
        intent = self.intent_classifier.classify(req.question)
        timings.intent_ms = (time.perf_counter() - t_intent_start) * 1000.0

        # Fast exit for out-of-scope / unanswerable intent
        if intent.category.value == "unanswerable":
            total_ms = (time.perf_counter() - t_start) * 1000.0
            timings.total_ms = total_ms
            return RelayResponse(
                request_id=req_id,
                question=req.question,
                answer=REFUSAL_MESSAGE,
                is_refusal=True,
                refusal_reason=f"Intent classified as unanswerable ({intent.reasoning})",
                status="ok",
                provider=provider_name,  # type: ignore
                model=target_model,
                is_simulated=is_simulated,
                source=source_val,
                first_token_ms=round(total_ms, 2),
                first_token_latency_ms=round(total_ms, 2) if is_simulated else None,
                total_ms=round(total_ms, 2),
                latency_ms=round(total_ms, 2),
                stage_timings=timings,
            )

        # -------------------------------------------------------------
        # STAGE 3: Retrieval Tool
        # -------------------------------------------------------------
        t_ret_start = time.perf_counter()
        reduced_mode = self.rate_limiter.is_near_budget_exhaustion(
            settings.budget_exhaustion_safety_margin
        )
        passages = self.retriever.retrieve(
            query=req.question,
            top_k=settings.top_k_passages,
            enable_reranking=settings.enable_reranking,
            reduced_budget_mode=reduced_mode,
        )
        timings.retrieval_ms = (time.perf_counter() - t_ret_start) * 1000.0

        # -------------------------------------------------------------
        # STAGE 4: Evidence Evaluation & Query Refinement Loop
        # -------------------------------------------------------------
        t_ev_start = time.perf_counter()
        ev_eval = self.evidence_evaluator.evaluate(req.question, passages)

        # Conditional loop: Refine query and retrieve again if marginal
        if ev_eval.needs_refinement and ev_eval.refined_query and not reduced_mode:
            t_refine_start = time.perf_counter()
            secondary_passages = self.retriever.retrieve(
                query=ev_eval.refined_query,
                top_k=settings.top_k_passages,
                enable_reranking=settings.enable_reranking,
            )
            passages = self.evidence_evaluator.deduplicate_passages(
                passages + secondary_passages
            )
            passages.sort(key=lambda p: p.final_score, reverse=True)
            ev_eval = self.evidence_evaluator.evaluate(
                req.question, passages, has_already_refined=True
            )
            timings.refinement_ms = (time.perf_counter() - t_refine_start) * 1000.0

        timings.evidence_eval_ms = (time.perf_counter() - t_ev_start) * 1000.0

        if not ev_eval.is_sufficient:
            # Refuse to guess when documents do not contain evidence
            total_ms = (time.perf_counter() - t_start) * 1000.0
            timings.total_ms = total_ms
            return RelayResponse(
                request_id=req_id,
                question=req.question,
                answer=REFUSAL_MESSAGE,
                is_refusal=True,
                refusal_reason=ev_eval.reason,
                status="ok",
                provider=provider_name,  # type: ignore
                model=target_model,
                is_simulated=is_simulated,
                source=source_val,
                first_token_ms=round(total_ms, 2),
                first_token_latency_ms=round(total_ms, 2) if is_simulated else None,
                total_ms=round(total_ms, 2),
                latency_ms=round(total_ms, 2),
                stage_timings=timings,
            )

        # -------------------------------------------------------------
        # STAGE 5: Scheduling, Concurrency & Answer Generation
        # -------------------------------------------------------------
        try:
            time_left = max(0.1, deadline_ts - time.perf_counter())
            wait_ms = await self.queue_manager.acquire_slot(timeout_seconds=time_left)
            timings.queue_wait_ms = wait_ms
        except QueueFullException:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            timings.total_ms = total_ms
            return RelayResponse(
                request_id=req_id,
                question=req.question,
                answer="",
                status="rate_limited",
                provider=provider_name,  # type: ignore
                model=target_model,
                is_simulated=is_simulated,
                source=source_val,
                total_ms=round(total_ms, 2),
                latency_ms=round(total_ms, 2),
                refusal_reason="Server queue saturated.",
                stage_timings=timings,
            )
        except asyncio.TimeoutError:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            timings.total_ms = total_ms
            return RelayResponse(
                request_id=req_id,
                question=req.question,
                answer="",
                status="timeout",
                provider=provider_name,  # type: ignore
                model=target_model,
                is_simulated=is_simulated,
                source=source_val,
                total_ms=round(total_ms, 2),
                latency_ms=round(total_ms, 2),
                refusal_reason="Client deadline expired while waiting in queue.",
                stage_timings=timings,
            )

        # Check circuit breaker
        if not self.retry_manager.circuit_breaker.can_attempt():
            self.queue_manager.release_slot()
            total_ms = (time.perf_counter() - t_start) * 1000.0
            timings.total_ms = total_ms
            return RelayResponse(
                request_id=req_id,
                question=req.question,
                answer="",
                status="error",
                provider=provider_name,  # type: ignore
                model=target_model,
                is_simulated=is_simulated,
                source=source_val,
                total_ms=round(total_ms, 2),
                latency_ms=round(total_ms, 2),
                refusal_reason="Circuit breaker OPEN due to repeated provider failures.",
                stage_timings=timings,
            )

        # Execute generation with bounded retry policy
        provider_resp: Optional[ProviderResponse] = None
        retries_count = 0
        total_retry_wait_ms = 0.0

        try:
            for attempt in range(1, self.retry_manager.max_retries + 1):
                if time.perf_counter() > deadline_ts:
                    raise RequestDeadlineExceededException("Overall deadline expired before generation.")

                t_gen_start = time.perf_counter()
                provider_resp = await self.generator.generate_answer(
                    req.question, passages, provider=active_provider
                )
                timings.generation_ms += (time.perf_counter() - t_gen_start) * 1000.0

                if not provider_resp.is_error:
                    self.retry_manager.circuit_breaker.record_success()
                    self.rate_limiter.record_usage(provider_resp.tokens_in + provider_resp.tokens_out)
                    break

                # Provider returned error
                self.retry_manager.circuit_breaker.record_failure()
                is_retryable, retry_reason = self.retry_manager.is_retryable(provider_resp)

                if not is_retryable or attempt == self.retry_manager.max_retries:
                    break

                retries_count += 1
                backoff_delay = self.retry_manager.compute_backoff(attempt, provider_resp)
                total_retry_wait_ms += (backoff_delay * 1000.0)

                if time.perf_counter() + backoff_delay > deadline_ts:
                    raise RequestDeadlineExceededException("Deadline exceeded during retry backoff.")

                await asyncio.sleep(backoff_delay)

        except RequestDeadlineExceededException:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            timings.total_ms = total_ms
            timings.retry_wait_ms = total_retry_wait_ms
            return RelayResponse(
                request_id=req_id,
                question=req.question,
                answer="",
                status="timeout",
                provider=provider_name,  # type: ignore
                model=target_model,
                is_simulated=is_simulated,
                source=source_val,
                retries=retries_count,
                total_ms=round(total_ms, 2),
                latency_ms=round(total_ms, 2),
                refusal_reason="Request deadline exceeded.",
                stage_timings=timings,
            )
        finally:
            self.queue_manager.release_slot()

        timings.retry_wait_ms = total_retry_wait_ms

        # Check final provider response
        if not provider_resp or provider_resp.is_error:
            status_val = "rate_limited" if (provider_resp and provider_resp.status_code == 429) else "error"
            total_ms = (time.perf_counter() - t_start) * 1000.0
            timings.total_ms = total_ms
            return RelayResponse(
                request_id=req_id,
                question=req.question,
                answer="",
                status=status_val,
                provider=provider_name,  # type: ignore
                model=target_model,
                is_simulated=is_simulated,
                source=source_val,
                retries=retries_count,
                total_ms=round(total_ms, 2),
                latency_ms=round(total_ms, 2),
                refusal_reason=provider_resp.error_message if provider_resp else "Unknown provider failure",
                stage_timings=timings,
            )

        # -------------------------------------------------------------
        # STAGE 6: Citation Verification
        # -------------------------------------------------------------
        t_ver_start = time.perf_counter()
        verification = self.citation_verifier.verify(
            answer_text=provider_resp.text,
            citations=[],
            retrieved_passages=passages,
        )
        timings.verification_ms = (time.perf_counter() - t_ver_start) * 1000.0

        final_answer = format_math_unicode(provider_resp.text)
        is_refusal = False
        refusal_reason = None

        if not verification.is_valid:
            final_answer = REFUSAL_MESSAGE
            is_refusal = True
            refusal_reason = f"Citation verification failed: {verification.rejection_reason}"

        total_ms = (time.perf_counter() - t_start) * 1000.0
        timings.total_ms = total_ms

        first_tok_ms = provider_resp.first_token_ms if provider_resp.first_token_ms > 0 else total_ms
        first_token_latency = first_tok_ms if (is_simulated or req.stream) else None

        return RelayResponse(
            request_id=req_id,
            question=req.question,
            answer=final_answer,
            citations=verification.verified_citations,
            is_refusal=is_refusal,
            refusal_reason=refusal_reason,
            status="ok",
            provider=provider_name,  # type: ignore
            model=target_model,
            is_simulated=is_simulated,
            source=source_val,
            input_tokens=provider_resp.tokens_in,
            output_tokens=provider_resp.tokens_out,
            total_tokens=provider_resp.tokens_in + provider_resp.tokens_out,
            tokens_in=provider_resp.tokens_in,
            tokens_out=provider_resp.tokens_out,
            first_token_ms=round(first_tok_ms, 2),
            first_token_latency_ms=round(first_token_latency, 2) if first_token_latency is not None else None,
            total_ms=round(total_ms, 2),
            latency_ms=round(total_ms, 2),
            retries=retries_count,
            stage_timings=timings,
        )

    async def execute_stream(
        self, req: RelayRequest
    ) -> AsyncGenerator[Tuple[str, Optional[RelayResponse]], None]:
        """Executes the pipeline with true token streaming from the selected provider."""
        t_start = time.perf_counter()
        req_id = req.request_id or f"req_{uuid.uuid4().hex[:8]}"
        deadline_s = req.timeout_seconds or settings.request_deadline
        deadline_ts = t_start + deadline_s
        timings = PipelineStageTimings()

        # Resolve provider
        try:
            active_provider, provider_name, is_simulated = self.get_provider(req.provider)
        except ValueError as e:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            timings.total_ms = total_ms
            err_resp = RelayResponse(
                request_id=req_id,
                question=req.question,
                answer="",
                is_refusal=True,
                refusal_reason=str(e),
                status="error",
                provider="standin",
                model="",
                is_simulated=True,
                source="stand-in",
                total_ms=round(total_ms, 2),
                latency_ms=round(total_ms, 2),
                stage_timings=timings,
            )
            yield ("", err_resp)
            return

        target_model = getattr(active_provider, "model_name", "standin-calibrated-lpu")
        source_val = "stand-in" if is_simulated else "real"

        # Validation
        t_val_start = time.perf_counter()
        is_valid, val_err = self.validate_request(req)
        timings.validation_ms = (time.perf_counter() - t_val_start) * 1000.0

        if not is_valid:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            timings.total_ms = total_ms
            err_resp = RelayResponse(
                request_id=req_id,
                question=req.question,
                answer="",
                is_refusal=True,
                refusal_reason=val_err,
                status="error",
                provider=provider_name,  # type: ignore
                model=target_model,
                is_simulated=is_simulated,
                source=source_val,
                total_ms=round(total_ms, 2),
                latency_ms=round(total_ms, 2),
                stage_timings=timings,
            )
            yield ("", err_resp)
            return

        # Intent
        t_intent_start = time.perf_counter()
        intent = self.intent_classifier.classify(req.question)
        timings.intent_ms = (time.perf_counter() - t_intent_start) * 1000.0

        if intent.category.value == "unanswerable":
            total_ms = (time.perf_counter() - t_start) * 1000.0
            timings.total_ms = total_ms
            yield (REFUSAL_MESSAGE, None)
            ref_resp = RelayResponse(
                request_id=req_id,
                question=req.question,
                answer=REFUSAL_MESSAGE,
                is_refusal=True,
                refusal_reason=f"Intent classified as unanswerable ({intent.reasoning})",
                status="ok",
                provider=provider_name,  # type: ignore
                model=target_model,
                is_simulated=is_simulated,
                source=source_val,
                first_token_ms=round(total_ms, 2),
                first_token_latency_ms=round(total_ms, 2) if is_simulated else None,
                total_ms=round(total_ms, 2),
                latency_ms=round(total_ms, 2),
                stage_timings=timings,
            )
            yield ("", ref_resp)
            return

        # Retrieval
        t_ret_start = time.perf_counter()
        passages = self.retriever.retrieve(
            query=req.question,
            top_k=settings.top_k_passages,
            enable_reranking=settings.enable_reranking,
        )
        timings.retrieval_ms = (time.perf_counter() - t_ret_start) * 1000.0

        # Evidence evaluation
        t_ev_start = time.perf_counter()
        ev_eval = self.evidence_evaluator.evaluate(req.question, passages)
        timings.evidence_eval_ms = (time.perf_counter() - t_ev_start) * 1000.0

        if not ev_eval.is_sufficient:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            timings.total_ms = total_ms
            yield (REFUSAL_MESSAGE, None)
            ref_resp = RelayResponse(
                request_id=req_id,
                question=req.question,
                answer=REFUSAL_MESSAGE,
                is_refusal=True,
                refusal_reason=ev_eval.reason,
                status="ok",
                provider=provider_name,  # type: ignore
                model=target_model,
                is_simulated=is_simulated,
                source=source_val,
                first_token_ms=round(total_ms, 2),
                first_token_latency_ms=round(total_ms, 2) if is_simulated else None,
                total_ms=round(total_ms, 2),
                latency_ms=round(total_ms, 2),
                stage_timings=timings,
            )
            yield ("", ref_resp)
            return

        # Acquire queue slot
        try:
            wait_ms = await self.queue_manager.acquire_slot(timeout_seconds=deadline_s)
            timings.queue_wait_ms = wait_ms
        except Exception as q_err:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            timings.total_ms = total_ms
            yield ("", RelayResponse(
                request_id=req_id,
                question=req.question,
                answer="",
                status="rate_limited",
                provider=provider_name,  # type: ignore
                model=target_model,
                is_simulated=is_simulated,
                source=source_val,
                total_ms=round(total_ms, 2),
                latency_ms=round(total_ms, 2),
                refusal_reason=str(q_err),
                stage_timings=timings,
            ))
            return

        accumulated_text: List[str] = []
        final_provider_resp: Optional[ProviderResponse] = None

        try:
            t_gen_start = time.perf_counter()
            async for delta, resp_final in self.generator.generate_answer_stream(
                req.question, passages, provider=active_provider
            ):
                if delta:
                    accumulated_text.append(delta)
                    yield (delta, None)
                if resp_final:
                    final_provider_resp = resp_final
            timings.generation_ms = (time.perf_counter() - t_gen_start) * 1000.0

        finally:
            self.queue_manager.release_slot()

        if not final_provider_resp or final_provider_resp.is_error:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            timings.total_ms = total_ms
            yield ("", RelayResponse(
                request_id=req_id,
                question=req.question,
                answer="".join(accumulated_text),
                status="error" if (not final_provider_resp or final_provider_resp.status_code != 429) else "rate_limited",
                provider=provider_name,  # type: ignore
                model=target_model,
                is_simulated=is_simulated,
                source=source_val,
                total_ms=round(total_ms, 2),
                latency_ms=round(total_ms, 2),
                refusal_reason=final_provider_resp.error_message if final_provider_resp else "Stream generation failed",
                stage_timings=timings,
            ))
            return

        full_answer = "".join(accumulated_text) or final_provider_resp.text

        # Citation verification
        t_ver_start = time.perf_counter()
        verification = self.citation_verifier.verify(
            answer_text=full_answer,
            citations=[],
            retrieved_passages=passages,
        )
        timings.verification_ms = (time.perf_counter() - t_ver_start) * 1000.0

        total_ms = (time.perf_counter() - t_start) * 1000.0
        timings.total_ms = total_ms

        final_relay_response = RelayResponse(
            request_id=req_id,
            question=req.question,
            answer=format_math_unicode(full_answer),
            citations=verification.verified_citations,
            is_refusal=not verification.is_valid,
            refusal_reason=verification.rejection_reason if not verification.is_valid else None,
            status="ok",
            provider=provider_name,  # type: ignore
            model=target_model,
            is_simulated=is_simulated,
            source=source_val,
            input_tokens=final_provider_resp.tokens_in,
            output_tokens=final_provider_resp.tokens_out,
            total_tokens=final_provider_resp.tokens_in + final_provider_resp.tokens_out,
            tokens_in=final_provider_resp.tokens_in,
            tokens_out=final_provider_resp.tokens_out,
            first_token_ms=round(final_provider_resp.first_token_ms, 2),
            first_token_latency_ms=round(final_provider_resp.first_token_ms, 2),
            total_ms=round(total_ms, 2),
            latency_ms=round(total_ms, 2),
            stage_timings=timings,
        )

        yield ("", final_relay_response)
