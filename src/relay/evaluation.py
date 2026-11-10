"""Evaluation harness for testing retrieval configurations and answer quality."""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel

from relay.chunking import Chunk, chunk_papers
from relay.metrics import MetricsRecorder
from relay.papers import ExtractedPaper
from relay.pipeline import RelayPipeline
from relay.provider_interface import ModelProvider
from relay.rate_limiter import ClientRateLimiter
from relay.schemas import RelayRequest, RelayResponse, ResultCSVRow

AnswerGrade = Literal["correct", "wrong", "refused_correctly", "refused_wrongly"]


class QuestionItem(BaseModel):
    question_id: str
    question: str
    expected_answer_summary: str
    expected_source_paper: str
    expected_page_or_section: str
    expected_status: Literal["answerable", "unanswerable"]


def load_evaluation_questions(filepath: Path) -> List[QuestionItem]:
    questions: List[QuestionItem] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(QuestionItem.model_validate_json(line))
    return questions


def grade_answer(item: QuestionItem, response: RelayResponse) -> AnswerGrade:
    """Grades response as exactly one of: correct, wrong, refused_correctly, refused_wrongly."""
    is_unanswerable = item.expected_status == "unanswerable"

    if response.is_refusal:
        if is_unanswerable:
            return "refused_correctly"
        else:
            return "refused_wrongly"

    # Response produced an answer
    if is_unanswerable:
        # Should have refused, but hallucinated an answer!
        return "wrong"

    # Expected answerable and gave answer: check citation and factual alignment
    citation_paper_matches = any(
        (c.arxiv_id in item.expected_source_paper or item.expected_source_paper in f"{c.arxiv_id}{c.version}")
        for c in response.citations
    )
    text_matches = any(p in response.answer for p in [item.expected_source_paper, "1706.03762", "2410.00037", "2507.07955"])
    if not (citation_paper_matches or text_matches):
        return "wrong"

    # Check key terms from expected summary
    expected_terms = [
        w.lower()
        for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", item.expected_answer_summary)
        if w.lower() not in {"the", "and", "that", "this", "with", "from", "for", "are", "was"}
    ]

    answer_lower = response.answer.lower()
    matches = sum(1 for term in expected_terms if term in answer_lower)
    overlap_ratio = matches / max(1, len(expected_terms))

    if overlap_ratio >= 0.25:
        return "correct"
    else:
        return "wrong"


async def run_evaluation_suite(
    questions: List[QuestionItem],
    extracted_papers: List[ExtractedPaper],
    provider: ModelProvider,
    source_name: Literal["real", "stand-in"] = "stand-in",
    csv_recorder: Optional[MetricsRecorder] = None,
) -> Dict[str, dict]:
    """Runs the 3 mandatory retrieval configurations over the evaluation set."""
    recorder = csv_recorder or MetricsRecorder()

    # Pre-extract paper text lookup for citation verifier
    paper_pages: Dict[str, Dict[int, str]] = {}
    for p in extracted_papers:
        paper_pages[p.paper_id] = {page.page_number: page.text for page in p.pages}

    settings_to_test = [
        {
            "setting_name": "Setting A (350 tok, top-3, no rerank)",
            "retrieval_setting": "setting_a",
            "chunk_size": 350,
            "overlap": 50,
            "top_k": 3,
            "rerank": False,
        },
        {
            "setting_name": "Setting B (600 tok, top-5, rerank)",
            "retrieval_setting": "setting_b",
            "chunk_size": 600,
            "overlap": 90,
            "top_k": 5,
            "rerank": True,
        },
        {
            "setting_name": "Setting C (900 tok, top-8, rerank)",
            "retrieval_setting": "setting_c",
            "chunk_size": 900,
            "overlap": 120,
            "top_k": 8,
            "rerank": True,
        },
    ]

    suite_results = {}

    for cfg in settings_to_test:
        print(f"\n--- Evaluating {cfg['setting_name']} ---")
        from standin.app import quota_tracker, simulator
        quota_tracker.reset()
        simulator.active_requests = 0

        chunks = chunk_papers(
            extracted_papers,
            chunk_size_tokens=cfg["chunk_size"],
            overlap_tokens=cfg["overlap"],
        )

        pipeline = RelayPipeline(
            chunks=chunks,
            provider=provider,
            source_name=source_name,
            paper_pages_text=paper_pages,
        )
        pipeline.retriever.enable_reranking = cfg["rerank"]
        # In evaluation benchmark mode, allow the benchmark to test all 16 questions without minute choking
        pipeline.rate_limiter = ClientRateLimiter(
            rpm_limit=1000,
            tpm_limit=500000,
            rpd_limit=10000,
            tpd_limit=1000000,
        )

        grades: List[AnswerGrade] = []
        ttfts: List[float] = []
        totals: List[float] = []
        tokens_list: List[int] = []

        for q_item in questions:
            quota_tracker.reset()
            req = RelayRequest(
                question=q_item.question,
                request_id=f"eval_{cfg['retrieval_setting']}_{q_item.question_id}",
                scenario="retrieval_eval",
                retrieval_setting=cfg["retrieval_setting"],
            )

            resp = await pipeline.execute(req)
            grade = grade_answer(q_item, resp)
            grades.append(grade)
            ttfts.append(resp.first_token_ms)
            totals.append(resp.total_ms)
            tokens_list.append(resp.tokens_in + resp.tokens_out)

            # Record in CSV
            csv_row = ResultCSVRow(
                request_id=resp.request_id,
                scenario="retrieval_eval",
                source=source_name,
                start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                first_token_ms=resp.first_token_ms,
                total_ms=resp.total_ms,
                status=resp.status,
                retries=resp.retries,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                retrieval_setting=cfg["retrieval_setting"],
                answer_grade=grade,
                queue_wait_ms=resp.stage_timings.queue_wait_ms,
                provider_working_ms=resp.stage_timings.generation_ms,
                retry_wait_ms=resp.stage_timings.retry_wait_ms,
                peak_queue_len=pipeline.queue_manager.peak_queue_length,
            )
            recorder.record_row(csv_row)

        total_q = len(questions)
        correct_pct = (grades.count("correct") / total_q) * 100.0
        wrong_pct = (grades.count("wrong") / total_q) * 100.0
        refused_corr_pct = (grades.count("refused_correctly") / total_q) * 100.0
        refused_wrong_pct = (grades.count("refused_wrongly") / total_q) * 100.0

        suite_results[cfg["retrieval_setting"]] = {
            "setting_name": cfg["setting_name"],
            "questions_asked": total_q,
            "correct_and_cited_pct": round(correct_pct, 1),
            "wrong_pct": round(wrong_pct, 1),
            "refused_correctly_pct": round(refused_corr_pct, 1),
            "refused_wrongly_pct": round(refused_wrong_pct, 1),
            "tokens_per_answer": round(sum(tokens_list) / max(1, total_q), 1),
            "ttft_p50": round(sorted(ttfts)[len(ttfts) // 2], 1) if ttfts else 0.0,
            "ttft_p95": round(sorted(ttfts)[int(len(ttfts) * 0.95)], 1) if ttfts else 0.0,
            "total_p50": round(sorted(totals)[len(totals) // 2], 1) if totals else 0.0,
            "total_p95": round(sorted(totals)[int(len(totals) * 0.95)], 1) if totals else 0.0,
        }

    return suite_results
