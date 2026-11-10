"""Unit tests for Intent Classification and Evidence Evaluation."""

import pytest
from relay.evidence import EvidenceEvaluator
from relay.intent import IntentClassifier
from relay.schemas import IntentCategory, RetrievedPassage


def test_intent_classification():
    clf = IntentClassifier()

    r1 = clf.classify("How many layers are in the Transformer encoder stack?")
    assert r1.category in {IntentCategory.ARCHITECTURAL_DETAILS, IntentCategory.FACTUAL_LOOKUP}

    r2 = clf.classify("Compare the attention mechanism in Vaswani to Dynamic Chunking.")
    assert r2.category == IntentCategory.COMPARATIVE

    r3 = clf.classify("What is the current price of Bitcoin and stock market futures?")
    assert r3.category == IntentCategory.UNANSWERABLE


def test_evidence_evaluation_sufficiency():
    evaluator = EvidenceEvaluator(min_threshold=0.35)

    good_passage = RetrievedPassage(
        passage_id="p1",
        paper_id="1706.03762v7",
        paper_title="Attention Is All You Need",
        arxiv_id="1706.03762",
        version="v7",
        page_number=3,
        text="The encoder consists of N = 6 identical layers, each with multi-head attention and feed-forward sub-layers.",
        final_score=0.85,
    )

    ev = evaluator.evaluate("How many layers in encoder?", [good_passage])
    assert ev.is_sufficient is True
    assert ev.needs_refinement is False


def test_evidence_evaluation_insufficient():
    evaluator = EvidenceEvaluator(min_threshold=0.35)

    weak_passage = RetrievedPassage(
        passage_id="p2",
        paper_id="1706.03762v7",
        paper_title="Attention Is All You Need",
        arxiv_id="1706.03762",
        version="v7",
        page_number=1,
        text="We thank Kevin Murphy, Lukasz Kaiser and Illia Polosukhin for fruitful discussions.",
        final_score=0.15,
    )

    ev = evaluator.evaluate("What is the quantum state vector of Mimi?", [weak_passage])
    assert ev.is_sufficient is False
    assert ev.needs_refinement is True  # triggers controlled query refinement

