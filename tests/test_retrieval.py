"""Unit tests for BM25 retrieval and deterministic reranking."""

import pytest
from relay.chunking import Chunk
from relay.retrieval import DocumentRetriever


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="c1",
            paper_id="1706.03762v7",
            paper_title="Attention Is All You Need",
            arxiv_id="1706.03762",
            version="v7",
            page_number=3,
            section="3.1 Encoder and Decoder Stacks",
            text="The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers: multi-head self-attention mechanism and feed forward network.",
            approx_token_count=35,
        ),
        Chunk(
            chunk_id="c2",
            paper_id="2410.00037v1",
            paper_title="Moshi: a speech-text foundation model for real-time dialogue",
            arxiv_id="2410.00037",
            version="v1",
            page_number=2,
            section="2 Architecture",
            text="Moshi uses the Mimi neural audio codec operating at 12.5 Hz with an 8-codebook Residual Vector Quantizer (RVQ).",
            approx_token_count=30,
        ),
        Chunk(
            chunk_id="c3",
            paper_id="2507.07955v2",
            paper_title="Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
            arxiv_id="2507.07955",
            version="v2",
            page_number=4,
            section="3 Dynamic Chunking",
            text="Dynamic chunking dynamically segments continuous sequence representations into hierarchical chunks using boundary prediction.",
            approx_token_count=25,
        ),
    ]


def test_bm25_retrieval_relevance(sample_chunks):
    retriever = DocumentRetriever(sample_chunks)
    res = retriever.retrieve("What is the frame rate of Mimi?", top_k=2)

    assert len(res) > 0
    top = res[0]
    assert top.paper_id == "2410.00037v1"
    assert "Mimi" in top.text
    assert top.page_number == 2


def test_reranking_phrase_boost(sample_chunks):
    retriever = DocumentRetriever(sample_chunks)
    res_rerank = retriever.retrieve("encoder is composed of a stack of N = 6", top_k=2, enable_reranking=True)
    res_no_rerank = retriever.retrieve("encoder is composed of a stack of N = 6", top_k=2, enable_reranking=False)

    assert res_rerank[0].paper_id == "1706.03762v7"
    assert res_rerank[0].final_score >= res_no_rerank[0].final_score


def test_reduced_budget_retrieval(sample_chunks):
    retriever = DocumentRetriever(sample_chunks)
    res = retriever.retrieve("What is self-attention and chunking?", top_k=5, reduced_budget_mode=True)
    # Under reduced budget mode, top_k is capped at 2
    assert len(res) <= 2

