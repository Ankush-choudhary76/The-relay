"""Simulation logic for generating realistic latency, load slowdown, and responses."""

import asyncio
import json
import random
import re
import time
from typing import AsyncGenerator, Dict, List, Optional, Tuple
from standin.scenarios import StandinConfig

REFUSAL_MESSAGE = "The provided research papers do not contain sufficient information to answer this question reliably."


# ── Structured synthetic answers with Unicode math ───────────────────────────

ANSWER_ATTENTION = (
    "**Scaled Dot-Product Attention**\n\n"
    "Scaled Dot-Product Attention is the core attention mechanism used in the "
    "Transformer architecture. It operates on three input matrices:\n\n"
    "- **Q** — Query matrix\n"
    "- **K** — Key matrix\n"
    "- **V** — Value matrix\n\n"
    "The attention function is computed as:\n\n"
    "> **Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V**\n\n"
    "**Explanation of each component:**\n\n"
    "- **QKᵀ** — Computes dot-product similarity scores between all query–key pairs "
    "via matrix multiplication of Q with the transpose of K.\n"
    "- **√dₖ** — Scaling factor equal to the square root of the key dimension dₖ. "
    "This prevents dot-product values from growing too large, which would push the "
    "softmax function into regions with extremely small gradients.\n"
    "- **softmax** — Normalises the scaled scores into a probability distribution "
    "(attention weights that sum to 1).\n"
    "- **V** — The value vectors are weighted by these attention probabilities "
    "to produce the final output.\n\n"
    "The authors note that the two most commonly used attention functions are additive "
    "attention and dot-product (multiplicative) attention; scaled dot-product attention "
    "is identical to dot-product attention except for the 1/√dₖ scaling factor "
    "[Citation: Attention Is All You Need, arXiv:1706.03762v7, Page 4, Section 3.2.1]."
)

ANSWER_ADAM_WARMUP = (
    "**Training Configuration — Adam Optimiser & Learning Rate**\n\n"
    "The Transformer was trained using the Adam optimiser with the following "
    "hyperparameters:\n\n"
    "- β₁ = 0.9\n"
    "- β₂ = 0.98\n"
    "- ε = 10⁻⁹\n\n"
    "The learning rate was varied over the course of training according to:\n\n"
    "> **lr = d_model⁻⁰·⁵ · min(step⁻⁰·⁵, step · warmup_steps⁻¹·⁵)**\n\n"
    "This schedule linearly increases the learning rate for the first "
    "**warmup_steps = 4,000** training steps, then decreases it proportionally "
    "to the inverse square root of the step number "
    "[Citation: Attention Is All You Need, arXiv:1706.03762v7, Page 7, Section 5.3]."
)

ANSWER_MIMI_FRAMERATE = (
    "**Mimi Neural Audio Codec — Frame Rate**\n\n"
    "The Mimi neural audio codec operates at a frame rate of **12.5 Hz** with an "
    "8-codebook Residual Vector Quantiser (RVQ), processing audio originally sampled "
    "at 24 kHz.\n\n"
    "Key specifications:\n\n"
    "- **Input sample rate:** 24 kHz\n"
    "- **Codec frame rate:** 12.5 Hz (one frame every 80 ms)\n"
    "- **Codebooks:** 8 (RVQ structure)\n"
    "- **Bitrate:** 1.1 kbps per codebook\n\n"
    "The low 12.5 Hz rate is critical for keeping the sequence length manageable "
    "for the Transformer-based language model that operates on top of the codec "
    "[Citation: Moshi: a speech-text foundation model for real-time dialogue, "
    "arXiv:2410.00037v1, Page 2, Section 2]."
)

ANSWER_MIMI_LATENCY = (
    "**Mimi Codec — Streaming Latency**\n\n"
    "The Mimi codec achieves an overall theoretical streaming latency of **80 ms**, "
    "corresponding to a single frame at its 12.5 Hz frame rate.\n\n"
    "This low latency enables real-time dialogue, as each audio frame can be "
    "encoded, processed by the language model, and decoded within the 80 ms window "
    "[Citation: Moshi: a speech-text foundation model for real-time dialogue, "
    "arXiv:2410.00037v1, Page 2, Section 2]."
)

ANSWER_FULL_DUPLEX = (
    "**Full-Duplex Spoken Dialogue in Moshi**\n\n"
    "Moshi models full-duplex spoken dialogue by processing **two parallel streams** "
    "simultaneously:\n\n"
    "1. **User stream** — Encodes the user's speech input\n"
    "2. **System stream** — Generates both speech and text tokens for the system's "
    "response\n\n"
    "This dual-stream architecture eliminates artificial turn-taking delays that "
    "are typical in pipeline-based voice assistants, allowing natural overlapping "
    "conversation "
    "[Citation: Moshi: a speech-text foundation model for real-time dialogue, "
    "arXiv:2410.00037v1, Page 2, Section 1]."
)

ANSWER_HELIUM = (
    "**Helium — Text Language Model Backbone**\n\n"
    "Moshi is built upon **Helium**, a 7-billion parameter language model "
    "trained on text data from scratch.\n\n"
    "Helium serves as the backbone that provides the language understanding "
    "and generation capabilities. The audio codec tokens from Mimi are "
    "interleaved with text tokens and fed into Helium for joint modelling "
    "[Citation: Moshi: a speech-text foundation model for real-time dialogue, "
    "arXiv:2410.00037v1, Page 2, Section 2]."
)

ANSWER_DYNAMIC_CHUNKING = (
    "**Dynamic Chunking — Hierarchical Sequence Modelling**\n\n"
    "Dynamic Chunking groups sequence representations into **variable-length "
    "chunks** determined by learned boundary predictors that detect local "
    "acoustic or textual boundaries.\n\n"
    "Key properties:\n\n"
    "- **Boundary prediction:** A lightweight predictor outputs a probability "
    "at each position; positions above a threshold become chunk boundaries.\n"
    "- **Compression:** Tokens within each chunk are aggregated (e.g., averaged) "
    "into a single representation, reducing the effective sequence length.\n"
    "- **Hierarchical processing:** The compressed sequence is processed by a "
    "higher-level model, enabling multi-scale representation learning.\n\n"
    "This approach reduces the effective sequence length for the upper-level "
    "Transformer, lowering the quadratic O(N²) attention cost to a function of "
    "the number of chunks rather than the raw token count "
    "[Citation: Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, "
    "arXiv:2507.07955v2, Page 4, Section 3.2]."
)

ANSWER_DYNAMIC_CHUNKING_EFFICIENCY = (
    "**Dynamic Chunking vs. Direct Byte-Level Models**\n\n"
    "Direct byte-level models like MambaByte incur prohibitive computational "
    "costs because they must attend over every individual byte in the sequence, "
    "resulting in O(N²) complexity for attention-based architectures.\n\n"
    "**H-Net** addresses this by dynamically compressing sequences via learned "
    "boundary predictors:\n\n"
    "- The raw sequence of length N is segmented into C chunks (where C ≪ N)\n"
    "- Each chunk is compressed into a single representation\n"
    "- The upper-level model operates on C tokens instead of N\n\n"
    "This hierarchical approach achieves scalable processing while preserving "
    "the ability to capture fine-grained local patterns through the lower-level "
    "within-chunk model "
    "[Citation: Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, "
    "arXiv:2507.07955v2, Page 2, Section 1]."
)

ANSWER_ENCODER_LAYERS = (
    "**Transformer Encoder Architecture**\n\n"
    "The Transformer encoder is composed of a stack of **N = 6 identical layers**. "
    "Each layer contains two sub-layers:\n\n"
    "1. **Multi-Head Self-Attention** — Applies scaled dot-product attention "
    "across multiple parallel heads, allowing the model to attend to information "
    "from different representation subspaces.\n"
    "2. **Position-wise Feed-Forward Network** — A fully connected network applied "
    "independently to each position, consisting of two linear transformations with "
    "a ReLU activation in between.\n\n"
    "Each sub-layer uses a **residual connection** followed by **layer normalisation**:\n\n"
    "> **output = LayerNorm(x + SubLayer(x))**\n\n"
    "All sub-layers and embedding layers produce outputs of dimension "
    "d_model = 512 "
    "[Citation: Attention Is All You Need, arXiv:1706.03762v7, Page 3, Section 3.1]."
)

ANSWER_BLEU = (
    "**BLEU Scores on WMT 2014**\n\n"
    "On the WMT 2014 English-to-German translation task:\n\n"
    "| Model | BLEU |\n"
    "|:------|-----:|\n"
    "| Transformer (base) | 27.3 |\n"
    "| Transformer (big) | **28.4** |\n\n"
    "The big model achieved a new **state-of-the-art BLEU score of 28.4**, "
    "surpassing all previously published models and ensembles at a fraction "
    "of the training cost "
    "[Citation: Attention Is All You Need, arXiv:1706.03762v7, Page 8, Section 6.1]."
)

ANSWER_DMODEL = (
    "**Base Transformer Model Configuration**\n\n"
    "The base Transformer model uses the following hyperparameters:\n\n"
    "| Parameter | Value |\n"
    "|:----------|------:|\n"
    "| d_model | 512 |\n"
    "| d_ff | 2,048 |\n"
    "| Attention heads | 8 |\n"
    "| Layers (N) | 6 |\n"
    "| dₖ = dᵥ | 64 |\n\n"
    "The model was trained for **100,000 steps** (approximately **12 hours**) "
    "on **8 NVIDIA P100 GPUs** "
    "[Citation: Attention Is All You Need, arXiv:1706.03762v7, Page 7, Section 5.1]."
)

ANSWER_TOKENIZATION_COMPARISON = (
    "**Tokenisation: Transformer vs. Moshi**\n\n"
    "The two architectures use fundamentally different tokenisation strategies:\n\n"
    "- **Original Transformer** — Uses static subword tokenisation (BPE) over "
    "discrete text tokens. The vocabulary is fixed at training time.\n"
    "- **Moshi (Mimi codec)** — Uses a neural audio codec with multi-codebook "
    "Residual Vector Quantisation (RVQ) operating at 12.5 Hz over temporal "
    "acoustic features. Audio is encoded into sequences of discrete codec tokens "
    "rather than text tokens.\n\n"
    "This difference reflects the distinct modalities: text sequences for NMT "
    "versus continuous audio waveforms for speech dialogue "
    "[Citation: Moshi: a speech-text foundation model for real-time dialogue, "
    "arXiv:2410.00037v1, Page 2, Section 2]."
)

ANSWER_HIERARCHICAL_COMPLEXITY = (
    "**Quadratic Complexity & Hierarchical Reduction**\n\n"
    "The original Transformer has **O(N²)** quadratic complexity with respect "
    "to the sequence length N, because each token must attend to every other token "
    "in the self-attention computation.\n\n"
    "Dynamic Chunking reduces this by:\n\n"
    "1. Segmenting the sequence into C variable-length chunks (C ≪ N)\n"
    "2. Compressing each chunk into a single representation\n"
    "3. Running the upper-level attention over C tokens instead of N\n\n"
    "The effective attention cost becomes **O(C²)** for the upper level plus "
    "linear-cost within-chunk processing, dramatically reducing total computation "
    "for long sequences "
    "[Citation: Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, "
    "arXiv:2507.07955v2, Page 2, Section 1]."
)


class StandinSimulator:
    def __init__(self, config: Optional[StandinConfig] = None):
        self.config = config or StandinConfig()
        self.active_requests = 0
        self._lock = asyncio.Lock()

    def update_config(self, new_config: StandinConfig):
        self.config = new_config

    async def increment_active(self):
        async with self._lock:
            self.active_requests += 1

    async def decrement_active(self):
        async with self._lock:
            self.active_requests = max(0, self.active_requests - 1)

    def calculate_ttft(self) -> float:
        """Calculates time to first token based on lognormal base + concurrency slowdown."""
        base = max(50.0, random.gauss(self.config.base_ttft_ms, self.config.ttft_stddev_ms))
        concurrency_delay = self.config.load_concurrency_factor_ms * self.active_requests
        total_ttft = base + concurrency_delay

        if random.random() < self.config.slow_request_prob:
            total_ttft += self.config.slow_request_delay_ms

        return total_ttft

    def generate_synthetic_answer(self, messages: List[Dict[str, str]]) -> str:
        """Constructs a deterministic cited answer grounded in the prompt context and query."""
        user_prompt = ""
        for m in messages:
            if m.get("role") == "user":
                user_prompt += m.get("content", "")

        # Extract only the actual question portion
        if "QUESTION:" in user_prompt:
            question_text = user_prompt.split("QUESTION:")[1].lower()
        elif "question:" in user_prompt.lower():
            question_text = user_prompt.lower().split("question:")[1]
        else:
            question_text = user_prompt.lower()

        # Out-of-scope topics
        if any(w in question_text for w in [
            "quantum", "rsa", "autonomous driving", "obstacle",
            "carbon footprint", "gpt-3",
        ]):
            return REFUSAL_MESSAGE

        # ── Route to specific answers ────────────────────────────

        # Scaled dot-product attention formula
        if "scaled dot-product" in question_text or (
            "formula" in question_text and "attention" in question_text
        ):
            return ANSWER_ATTENTION

        # Learning rate warmup / Adam
        if any(kw in question_text for kw in ["adam", "warmup", "learning rate"]):
            return ANSWER_ADAM_WARMUP

        # Mimi frame rate / RVQ
        if "frame rate" in question_text or "mimi" in question_text:
            if "latency" in question_text:
                return ANSWER_MIMI_LATENCY
            return ANSWER_MIMI_FRAMERATE

        # Moshi full-duplex / dual streams
        if any(kw in question_text for kw in [
            "full-duplex", "dialogue", "parallel streams",
        ]):
            return ANSWER_FULL_DUPLEX

        # Helium text LM
        if any(kw in question_text for kw in ["helium", "backbone"]):
            return ANSWER_HELIUM

        # Dynamic chunking / H-Net boundaries
        if any(kw in question_text for kw in [
            "dynamic chunking", "boundaries", "boundary",
        ]):
            if any(kw in question_text for kw in ["mambabyte", "efficiency"]):
                return ANSWER_DYNAMIC_CHUNKING_EFFICIENCY
            return ANSWER_DYNAMIC_CHUNKING

        # Encoder layers & sublayers
        if "encoder" in question_text and any(
            kw in question_text for kw in ["layers", "identical"]
        ):
            return ANSWER_ENCODER_LAYERS

        # BLEU score on WMT 2014
        if "bleu" in question_text:
            return ANSWER_BLEU

        # d_model and training time
        if any(kw in question_text for kw in ["d_model", "p100"]):
            return ANSWER_DMODEL

        # Cross-paper comparisons
        if any(kw in question_text for kw in [
            "audio tokenization", "tokenisation", "differ",
        ]):
            return ANSWER_TOKENIZATION_COMPARISON

        if any(kw in question_text for kw in ["quadratic", "hierarchical"]):
            return ANSWER_HIERARCHICAL_COMPLEXITY

        return REFUSAL_MESSAGE

    async def stream_response(
        self, text: str, ttft_ms: float
    ) -> AsyncGenerator[str, None]:
        await asyncio.sleep(ttft_ms / 1000.0)
        tokens = text.split(" ")
        for i, token in enumerate(tokens):
            chunk_content = token + (" " if i < len(tokens) - 1 else "")
            chunk_dict = {
                "id": f"standin-chat-{int(time.time()*1000)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "standin-calibrated-lpu",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk_content},
                        "finish_reason": None if i < len(tokens) - 1 else "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(chunk_dict)}\n\n"
            await asyncio.sleep(self.config.inter_token_delay_ms / 1000.0)

        final_chunk = {
            "id": f"standin-chat-{int(time.time()*1000)}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "standin-calibrated-lpu",
            "choices": [],
            "usage": {
                "prompt_tokens": max(50, len(text.split()) * 2),
                "completion_tokens": len(tokens),
                "total_tokens": max(50, len(text.split()) * 2) + len(tokens),
            },
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
