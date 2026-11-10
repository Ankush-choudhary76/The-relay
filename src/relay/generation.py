"""Answer generation orchestrator with grounded prompting and citation formatting."""

from typing import AsyncGenerator, Dict, List, Optional, Tuple
from relay.provider_interface import ModelProvider
from relay.schemas import ProviderResponse, RetrievedPassage

REFUSAL_MESSAGE = "The provided research papers do not contain sufficient information to answer this question reliably."

GROUNDED_SYSTEM_PROMPT = """You are a rigorous ML research assistant answering questions using ONLY the provided passages from three research papers:
1. Attention Is All You Need (1706.03762v7)
2. Moshi: a speech-text foundation model for real-time dialogue (2410.00037v1)
3. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (2507.07955v2)

STRICT GROUNDING RULES:
1. Base your answer EXCLUSIVELY on the provided passages below.
2. NEVER introduce outside knowledge, training data facts, or assumptions.
3. Every factual statement or claim MUST include a citation at the end of the sentence formatted exactly as:
   [Citation: Paper Title, arXiv:IDvVersion, Page N, Section SectionName]
   Example: [Citation: Attention Is All You Need, arXiv:1706.03762v7, Page 3, Section 3.1]
4. Clearly distinguish between direct textual evidence and logical deduction.
5. If the provided passages do not contain sufficient evidence to answer the question reliably, output EXACTLY:
   "The provided research papers do not contain sufficient information to answer this question reliably."
6. Do not guess, speculate, or fabricate citations.

MATHEMATICAL FORMATTING RULES:
When your answer includes mathematical expressions, equations, or formulas:
1. Use clean Unicode symbols instead of LaTeX:
   - Square roots: √ (e.g., √dₖ instead of \\sqrt{d_k} or sqrt(d_k))
   - Transpose: ᵀ (e.g., QKᵀ instead of QK^T or QK^{\\top})
   - Superscripts: ² ³ ⁴ ⁻¹ ⁿ (e.g., x² instead of x^2)
   - Subscripts: ₀ ₁ ₂ ₖ ᵥ ₙ (e.g., dₖ instead of d_k, dᵥ instead of d_v)
   - Greek letters: α β γ δ ε θ λ σ π (e.g., β₁ instead of beta1)
   - Operators: × · ≤ ≥ ≠ ≈ ∑ (e.g., A · B instead of A * B)
2. Fractions: Write inline with '/' (e.g., QKᵀ / √dₖ instead of \\frac{QK^{\\top}}{\\sqrt{d_k}}).
3. Present important equations on their own line as a Markdown blockquote, e.g.:
   > **Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V**
4. After each equation, define and explain every variable and symbol using bullet points.
5. Strictly avoid raw LaTeX markup:
   - NEVER output LaTeX delimiters like \\[ \\], \\( \\), or $ $.
   - NEVER output raw LaTeX commands like \\sqrt, \\frac, \\top, \\left, \\right, or \\operatorname.
   - NEVER output exclamation marks in formulas (e.g., do NOT write softmax! or \\!).
   - Do NOT wrap isolated variables in parentheses: write matrices Q, K, V and dimension dₖ, NOT (Q), (K), (V), (dₖ).
   - Do NOT use plain-text caret (^) or underscores (_) for powers and indices.

ANSWER STRUCTURE RULES:
1. Start with a bold title summarising the topic.
2. Provide context before presenting equations.
3. List and define key variables using bullet points.
4. Present the equation prominently.
5. Explain each component of the equation in detail.
6. Keep answers complete — do not omit important formulas from the source.
"""


def format_context_prompt(question: str, passages: List[RetrievedPassage]) -> str:
    """Formats retrieved passages into a grounded user prompt."""
    context_blocks = []
    for idx, p in enumerate(passages, 1):
        context_blocks.append(
            f"--- PASSAGE {idx} ---\n"
            f"Paper: {p.paper_title}\n"
            f"arXiv ID: {p.arxiv_id}{p.version}\n"
            f"Page: {p.page_number}\n"
            f"Section: {p.section or 'General'}\n"
            f"Content:\n{p.text}\n"
        )

    passages_text = "\n".join(context_blocks)
    return (
        f"RETRIEVED PASSAGES:\n{passages_text}\n\n"
        f"QUESTION: {question}\n\n"
        f"Provide a concise, factual answer with exact citations strictly based on the passages above."
    )


class AnswerGenerator:
    def __init__(self, provider: Optional[ModelProvider] = None):
        self.provider = provider

    async def generate_answer(
        self,
        question: str,
        passages: List[RetrievedPassage],
        provider: Optional[ModelProvider] = None,
        temperature: float = 0.1,
    ) -> ProviderResponse:
        """Executes grounded answer generation via the configured provider."""
        active_provider = provider or self.provider
        if not active_provider:
            raise ValueError("No model provider configured for answer generation.")
        user_content = format_context_prompt(question, passages)
        messages = [
            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return await active_provider.generate(messages, temperature=temperature)

    async def generate_answer_stream(
        self,
        question: str,
        passages: List[RetrievedPassage],
        provider: Optional[ModelProvider] = None,
        temperature: float = 0.1,
    ) -> AsyncGenerator[Tuple[str, Optional[ProviderResponse]], None]:
        """Streams grounded answer tokens via the configured provider."""
        active_provider = provider or self.provider
        if not active_provider:
            raise ValueError("No model provider configured for answer generation.")
        user_content = format_context_prompt(question, passages)
        messages = [
            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        async for delta, final_resp in active_provider.generate_stream(messages, temperature=temperature):
            yield (delta, final_resp)
