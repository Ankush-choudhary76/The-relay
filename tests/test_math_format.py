"""Tests for the Unicode math formatter."""

from relay.math_format import format_math_unicode


class TestSquareRoot:
    def test_sqrt_function_notation(self):
        assert "√dₖ" in format_math_unicode("sqrt(d_k)")

    def test_sqrt_simple(self):
        assert format_math_unicode("sqrt(x)") == "√x"

    def test_latex_sqrt_braces(self):
        assert "√dₖ" in format_math_unicode("\\sqrt{d_k}")

    def test_sqrt_in_formula(self):
        result = format_math_unicode("softmax(QK^T / sqrt(d_k))")
        assert "√dₖ" in result
        assert "sqrt" not in result


class TestSuperscripts:
    def test_transpose(self):
        assert format_math_unicode("K^T") == "Kᵀ"

    def test_squared(self):
        assert format_math_unicode("x^2") == "x²"

    def test_cubed(self):
        assert format_math_unicode("x^3") == "x³"

    def test_negative_exponent(self):
        result = format_math_unicode("10^{-9}")
        assert "⁻" in result
        assert "⁹" in result

    def test_on2_quadratic(self):
        assert "N²" in format_math_unicode("O(N^2)")

    def test_no_caret_in_output(self):
        result = format_math_unicode("K^T, x^2, N^3")
        assert "^" not in result


class TestSubscripts:
    def test_d_k(self):
        assert format_math_unicode("d_k") == "dₖ"

    def test_d_v(self):
        assert format_math_unicode("d_v") == "dᵥ"

    def test_beta_subscripts(self):
        assert format_math_unicode("beta1") == "β₁"
        assert format_math_unicode("beta_2") == "β₂"

    def test_epsilon(self):
        assert format_math_unicode("epsilon") == "ε"

    def test_latex_subscript_braces(self):
        result = format_math_unicode("x_{model}")
        assert "ₘ" in result
        assert "_{" not in result


class TestLatexSymbols:
    def test_times(self):
        assert format_math_unicode("a \\times b") == "a × b"

    def test_cdot(self):
        assert format_math_unicode("a \\cdot b") == "a · b"

    def test_greek_letters(self):
        assert format_math_unicode("\\alpha") == "α"
        assert format_math_unicode("\\beta") == "β"
        assert format_math_unicode("\\sigma") == "σ"
        assert format_math_unicode("\\theta") == "θ"

    def test_inequality(self):
        assert format_math_unicode("a \\leq b") == "a ≤ b"
        assert format_math_unicode("a \\geq b") == "a ≥ b"

    def test_sum_infinity(self):
        assert format_math_unicode("\\sum to \\infty") == "∑ to ∞"


class TestLatexFractions:
    def test_simple_frac(self):
        result = format_math_unicode("\\frac{a}{b}")
        assert result == "a/b"

    def test_frac_in_context(self):
        result = format_math_unicode("\\frac{QK^T}{\\sqrt{d_k}}")
        assert "QKᵀ" in result
        assert "√dₖ" in result
        assert "\\frac" not in result


class TestDollarSignStripping:
    def test_inline_math(self):
        result = format_math_unicode("The value $x^2$ is big")
        assert "x²" in result
        assert "$" not in result

    def test_display_math(self):
        result = format_math_unicode("$$x^2 + y^2 = z^2$$")
        assert "²" in result
        assert "$$" not in result


class TestCodeSpanPreservation:
    def test_code_span_not_modified(self):
        result = format_math_unicode("Use `d_k` in code and d_k in text")
        assert "`d_k`" in result
        assert "dₖ" in result

    def test_code_span_with_caret(self):
        result = format_math_unicode("Run `x^2` but x^2 is squared")
        assert "`x^2`" in result
        assert "x²" in result


class TestCompleteFormulas:
    def test_attention_formula(self):
        raw = "Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V"
        result = format_math_unicode(raw)
        assert "Kᵀ" in result or "Kᵀ" in result
        assert "√dₖ" in result
        assert "^" not in result
        assert "sqrt" not in result

    def test_unicode_passthrough(self):
        """Text that's already Unicode should pass through unchanged."""
        text = "Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V"
        assert format_math_unicode(text) == text


class TestGroqMathFormatting:
    def test_groq_latex_display_attention(self):
        raw = r"\[ \text{Attention}(Q,K,V)=\operatorname{softmax}!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V \]"
        result = format_math_unicode(raw)
        assert "Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V" in result
        assert r"\frac" not in result
        assert r"\top" not in result
        assert "!" not in result

    def test_groq_bracketed_attention(self):
        raw = r"[ Attention(Q,K,V)=softmax!(\frac{QK!\top}{√dₖ})V ]"
        result = format_math_unicode(raw)
        assert "> **Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V**" in result
        assert r"\frac" not in result
        assert r"\top" not in result

    def test_groq_bracketed_attention_with_period(self):
        raw = r"[ Attention(Q,K,V)=softmax!(\frac{QK!\top}{√dₖ})V ]."
        result = format_math_unicode(raw)
        assert "> **Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V**" in result

    def test_matrix_symbol_parentheses_cleanup(self):
        raw = "Given query, key, and value matrices (Q), (K), and (V) (each of dimension (d_k) for keys/queries and (d_v) for values)"
        result = format_math_unicode(raw)
        assert "matrices Q, K, and V" in result
        assert "(dₖ)" not in result
        assert "dₖ" in result
        assert "dᵥ" in result


