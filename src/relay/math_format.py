"""Unicode math formatting post-processor for The Relay.

Converts raw math notation (^, _, sqrt, LaTeX commands) into clean
Unicode symbols so answers render professionally in any context
(HTML, Markdown, plain text) without requiring a LaTeX engine.
"""

import re
from typing import List, Optional, Tuple

# ── Unicode lookup tables ──────────────────────────────────────

_SUPERSCRIPTS = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
    'n': 'ⁿ', 'i': 'ⁱ',
}

_SUBSCRIPTS = {
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
    '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
    '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎',
    'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ',
    'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ',
    'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ',
    'v': 'ᵥ', 'x': 'ₓ',
}

# ── LaTeX command → Unicode replacement ────────────────────────

_LATEX_SYMBOLS: List[Tuple[str, str]] = [
    ('\\times', '×'),
    ('\\cdot', '·'),
    ('\\leq', '≤'),
    ('\\geq', '≥'),
    ('\\neq', '≠'),
    ('\\approx', '≈'),
    ('\\sum', '∑'),
    ('\\prod', '∏'),
    ('\\infty', '∞'),
    ('\\alpha', 'α'),
    ('\\beta', 'β'),
    ('\\gamma', 'γ'),
    ('\\delta', 'δ'),
    ('\\epsilon', 'ε'),
    ('\\varepsilon', 'ε'),
    ('\\theta', 'θ'),
    ('\\lambda', 'λ'),
    ('\\sigma', 'σ'),
    ('\\pi', 'π'),
    ('\\omega', 'ω'),
    ('\\mu', 'μ'),
    ('\\tau', 'τ'),
    ('\\phi', 'φ'),
    ('\\psi', 'ψ'),
    ('\\rho', 'ρ'),
    ('\\eta', 'η'),
    ('\\nabla', '∇'),
    ('\\partial', '∂'),
    ('\\in', '∈'),
    ('\\rightarrow', '→'),
    ('\\leftarrow', '←'),
    ('\\Rightarrow', '⇒'),
    ('\\log', 'log'),
    ('\\exp', 'exp'),
    ('\\max', 'max'),
    ('\\min', 'min'),
    ('\\quad', ' '),
    ('\\qquad', '  '),
    ('\\,', ' '),
    ('\\;', ' '),
    ('\\:', ' '),
    ('\\!', ''),
]


def _to_superscript(s: str) -> str:
    """Convert a string of digits/signs to Unicode superscript."""
    return ''.join(_SUPERSCRIPTS.get(c, c) for c in s)


def _to_subscript(s: str) -> str:
    """Convert a string to Unicode subscript characters."""
    return ''.join(_SUBSCRIPTS.get(c, c) for c in s)


def _extract_braced(text: str, start_idx: int) -> Tuple[Optional[str], int]:
    """Find matching { ... } starting at or after start_idx, handling nested braces."""
    idx = text.find('{', start_idx)
    if idx == -1:
        return None, start_idx
    depth = 0
    start_content = idx + 1
    for i in range(idx, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start_content:i], i + 1
    return None, start_idx


def _replace_fractions(text: str) -> str:
    """Recursively replace LaTeX \\frac{num}{den} with clean inline num/den fractions."""
    while True:
        pos = text.find(r'\frac')
        if pos == -1:
            break
        num, next_pos = _extract_braced(text, pos + 5)
        if num is None:
            break
        den, end_pos = _extract_braced(text, next_pos)
        if den is None:
            break
        num_fmt = _format_segment(num).strip()
        den_fmt = _format_segment(den).strip()
        if len(num_fmt) <= 2 and len(den_fmt) <= 2:
            frac_str = f'{num_fmt}/{den_fmt}'
        else:
            frac_str = f'{num_fmt} / {den_fmt}'
        text = text[:pos] + frac_str + text[end_pos:]
    return text


def format_math_unicode(text: str) -> str:
    """Convert raw math notation to clean Unicode symbols.

    Handles patterns commonly found in ML research paper answers:
      sqrt(x)  → √x          K^T     → Kᵀ
      x^2      → x²          d_k     → dₖ
      10^{-9}  → 10⁻⁹        \\alpha  → α

    Code spans (backtick-enclosed) are preserved untouched.
    """
    # Split on code spans so we never modify code
    parts = re.split(r'(`[^`]+`)', text)
    processed = []
    for part in parts:
        if part.startswith('`') and part.endswith('`'):
            processed.append(part)
        else:
            processed.append(_format_segment(part))
    return ''.join(processed)


def _format_segment(text: str) -> str:
    """Apply all math formatting rules to a non-code text segment."""

    # ── 0. Pre-clean LaTeX wrappers, commands, and delimiters ──
    text = text.replace(r'\operatorname{softmax}', 'softmax')
    text = text.replace(r'\mathrm{softmax}', 'softmax')
    text = text.replace(r'\mathbf{softmax}', 'softmax')
    text = text.replace(r'\text{Attention}', 'Attention')
    text = text.replace(r'\mathrm{Attention}', 'Attention')

    text = re.sub(r'\\left\b\s*', '', text)
    text = re.sub(r'\\right\b\s*', '', text)
    text = text.replace(r'\!', '')
    text = text.replace('!\\top', '\\top')
    text = re.sub(r'(\b(?:softmax|Attention|sin|cos|tan|log|exp|min|max))!\s*', r'\1', text)

    # Display delimiters: \[ ... \] and $$ ... $$ become display blockquotes
    text = re.sub(r'\\\[\s*(.+?)\s*\\\]', r'\n> **\1**\n', text, flags=re.DOTALL)
    text = re.sub(r'\$\$(.+?)\$\$', r'\n> **\1**\n', text, flags=re.DOTALL)

    # Inline math \( ... \) and $ ... $ have delimiters stripped
    text = re.sub(r'\\\(\s*(.+?)\s*\\\)', r'\1', text)
    text = re.sub(r'\$(.+?)\$', r'\1', text)

    # ── 1. Transpose: ^T, ^{T}, \top, ^{\top}, ^\top ──
    text = text.replace(r'^{\top}', 'ᵀ').replace(r'^\top', 'ᵀ')
    text = text.replace(r'!^{\top}', 'ᵀ').replace(r'!^\top', 'ᵀ').replace(r'!\top', 'ᵀ')
    text = text.replace(r'^{T}', 'ᵀ').replace(r'^T', 'ᵀ')
    text = text.replace(r'\top', 'ᵀ')
    text = re.sub(r'\^\{?T\}?(?=[\s,;.)·*/\]]|$)', 'ᵀ', text)

    # ── 2. LaTeX \sqrt{x} and plain sqrt(x) ──
    text = re.sub(r'\\sqrt\{([^}]+)\}', lambda m: '√' + _format_segment(m.group(1)), text)
    text = re.sub(r'\\sqrt\(([^)]+)\)', lambda m: '√' + _format_segment(m.group(1)), text)
    text = re.sub(r'(?<!\w)sqrt\(([^)]+)\)', lambda m: '√' + _format_segment(m.group(1)), text)

    # ── 3. Fractions (handles nested braces) ──
    text = _replace_fractions(text)
    # Fallback regex for simple fractions
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', text)

    # ── 4. Known subscript variables ──
    text = re.sub(r'\bd_k\b', 'dₖ', text)
    text = re.sub(r'\bd_v\b', 'dᵥ', text)
    text = re.sub(r'\bd_q\b', 'dq', text)
    text = re.sub(r'\bd_ff\b', 'd_ff', text)
    text = re.sub(r'\bbeta1\b', 'β₁', text)
    text = re.sub(r'\bbeta2\b', 'β₂', text)
    text = re.sub(r'\bbeta_1\b', 'β₁', text)
    text = re.sub(r'\bbeta_2\b', 'β₂', text)
    text = re.sub(r'\bepsilon\b', 'ε', text)

    # ── 5. Superscript braces & single digits: ^{-1}, ^{2n}, ^2 ──
    text = re.sub(r'\^{([^}]+)}', lambda m: _to_superscript(m.group(1)), text)
    text = re.sub(r'\^(-?\d)', lambda m: _to_superscript(m.group(1)), text)
    text = re.sub(r'\^([ni])\b', lambda m: _to_superscript(m.group(1)), text)

    # ── 6. LaTeX subscript braces: _{model}, _{k} ──
    text = re.sub(r'_\{([^}]+)\}', lambda m: _to_subscript(m.group(1)), text)

    # ── 7. Word-bounded inequalities before symbol table ──
    text = re.sub(r'\\le\b', '≤', text)
    text = re.sub(r'\\ge\b', '≥', text)
    text = re.sub(r'\\ne\b', '≠', text)

    # LaTeX symbol commands
    for cmd, symbol in _LATEX_SYMBOLS:
        text = text.replace(cmd, symbol)

    # ── 8. Strip remaining text wrappers ──
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', text)

    # ── 9. Spacing & clean equation formatting ──
    # Format Attention(Q,K,V) -> Attention(Q, K, V)
    text = re.sub(
        r'Attention\(([A-Za-z,\s]+)\)',
        lambda m: 'Attention(' + ', '.join(x.strip() for x in m.group(1).split(',')) + ')' if ',' in m.group(1) else m.group(0),
        text,
    )
    # Ensure space around = in equations
    text = re.sub(r'([A-Za-z0-9_)]+)\s*=\s*([A-Za-z0-9_]+)', r'\1 = \2', text)
    # Ensure space around transpose division: QKᵀ / √dₖ
    text = re.sub(r'(\w)ᵀ\s*/\s*', r'\1ᵀ / ', text)
    # Ensure space between closing parenthesis and trailing factor: )V -> ) V
    text = re.sub(r'\)\s*([A-Za-z])', r') \1', text)

    # ── 10. Display equations wrapped in square brackets: [ Equation ] ──
    text = re.sub(r'(?m)^\[\s*([A-Za-z].*?=.*?)\s*\]\.?$', r'> **\1**', text)

    # ── 11. Clean unnecessary parentheses around single variables ──
    text = re.sub(r'\b(matrices|vectors|tensors|variables)\s+\(([A-Z])\)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<=[,\s])\(([A-Z])\)(?=[,\s\)])', r'\1', text)
    text = re.sub(r'\((\s*[d][ₖᵥq]\s*)\)', r'\1', text)

    return text

