"""Distinguish currency prose from dollar-delimited inline mathematics."""

from __future__ import annotations

import pytest

from remove_ai_watermarks.text import remove_text_invisible, split_protected_text_spans


@pytest.mark.parametrize(
    "source",
    [
        "Staffing costs $500 per week and updates cost $1,750 per phase.",
        "The estimates are $20,000 and $30,000.",
        "Prices range from $5 to $10, with a $20 deposit.",
        "The price range is $5-$10.",
        "A closing delimiter followed by a digit, $x$2, is literal text.",
        "The first item costs $5. The second item costs $10.",
        "An amount $ 5 and another $ 10 are recorded.",
        "Literal delimiters $ x $ have spaces inside them.",
        "A spaced opening $ x$ is literal text.",
        "A spaced closing $x $ is literal text.",
        "Costs are $500\r\nper week and $1,750 per phase.",
    ],
)
def test_currency_and_spaced_delimiters_remain_prose(source: str) -> None:
    spans = split_protected_text_spans(source)
    assert "".join(span.text for span in spans) == source
    assert not any(span.kind == "math_inline" for span in spans)


@pytest.mark.parametrize(
    ("source", "formula"),
    [
        ("A $500 budget uses $x + 1$ as its formula.", "$x + 1$"),
        ("Formula $x + 1$ precedes the $500 and $1,750 amounts.", "$x + 1$"),
        ("The value $5 + 2$ is seven.", "$5 + 2$"),
        ("The variable $x$ is positive.", "$x$"),
        (r"A $500 budget uses \(x + 1\) as its formula.", r"\(x + 1\)"),
    ],
)
def test_real_inline_formulas_remain_locked(source: str, formula: str) -> None:
    spans = split_protected_text_spans(source)
    assert "".join(span.text for span in spans) == source
    assert [span.text for span in spans if span.kind == "math_inline"] == [formula]
    assert all(span.locked for span in spans if span.kind == "math_inline")


def test_currency_prose_reaches_rewriter_without_a_cross_price_placeholder() -> None:
    source = "Staffing costs $500 per week and updates cost $1,750 per phase."
    seen: list[str] = []

    def rewrite(passage: str) -> str:
        seen.append(passage)
        return passage.replace("Staffing costs", "Personnel costs")

    result = remove_text_invisible(source, rewriter=rewrite)
    assert seen == [source]
    assert result.cleaned_text == "Personnel costs $500 per week and updates cost $1,750 per phase."
