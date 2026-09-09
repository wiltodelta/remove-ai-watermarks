"""Pure text-normalization helpers shared by evaluation scripts."""

from __future__ import annotations

import unicodedata


def normalize_text(text: str) -> str:
    """Normalize text for layout-independent evaluation comparisons."""
    return "".join(unicodedata.normalize("NFC", text).casefold().split())


def levenshtein_normalized(left: str, right: str) -> float:
    """Return Levenshtein distance divided by the longer input length (not CER)."""
    if not left and not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right))


def normalized_edit_distance(left: str, right: str) -> float:
    """Normalize text, then return Levenshtein distance over the result."""
    return levenshtein_normalized(normalize_text(left), normalize_text(right))
