"""Scalar implementation of the open SynthID Text Threefry bit and mask rules.

Algorithm reference: Google DeepMind synthid-text at
72810caf309d51f5985dc871a4bce42c3d0cb54d (Apache-2.0).
"""

from __future__ import annotations

import math
from collections import Counter, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_U32 = 0xFFFFFFFF
_PARITY = 0x1BD11BDA
_ROTATIONS = ((13, 15, 26, 6), (17, 29, 16, 24))


def threefry2x32(key: tuple[int, int], first: int, second: int) -> tuple[int, int]:
    """Apply the twenty-round Random123 Threefry2x32 block function."""
    left, right = key
    words = (left, right, left ^ right ^ _PARITY)
    first = (first + left) & _U32
    second = (second + right) & _U32
    for group in range(5):
        for shift in _ROTATIONS[group % 2]:
            first = (first + second) & _U32
            second = first ^ (((second << shift) | (second >> (32 - shift))) & _U32)
        injection = group + 1
        first = (first + words[injection % 3]) & _U32
        second = (second + words[(injection + 1) % 3] + injection) & _U32
    return first, second


def count_g_values(
    tokens: Sequence[int],
    key: int,
    *,
    ngram_len: int,
    depth: int,
    context_history_size: int,
) -> tuple[int, int]:
    """Count one bits and scored positions for the EOS-free continuation."""
    base_key = (key >> 32, key & _U32)
    width = ngram_len - 1
    recent: deque[tuple[int, int]] = deque()
    counts: Counter[tuple[int, int]] = Counter()
    ones = positions = 0
    for end in range(width, len(tokens)):
        context_key = base_key
        for previous in tokens[end - width : end]:
            context_key = threefry2x32(context_key, 0, previous)

        initial_zero = context_key == (0, 0) and len(recent) < context_history_size
        if not counts[context_key] and not initial_zero:
            positions += 1
            for block, offset in enumerate(range(0, depth, 64)):
                high, low = threefry2x32(context_key, tokens[end], block)
                mask = (1 << min(depth - offset, 64)) - 1
                ones += (((high << 32) | low) & mask).bit_count()

        if len(recent) == context_history_size:
            oldest = recent.popleft()
            counts[oldest] -= 1
            if not counts[oldest]:
                del counts[oldest]
        recent.append(context_key)
        counts[context_key] += 1
    return ones, positions


def binomial_half_survival(ones: int, trials: int) -> float:
    """Return P(X >= ones) for a binomial with probability one half."""
    if ones == 0:
        return 1.0
    if ones <= trials // 2:
        return 1.0 - binomial_half_survival(trials - ones + 1, trials)

    first_coefficient = math.comb(trials, ones)
    scale = max(0, trials - first_coefficient.bit_length() - 1000)

    def upper_tail() -> Iterator[float]:
        term = first_coefficient / (1 << (trials - scale))
        for count in range(ones, trials + 1):
            if not term:
                break
            yield term
            term *= (trials - count) / (count + 1)

    return math.ldexp(math.fsum(upper_tail()), -scale)
