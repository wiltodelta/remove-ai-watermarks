"""Research implementation of the response statistic in SAMark v3, §III-C."""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ResponseScore:
    z: float | None
    flags: tuple[int, ...]
    count: int


def score_responses(rows: Sequence[Sequence[float]], sharpness: float = 30.0) -> ResponseScore:
    """Compute Algorithm 2 from supplied cosine responses, without key derivation.

    A zero channel sum yields sign zero. Constant alignment has no defined
    z statistic, so its result carries None rather than a detector decision.
    """
    if not math.isfinite(sharpness) or sharpness <= 0.0:
        raise ValueError("Sharpness must be finite and positive.")
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Responses must be a nonempty rectangular matrix.")
    if any(not -1.0 <= value <= 1.0 for row in rows for value in row):
        raise ValueError("Cosine responses must be within [-1, 1].")

    channel_sums = (math.fsum(channel) for channel in zip(*rows, strict=True))
    flags = tuple((value > 0.0) - (value < 0.0) for value in channel_sums)
    alignment = [math.tanh(sharpness * value * flag) for row in rows for value, flag in zip(row, flags, strict=True)]
    count = len(alignment)
    mean = math.fsum(alignment) / count
    variance = statistics.pvariance(alignment)
    z_score = mean * math.sqrt(count / variance) if variance else None
    return ResponseScore(z_score, flags, count)
