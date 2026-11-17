"""Bounded known-message correlation over every frame and complete windows.

The reference must be known independently of decoding the tested file. These
project criteria do not infer arbitrary watermark presence or AI origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

AGGREGATIONS = ("avg", "squared_avg", "l1norm_avg", "l2norm_avg")
WINDOW_FRAMES = 12
THRESHOLD = 0.9


@dataclass(frozen=True)
class MessageWindow:
    """Scores at native, half-open frame indices."""

    start_frame: int
    end_frame: int
    scores: dict[str, float]


@dataclass(frozen=True)
class VideoMessageReading:
    """Known-target correlation under a named experimental receiver profile."""

    frames: int
    whole_clip: dict[str, float]
    windows: tuple[MessageWindow, ...]
    per_frame_max: float
    max_bit_accuracy: float

    @property
    def target_matched(self) -> bool:
        return self.max_bit_accuracy >= THRESHOLD


def reference_bits(reference_hex: str) -> tuple[int, ...]:
    """Parse exactly 256 independently known bits, represented by 64 hex digits."""
    if len(reference_hex) != 64 or any(char not in "0123456789abcdefABCDEF" for char in reference_hex):
        raise ValueError("The independent reference must contain exactly 64 hexadecimal digits")
    return tuple(int(bit) for byte in bytes.fromhex(reference_hex) for bit in format(byte, "08b"))


class MessageAccumulator:
    """Stream float64 reductions, retaining at most a window and one input batch."""

    def __init__(self, expected: Sequence[int]) -> None:
        import numpy as np

        if len(expected) != 256 or any(bit not in (0, 1) for bit in expected):
            raise ValueError("An independently known 256-bit reference is required")
        self._expected = np.asarray(expected, dtype=np.bool_)
        self._sums: NDArray[Any] = np.zeros((4, 256), dtype=np.float64)
        self._pending: NDArray[Any] = np.empty((0, 4, 256), dtype=np.float64)
        self._count = 0
        self._next_start = 0
        self._frame_max = 0.0
        self._windows: list[MessageWindow] = []
        self._closed = False

    @staticmethod
    def _weighted_terms(values: NDArray[Any]) -> NDArray[Any]:
        import numpy as np

        try:
            with np.errstate(over="raise", invalid="raise"):
                magnitudes = np.abs(values)
                weights_l1 = magnitudes.sum(axis=1)
                weights_l2 = np.sqrt((values * values).sum(axis=1))
                terms = np.stack(
                    [
                        values,
                        values * magnitudes,
                        values * weights_l1[:, None],
                        values * weights_l2[:, None],
                    ],
                    axis=1,
                )
        except FloatingPointError as exc:
            raise RuntimeError("Video message aggregation overflowed") from exc
        if not np.isfinite(terms).all():
            raise RuntimeError("Video message aggregation returned nonfinite values")
        return terms

    @staticmethod
    def _sum_terms(terms: NDArray[Any]) -> NDArray[Any]:
        import numpy as np

        try:
            with np.errstate(over="raise", invalid="raise"):
                totals = terms.sum(axis=0)
        except FloatingPointError as exc:
            raise RuntimeError("Video message aggregation overflowed") from exc
        if not np.isfinite(totals).all():
            raise RuntimeError("Video message aggregation returned nonfinite values")
        return totals

    def _scores(self, sums: NDArray[Any]) -> dict[str, float]:
        import numpy as np

        counts = np.count_nonzero((sums > 0) == self._expected[None], axis=1).tolist()
        return {name: count / 256 for name, count in zip(AGGREGATIONS, counts, strict=True)}

    def update(self, logits: object) -> None:
        """Check a nonempty, sequential batch; invalid checks cannot become negatives."""
        import numpy as np

        if self._closed:
            raise RuntimeError("Video message accumulator is already finalized")
        # Any interrupted or invalid update poisons the stream, including a
        # caller that catches the error and tries to finalize a valid prefix.
        self._closed = True
        values = np.asarray(logits, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 256 or not len(values) or not np.isfinite(values).all():
            raise RuntimeError("Video message decoder returned invalid raw logits")
        terms = self._weighted_terms(values)
        self._sums += self._sum_terms(terms)
        if not np.isfinite(self._sums).all():
            raise RuntimeError("Video message accumulation returned nonfinite values")
        matches = np.count_nonzero((values > 0) == self._expected[None], axis=1)
        self._frame_max = max(self._frame_max, int(matches.max()) / 256)
        self._count += len(values)
        remaining = np.concatenate((self._pending, terms), axis=0)
        while len(remaining) >= WINDOW_FRAMES:
            end = self._next_start + WINDOW_FRAMES
            self._windows.append(
                MessageWindow(self._next_start, end, self._scores(self._sum_terms(remaining[:WINDOW_FRAMES])))
            )
            self._next_start += 1
            remaining = remaining[1:]
        self._pending = remaining.copy()
        self._closed = False

    def finish(self) -> VideoMessageReading:
        """Finalize only after the complete video stream has been decoded."""
        if self._closed or not self._count:
            raise RuntimeError("Video message decoding has no complete, unfinalized stream")
        self._closed = True
        if not self._windows:
            self._windows.append(MessageWindow(0, self._count, self._scores(self._sum_terms(self._pending))))
        whole = self._scores(self._sums)
        maximum = max(self._frame_max, *whole.values(), *(score for w in self._windows for score in w.scores.values()))
        return VideoMessageReading(self._count, whole, tuple(self._windows), self._frame_max, maximum)
