"""Bounded neighboring context for independent-channel audio transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from numpy.typing import NDArray


def transform_chunks(
    chunks: Iterable[NDArray[Any]],
    transform: Callable[[NDArray[Any]], NDArray[Any]],
    *,
    context_frames: int,
) -> Iterator[NDArray[Any]]:
    """Apply a transform with input context, retaining exact central frame counts."""
    import numpy as np

    if context_frames < 1:
        raise ValueError("Audio context must contain at least one frame")
    iterator = iter(chunks)

    def block(value: NDArray[Any]) -> NDArray[Any]:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or not array.shape[0] or not array.shape[1] or not np.isfinite(array).all():
            raise ValueError("Audio chunks must contain finite nonempty sample frames")
        return array

    try:
        value = next(iterator, None)
        if value is None:
            return
        current = block(value)
        previous = current[:0]
        while True:
            value = next(iterator, None)
            following = None if value is None else block(value)
            if following is not None and len(current) < context_frames:
                raise ValueError("Nonfinal audio chunks must contain enough transform context")
            right = current[:0] if following is None else following[:context_frames]
            padded = np.concatenate((previous, current, right), axis=0)
            result = np.asarray(transform(padded), dtype=np.float32)
            if result.shape != padded.shape or not np.isfinite(result).all():
                raise RuntimeError("Audio transform changed shape or returned nonfinite samples")
            yield result[len(previous) : len(previous) + len(current)]
            if following is None:
                break
            previous = current[-context_frames:].copy()
            current = following
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
