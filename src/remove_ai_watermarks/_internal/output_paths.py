from __future__ import annotations

import os
from pathlib import Path


def unique_output_path(
    desired: str | Path,
    *,
    original: str | Path | None = None,
) -> Path:
    """Return an unused output path, preserving intentional in-place writes."""
    path = Path(desired)

    if original is not None:
        try:
            if path.resolve() == Path(original).resolve():
                return path
        except (OSError, RuntimeError):
            pass

    if not os.path.lexists(path):
        return path

    for index in range(1, 100_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not os.path.lexists(candidate):
            return candidate

    raise FileExistsError(f"No available output name for {path}")
