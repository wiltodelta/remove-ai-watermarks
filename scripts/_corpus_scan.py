"""Shared deterministic corpus walks and append-safe JSONL checkpoints."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path


def corpus_day(path: Path, corpus: Path) -> dt.date | None:
    """Read the leading YYYY-MM-DD corpus segment, if present."""
    try:
        return dt.date.fromisoformat(path.relative_to(corpus).parts[0])
    except (ValueError, IndexError):
        return None


def corpus_files(corpus: Path, since: dt.date | None, *, suffixes: Collection[str] | None = None) -> list[Path]:
    """Return a deterministic recursive walk, optionally date- and suffix-bounded."""
    if since is None:
        candidates = corpus.rglob("*")
    else:
        roots = [
            path
            for path in corpus.iterdir()
            if path.is_dir() and (day := corpus_day(path, corpus)) is not None and day >= since
        ]
        candidates = (path for root in roots for path in root.rglob("*"))
    return sorted(
        path for path in candidates if path.is_file() and (suffixes is None or path.suffix.lower() in suffixes)
    )


def repair_jsonl_tail(path: Path) -> None:
    """Truncate an interrupted final JSONL fragment before another append."""
    if not path.exists():
        return
    with path.open("rb+") as stream:
        data = stream.read()
        if data and not data.endswith(b"\n"):
            stream.truncate(data.rfind(b"\n") + 1)
