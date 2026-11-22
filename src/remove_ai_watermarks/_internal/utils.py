"""Small path helpers shared by optional image pipelines."""

from __future__ import annotations

import contextlib
import os
import secrets
from pathlib import Path
from typing import TYPE_CHECKING

from remove_ai_watermarks._internal.constants import JPEG_SUFFIXES, SUPPORTED_FORMATS

if TYPE_CHECKING:
    from collections.abc import Generator


def is_supported_format(file_path: Path) -> bool:
    """Return whether ``file_path`` has a supported raster suffix."""
    return file_path.suffix.casefold() in SUPPORTED_FORMATS


def get_image_format(file_path: Path) -> str:
    """Return the Pillow save format used by the legacy metadata API.

    The metadata writer only has specialized PNG and JPEG paths. Other accepted
    inputs therefore use its PNG fallback, matching the established API contract.
    """
    return "JPEG" if file_path.suffix.casefold() in JPEG_SUFFIXES else "PNG"


@contextlib.contextmanager
def staged_sibling(output: Path) -> Generator[Path]:
    """Yield an unused sibling of ``output``, published over it only if the block succeeds.

    A strip that leaves AI metadata behind raises inside the block, so nothing reaches
    ``output``. The name is not pre-created (``mkstemp`` would), so the writer gives the
    published file the default permissions rather than ``mkstemp``'s 0600.
    """
    staged = output.with_name(f".{output.stem}-{secrets.token_hex(8)}{output.suffix}")
    try:
        yield staged
        os.replace(staged, output)
    finally:
        staged.unlink(missing_ok=True)


def model_cache_dir() -> Path:
    """Where this library persists downloaded and derived model assets.

    ``HF_HOME`` comes first because on a scale-to-zero runner it is the one path
    mounted persistently; anything under a container-local cache is re-derived on
    every request.
    """
    root = os.environ.get("HF_HOME") or os.environ.get("XDG_CACHE_HOME")
    base = Path(root).expanduser() if root else Path.home() / ".cache"
    return base / "remove-ai-watermarks"
