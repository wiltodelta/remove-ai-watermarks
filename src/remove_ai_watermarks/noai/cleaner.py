"""AI metadata cleaning and removal.

Provides functions to identify and strip AI-generation metadata from
PNG and JPEG images while optionally preserving standard fields like
Author, Title, and Copyright.

The removal pipeline:
1. Opens the image and iterates over all metadata keys.
2. Classifies each key as AI-related (using ``constants.AI_METADATA_KEYS``
   and ``constants.AI_KEYWORDS``) or standard.
3. Rebuilds the image with only the desired metadata retained.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import piexif
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from remove_ai_watermarks.noai.constants import AI_KEYWORDS, AI_METADATA_KEYS, PNG_METADATA_KEYS
from remove_ai_watermarks.noai.utils import get_image_format

# Pre-compute a lowercase set for O(1) key lookups.
_AI_KEYS_LOWER: frozenset[str] = frozenset(k.lower() for k in AI_METADATA_KEYS)


def remove_ai_metadata(
    source_path: Path,
    output_path: Path | None = None,
    keep_standard: bool = True,
) -> Path:
    """
    Remove all AI-generated metadata from an image.

    Removes:
    - AI parameters (Stable Diffusion, ComfyUI, etc.)
    - C2PA metadata (Google Imagen, OpenAI, etc.)
    - Any metadata with AI-related keywords

    Args:
        source_path: Path to the source image file.
        output_path: Optional output path. If not provided, modifies source in place.
        keep_standard: If True, keeps standard metadata (Author, Title, etc.).

    Returns:
        Path to the output file with AI metadata removed.
    """
    if output_path is None:
        output_path = source_path

    cleaned_metadata = _extract_non_ai_metadata(source_path, keep_standard)

    with Image.open(source_path) as img:
        img = img.copy()

        output_format = get_image_format(output_path)
        save_kwargs: dict[str, Any] = {"format": output_format}

        # Handle EXIF data (keep it, just don't include AI-related fields)
        if "exif" in cleaned_metadata:
            save_kwargs["exif"] = cleaned_metadata["exif"]

        if output_format == "PNG":
            save_kwargs = _prepare_clean_png_kwargs(save_kwargs, cleaned_metadata)
        elif output_format == "JPEG":
            save_kwargs = _prepare_clean_jpeg_kwargs(save_kwargs, cleaned_metadata)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "JPEG" and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        img.save(output_path, **save_kwargs)

    return output_path


def _extract_non_ai_metadata(source_path: Path, keep_standard: bool) -> dict[str, Any]:
    """Extract metadata excluding AI-related fields."""
    cleaned_metadata: dict[str, Any] = {}

    with Image.open(source_path) as img:
        # Handle EXIF data
        if "exif" in img.info:
            with contextlib.suppress(Exception):
                exif_dict = piexif.load(img.info["exif"])
                cleaned_metadata["exif"] = exif_dict

        # Extract non-AI metadata
        for key, value in img.info.items():
            if _is_ai_metadata_key(key):
                continue

            is_standard = keep_standard and key in PNG_METADATA_KEYS
            is_nonstandard = not keep_standard and key not in ["exif", "dpi", "gamma"] and key not in PNG_METADATA_KEYS
            if is_standard or is_nonstandard:
                cleaned_metadata[key] = value

        # Keep DPI and gamma
        if "dpi" in img.info:
            cleaned_metadata["dpi"] = img.info["dpi"]
        if "gamma" in img.info:
            cleaned_metadata["gamma"] = img.info["gamma"]

    return cleaned_metadata


def _is_ai_metadata_key(key: str) -> bool:
    """Return True if *key* is an AI-generation metadata field.

    Detection uses two layers:
    1. Exact match against the canonical ``AI_METADATA_KEYS`` list.
    2. Substring match against ``AI_KEYWORDS`` (covers partial hits
       like ``"stable_diffusion_model"``).
    """
    key_lower = key.lower()
    if key_lower in _AI_KEYS_LOWER:
        return True
    return any(kw in key_lower for kw in AI_KEYWORDS)


def _prepare_clean_png_kwargs(save_kwargs: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Prepare save kwargs for clean PNG."""
    pnginfo = {}
    exclude_keys = ["exif", "exif_raw", "dpi", "gamma"]

    for key, value in metadata.items():
        if key not in exclude_keys:
            pnginfo[key] = value

    if pnginfo:
        pnginfo_obj = PngInfo()
        for key, value in pnginfo.items():
            if isinstance(value, str):
                pnginfo_obj.add_text(key, value)
            elif isinstance(value, bytes):
                pnginfo_obj.add_text(key, value.decode("utf-8", errors="replace"))
        save_kwargs["pnginfo"] = pnginfo_obj

    if "dpi" in metadata:
        save_kwargs["dpi"] = metadata["dpi"]

    return save_kwargs


def _prepare_clean_jpeg_kwargs(save_kwargs: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Prepare save kwargs for clean JPEG."""
    exif_dict = metadata.get("exif", {"0th": {}, "Exif": {}, "1st": {}, "GPS": {}, "Interop": {}})

    with contextlib.suppress(Exception):
        exif_bytes = piexif.dump(exif_dict)
        save_kwargs["exif"] = exif_bytes

    if "dpi" in metadata:
        save_kwargs["dpi"] = metadata["dpi"]

    return save_kwargs


def has_ai_content(image_path: Path) -> bool:
    """
    Check if an image has any AI-generated content or metadata.

    Args:
        image_path: Path to the image file.

    Returns:
        True if the image contains AI metadata.
    """
    from remove_ai_watermarks.noai.extractor import has_ai_metadata

    return has_ai_metadata(image_path)
