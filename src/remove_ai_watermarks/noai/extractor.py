"""Read-only metadata extraction from PNG and JPEG images.

Provides functions to pull all metadata, AI-only metadata, or a
human-readable summary without modifying the source file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import piexif
from PIL import Image

from remove_ai_watermarks.noai.c2pa import extract_c2pa_chunk, extract_c2pa_info, has_c2pa_metadata
from remove_ai_watermarks.noai.constants import AI_KEYWORDS, AI_METADATA_KEYS, PNG_METADATA_KEYS


def extract_metadata(source_path: Path) -> dict[str, Any]:
    """
    Extract all metadata from a PNG or JPG file.

    Args:
        source_path: Path to the source image file.

    Returns:
        Dictionary containing all extracted metadata.
    """
    metadata: dict[str, Any] = {}

    with Image.open(source_path) as img:
        # Extract EXIF data
        if "exif" in img.info:
            try:
                exif_dict = piexif.load(img.info["exif"])
                metadata["exif"] = exif_dict
            except Exception:
                metadata["exif_raw"] = img.info["exif"]

        # Extract standard PNG metadata
        for key in PNG_METADATA_KEYS:
            if key in img.info:
                metadata[key] = img.info[key]

        # Extract all other metadata including AI-specific
        for key, value in img.info.items():
            if key not in metadata and key not in ["exif"]:
                metadata[key] = value

        # Extract DPI and gamma if present
        if "dpi" in img.info:
            metadata["dpi"] = img.info["dpi"]
        if "gamma" in img.info:
            metadata["gamma"] = img.info["gamma"]

    # Check for C2PA metadata
    if has_c2pa_metadata(source_path):
        metadata["c2pa"] = extract_c2pa_info(source_path)
        c2pa_chunk = extract_c2pa_chunk(source_path)
        if c2pa_chunk:
            metadata["c2pa_chunk"] = c2pa_chunk

    return metadata


def extract_ai_metadata(source_path: Path) -> dict[str, Any]:
    """
    Extract only AI-generated metadata from a PNG or JPG file.

    Args:
        source_path: Path to the source image file.

    Returns:
        Dictionary containing only AI-related metadata.
    """
    ai_metadata: dict[str, Any] = {}

    with Image.open(source_path) as img:
        for key in AI_METADATA_KEYS:
            if key in img.info:
                ai_metadata[key] = img.info[key]

        for key, value in img.info.items():
            key_lower = key.lower()
            if key not in ai_metadata:
                if any(kw in key_lower for kw in AI_KEYWORDS):
                    ai_metadata[key] = value

    # Check for C2PA metadata
    if has_c2pa_metadata(source_path):
        ai_metadata["c2pa"] = extract_c2pa_info(source_path)
        c2pa_chunk = extract_c2pa_chunk(source_path)
        if c2pa_chunk:
            ai_metadata["c2pa_chunk"] = c2pa_chunk

    return ai_metadata


def has_ai_metadata(image_path: Path) -> bool:
    """
    Check if an image contains AI-generated metadata.

    Args:
        image_path: Path to the image file.

    Returns:
        True if AI metadata is detected, False otherwise.
    """
    with Image.open(image_path) as img:
        for key in AI_METADATA_KEYS:
            if key in img.info:
                return True

    if has_c2pa_metadata(image_path):
        return True

    return False


def get_ai_metadata_summary(source_path: Path) -> str:
    """
    Get a human-readable summary of AI metadata.

    Args:
        source_path: Path to the source image file.

    Returns:
        Formatted string with AI metadata summary.
    """
    ai_meta = extract_ai_metadata(source_path)

    if not ai_meta:
        return "No AI metadata found."

    lines = ["AI Image Metadata:"]
    lines.append("-" * 40)

    for key, value in ai_meta.items():
        if key == "c2pa_chunk":
            continue
        elif key == "c2pa" and isinstance(value, dict):
            lines.append("C2PA Metadata:")
            for ck, cv in value.items():
                lines.append(f"  {ck}: {cv}")
        elif isinstance(value, str) and len(value) > 100:
            value = value[:100] + "..."
            lines.append(f"{key}: {value}")
        elif isinstance(value, bytes):
            lines.append(f"{key}: <binary data ({len(value)} bytes)>")
        else:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)
