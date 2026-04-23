"""AI metadata detection and removal.

Wraps the noai-watermark metadata handling for stripping AI-generation
metadata (EXIF, PNG text chunks, C2PA provenance) from images.

For metadata-only operations, the heavy ML dependencies are NOT required.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ── Known AI metadata keys ──────────────────────────────────────────

AI_METADATA_KEYS: frozenset[str] = frozenset(
    k.lower()
    for k in [
        "parameters",
        "prompt",
        "negative_prompt",
        "workflow",
        "comfyui",
        "sd-metadata",
        "invokeai_metadata",
        "generation_data",
        "ai_metadata",
        "dream",
        "sd:prompt",
        "sd:negative_prompt",
        "sd:seed",
        "sd:steps",
        "sd:sampler",
        "sd:cfg_scale",
        "sd:model_hash",
        "c2pa",
        "c2pa_chunk",
        "Software",
    ]
)

AI_KEYWORDS: tuple[str, ...] = (
    "stable_diffusion",
    "comfyui",
    "automatic1111",
    "invokeai",
    "midjourney",
    "dall-e",
    "dalle",
    "imagen",
    "synthid",
    "google_ai",
    "openai",
    "c2pa",
)

STANDARD_METADATA_KEYS: frozenset[str] = frozenset(
    [
        "Author",
        "Title",
        "Description",
        "Copyright",
        "Creation Time",
        "Software",
        "Comment",
        "Disclaimer",
        "Source",
        "Warning",
    ]
)


def _is_ai_key(key: str) -> bool:
    """Check if a metadata key is AI-related."""
    key_lower = key.lower()
    if key_lower in AI_METADATA_KEYS:
        return True
    return any(kw in key_lower for kw in AI_KEYWORDS)


def has_ai_metadata(image_path: Path) -> bool:
    """Check if an image contains AI-generation metadata.

    Args:
        image_path: Path to the image.

    Returns:
        True if AI metadata is detected.
    """
    from PIL import Image

    with Image.open(image_path) as img:
        for key in img.info:
            if _is_ai_key(key):
                return True

    # Check C2PA
    try:
        from c2pa import has_c2pa_metadata

        if has_c2pa_metadata(image_path):
            return True
    except ImportError:
        # Try simple binary scan (read only first 512KB to avoid OOM on huge files)
        with open(image_path, "rb") as f:
            data = f.read(512 * 1024)
        if b"c2pa" in data.lower() or b"C2PA" in data:
            return True

    return False


def _scan_png_c2pa_chunk(image_path: Path) -> dict[str, str]:
    """Extract a human-readable summary of the C2PA manifest in a PNG file.

    PIL does not expose the caBX JUMBF box via ``img.info``, so we delegate
    chunk extraction to the existing ``extract_c2pa_chunk`` helper and pull
    key fields from the JUMBF payload without a full CBOR parser.
    """
    import re

    from remove_ai_watermarks.noai.c2pa import extract_c2pa_chunk

    raw = extract_c2pa_chunk(image_path)
    if raw is None:
        return {}

    # extract_c2pa_chunk returns chunk_header (8 bytes) + data + crc (4 bytes).
    payload = raw[8:-4]
    result: dict[str, str] = {"c2pa_manifest": f"C2PA manifest ({len(payload)} bytes)"}

    def _cbor_text_after(key: bytes) -> str | None:
        """Return the CBOR text-string immediately following ``key``.

        Handles CBOR major-type 3 length prefixes: direct (0x60-0x77),
        1-byte (0x78 NN), and 2-byte (0x79 NN NN).
        """
        idx = payload.find(key)
        if idx < 0:
            return None
        p = idx + len(key)
        if p >= len(payload):
            return None
        head = payload[p]
        if 0x60 <= head <= 0x77:
            length, start = head - 0x60, p + 1
        elif head == 0x78 and p + 1 < len(payload):
            length, start = payload[p + 1], p + 2
        elif head == 0x79 and p + 2 < len(payload):
            length, start = (payload[p + 1] << 8) | payload[p + 2], p + 3
        else:
            return None
        raw_str = payload[start : start + length]
        try:
            return raw_str.decode("utf-8")
        except UnicodeDecodeError:
            return raw_str.decode("latin1", errors="replace")

    if generator := _cbor_text_after(b"name"):
        result["claim_generator"] = generator

    if spec := _cbor_text_after(b"specVersion"):
        result["c2pa_spec"] = spec

    dst_match = re.search(
        rb"(http://cv\.iptc\.org/newscodes/digitalsourcetype/[A-Za-z0-9_-]+)",
        payload,
    )
    if dst_match:
        result["digital_source_type"] = dst_match.group(1).decode("latin1")

    actions = sorted(
        {m.decode("latin1") for m in re.findall(rb"c2pa\.(created|converted|edited|opened|placed)", payload)}
    )
    if actions:
        result["c2pa_actions"] = ", ".join(actions)

    # Scan cert DN printable strings for the signer org name.
    signer_match = re.search(
        rb"([A-Za-z][A-Za-z0-9 .,&'()\-]{2,48}OpenAI[A-Za-z0-9 .,&'()\-]{0,48})",
        payload,
    )
    if signer_match:
        result["signer"] = signer_match.group(1).decode("latin1").strip()

    return result


def get_ai_metadata(image_path: Path) -> dict[str, str]:
    """Extract AI-related metadata from an image.

    Args:
        image_path: Path to the image.

    Returns:
        Dictionary of AI metadata key-value pairs.
    """
    from PIL import Image

    result: dict[str, str] = {}

    with Image.open(image_path) as img:
        for key, value in img.info.items():
            if _is_ai_key(key):
                if isinstance(value, bytes):
                    result[key] = f"<binary {len(value)} bytes>"
                elif isinstance(value, str) and len(value) > 200:
                    result[key] = value[:200] + "…"
                else:
                    result[key] = str(value)

    result.update(_scan_png_c2pa_chunk(image_path))
    return result


def remove_ai_metadata(
    source_path: Path,
    output_path: Path | None = None,
    keep_standard: bool = True,
) -> Path:
    """Remove AI-generation metadata from an image.

    Strips EXIF AI tags, PNG text chunks, and C2PA provenance manifests
    while optionally preserving standard metadata (Author, Title, etc.).

    Args:
        source_path: Path to the source image.
        output_path: Output path (None = overwrite source).
        keep_standard: If True, preserve standard metadata fields.

    Returns:
        Path to the cleaned image.
    """
    import piexif
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    if output_path is None:
        output_path = source_path

    # Read image and filter metadata
    with Image.open(source_path) as img:
        img = img.copy()
        fmt = output_path.suffix.lower()

        save_kwargs: dict = {}
        if fmt in (".jpg", ".jpeg"):
            save_kwargs["format"] = "JPEG"
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
        else:
            save_kwargs["format"] = "PNG"

        # Collect non-AI metadata
        kept_meta: dict[str, str] = {}
        exif_data = None

        for key, value in img.info.items():
            if _is_ai_key(key):
                continue
            if key == "exif":
                with contextlib.suppress(Exception):
                    exif_data = piexif.load(value)
                continue
            if key in ("dpi", "gamma"):
                save_kwargs[key] = value
                continue
            if keep_standard and key in STANDARD_METADATA_KEYS:
                kept_meta[key] = str(value) if not isinstance(value, str) else value

        # Apply cleaned metadata
        if save_kwargs["format"] == "PNG" and kept_meta:
            pnginfo = PngInfo()
            for k, v in kept_meta.items():
                pnginfo.add_text(k, v)
            save_kwargs["pnginfo"] = pnginfo

        if exif_data and save_kwargs["format"] == "JPEG":
            with contextlib.suppress(Exception):
                save_kwargs["exif"] = piexif.dump(exif_data)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, **save_kwargs)

    logger.info("Stripped AI metadata → %s", output_path)
    return output_path
