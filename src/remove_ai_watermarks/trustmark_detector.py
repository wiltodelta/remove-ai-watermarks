"""Detect and remove Adobe TrustMark invisible watermarks.

TrustMark (github.com/adobe/trustmark, MIT) is the open, keyless image watermark
behind Adobe "Durable Content Credentials": when a C2PA manifest is stripped, a
TrustMark soft binding can still re-link the asset to its manifest in a
repository. Unlike SynthID it has a PUBLIC decoder and remover with no secret
key, so a TrustMark-stamped image can be identified and scrubbed locally.
Adobe's shipping products default to Variant P (the ``com.adobe.trustmark.P``
soft-binding ``alg``); Variants Q (``com.adobe.trustmark.Q``) and C
(``com.adobe.trustmark.C``) are also supported when explicitly requested or
corroborated by a C2PA soft-binding assertion.

Optional dependency (extra: ``trustmark``); the model weights download on first
use. ``detect_trustmark`` returns None when the package is absent. This detects
provenance (Adobe Content Credentials), NOT AI generation as such -- TrustMark
also marks human-authored content -- so callers should treat it as a watermark
signal, not proof of AI origin.
"""

# trustmark ships no type stubs; relax untyped-library diagnostics for this thin
# wrapper module only.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingImports=false

from __future__ import annotations

import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

TrustMarkVariant = Literal["P", "Q", "C"]
SUPPORTED_VARIANTS: tuple[TrustMarkVariant, ...] = ("P", "Q", "C")
# Adobe ships Variant P in production (com.adobe.trustmark.P).
_MODEL_TYPE: TrustMarkVariant = "P"
# Schema 3 (BCH_3) is below the detector's measured precision threshold; the
# calibration history is canonical in docs/module-internals.md.
_SUPPORTED_SCHEMAS = frozenset({0, 1, 2})
# Lazily constructed singletons -- model load + first-use download is expensive.
# Guarded by a lock so concurrent callers don't double-construct/double-download.
_tm: Any = None
_tm_by_variant: dict[TrustMarkVariant, Any] = {}
_tm_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class TrustMarkDetection:
    """A durable Adobe TrustMark watermark decoded from image pixels."""

    variant: TrustMarkVariant
    schema: int
    payload: str

    @property
    def label(self) -> str:
        """Human-readable scheme label matching :func:`detect_trustmark`."""
        return f"Adobe TrustMark (variant {self.variant}, schema {self.schema})"


@dataclass(frozen=True, slots=True)
class TrustMarkRemoval(TrustMarkDetection):
    """A removed durable Adobe TrustMark watermark and its output path."""

    output: Path


def _validate_variant(variant: str) -> TrustMarkVariant:
    normalized = variant.strip().upper()
    if normalized not in SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported TrustMark variant {variant!r}; expected one of {', '.join(SUPPORTED_VARIANTS)}")
    return normalized


def is_available() -> bool:
    """True if the optional ``trustmark`` package is installed."""
    from .optional_deps import module_available

    return module_available("trustmark")


def _decoder() -> Any:
    global _tm
    if _tm is None:
        with _tm_lock:
            if _tm is None:
                from trustmark import TrustMark

                _tm = TrustMark(verbose=False, model_type=_MODEL_TYPE)
    return _tm


def _decoder_for_variant(variant: TrustMarkVariant) -> Any:
    resolved = _validate_variant(variant)
    if resolved == _MODEL_TYPE:
        return _decoder()
    cached = _tm_by_variant.get(resolved)
    if cached is None:
        with _tm_lock:
            cached = _tm_by_variant.get(resolved)
            if cached is None:
                from trustmark import TrustMark

                cached = TrustMark(verbose=False, model_type=resolved)
                _tm_by_variant[resolved] = cached
    return cached


# JPEG quality for the false-positive durability gate (see detect_trustmark).
# Deliberately mild: a genuine TrustMark survives far harsher. The round-trip
# still needs payload and schema validation because content-correlated false
# positives can survive this compression level.
_REENCODE_QUALITY = 95


def _detect_cover_detailed(
    cover: Image.Image,
    source: object,
    *,
    variant: TrustMarkVariant = _MODEL_TYPE,
) -> TrustMarkDetection | None:
    """Decode one RGB cover after availability and image loading are resolved."""
    resolved = _validate_variant(variant)
    try:
        decoder = _decoder_for_variant(resolved)
        wm_secret, wm_present, wm_schema = decoder.decode(cover, "binary")
        if not wm_present:
            return None
        if wm_schema not in _SUPPORTED_SCHEMAS:
            logger.debug(
                "TrustMark decode for %s used weak schema %s; treating as false positive",
                source,
                wm_schema,
            )
            return None
        if not _survives_reencode(decoder, cover, wm_secret, wm_schema):
            logger.debug("TrustMark decode for %s did not survive re-encode; treating as false positive", source)
            return None
    except Exception as exc:  # model download / decode failure
        logger.debug("TrustMark decode failed for %s: %s", source, exc)
        return None
    return TrustMarkDetection(variant=resolved, schema=int(wm_schema), payload=str(wm_secret))


def detect_trustmark_image(
    image: Image.Image,
    *,
    source: object = "decoded image",
    variant: TrustMarkVariant = _MODEL_TYPE,
) -> str | None:
    """Detect a durable TrustMark in an already decoded Pillow image."""
    resolved = _validate_variant(variant)
    if not is_available():
        return None
    try:
        cover = image.convert("RGB")
    except Exception as exc:
        logger.debug("TrustMark image conversion failed for %s: %s", source, exc)
        return None
    detection = _detect_cover_detailed(cover, source, variant=resolved)
    return detection.label if detection is not None else None


def detect_trustmark_detailed(
    image_path: str | Path,
    *,
    variant: TrustMarkVariant = _MODEL_TYPE,
) -> TrustMarkDetection | None:
    """Return structured TrustMark detection details if a durable mark is decoded, else None."""
    resolved = _validate_variant(variant)
    if not is_available():
        return None
    path = Path(image_path)
    try:
        from PIL import Image

        with Image.open(path) as img:
            cover = img.convert("RGB")
    except Exception as exc:  # unreadable image
        logger.debug("TrustMark decode failed for %s: %s", path, exc)
        return None
    return _detect_cover_detailed(cover, path, variant=resolved)


def detect_trustmark(
    image_path: str | Path,
    *,
    variant: TrustMarkVariant = _MODEL_TYPE,
) -> str | None:
    """Return a TrustMark scheme note if a *durable* TrustMark watermark is decoded, else None.

    Returns e.g. ``"Adobe TrustMark (variant P, schema 0)"`` when the decoder
    reports the watermark present AND it survives a mild JPEG re-encode, or None
    if it is absent, the optional ``trustmark`` package is not installed, or the
    image cannot be read/decoded.

    **False-positive gate.** TrustMark's ``wm_present`` flag is a BCH
    error-correction validity check, which spuriously validates on a small
    fraction of un-watermarked images -- content-correlated, so AI-generated
    textures trip it more often than camera photos (verified 2026-05-29 on real
    files: the false "detections" were on Gemini / OpenAI / Doubao output that
    cannot carry Adobe's watermark, and decoded a random-bytes secret). A genuine
    TrustMark is a *durable* soft binding engineered to survive re-encoding (that
    is its entire purpose once C2PA is stripped), so we re-decode after a mild
    JPEG round-trip and require the same binary payload and schema both times.
    Only the calibrated schemas 0-2 count as high-precision positives.
    """
    detection = detect_trustmark_detailed(image_path, variant=variant)
    return detection.label if detection is not None else None


def remove_trustmark(
    source: str | Path,
    output: str | Path,
    *,
    variant: TrustMarkVariant = _MODEL_TYPE,
) -> TrustMarkRemoval | None:
    """Remove a confirmed durable TrustMark watermark with the official neural remover.

    No output is written unless ``detect_trustmark_detailed`` first confirms a
    durable mark for ``variant`` and the encoded output no longer decodes as
    watermarked under that variant. Alpha channels and display tags are
    preserved via ``image_io.write_bgr_with_alpha``.
    """
    resolved = _validate_variant(variant)
    if not is_available():
        return None

    import numpy as np
    from PIL import Image

    from remove_ai_watermarks import image_io

    source_path, output_path = Path(source), Path(output)
    try:
        with Image.open(source_path) as img:
            cover = img.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not read image: {source}") from exc

    detection = _detect_cover_detailed(cover, source_path, variant=resolved)
    if detection is None:
        return None

    bgr, alpha = image_io.read_bgr_and_alpha(source_path)
    if bgr is None:
        raise ValueError(f"Could not read image: {source}")

    decoder = _decoder_for_variant(resolved)
    cleaned_rgb_img = decoder.remove_watermark(cover).convert("RGB")
    if cleaned_rgb_img.size != (bgr.shape[1], bgr.shape[0]):
        cleaned_rgb_img = cleaned_rgb_img.resize((bgr.shape[1], bgr.shape[0]), Image.Resampling.LANCZOS)
    cleaned_rgb = np.asarray(cleaned_rgb_img, dtype=np.uint8)
    cleaned_bgr = np.ascontiguousarray(cleaned_rgb[..., ::-1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}-", suffix=output_path.suffix, dir=output_path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        if not image_io.write_bgr_with_alpha(
            temporary_path,
            cleaned_bgr,
            alpha,
            display_tags_from=source_path,
            orientation_applied=False,
        ):
            raise OSError(f"failed to write output (is the destination writable?): {output_path}")
        if detect_trustmark_detailed(temporary_path, variant=resolved) is not None:
            return None
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return TrustMarkRemoval(
        variant=detection.variant,
        schema=detection.schema,
        payload=detection.payload,
        output=output_path,
    )


def _survives_reencode(decoder: Any, cover: Any, payload: str, schema: int) -> bool:
    """True if the same watermark re-decodes after a mild JPEG round-trip."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    cover.save(buffer, "JPEG", quality=_REENCODE_QUALITY)
    buffer.seek(0)
    with Image.open(buffer) as reencoded:
        reencoded_payload, present, reencoded_schema = decoder.decode(reencoded.convert("RGB"), "binary")
    return bool(present) and reencoded_schema == schema and reencoded_payload == payload
