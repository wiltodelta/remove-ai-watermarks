"""Detect Adobe TrustMark invisible watermarks.

TrustMark (github.com/adobe/trustmark, MIT) is the open, keyless image watermark
behind Adobe "Durable Content Credentials": when a C2PA manifest is stripped, a
TrustMark soft binding can still re-link the asset to its manifest in a
repository. Unlike SynthID it has a PUBLIC decoder with no secret key, so a
TrustMark-stamped image can be identified locally. Adobe's shipping products use
Variant P (the ``com.adobe.trustmark.P`` soft-binding ``alg``); this wrapper
loads that model.

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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

# Adobe ships Variant P in production (com.adobe.trustmark.P).
_MODEL_TYPE = "P"
# Lazily constructed singleton -- model load + first-use download is expensive.
_tm: Any = None


def is_available() -> bool:
    """True if the optional ``trustmark`` package is installed."""
    import importlib.util

    return importlib.util.find_spec("trustmark") is not None


def _decoder() -> Any:
    global _tm
    if _tm is None:
        from trustmark import TrustMark

        _tm = TrustMark(verbose=False, model_type=_MODEL_TYPE)
    return _tm


# JPEG quality for the false-positive durability gate (see detect_trustmark).
# Deliberately mild: a genuine TrustMark survives far harsher, while every
# observed false positive collapsed even at this quality.
_REENCODE_QUALITY = 95


def detect_trustmark(image_path: Path) -> str | None:
    """Return a TrustMark scheme note if a *durable* TrustMark watermark is
    decoded, else None.

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
    JPEG round-trip and require the same schema both times. Every observed false
    positive collapsed under this gate.
    """
    if not is_available():
        return None
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            cover = img.convert("RGB")
        decoder = _decoder()
        _wm_secret, wm_present, wm_schema = decoder.decode(cover)
        if not wm_present:
            return None
        if not _survives_reencode(decoder, cover, wm_schema):
            log.debug("TrustMark decode for %s did not survive re-encode; treating as false positive", image_path)
            return None
    except Exception as exc:  # model download / decode failure / unreadable image
        log.debug("TrustMark decode failed for %s: %s", image_path, exc)
        return None
    return f"Adobe TrustMark (variant {_MODEL_TYPE}, schema {wm_schema})"


def _survives_reencode(decoder: Any, cover: Any, schema: int) -> bool:
    """True if the watermark re-decodes with the same schema after a mild JPEG
    round-trip -- the durability a genuine TrustMark guarantees, which a BCH
    false positive (content noise) does not."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    cover.save(buffer, "JPEG", quality=_REENCODE_QUALITY)
    buffer.seek(0)
    with Image.open(buffer) as reencoded:
        _secret, present, reencoded_schema = decoder.decode(reencoded.convert("RGB"))
    return bool(present) and reencoded_schema == schema
