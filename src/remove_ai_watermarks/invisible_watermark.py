"""Detect open invisible watermarks embedded by the ``invisible-watermark``
(imwatermark) library -- used by Stable Diffusion, SDXL, and FLUX.

Unlike SynthID (proprietary, no local decoder), these are DWT-DCT watermarks
with a PUBLIC decoder and no secret key, so a fresh, un-re-encoded output can be
identified locally. The known fixed patterns were verified against upstream
source:

- **Stable Diffusion XL** -- diffusers ``StableDiffusionXLWatermarker``
  ``WATERMARK_MESSAGE`` (48-bit).
- **FLUX.2** -- ``black-forest-labs/flux2`` ``src/flux2/watermark.py`` (48-bit).
- **Stable Diffusion 1.x / 2.x** -- the library's default ``"StableDiffusionV1"``
  string (136-bit).

The watermark is fragile: it does NOT survive JPEG re-encoding or resizing
(verified -- gone after JPEG q90), so detection works only on pristine PNG
originals. Absence is never proof. Requires the optional ``invisible-watermark``
package (extra: ``detect``), or the torch-free in-tree decoder via PyWavelets
(extra: ``detect-pywavelets``). ``detect_invisible_watermark`` returns None when
neither is installed. When both are present, ``detect`` / imwatermark is used.
"""

# imwatermark ships no type stubs (like cv2); its decoder returns are Unknown.
# Relax the untyped-library diagnostics for this thin wrapper module only.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Known 48-bit ``bits`` watermarks (dwtDct, no key), name -> message integer.
_BITS_48: dict[str, int] = {
    "Stable Diffusion XL": 0b101100111110110010010000011110111011000110011110,
    "FLUX.2 (Black Forest Labs)": 0b001010101111111010000111100111001111010100101110,
}
# The invisible-watermark default string watermark (SD 1.x / 2.x).
_SD1_STRING = b"StableDiffusionV1"

# Decoded bits/bytes never match a 48-bit pattern by chance: random decode lands
# near 24/48, an exact embed at 48/48 (measured). 44 (<=4 bit errors) is a safe
# floor that tolerates light perturbation without risking a false positive.
_MATCH_48 = 44
_MATCH_SD1_FRAC = 0.92  # fraction of the 136 string bits that must match


def is_available() -> bool:
    """True if an optional open-watermark decoder is installed (``detect`` or ``detect-pywavelets``)."""
    from .optional_deps import module_available

    return module_available("imwatermark") or module_available("pywt")


def _decoder_backend() -> Literal["imwatermark", "dwt_dct"] | None:
    """Which decode implementation will run, or None if no detect extra is present.

    Prefer upstream ``imwatermark`` whenever it is installed so ``[detect]`` keeps
    the historical decoder (PyWavelets alone is not enough to switch -- it is also
    a transitive dep of ``invisible-watermark``).
    """
    from .optional_deps import module_available

    if module_available("imwatermark"):
        return "imwatermark"
    if module_available("pywt"):
        return "dwt_dct"
    return None


def _bits_match(value: int, ref: int, width: int = 48) -> int:
    """Number of matching bits between two ``width``-bit integers."""
    return width - bin(value ^ ref).count("1")


def _bytes_match_frac(a: bytes, b: bytes) -> float:
    """Fraction of matching bits between two equal-length byte strings."""
    if len(a) != len(b) or not a:
        return 0.0
    diff = sum(bin(x ^ y).count("1") for x, y in zip(a, b, strict=True))
    return 1.0 - diff / (8 * len(b))


def _bits_to_int(bits: object) -> int:
    value = 0
    for bit in bits:  # type: ignore[attr-defined]
        value = (value << 1) | (1 if bit else 0)
    return value


def _bits_to_bytes(bits: object, nbytes: int) -> bytes:
    import struct

    import numpy as np

    packed = np.packbits([1 if b else 0 for b in bits])  # type: ignore[attr-defined]
    out = b""
    for i in range(nbytes):
        out += struct.pack(">B", int(packed[i]))
    return out


def _decode_dwt_dct_bits(img: NDArray[Any], wm_len: int) -> object:
    """Extract ``wm_len`` bits via the active backend (imwatermark, else dwt_dct)."""
    backend = _decoder_backend()
    if backend == "imwatermark":
        from imwatermark import WatermarkDecoder

        return WatermarkDecoder("bits", wm_len).decode(img, "dwtDct")
    if backend == "dwt_dct":
        from remove_ai_watermarks.dwt_dct import decode_dwt_dct

        return decode_dwt_dct(img, wm_len=wm_len)
    raise RuntimeError("no open-watermark decoder backend available")


def detect_invisible_watermark(image_path: Path) -> str | None:
    """Return the embedding scheme name if a known open watermark is decoded.

    Returns e.g. ``"Stable Diffusion XL"`` / ``"FLUX.2 (Black Forest Labs)"`` /
    ``"Stable Diffusion 1.x / 2.x"``, or None if none matches, the decoder is
    unavailable, or the image can't be read. Meaningful only on pristine
    (un-re-encoded) images.
    """
    if not is_available():
        return None
    from remove_ai_watermarks import image_io

    img = image_io.imread(image_path)
    if img is None:
        return None

    # 48-bit fixed-message watermarks (SDXL, FLUX.2).
    try:
        bits = _decode_dwt_dct_bits(img, wm_len=48)
        value = _bits_to_int(bits)
        for name, ref in _BITS_48.items():
            if _bits_match(value, ref) >= _MATCH_48:
                return name
    except Exception as exc:  # decode can fail on tiny images
        logger.debug("48-bit watermark decode failed for %s: %s", image_path, exc)

    # 136-bit default string watermark (SD 1.x / 2.x).
    try:
        bits = _decode_dwt_dct_bits(img, wm_len=8 * len(_SD1_STRING))
        raw = _bits_to_bytes(bits, len(_SD1_STRING))
        if _bytes_match_frac(raw, _SD1_STRING) >= _MATCH_SD1_FRAC:
            return "Stable Diffusion 1.x / 2.x"
    except Exception as exc:
        logger.debug("string watermark decode failed for %s: %s", image_path, exc)

    return None
