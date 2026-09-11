"""Rebuild the visible-watermark alpha assets from controlled captures.

The committed, reproducible build of the bundled visible-mark assets. The inputs
live in ``data/calibration/<engine>/`` (committed solid-color captures run
through the generator). Re-run after re-capturing.

**Doubao "豆包AI生成" strip and Jimeng "★ 即梦AI" wordmark** are fixed
semi-transparent white overlays; the asset is their recovered per-pixel alpha map
(``assets/<engine>_alpha.png``). The "careful" solve (issue #13) -- a naive build
(max-over-channels, coarse background, blur, truncated halo, or a black-dominated
least-squares fit) leaves a visible outline because the alpha is wrong at the glyph
edges:

1. Locate the mark on the BLACK capture (bright pixels in the bottom-right).
2. Fit a smooth CUBIC background per channel over the GRAY capture's non-glyph
   pixels (a cubic captures the gentle gradient without bleeding glyph values).
3. Solve ``a = (I - B) / (255 - B)`` on the gray capture, AVERAGED over channels,
   at FULL halo extent (down to a~0.02) and UNBLURRED. Gray (background ~130-200)
   is the reference because the mark sits on bright photo content in real use, not
   on black; the white capture only confirms the logo is white.

**Gemini sparkle** is a different type: a single icon stamped on PURE BLACK, so the
engine reads ``alpha = max(R,G,B)/255`` directly (no background fit). Its assets are
the sparkle-on-black capture cropped to two fixed logo sizes (``gemini_bg_{96,48}.png``).

**Qwen symbol** is solved from the cleared Qwen Create provider fixture. The symbol
occupies a recorded square in a smooth linen region. A cubic surface fitted outside
that square estimates the local background; the three interior components of the
positive residual are the symbol. Keeping the extraction as code makes the bundled
asset reproducible instead of a hand-cropped copy of the fixture.

Usage::

    uv run python scripts/visible_alpha_solve.py doubao
    uv run python scripts/visible_alpha_solve.py jimeng
    uv run python scripts/visible_alpha_solve.py gemini
    uv run python scripts/visible_alpha_solve.py qwen_symbol
    uv run python scripts/visible_alpha_solve.py all
"""

# cv2/numpy boundary: third-party libs ship no usable element types; relax the
# unknown-type rules for this file only (mirrors the engine modules).
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from remove_ai_watermarks import image_io

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EngineSpec:
    """Per-engine capture inputs and the alpha asset to rebuild."""

    name: str
    capture_dir: Path
    black: str
    gray: str
    asset: Path
    native_width: int = 2048
    corner: str = "br"  # which corner the mark sits in: "br" (Doubao/Jimeng) or "bl" (Samsung)


_SPECS: dict[str, EngineSpec] = {
    "doubao": EngineSpec(
        "doubao",
        _ROOT / "data" / "calibration" / "doubao",
        "doubao_black_1x1_1.png",
        "doubao_gray_1x1_1.png",
        _ROOT / "src" / "remove_ai_watermarks" / "assets" / "doubao_alpha.png",
    ),
    "jimeng": EngineSpec(
        "jimeng",
        _ROOT / "data" / "calibration" / "jimeng",
        "jimeng_cap_A.png",  # black seed
        "jimeng_cap_C.png",  # gray seed
        _ROOT / "src" / "remove_ai_watermarks" / "assets" / "jimeng_alpha.png",
    ),
    "samsung": EngineSpec(
        "samsung",
        _ROOT / "data" / "calibration" / "samsung",
        "samsung_black_1.png",  # black flat edit (mark on true black, bottom-left)
        "samsung_gray_1.png",  # gray flat edit
        _ROOT / "src" / "remove_ai_watermarks" / "assets" / "samsung_alpha.png",
        # The flat captures arrive at the phone's flat-edit size (1086 wide); the
        # mark is a fixed FRACTION of width (~0.31), consistent with the 2958-wide
        # real photos, so geometry is emitted relative to the capture width.
        native_width=1086,
        corner="bl",
    ),
}

_CUBIC_BG_PAD = 30  # px of background margin around the mark for the cubic fit
_GLYPH_BODY = 0.08  # alpha above this is the solid glyph body (for the bbox)
_MIN_PART_AREA = 25  # drop connected glyph-mask blobs smaller than this (cubic-fit specks)
_HALO_PAD = 7  # keep this many px of halo around the glyph body in the saved asset

# Gemini is a different watermark TYPE: a single sparkle icon stamped on a
# PURE-BLACK background (so the engine reads alpha = max(R,G,B)/255 directly, no
# background fit). Its assets are the sparkle-on-black CAPTURE at two fixed logo
# sizes (the engine interpolates between them), not an alpha map.
_GEMINI_CAPTURE = _ROOT / "data" / "calibration" / "gemini" / "gemini_black_2048.png"
_GEMINI_ASSETS: dict[int, Path] = {
    96: _ROOT / "src" / "remove_ai_watermarks" / "assets" / "gemini_bg_96.png",
    48: _ROOT / "src" / "remove_ai_watermarks" / "assets" / "gemini_bg_48.png",
}

_QWEN_SYMBOL_CAPTURE = _ROOT / "data" / "fixtures" / "provenance" / "qwen-create-qwen-image-2.png"
_QWEN_SYMBOL_ASSET = _ROOT / "src" / "remove_ai_watermarks" / "assets" / "qwen_symbol_alpha.png"
# Recorded provider-symbol box (x, y, width, height) in the 2048-square fixture.
_QWEN_SYMBOL_BOX = (1925, 1917, 70, 70)
_QWEN_SYMBOL_BG_PAD = 45


def _union_bbox(mask: NDArray[np.uint8], err: str) -> tuple[int, int, int, int]:
    """Union bbox ``(x0, x1, y0, y1)`` of ``mask``'s connected components with area
    >= ``_MIN_PART_AREA``. The mark is several separate glyphs, so the union spans
    the whole word while a stray small speck/blotch is dropped by the area filter.
    Raises ``ValueError(err)`` if nothing qualifies."""
    n, _labels, stats, _c = cv2.connectedComponentsWithStats(mask, connectivity=8)
    parts = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= _MIN_PART_AREA]
    if not parts:
        raise ValueError(err)
    x0 = min(int(stats[i, cv2.CC_STAT_LEFT]) for i in parts)
    y0 = min(int(stats[i, cv2.CC_STAT_TOP]) for i in parts)
    x1 = max(int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]) for i in parts)
    y1 = max(int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) for i in parts)
    return x0, x1, y0, y1


def _locate_on_black(black: NDArray[np.float32], corner: str = "br") -> tuple[int, int, int, int]:
    """Bounding box of the white mark on the black capture, in the given corner.

    Thresholds well above the blotchy near-black background, then unions the
    sufficiently-large bright components so the box spans the whole word. ``corner``
    is ``"br"`` (bottom-right, Doubao/Jimeng) or ``"bl"`` (bottom-left, Samsung).
    The horizontal window is kept generous (the Samsung text strip is ~0.31 of the
    width, so a corner *quarter* would clip it) while still excluding any centered
    generated content the flat edit hallucinated.
    """
    h, w = black.shape[:2]
    lum = black.mean(axis=2)
    br = lum > 40  # comfortably above the ~5-30 background blotches
    br[: h * 3 // 4, :] = False  # bottom quarter only
    if corner == "bl":
        br[:, w // 2 :] = False  # left half only
    else:
        br[:, : w * 3 // 4] = False  # right quarter only
    bright = cv2.morphologyEx(br.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return _union_bbox(bright, f"no mark found on the black capture ({corner} corner is empty)")


def _cubic_background(crop: NDArray[np.float32], glyph: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Per-channel cubic surface fit over the non-glyph pixels of ``crop``."""
    h, w = crop.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy /= h
    xx /= w
    terms = [np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy, xx**3, xx * xx * yy, xx * yy * yy, yy**3]
    basis = np.stack(terms, axis=-1).reshape(-1, len(terms))
    keep = (~glyph).reshape(-1)
    out = np.zeros_like(crop)
    for ch in range(3):
        values = crop[..., ch].reshape(-1)
        coef, *_ = np.linalg.lstsq(basis[keep], values[keep], rcond=None)
        out[..., ch] = (basis @ coef).reshape(h, w)
    return out


def solve_alpha(spec: EngineSpec) -> NDArray[np.uint8]:
    """Solve the careful gray-self alpha map for one engine (uint8, a*255)."""
    black = image_io.imread(str(spec.capture_dir / spec.black), cv2.IMREAD_COLOR)
    gray = image_io.imread(str(spec.capture_dir / spec.gray), cv2.IMREAD_COLOR)
    if black is None or gray is None:
        raise FileNotFoundError(f"missing captures in {spec.capture_dir} (expected {spec.black}, {spec.gray})")
    black_f = black.astype(np.float32)
    gray_f = gray.astype(np.float32)

    img_h, img_w = black_f.shape[:2]
    mx0, mx1, my0, my1 = _locate_on_black(black_f, spec.corner)
    pad = _CUBIC_BG_PAD
    rx0, rx1 = max(0, mx0 - pad), min(img_w, mx1 + pad)
    ry0, ry1 = max(0, my0 - pad), min(img_h, my1 + pad)
    cg = gray_f[ry0:ry1, rx0:rx1]
    cb = black_f[ry0:ry1, rx0:rx1]

    glyph = cv2.dilate((cb.mean(axis=2) > 8).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    bg = _cubic_background(cg, glyph)
    alpha = np.clip((cg - bg).mean(axis=2) / np.clip(255.0 - bg.mean(axis=2), 1e-3, None), 0.0, 1.0)

    # Crop to the UNION of the glyph parts (the mark is several disconnected
    # glyphs), padded by _HALO_PAD -- this keeps the real anti-aliased halo while
    # dropping the small cubic-fit specks at the crop edges (< _MIN_PART_AREA) that
    # a bare a>floor box would otherwise inflate the asset with.
    body = (alpha > _GLYPH_BODY).astype(np.uint8)
    bx, bex, by, bey = _union_bbox(body, "solved alpha has no glyph body -- check the gray capture background")
    cx0 = max(0, bx - _HALO_PAD)
    cy0 = max(0, by - _HALO_PAD)
    cx1 = min(alpha.shape[1], bex + _HALO_PAD)
    cy1 = min(alpha.shape[0], bey + _HALO_PAD)
    tight = alpha[cy0:cy1, cx0:cx1]
    aw, ah = tight.shape[1], tight.shape[0]
    # Absolute asset position in the capture, for the engine's geometry constants.
    abs_x0, abs_y0 = rx0 + cx0, ry0 + cy0
    # Horizontal margin depends on the anchor corner: left margin for "bl", right
    # margin (distance from the right edge) for "br".
    h_margin = abs_x0 if spec.corner == "bl" else img_w - (abs_x0 + aw)
    log.info(
        "%s: alpha %dx%d max %.3f | WIDTH_FRAC %.4f HEIGHT_FRAC %.4f "
        "MARGIN_%s_FRAC %.4f MARGIN_BOTTOM_FRAC %.4f (native_width %d)",
        spec.name,
        aw,
        ah,
        float(tight.max()),
        aw / spec.native_width,
        ah / spec.native_width,
        "LEFT" if spec.corner == "bl" else "RIGHT",
        h_margin / spec.native_width,
        (img_h - (abs_y0 + ah)) / spec.native_width,
        spec.native_width,
    )
    return (np.clip(tight, 0.0, 1.0) * 255.0).astype(np.uint8)


def solve_gemini() -> dict[int, NDArray[np.uint8]]:
    """Extract the Gemini sparkle-on-black region from the black capture at each
    bundled logo size (the bg-capture asset format; the engine derives the alpha).
    Returns ``{size: bgr_image}``."""
    black = image_io.imread(str(_GEMINI_CAPTURE), cv2.IMREAD_COLOR)
    if black is None:
        raise FileNotFoundError(f"missing Gemini capture {_GEMINI_CAPTURE}")
    h, w = black.shape[:2]
    bright = np.zeros((h, w), np.uint8)
    reg = black.astype(np.float32).mean(axis=2) > 60  # sparkle is ~0.5*255 on black
    reg[: h * 3 // 4, :] = False
    reg[:, : w * 3 // 4] = False
    bright[reg] = 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    x0, x1, y0, y1 = _union_bbox(bright, "no sparkle found on the Gemini black capture")
    crop = black[y0:y1, x0:x1]
    log.info("gemini: sparkle %dx%d at margin_frac %.4f", x1 - x0, y1 - y0, (w - x1) / w)
    return {size: cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA) for size in _GEMINI_ASSETS}


def solve_qwen_symbol() -> NDArray[np.uint8]:
    """Recover the three-lobe Qwen symbol from its cleared provider fixture."""
    source = image_io.imread(str(_QWEN_SYMBOL_CAPTURE), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(f"missing Qwen symbol capture {_QWEN_SYMBOL_CAPTURE}")
    x, y, width, height = _QWEN_SYMBOL_BOX
    pad = _QWEN_SYMBOL_BG_PAD
    crop = source[y - pad : y + height + pad, x - pad : x + width + pad].astype(np.float32)
    excluded = np.zeros(crop.shape[:2], dtype=bool)
    excluded[pad - 8 : pad + height + 8, pad - 8 : pad + width + 8] = True
    background = _cubic_background(crop, excluded)
    alpha = np.clip(
        ((crop - background) / np.clip(255.0 - background, 1e-3, None)).mean(axis=2),
        0.0,
        1.0,
    )[pad : pad + height, pad : pad + width]

    body = (alpha > 0.10).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(body, connectivity=8)
    parts = [
        index
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= 300
        and stats[index, cv2.CC_STAT_LEFT] > 0
        and stats[index, cv2.CC_STAT_TOP] > 0
        and stats[index, cv2.CC_STAT_LEFT] + stats[index, cv2.CC_STAT_WIDTH] < width
        and stats[index, cv2.CC_STAT_TOP] + stats[index, cv2.CC_STAT_HEIGHT] < height
    ]
    if len(parts) != 3:
        raise ValueError(f"expected three Qwen symbol lobes, found {len(parts)}")
    selected = np.isin(labels, parts)
    ys, xs = np.where(selected)
    tight = np.where(selected, alpha, 0.0)[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    tight[tight < 0.08] = 0.0
    log.info("qwen-symbol: alpha %dx%d max %.3f", tight.shape[1], tight.shape[0], float(tight.max()))
    return (tight * 255.0).astype(np.uint8)


@click.command()
@click.argument("engine", type=click.Choice([*_SPECS, "gemini", "qwen_symbol", "all"]))
def main(engine: str) -> None:
    """Rebuild the alpha asset(s) for ENGINE."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    def _write(path: Path, img: NDArray[np.uint8], label: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not image_io.imwrite(str(path), img):
            raise OSError(f"failed to write {path}")
        log.info("%s: wrote %s", label, path.relative_to(_ROOT))

    if engine in (*_SPECS, "all"):
        specs = list(_SPECS.values()) if engine == "all" else [_SPECS[engine]]
        for spec in specs:
            _write(spec.asset, solve_alpha(spec), spec.name)
    if engine in ("gemini", "all"):
        for size, img in solve_gemini().items():
            _write(_GEMINI_ASSETS[size], img, f"gemini-{size}")
    if engine in ("qwen_symbol", "all"):
        _write(_QWEN_SYMBOL_ASSET, solve_qwen_symbol(), "qwen-symbol")


if __name__ == "__main__":
    main()
