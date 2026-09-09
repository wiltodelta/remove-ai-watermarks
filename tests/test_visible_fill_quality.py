"""Cheap constructed-reference regressions for every visible image mark."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from remove_ai_watermarks import watermark_registry as wr
from scripts.detector_response import ALPHAS, SIZES
from scripts.fill_quality import psnr, stamp_any
from scripts.invisible_quality_audit import _ssim
from scripts.render_visible_examples import _glyph_asset, build_pair, stamp_image_mark

_IMAGE_KEYS = tuple(mark.key for mark in wr.known_marks())

# The second case changes both the generated background and the geometry. Samsung's
# faint overlay needs the larger width its calibrated detector covers.
_ALTERNATE_SIZE = {
    "gemini": (1280, 960),
    "doubao": (1280, 960),
    "jimeng": (1280, 960),
    "qwen": (1280, 1280),
    "kling": (1280, 960),
    "yuanbao": (1280, 960),
    "samsung": (2304, 1728),
    "runninghub": (1280, 960),
    "baidu": (1280, 960),
    "liblib": (960, 1280),
    "liblib_pill": (960, 1280),
    "microsoft": (1280, 960),
    "jimeng_pill": (960, 1280),
}

_MIN_MASK_COVERAGE = 0.94
_MIN_FILLED_PSNR = 25.0
_MIN_FILLED_SSIM = 0.90


def _score_box(clean: np.ndarray, filled: np.ndarray, changed: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(changed)
    assert ys.size > 0
    pad = 8
    y0, y1 = max(0, int(ys.min()) - pad), min(clean.shape[0], int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(clean.shape[1], int(xs.max()) + 1 + pad)
    truth = clean[y0:y1, x0:x1]
    output = filled[y0:y1, x0:x1]
    return psnr(output, truth), _ssim(
        cv2.cvtColor(output, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY),
    )


def test_samsung_stamp_uses_the_solved_alpha_exactly_once() -> None:
    """The asset already stores the measured ~0.38 peak opacity."""
    clean = np.full((1448, 1086, 3), 100, np.uint8)
    stamped = stamp_image_mark("samsung", clean)
    assert stamped is not None
    marked, (x, y, w, h) = stamped
    alpha = cv2.resize(_glyph_asset("samsung_alpha.png"), (w, h), interpolation=cv2.INTER_LINEAR)
    expected = np.clip(100.0 * (1.0 - alpha) + 255.0 * alpha, 0, 255).astype(np.uint8)
    assert np.array_equal(marked[y : y + h, x : x + w, 0], expected)


def test_samsung_detects_its_faint_mark_on_a_textured_background() -> None:
    """Continuous top-hat preserves a mark that binary thresholding shatters."""
    height, width = 800, 600
    rng = np.random.default_rng(0)
    xx = np.arange(width, dtype=np.float32)[None, :]
    texture = 120 + 25 * np.sin(xx / 9)
    texture = texture + cv2.GaussianBlur(rng.normal(0, 35, (height, width)).astype(np.float32), (0, 0), 2)
    clean = cv2.cvtColor(np.clip(texture, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    stamped = stamp_image_mark("samsung", clean)
    assert stamped is not None

    marked, _box = stamped
    detection = wr.get_mark("samsung").detect(marked)
    assert detection.detected
    assert detection.confidence >= 0.80


def test_detector_response_can_construct_every_declared_grid_cell() -> None:
    """An out-of-search-range mark is a measured miss, not an omitted row."""
    for key in _IMAGE_KEYS:
        clean, _marked = build_pair(key)
        for size_mult in SIZES:
            for alpha_mult in ALPHAS:
                assert stamp_any(clean, key, size_mult=size_mult, alpha_mult=alpha_mult) is not None, (
                    key,
                    size_mult,
                    alpha_mult,
                )


@pytest.mark.parametrize("key", _IMAGE_KEYS)
@pytest.mark.parametrize("case", ["canonical", "alternate"])
def test_cv2_mask_quality_smoke(key: str, case: str) -> None:
    """One fill covers the stamp, clears detection, and preserves local quality."""
    size = None if case == "canonical" else _ALTERNATE_SIZE[key]
    seed = 7 if case == "canonical" else 23
    clean, marked = build_pair(key, size=size, seed=seed)
    changed = np.any(marked != clean, axis=2)
    assert np.any(changed), key

    mark = wr.get_mark(key)
    corroborated = key == "liblib_pill"
    before = mark.detect(marked, provenance=corroborated)
    assert before.detected, f"{key}/{case}: confidence {before.confidence:.3f}"
    localization = mark.localize(marked, force=False, detection=before)
    assert localization.mask is not None, f"{key}/{case}: no mask"

    coverage = float(np.count_nonzero(localization.mask[changed])) / float(np.count_nonzero(changed))
    assert coverage >= _MIN_MASK_COVERAGE, f"{key}/{case}: mask coverage {coverage:.3f}"

    filled, region = mark.remove(marked, backend="cv2", detection=before)
    assert region is not None, key
    assert np.array_equal(filled[localization.mask == 0], marked[localization.mask == 0]), key

    after = mark.detect(filled, provenance=corroborated)
    assert not after.detected, f"{key}/{case}: residual confidence {after.confidence:.3f}"
    filled_psnr, filled_ssim = _score_box(clean, filled, changed)
    assert filled_psnr >= _MIN_FILLED_PSNR, f"{key}/{case}: PSNR {filled_psnr:.2f} dB"
    assert filled_ssim >= _MIN_FILLED_SSIM, f"{key}/{case}: SSIM {filled_ssim:.4f}"
