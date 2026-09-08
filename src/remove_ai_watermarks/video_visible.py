"""Visible AI-watermark localization and removal for video.

Supported marks use fully synthetic silhouettes made from geometric primitives,
OpenCV's built-in font, and Pillow's bundled font. Sora detection searches the
full frame because the wordmark moves. Veo detection covers both the current
four-point diamond and legacy ``Veo`` text. Seedance detects the boxed ``AI``
label, Dola detects its compact text label, Hailuo AI detects the composite
MINIMAX/Hailuo AI label, and Kling AI detects its version-independent wordmark core.
A single frame is never enough to authorize removal: the temporal arbiter
requires the candidate to recur at the same location across adjacent frames.
This keeps isolated lookalikes in clean videos from becoming removal masks.

Video pixels are decoded with OpenCV and encoded with the system ``ffmpeg``.
Variable frame timestamps cross the pipe in a PyAV-muxed NUT stream; uniform
inputs retain the cheaper raw-BGR pipe. Complete audio is stream-copied from
the source. The video stream must be transcoded because visible-mark removal
changes pixels.
"""

# cv2/numpy boundary: these packages do not expose usable types for many array
# operations. Public signatures remain annotated while unknown third-party types
# are relaxed only in this module.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from remove_ai_watermarks.video import VIDEO_VISIBLE_MARKS
from remove_ai_watermarks.video_encoding import (
    abort_raw_video_encoder,
    finish_raw_video_encoder,
    mux_encoded_video,
    probe_video_encode_profile,
    probe_video_timestamps,
    raw_video_command,
    staged_video_output,
    start_raw_video_encoder,
)
from remove_ai_watermarks.video_temporal import stabilize_filled_frame

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from remove_ai_watermarks.watermark_registry import Backend


log = logging.getLogger(__name__)

Region = tuple[int, int, int, int]

_NORMALIZED_SHORT_SIDE = 480
_SORA_TEMPLATE_SIZE = (180, 64)
_SORA_RELATIVE_HEIGHTS = (0.065, 0.075, 0.085, 0.095, 0.105)
_SORA_PROVENANCE_WEAK_CONFIDENCE = 0.58
_SORA_STRICT_WEAK_CONFIDENCE = 0.60
_SORA_STRONG_CONFIDENCE = 0.65
_VEO_PROVENANCE_WEAK_CONFIDENCE = 0.45
_VEO_STRICT_WEAK_CONFIDENCE = 0.50
_VEO_STRONG_CONFIDENCE = 0.55
_SEEDANCE_WEAK_CONFIDENCE = 0.38
_SEEDANCE_STRONG_CONFIDENCE = 0.43
_DOLA_PROVENANCE_WEAK_CONFIDENCE = 0.48
_DOLA_STRICT_WEAK_CONFIDENCE = 0.50
_DOLA_STRONG_CONFIDENCE = 0.52
_HAILUO_WEAK_CONFIDENCE = 0.30
_HAILUO_STRONG_CONFIDENCE = 0.34
_KLING_WEAK_CONFIDENCE = 0.20
_KLING_STRONG_CONFIDENCE = 0.24
_KLING_MIN_WHITE_FRACTION = 0.02
_MIN_STABLE_FRAMES = 5
_MIN_VEO_STABLE_FRAMES = 12
_MIN_FIXED_MARK_STABLE_FRAMES = 12
_MAX_STABLE_GAP = 2
_STABLE_IOU = 0.55
_VEO_REFERENCE_SHORT_SIDE = 720
_VEO_DIAMOND_PROFILES = (
    (56, 92, 92),
    (48, 72, 72),
    (44, 29, 40),
)
_DOLA_RELATIVE_HEIGHTS = tuple(value / 1000 for value in range(22, 41))
_HAILUO_RELATIVE_HEIGHTS = tuple(value / 1000 for value in range(28, 56, 3))
_KLING_RELATIVE_HEIGHTS = tuple(value / 1000 for value in range(24, 49, 3))
_HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})


@dataclass(frozen=True)
class FrameLocalization:
    """Best untrusted visible-mark candidate found in one decoded frame."""

    frame_index: int
    confidence: float
    region: Region | None


@dataclass(frozen=True)
class VideoScan:
    """Decoded video geometry plus one localization candidate per frame."""

    width: int
    height: int
    fps: float
    detections: tuple[FrameLocalization, ...]
    timestamps: tuple[float, ...] = ()


@dataclass(frozen=True)
class _PreparedFrame:
    """Frame representations shared by every detector in one scan pass."""

    gray: NDArray[Any]
    normalized_gray: NDArray[Any]
    normalized_scale: float


def _scalable_default_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load Pillow's bundled scalable font, with a Pillow 10.0 fallback."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # ``size=`` was added after Pillow 10.0, which is still within the
        # package's supported dependency range. Resize that bundled bitmap font
        # before compositing it into the synthetic template.
        return ImageFont.load_default()


def _crop_nonzero(image: NDArray[Any]) -> NDArray[Any]:
    """Crop a synthetic template to its nonzero footprint."""
    ys, xs = np.where(image > 0)
    if len(xs) == 0:
        return image
    return image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


@lru_cache(maxsize=1)
def _sora_templates() -> tuple[NDArray[Any], NDArray[Any]]:
    """Return synthetic full-wordmark and mascot-only silhouettes.

    No source frame or provider logo asset contributes pixels to these templates.
    The cloud-like mascot is assembled from primitive shapes and the word is
    rendered with Pillow's bundled font.
    """
    width, height = _SORA_TEMPLATE_SIZE
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((4, 8, 58, 57), radius=22, fill=255)
    draw.ellipse((0, 18, 26, 50), fill=255)
    draw.ellipse((38, 16, 64, 51), fill=255)
    draw.ellipse((16, 18, 29, 43), fill=0)
    draw.ellipse((35, 18, 48, 43), fill=0)

    font = _scalable_default_font(50)
    if isinstance(font, ImageFont.FreeTypeFont):
        draw.text((67, 0), "Sora", font=font, fill=255, stroke_width=1, stroke_fill=255)
    else:
        text_box = font.getbbox("Sora")
        text = Image.new("L", (max(1, text_box[2]), max(1, text_box[3])), 0)
        ImageDraw.Draw(text).text((0, 0), "Sora", font=font, fill=255)
        text = text.resize((102, 50), Image.Resampling.NEAREST)
        canvas.paste(text, (67, 0), text)

    full = np.asarray(canvas, dtype=np.uint8)
    return full, full[:, :64]


@lru_cache(maxsize=1)
def _veo_templates() -> tuple[NDArray[Any], NDArray[Any]]:
    """Return synthetic current-diamond and legacy-text Veo silhouettes."""
    size = 256
    diamond_canvas = Image.new("L", (size, size), 0)
    diamond_points = (
        (0.50, 0.02),
        (0.60, 0.39),
        (0.98, 0.50),
        (0.60, 0.61),
        (0.50, 0.98),
        (0.40, 0.61),
        (0.02, 0.50),
        (0.40, 0.39),
    )
    ImageDraw.Draw(diamond_canvas).polygon(
        [(round(x * size), round(y * size)) for x, y in diamond_points],
        fill=255,
    )

    text_canvas = Image.new("L", (140, 60), 0)
    text_draw = ImageDraw.Draw(text_canvas)
    text_draw.text((2, 0), "Veo", font=_scalable_default_font(48), fill=255)
    text = _crop_nonzero(np.asarray(text_canvas, dtype=np.uint8))
    return np.asarray(diamond_canvas, dtype=np.uint8), text


@lru_cache(maxsize=1)
def _seedance_template() -> NDArray[Any]:
    """Return a synthetic boxed-AI silhouette for Seedance exports."""
    canvas = Image.new("L", (160, 120), 0)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((8, 8, 142, 105), radius=28, outline=255, width=7)
    draw.text(
        (35, 17),
        "AI",
        font=_scalable_default_font(70),
        fill=255,
        stroke_width=1,
        stroke_fill=255,
    )
    draw.rounded_rectangle((142, 94, 157, 109), radius=3, outline=255, width=2)
    draw.text((145, 94), "AI", font=_scalable_default_font(8), fill=255)
    return np.asarray(canvas, dtype=np.uint8)


@lru_cache(maxsize=1)
def _dola_template() -> NDArray[Any]:
    """Return a synthetic Dola AI text silhouette using OpenCV's font."""
    canvas = np.zeros((100, 400), dtype=np.uint8)
    cv2.putText(
        canvas,
        "Dola AI",
        (2, 72),
        cv2.FONT_HERSHEY_DUPLEX,
        2.2,
        255,
        3,
        cv2.LINE_AA,
    )
    return _crop_nonzero(canvas)


@lru_cache(maxsize=1)
def _hailuo_template() -> NDArray[Any]:
    """Return a synthetic MINIMAX/Hailuo composite-label silhouette."""
    canvas = Image.new("L", (680, 112), 0)
    draw = ImageDraw.Draw(canvas)

    # Five symmetric waveform strokes approximate the provider-independent
    # geometry at the left edge without copying pixels from an export.
    waveform_heights = (42, 70, 96, 70, 42)
    center_y = 56
    for index, height in enumerate(waveform_heights):
        x = 10 + index * 13
        draw.rounded_rectangle(
            (x, center_y - height // 2, x + 5, center_y + height // 2),
            radius=2,
            fill=255,
        )

    font = _scalable_default_font(58)
    draw.text((84, 18), "MINIMAX", font=font, fill=255, stroke_width=1, stroke_fill=255)
    draw.rectangle((342, 18, 347, 92), fill=255)

    # The Hailuo symbol is a ring with a small offset highlight.
    draw.ellipse((370, 20, 446, 96), outline=255, width=12)
    draw.ellipse((397, 38, 434, 76), fill=255)
    draw.ellipse((397, 29, 420, 52), fill=0)
    draw.text((456, 18), "hailuo AI", font=font, fill=255, stroke_width=1, stroke_fill=255)
    return np.asarray(canvas, dtype=np.uint8)


@lru_cache(maxsize=1)
def _kling_templates() -> tuple[NDArray[Any], ...]:
    """Return synthetic font and capitalization variants for the Kling wordmark core."""
    templates: list[NDArray[Any]] = []
    for text in ("KLING AI", "KlingAI"):
        canvas = Image.new("L", (430, 104), 0)
        ImageDraw.Draw(canvas).text(
            (2, 12),
            text,
            font=_scalable_default_font(64),
            fill=255,
            stroke_width=1,
            stroke_fill=255,
        )
        templates.append(_crop_nonzero(np.asarray(canvas, dtype=np.uint8)))
        for font in (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX):
            cv_template = np.zeros((100, 500), dtype=np.uint8)
            cv2.putText(
                cv_template,
                text,
                (2, 72),
                font,
                2.2,
                255,
                3,
                cv2.LINE_AA,
            )
            templates.append(_crop_nonzero(cv_template))
    return tuple(templates)


@lru_cache(maxsize=1)
def _kling_logo_template() -> NDArray[Any]:
    """Return a synthetic ring approximation of the Kling swirl."""
    template = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(template, (50, 50), 34, 255, 14, cv2.LINE_AA)
    return _crop_nonzero(template)


def _top_hat(gray: NDArray[Any]) -> NDArray[Any]:
    kernel = np.ones((7, 7), dtype=np.uint8)
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


@lru_cache(maxsize=1)
def _asset_template(name: str) -> NDArray[Any]:
    """Load a package asset as a video detect template (white glyph on black).

    The image-registry alpha maps are the same synthetic silhouettes the image
    engines match; a video mark that shares the glyph reuses the asset instead of
    re-rendering it. No source frame contributes pixels.
    """
    from pathlib import Path as _P

    from remove_ai_watermarks.image_io import imread as _imread

    at = _imread(str(_P(__file__).parent / "assets" / name), 2)
    if at is None:
        raise RuntimeError(f"missing video template asset: {name}")
    return at


@lru_cache(maxsize=1)
def _doubao_template() -> NDArray[Any]:
    return _asset_template("doubao_alpha.png")


def _template_sources() -> dict[str, NDArray[Any]]:
    """Return every immutable synthetic template by detector-local key."""
    sora_word, sora_icon = _sora_templates()
    veo_diamond, veo_text = _veo_templates()
    sources = {
        "sora-word": sora_word,
        "sora-icon": sora_icon,
        "veo-diamond": veo_diamond,
        "veo-text": veo_text,
        "seedance": _seedance_template(),
        "dola": _dola_template(),
        "hailuo": _hailuo_template(),
        "kling-logo": _kling_logo_template(),
        # The Doubao video mark is the same "豆包AI生成" run the image engine
        # matches (its synthetic alpha asset), bottom-right on video frames.
        "doubao": _doubao_template(),
    }
    sources.update({f"kling-{index}": template for index, template in enumerate(_kling_templates())})
    return sources


@lru_cache(maxsize=512)
def _resized_template_feature(
    template_key: str,
    width: int,
    height: int,
    kernel_size: int,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Resize one template and cache its invariant top-hat representation."""
    template = cv2.resize(
        _template_sources()[template_key],
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    feature = cv2.morphologyEx(template, cv2.MORPH_TOPHAT, kernel)
    return template, feature


def _normalized_gray(image_bgr: NDArray[Any]) -> tuple[NDArray[Any], float]:
    gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    short_side = min(height, width)
    if short_side <= 0:
        return gray, 1.0
    scale = min(1.0, _NORMALIZED_SHORT_SIDE / short_side)
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return gray, scale


def _prepare_frame(image_bgr: NDArray[Any]) -> _PreparedFrame:
    """Compute the shared grayscale representations for one decoded frame."""
    gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    normalized_gray, normalized_scale = _normalized_gray(gray)
    return _PreparedFrame(gray, normalized_gray, normalized_scale)


def _expanded_region(
    location: tuple[int, int],
    template_width: int,
    template_height: int,
    *,
    icon_only: bool,
    scale: float,
    frame_width: int,
    frame_height: int,
) -> Region:
    x = round(location[0] / scale)
    y = round(location[1] / scale)
    height = max(1, round(template_height / scale))
    width = max(1, round(template_width / scale))
    if icon_only:
        width = round(height * _SORA_TEMPLATE_SIZE[0] / _SORA_TEMPLATE_SIZE[1])
    width = min(width, frame_width - x)
    height = min(height, frame_height - y)
    return x, y, max(1, width), max(1, height)


def detect_sora_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the strongest synthetic Sora-wordmark match in one frame.

    The returned candidate is intentionally untrusted. Call
    :func:`stabilize_localizations` across the full sequence before building
    any removal mask.
    """
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)

    frame_height, frame_width = image_bgr.shape[:2]
    prepared = prepared or _prepare_frame(image_bgr)
    gray, scale = prepared.normalized_gray, prepared.normalized_scale
    normalized_height, normalized_width = gray.shape[:2]
    feature = _top_hat(gray)
    best_confidence = 0.0
    best_region: Region | None = None

    for template_index, base_template in enumerate(_sora_templates()):
        icon_only = template_index == 1
        template_key = "sora-icon" if icon_only else "sora-word"
        for relative_height in _SORA_RELATIVE_HEIGHTS:
            template_height = max(16, round(min(normalized_height, normalized_width) * relative_height))
            template_width = max(1, round(base_template.shape[1] * template_height / base_template.shape[0]))
            if template_height >= normalized_height or template_width >= normalized_width:
                continue
            _, template_feature = _resized_template_feature(
                template_key,
                template_width,
                template_height,
                7,
            )
            scores = cv2.matchTemplate(feature, template_feature, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, location = cv2.minMaxLoc(scores)
            if confidence <= best_confidence:
                continue
            best_confidence = float(confidence)
            best_region = _expanded_region(
                location,
                template_width,
                template_height,
                icon_only=icon_only,
                scale=scale,
                frame_width=frame_width,
                frame_height=frame_height,
            )

    return FrameLocalization(frame_index, best_confidence, best_region)


def _match_template(
    gray: NDArray[Any],
    template: NDArray[Any],
    *,
    region: Region,
    kernel_size: int,
    template_feature: NDArray[Any] | None = None,
) -> tuple[float, Region | None]:
    """Match one synthetic silhouette inside a bounded frame region."""
    x, y, width, height = region
    roi = gray[y : y + height, x : x + width]
    template_height, template_width = template.shape[:2]
    if roi.size == 0 or template_height >= roi.shape[0] or template_width >= roi.shape[1]:
        return 0.0, None
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    feature = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, kernel)
    if template_feature is None:
        template_feature = cv2.morphologyEx(template, cv2.MORPH_TOPHAT, kernel)
    scores = cv2.matchTemplate(feature, template_feature, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, location = cv2.minMaxLoc(scores)
    return float(confidence), (
        x + location[0],
        y + location[1],
        template_width,
        template_height,
    )


def _restore_region(
    region: Region | None,
    *,
    scale: float,
    frame_width: int,
    frame_height: int,
) -> Region | None:
    """Map a localization from normalized pixels back to the source frame."""
    if region is None:
        return None
    x, y, width, height = region
    source_x = round(x / scale)
    source_y = round(y / scale)
    source_width = min(frame_width - source_x, max(1, round(width / scale)))
    source_height = min(frame_height - source_y, max(1, round(height / scale)))
    return source_x, source_y, source_width, source_height


def _bounded_region(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    frame_width: int,
    frame_height: int,
) -> Region:
    """Clip an expanded region to the frame without changing its anchor."""
    bounded_x = max(0, x)
    bounded_y = max(0, y)
    return (
        bounded_x,
        bounded_y,
        min(frame_width - bounded_x, width + x - bounded_x),
        min(frame_height - bounded_y, height + y - bounded_y),
    )


def _detect_fixed_mark(
    image_bgr: NDArray[Any],
    template_key: str,
    *,
    relative_heights: tuple[float, ...],
    search_origin: tuple[float, float],
    kernel_fraction: float,
    prefer_larger_within: float = 0.0,
    normalized: tuple[NDArray[Any], float] | None = None,
    frame_index: int,
) -> FrameLocalization:
    """Match one fixed synthetic mark inside a normalized-frame search region."""
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)

    frame_height, frame_width = image_bgr.shape[:2]
    gray, scale = normalized if normalized is not None else _normalized_gray(image_bgr)
    normalized_height, normalized_width = gray.shape[:2]
    short_side = min(normalized_height, normalized_width)
    search_x = round(normalized_width * search_origin[0])
    search_y = round(normalized_height * search_origin[1])
    search_region = (
        search_x,
        search_y,
        normalized_width - search_x,
        normalized_height - search_y,
    )
    matches: list[tuple[float, Region]] = []
    base_template = _template_sources()[template_key]
    for relative_height in relative_heights:
        template_height = max(6, round(short_side * relative_height))
        template_width = max(1, round(base_template.shape[1] * template_height / base_template.shape[0]))
        kernel_size = max(3, round(template_height * kernel_fraction) | 1)
        resized, template_feature = _resized_template_feature(
            template_key,
            template_width,
            template_height,
            kernel_size,
        )
        confidence, candidate = _match_template(
            gray,
            resized,
            region=search_region,
            kernel_size=kernel_size,
            template_feature=template_feature,
        )
        if candidate is not None and confidence > 0:
            matches.append((confidence, candidate))
    if not matches:
        return FrameLocalization(frame_index, 0.0, None)
    best_confidence, best_region = max(matches, key=lambda match: match[0])
    if prefer_larger_within > 0:
        eligible = [
            (confidence, candidate)
            for confidence, candidate in matches
            if confidence >= best_confidence - prefer_larger_within
        ]
        best_confidence, best_region = max(
            eligible,
            key=lambda match: (match[1][2] * match[1][3], match[0]),
        )

    return FrameLocalization(
        frame_index,
        best_confidence,
        _restore_region(
            best_region,
            scale=scale,
            frame_width=frame_width,
            frame_height=frame_height,
        ),
    )


def detect_seedance_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the strongest fixed Seedance boxed-AI candidate."""
    return _detect_fixed_mark(
        image_bgr,
        "seedance",
        relative_heights=(0.065, 0.075, 0.085, 0.095, 0.105),
        search_origin=(0.68, 0.72),
        kernel_fraction=0.12,
        normalized=None if prepared is None else (prepared.normalized_gray, prepared.normalized_scale),
        frame_index=frame_index,
    )


def detect_doubao_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the strongest fixed Doubao "豆包AI生成" text candidate.

    The ByteDance Doubao mark registered for IMAGES also appears burned into the
    bottom-right of Doubao VIDEO exports; this detector reuses the image engine's
    synthetic alpha as the template. Search profile measured on the local private
    corpus (48 TC260-confirmed Doubao videos, 2026-08-28): the run sits at
    cx 0.83-0.90 / bottom-flush, heights swept 3.2-6.0 percent of the short side.
    """
    return _detect_fixed_mark(
        image_bgr,
        "doubao",
        relative_heights=tuple(value / 1000 for value in range(32, 61, 4)),
        search_origin=(0.60, 0.76),
        kernel_fraction=0.14,
        prefer_larger_within=0.04,
        normalized=None if prepared is None else (prepared.normalized_gray, prepared.normalized_scale),
        frame_index=frame_index,
    )


def detect_dola_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the strongest fixed Dola AI text candidate."""
    return _detect_fixed_mark(
        image_bgr,
        "dola",
        relative_heights=_DOLA_RELATIVE_HEIGHTS,
        search_origin=(0.65, 0.85),
        kernel_fraction=0.50,
        normalized=None if prepared is None else (prepared.normalized_gray, prepared.normalized_scale),
        frame_index=frame_index,
    )


def detect_hailuo_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the strongest fixed MINIMAX/Hailuo composite-label candidate."""
    detection = _detect_fixed_mark(
        image_bgr,
        "hailuo",
        relative_heights=_HAILUO_RELATIVE_HEIGHTS,
        search_origin=(0.28, 0.76),
        kernel_fraction=0.18,
        normalized=None if prepared is None else (prepared.normalized_gray, prepared.normalized_scale),
        frame_index=frame_index,
    )
    if detection.region is None:
        return detection
    frame_width = image_bgr.shape[1]
    x, y, width, height = detection.region
    horizontal_padding = round(height * 1.25)
    region = _bounded_region(
        x - horizontal_padding,
        y,
        width + horizontal_padding * 2,
        height,
        frame_width=frame_width,
        frame_height=image_bgr.shape[0],
    )
    return FrameLocalization(
        frame_index,
        detection.confidence,
        region,
    )


def detect_kling_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the fixed Kling wordmark core and include its version suffix."""
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)
    frame_height, frame_width = image_bgr.shape[:2]
    prepared = prepared or _prepare_frame(image_bgr)
    normalized = prepared.normalized_gray, prepared.normalized_scale
    expanded: list[FrameLocalization] = []
    for template_index, _template in enumerate(_kling_templates()):
        detection = _detect_fixed_mark(
            image_bgr,
            f"kling-{template_index}",
            relative_heights=_KLING_RELATIVE_HEIGHTS,
            search_origin=(0.64, 0.84),
            kernel_fraction=0.18,
            prefer_larger_within=0.06,
            normalized=normalized,
            frame_index=frame_index,
        )
        if detection.region is None:
            continue
        x, y, width, height = detection.region
        left_padding = round(height * 1.8)
        right_padding = round(height * 4.0)
        vertical_padding = round(height * 0.4)
        region = _bounded_region(
            x - left_padding,
            y - vertical_padding,
            width + left_padding + right_padding,
            height + vertical_padding * 2,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        expanded.append(
            FrameLocalization(
                frame_index,
                detection.confidence,
                region,
            )
        )
    if not expanded:
        return FrameLocalization(frame_index, 0.0, None)
    edge_candidates = [
        candidate
        for candidate in expanded
        if candidate.region is not None
        and candidate.region[0] + candidate.region[2] >= frame_width * 0.96
        and candidate.region[1] + candidate.region[3] >= frame_height * 0.94
    ]
    font_candidate = None if not edge_candidates else max(edge_candidates, key=lambda candidate: candidate.confidence)

    logo = _detect_fixed_mark(
        image_bgr,
        "kling-logo",
        relative_heights=tuple(value / 1000 for value in range(20, 61, 3)),
        search_origin=(0.62, 0.90),
        kernel_fraction=0.18,
        prefer_larger_within=0.03,
        normalized=normalized,
        frame_index=frame_index,
    )
    logo_candidate: FrameLocalization | None = None
    logo_x: int | None = None
    if logo.region is not None and logo.confidence >= 0.44:
        logo_x, logo_y, _, logo_height = logo.region
        logo_candidate = FrameLocalization(
            frame_index,
            logo.confidence,
            _bounded_region(
                logo_x - round(logo_height * 0.2),
                logo_y - round(logo_height * 0.25),
                round(logo_height * 7.8),
                round(logo_height * 1.5),
                frame_width=frame_width,
                frame_height=frame_height,
            ),
        )

    best = font_candidate or logo_candidate
    if (
        font_candidate is not None
        and font_candidate.region is not None
        and logo_candidate is not None
        and logo_x is not None
    ):
        font_x, _, font_width, _ = font_candidate.region
        if font_x - round(font_width * 0.08) <= logo_x <= font_x + round(font_width * 0.50):
            best = logo_candidate
    if best is None or best.region is None:
        return FrameLocalization(frame_index, 0.0, None)
    x, y, width, height = best.region
    roi = image_bgr[y : y + height, x : x + width]
    if roi.ndim == 2:
        white_fraction = float(np.mean(roi >= 180))
    else:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_fraction = float(np.mean((hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 180)))
    if white_fraction < _KLING_MIN_WHITE_FRACTION:
        return FrameLocalization(frame_index, 0.0, None)
    return best


def detect_veo_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the strongest current-diamond or legacy-text Veo candidate."""
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)

    frame_height, frame_width = image_bgr.shape[:2]
    gray = (prepared or _prepare_frame(image_bgr)).gray
    short_scale = min(frame_height, frame_width) / _VEO_REFERENCE_SHORT_SIDE
    _, text_base = _veo_templates()
    best_confidence = 0.0
    best_region: Region | None = None

    diamond_sizes: set[int] = set()
    for base_size, right_margin, bottom_margin in _VEO_DIAMOND_PROFILES:
        diamond_size = max(16, round(base_size * short_scale))
        diamond_sizes.add(diamond_size)
        kernel_size = max(3, round(7 * short_scale) | 1)
        template, template_feature = _resized_template_feature(
            "veo-diamond",
            diamond_size,
            diamond_size,
            kernel_size,
        )
        expected_x = round(frame_width - (right_margin + base_size) * short_scale)
        expected_y = round(frame_height - (bottom_margin + base_size) * short_scale)
        search_padding = max(6, round(diamond_size * 0.25))
        search_x = max(0, expected_x - search_padding)
        search_y = max(0, expected_y - search_padding)
        search_width = min(frame_width - search_x, diamond_size + search_padding * 2)
        search_height = min(frame_height - search_y, diamond_size + search_padding * 2)
        confidence, candidate = _match_template(
            gray,
            template,
            region=(search_x, search_y, search_width, search_height),
            kernel_size=kernel_size,
            template_feature=template_feature,
        )
        if confidence > best_confidence:
            best_confidence = confidence
            best_region = candidate

    # Provider layouts have moved before. A bounded corner search is a safety
    # net for a relocated diamond, but it is admitted only at a much stronger
    # per-frame score than the known-profile path. Without this gate, recurring
    # bright scene details in clean API exports can become stable false matches.
    corner_x = round(frame_width * 0.65)
    corner_y = round(frame_height * 0.65)
    corner_region = (corner_x, corner_y, frame_width - corner_x, frame_height - corner_y)
    for diamond_size in diamond_sizes:
        kernel_size = max(3, round(7 * short_scale) | 1)
        template, template_feature = _resized_template_feature(
            "veo-diamond",
            diamond_size,
            diamond_size,
            kernel_size,
        )
        confidence, candidate = _match_template(
            gray,
            template,
            region=corner_region,
            kernel_size=kernel_size,
            template_feature=template_feature,
        )
        if confidence >= 0.70 and confidence > best_confidence:
            best_confidence = confidence
            best_region = candidate

    text_region_width = min(frame_width, max(32, round(180 * short_scale)))
    text_region_height = min(frame_height, max(24, round(120 * short_scale)))
    text_region = (
        frame_width - text_region_width,
        frame_height - text_region_height,
        text_region_width,
        text_region_height,
    )
    text_heights = sorted({max(5, round(height * short_scale)) for height in range(8, 22)})
    for text_height in text_heights:
        text_width = max(1, round(text_base.shape[1] * text_height / text_base.shape[0]))
        kernel_size = max(3, round(3 * short_scale) | 1)
        template, template_feature = _resized_template_feature(
            "veo-text",
            text_width,
            text_height,
            kernel_size,
        )
        confidence, candidate = _match_template(
            gray,
            template,
            region=text_region,
            kernel_size=kernel_size,
            template_feature=template_feature,
        )
        if confidence > best_confidence:
            best_confidence = confidence
            best_region = candidate

    return FrameLocalization(frame_index, best_confidence, best_region)


def _region_iou(left: Region, right: Region) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x0 = max(lx, rx)
    y0 = max(ly, ry)
    x1 = min(lx + lw, rx + rw)
    y1 = min(ly + lh, ry + rh)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class VisibleMarkPolicy:
    """One provider's temporal-arbiter tuning, plus the fill geometry it needs.

    Every value here is MEASURED per provider; the arbiter itself
    (:func:`_stabilize_localizations`) is shared and knows nothing about providers.

    ``accepts_provenance`` is load-bearing rather than cosmetic. A vendor with no
    metadata that could confirm it forces ``provenance=False``; that used to be
    guaranteed structurally by wrappers that took no ``provenance`` parameter at
    all, and this flag is what preserves the guarantee now that one entry point
    serves every mark. Kling keeps the flag off (its TC260 producer code is image
    evidence; no kling video corpus row ties a producer to the moving mark), while
    Hailuo accepts it through the MiniMax TC260 label
    (:func:`has_hailuo_video_provenance`).

    ``padding_fraction`` and ``mask_style`` belong to the removal plan rather than the
    arbiter, but they are per-provider constants like the rest, so they live on the same
    row instead of in a parallel branch in ``video.py``.
    """

    weak_floor: float
    strong_floor: float
    transition_floor: float
    min_stable_frames: int
    cover_after_confirmation: bool
    padding_fraction: float
    mask_style: Literal["box", "veo"]
    anchor_iou: float | None = None
    accepts_provenance: bool = True
    # Weak floor to use when provenance confirms the vendor. None = the mark has no
    # relaxed band; metadata never creates a detection, it only lets a stable visual
    # run below the strict floor through.
    provenance_weak_floor: float | None = None


VISIBLE_MARK_POLICIES: dict[str, VisibleMarkPolicy] = {
    "sora": VisibleMarkPolicy(
        weak_floor=_SORA_STRICT_WEAK_CONFIDENCE,
        provenance_weak_floor=_SORA_PROVENANCE_WEAK_CONFIDENCE,
        strong_floor=_SORA_STRONG_CONFIDENCE,
        transition_floor=0.45,
        min_stable_frames=_MIN_STABLE_FRAMES,
        cover_after_confirmation=False,
        padding_fraction=0.28,
        mask_style="box",
    ),
    "veo": VisibleMarkPolicy(
        weak_floor=_VEO_STRICT_WEAK_CONFIDENCE,
        provenance_weak_floor=_VEO_PROVENANCE_WEAK_CONFIDENCE,
        strong_floor=_VEO_STRONG_CONFIDENCE,
        transition_floor=0.35,
        min_stable_frames=_MIN_VEO_STABLE_FRAMES,
        cover_after_confirmation=True,
        padding_fraction=0.18,
        mask_style="veo",
    ),
    "seedance": VisibleMarkPolicy(
        # One floor at either trust level: provenance still gates the run acceptance
        # inside the arbiter, but the confidence bar does not move.
        weak_floor=_SEEDANCE_WEAK_CONFIDENCE,
        strong_floor=_SEEDANCE_STRONG_CONFIDENCE,
        transition_floor=0.30,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.0,
        mask_style="box",
    ),
    "doubao": VisibleMarkPolicy(
        # One floor at either trust level (seedance-style: the TC260 producer
        # confirms the vendor inside the arbiter; the confidence bar does not
        # move). Floors measured 2026-08-28 on the local private corpus with
        # sequential decode: a >=0.35 stable run of 12+ frames accepts 33/39
        # TC260-confirmed Doubao videos and 0/25 no-Doubao negatives; 0.30
        # already accepted 1 negative, so the bar stays at 0.35.
        weak_floor=0.35,
        strong_floor=0.55,
        transition_floor=0.30,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.0,
        mask_style="box",
    ),
    "dola": VisibleMarkPolicy(
        weak_floor=_DOLA_STRICT_WEAK_CONFIDENCE,
        provenance_weak_floor=_DOLA_PROVENANCE_WEAK_CONFIDENCE,
        strong_floor=_DOLA_STRONG_CONFIDENCE,
        transition_floor=0.40,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.20,
        mask_style="box",
    ),
    "hailuo": VisibleMarkPolicy(
        weak_floor=_HAILUO_WEAK_CONFIDENCE,
        strong_floor=_HAILUO_STRONG_CONFIDENCE,
        transition_floor=0.28,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.12,
        mask_style="box",
        # No provenance_weak_floor: the measured weak floor stays the entry bar;
        # a MiniMax TC260 label only drops the strong-frame requirement for an
        # already-stable run (see has_hailuo_video_provenance).
    ),
    "kling": VisibleMarkPolicy(
        weak_floor=_KLING_WEAK_CONFIDENCE,
        strong_floor=_KLING_STRONG_CONFIDENCE,
        transition_floor=0.30,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.12,
        mask_style="box",
        accepts_provenance=False,
    ),
}


def stabilize_localizations(
    mark: str,
    detections: tuple[FrameLocalization, ...] | list[FrameLocalization],
    *,
    provenance: bool = False,
) -> list[Region | None]:
    """Accept only spatially/temporally recurring candidates for ``mark``.

    Metadata never creates a detection. It only allows a stable visual run whose scores
    remain below the strict confidence floor, which covers a low-contrast mark while
    clean metadata-bearing exports stay untouched. A mark whose policy sets
    ``accepts_provenance=False`` ignores the argument entirely.
    """
    policy = VISIBLE_MARK_POLICIES[mark]
    trusted = provenance and policy.accepts_provenance
    weak_floor = policy.provenance_weak_floor if (trusted and policy.provenance_weak_floor is not None) else None
    return _stabilize_localizations(
        detections,
        provenance=trusted,
        weak_floor=weak_floor if weak_floor is not None else policy.weak_floor,
        strong_floor=policy.strong_floor,
        transition_floor=policy.transition_floor,
        min_stable_frames=policy.min_stable_frames,
        cover_after_confirmation=policy.cover_after_confirmation,
        **({"anchor_iou": policy.anchor_iou} if policy.anchor_iou is not None else {}),
    )


def _stabilize_localizations(
    detections: tuple[FrameLocalization, ...] | list[FrameLocalization],
    *,
    provenance: bool,
    weak_floor: float,
    strong_floor: float,
    transition_floor: float,
    min_stable_frames: int,
    cover_after_confirmation: bool,
    anchor_iou: float | None = None,
) -> list[Region | None]:
    """Apply the shared recurrence policy to provider-specific candidates."""
    accepted: list[Region | None] = [None] * len(detections)
    runs: list[list[int]] = []
    current: list[int] = []

    for position, detection in enumerate(detections):
        if detection.region is None or detection.confidence < weak_floor:
            continue
        if current:
            previous = detections[current[-1]]
            anchor = detections[current[0]]
            frame_gap = detection.frame_index - previous.frame_index
            if (
                previous.region is None
                or anchor.region is None
                or frame_gap > _MAX_STABLE_GAP + 1
                or _region_iou(previous.region, detection.region) < _STABLE_IOU
                or (anchor_iou is not None and _region_iou(anchor.region, detection.region) < anchor_iou)
            ):
                runs.append(current)
                current = []
        current.append(position)
    if current:
        runs.append(current)

    for run in runs:
        strong = max(detections[position].confidence for position in run) >= strong_floor
        if len(run) < min_stable_frames or (not provenance and not strong):
            continue
        for position in run:
            accepted[position] = detections[position].region
        for left_position, right_position in pairwise(run):
            if right_position - left_position <= 1:
                continue
            left = detections[left_position]
            right = detections[right_position]
            if left.region is None or right.region is None or _region_iou(left.region, right.region) < _STABLE_IOU:
                continue
            for missing_position in range(left_position + 1, right_position):
                distance_left = missing_position - left_position
                distance_right = right_position - missing_position
                accepted[missing_position] = left.region if distance_left <= distance_right else right.region

    # Provider provenance plus a confirmed run establishes a continuously
    # watermarked app export rather than a clean API export that merely shares
    # the generator name. Veo may also cover the sequence without provenance
    # after its longer, strong fixed-position run. Cover low-contrast transition
    # frames with the nearest confirmed position.
    confirmed_positions = [position for position, region in enumerate(accepted) if region is not None]
    if (provenance or cover_after_confirmation) and confirmed_positions:
        confirmed_regions = [accepted[position] for position in confirmed_positions]
        carry_position = confirmed_positions[0]
        for position, region in enumerate(accepted):
            if region is not None:
                carry_position = position
                continue
            raw = detections[position]
            if (
                raw.region is not None
                and raw.confidence >= transition_floor
                and any(
                    confirmed_region is not None and _region_iou(raw.region, confirmed_region) >= _STABLE_IOU
                    for confirmed_region in confirmed_regions
                )
            ):
                accepted[position] = raw.region
                continue
            accepted[position] = accepted[carry_position]

    return accepted


def _scan_video_detectors(
    source: Path,
    detectors: dict[str, Any],
    *,
    collect_timestamps: bool = True,
) -> dict[str, VideoScan]:
    """Decode once and collect one untrusted candidate per detector and frame."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not decode video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0 or fps <= 0:
        capture.release()
        raise RuntimeError(f"Video has invalid stream geometry or frame rate: {source}")

    detections: dict[str, list[FrameLocalization]] = {mark: [] for mark in detectors}
    timestamps: list[float] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if collect_timestamps:
            timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000)
        if frame.shape[:2] != (height, width):
            capture.release()
            raise RuntimeError(f"Video changes frame dimensions at frame {frame_index}: {source}")
        prepared = _prepare_frame(frame)
        for mark, detector in detectors.items():
            detections[mark].append(
                detector(
                    frame,
                    frame_index=frame_index,
                    prepared=prepared,
                )
            )
        frame_index += 1
    capture.release()
    if frame_index == 0:
        raise RuntimeError(f"Video contains no decodable frames: {source}")
    shared_timestamps: tuple[float, ...] = ()
    if collect_timestamps:
        probed_timestamps = probe_video_timestamps(source)
        if len(probed_timestamps) == frame_index:
            shared_timestamps = probed_timestamps
        else:
            if probed_timestamps:
                log.warning(
                    "ffprobe/OpenCV frame-count mismatch for %s: timestamps=%s decoded=%s; using decoder timestamps",
                    source,
                    len(probed_timestamps),
                    frame_index,
                )
            shared_timestamps = tuple(timestamps)
    return {
        mark: VideoScan(width, height, fps, tuple(mark_detections), shared_timestamps)
        for mark, mark_detections in detections.items()
    }


def scan_video_marks(
    source: Path,
    marks: tuple[str, ...] = VIDEO_VISIBLE_MARKS,
    *,
    collect_timestamps: bool = True,
) -> dict[str, VideoScan]:
    """Decode once and collect candidates for every requested provider mark.

    Timestamp probing is optional because identification never encodes frames.
    Removal keeps it enabled so variable and non-zero-start timing is preserved.
    """
    detectors = dict(
        zip(
            VIDEO_VISIBLE_MARKS,
            (
                detect_sora_frame,
                detect_veo_frame,
                detect_seedance_frame,
                detect_doubao_frame,
                detect_dola_frame,
                detect_hailuo_frame,
                detect_kling_frame,
            ),
            strict=True,
        )
    )
    unsupported = sorted(set(marks) - detectors.keys())
    if unsupported:
        raise ValueError(f"Unsupported visible video mark: {', '.join(unsupported)}")
    return _scan_video_detectors(
        source,
        {mark: detectors[mark] for mark in marks},
        collect_timestamps=collect_timestamps,
    )


def _mask_for_region(
    frame_bgr: NDArray[Any],
    region: Region,
    *,
    padding_fraction: float,
    mask_style: Literal["box", "veo"],
) -> NDArray[Any]:
    height, width = frame_bgr.shape[:2]
    x, y, region_width, region_height = region
    if mask_style == "veo" and 0.80 <= region_width / region_height <= 1.25:
        mask = np.zeros((height, width), dtype=np.uint8)
        diamond_base, _ = _veo_templates()
        diamond = cv2.resize(
            diamond_base,
            (region_width, region_height),
            interpolation=cv2.INTER_AREA,
        )
        diamond = np.where(diamond >= 24, 255, 0).astype(np.uint8)
        dilation = max(2, round(region_height * 0.08))
        kernel_size = dilation * 2 + 1
        diamond = cv2.dilate(
            diamond,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        )
        x1 = min(width, x + region_width)
        y1 = min(height, y + region_height)
        mask[y:y1, x:x1] = diamond[: y1 - y, : x1 - x]
        return mask

    # A glyph-shaped mask leaves a thin translucent rim outside the approximate
    # synthetic silhouette. Classical inpainting then pulls that white rim back
    # into the hole, recreating the mascot as a bright blob. The measured clean
    # floor on real Sora frames is a full box with roughly 0.28 mark-heights of
    # context on every side.
    # Padded rectangle + no dilation is exactly region_eraser.boxes_to_mask (the same
    # primitive the image fill uses); the padding IS the growth, so `dilate=0`.
    from remove_ai_watermarks.region_eraser import boxes_to_mask

    padding = max(4, round(region_height * padding_fraction))
    return boxes_to_mask(
        (height, width),
        [(x - padding, y - padding, region_width + 2 * padding, region_height + 2 * padding)],
        dilate=0,
    )


def _timestamp_time_base(profile_time_base: str | None) -> Fraction:
    """Use the source time base when known, with a microsecond fallback."""
    return Fraction(profile_time_base) if profile_time_base is not None else Fraction(1, 1_000_000)


def _timestamps_are_variable(scan: VideoScan, *, time_base: Fraction) -> bool:
    """Whether OpenCV exposed valid timestamps with non-uniform frame intervals."""
    if len(scan.timestamps) != len(scan.detections) or len(scan.timestamps) < 3:
        return False
    ticks = tuple(round(timestamp / float(time_base)) for timestamp in scan.timestamps)
    intervals = tuple(current - previous for previous, current in pairwise(ticks))
    return all(interval > 0 for interval in intervals) and len(set(intervals)) > 1


class _TimestampedNutWriter:
    """Mux BGR frames with explicit PTS into ffmpeg's standard-input pipe."""

    def __init__(
        self,
        pipe: Any,
        *,
        width: int,
        height: int,
        time_base: Fraction,
        start_pts: int = 0,
    ) -> None:
        import av

        self._av = av
        self._time_base = time_base
        self._start_pts = start_pts
        self._origin: float | None = None
        self._container = av.open(pipe, mode="w", format="nut")
        self._stream = self._container.add_stream("rawvideo", rate=None)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "bgr24"
        self._stream.time_base = time_base
        self._stream.codec_context.time_base = time_base

    def write(self, frame_bgr: NDArray[Any], timestamp: float) -> None:
        """Mux one contiguous BGR frame at its source timeline timestamp."""
        if self._origin is None:
            self._origin = timestamp
        frame = self._av.VideoFrame.from_ndarray(
            np.ascontiguousarray(frame_bgr),
            format="bgr24",
        )
        frame.pts = self._start_pts + round((timestamp - self._origin) / float(self._time_base))
        frame.time_base = self._time_base
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        """Flush the rawvideo encoder and NUT container without closing ffmpeg stdin."""
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


def encode_clean_video(
    source: Path,
    output: Path,
    scan: VideoScan,
    regions: list[Region | None],
    *,
    backend: Backend,
    strip_metadata: bool,
    padding_fraction: float = 0.28,
    mask_style: Literal["box", "veo"] = "box",
    temporal_consistency: bool = True,
) -> int:
    """Decode again, fill accepted regions, and atomically encode with complete audio."""
    from remove_ai_watermarks.watermark_registry import fill, resolve_backend

    if len(regions) != len(scan.detections):
        raise ValueError("Temporal localization count does not match the scanned frame count")

    with staged_video_output(output) as (encoded_video, temporary_output):
        profile = probe_video_encode_profile(source)
        if (profile.component_depth or 0) > 8 or profile.color_transfer in _HDR_TRANSFERS:
            source_format = profile.source_pixel_format or "unknown pixel format"
            raise RuntimeError(
                "Visible video removal currently supports SDR 8-bit input only; "
                f"refusing to silently reduce {source_format} / "
                f"{profile.color_transfer or 'unknown transfer'} to 8-bit SDR"
            )
        time_base = _timestamp_time_base(profile.time_base)
        preserve_start_offset = profile.start_pts not in (None, 0)
        timestamped_input = preserve_start_offset or _timestamps_are_variable(scan, time_base=time_base)
        if timestamped_input and profile.time_base is None:
            profile = replace(
                profile,
                time_base=f"{time_base.numerator}/{time_base.denominator}",
            )
        process = start_raw_video_encoder(
            raw_video_command(
                encoded_video,
                width=scan.width,
                height=scan.height,
                fps=scan.fps,
                crf=14,
                profile=profile,
                timestamped_input=timestamped_input,
                copy_input_timestamps=preserve_start_offset,
            )
        )
        frame_pipe = process.stdin

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            abort_raw_video_encoder(process)
            raise RuntimeError(f"OpenCV could not reopen video for removal: {source}")

        removed_frames = 0
        timestamped_writer: _TimestampedNutWriter | None = None
        previous_source: NDArray[Any] | None = None
        previous_cleaned: NDArray[Any] | None = None
        previous_mask: NDArray[Any] | None = None
        try:
            resolved_backend: Literal["cv2", "migan", "lama"] = resolve_backend(backend)
            if timestamped_input:
                timestamped_writer = _TimestampedNutWriter(
                    frame_pipe,
                    width=scan.width,
                    height=scan.height,
                    time_base=time_base,
                    start_pts=profile.start_pts or 0,
                )
            for frame_index, region in enumerate(regions):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Video ended while reading frame {frame_index}: {source}")
                source_frame = frame
                mask: NDArray[Any] | None = None
                if region is not None:
                    mask = _mask_for_region(
                        frame,
                        region,
                        padding_fraction=padding_fraction,
                        mask_style=mask_style,
                    )
                    frame = fill(frame, mask, backend=resolved_backend)
                    if (
                        temporal_consistency
                        and previous_source is not None
                        and previous_cleaned is not None
                        and previous_mask is not None
                    ):
                        frame = stabilize_filled_frame(
                            previous_source,
                            previous_cleaned,
                            previous_mask,
                            source_frame,
                            frame,
                            mask,
                            copy=False,
                        )
                    removed_frames += 1
                if timestamped_writer is None:
                    frame_pipe.write(frame.tobytes())
                else:
                    timestamped_writer.write(frame, scan.timestamps[frame_index])
                previous_source = source_frame
                previous_cleaned = frame
                previous_mask = mask
            if timestamped_writer is not None:
                timestamped_writer.close()
                timestamped_writer = None
            finish_raw_video_encoder(
                process,
                encoded_video,
                operation="visible-watermark encode",
            )
            mux_encoded_video(
                encoded_video,
                source,
                temporary_output,
                strip_metadata=strip_metadata,
                copy_input_timestamps=preserve_start_offset,
            )
        except Exception:
            if timestamped_writer is not None:
                with suppress(Exception):
                    timestamped_writer.close()
            abort_raw_video_encoder(process)
            raise
        finally:
            capture.release()

    return removed_frames


def has_sora_provenance(markers: dict[str, str]) -> bool:
    """Whether container provenance specifically names the Sora generator."""
    return "sora" in markers.get("claim_generator", "").lower()


def has_veo_provenance(markers: dict[str, str]) -> bool:
    """Whether container provenance names Google as the AI-video generator."""
    identity = " ".join(
        (
            markers.get("claim_generator", ""),
            markers.get("issuer", ""),
        )
    ).lower()
    return "google" in identity and "trainedalgorithmicmedia" in markers.get("source_type", "").lower()


def has_bytedance_video_provenance(markers: dict[str, str]) -> bool:
    """Whether container provenance names ByteDance or BytePlus AI video."""
    identity = " ".join(
        (
            markers.get("claim_generator", ""),
            markers.get("issuer", ""),
        )
    ).lower()
    source_type = markers.get("source_type", "").lower()
    return ("bytedance" in identity or "byteplus" in identity) and "trainedalgorithmicmedia" in source_type


def has_hailuo_video_provenance(markers: dict[str, str]) -> bool:
    """Whether a TC260 label names MiniMax, Hailuo's maker, as the producer.

    Hailuo video exports carry no C2PA; their TC260 ``ContentProducer`` is the
    bare name ``MiniMax`` (verified on retained MiniMax-hailuo clips). The
    producer travels as its own structural marker (``aigc_producer``, set in
    ``metadata.get_ai_metadata``), so this matches the field exactly rather
    than parsing the human-readable ``aigc_label`` sentence.
    """
    return markers.get("aigc_producer", "").strip().lower() == "minimax"
