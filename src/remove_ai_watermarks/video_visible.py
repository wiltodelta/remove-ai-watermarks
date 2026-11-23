"""Visible AI-watermark localization and removal for video.

Supported marks use fully synthetic silhouettes made from geometric primitives,
frozen renders of OpenCV 4's built-in font, and Pillow's bundled font. Sora detection searches the
full frame because the wordmark moves. Veo detection covers both the current
four-point diamond and legacy ``Veo`` text. Seedance detects the boxed ``AI``
label, Dola detects its compact text label, Hailuo AI detects the composite
MINIMAX/Hailuo AI label, and Kling AI detects its version-independent wordmark core.
A single frame is never enough to authorize removal: the temporal arbiter
requires the candidate to recur at the same location across adjacent frames.
This keeps isolated lookalikes in clean videos from becoming removal masks.

Video pixels are decoded with OpenCV, with a PyAV codec fallback, and encoded
with the resolved ``ffmpeg`` executable.
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
import math
import subprocess
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import cache, lru_cache
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from remove_ai_watermarks._internal.tc260_producers import producer_codes_for_mark, producer_for_code
from remove_ai_watermarks._internal.tc260_producers import tc260_producer_in as tc260_producer_in
from remove_ai_watermarks.video import VIDEO_VISIBLE_MARKS
from remove_ai_watermarks.video_decode import open_video_capture, open_video_capture_16bit
from remove_ai_watermarks.video_encoding import (
    HIGH_BIT_DEPTH_PIXEL_FORMATS,
    abort_raw_video_encoder,
    container_format_args,
    count_decoded_video_frames,
    finish_raw_video_encoder,
    mp4_timescale_args,
    mux_encoded_video,
    probe_video_encode_profile,
    probe_video_keyframes,
    probe_video_timestamps,
    raw_video_command,
    staged_video_output,
    start_raw_video_encoder,
)
from remove_ai_watermarks.video_temporal import (
    ClipAlphaMatte,
    estimate_clip_alpha_matte,
    recover_unobscured_background,
    sobel_magnitude,
    stabilize_filled_frame_bidirectional,
    unblend_ycbcr_roi,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

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
_SORA_ICON_SHAPE_FLOOR = 0.50
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
# The Hailuo label is white. A Hailuo 2.3 clip served by Higgsfield (2026-09-25)
# scored 0.33-0.35 on wood grain, above the real label's 0.31, with no bright
# low-saturation pixel in the box; the real label and the synthetic example
# measure 1.00 and 0.15.
_HAILUO_MIN_WHITE_FRACTION = 0.02
# Vidu (ShengShu) "Vidu AI" logo plus wordmark, bottom-right. Measured 2026-09-24 on
# the one real Vidu Q3 export (fixture visible/vidu/provider-original.mp4): 0.72-0.73
# on every frame at 1080p, 720p and 480p re-encodes, 0.60 at 360p. The maximum over
# the 956 other videos of the local corpus and every tracked fixture (every 12th
# frame) was 0.41; none reached 0.52. One positive clip is a thin cohort: the
# floors sit between the two with margin on both sides.
_VIDU_WEAK_CONFIDENCE = 0.52
_VIDU_STRONG_CONFIDENCE = 0.58
_DREAMINA_WEAK_CONFIDENCE = 0.40
_DREAMINA_STRONG_CONFIDENCE = 0.45
_WAN_WEAK_CONFIDENCE = 0.50
_WAN_STRONG_CONFIDENCE = 0.56
_JIMENG_WEAK_CONFIDENCE = 0.45
_JIMENG_STRONG_CONFIDENCE = 0.52
_JIMENG_PILL_WEAK_CONFIDENCE = 0.28
_JIMENG_PILL_STRONG_CONFIDENCE = 0.31
_YUANBAO_WEAK_CONFIDENCE = 0.40
_YUANBAO_STRONG_CONFIDENCE = 0.46
_QWEN_WEAK_CONFIDENCE = 0.40
_QWEN_STRONG_CONFIDENCE = 0.45
_BAIDU_WEAK_CONFIDENCE = 0.35
_BAIDU_STRONG_CONFIDENCE = 0.45
_RUNNINGHUB_WEAK_CONFIDENCE = 0.34
_RUNNINGHUB_STRONG_CONFIDENCE = 0.40
_MICROSOFT_WEAK_CONFIDENCE = 0.38
_MICROSOFT_STRONG_CONFIDENCE = 0.45
_PIXVERSE_WEAK_CONFIDENCE = 0.36
_PIXVERSE_STRONG_CONFIDENCE = 0.42
_LUMA_WEAK_CONFIDENCE = 0.45
_LUMA_STRONG_CONFIDENCE = 0.52
_HEYGEN_WEAK_CONFIDENCE = 0.34
_HEYGEN_STRONG_CONFIDENCE = 0.39
_SYNC_LABS_WEAK_CONFIDENCE = 0.44
_SYNC_LABS_STRONG_CONFIDENCE = 0.50
_TIKTOK_AI_WEAK_CONFIDENCE = 0.42
_TIKTOK_AI_STRONG_CONFIDENCE = 0.48
_GENERIC_AI_LABEL_WEAK_CONFIDENCE = 0.48
_GENERIC_AI_LABEL_STRONG_CONFIDENCE = 0.55
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
_VIDU_RELATIVE_HEIGHTS = tuple(value / 1000 for value in range(34, 61, 3))
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
    feature_cache: dict[tuple[Any, ...], NDArray[Any]] = field(default_factory=dict, repr=False, compare=False)


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


def _dola_template() -> NDArray[Any]:
    """Return the synthetic Dola AI text silhouette drawn in OpenCV 4's Hershey font.

    The Hershey templates are frozen assets, not rendered here: OpenCV 5 draws the
    same ``putText`` call with smaller, heavier glyphs than the ones the detectors were
    calibrated on. ``scripts/render_video_hershey_templates.py`` is the recipe.
    """
    return _asset_template("video_dola_hershey_duplex.png")


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
    for text, asset_stem in (("KLING AI", "kling_ai"), ("KlingAI", "klingai")):
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
        # Frozen OpenCV 4 Hershey renders; see _dola_template for why.
        templates.extend(_asset_template(f"video_{asset_stem}_hershey_{font}.png") for font in ("simplex", "duplex"))
    return tuple(templates)


@lru_cache(maxsize=1)
def _kling_logo_template() -> NDArray[Any]:
    """Return a synthetic ring approximation of the Kling swirl."""
    template = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(template, (50, 50), 34, 255, 14, cv2.LINE_AA)
    return _crop_nonzero(template)


def _vidu_capsule(height: int, width: int, angle: float, stroke: int) -> NDArray[Any]:
    canvas = Image.new("L", (200, 200), 0)
    ImageDraw.Draw(canvas).rounded_rectangle(
        (100 - width // 2, 100 - height // 2, 100 + width // 2, 100 + height // 2),
        radius=width // 2,
        outline=255,
        width=stroke,
    )
    rotation = cv2.getRotationMatrix2D((100, 100), angle, 1.0)
    return _crop_nonzero(cv2.warpAffine(np.asarray(canvas, dtype=np.uint8), rotation, (200, 200)))


@lru_cache(maxsize=1)
def _vidu_template() -> NDArray[Any]:
    """Return a synthetic Vidu logo plus "Vidu AI" wordmark silhouette.

    The logo is two outlined capsules leaning into a V; the text is font-rendered.
    Proportions follow the measured 1080p mark (232 x 50 px: logo 65 x 50, text
    155 x 35 centered 12 px after it). No pixels come from an export.
    """
    canvas = np.zeros((100, 464), dtype=np.uint8)
    logo = np.zeros((100, 130), dtype=np.uint8)
    left = _vidu_capsule(96, 40, 24, 11)
    left = cv2.resize(left, (left.shape[1] * 100 // left.shape[0], 100))
    logo[:, : left.shape[1]] = np.maximum(logo[:, : left.shape[1]], left)
    right = _vidu_capsule(96, 40, -24, 11)
    right = cv2.resize(right, (right.shape[1] * 100 // right.shape[0], 100))
    offset = 130 - right.shape[1]
    logo[:, offset:] = np.maximum(logo[:, offset:], right)
    canvas[:, :130] = logo
    text = Image.new("L", (600, 140), 0)
    ImageDraw.Draw(text).text(
        (4, 10), "Vidu AI", font=_scalable_default_font(96), fill=255, stroke_width=1, stroke_fill=255
    )
    word = cv2.resize(_crop_nonzero(np.asarray(text, dtype=np.uint8)), (310, 70), interpolation=cv2.INTER_AREA)
    canvas[15:85, 154:464] = np.maximum(canvas[15:85, 154:464], word)
    return canvas


def _top_hat(gray: NDArray[Any]) -> NDArray[Any]:
    kernel = np.ones((7, 7), dtype=np.uint8)
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


@cache
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


def _doubao_template() -> NDArray[Any]:
    return _asset_template("doubao_alpha.png")


def _wan_template() -> NDArray[Any]:
    return _asset_template("wan_alpha.png")


@lru_cache(maxsize=1)
def _dreamina_template() -> NDArray[Any]:
    """Return a synthetic rounded-box + italic AI silhouette for Dreamina's top-left badge."""
    w, h = 216, 164
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((6, 6, w - 7, h - 7), radius=48, outline=255, width=8)
    txt = Image.new("L", (160, 110), 0)
    ImageDraw.Draw(txt).text((18, 8), "AI", font=_scalable_default_font(82), fill=255, stroke_width=2, stroke_fill=255)
    txt_arr = _crop_nonzero(np.asarray(txt, dtype=np.uint8))
    th, tw = txt_arr.shape[:2]
    shear = -0.24
    extra = round(abs(shear) * th)
    sheared = cv2.warpAffine(
        txt_arr,
        np.float32([[1.0, shear, extra], [0.0, 1.0, 0.0]]),
        (tw + extra, th),
        flags=cv2.INTER_LINEAR,
    )
    sheared = _crop_nonzero(sheared)
    target_h = 72
    target_w = max(1, round(sheared.shape[1] * target_h / sheared.shape[0]))
    glyph = cv2.resize(sheared, (target_w, target_h), interpolation=cv2.INTER_AREA)
    gx = (w - target_w) // 2 + 2
    gy = (h - target_h) // 2 + 2
    out = np.asarray(canvas, dtype=np.uint8).copy()
    out[gy : gy + target_h, gx : gx + target_w] = np.maximum(out[gy : gy + target_h, gx : gx + target_w], glyph)
    return _crop_nonzero(out)


@lru_cache(maxsize=1)
def _pixverse_template() -> NDArray[Any]:
    """Return a synthetic 'PixVerse.ai' top-right wordmark silhouette."""
    canvas = Image.new("L", (340, 80), 0)
    draw = ImageDraw.Draw(canvas)
    font = _scalable_default_font(52)
    draw.text((8, 8), "PixVerse.ai", font=font, fill=255, stroke_width=2, stroke_fill=255)
    return _crop_nonzero(np.asarray(canvas, dtype=np.uint8))


@lru_cache(maxsize=1)
def _luma_template() -> NDArray[Any]:
    """Return a synthetic Luma 2-facet isometric corner prism + 'Luma' silhouette."""
    h, w = 106, 456
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    draw.polygon([(0, 36), (38, 0), (38, 70), (0, 70)], fill=200)
    draw.polygon([(0, 70), (38, 70), (76, 87), (38, 105)], fill=255)
    font = _scalable_default_font(104)
    txt = Image.new("L", (400, 130), 0)
    ImageDraw.Draw(txt).text((0, 0), "Luma", font=font, fill=255, stroke_width=2, stroke_fill=255)
    txt_arr = cv2.resize(_crop_nonzero(np.asarray(txt, dtype=np.uint8)), (330, 88), interpolation=cv2.INTER_AREA)
    out = np.asarray(canvas, dtype=np.uint8).copy()
    out[12:100, 124:454] = np.maximum(out[12:100, 124:454], txt_arr)
    return _crop_nonzero(out)


@lru_cache(maxsize=1)
def _heygen_template() -> NDArray[Any]:
    """Return a synthetic two-line 'Hey [4-facet diamond] / Gen' corner badge silhouette."""
    w, h = 260, 142
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    font = _scalable_default_font(64)
    t_hey = Image.new("L", (200, 90), 0)
    ImageDraw.Draw(t_hey).text((0, 0), "Hey", font=font, fill=255, stroke_width=2, stroke_fill=255)
    a_hey = cv2.resize(_crop_nonzero(np.asarray(t_hey, dtype=np.uint8)), (142, 68), interpolation=cv2.INTER_AREA)
    t_gen = Image.new("L", (200, 90), 0)
    ImageDraw.Draw(t_gen).text((0, 0), "Gen", font=font, fill=255, stroke_width=2, stroke_fill=255)
    a_gen = cv2.resize(_crop_nonzero(np.asarray(t_gen, dtype=np.uint8)), (148, 56), interpolation=cv2.INTER_AREA)
    cx, cy, rx, ry, gap = 206, 34, 48, 31, 3
    draw.polygon([(cx - gap, cy - gap), (cx - rx, cy - gap), (cx - gap, cy - ry)], fill=235)
    draw.polygon([(cx + gap, cy - gap), (cx + rx, cy - gap), (cx + gap, cy - ry)], fill=200)
    draw.polygon([(cx - gap, cy + gap), (cx - rx, cy + gap), (cx - gap, cy + ry)], fill=200)
    draw.polygon([(cx + gap, cy + gap), (cx + rx, cy + gap), (cx + gap, cy + ry)], fill=235)
    out = np.asarray(canvas, dtype=np.uint8).copy()
    out[6:74, 4:146] = np.maximum(out[6:74, 4:146], a_hey)
    out[78:134, 48:196] = np.maximum(out[78:134, 48:196], a_gen)
    return _crop_nonzero(out)


@lru_cache(maxsize=1)
def _sync_labs_template() -> NDArray[Any]:
    """Return a synthetic Sync Labs lens-circle + 'sync.so' silhouette."""
    w, h = 380, 84
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    r, cx, cy = 34, 38, 40
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    for sx_f, sy_f, fill_val in ((0.96, 0.44, 0), (0.78, 0.26, 255), (0.62, 0.14, 0)):
        pts = [
            (cx + round(r * sx_f * math.cos(math.radians(d))), cy + round(r * sy_f * math.sin(math.radians(d))))
            for d in range(0, 360, 10)
        ]
        draw.polygon(pts, fill=fill_val)
    font = _scalable_default_font(60)
    draw.text((88, 4), "sync.so", font=font, fill=255, stroke_width=1, stroke_fill=255)
    return _crop_nonzero(np.asarray(canvas, dtype=np.uint8))


@lru_cache(maxsize=1)
def _tiktok_ai_template() -> NDArray[Any]:
    """Return a synthetic 'AI-generated' label silhouette for TikTok's bottom-left pill."""
    txt = Image.new("L", (360, 62), 0)
    draw = ImageDraw.Draw(txt)
    font = _scalable_default_font(40)
    draw.text((16, 10), "AI-generated", font=font, fill=255)
    return _crop_nonzero(np.asarray(txt, dtype=np.uint8))


@lru_cache(maxsize=1)
def _qwen_en_template() -> NDArray[Any]:
    """Return a synthetic 3-lobe triangular pinwheel + 'Qwen' silhouette."""
    w, h = 340, 92
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    cx, cy = 46.0, 46.0
    base_lobe = (
        (-16.0, -34.0),
        (-4.0, -34.0),
        (2.0, -23.0),
        (24.0, -23.0),
        (30.0, -12.0),
        (-10.0, -12.0),
        (-22.0, -24.0),
    )
    for k in range(3):
        ang = math.radians(120.0 * k)
        ca, sa = math.cos(ang), math.sin(ang)
        pts = [(round(cx + x * ca - y * sa), round(cy + x * sa + y * ca)) for x, y in base_lobe]
        draw.polygon(pts, fill=255)
    font = _scalable_default_font(66)
    draw.text((98, 8), "Qwen", font=font, fill=255, stroke_width=2, stroke_fill=255)
    return _crop_nonzero(np.asarray(canvas, dtype=np.uint8))


@lru_cache(maxsize=1)
def _template_sources() -> dict[str, NDArray[Any]]:
    """Return every immutable synthetic template by detector-local key."""
    sora_word, sora_icon = _sora_templates()
    veo_diamond, veo_text = _veo_templates()
    sources = {
        "sora-word": sora_word,
        "sora-icon": sora_icon,
        "veo-diamond": veo_diamond,
        "veo-text": veo_text,
        "dreamina": _dreamina_template(),
        "seedance": _seedance_template(),
        "dola": _dola_template(),
        "hailuo": _hailuo_template(),
        "vidu": _vidu_template(),
        "wan": _wan_template(),
        "jimeng": _asset_template("jimeng_alpha.png"),
        "jimeng_pill": _asset_template("jimeng_pill.png"),
        "yuanbao": _asset_template("yuanbao_alpha.png"),
        "qwen": _asset_template("qwen_alpha.png"),
        "qwen-en": _qwen_en_template(),
        "baidu": _asset_template("baidu_alpha.png"),
        "runninghub": _asset_template("runninghub_alpha.png"),
        "microsoft": _asset_template("microsoft_alpha.png"),
        "pixverse": _pixverse_template(),
        "luma": _luma_template(),
        "heygen": _heygen_template(),
        "sync_labs": _sync_labs_template(),
        "tiktok_ai": _tiktok_ai_template(),
        "generic_ai_label": _asset_template("generic_ai_label_alpha.png"),
        "kling-logo": _kling_logo_template(),
        # The Doubao video mark is the same "豆包AI生成" run the image engine
        # matches (its synthetic alpha asset), bottom-right on video frames.
        "doubao": _doubao_template(),
    }
    sources.update({f"kling-{index}": template for index, template in enumerate(_kling_templates())})
    return sources


@lru_cache(maxsize=2048)
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


@lru_cache(maxsize=512)
def _resized_raw_template(template_key: str, width: int, height: int) -> NDArray[Any]:
    return cv2.resize(_template_sources()[template_key], (width, height), interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=512)
def _resized_sobel_template(template_key: str, width: int, height: int) -> NDArray[Any]:
    raw = _resized_raw_template(template_key, width, height)
    return cv2.normalize(sobel_magnitude(raw), None, 0, 255, cv2.NORM_MINMAX)


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
            template, template_feature = _resized_template_feature(
                template_key,
                template_width,
                template_height,
                7,
            )
            scores = cv2.matchTemplate(feature, template_feature, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, location = cv2.minMaxLoc(scores)
            if confidence <= best_confidence:
                continue
            if icon_only:
                # Edge-only correlation also accepts diamonds and scene texture.
                # Confirm the full mascot silhouette, including its two eyes.
                x, y = location
                candidate = gray[y : y + template_height, x : x + template_width]
                shape_confidence = float(cv2.matchTemplate(candidate, template, cv2.TM_CCOEFF_NORMED)[0, 0])
                if shape_confidence < _SORA_ICON_SHAPE_FLOOR:
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
    patch = feature[location[1] : location[1] + template_height, location[0] : location[0] + template_width]
    if patch.size == 0 or int(patch.max()) < 8:
        return 0.0, None
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


def detect_dreamina_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate Dreamina's top-left rounded-box italic AI badge."""
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)
    frame_height, frame_width = image_bgr.shape[:2]
    gray = (prepared or _prepare_frame(image_bgr)).gray
    short = min(frame_height, frame_width)
    roi_w = min(frame_width, max(24, round(short * 0.12)))
    roi_h = min(frame_height, max(20, round(short * 0.10)))
    roi = gray[:roi_h, :roi_w]
    tmpl_base = _template_sources()["dreamina"]
    feat_cache: dict[int, NDArray[Any]] = {}
    best_conf = 0.0
    best_region: Region | None = None
    for rel_h in (0.030, 0.034, 0.038, 0.042, 0.046):
        th = max(12, round(short * rel_h))
        tw = max(14, round(tmpl_base.shape[1] * th / tmpl_base.shape[0]))
        if th >= roi_h or tw >= roi_w:
            continue
        kernel_size = max(3, round(th * 0.18) | 1)
        _tmpl, t_feat = _resized_template_feature("dreamina", tw, th, kernel_size)
        feat = feat_cache.get(kernel_size)
        if feat is None:
            feat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, np.ones((kernel_size, kernel_size), np.uint8))
            feat_cache[kernel_size] = feat
        scores = cv2.matchTemplate(feat, t_feat, cv2.TM_CCOEFF_NORMED)
        _, conf, _, loc = cv2.minMaxLoc(scores)
        x, y = loc
        patch = feat[y : y + th, x : x + tw]
        if patch.size == 0 or int(patch.max()) < 6:
            continue
        if not (short * 0.006 <= x <= short * 0.035 and short * 0.006 <= y <= short * 0.035):
            continue
        if conf > best_conf:
            best_conf = float(conf)
            best_region = (x, y, tw, th)
    return FrameLocalization(frame_index, best_conf, best_region)


def detect_wan_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate Alibaba Wan's bottom-right logo + AIGenerated badge."""
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)
    frame_height, frame_width = image_bgr.shape[:2]
    prep = prepared or _prepare_frame(image_bgr)
    gray, scale = prep.normalized_gray, prep.normalized_scale
    nh, nw = gray.shape[:2]
    nshort = min(nh, nw)
    sx, sy = round(nw * 0.72), round(nh * 0.84)
    roi = gray[sy:, sx:]
    tmpl_base = _template_sources()["wan"]
    feat_cache: dict[int, NDArray[Any]] = {}
    best_conf = 0.0
    best_region: Region | None = None
    for rel_h in tuple(v / 1000 for v in range(36, 65, 4)):
        th = max(10, round(nshort * rel_h))
        tw = max(14, round(tmpl_base.shape[1] * th / tmpl_base.shape[0]))
        if th >= roi.shape[0] or tw >= roi.shape[1]:
            continue
        kernel_size = max(3, round(th * 0.22) | 1)
        _tmpl, t_feat = _resized_template_feature("wan", tw, th, kernel_size)
        feat = feat_cache.get(kernel_size)
        if feat is None:
            feat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, np.ones((kernel_size, kernel_size), np.uint8))
            feat_cache[kernel_size] = feat
        scores = cv2.matchTemplate(feat, t_feat, cv2.TM_CCOEFF_NORMED)
        _, conf, _, loc = cv2.minMaxLoc(scores)
        x, y = sx + loc[0], sy + loc[1]
        patch = feat[loc[1] : loc[1] + th, loc[0] : loc[0] + tw]
        if patch.size == 0 or int(patch.max()) < 8:
            continue
        if (nw - (x + tw)) > nshort * 0.025 or (nh - (y + th)) > nshort * 0.025:
            continue
        if conf > best_conf:
            best_conf = float(conf)
            best_region = _restore_region(
                (x, y, tw, th),
                scale=scale,
                frame_width=frame_width,
                frame_height=frame_height,
            )
    return FrameLocalization(frame_index, best_conf, best_region)


def _sobel_norm_u8(gray: NDArray[Any]) -> NDArray[Any]:
    mag = sobel_magnitude(gray)
    denom = max(float(mag.max()), 48.0)
    return np.clip(mag * (255.0 / denom), 0.0, 255.0).astype(np.float32)


def _detect_corner_asset_mark(
    image_bgr: NDArray[Any],
    *,
    frame_index: int,
    prepared: _PreparedFrame | None,
    key: str,
    corner: Literal["br", "tl", "tr"],
    basis: Literal["short", "width", "long"],
    rel_sizes: tuple[float, ...],
    max_rx: float,
    max_ry: float,
    min_peak: int = 8,
    rival_keys: tuple[str, ...] = (),
    extend_right_margin: float | None = None,
    roi_origin_x: float | None = None,
    kernel_factor: float = 0.22,
    min_kernel: int = 3,
) -> FrameLocalization:
    """Shared normalized-frame corner matcher for calibrated alpha-silhouette marks."""
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)
    frame_height, frame_width = image_bgr.shape[:2]
    prep = prepared or _prepare_frame(image_bgr)
    gray, scale = prep.normalized_gray, prep.normalized_scale
    nh, nw = gray.shape[:2]
    nshort, nlong = min(nh, nw), max(nh, nw)
    dim = {"short": nshort, "width": nw, "long": nlong}[basis]
    templates = _template_sources()
    tmpl_base = templates[key]
    if corner == "br":
        sx, sy = round(nw * 0.62), round(nh * 0.78)
        roi = gray[sy:, sx:]
    elif corner == "tl":
        sx, sy = 0, 0
        roi = gray[: max(20, round(nh * 0.18)), : max(32, round(nw * 0.45))]
    else:
        sx, sy = round(nw * (0.72 if roi_origin_x is None else roi_origin_x)), 0
        roi = gray[: max(24 if roi_origin_x is not None else 20, round(nh * 0.16)), sx:]

    feat_cache: dict[int, NDArray[Any]] = {}
    best_conf = 0.0
    best_region: Region | None = None
    for rel in rel_sizes:
        if basis in {"width", "long"}:
            tw = max(14, round(dim * rel))
            th = max(6, round(tmpl_base.shape[0] * tw / tmpl_base.shape[1]))
        else:
            th = max(8, round(dim * rel))
            tw = max(12, round(tmpl_base.shape[1] * th / tmpl_base.shape[0]))
        if th >= roi.shape[0] or tw >= roi.shape[1]:
            continue
        kernel_size = max(min_kernel, round(th * kernel_factor) | 1)
        if key == "microsoft":
            t_res = _resized_raw_template("microsoft", tw, th)
            scores = cv2.matchTemplate(roi, t_res, cv2.TM_CCOEFF_NORMED)
            _, conf, _, loc = cv2.minMaxLoc(scores)
            lx, ly = loc
            patch_raw = roi[ly : ly + th, lx : lx + tw]
            fg = patch_raw[t_res > 160]
            bg = patch_raw[t_res < 80]
            if fg.size == 0 or bg.size == 0 or (float(fg.mean()) - float(bg.mean())) < 22.0:
                continue
            patch_feat: NDArray[Any] | None = None
        else:
            _tmpl, t_feat = _resized_template_feature(key, tw, th, kernel_size)
            cache_key = ("corner_tophat", corner, sx, kernel_size)
            feat = prep.feature_cache.get(cache_key)
            if feat is None:
                feat = feat_cache.get(kernel_size)
                if feat is None:
                    feat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, np.ones((kernel_size, kernel_size), np.uint8))
                    feat_cache[kernel_size] = feat
                prep.feature_cache[cache_key] = feat
            scores = cv2.matchTemplate(feat, t_feat, cv2.TM_CCOEFF_NORMED)
            _, conf, _, loc = cv2.minMaxLoc(scores)
            lx, ly = loc
            patch_feat = feat[ly : ly + th, lx : lx + tw]
            if patch_feat.size == 0 or int(patch_feat.max()) < min_peak:
                continue

        x, y = sx + lx, sy + ly
        if corner == "br":
            if (nw - (x + tw)) > nw * max_rx or (nh - (y + th)) > nh * max_ry:
                continue
        elif corner == "tl":
            if x > nw * max_rx or y > nh * max_ry:
                continue
        elif (nw - (x + tw)) > nw * max_rx or y > nh * max_ry:
            continue

        if rival_keys and patch_feat is not None and conf >= 0.32 and conf > best_conf:
            rival_won = False
            for r_key in rival_keys:
                _, r_feat = _resized_template_feature(r_key, tw, th, kernel_size)
                r_conf = float(cv2.matchTemplate(patch_feat, r_feat, cv2.TM_CCOEFF_NORMED)[0, 0])
                if r_conf >= conf - 0.02:
                    rival_won = True
                    break
            if rival_won:
                continue

        if conf > best_conf:
            best_conf = float(conf)
            reg_w = tw
            if extend_right_margin is not None:
                target_x1 = nw - max(4, round(nw * extend_right_margin))
                if target_x1 > x + tw:
                    reg_w = target_x1 - x
            best_region = _restore_region(
                (x, y, reg_w, th),
                scale=scale,
                frame_width=frame_width,
                frame_height=frame_height,
            )
    return FrameLocalization(frame_index, best_conf, best_region)


def detect_jimeng_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate ByteDance Jimeng's bottom-right 即梦AI wordmark."""
    return _detect_corner_asset_mark(
        image_bgr,
        frame_index=frame_index,
        prepared=prepared,
        key="jimeng",
        corner="br",
        basis="width",
        rel_sizes=(0.16, 0.18, 0.20, 0.22, 0.25),
        max_rx=0.035,
        max_ry=0.035,
        rival_keys=("doubao",),
    )


def detect_jimeng_pill_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate ByteDance Jimeng's top-left AI生成 rounded pill."""
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)
    frame_height, frame_width = image_bgr.shape[:2]
    prep = prepared or _prepare_frame(image_bgr)
    gray, scale = prep.normalized_gray, prep.normalized_scale
    nh, nw = gray.shape[:2]
    roi = gray[: max(20, round(nh * 0.16)), : max(36, round(nw * 0.36))]
    roi_grad = prep.feature_cache.get(("pill_sobel",))
    if roi_grad is None:
        roi_grad = _sobel_norm_u8(roi)
        prep.feature_cache[("pill_sobel",)] = roi_grad
    sil = _template_sources()["jimeng_pill"]
    best_conf = 0.0
    best_region: Region | None = None
    for wf in (0.135, 0.150, 0.161, 0.175, 0.190):
        tw = max(20, round(nw * wf))
        th = max(10, round(tw * sil.shape[0] / sil.shape[1]))
        if th >= roi.shape[0] or tw >= roi.shape[1]:
            continue
        t_grad = _resized_sobel_template("jimeng_pill", tw, th)
        _, conf, _, loc = cv2.minMaxLoc(cv2.matchTemplate(roi_grad, t_grad, cv2.TM_CCOEFF_NORMED))
        lx, ly = loc
        if lx > nw * 0.04 or ly > nh * 0.04:
            continue
        if conf > best_conf:
            best_conf = float(conf)
            best_region = _restore_region(
                (lx, ly, tw, th),
                scale=scale,
                frame_width=frame_width,
                frame_height=frame_height,
            )
    return FrameLocalization(frame_index, best_conf, best_region)


def detect_yuanbao_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate Tencent Yuanbao's bottom-right 元宝 / AI生成 mark."""
    return _detect_corner_asset_mark(
        image_bgr,
        frame_index=frame_index,
        prepared=prepared,
        key="yuanbao",
        corner="br",
        basis="short",
        rel_sizes=(0.038, 0.044, 0.050, 0.056),
        max_rx=0.040,
        max_ry=0.040,
    )


def detect_qwen_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate Alibaba Cloud Qwen's bottom-right 通义 or international Qwen mark."""
    prep = prepared or (_prepare_frame(image_bgr) if image_bgr.size > 0 else None)
    cn_loc = _detect_corner_asset_mark(
        image_bgr,
        frame_index=frame_index,
        prepared=prep,
        key="qwen",
        corner="br",
        basis="short",
        rel_sizes=(0.030, 0.036, 0.042, 0.048, 0.054),
        max_rx=0.040,
        max_ry=0.040,
        rival_keys=("doubao", "kling-1"),
    )
    en_loc = _detect_corner_asset_mark(
        image_bgr,
        frame_index=frame_index,
        prepared=prep,
        key="qwen-en",
        corner="br",
        basis="short",
        rel_sizes=(0.042, 0.048, 0.054, 0.060, 0.066),
        max_rx=0.060,
        max_ry=0.050,
    )
    if en_loc.confidence >= 0.48 and en_loc.confidence > cn_loc.confidence:
        return en_loc
    return cn_loc


def detect_pixverse_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate PixVerse's top-right 'PixVerse.ai' wordmark."""
    return _detect_corner_asset_mark(
        image_bgr,
        frame_index=frame_index,
        prepared=prepared,
        key="pixverse",
        corner="tr",
        basis="short",
        rel_sizes=(0.022, 0.026, 0.030, 0.035, 0.040, 0.046),
        max_rx=0.080,
        max_ry=0.070,
        min_peak=10,
        roi_origin_x=0.60,
        kernel_factor=0.25,
    )


def detect_luma_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate Luma AI's top-left isometric prism + 'Luma [AI]' semi-transparent mark."""
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)
    frame_height, frame_width = image_bgr.shape[:2]
    prep = prepared or _prepare_frame(image_bgr)
    gray, scale = prep.normalized_gray, prep.normalized_scale
    nh, nw = gray.shape[:2]
    max_luma_w = min(nw, round(nw * 0.12) + round(nh * 0.13 * 4.5))
    roi = gray[: max(30, round(nh * 0.20)), : max(64, max_luma_w)]
    tmpl_base = _template_sources()["luma"]
    kernel_size = max(5, round(nh * 0.046) | 1)
    feat = cv2.morphologyEx(
        roi, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    )
    best_conf = 0.0
    best_region: Region | None = None
    for rh in (0.095, 0.103, 0.110, 0.118, 0.126):
        th = max(18, round(nh * rh))
        tw = max(40, round(tmpl_base.shape[1] * th / tmpl_base.shape[0]))
        if th >= roi.shape[0] or tw >= roi.shape[1]:
            continue
        _tmpl, t_feat = _resized_template_feature("luma", tw, th, kernel_size)
        _, conf, _, loc = cv2.minMaxLoc(cv2.matchTemplate(feat, t_feat, cv2.TM_CCOEFF_NORMED))
        lx, ly = loc
        patch = feat[ly : ly + th, lx : lx + tw]
        if patch.size == 0 or int(patch.max()) < 8:
            continue
        if not (nw * 0.03 <= lx <= nw * 0.11 and nh * 0.015 <= ly <= nh * 0.075):
            continue
        if conf > best_conf:
            best_conf = float(conf)
            ext_w = min(nw - lx, round(tw * 1.48))
            best_region = _restore_region(
                (lx, ly, ext_w, th),
                scale=scale,
                frame_width=frame_width,
                frame_height=frame_height,
            )
    return FrameLocalization(frame_index, best_conf, best_region)


def detect_heygen_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate HeyGen's bottom-right two-line 'Hey [diamond] / Gen' corner badge."""
    return _detect_corner_asset_mark(
        image_bgr,
        frame_index=frame_index,
        prepared=prepared,
        key="heygen",
        corner="br",
        basis="width",
        rel_sizes=(0.16, 0.19, 0.22, 0.25, 0.28),
        max_rx=0.10,
        max_ry=0.06,
        min_peak=12,
        kernel_factor=0.28,
        min_kernel=5,
    )


def detect_sync_labs_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate Sync Labs' top-left lens-circle + 'sync.so' mark."""
    return _detect_corner_asset_mark(
        image_bgr,
        frame_index=frame_index,
        prepared=prepared,
        key="sync_labs",
        corner="tl",
        basis="short",
        rel_sizes=(0.028, 0.033, 0.038, 0.044, 0.050, 0.056),
        max_rx=0.090,
        max_ry=0.090,
        min_peak=10,
    )


def detect_tiktok_ai_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate TikTok's bottom-left 'AI-generated' rounded pill."""
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)
    frame_height, frame_width = image_bgr.shape[:2]
    prep = prepared or _prepare_frame(image_bgr)
    gray, scale = prep.normalized_gray, prep.normalized_scale
    nh, nw = gray.shape[:2]
    sy = round(nh * 0.86)
    roi = gray[sy:, : max(48, round(nw * 0.42))]
    tmpl_base = _template_sources()["tiktok_ai"]
    feat_cache: dict[int, NDArray[Any]] = {}
    best_conf = 0.0
    best_region: Region | None = None
    for rw in (0.14, 0.16, 0.18, 0.20, 0.22, 0.25):
        pw = max(28, round(nw * rw))
        tw = max(22, round(pw * 0.84))
        th = max(7, round(tmpl_base.shape[0] * tw / tmpl_base.shape[1]))
        if th >= roi.shape[0] or tw >= roi.shape[1]:
            continue
        kernel_size = max(3, round(th * 0.30) | 1)
        t_raw, t_feat = _resized_template_feature("tiktok_ai", tw, th, kernel_size)
        feat = feat_cache.get(kernel_size)
        if feat is None:
            feat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, np.ones((kernel_size, kernel_size), np.uint8))
            feat_cache[kernel_size] = feat
        _, conf, _, loc = cv2.minMaxLoc(cv2.matchTemplate(feat, t_feat, cv2.TM_CCOEFF_NORMED))
        lx, ly = loc
        patch = feat[ly : ly + th, lx : lx + tw]
        if patch.size == 0 or int(patch.max()) < 12:
            continue
        raw_patch = roi[ly : ly + th, lx : lx + tw]
        fg = raw_patch[t_raw > 140]
        bg = raw_patch[t_raw < 40]
        if fg.size == 0 or bg.size == 0 or (float(fg.mean()) - float(bg.mean())) < 12.0:
            continue
        x, y = lx, sy + ly
        if x > nw * 0.09 or (nh - (y + th)) > nh * 0.06:
            continue
        if conf > best_conf:
            best_conf = float(conf)
            pad_x = max(2, round((pw - tw) * 0.55))
            pad_y = max(2, round(th * 0.48))
            bx = max(0, x - pad_x)
            by = max(0, y - pad_y)
            bw = min(nw - bx, tw + 2 * pad_x)
            bh = min(nh - by, th + 2 * pad_y)
            best_region = _restore_region(
                (bx, by, bw, bh),
                scale=scale,
                frame_width=frame_width,
                frame_height=frame_height,
            )
    return FrameLocalization(frame_index, best_conf, best_region)


def detect_baidu_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate Baidu's bottom-right 百度 AI生成 mark."""
    return _detect_corner_asset_mark(
        image_bgr,
        frame_index=frame_index,
        prepared=prepared,
        key="baidu",
        corner="br",
        basis="short",
        rel_sizes=(0.040, 0.046, 0.052),
        max_rx=0.140,
        max_ry=0.040,
        rival_keys=("doubao", "qwen"),
        extend_right_margin=0.010,
    )


def detect_runninghub_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate RunningHub's top-left RunningHub AI生成 mark."""
    return _detect_corner_asset_mark(
        image_bgr,
        frame_index=frame_index,
        prepared=prepared,
        key="runninghub",
        corner="tl",
        basis="width",
        rel_sizes=(0.26, 0.29, 0.32, 0.35),
        max_rx=0.025,
        max_ry=0.020,
        min_peak=6,
    )


def detect_microsoft_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate Microsoft's top-right Made with AI badge."""
    return _detect_corner_asset_mark(
        image_bgr,
        frame_index=frame_index,
        prepared=prepared,
        key="microsoft",
        corner="tr",
        basis="long",
        rel_sizes=(0.12, 0.136, 0.152, 0.17, 0.19),
        max_rx=0.025,
        max_ry=0.025,
    )


def detect_generic_ai_label_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the generic Chinese GB 45438-2025 bottom-right AI生成 mark."""
    return _detect_corner_asset_mark(
        image_bgr,
        frame_index=frame_index,
        prepared=prepared,
        key="generic_ai_label",
        corner="br",
        basis="width",
        rel_sizes=(0.135, 0.155, 0.172, 0.195, 0.215),
        max_rx=0.030,
        max_ry=0.030,
        rival_keys=("jimeng", "doubao", "yuanbao", "qwen", "baidu"),
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


def _white_fraction(roi: NDArray[Any]) -> float:
    """Share of bright, low-saturation pixels, the color of the white video labels.

    A light UI card or navigation bar with dark text has a high share of white
    background pixels (``median >= 180``), and its narrow inter-letter gaps can
    correlate with a morphological top-hat template. In that bright-background
    inverted polarity the dark excursion below the patch median exceeds the
    bright excursion above it, whereas a white label over a non-white scene
    (including two-tone midtone-plus-shadow backgrounds where ``median < 180``)
    keeps its white-stroke share.
    """
    if roi.size == 0:
        return 0.0
    if roi.ndim == 2:
        value = roi
        bright = value >= 180
        white = bright
    else:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        bright = value >= 180
        white = (hsv[:, :, 1] <= 55) & bright
    if int(np.count_nonzero(bright)) * 2 >= value.size:
        median = float(np.median(value))
        if median >= 180.0 and float(value.mean()) < median:
            return 0.0
    return float(np.mean(white))


def _reaches_kling_edge(region: Region, frame_width: int, frame_height: int) -> bool:
    """Whether a Kling candidate box reaches the bottom-right frame edge."""
    x, y, width, height = region
    return x + width >= frame_width * 0.96 and y + height >= frame_height * 0.94


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
    if _white_fraction(image_bgr[y : y + height, x : x + width]) < _HAILUO_MIN_WHITE_FRACTION:
        return FrameLocalization(frame_index, 0.0, None)
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


def detect_vidu_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the strongest fixed Vidu logo-plus-wordmark candidate."""
    return _detect_fixed_mark(
        image_bgr,
        "vidu",
        relative_heights=_VIDU_RELATIVE_HEIGHTS,
        search_origin=(0.70, 0.84),
        kernel_fraction=0.18,
        normalized=None if prepared is None else (prepared.normalized_gray, prepared.normalized_scale),
        frame_index=frame_index,
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
        if candidate.region is not None and _reaches_kling_edge(candidate.region, frame_width, frame_height)
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
        logo_region = _bounded_region(
            logo_x - round(logo_height * 0.2),
            logo_y - round(logo_height * 0.25),
            round(logo_height * 7.8),
            round(logo_height * 1.5),
            frame_width=frame_width,
            frame_height=frame_height,
        )
        # The swirl arm must reach the same frame edge as the text arm: a swirl
        # match on wood grain 11% short of the right edge covered every frame of
        # a Higgsfield Cinema Studio clip (2026-09-25).
        if _reaches_kling_edge(logo_region, frame_width, frame_height):
            logo_candidate = FrameLocalization(frame_index, logo.confidence, logo_region)

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
    if _white_fraction(image_bgr[y : y + height, x : x + width]) < _KLING_MIN_WHITE_FRACTION:
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


def _region_intersection(left: Region, right: Region) -> int:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return max(0, min(lx + lw, rx + rw) - max(lx, rx)) * max(0, min(ly + lh, ry + rh) - max(ly, ry))


def _region_iou(left: Region, right: Region) -> float:
    intersection = _region_intersection(left, right)
    union = left[2] * left[3] + right[2] * right[3] - intersection
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
    "dreamina": VisibleMarkPolicy(
        weak_floor=_DREAMINA_WEAK_CONFIDENCE,
        strong_floor=_DREAMINA_STRONG_CONFIDENCE,
        transition_floor=0.34,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.12,
        mask_style="box",
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
    "vidu": VisibleMarkPolicy(
        weak_floor=_VIDU_WEAK_CONFIDENCE,
        strong_floor=_VIDU_STRONG_CONFIDENCE,
        transition_floor=0.45,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.12,
        mask_style="box",
        # No provenance_weak_floor: a ShengShu TC260 label confirms the vendor
        # (has_vidu_video_provenance) but does not lower the measured bar.
    ),
    "wan": VisibleMarkPolicy(
        weak_floor=_WAN_WEAK_CONFIDENCE,
        strong_floor=_WAN_STRONG_CONFIDENCE,
        transition_floor=0.42,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.12,
        mask_style="box",
    ),
    "jimeng": VisibleMarkPolicy(
        weak_floor=_JIMENG_WEAK_CONFIDENCE,
        strong_floor=_JIMENG_STRONG_CONFIDENCE,
        transition_floor=0.38,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.12,
        mask_style="box",
    ),
    "jimeng_pill": VisibleMarkPolicy(
        weak_floor=_JIMENG_PILL_WEAK_CONFIDENCE,
        strong_floor=_JIMENG_PILL_STRONG_CONFIDENCE,
        transition_floor=0.24,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.24,
        mask_style="box",
    ),
    "qwen": VisibleMarkPolicy(
        weak_floor=_QWEN_WEAK_CONFIDENCE,
        strong_floor=_QWEN_STRONG_CONFIDENCE,
        transition_floor=0.35,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.12,
        mask_style="box",
    ),
    "yuanbao": VisibleMarkPolicy(
        weak_floor=_YUANBAO_WEAK_CONFIDENCE,
        strong_floor=_YUANBAO_STRONG_CONFIDENCE,
        transition_floor=0.34,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.12,
        mask_style="box",
    ),
    "baidu": VisibleMarkPolicy(
        weak_floor=_BAIDU_WEAK_CONFIDENCE,
        provenance_weak_floor=0.33,
        strong_floor=_BAIDU_STRONG_CONFIDENCE,
        transition_floor=0.30,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.12,
        mask_style="box",
    ),
    "runninghub": VisibleMarkPolicy(
        weak_floor=_RUNNINGHUB_WEAK_CONFIDENCE,
        strong_floor=_RUNNINGHUB_STRONG_CONFIDENCE,
        transition_floor=0.28,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.12,
        mask_style="box",
    ),
    "microsoft": VisibleMarkPolicy(
        weak_floor=_MICROSOFT_WEAK_CONFIDENCE,
        strong_floor=_MICROSOFT_STRONG_CONFIDENCE,
        transition_floor=0.32,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.14,
        mask_style="box",
    ),
    "pixverse": VisibleMarkPolicy(
        weak_floor=_PIXVERSE_WEAK_CONFIDENCE,
        strong_floor=_PIXVERSE_STRONG_CONFIDENCE,
        transition_floor=0.30,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.16,
        mask_style="box",
    ),
    "luma": VisibleMarkPolicy(
        weak_floor=_LUMA_WEAK_CONFIDENCE,
        strong_floor=_LUMA_STRONG_CONFIDENCE,
        transition_floor=0.36,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.14,
        mask_style="box",
        accepts_provenance=False,
    ),
    "heygen": VisibleMarkPolicy(
        weak_floor=_HEYGEN_WEAK_CONFIDENCE,
        strong_floor=_HEYGEN_STRONG_CONFIDENCE,
        transition_floor=0.29,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.85,
        padding_fraction=0.14,
        mask_style="box",
        accepts_provenance=False,
    ),
    "sync_labs": VisibleMarkPolicy(
        weak_floor=_SYNC_LABS_WEAK_CONFIDENCE,
        strong_floor=_SYNC_LABS_STRONG_CONFIDENCE,
        transition_floor=0.36,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.16,
        mask_style="box",
    ),
    "tiktok_ai": VisibleMarkPolicy(
        weak_floor=_TIKTOK_AI_WEAK_CONFIDENCE,
        strong_floor=_TIKTOK_AI_STRONG_CONFIDENCE,
        transition_floor=0.35,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.16,
        mask_style="box",
    ),
    "generic_ai_label": VisibleMarkPolicy(
        weak_floor=_GENERIC_AI_LABEL_WEAK_CONFIDENCE,
        strong_floor=_GENERIC_AI_LABEL_STRONG_CONFIDENCE,
        transition_floor=0.40,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
        padding_fraction=0.12,
        mask_style="box",
        accepts_provenance=False,
    ),
    "kling": VisibleMarkPolicy(
        weak_floor=_KLING_WEAK_CONFIDENCE,
        strong_floor=_KLING_STRONG_CONFIDENCE,
        transition_floor=0.30,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        # Kling's per-frame score does not separate texture from the label: over
        # 1008 local videos the top clip, an OpenArt export with no Kling mark,
        # scored 0.81 against 0.63 for the real overlay, and none of 68 inspected
        # clips between 0.24 and 0.81 showed one (2026-09-25). The overlay's box
        # is identical on every frame, while texture matches drift a few pixels, so
        # a run must stay anchored far tighter than the shared 0.80.
        anchor_iou=0.95,
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
    #
    # The carry stops ``min_stable_frames`` after the last frame that still shows the
    # mark. China's labeling rules (CAC Measures, Article 4) require a video label only
    # on the opening frames; without this bound a label seen for 2 s had its box
    # inpainted over the whole clip (measured 2026-09-23: 240 of 240 frames for a
    # 48-frame label on every policy with cover_after_confirmation). No committed
    # provider clip relied on a tail carry longer than zero frames.
    confirmed_positions = [position for position, region in enumerate(accepted) if region is not None]
    if (provenance or cover_after_confirmation) and confirmed_positions:
        confirmed_regions = [accepted[position] for position in confirmed_positions]
        carry_position = confirmed_positions[0]
        last_supported = confirmed_positions[-1]
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
                last_supported = max(last_supported, position)
                continue
            if position > last_supported + min_stable_frames:
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
    capture = open_video_capture(source)
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
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if collect_timestamps:
                timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000)
            if frame.shape[:2] != (height, width):
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
    finally:
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
                    "ffprobe/decoder frame-count mismatch for %s: timestamps=%s decoded=%s; using decoder timestamps",
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
    detectors = {
        "sora": detect_sora_frame,
        "veo": detect_veo_frame,
        "dreamina": detect_dreamina_frame,
        "jimeng": detect_jimeng_frame,
        "jimeng_pill": detect_jimeng_pill_frame,
        "seedance": detect_seedance_frame,
        "doubao": detect_doubao_frame,
        "dola": detect_dola_frame,
        "hailuo": detect_hailuo_frame,
        "vidu": detect_vidu_frame,
        "wan": detect_wan_frame,
        "qwen": detect_qwen_frame,
        "yuanbao": detect_yuanbao_frame,
        "baidu": detect_baidu_frame,
        "runninghub": detect_runninghub_frame,
        "microsoft": detect_microsoft_frame,
        "pixverse": detect_pixverse_frame,
        "luma": detect_luma_frame,
        "heygen": detect_heygen_frame,
        "sync_labs": detect_sync_labs_frame,
        "tiktok_ai": detect_tiktok_ai_frame,
        "generic_ai_label": detect_generic_ai_label_frame,
        "kling": detect_kling_frame,
    }
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
        pix_fmt: str = "bgr24",
    ) -> None:
        import av

        self._av = av
        self._time_base = time_base
        self._start_pts = start_pts
        self._pix_fmt = pix_fmt
        self._origin: float | None = None
        self._container = av.open(pipe, mode="w", format="nut")
        self._stream = self._container.add_stream("rawvideo", rate=None)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = pix_fmt
        self._stream.time_base = time_base
        self._stream.codec_context.time_base = time_base

    def write(self, frame_bgr: NDArray[Any], timestamp: float) -> None:
        """Mux one contiguous BGR/BGR48LE frame at its source timeline timestamp."""
        if self._origin is None:
            self._origin = timestamp
        frame = self._av.VideoFrame.from_ndarray(
            np.ascontiguousarray(frame_bgr),
            format=self._pix_fmt,
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


def _normalize_frame_regions(entry: Region | Sequence[Region | None] | None) -> tuple[Region | None, ...]:
    if entry is None:
        return ()
    if len(entry) == 4 and all(isinstance(v, int) for v in entry):
        return (entry,)  # type: ignore[return-value]
    return tuple(entry)  # type: ignore[arg-type]


def _masks_for_regions(
    frame_bgr: NDArray[Any],
    frame_regions: tuple[Region | None, ...],
    *,
    padding_fraction: float,
    mask_style: Literal["box", "veo"],
    slot_padding_fractions: Sequence[float] | None = None,
    slot_mask_styles: Sequence[Literal["box", "veo"]] | None = None,
    dilate_px: int | None = None,
) -> NDArray[Any] | None:
    present = [reg for reg in frame_regions if reg is not None]
    if not present:
        return None
    height, width = frame_bgr.shape[:2]
    if dilate_px is not None:
        from remove_ai_watermarks.region_eraser import boxes_to_mask

        return boxes_to_mask((height, width), present, dilate=max(0, int(dilate_px)))
    combined = np.zeros((height, width), dtype=np.uint8)
    for slot_idx, reg in enumerate(frame_regions):
        if reg is None:
            continue
        pad_f = (
            slot_padding_fractions[slot_idx]
            if slot_padding_fractions is not None and slot_idx < len(slot_padding_fractions)
            else padding_fraction
        )
        style: Literal["box", "veo"] = (
            slot_mask_styles[slot_idx]
            if slot_mask_styles is not None and slot_idx < len(slot_mask_styles)
            else mask_style
        )
        combined = np.maximum(
            combined,
            _mask_for_region(frame_bgr, reg, padding_fraction=pad_f, mask_style=style),
        )
    return combined


def _precalibrate_stationary_mattes(
    source: Path,
    normalized_regions: list[tuple[Region | None, ...]],
    *,
    width: int,
    height: int,
    padding_fraction: float,
    slot_padding_fractions: Sequence[float] | None = None,
    dilate_px: int | None = None,
    high_depth: bool = False,
) -> tuple[ClipAlphaMatte, ...]:
    """Estimate self-calibrated YCrCb alpha mattes for fixed watermark/erase boxes."""
    max_regs = max((len(regs) for regs in normalized_regions), default=0)
    if max_regs == 0:
        return ()

    candidate_bounds: list[tuple[tuple[int, int, int, int], list[int]]] = []
    for slot in range(max_regs):
        slot_pairs = [
            (idx, reg)
            for idx, regs in enumerate(normalized_regions)
            if len(regs) > slot and (reg := regs[slot]) is not None
        ]
        if len(slot_pairs) < 12:
            continue
        first_reg = slot_pairs[0][1]
        anchored_pairs = [(idx, reg) for idx, reg in slot_pairs if _region_iou(reg, first_reg) >= 0.85]
        if len(anchored_pairs) < 12:
            continue
        anchored = [idx for idx, _reg in anchored_pairs]
        valid_boxes = [reg for _idx, reg in anchored_pairs]
        xs0 = [b[0] for b in valid_boxes]
        ys0 = [b[1] for b in valid_boxes]
        xs1 = [b[0] + b[2] for b in valid_boxes]
        ys1 = [b[1] + b[3] for b in valid_boxes]
        med_h = int(np.median([b[3] for b in valid_boxes]))
        slot_pad = (
            slot_padding_fractions[slot]
            if slot_padding_fractions is not None and slot < len(slot_padding_fractions)
            else padding_fraction
        )
        pad = max(0, int(dilate_px)) if dilate_px is not None else max(4, round(med_h * slot_pad))
        bounds = (
            max(0, min(xs0) - pad),
            max(0, min(ys0) - pad),
            min(width, max(xs1) + pad),
            min(height, max(ys1) + pad),
        )
        candidate_bounds.append((bounds, anchored))

    if not candidate_bounds:
        return ()

    max_samples = 36
    slot_samples: list[list[int]] = []
    union_indices: set[int] = set()
    for _bounds, anchored in candidate_bounds:
        if len(anchored) > max_samples:
            step_idx = np.linspace(0, len(anchored) - 1, max_samples, dtype=int)
            picked = [anchored[int(i)] for i in step_idx]
        else:
            picked = list(anchored)
        slot_samples.append(picked)
        union_indices.update(picked)
    max_needed = max(union_indices)

    decoded_by_idx: dict[int, NDArray[Any]] = {}
    cap: Any = (
        open_video_capture_16bit(source, width=width, height=height) if high_depth else open_video_capture(source)
    )
    if not cap.isOpened():
        return ()
    has_grab = hasattr(cap, "grab") and hasattr(cap, "retrieve")
    try:
        for idx in range(max_needed + 1):
            if has_grab and idx not in union_indices:
                if not cap.grab():
                    break
                continue
            ok, raw = cap.read()
            if not ok or raw is None:
                break
            if idx in union_indices:
                decoded_by_idx[idx] = raw
    finally:
        cap.release()

    mattes: list[ClipAlphaMatte] = []
    for (bounds, _anchored), picked in zip(candidate_bounds, slot_samples, strict=True):
        seq = [decoded_by_idx[i] for i in picked if i in decoded_by_idx]
        matte = estimate_clip_alpha_matte(seq, bounds)
        if matte is not None:
            mattes.append(matte)
    return tuple(mattes)


def _split_disjoint_masks(mask: NDArray[Any], *, multi_region: bool) -> list[NDArray[Any]]:
    """Split spatially separated watermark regions so flow/inpaint crops stay local to each corner."""
    if not multi_region:
        return [mask]
    binary = (mask > 0).astype(np.uint8)
    bridged = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    num_labels, labels = cv2.connectedComponents(bridged, connectivity=8)
    if num_labels <= 2:
        return [mask]
    parts: list[NDArray[Any]] = []
    for label_id in range(1, num_labels):
        sub = np.where(labels == label_id, mask, 0).astype(mask.dtype, copy=False)
        if np.any(sub > 0):
            parts.append(sub)
    return parts or [mask]


def iter_clean_video_frames(
    source: Path,
    scan: VideoScan,
    regions: Sequence[Region | Sequence[Region | None] | None],
    *,
    backend: Backend,
    padding_fraction: float = 0.28,
    mask_style: Literal["box", "veo"] = "box",
    slot_padding_fractions: Sequence[float] | None = None,
    slot_mask_styles: Sequence[Literal["box", "veo"]] | None = None,
    temporal_consistency: bool = True,
    dilate_px: int | None = None,
    allow_unmasked_donors: bool = True,
    high_depth: bool = False,
    start_frame: int = 0,
    stop_frame: int | None = None,
) -> Iterator[tuple[NDArray[Any], bool]]:
    """Stream cleaned video frames and per-frame removal flags without forcing an intermediate encode."""
    from remove_ai_watermarks.watermark_registry import (
        backend_prefers_batching,
        fill,
        fill_batch,
        resolve_backend,
    )

    normalized_regions = [_normalize_frame_regions(item) for item in regions]
    total_frames = len(normalized_regions)
    eff_start = max(0, start_frame)
    eff_stop = total_frames if stop_frame is None else min(total_frames, stop_frame)
    if eff_start >= eff_stop:
        return

    dirty_indices = [
        idx for idx in range(eff_start, eff_stop) if any(reg is not None for reg in normalized_regions[idx])
    ]
    clip_mattes = (
        _precalibrate_stationary_mattes(
            source,
            normalized_regions,
            width=scan.width,
            height=scan.height,
            padding_fraction=padding_fraction,
            slot_padding_fractions=slot_padding_fractions,
            dilate_px=dilate_px,
            high_depth=high_depth,
        )
        if temporal_consistency and dirty_indices
        else ()
    )
    capture: Any = (
        open_video_capture_16bit(source, width=scan.width, height=scan.height)
        if high_depth
        else open_video_capture(source)
    )
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not reopen video for removal: {source}")

    previous_source: NDArray[Any] | None = None
    previous_cleaned: NDArray[Any] | None = None
    previous_mask: NDArray[Any] | None = None
    previous_clean_plate: tuple[NDArray[Any], NDArray[Any] | None] | None = None
    pending: tuple[NDArray[Any], NDArray[Any], list[NDArray[Any]], NDArray[Any] | None, bool] | None = None
    try:
        resolved_backend: Literal["cv2", "migan", "lama"] = resolve_backend(backend)
        use_batch_inpaint = resolved_backend in ("lama", "migan") and backend_prefers_batching(resolved_backend)
        inpaint_batch_size = 8 if use_batch_inpaint else 1
        donor_radius = 12 if temporal_consistency else 0
        has_grab = callable(getattr(capture, "grab", None)) and callable(getattr(capture, "retrieve", None))

        def _skip_to(target_idx: int, cur_idx: int) -> int:
            while cur_idx < target_idx:
                if has_grab:
                    if not capture.grab():
                        raise RuntimeError(f"Video ended while advancing to frame {cur_idx}: {source}")
                else:
                    ok_s, _ = capture.read()
                    if not ok_s:
                        raise RuntimeError(f"Video ended while advancing to frame {cur_idx}: {source}")
                cur_idx += 1
            return cur_idx

        def _passthrough_until(limit: int) -> Iterator[tuple[NDArray[Any], bool]]:
            nonlocal next_decode
            while next_decode < limit:
                ok_f, raw_f = capture.read()
                if not ok_f or raw_f is None:
                    raise RuntimeError(f"Video ended while reading frame {next_decode}: {source}")
                next_decode += 1
                yield raw_f, False

        next_decode = _skip_to(eff_start, 0)
        if not dirty_indices:
            yield from _passthrough_until(eff_stop)
            return

        active_start = max(eff_start, dirty_indices[0] - donor_radius)
        active_end = min(eff_stop - 1, dirty_indices[-1] + donor_radius)
        yield from _passthrough_until(active_start)

        window: deque[tuple[int, NDArray[Any], NDArray[Any] | None]] = deque()
        prepared_batch: list[tuple[NDArray[Any], list[NDArray[Any]], NDArray[Any] | None, bool]] = []

        def _crop_neighbor_mask(
            nb_mask: NDArray[Any] | None, sub_mask: NDArray[Any], single_sub: bool
        ) -> NDArray[Any] | None:
            if nb_mask is None:
                return None
            if single_sub:
                return nb_mask
            ys, xs = np.where(sub_mask > 0)
            pad_m = 48
            y0_m = max(0, int(ys.min()) - pad_m)
            y1_m = min(nb_mask.shape[0], int(ys.max()) + 1 + pad_m)
            x0_m = max(0, int(xs.min()) - pad_m)
            x1_m = min(nb_mask.shape[1], int(xs.max()) + 1 + pad_m)
            sub = np.zeros_like(nb_mask)
            sub[y0_m:y1_m, x0_m:x1_m] = nb_mask[y0_m:y1_m, x0_m:x1_m]
            return sub if np.any(sub > 0) else None

        def _flush_pending(
            nxt_entry: tuple[NDArray[Any], NDArray[Any], NDArray[Any] | None] | None,
        ) -> tuple[NDArray[Any], bool] | None:
            nonlocal pending, previous_source, previous_cleaned, previous_mask
            if pending is None:
                return None
            p_src, p_frame, p_sub_masks, p_stage_mask, p_removed = pending
            out_frame = p_frame
            if temporal_consistency and p_sub_masks:
                single_sub = len(p_sub_masks) == 1
                for sub_mask in p_sub_masks:
                    prev_sub = (
                        _crop_neighbor_mask(previous_mask, sub_mask, single_sub)
                        if (previous_source is not None and previous_cleaned is not None)
                        else None
                    )
                    nxt_sub = _crop_neighbor_mask(nxt_entry[2], sub_mask, single_sub) if nxt_entry is not None else None
                    prev_arg = (
                        (previous_source, previous_cleaned, prev_sub)
                        if (previous_source is not None and previous_cleaned is not None and prev_sub is not None)
                        else None
                    )
                    nxt_arg = (
                        (nxt_entry[0], nxt_entry[1], nxt_sub)
                        if (nxt_entry is not None and nxt_sub is not None)
                        else None
                    )
                    if prev_arg is not None or nxt_arg is not None:
                        out_frame = stabilize_filled_frame_bidirectional(
                            p_src,
                            out_frame,
                            sub_mask,
                            previous=prev_arg,
                            nxt=nxt_arg,
                            copy=False,
                        )
            previous_source = p_src
            previous_cleaned = out_frame
            previous_mask = p_stage_mask
            pending = None
            return out_frame, p_removed

        def _pull_frame() -> None:
            nonlocal next_decode
            ok, raw = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended while reading frame {next_decode}: {source}")
            f_regs = normalized_regions[next_decode]
            m = _masks_for_regions(
                raw,
                f_regs,
                padding_fraction=padding_fraction,
                mask_style=mask_style,
                slot_padding_fractions=slot_padding_fractions,
                slot_mask_styles=slot_mask_styles,
                dilate_px=dilate_px,
            )
            window.append((next_decode, raw, m))
            next_decode += 1

        def _drain_prepared_batch() -> Iterator[tuple[NDArray[Any], bool]]:
            nonlocal pending
            if not prepared_batch:
                return
            filled_frames: list[NDArray[Any]] = [entry[0] for entry in prepared_batch]
            if use_batch_inpaint:
                max_slots = max((len(entry[1]) for entry in prepared_batch), default=0)
                for slot_idx in range(max_slots):
                    slot_indices: list[int] = []
                    slot_items: list[tuple[NDArray[Any], NDArray[Any]]] = []
                    for b_idx, entry in enumerate(prepared_batch):
                        if slot_idx < len(entry[1]):
                            slot_indices.append(b_idx)
                            slot_items.append((filled_frames[b_idx], entry[1][slot_idx]))
                    if slot_items:
                        batched_outs = fill_batch(slot_items, backend=resolved_backend)
                        for b_idx, out_img in zip(slot_indices, batched_outs, strict=True):
                            filled_frames[b_idx] = out_img
            else:
                for b_idx, (b_src, b_subs, _b_stage_mask, _b_removed) in enumerate(prepared_batch):
                    if not b_subs:
                        continue
                    cur_frame = b_src
                    for sub_mask in b_subs:
                        cur_frame = fill(cur_frame, sub_mask, backend=resolved_backend)
                    filled_frames[b_idx] = cur_frame
            for b_idx, (b_src, b_subs, b_stage_mask, b_removed) in enumerate(prepared_batch):
                out_frame = filled_frames[b_idx]
                if temporal_consistency:
                    flushed = _flush_pending((b_src, out_frame, b_stage_mask))
                    if flushed is not None:
                        yield flushed
                    pending = (b_src, out_frame, b_subs, b_stage_mask, b_removed)
                else:
                    yield out_frame, b_removed
            prepared_batch.clear()

        for frame_index in range(active_start, active_end + 1):
            while next_decode <= active_end and next_decode <= frame_index + donor_radius:
                _pull_frame()
            while window and window[0][0] < frame_index - donor_radius:
                window.popleft()

            _cur_idx, source_frame, mask = next(item for item in window if item[0] == frame_index)
            stage_source = source_frame
            stage_mask: NDArray[Any] | None = None
            remaining_sub_masks: list[NDArray[Any]] = []
            was_removed = False
            if mask is not None:
                working_mask = mask
                for matte in clip_mattes:
                    stage_source, working_mask = unblend_ycbcr_roi(stage_source, working_mask, matte)
                cur_regs = [r for r in normalized_regions[frame_index] if r is not None]
                sub_masks = _split_disjoint_masks(working_mask, multi_region=len(cur_regs) > 1)
                donors: list[tuple[NDArray[Any], NDArray[Any] | None]] = []
                if temporal_consistency and np.any(working_mask > 0):
                    if len(window) > 1:
                        disjoint_donors: list[tuple[int, NDArray[Any], NDArray[Any] | None]] = []
                        overlap_donors: list[tuple[int, NDArray[Any], NDArray[Any] | None]] = []
                        for d_idx, d_src, d_mask in window:
                            if d_idx == frame_index or not (allow_unmasked_donors or d_mask is not None):
                                continue
                            is_disjoint = (
                                d_mask is None
                                or not cur_regs
                                or all(
                                    _region_iou(dr, cr) < 0.25
                                    for dr in normalized_regions[d_idx]
                                    if dr is not None
                                    for cr in cur_regs
                                )
                            )
                            (disjoint_donors if is_disjoint else overlap_donors).append((d_idx, d_src, d_mask))
                        disjoint_donors.sort(key=lambda it: abs(it[0] - frame_index))
                        # For overlapping stationary marks, stride nearest-to-farthest across
                        # the full +/-donor_radius window so nearby frames get highest weight
                        # while distant frames still uncover wide logos on slow camera pans.
                        overlap_donors.sort(key=lambda it: (abs(it[0] - frame_index), it[0] > frame_index))
                        picked_disjoint = disjoint_donors[:8]
                        remaining_slots = 8 - len(picked_disjoint)
                        if remaining_slots > 0 and overlap_donors:
                            if len(overlap_donors) <= remaining_slots:
                                picked_overlap = overlap_donors
                            else:
                                idxs = np.linspace(0, len(overlap_donors) - 1, remaining_slots, dtype=int)
                                picked_overlap = [overlap_donors[int(i)] for i in idxs]
                        else:
                            picked_overlap = []
                        donors = [(d_src, d_mask) for _i, d_src, d_mask in (*picked_disjoint, *picked_overlap)]
                    # Transitive clean-plate donor: genuine unobscured/recovered pixels from t-1
                    # (strictly before single-frame inpainting) propagate across slow pans > donor_radius.
                    if previous_clean_plate is not None and (
                        allow_unmasked_donors or previous_clean_plate[1] is not None
                    ):
                        donors.append(previous_clean_plate)
                for sub_mask in sub_masks:
                    if donors and np.any(sub_mask > 0):
                        stage_source, sub_mask = recover_unobscured_background(stage_source, sub_mask, donors)
                    if np.any(sub_mask > 0):
                        remaining_sub_masks.append(sub_mask)
                combined_stage_mask: NDArray[Any] | None = None
                single_sub = len(remaining_sub_masks) == 1
                for sub_mask in remaining_sub_masks:
                    combined_stage_mask = (
                        sub_mask
                        if single_sub
                        else (
                            sub_mask.copy()
                            if combined_stage_mask is None
                            else np.maximum(combined_stage_mask, sub_mask)
                        )
                    )
                stage_mask = combined_stage_mask
                was_removed = True

            previous_clean_plate = (stage_source, stage_mask) if temporal_consistency else None
            prepared_batch.append((stage_source, remaining_sub_masks, stage_mask, was_removed))
            if len(prepared_batch) >= inpaint_batch_size:
                yield from _drain_prepared_batch()

        yield from _drain_prepared_batch()
        flushed = _flush_pending(None)
        if flushed is not None:
            yield flushed
        window.clear()
        yield from _passthrough_until(eff_stop)
    finally:
        capture.release()


def _source_has_bitstream_ai_sei(source: Path) -> bool:
    """Return True if the source bitstream carries in-band AI SEI/OBU payloads that stream-copy would preserve."""
    suffix = source.suffix.lower()
    if suffix in {".mp4", ".mov", ".m4v"}:
        from remove_ai_watermarks._internal import isobmff

        sei = isobmff.scan_isobmff_video_sei(source, stop_on_first_kling=True)
        return bool(sei.kling_tag_offsets or sei.ai_sei_spans or sei.av1_obu_patches)
    if suffix in {".mkv", ".webm"}:
        from remove_ai_watermarks._internal import ebml

        return bool(ebml.scan_ebml_av1_ai_obus(source, stop_on_first=True))
    return False


def _try_selective_gop_clean_encode(
    source: Path,
    temporary_output: Path,
    active_scan: VideoScan,
    active_regions: Sequence[Region | Sequence[Region | None] | None],
    *,
    backend: Backend,
    strip_metadata: bool,
    padding_fraction: float,
    mask_style: Literal["box", "veo"],
    slot_padding_fractions: Sequence[float] | None,
    slot_mask_styles: Sequence[Literal["box", "veo"]] | None,
    temporal_consistency: bool,
    dilate_px: int | None,
    allow_unmasked_donors: bool,
    trim_end_frame: int | None,
    trim_end_seconds: float | None,
    crf: int,
    profile: Any,
    timestamped_input: bool,
    high_depth: bool,
) -> int | None:
    """Losslessly copy untouched H.264 keyframe GOPs when verified frame-exact, else return None."""
    from remove_ai_watermarks.video_tools import require_video_tool

    total_active = len(active_regions)
    if total_active < 15 or timestamped_input or high_depth or profile.pixel_format != "yuv420p":
        return None
    suffix = temporary_output.suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".mkv"}:
        return None
    normalized = [_normalize_frame_regions(item) for item in active_regions]
    dirty_indices = [idx for idx, regs in enumerate(normalized) if any(r is not None for r in regs)]
    donor_radius = 12 if temporal_consistency else 0
    if not dirty_indices:
        if trim_end_frame is None:
            return None
    else:
        early_start = max(0, dirty_indices[0] - donor_radius)
        early_end = min(total_active - 1, dirty_indices[-1] + donor_radius)
        if early_start < 30 and (trim_end_frame is not None or early_end >= total_active - 30):
            return None

    if strip_metadata and _source_has_bitstream_ai_sei(source):
        return None

    kf_spec = probe_video_keyframes(source)
    if kf_spec is None or kf_spec.start_pts not in (None, 0):
        return None

    ffmpeg = require_video_tool("ffmpeg")
    # Fast path 1: pure trailing bumper trim (zero dirty frames) aligned on a keyframe boundary.
    if (
        not dirty_indices
        and trim_end_frame is not None
        and trim_end_seconds is not None
        and kf_spec.has_b_frames == 0
        and any(idx == trim_end_frame for idx, _ in kf_spec.keyframes)
    ):
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-frames:v",
            str(trim_end_frame),
            "-to",
            f"{trim_end_seconds:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c",
            "copy",
            "-map_metadata",
            "-1" if strip_metadata else "0",
            "-map_chapters",
            "-1" if strip_metadata else "0",
            *(["-movflags", "+faststart"] if suffix in {".mp4", ".mov", ".m4v"} else []),
            *container_format_args(temporary_output.suffix),
            str(temporary_output),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        if res.returncode == 0 and count_decoded_video_frames(temporary_output) == trim_end_frame:
            log.info("Selective GOP stream-copy: pure keyframe bumper trim (%d frames)", trim_end_frame)
            return 0
        temporary_output.unlink(missing_ok=True)

    # Fast path 2: hybrid GOP stream-copy for H.264 clips with >= 30 untouched GOP frames.
    if kf_spec.codec_name != "h264" or kf_spec.has_b_frames > 0:
        return None
    if trim_end_frame is None and kf_spec.packet_count != total_active:
        return None
    prefix_limit = total_active - 1 if not dirty_indices else max(0, dirty_indices[0] - donor_radius)
    prefix_kfs = [(idx, ts) for idx, ts in kf_spec.keyframes if 30 <= idx <= prefix_limit]
    prefix_end_idx = prefix_kfs[-1][0] if prefix_kfs else 0
    suffix_start_idx: int | None = None
    suffix_ts: float | None = None
    if dirty_indices and trim_end_frame is None:
        active_end = min(total_active - 1, dirty_indices[-1] + donor_radius)
        suffix_kfs = [(idx, ts) for idx, ts in kf_spec.keyframes if active_end < idx <= total_active - 30]
        if suffix_kfs:
            suffix_start_idx, suffix_ts = suffix_kfs[0]
    if prefix_end_idx < 30 and suffix_start_idx is None:
        return None

    mid_start = prefix_end_idx
    mid_end = suffix_start_idx if suffix_start_idx is not None else total_active
    h264_prof_map = {"Constrained Baseline": "baseline", "Baseline": "baseline", "Main": "main", "High": "high"}
    h264_prof = h264_prof_map.get(kf_spec.profile_name or "", "high")
    seg_ext = ".mkv" if suffix == ".mkv" else ".mp4"
    timescale_args = mp4_timescale_args(seg_ext, profile.time_base)

    with TemporaryDirectory(prefix=".raiw-gop-", dir=temporary_output.parent) as gop_dir_str:
        gop_dir = Path(gop_dir_str)
        segments: list[Path] = []
        if prefix_end_idx >= 30:
            seg_pre = gop_dir / f"seg_prefix{seg_ext}"
            pre_cmd = [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-frames:v",
                str(prefix_end_idx),
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                *timescale_args,
                str(seg_pre),
            ]
            if subprocess.run(pre_cmd, capture_output=True, check=False).returncode != 0:  # noqa: S603
                return None
            segments.append(seg_pre)

        removed_frames = 0
        if mid_end > mid_start:
            seg_mid = gop_dir / f"seg_mid{seg_ext}"
            mid_cmd = raw_video_command(
                seg_mid,
                width=active_scan.width,
                height=active_scan.height,
                fps=active_scan.fps,
                crf=crf,
                profile=profile,
                input_pix_fmt="bgr24",
                h264_profile=h264_prof if seg_ext == ".mp4" else None,
            )
            proc = start_raw_video_encoder(mid_cmd)
            mid_written = 0
            mid_iter = iter_clean_video_frames(
                source,
                active_scan,
                active_regions,
                backend=backend,
                padding_fraction=padding_fraction,
                mask_style=mask_style,
                slot_padding_fractions=slot_padding_fractions,
                slot_mask_styles=slot_mask_styles,
                temporal_consistency=temporal_consistency,
                dilate_px=dilate_px,
                allow_unmasked_donors=allow_unmasked_donors,
                high_depth=False,
                start_frame=mid_start,
                stop_frame=mid_end,
            )
            try:
                for frm, was_rem in mid_iter:
                    if was_rem:
                        removed_frames += 1
                    proc.stdin.write(frm.tobytes())
                    mid_written += 1
                finish_raw_video_encoder(proc, seg_mid, operation="selective GOP middle encode")
            except Exception:
                abort_raw_video_encoder(proc)
                return None
            finally:
                close_mid = getattr(mid_iter, "close", None)
                if callable(close_mid):
                    close_mid()
            if mid_written != (mid_end - mid_start):
                return None
            segments.append(seg_mid)

        if suffix_start_idx is not None and suffix_ts is not None:
            seg_suf = gop_dir / f"seg_suffix{seg_ext}"
            suf_cmd = [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{suffix_ts:.6f}",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                *timescale_args,
                str(seg_suf),
            ]
            if subprocess.run(suf_cmd, capture_output=True, check=False).returncode != 0:  # noqa: S603
                return None
            segments.append(seg_suf)

        concat_list = gop_dir / "concat.txt"
        escaped_lines = [
            f"file '{seg.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for seg in segments
        ]
        concat_list.write_text("".join(escaped_lines), encoding="utf-8")
        mux_ok = False
        with suppress(RuntimeError):
            mux_encoded_video(
                concat_list,
                source,
                temporary_output,
                strip_metadata=strip_metadata,
                trim_end_seconds=trim_end_seconds,
                video_input_args=("-f", "concat", "-safe", "0"),
            )
            mux_ok = True
        if mux_ok and count_decoded_video_frames(temporary_output) == total_active:
            log.info(
                "Selective GOP stream-copy succeeded: re-encoded [%d, %d) of %d frames",
                mid_start,
                mid_end,
                total_active,
            )
            return removed_frames
        temporary_output.unlink(missing_ok=True)
        return None


def encode_clean_video(
    source: Path,
    output: Path,
    scan: VideoScan,
    regions: Sequence[Region | Sequence[Region | None] | None],
    *,
    backend: Backend,
    strip_metadata: bool,
    padding_fraction: float = 0.28,
    mask_style: Literal["box", "veo"] = "box",
    slot_padding_fractions: Sequence[float] | None = None,
    slot_mask_styles: Sequence[Literal["box", "veo"]] | None = None,
    temporal_consistency: bool = True,
    dilate_px: int | None = None,
    allow_unmasked_donors: bool = True,
    trim_end_frame: int | None = None,
    crf: int = 14,
) -> int:
    """Decode again, fill accepted regions, and atomically encode with complete audio."""
    if len(regions) != len(scan.detections):
        raise ValueError("Temporal localization count does not match the scanned frame count")

    active_regions = regions[:trim_end_frame] if trim_end_frame is not None else regions
    trim_end_seconds: float | None = None
    active_scan = scan
    if trim_end_frame is not None and trim_end_frame < len(scan.detections):
        if scan.timestamps and len(scan.timestamps) > trim_end_frame:
            trim_end_seconds = float(scan.timestamps[trim_end_frame])
        elif scan.fps > 0:
            trim_end_seconds = float(trim_end_frame) / float(scan.fps)
        active_scan = replace(
            scan,
            detections=scan.detections[:trim_end_frame],
            timestamps=scan.timestamps[:trim_end_frame] if scan.timestamps else (),
        )

    with staged_video_output(output) as (encoded_video, temporary_output):
        profile = probe_video_encode_profile(source)
        high_depth = profile.pixel_format in HIGH_BIT_DEPTH_PIXEL_FORMATS
        if ((profile.component_depth or 0) > 8 or profile.color_transfer in _HDR_TRANSFERS) and not high_depth:
            source_format = profile.source_pixel_format or "unknown pixel format"
            raise RuntimeError(
                "Visible video removal currently supports SDR 8-bit input only; "
                f"refusing to silently reduce {source_format} / "
                f"{profile.color_transfer or 'unknown transfer'} to 8-bit SDR"
            )
        time_base = _timestamp_time_base(profile.time_base)
        preserve_start_offset = profile.start_pts not in (None, 0)
        timestamped_input = preserve_start_offset or _timestamps_are_variable(active_scan, time_base=time_base)
        if timestamped_input and profile.time_base is None:
            profile = replace(
                profile,
                time_base=f"{time_base.numerator}/{time_base.denominator}",
            )
        selective_removed = _try_selective_gop_clean_encode(
            source,
            temporary_output,
            active_scan,
            active_regions,
            backend=backend,
            strip_metadata=strip_metadata,
            padding_fraction=padding_fraction,
            mask_style=mask_style,
            slot_padding_fractions=slot_padding_fractions,
            slot_mask_styles=slot_mask_styles,
            temporal_consistency=temporal_consistency,
            dilate_px=dilate_px,
            allow_unmasked_donors=allow_unmasked_donors,
            trim_end_frame=trim_end_frame,
            trim_end_seconds=trim_end_seconds,
            crf=crf,
            profile=profile,
            timestamped_input=timestamped_input,
            high_depth=high_depth,
        )
        if selective_removed is not None:
            return selective_removed
        raw_pix_fmt = "bgr48le" if high_depth else "bgr24"
        process = start_raw_video_encoder(
            raw_video_command(
                encoded_video,
                width=active_scan.width,
                height=active_scan.height,
                fps=active_scan.fps,
                crf=crf,
                profile=profile,
                timestamped_input=timestamped_input,
                copy_input_timestamps=preserve_start_offset,
                input_pix_fmt=raw_pix_fmt,
            )
        )
        frame_pipe = process.stdin
        removed_frames = 0
        timestamped_writer: _TimestampedNutWriter | None = None
        frame_iter = iter_clean_video_frames(
            source,
            active_scan,
            active_regions,
            backend=backend,
            padding_fraction=padding_fraction,
            mask_style=mask_style,
            slot_padding_fractions=slot_padding_fractions,
            slot_mask_styles=slot_mask_styles,
            temporal_consistency=temporal_consistency,
            dilate_px=dilate_px,
            allow_unmasked_donors=allow_unmasked_donors,
            high_depth=high_depth,
        )
        try:
            if timestamped_input:
                timestamped_writer = _TimestampedNutWriter(
                    frame_pipe,
                    width=active_scan.width,
                    height=active_scan.height,
                    time_base=time_base,
                    start_pts=profile.start_pts or 0,
                    pix_fmt=raw_pix_fmt,
                )
            for frame_index, (frame, was_removed) in enumerate(frame_iter):
                if was_removed:
                    removed_frames += 1
                if timestamped_writer is None:
                    frame_pipe.write(frame.tobytes())
                else:
                    timestamped_writer.write(frame, active_scan.timestamps[frame_index])
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
                trim_end_seconds=trim_end_seconds,
            )
        except Exception:
            if timestamped_writer is not None:
                with suppress(Exception):
                    timestamped_writer.close()
            abort_raw_video_encoder(process)
            raise
        finally:
            close_fn = getattr(frame_iter, "close", None)
            if callable(close_fn):
                close_fn()

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
            markers.get("ai_tool", ""),
            markers.get("issuer", ""),
        )
    ).lower()
    source_type = markers.get("source_type", "").lower()
    return (
        any(token in identity for token in ("bytedance", "byteplus", "spring (sg)", "dola"))
        and "trainedalgorithmicmedia" in source_type
    )


def has_doubao_video_provenance(markers: dict[str, str]) -> bool:
    """Whether the structural TC260 producer identifies registered Doubao provenance."""
    return tc260_producer_in(markers.get("aigc_producer", ""), producer_codes_for_mark("doubao"))


def has_dreamina_video_provenance(markers: dict[str, str]) -> bool:
    """Whether C2PA or TC260 provenance identifies ByteDance Dreamina / Jimeng."""
    if has_bytedance_video_provenance(markers):
        return True
    if tc260_producer_in(markers.get("aigc_producer", ""), producer_codes_for_mark("dreamina")):
        return True
    return "dreamina" in markers.get("claim_generator", "").casefold()


def has_wan_video_provenance(markers: dict[str, str]) -> bool:
    """Whether a TC260 label names Alibaba Tongyi (Wan) as the producer."""
    return tc260_producer_in(markers.get("aigc_producer", ""), producer_codes_for_mark("wan"))


def has_qwen_video_provenance(markers: dict[str, str]) -> bool:
    """Whether a TC260 label names Alibaba Cloud Qwen as the producer."""
    return tc260_producer_in(markers.get("aigc_producer", ""), producer_codes_for_mark("qwen"))


def has_yuanbao_video_provenance(markers: dict[str, str]) -> bool:
    """Whether a TC260 label names Tencent Yuanbao as the producer."""
    return tc260_producer_in(markers.get("aigc_producer", ""), producer_codes_for_mark("yuanbao"))


def has_baidu_video_provenance(markers: dict[str, str]) -> bool:
    """Whether a TC260 label names Baidu as the producer."""
    return tc260_producer_in(markers.get("aigc_producer", ""), producer_codes_for_mark("baidu"))


def has_runninghub_video_provenance(markers: dict[str, str]) -> bool:
    """Whether a TC260 label names RunningHub as the producer."""
    return tc260_producer_in(markers.get("aigc_producer", ""), producer_codes_for_mark("runninghub"))


def _has_valid_c2pa_video_provenance(
    markers: dict[str, str],
    *,
    issuer_tokens: tuple[str, ...] = (),
    identity_tokens: tuple[str, ...] = (),
    generator_claim_tokens: tuple[str, ...] = (),
) -> bool:
    """Shared valid-C2PA AI provenance check across issuer/claim_generator tokens."""
    from remove_ai_watermarks._internal.c2pa import c2pa_credential_level

    if c2pa_credential_level(markers) == "invalid":
        return False
    issuer = markers.get("issuer", "").casefold()
    generator = markers.get("claim_generator", "").casefold()
    identity = f"{issuer} {generator}"
    source_type = markers.get("source_type", "").casefold()
    vendor_match = any(tok in issuer for tok in issuer_tokens) or any(tok in identity for tok in identity_tokens)
    ai_match = "trainedalgorithmicmedia" in source_type or any(tok in generator for tok in generator_claim_tokens)
    return vendor_match and ai_match


def has_microsoft_video_provenance(markers: dict[str, str]) -> bool:
    """Whether C2PA provenance names Microsoft."""
    return _has_valid_c2pa_video_provenance(markers, issuer_tokens=("microsoft",))


def has_pixverse_video_provenance(markers: dict[str, str]) -> bool:
    """Whether C2PA provenance names PixVerse."""
    return _has_valid_c2pa_video_provenance(markers, issuer_tokens=("pixverse",))


def has_sync_labs_video_provenance(markers: dict[str, str]) -> bool:
    """Whether C2PA provenance names Sync Labs / Synchronicity Labs."""
    return _has_valid_c2pa_video_provenance(markers, identity_tokens=("sync labs", "synchronicity"))


def has_tiktok_ai_video_provenance(markers: dict[str, str]) -> bool:
    """Whether C2PA provenance names TikTok AI generation."""
    return _has_valid_c2pa_video_provenance(
        markers,
        identity_tokens=("tiktok",),
        generator_claim_tokens=("tiktok ad creative toolbox",),
    )


# Tokens of the C2PA issuer each mark's own vendor signs with, compared casefolded.
_VIDEO_MARK_ISSUER_TOKENS: dict[str, tuple[str, ...]] = {
    "sora": ("openai",),
    "veo": ("google",),
    "dreamina": ("bytedance", "byteplus", "spring (sg)", "dreamina"),
    "jimeng": ("bytedance", "byteplus", "spring (sg)", "dreamina"),
    "jimeng_pill": ("bytedance", "byteplus", "spring (sg)", "dreamina"),
    "seedance": ("bytedance", "byteplus", "spring (sg)"),
    "doubao": ("bytedance", "byteplus", "spring (sg)"),
    "dola": ("bytedance", "byteplus", "spring (sg)"),
    "hailuo": ("minimax",),
    "vidu": ("shengshu", "vidu"),
    "wan": ("alibaba", "tongyi", "wan"),
    "qwen": ("alibaba", "tongyi", "qwen"),
    "yuanbao": ("tencent", "yuanbao", "hunyuan"),
    "baidu": ("baidu", "ernie"),
    "runninghub": ("runninghub",),
    "microsoft": ("microsoft",),
    "pixverse": ("pixverse",),
    "luma": ("luma",),
    "heygen": ("heygen",),
    "sync_labs": ("sync labs", "synchronicity"),
    "tiktok_ai": ("tiktok", "bytedance"),
    "generic_ai_label": (),
    "kling": ("kling", "kuaishou"),
}


def confirms_video_provenance(mark: str, markers: dict[str, str]) -> bool:
    """Whether provenance names the visible mark's own vendor; Kling has no such predicate."""
    confirms = {
        "sora": has_sora_provenance,
        "veo": has_veo_provenance,
        "dreamina": has_dreamina_video_provenance,
        "jimeng": has_dreamina_video_provenance,
        "jimeng_pill": has_dreamina_video_provenance,
        "seedance": has_bytedance_video_provenance,
        "doubao": has_doubao_video_provenance,
        "dola": has_bytedance_video_provenance,
        "hailuo": has_hailuo_video_provenance,
        "vidu": has_vidu_video_provenance,
        "wan": has_wan_video_provenance,
        "qwen": has_qwen_video_provenance,
        "yuanbao": has_yuanbao_video_provenance,
        "baidu": has_baidu_video_provenance,
        "runninghub": has_runninghub_video_provenance,
        "microsoft": has_microsoft_video_provenance,
        "pixverse": has_pixverse_video_provenance,
        "sync_labs": has_sync_labs_video_provenance,
        "tiktok_ai": has_tiktok_ai_video_provenance,
    }.get(mark)
    return bool(confirms and confirms(markers))


def contradicts_video_provenance(mark: str, markers: dict[str, str]) -> bool:
    """Whether provenance names a producer other than the visible mark's vendor.

    A present TC260 ``ContentProducer`` naming another vendor vetoes any mark whose
    own vendor is not named: Kling scored 0.56 on a Vidu Q3 export (2026-09-23),
    Hailuo scored 0.35 on all 146 frames of a ByteDance Jimeng video (2026-10-08),
    and Seedance scored 0.45 on all 294 frames of two Agnes AI videos (2026-10-10).
    A missing label is not evidence either way and leaves the visual path alone.

    A C2PA manifest that itself claims AI generation by another vendor contradicts
    any mark: a Veo 3.1 Lite clip with Google's C2PA (Higgsfield, 2026-09-24) scored
    0.61 for Kling on wood texture, against 0.63 on the real Kling original, a stable
    192-frame run; a Gemini Omni 1.1 Flash clip with Google's C2PA (Runway,
    2026-09-25) scored 0.61-0.67 for Sora on a steam wisp over five frames. A
    re-signing platform (YouTube) records no AI source type, so it vetoes nothing.
    An invalid credential vetoes nothing either, and provenance that also confirms
    the mark's own vendor (a Sora claim generator under a Google issuer) confirms
    instead of vetoing.
    """
    if mark == "generic_ai_label" or confirms_video_provenance(mark, markers):
        return False
    producer = markers.get("aigc_producer", "").strip()
    if producer:
        row = producer_for_code(producer)
        if row is None:
            return True
        if mark in {"dreamina", "jimeng", "jimeng_pill", "seedance", "dola", "tiktok_ai"}:
            return row.vendor != "ByteDance"
        return mark not in (row.image_mark, row.video_mark)
    from remove_ai_watermarks._internal.c2pa import c2pa_credential_level

    issuer = markers.get("issuer", "").casefold()
    claims_ai = "trainedalgorithmicmedia" in markers.get("source_type", "").casefold()
    return (
        bool(issuer)
        and claims_ai
        and c2pa_credential_level(markers) != "invalid"
        and not any(token in issuer for token in _VIDEO_MARK_ISSUER_TOKENS[mark])
    )


def has_vidu_video_provenance(markers: dict[str, str]) -> bool:
    """Whether a TC260 label names ShengShu, Vidu's maker, as the producer."""
    return tc260_producer_in(markers.get("aigc_producer", ""), producer_codes_for_mark("vidu"))


def has_hailuo_video_provenance(markers: dict[str, str]) -> bool:
    """Whether a TC260 label names MiniMax, Hailuo's maker, as the producer.

    Hailuo video exports carry no C2PA; their TC260 ``ContentProducer`` is the
    bare name ``MiniMax`` (verified on retained MiniMax-hailuo clips). The
    producer travels as its own structural marker (``aigc_producer``, set in
    ``metadata.get_ai_metadata``), so this matches the field exactly rather
    than parsing the human-readable ``aigc_label`` sentence.
    """
    return tc260_producer_in(markers.get("aigc_producer", ""), producer_codes_for_mark("hailuo"))
