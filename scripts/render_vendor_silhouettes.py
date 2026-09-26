"""Render synthetic detection silhouettes for vendor text marks.

Committed assets must be font-rendered and contain no source-image pixels. Local
evaluation inputs may be used only to learn glyphs, weight, layout, and detector
thresholds. Candidate assets stay outside the installed package until calibrated.

Regenerate with:
    uv run python scripts/render_vendor_silhouettes.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from collections.abc import Callable

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from remove_ai_watermarks.watermark_registry import mark_keys  # noqa: E402

_PACKAGE_ASSETS = _ROOT / "src" / "remove_ai_watermarks" / "assets"
_CANDIDATE_ASSETS = _ROOT / "scripts" / "assets" / "visible-mark-candidates"
# STHeiti Medium approximates the semibold CJK sans these marks are set in; the exact
# family is unpublished for every vendor (GB 45438-2025 only requires a legible face).
_FONT = "/System/Library/Fonts/STHeiti Medium.ttc"

MARKS = {
    # Brand-less TC260 fallback (no vendor wordmark exists to tune against -- see
    # generic_ai_label_engine.py's module docstring for the vivo/Xiaomi/Samsung
    # evidence). Bare glyph, no prefix, unlike every other row below.
    "generic_ai_label_alpha.png": "AI生成",
    "qwen_alpha.png": "千问AI生成",
    "xinghui_alpha.png": "星绘AI生成",
    # Yuanbao's stamp is a TWO-LINE block (元宝 over AI生成), left-aligned, tightly
    # stacked and ITALIC-SLANTED. A rare one-line variant exists, but the stacked block
    # is dominant.
    "yuanbao_alpha.png": "元宝\nAI生成",
    # Older Kling (可灵) exports stamp a thin light-gray one-line "可灵AI 3.0"
    # bottom-right. The leading spiral logo is NOT rendered (logos vary; the text
    # run discriminates).
    "kling_alpha.png": "可灵AI 3.0",
    # Current IMAGE 3.0 exports use the Latin ``KlingAI 3.0`` run. Keep it as a
    # second installed silhouette rather than replacing the measured CJK variant.
    "kling_latin_alpha.png": "KlingAI 3.0",
    # The "cat-logo" candidate stamps an outline cat-head plus bold "AI生成",
    # bottom-right. It remains unregistered pending sufficient calibration coverage.
    "catlogo_alpha.png": "cat logo + AI生成",
    # RunningHub top-left text mark.
    "runninghub_alpha.png": "RunningHub AI生成",
    # LibLibAI bottom-center wordmark.
    "liblib_alpha.png": "LibLibAI",
    # Zhipu Qingyan candidate text mark.
    "qingyan_alpha.png": "清言·AI生成",
    # MiniMax / Hailuo candidate wordmark.
    "hailuo_alpha.png": "Hailuo AI",
    # Baidu bottom-right text run.
    "baidu_alpha.png": "百度",
    # Measured Microsoft top-right white AI-badge variant. Sentinel: drawn by
    # draw_msbadge(), not font-rendered.
    "microsoft_alpha.png": "Made with AI",
    # Samsung Galaxy AI label, English locale (the registered samsung_alpha.png is
    # the Italian "Contenuti generati dall'AI" silhouette; EN is the literal
    # translation with the same leading sparkle). Sentinel: draw_samsung_en().
    "samsung_en_alpha.png": "AI-generated content",
    # Gemini text-form label (the registered gemini mark is the sparkle icon).
    "gemini_text_alpha.png": "Generated with Gemini",
    # Candidate wordmarks measured on a local evaluation corpus (unregistered).
    "notebooklm_alpha.png": "NotebookLM",
    "dola_alpha.png": "DolaAI",
    "mindvideo_alpha.png": "MindVideo.AI",
    "higgsfield_alpha.png": "HIGGSFIELD AI",
    "capcut_alpha.png": "CapCut AI",
    "zsky_alpha.png": "MADE WITH zsky.ai",
    "chromastudio_alpha.png": "ChromaStudio.ai",
    "digenai_alpha.png": "DIGENAI",
    "gendo_alpha.png": "GendoAI",
    # Alibaba Wan (通义万相) bottom-right logo plus "Wan" wordmark. Sentinel:
    # drawn by draw_wan(), not font-rendered alone.
    "wan_alpha.png": "Wan logo + Wan",
    # Jianying / 剪映, the China product in the CapCut family, stamps 剪映AI
    # bottom-right (the international CapCut pill sits top-left).
    "jianying_alpha.png": "剪映AI",
}
_REGISTERED = {f"{key}_alpha.png" for key in mark_keys()} & MARKS.keys()

# Per-mark post-processing for the multi-line / slanted stamps (see render()).
MARK_OPTS: dict[str, dict[str, Any]] = {
    # Hiragino Sans GB W6, tight leading, dilation, and negative shear match the
    # standard Yuanbao stamp without clipping the lower line.
    "yuanbao_alpha.png": {
        "gap_frac": 0.05,
        "dilate": 2,
        "shear": -0.60,
        "font": "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "font_index": 2,
    },
    # Qingyan uses a heavier weight than STHeiti Medium.
    "qingyan_alpha.png": {"font": "/System/Library/Fonts/Hiragino Sans GB.ttc", "font_index": 2},
    # LibLibAI uses an Arial-class grotesque.
    "liblib_alpha.png": {"font": "/System/Library/Fonts/Supplemental/Arial.ttf"},
    "kling_latin_alpha.png": {"font": "/System/Library/Fonts/HelveticaNeue.ttc", "font_index": 1},
}

# ``kling_latin`` is a variant of the registered ``kling`` row, not a second
# registry key, so it cannot be derived from ``mark_keys()`` above.
_REGISTERED.add("kling_latin_alpha.png")


def _fit_font(
    font_path: str,
    reaches_target: Callable[[ImageFont.FreeTypeFont], bool],
    *,
    index: int = 0,
) -> ImageFont.FreeTypeFont:
    """Return the smallest 8-200 px font that reaches a render target."""
    low, high = 8, 200
    while low < high:
        size = (low + high) // 2
        font = ImageFont.truetype(font_path, size, index=index)
        if reaches_target(font):
            high = size
        else:
            low = size + 1
    return ImageFont.truetype(font_path, low, index=index)


def render(text: str, width: int = 335, opts: dict[str, Any] | None = None) -> np.ndarray:
    """Binary glyph silhouette (255 = glyph), sized to the doubao asset's convention.

    Matching doubao's 335px asset width keeps the `alpha_*_frac` numbers transferable,
    since these marks are the same house style and scale. A "\n" in ``text`` renders a
    multi-line block: lines drawn left-aligned at one shared font size with a tight
    gap, then optional stroke dilation and an italic shear (see MARK_OPTS).
    """
    opts = opts or {}
    gap_frac = float(opts.get("gap_frac", 0.15))
    dilate = int(opts.get("dilate", 0))
    shear_k = float(opts.get("shear", 0.0))
    font_path = str(opts.get("font", _FONT))
    font_index = int(opts.get("font_index", 0))
    probe = Image.new("L", (10, 10))
    d0 = ImageDraw.Draw(probe)
    lines = text.split("\n")
    font = _fit_font(
        font_path,
        lambda f: max(d0.textbbox((0, 0), ln, font=f)[2] for ln in lines) >= width * 0.98,
        index=font_index,
    )
    boxes = [d0.textbbox((0, 0), ln, font=font) for ln in lines]
    line_h = max(bb[3] - bb[1] for bb in boxes)
    gap = max(1, int(line_h * gap_frac))
    w = max(bb[2] - bb[0] for bb in boxes)
    h = line_h * len(lines) + gap * (len(lines) - 1)
    pad = max(2, int(line_h * 0.12))
    im = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    draw = ImageDraw.Draw(im)
    y = pad
    for ln, bb in zip(lines, boxes, strict=True):
        draw.text((pad - bb[0], y - bb[1]), ln, font=font, fill=255)
        y += line_h + gap
    sil = np.array(im)
    if dilate or shear_k:
        import cv2

        if dilate:
            sil = cv2.dilate(sil, np.ones((dilate, dilate), np.uint8))
        if shear_k:
            hh, ww = sil.shape
            extra = int(abs(shear_k) * hh)
            offset = extra if shear_k < 0 else 0
            sil = cv2.warpAffine(
                sil,
                np.float32([[1, shear_k, offset], [0, 1, 0]]),
                (ww + extra, hh),
            )
            ys, xs = np.where(sil > 0)
            if xs.size:
                sil = sil[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return sil


def draw_catlogo(width: int = 335) -> np.ndarray:
    """The cat-logo mark: an outline cat-head (integrated pointy ears, two dot eyes)
    + a bold "AI生成" run, drawn synthetically from the calibrated layout. The outline
    form is the parked candidate described in MARKS."""
    probe = Image.new("L", (10, 10))
    d0 = ImageDraw.Draw(probe)
    text = "AI生成"
    font = _fit_font(_FONT, lambda f: d0.textbbox((0, 0), text, font=f)[2] >= width * 0.60)
    bb = d0.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    cs = int(th * 1.08)
    stroke = max(2, int(th * 0.09))
    gap = int(th * 0.35)

    def head(s: int) -> Image.Image:
        im = Image.new("L", (s, s), 0)
        d = ImageDraw.Draw(im)
        f = float(s)
        pts = [
            (0.12 * f, 0.95 * f),
            (0.10 * f, 0.45 * f),
            (0.12 * f, 0.30 * f),
            (0.20 * f, 0.05 * f),  # left ear tip
            (0.40 * f, 0.24 * f),  # left ear valley
            (0.60 * f, 0.24 * f),  # right ear valley
            (0.80 * f, 0.05 * f),  # right ear tip
            (0.88 * f, 0.30 * f),
            (0.90 * f, 0.45 * f),
            (0.88 * f, 0.95 * f),
        ]
        d.line([*pts, pts[0]], fill=255, width=stroke, joint="curve")
        r = max(1.5, stroke * 0.7)
        d.ellipse([0.35 * f - r, 0.60 * f - r, 0.35 * f + r, 0.60 * f + r], fill=255)
        d.ellipse([0.65 * f - r, 0.60 * f - r, 0.65 * f + r, 0.60 * f + r], fill=255)
        return im

    w = cs + gap + tw
    h = max(th, cs)
    pad = max(2, int(h * 0.12))
    im = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    im.paste(head(cs), (pad, pad + (h - cs) // 2))
    ImageDraw.Draw(im).text((pad + cs + gap - bb[0], pad + (h - th) // 2 - bb[1]), text, font=font, fill=255)
    return np.array(im)


def _star_pts(cx: float, cy: float, r: float, waist: float) -> list[tuple[float, float]]:
    return [
        (cx, cy - r),
        (cx + r * waist, cy - r * waist),
        (cx + r, cy),
        (cx + r * waist, cy + r * waist),
        (cx, cy + r),
        (cx - r * waist, cy + r * waist),
        (cx - r, cy),
        (cx - r * waist, cy - r * waist),
    ]


def _sparkle(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float) -> None:
    """Draw the four-point cutout used by synthetic candidate silhouettes."""
    draw.polygon(_star_pts(cx, cy, r, 0.22), fill=0)


def draw_msbadge(width: int = 335) -> np.ndarray:
    """Synthetic silhouette for the measured Microsoft top-right white pill.

    The top-hat front-end sees the bright pill with dark-text holes, so the template
    carries the same holes -- that is what discriminates this pill from any other
    white rounded element in the top-right corner. The text and four-point cutout
    approximate the measured internal shape; they do not assert one universal
    Microsoft icon or wording. Geometry measured on 17 visually confirmed carriers
    on 2026-08-27: pill 0.152W x 0.040W, margins ~0.010W right / ~0.007W top,
    glyph height ~0.39 of pill height.
    """
    h = round(width / 3.78)
    im = Image.new("L", (width, h), 0)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, width - 1, h - 1], radius=h // 2, fill=255)
    font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    text = "Made with AI"
    probe = Image.new("L", (10, 10))
    d0 = ImageDraw.Draw(probe)
    font = _fit_font(font_path, lambda f: d0.textbbox((0, 0), text, font=f)[3] >= h * 0.39)
    bb = d0.textbbox((0, 0), text, font=font)
    th = bb[3] - bb[1]
    r = h * 0.20  # sparkle radius, ~half the text height
    pad_l = h * 0.22
    cx = pad_l + r
    cy = h / 2 - 1
    tx = int(pad_l + 2 * r + h * 0.22)
    d.text((tx - bb[0], (h - th) // 2 - bb[1]), text, font=font, fill=0)
    _sparkle(d, cx, cy, r)
    return np.array(im)


def draw_samsung_en(width: int = 335) -> np.ndarray:
    """Samsung Galaxy AI English label: "AI-generated content" with the leading
    4-point sparkle, light-gray glyphs (same class as the registered Italian asset)."""
    text = "AI-generated content"
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    probe = Image.new("L", (10, 10))
    d0 = ImageDraw.Draw(probe)
    font = _fit_font(font_path, lambda f: d0.textbbox((0, 0), text, font=f)[2] >= width * 0.80)
    bb = d0.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    r = th * 0.55
    gap = th * 0.45
    im = Image.new("L", (int(tw + gap + 2 * r + 8), th + 8), 0)
    d = ImageDraw.Draw(im)
    # sparkle as bright glyph (this silhouette is light-glyph class, not a pill)
    d.polygon(_star_pts(4 + r, 4 + th / 2, r, 0.22), fill=255)
    d.text((4 + 2 * r + gap - bb[0], 4 - bb[1]), text, font=font, fill=255)
    arr = np.array(im)
    ys, xs = np.where(arr > 0)
    return arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def draw_wan(width: int = 335) -> np.ndarray:
    """Alibaba Wan logo plus "Wan" wordmark, bottom-right, light-glyph class.

    Geometry measured on the one real Wan 2.7 Pro export (2048 px square,
    2026-09-23): logo 102 x 103 px, a 19 px gap, then "Wan" 147 px wide with a
    51 px cap height centered on the logo; 268 x 103 overall. The logo is
    approximated as a hexagon with a central triangular hole and three pinwheel
    cuts; the wordmark is Avenir Next Demi Bold, the nearest installed geometric sans.
    Hole size, cut width and weight were chosen by the best match to the real mark
    over a small grid (0.66 vs 0.60-0.65 for the neighbors).
    """
    import math

    k = width / 268
    h = round(103 * k)
    im = Image.new("L", (width, h), 0)
    d = ImageDraw.Draw(im)
    logo = round(102 * k)
    cx, cy, radius = logo / 2, h / 2, logo / 2
    d.polygon(
        [
            (cx + radius * math.cos(math.radians(60 * i)), cy + radius * 0.98 * math.sin(math.radians(60 * i)))
            for i in range(6)
        ],
        fill=255,
    )
    inner = [
        (
            cx + radius * 0.42 * math.cos(math.radians(90 + 120 * i)),
            cy + radius * 0.42 * math.sin(math.radians(90 + 120 * i)),
        )
        for i in range(3)
    ]
    d.polygon(inner, fill=0)
    cut = max(2, round(logo * 0.05))
    for i, (px, py) in enumerate(inner):
        angle = math.radians(90 + 120 * i + 90)
        d.line(
            [(px, py), (px + radius * 1.2 * math.cos(angle), py + radius * 1.2 * math.sin(angle))], fill=0, width=cut
        )
    text = "Wan"
    probe = ImageDraw.Draw(Image.new("L", (10, 10)))
    font = _fit_font(
        "/System/Library/Fonts/Avenir Next.ttc",
        lambda f: probe.textbbox((0, 0), text, font=f)[3] - probe.textbbox((0, 0), text, font=f)[1] >= round(51 * k),
        index=2,
    )
    bb = probe.textbbox((0, 0), text, font=font)
    d.text((round((102 + 19) * k) - bb[0], round(h / 2 - (bb[3] - bb[1]) / 2) - bb[1]), text, font=font, fill=255)
    arr = np.array(im)
    ys, xs = np.where(arr > 0)
    return arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


_CUSTOM_RENDERERS = {
    "catlogo_alpha.png": draw_catlogo,
    "microsoft_alpha.png": draw_msbadge,
    "samsung_en_alpha.png": draw_samsung_en,
    "wan_alpha.png": draw_wan,
}


def main() -> None:
    try:
        for name, text in MARKS.items():
            renderer = _CUSTOM_RENDERERS.get(name)
            sil = renderer() if renderer is not None else render(text, opts=MARK_OPTS.get(name))
            output_dir = _PACKAGE_ASSETS if name in _REGISTERED else _CANDIDATE_ASSETS
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / name
            Image.fromarray(sil).save(output)
            print(f"wrote {output}  ({sil.shape[1]}x{sil.shape[0]})  text={text!r}")
    except OSError as e:
        print(f"Font not found ({e}); install a CJK font or edit _FONT.", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
