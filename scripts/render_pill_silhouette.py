"""Render the SYNTHETIC 'AI生成' pill silhouette asset (data-safe, font-rendered).

The Jimeng-basic TC260 visible label is a rounded pill with 'AI生成' in the TOP-LEFT
corner (issue #54). Unlike the reverse-alpha marks, this is a CAPTURE-LESS mark:
the committed asset is a font-rendered binary SILHOUETTE (mark shape only, zero photo
content), used ONLY to (a) detect the pill by edge-NCC in the top-left corner and
(b) build the inpaint mask. It is NOT an alpha map -- removal quality comes from the
inpaint backend (MI-GAN/cv2), so the silhouette need not be pixel-accurate, and the
synthetic render keeps corpus/user content out of the tracked repo (data-safety).

Detection was calibrated on the retained local corpus (61 real positives + jimeng
negatives): edge-NCC threshold ~0.22 in the top-left ROI. Re-run to regenerate the
asset:  uv run python scripts/render_pill_silhouette.py

Requires a CJK font (macOS STHeiti by default); the asset itself is committed, so this
script only runs when regenerating it (never in CI).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ASSET = Path(__file__).resolve().parents[1] / "src" / "remove_ai_watermarks" / "assets" / "jimeng_pill.png"
_FONT = "/System/Library/Fonts/STHeiti Medium.ttc"


def render_silhouette(w: int = 320) -> np.ndarray:
    """Rounded-pill outline + 'AI生成' text as a binary silhouette (255 = mark)."""
    h = int(w * 0.5)
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    pad = int(w * 0.03)
    r = (h - 2 * pad) // 3
    d.rounded_rectangle([pad, pad, w - pad, h - pad], radius=r, outline=255, width=max(2, w // 90))
    fsz = int(h * 0.42)
    font = ImageFont.truetype(_FONT, fsz)
    txt = "AI生成"
    tb = d.textbbox((0, 0), txt, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    d.text(((w - tw) // 2 - tb[0], (h - th) // 2 - tb[1]), txt, font=font, fill=255)
    return np.array(im)


def main() -> None:
    try:
        sil = render_silhouette()
    except OSError as e:
        print(f"Font not found ({e}); install a CJK font or edit _FONT.", file=sys.stderr)
        raise SystemExit(1) from e
    _ASSET.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sil).save(_ASSET)
    print(f"wrote {_ASSET}  ({sil.shape[1]}x{sil.shape[0]})")


if __name__ == "__main__":
    main()
