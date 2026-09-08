"""Render the visible-mark example gallery under data/fixtures/visible/.

One committed example per registered image mark, so the repo carries a working
sample of everything it supports and a canary test can hold both sides to it
(mark registered without example; engine regressed on its canonical example).

Every example is SYNTHETIC: a deterministic generated base photo with the mark's
own committed alpha or silhouette composited at the engine's measured geometry.
Some alpha assets were solved from controlled provider captures; the other shapes
are repository-owned renders rather than copied vendor rasters.

Regenerate with:
    uv run python scripts/render_visible_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from remove_ai_watermarks import watermark_registry as wr  # noqa: E402
from remove_ai_watermarks.image_io import imread  # noqa: E402

OUT = _ROOT / "data" / "fixtures" / "visible"

# Composite strength per key: glyph target luma for the light-overlay class.
# Kling is a thin light-gray run. Samsung's solved asset already stores its faint
# measured opacity and was solved against pure white, so it must use 255 without a
# second attenuation. Everything else uses the bold near-white house style.
_STRENGTH: dict[str, int] = {
    "kling": 208,
    "samsung": 255,
}
_DEFAULT_STRENGTH = 238

# Base geometry: one canonical size per mark where the engine's size modes or
# calibrated orientation matter, otherwise a plain 3:2 landscape frame.
# Samsung keeps the larger base because real marks live on ~2958px phone photos.
_SIZE: dict[str, tuple[int, int]] = {
    "qwen": (1536, 1536),
    "liblib": (1152, 1536),
    "liblib_pill": (1152, 1536),
    "samsung": (2048, 1536),
    "jimeng_pill": (1152, 1536),
}


def base_photo(w: int, h: int, seed: int = 7) -> np.ndarray:
    """A deterministic synthetic 'photo': gradient sky, soft blobs, mild noise."""
    rng = np.random.default_rng(seed)
    top, bottom = 96, 168
    grad = np.linspace(top, bottom, h, dtype=np.float32)[:, None]
    img = np.repeat(grad[:, :, None], w, axis=1)  # (h, w, 1)
    for _ in range(5):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(w * 0.12, w * 0.35)
        blob = rng.uniform(-52, 52)
        yy, xx = np.ogrid[:h, :w]
        gauss = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (r * 0.55) ** 2))).astype(np.float32)
        img = img + (blob * gauss)[:, :, None]
    img = img + rng.normal(0, 1.6, img.shape).astype(np.float32)
    return cv2.merge([np.clip(img, 0, 255).astype(np.uint8)] * 3)


def _glyph_asset(name: str) -> np.ndarray:
    from remove_ai_watermarks._text_mark_engine import load_alpha_template

    at = load_alpha_template(name)
    if at is None:
        raise RuntimeError(f"missing silhouette asset: {name}")
    return at


def _composite_light(base: np.ndarray, alpha: np.ndarray, x: int, y: int, strength: int) -> np.ndarray:
    out = base.copy()
    h, w = alpha.shape[:2]
    roi = out[y : y + h, x : x + w].astype(np.float32)
    alpha = np.clip(alpha, 0.0, 1.0)
    a3 = alpha[:, :, None] if alpha.ndim == 2 else alpha
    out[y : y + h, x : x + w] = np.clip(roi * (1 - a3) + strength * a3, 0, 255).astype(np.uint8)
    return out


def _stamp_text_mark(
    key: str,
    base: np.ndarray,
    *,
    size_mult: float,
    alpha_mult: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    engine = wr._engine(key)  # the generator drives the engine's own config
    cfg = engine.config
    h, w = base.shape[:2]
    if key == "microsoft":
        # Opaque white pill with dark text/sparkle holes, at the measured inset.
        at = _glyph_asset("microsoft_alpha.png")
        long_side = max(w, h)
        pw = int(0.152 * long_side * size_mult)
        ph = max(4, int(pw / (at.shape[1] / at.shape[0])))
        pad = int(0.010 * long_side)
        if pw > w - pad or ph > h - pad:
            return None
        pill = cv2.resize(at, (pw, ph))
        x, y = w - pad - pw, pad
        out = base.copy()
        bright = (pill > 0.6)[:, :, None]
        target = np.where(bright, 245.0, 46.0).astype(np.float32)
        blend = float(np.clip(alpha_mult, 0.0, 1.0))
        roi = out[y : y + ph, x : x + pw].astype(np.float32)
        out[y : y + ph, x : x + pw] = np.clip(roi * (1 - blend) + target * blend, 0, 255).astype(np.uint8)
        return out, (x, y, pw, ph)
    base_dim = {"short": min(w, h), "width": w, "long": max(w, h)}[cfg.scale_basis]
    # Size the glyph ON a ladder rung: the continuous front ends sweep only the
    # configured rungs, and a glyph sized between rungs collapses the NCC (the
    # comb-collapse qwen's own two-rung ladder exists to avoid).
    rung = max(cfg.ladder) if cfg.detect_frontend != "binary" else 1.0
    gw = max(cfg.min_gw, int(cfg.alpha_width_frac * base_dim * rung * size_mult))
    gh = max(4, int(cfg.alpha_height_frac * base_dim * rung * size_mult))
    loc = engine.locate(base)
    # Corner-hugging placement: the yuanbao/runninghub anchor gates demote a match
    # that does not hug the corner, and the real marks sit flush on the box's
    # corner side (never centered).
    if cfg.corner in ("br", "tr"):
        x = loc.x + loc.w - gw
    elif cfg.corner == "bc":
        x = loc.x + (loc.w - gw) // 2
    else:  # bl, tl: flush left
        x = loc.x
    y = loc.y if cfg.corner in ("tl", "tr") else loc.y + loc.h - gh
    x, y = max(0, x), max(0, y)
    if x + gw > w or y + gh > h:
        return None
    at = _glyph_asset(f"{key}_alpha.png")
    alpha = cv2.resize(at, (gw, gh))
    alpha = alpha * alpha_mult
    return _composite_light(base, alpha, x, y, _STRENGTH.get(key, _DEFAULT_STRENGTH)), (x, y, gw, gh)


def _stamp_gemini(
    base: np.ndarray,
    *,
    size_mult: float,
    alpha_mult: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    from remove_ai_watermarks.gemini_engine import _shared_engine, get_watermark_config, get_watermark_size

    h, w = base.shape[:2]
    eng = _shared_engine()
    size = get_watermark_size(w, h)
    alpha = eng.get_alpha_map(size)
    cfg = get_watermark_config(w, h)
    x, y = cfg.get_position(w, h)
    ah, aw = alpha.shape[:2]
    gw, gh = max(8, int(aw * size_mult)), max(8, int(ah * size_mult))
    x, y = x + aw - gw, y + ah - gh
    if x < 0 or y < 0 or x + gw > w or y + gh > h:
        return None
    resized = cv2.resize(alpha, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32) * alpha_mult
    return _composite_light(base, resized, x, y, 255), (x, y, gw, gh)


def _stamp_pill(
    base: np.ndarray,
    *,
    size_mult: float,
    alpha_mult: float,
    width_frac: float = 0.161,
    height_frac: float | None = None,
    margin_frac: float = 0.03,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    h, w = base.shape[:2]
    at = _glyph_asset("jimeng_pill.png")
    pw = max(24, int(width_frac * w * size_mult))
    ph = (
        max(8, int(height_frac * w * size_mult))
        if height_frac is not None
        else max(8, int(pw * at.shape[0] / at.shape[1]))
    )
    x, y = int(margin_frac * w), int(margin_frac * h)
    if x + pw > w or y + ph > h:
        return None
    alpha = cv2.resize(at, (pw, ph)) * alpha_mult
    return _composite_light(base, alpha, x, y, 232), (x, y, pw, ph)


def _stamp_liblib_pill(
    base: np.ndarray,
    *,
    size_mult: float,
    alpha_mult: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Composite LiblibAI's shorter top-left pill variant."""
    return _stamp_pill(
        base,
        size_mult=size_mult,
        alpha_mult=alpha_mult,
        width_frac=0.15,
        height_frac=0.05,
        margin_frac=0.025,
    )


def stamp_image_mark(
    key: str,
    base: np.ndarray,
    *,
    size_mult: float = 1.0,
    alpha_mult: float = 1.0,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Composite one image mark using the gallery's verified forward model."""
    if key == "gemini":
        return _stamp_gemini(base, size_mult=size_mult, alpha_mult=alpha_mult)
    if key == "jimeng_pill":
        return _stamp_pill(base, size_mult=size_mult, alpha_mult=alpha_mult)
    if key == "liblib_pill":
        return _stamp_liblib_pill(base, size_mult=size_mult, alpha_mult=alpha_mult)
    return _stamp_text_mark(key, base, size_mult=size_mult, alpha_mult=alpha_mult)


def build(
    key: str,
    *,
    size: tuple[int, int] | None = None,
    seed: int = 7,
) -> np.ndarray:
    return build_pair(key, size=size, seed=seed)[1]


def build_pair(
    key: str,
    *,
    size: tuple[int, int] | None = None,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the clean synthetic source and the matching marked image."""
    size = size or _SIZE.get(key, (1536, 1152))
    clean = base_photo(*size, seed=seed)
    stamped = stamp_image_mark(key, clean)
    if stamped is None:
        raise RuntimeError(f"could not stamp {key} at its nominal geometry")
    marked, _box = stamped
    return clean, marked


# ── Video mark examples ──────────────────────────────────────────────────────
# One short clip per registered video mark: the detector's own synthetic
# template composited at a scale inside its calibrated search profile, on every
# frame of a generated base. The canary asserts the SHIPPED selection accepts
# the clip (identify_video -> visible_mark), not just the per-frame detector.

_VIDEO_FRAMES = 90
_VIDEO_FPS = 30


def _video_mark_frame(key: str, w: int, h: int) -> np.ndarray:
    from remove_ai_watermarks.video_visible import _template_sources

    base = base_photo(w, h, seed=11)
    templates = _template_sources()
    short = min(w, h)
    if key == "sora":
        tmpl, scale, x, y = templates["sora-icon"], 0.10, int(w * 0.72), int(h * 0.90)
    elif key == "veo":
        # The legacy "Veo" TEXT form: a perfect synthetic diamond also matches the
        # Sora icon template (both are 4-point stars) and table order hands the
        # tie to Sora, so the gallery carries the discriminative text variant.
        tmpl = templates["veo-text"]
        th = max(6, round(14 * short / 720))
        tw = max(1, round(tmpl.shape[1] * th / tmpl.shape[0]))
        x, y = w - tw - int(0.045 * w), h - th - int(0.045 * h)
        return _composite_light(base, cv2.resize(tmpl, (tw, th)).astype(np.float32) / 255.0, x, y, 250)
    elif key == "seedance":
        tmpl, scale, x, y = templates["seedance"], 0.095, int(w * 0.74), int(h * 0.80)
    elif key == "doubao":
        tmpl, scale, x, y = templates["doubao"], 0.048, int(w * 0.72), int(h * 0.82)
    elif key == "dola":
        tmpl, scale, x, y = templates["dola"], 0.036, int(w * 0.70), int(h * 0.88)
    elif key == "hailuo":
        tmpl, scale, x, y = templates["hailuo"], 0.052, int(w * 0.34), int(h * 0.82)
    elif key == "kling":
        # The FULL mark: swirl logo left of the text run, flush against the
        # bottom-right EDGE. The font arm is edge-gated (region must reach
        # >=0.96W / >=0.94H), and a text-only composite away from the edge both
        # fails that gate and cross-fires the Seedance detector.
        tmpl = templates["kling-1"]
        th = max(8, round(short * 0.040))
        tw = max(1, round(tmpl.shape[1] * th / tmpl.shape[0]))
        tx, ty = w - tw - 6, h - th - 6
        out = _composite_light(base, cv2.resize(tmpl, (tw, th)).astype(np.float32) / 255.0, tx, ty, 250)
        logo = templates["kling-logo"]
        lh = max(6, round(short * 0.046))
        lw = max(1, round(logo.shape[1] * lh / logo.shape[0]))
        lx, ly = tx - lw - round(th * 0.5), h - lh - 6
        return _composite_light(out, cv2.resize(logo, (lw, lh)).astype(np.float32) / 255.0, lx, ly, 250)
    else:
        raise ValueError(key)
    th = max(8, round(short * scale))
    tw = max(1, round(tmpl.shape[1] * th / tmpl.shape[0]))
    alpha = cv2.resize(tmpl, (tw, th)).astype(np.float32) / 255.0
    return _composite_light(base, alpha, x, y, 250)


def build_video(key: str) -> None:
    w, h = 960, 540
    out_dir = OUT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "example.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), _VIDEO_FPS, (w, h))
    if not writer.isOpened():
        raise RuntimeError("mp4v writer unavailable")
    frame = _video_mark_frame(key, w, h)
    for _ in range(_VIDEO_FRAMES):
        writer.write(frame)
    writer.release()


def verify_video(key: str) -> tuple[float, str | None, int]:
    from remove_ai_watermarks.video import identify_video

    rep = identify_video(OUT / key / "example.mp4", check_visible=True)
    return float(rep.visible_detected_frames or 0), rep.visible_mark, rep.total_frames


def render_videos() -> list[str]:
    from remove_ai_watermarks.video import VIDEO_VISIBLE_MARKS

    failures: list[str] = []
    for key in VIDEO_VISIBLE_MARKS:
        build_video(key)
        frames, mark, total = verify_video(key)
        status = "OK " if mark == key else "MISS"
        print(f"{status} {key:10s} video: {mark} on {frames}/{total} frames -> data/fixtures/visible/{key}/example.mp4")
        if mark != key:
            failures.append(key)
    return failures


def main() -> None:
    failures: list[str] = []
    for mark in wr.known_marks():
        key = mark.key
        img = build(key)
        out_dir = OUT / key
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "example.png"
        cv2.imwrite(str(path), img)
        det = wr.get_mark(key).detect(imread(str(path)), provenance=key == "liblib_pill")
        status = "OK " if det.detected else "MISS"
        print(f"{status} {key:12s} conf={det.confidence:.3f} -> {path.relative_to(_ROOT)}")
        if not det.detected:
            failures.append(key)
    failures += render_videos()
    if failures:
        print(f"\nNOT DETECTED on their own examples: {failures}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
