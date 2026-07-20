"""Tier B1: measure visible-mark FILL quality against a constructed ground truth.

THE PROBLEM THIS SOLVES
  A real watermarked image has no clean counterpart, so fill quality has only ever been
  eyeballed ("cv2 smears on texture, LaMa is best") and never carried a number. The docs
  state a preference order for `--backend auto` that rests on no measurement.

THE CONSTRUCTION
  Take a VERIFIED-CLEAN corpus image, stamp a known mark onto it using the mark's own
  captured alpha map (the forward model the reverse-alpha work established:
  `stamped = (1-a)*bg + a*white`), then remove it and compare against the TRUE original.
  The original is the answer key that reality never provides.

TWO MEASUREMENT DECISIONS THAT MAKE OR BREAK THIS
  * Score INSIDE the footprint only. The fill touches a tiny corner, so whole-frame PSNR
    sits near 60 dB whatever the backend does and would rank them all "excellent".
  * Report the DAMAGE baseline (stamped vs original) next to the recovery (filled vs
    original). Without it a PSNR is uninterpretable: 30 dB is a triumph if the mark cost
    12 dB and a failure if it cost 29 dB. The honest metric is how much of the gap the
    fill closes.

WHAT IT DOES NOT MEASURE
  Detection. The mark is localized with `force=True`, so a miss cannot contaminate the
  fill numbers -- this isolates the FILL. Detection accuracy is Tier C's job.

DATA SAFETY
  Corpus images are user uploads: read-only, local analysis, output under a gitignored
  data/spaces/ path. No image content is written into the report.

    uv run python scripts/fill_quality.py --n 60
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from invisible_quality_audit import _ssim  # reuse, do not reimplement a third SSIM

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data" / "spaces" / "originals"
OUT = REPO / "data" / "spaces" / "_fill_quality.jsonl"

# Text marks: a bundled alpha PNG plus the engine's own corner geometry.
STAMPABLE = ("doubao", "jimeng", "samsung")
# Marks with no bundled alpha asset. Both expose a default footprint via
# `footprint_mask(force=True)`, so they are stamped by fitting their own alpha source
# into that slot: Gemini's alpha is derived from its background captures, the pill's
# from its synthetic font-rendered silhouette.
SLOT_STAMPABLE = ("gemini", "jimeng_pill")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * float(np.log10(255.0**2 / mse))


def texture_of(box: np.ndarray) -> float:
    """Median Sobel magnitude -- the same texture proxy the pill's flatness gate uses."""
    g = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY) if box.ndim == 3 else box
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.median(cv2.magnitude(gx, gy)))


def engine_for(mark_key: str) -> Any:
    """The TextMarkEngine instance for a mark key (the registry does not expose one)."""
    import importlib

    mod = importlib.import_module(f"remove_ai_watermarks.{mark_key}_engine")
    cls = next(
        obj
        for name, obj in vars(mod).items()
        # Exclude the imported base class: it takes a `config` arg, the subclasses do not.
        if isinstance(obj, type) and name.endswith("Engine") and name != "TextMarkEngine"
    )
    return cls()


def stamp(image: np.ndarray, mark_key: str) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Composite a mark's alpha glyph into its canonical corner. Returns (stamped, bbox)."""
    from remove_ai_watermarks._text_mark_engine import load_alpha_template

    engine = engine_for(mark_key)
    cfg = engine.config
    alpha = load_alpha_template(cfg.asset_name)
    if alpha is None:
        return None

    loc = engine.locate(image)
    base = engine.scale_base(image)
    gw = max(cfg.min_gw, int(cfg.alpha_width_frac * base))
    gh = max(4, int(cfg.alpha_height_frac * base))
    if gw < 8 or gh < 4 or gw > loc.w or gh > loc.h:
        return None

    a = cv2.resize(alpha, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32)
    x = loc.x + (loc.w - gw) // 2
    y = loc.y + (loc.h - gh) // 2
    out = image.copy()
    roi = out[y : y + gh, x : x + gw].astype(np.float32)
    a3 = a[..., None]
    out[y : y + gh, x : x + gw] = np.clip(roi * (1 - a3) + 255.0 * a3, 0, 255).astype(np.uint8)
    return out, (x, y, gw, gh)


def _slot_alpha(mark_key: str) -> np.ndarray | None:
    """The alpha source for a mark that ships no bundled alpha PNG."""
    if mark_key == "gemini":
        from remove_ai_watermarks.gemini_engine import _shared_engine

        # _shared_engine is the package's own lru_cache singleton: constructing
        # GeminiEngine() per image reloads its captures, recomputes both alpha maps and
        # rebuilds the whole template ladder -- the exact cost that singleton exists to
        # avoid. Private attribute access is deliberate (no accessor exists); a
        # measurement script may reach in, product code must not.
        return np.asarray(_shared_engine()._alpha_large, dtype=np.float32)
    if mark_key == "jimeng_pill":
        from remove_ai_watermarks._text_mark_engine import load_alpha_template

        return load_alpha_template("jimeng_pill.png")
    return None


def stamp_slot(image: np.ndarray, mark_key: str) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Stamp a mark into its OWN default footprint slot (the `--no-detect` geometry).

    Used for gemini and the pill, which have no bundled alpha asset with corner
    fractions. The mark is fitted into the middle of its slot so the fill has to
    recover the same kind of region it would in production.
    """
    from remove_ai_watermarks.watermark_registry import get_mark

    alpha = _slot_alpha(mark_key)
    if alpha is None:
        return None
    loc = get_mark(mark_key).localize(image, force=True)
    if loc.mask is None:
        return None
    ys, xs = np.nonzero(loc.mask)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    gw, gh = max(8, int(bw * 0.7)), max(8, int(bh * 0.7))
    if gw < 8 or gh < 8 or gw > image.shape[1] or gh > image.shape[0]:
        return None
    a = cv2.resize(alpha, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32)
    a = np.clip(a, 0.0, 1.0)
    x, y = x0 + (bw - gw) // 2, y0 + (bh - gh) // 2
    out = image.copy()
    roi = out[y : y + gh, x : x + gw].astype(np.float32)
    a3 = a[..., None]
    out[y : y + gh, x : x + gw] = np.clip(roi * (1 - a3) + 255.0 * a3, 0, 255).astype(np.uint8)
    return out, (x, y, gw, gh)


def stamp_any(image: np.ndarray, mark_key: str) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    return stamp(image, mark_key) if mark_key in STAMPABLE else stamp_slot(image, mark_key)


def clean_sources(n: int, seed: int = 11) -> list[Path]:
    """Corpus images with NO metadata signal and NO mark detection -- verified clean."""
    from remove_ai_watermarks.identify import identify
    from remove_ai_watermarks.image_io import imread
    from remove_ai_watermarks.watermark_registry import detect_marks

    pool = glob.glob(str(CORPUS / "*" / "*"))
    random.Random(seed).shuffle(pool)  # noqa: S311 -- deterministic sampling, not crypto
    out: list[Path] = []
    for p in pool:
        if len(out) >= n:
            break
        path = Path(p)
        try:
            img = imread(str(path))
            if img is None or min(img.shape[:2]) < 400:
                continue
            # check_visible=False: identify would otherwise decode the file a second
            # time and run the very same visible detectors as detect_marks below.
            if identify(path, check_visible=False).signals:
                continue
            if any(d.detected for d in detect_marks(img)):
                continue
        except Exception:  # noqa: S112 -- a bad corpus file just is not a candidate
            continue
        out.append(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="clean source images")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    from remove_ai_watermarks.image_io import imread
    from remove_ai_watermarks.region_eraser import lama_available, migan_available
    from remove_ai_watermarks.watermark_registry import fill, get_mark

    backends = ["cv2"] + (["migan"] if migan_available() else []) + (["lama"] if lama_available() else [])
    print(f"backends under test: {backends}")
    print(f"selecting {a.n} verified-clean sources...", flush=True)
    sources = clean_sources(a.n)
    print(f"got {len(sources)}\n", flush=True)

    rows: list[dict[str, Any]] = []
    with open(a.out, "w", encoding="utf-8") as fh:
        for i, src in enumerate(sources, 1):
            base = imread(str(src))
            if base is None:
                continue
            for key in (*STAMPABLE, *SLOT_STAMPABLE):
                st = stamp_any(base, key)
                if st is None:
                    continue
                stamped, (x, y, w, h) = st
                # Score a slightly padded box: the fill dilates, so a tight glyph box
                # would miss damage the backend does just outside the glyph.
                pad = 8
                y0, y1 = max(0, y - pad), min(base.shape[0], y + h + pad)
                x0, x1 = max(0, x - pad), min(base.shape[1], x + w + pad)
                truth = base[y0:y1, x0:x1]
                dmg_psnr = psnr(stamped[y0:y1, x0:x1], truth)
                dmg_ssim = _ssim(
                    cv2.cvtColor(stamped[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY)
                )
                tex = texture_of(truth)

                mark = get_mark(key)
                loc = mark.localize(stamped, force=True)
                if loc.mask is None:
                    continue
                for backend in backends:
                    try:
                        filled = fill(stamped, loc.mask, backend=backend)
                    except Exception as e:
                        rows.append({"src": src.name, "mark": key, "backend": backend, "error": str(e)[:150]})
                        continue
                    # Invariant: the fill must not touch anything outside its mask.
                    outside = loc.mask == 0
                    untouched = bool(np.array_equal(filled[outside], stamped[outside]))
                    rec = {
                        "src": src.name,
                        "mark": key,
                        "backend": backend,
                        "texture": round(tex, 2),
                        "damage_psnr": round(dmg_psnr, 2),
                        "damage_ssim": round(dmg_ssim, 4),
                        "filled_psnr": round(psnr(filled[y0:y1, x0:x1], truth), 2),
                        "filled_ssim": round(
                            _ssim(
                                cv2.cvtColor(filled[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY),
                                cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY),
                            ),
                            4,
                        ),
                        "outside_mask_untouched": untouched,
                    }
                    rows.append(rec)
                    fh.write(json.dumps(rec) + "\n")
            if i % 10 == 0:
                fh.flush()
                print(f"  {i}/{len(sources)}", flush=True)

    report(rows, a.out)


def report(rows: list[dict[str, Any]], out: Path) -> None:
    good = [r for r in rows if "error" not in r]
    if not good:
        print("no measurements")
        return
    tex = sorted(r["texture"] for r in good)
    t1, t2 = tex[len(tex) // 3], tex[2 * len(tex) // 3]

    def bucket(t: float) -> str:
        return "flat" if t <= t1 else ("mid" if t <= t2 else "textured")

    print(f"\n{'=' * 78}\nFILL QUALITY  n={len(good)} measurements  (texture terciles at {t1:.1f} / {t2:.1f})")
    print(f"{'=' * 78}")
    print("recovered = filled PSNR minus damaged PSNR; how much of the mark's damage the fill undoes\n")
    hdr = f"{'mark':9s} {'backend':8s} {'bg':9s} {'n':>4s} {'damaged':>9s} {'filled':>9s}"
    print(f"{hdr} {'recovered':>10s} {'ssim':>7s}")
    agg: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in good:
        agg[(r["mark"], r["backend"], bucket(r["texture"]))].append(r)
    for (mark, backend, bg), rs in sorted(agg.items()):
        # MEDIAN, not mean: a fill that reproduces a flat background EXACTLY scores
        # PSNR=inf, and a single inf makes the mean inf -- which reported "+inf" for
        # every flat bucket on the first run and hid the real numbers.
        finite = [r for r in rs if np.isfinite(r["filled_psnr"])]
        perfect = len(rs) - len(finite)
        d = float(np.median([r["damage_psnr"] for r in rs]))
        f = float(np.median([r["filled_psnr"] for r in finite])) if finite else float("inf")
        sv = float(np.median([r["filled_ssim"] for r in rs]))
        # Recovery is the MEDIAN OF THE PER-IMAGE DIFFERENCES, not the difference of the
        # two medians. Those are not the same statistic on skewed data and they can even
        # disagree in SIGN: the first version of this report subtracted medians and
        # produced a backend ranking that a paired comparison did not support.
        rec = float(np.median([r["filled_psnr"] - r["damage_psnr"] for r in finite])) if finite else float("inf")
        tail = f"  ({perfect} exact)" if perfect else ""
        print(f"{mark:9s} {backend:8s} {bg:9s} {len(rs):4d} {d:9.2f} {f:9.2f} {rec:+10.2f} {sv:7.4f}{tail}")

    bad = [r for r in good if not r["outside_mask_untouched"]]
    print(f"\noutside-mask invariant violations: {len(bad)}" + (" <-- BUG" if bad else " (none)"))
    print(f"records: {out}")


if __name__ == "__main__":
    main()
