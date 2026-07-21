"""Calibrate a candidate text-mark detector for an UNCOVERED vendor, on real positives.

WHERE THE POSITIVES COME FROM
  `vendor_cohort_harvest.py` partitions China-AIGC carriers into per-entity cohorts by
  their TC260 producer code, so a cohort is a vendor LABEL that owes nothing to any
  pixel detector. That is what makes this calibration non-circular: the previous attempt
  (`render_vendor_silhouettes.py`, 2026-07-18) died at n=1 because the only way it knew
  to find 千问 frames was to eyeball the misses of a detector that cannot see them.

  A cohort is NOT automatically a set of visible-mark positives: TC260 provenance is
  metadata, and a vendor may label a frame without stamping it. So the cohort is the
  CANDIDATE pool, and mark presence is settled by eye -- `--sheets` writes the corner
  crops sorted by score, which makes that pass cheap and makes the separation (or its
  absence) visible directly.

NEGATIVES
  The 432 frames hand-labelled `present: []` in the 2026-07-18 round -- already-adjudicated
  no-visible-mark images, so the false-fire arm rests on human labels rather than on the
  absence of a detection.

THE TRAPS, INHERITED FROM THE 2026-07-18 MEASUREMENT
  Both are encoded below rather than left to the caller:
    * size the template with `alpha_height_frac`, NOT the silhouette's own aspect ratio
      (the latter inflated the clean p99 from 0.30 to 0.58 and made comparison meaningless)
    * keep the ladder at the shipped 3 rungs -- a wide sweep hands clean corners extra
      chances to match, which flatters the positives and the negatives alike

DATA SAFETY
  Corpus images are real user uploads: read-only, local, gitignored output. The template
  is font-rendered synthetic (`render_vendor_silhouettes.py`), never cut from an upload.

    uv run python scripts/vendor_mark_calibrate.py --cohort 91440101MA9Y9T4H7A \\
        --asset qwen_alpha.png --sheets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

REPO = Path(__file__).resolve().parents[1]
COHORTS = REPO / "data" / "spaces" / "_vendor_cohorts.jsonl"
SHEET_DIR = REPO / "data" / "spaces" / "_vendor_calib_sheets"
OUT = REPO / "data" / "spaces" / "_vendor_calibration.jsonl"


def build_config(
    asset: str,
    name: str,
    scale_basis: str = "short",
    overrides: dict[str, Any] | None = None,
) -> Any:
    """A candidate config: doubao's tuned geometry with this vendor's silhouette.

    Transferring doubao's numbers is justified by LAYOUT, not by hope: every one of these
    marks is the same GB 45438-2025 house style -- a 2-glyph vendor prefix, then the
    mandated `AI生成` tail, set in a semibold CJK sans in the bottom-right corner. So
    `豆包AI生成` and `千问AI生成` are the same 6 glyph cells at the same scale, and the
    width/height fractions carry over. The NCC gate does NOT carry over and is what this
    script exists to measure. Any tuned value can be overridden with what
    `--fit-geometry` measured -- inheriting the locate box blindly clipped the big-mode
    qwen mark, which is exactly the trap this tool exists to avoid.
    """
    import dataclasses

    from remove_ai_watermarks._text_mark_engine import TextMarkConfig
    from remove_ai_watermarks.doubao_engine import _CONFIG

    return dataclasses.replace(
        TextMarkConfig(**dataclasses.asdict(_CONFIG)),
        name=name,
        asset_name=asset,
        scale_basis=scale_basis,
        **(overrides or {}),
    )


ScoreArgs = tuple[str, str, str, str, "dict[str, Any]"]


def _score(args: ScoreArgs) -> dict[str, Any] | None:
    path_str, asset, name, basis, overrides = args
    from remove_ai_watermarks._text_mark_engine import TextMarkEngine
    from remove_ai_watermarks.image_io import imread

    img = imread(path_str)
    if img is None or min(img.shape[:2]) < 64:
        return None
    eng = TextMarkEngine(build_config(asset, name, basis, overrides))
    loc = eng.locate(img)
    try:
        score, box = eng._tophat_best(img, loc)
    except Exception:
        return None
    return {"path": path_str, "score": round(float(score), 4), "box": box}


NEGATIVES = REPO / "data" / "spaces" / "_research_20260718_textmark_relaxation" / "groundtruth.jsonl"


def load_sets(cohort: str) -> tuple[list[str], list[str]]:
    pos = [
        json.loads(x)["path"]
        for x in COHORTS.read_text(encoding="utf-8").splitlines()
        if x.strip() and json.loads(x)["uscc"] == cohort
    ]
    # The 2026-07-18 labels are in the vocabulary of the REGISTERED marks only
    # (gemini/doubao/jimeng/jimeng_pill): `present: []` means "no registered mark", NOT
    # "no mark at all" -- 146 of the 432 sit in a TC260 cohort, and qwen-cohort frames
    # visibly carrying 千问AI生成 are labelled `present: []` there (measured 2026-07-21:
    # they made up the clean arm's whole top tail, clean p99 0.37 -> 0.69). A gate read
    # off that arm is meaningless, so the clean arm excludes every frame in ANY TC260
    # cohort -- cohort membership is the cheap proxy for "may carry a CJK AI label".
    in_any_cohort = {
        str(Path(json.loads(x)["path"]).resolve())
        for x in COHORTS.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }
    neg: list[str] = []
    dropped = 0
    for line in NEGATIVES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("present"):
            continue
        p = str((REPO / rec["path"]).resolve())
        if p in in_any_cohort:
            dropped += 1
            continue
        neg.append(p)
    if dropped:
        print(f"clean arm: dropped {dropped} negatives that sit in a TC260 cohort (contamination guard)")
    return pos, neg


def run(
    paths: list[str],
    asset: str,
    name: str,
    workers: int,
    basis: str = "short",
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_score, (p, asset, name, basis, overrides or {})) for p in paths]
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception:  # noqa: S112 -- one bad file must not kill the sweep
                continue
            if r:
                out.append(r)
    return out


def report(pos: list[dict[str, Any]], neg: list[dict[str, Any]], name: str) -> None:
    import numpy as np

    p = np.array([r["score"] for r in pos])
    n = np.array([r["score"] for r in neg])
    print(f"\n{'=' * 78}\n{name}: candidate-cohort vs hand-labelled clean\n{'=' * 78}")
    print(f"\n{'arm':10s} {'n':>5s} {'p10':>7s} {'p50':>7s} {'p90':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s}")
    for label, arr in (("cohort", p), ("clean", n)):
        if not len(arr):
            continue
        qs = [np.percentile(arr, q) for q in (10, 50, 90, 95, 99)]
        print(f"{label:10s} {len(arr):5d} " + " ".join(f"{q:7.3f}" for q in qs) + f" {arr.max():7.3f}")

    if len(p) and len(n):
        print("\n\nOPERATING POINTS -- gate set on the CLEAN arm")
        print("`cohort fire` is an UPPER BOUND on recall: the cohort also holds")
        print("metadata-only frames that carry no visible mark to find.\n")
        print(f"{'gate':>7s} {'clean fire':>12s} {'cohort fire':>13s} {'cohort n':>10s}")
        for q in (90, 95, 99, 99.5, 100):
            t = float(np.percentile(n, q))
            cf, pf = 100 * float((n >= t).mean()), 100 * float((p >= t).mean())
            print(f"{t:7.3f} {cf:11.2f}% {pf:12.1f}% {int((p >= t).sum()):10d}")


def sheets(pos: list[dict[str, Any]], name: str, per_sheet: int = 24) -> None:
    """Corner crops sorted by score, so mark presence and the gate are read in one pass."""
    import cv2
    import numpy as np

    from remove_ai_watermarks._text_mark_engine import TextMarkEngine
    from remove_ai_watermarks.image_io import imread

    eng = TextMarkEngine(build_config("doubao_alpha.png", "roi"))
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    ranked = sorted(pos, key=lambda r: -r["score"])
    width = 660
    for start in range(0, len(ranked), per_sheet):
        chunk = ranked[start : start + per_sheet]
        tiles: list[Any] = []
        for rank, r in enumerate(chunk, start + 1):
            img = imread(r["path"])
            if img is None:
                continue
            loc = eng.locate(img)
            crop = img[loc.y : loc.y + loc.h, loc.x : loc.x + loc.w]
            if not crop.size:
                continue
            tile = cv2.resize(crop, (width, 96), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (118, 22), (0, 0, 0), -1)
            cv2.putText(tile, f"#{rank} {r['score']:.3f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(tile)
            tiles.append(np.full((2, width, 3), 70, np.uint8))
        if tiles:
            dest = SHEET_DIR / f"{name}_ranked_{start // per_sheet:02d}.png"
            cv2.imwrite(str(dest), np.vstack(tiles))
            print(f"  {dest.name}  (#{start + 1}..#{start + len(chunk)})")


# Wide and dense, for the geometry FIT only. An unregistered vendor's glyph size is
# genuinely unknown, which is the one case a dense ladder earns its cost -- but it also
# hands clean corners extra chances to match, so it must never set a gate.
_FIT_SCALES = tuple(round(0.4 * (1.03**i), 4) for i in range(80))  # 0.40 .. ~4.1
# The ladder the product actually ships (`_text_mark_engine._tophat_best`).
_SHIPPED_LADDER = (0.8, 1.0, 1.25)


def _fit_one(args: tuple[str, str]) -> dict[str, Any] | None:
    """Best match over the WIDE ladder, reported as a mark width in pixels.

    Also measures the template ASPECT at the winning width: the mark's true height is
    fitted by sweeping gh at the winning gw and reading the argmax, because
    `alpha_height_frac` must be measured, not taken from the silhouette's own aspect
    (that inflated the clean p99 from 0.30 to 0.58 on the 2026-07-18 attempt) and not
    inherited from doubao.
    """
    path_str, asset = args
    import cv2
    import numpy as np

    from remove_ai_watermarks._text_mark_engine import TextMarkEngine
    from remove_ai_watermarks.image_io import imread

    cfg = build_config(asset, "fit", "width")
    eng = TextMarkEngine(cfg)
    img = imread(path_str)
    if img is None:
        return None
    loc = eng.locate(img)
    resp = eng.tophat_response(img, loc)
    sil = eng._glyph_silhouette()
    if resp is None or sil is None:
        return None
    w = img.shape[1]
    best, best_gw = 0.0, 0
    best_tl = (0, 0)
    for s in _FIT_SCALES:
        gw = max(cfg.min_gw, int(cfg.alpha_width_frac * w * s))
        gh = max(4, int(cfg.alpha_height_frac * w * s))
        if gw >= resp.shape[1] or gh >= resp.shape[0]:
            continue
        t = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(resp, t, cv2.TM_CCOEFF_NORMED)
        _, v, _, tl = cv2.minMaxLoc(res)
        if v > best:
            best, best_gw, best_tl = v, gw, (int(tl[0]), int(tl[1]))
    # Aspect fit at the winning width: sweep gh/gw and keep the argmax. Range covers
    # everything between samsung's 0.12 and jimeng's 0.29 house styles, plus slack.
    best_aspect = 0.0
    if best_gw > 0:
        best_gh_score = -1.0
        for ratio in np.arange(0.12, 0.42, 0.01):
            gh = max(4, int(best_gw * float(ratio)))
            if gh >= resp.shape[0]:
                continue
            t = cv2.resize(sil, (best_gw, gh), interpolation=cv2.INTER_AREA)
            v = float(cv2.matchTemplate(resp, t, cv2.TM_CCOEFF_NORMED).max())
            if v > best_gh_score:
                best_gh_score, best_aspect = v, float(ratio)
    # The ABSOLUTE mark rect, so the LOCATE box fractions can be fitted too: inheriting
    # doubao's corner box clipped the big-mode qwen mark's first glyph (the qwen mark
    # sits ~0.025 of the short side off the right edge, doubao's box assumes ~0.004),
    # which collapsed an exact-size template to 0.26.
    ax = loc.x + best_tl[0]
    ay = loc.y + best_tl[1]
    return {
        "path": path_str,
        "best": round(best, 4),
        "mark_w": best_gw,
        "aspect": round(best_aspect, 3),
        "x": ax,
        "y": ay,
        "w": w,
        "h": img.shape[0],
    }


def fit_geometry(paths: list[str], asset: str, workers: int, floor: float = 0.50, paths_name: str = "cohort") -> None:
    """Which basis and fraction does this vendor's mark actually scale with?

    Only frames matching above ``floor`` are used: below it the winning size is the
    ladder's best fit to background texture, not a measurement of the mark.
    """
    import numpy as np

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_fit_one, (p, asset)) for p in paths]):
            try:
                r = f.result()
            except Exception:  # noqa: S112 -- one bad file must not kill the fit
                continue
            if r:
                rows.append(r)

    strong = [r for r in rows if r["best"] >= floor]
    fit_out = REPO / "data" / "spaces" / f"_vendor_fit_{paths_name}.jsonl"
    fit_out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"\n{'=' * 78}\nGEOMETRY FIT  (n={len(rows)}, usable best>={floor}: {len(strong)})\n{'=' * 78}")
    print(f"rows -> {fit_out}")
    if len(strong) < 20:
        print("too few strong frames to fit a basis -- do not ship a fraction off this")
        return

    mw = np.array([r["mark_w"] for r in strong], float)
    w = np.array([r["w"] for r in strong], float)
    h = np.array([r["h"] for r in strong], float)
    bases = {
        "width": w,
        "height": h,
        "short": np.minimum(w, h),
        "long": np.maximum(w, h),
        "sqrt(w*h)": np.sqrt(w * h),
        "diagonal": np.hypot(w, h),
    }
    print(f"\n{'basis':12s} {'mean frac':>10s} {'CV':>8s} {'p10':>8s} {'p90':>8s}")
    print("-" * 52)
    for nm, b in sorted(bases.items(), key=lambda kv: float(np.std(mw / kv[1]) / np.mean(mw / kv[1]))):
        r = mw / b
        print(
            f"{nm:12s} {np.mean(r):10.4f} {float(np.std(r) / np.mean(r)):8.3f} "
            f"{np.percentile(r, 10):8.4f} {np.percentile(r, 90):8.4f}"
        )

    lo_l, hi_l = _SHIPPED_LADDER[0], _SHIPPED_LADDER[-1]
    print(f"\nCoverage by a single fraction on the SHIPPED ladder ({lo_l} .. {hi_l}, span {hi_l / lo_l:.3f}x):")
    print(f"{'basis':12s} {'frac':>7s} {'window':>16s} {'covered':>9s}")
    for nm in ("short", "sqrt(w*h)", "width"):
        fs = mw / bases[nm]
        best_f, best_cov = 0.0, -1.0
        for f in np.arange(float(fs.min()) * 0.9, float(fs.max()) * 1.1, 0.002):
            cov = float(((fs >= f * lo_l) & (fs <= f * hi_l)).mean())
            if cov > best_cov:
                best_f, best_cov = float(f), cov
        print(f"{nm:12s} {best_f:7.3f} {best_f * lo_l:7.3f}-{best_f * hi_l:.3f} {100 * best_cov:8.1f}%")

    # The raw distribution behind the coverage number: where the mark actually sits,
    # so the mode structure (and what a 4th rung would recover) is visible directly.
    fs = mw / bases["short"]
    qs = [np.percentile(fs, q) for q in (5, 25, 50, 75, 95)]
    print(f"\nfrac_short distribution: p5 {qs[0]:.3f} p25 {qs[1]:.3f} p50 {qs[2]:.3f} p75 {qs[3]:.3f} p95 {qs[4]:.3f}")
    hist, edges = np.histogram(fs, bins=16)
    for c, e0, e1 in zip(hist, edges[:-1], edges[1:], strict=True):
        print(f"  {e0:.3f}-{e1:.3f}  {'#' * c}")

    # Template aspect at the winning width -> the alpha_height_frac recommendation.
    # Measured, per the standing rule: not the silhouette's own aspect, not doubao's.
    aspects = np.array([r["aspect"] for r in strong if r["aspect"] > 0], float)
    if len(aspects) >= 20:
        med = float(np.median(aspects))
        print(
            f"\nASPECT FIT (n={len(aspects)}): p10 {np.percentile(aspects, 10):.3f} "
            f"p50 {med:.3f} p90 {np.percentile(aspects, 90):.3f}"
        )
        print("alpha_height_frac = alpha_width_frac * p50(aspect), per basis:")
        for nm in ("short", "sqrt(w*h)", "width"):
            fxs = mw / bases[nm]
            best_f = max(
                np.arange(float(fxs.min()) * 0.9, float(fxs.max()) * 1.1, 0.002),
                key=lambda f: float(((fxs >= f * lo_l) & (fxs <= f * hi_l)).mean()),
            )
            print(f"  {nm:12s} width {best_f:.3f}  ->  height {best_f * med:.4f}")

    # LOCATE-box fit. The box fractions are as mark-specific as the template size:
    # doubao's box clipped qwen's big-mode mark (see _fit_one). Derive the box from the
    # measured absolute mark rects: margins must not exceed the mark's own (else the
    # mark exits the anchored box), and the box must cover the mark plus NCC slack.
    if len(aspects) >= 20:
        short = np.minimum(w, h).astype(float)
        mark_h = np.array([r["mark_w"] * r["aspect"] for r in strong], float)
        ax = np.array([r["x"] for r in strong], float)
        ay = np.array([r["y"] for r in strong], float)
        right = (w - (ax + mw)) / short  # frame right edge to mark right edge
        bottom = (h - (ay + mark_h)) / short
        print(f"\nLOCATE FIT (basis=short, n={len(strong)}):")
        print(f"  right-margin frac  p5 {np.percentile(right, 5):.4f} p50 {np.percentile(right, 50):.4f}")
        print(f"  bottom-margin frac p5 {np.percentile(bottom, 5):.4f} p50 {np.percentile(bottom, 50):.4f}")
        print(
            f"  mark height frac   p50 {np.percentile(mark_h / short, 50):.4f} "
            f"p95 {np.percentile(mark_h / short, 95):.4f}"
        )
        mx = max(0.002, float(np.percentile(right, 5)) - 0.004)
        mb = max(0.002, float(np.percentile(bottom, 5)) - 0.004)
        need_w = float(np.percentile(mw / short + right, 95)) - mx + 0.02
        need_h = float(np.percentile(mark_h / short + bottom, 95)) - mb + 0.015
        print(f"  recommended: margin_x_frac={mx:.4f} margin_bottom_frac={mb:.4f}")
        print(f"               width_frac={need_w:.3f} height_frac={need_h:.3f}")

    print("\nThese are DIAGNOSTIC. Re-score both arms on the shipped ladder with the")
    print("fitted geometry before reading any gate off the clean arm.")


FIRED = REPO / "data" / "spaces" / "_visible_positives.jsonl"


def _fired_pool(mark: str, limit: int, seed: int = 7) -> list[str]:
    """Paths where ``mark`` fired, from the COMPLETED full-corpus artifact -- the
    standing rule: detector firings are joined, never re-run."""
    import random

    pool = [
        json.loads(x)["path"]
        for x in FIRED.read_text(encoding="utf-8").splitlines()
        if x.strip() and mark in (json.loads(x).get("keys") or [])
    ]
    rng = random.Random(seed)  # noqa: S311 -- reproducible sampling, not crypto
    rng.shuffle(pool)
    return pool[:limit]


def _cross_score(args: tuple[str, Any, Any]) -> dict[str, Any] | None:
    """One frame scored by BOTH the candidate and the doubao production configs."""
    path_str, cand_cfg, db_cfg = args
    from remove_ai_watermarks._text_mark_engine import TextMarkEngine
    from remove_ai_watermarks.image_io import imread

    img = imread(path_str)
    if img is None or min(img.shape[:2]) < 200:
        return None
    out: dict[str, Any] = {"path": path_str}
    for key, cfg in (("cand", cand_cfg), ("doubao", db_cfg)):
        eng = TextMarkEngine(cfg)
        loc = eng.locate(img)
        try:
            score, _ = eng._tophat_best(img, loc)
        except Exception:
            return None
        out[key] = round(float(score), 4)
    return out


def crossfire(
    pools: dict[str, list[str]],
    cand_cfg: Any,
    workers: int,
    gate: float,
    margin: float = 0.10,
) -> None:
    """Score the candidate AND doubao's production template on the same frames.

    The registration question a cohort-vs-clean run cannot answer: the candidate shares
    the mandated `AI生成` tail with doubao (4 of 6 glyph cells), so its template will
    correlate with doubao marks too. If the candidate fires on the doubao pool at the
    candidate gate, registering it double-fills every doubao frame and mislabels it --
    unless the rival margin suppresses it, which then has to be shown NOT to kill the
    candidate on its own marks. Measured here in the tophat domain (the gate's domain);
    production's `_rival_margin_ok` runs the same comparison on the binary blob.
    """
    import numpy as np

    from remove_ai_watermarks.doubao_engine import _CONFIG as db_cfg

    print(f"\n{'=' * 78}\nCROSSFIRE -- candidate vs doubao, same frames, tophat domain\n{'=' * 78}")
    print(f"candidate gate {gate:.3f} | rival margin {margin:.2f}\n")
    print(
        f"{'pool':8s} {'n':>5s} {'cand p50':>9s} {'cand p90':>9s} {'db p50':>7s} {'db p90':>7s} "
        f"{'m-d p10':>8s} {'m-d p50':>8s} {'fire':>7s} {'fire+m':>7s}"
    )
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for pool_name, paths in pools.items():
            rows: list[dict[str, Any]] = []
            futs = [ex.submit(_cross_score, (p, cand_cfg, db_cfg)) for p in paths]
            for f in as_completed(futs):
                try:
                    r = f.result()
                except Exception:  # noqa: S112 -- one bad file must not kill the pool
                    continue
                if r:
                    rows.append(r)
            if not rows:
                continue
            c = np.array([r["cand"] for r in rows])
            d = np.array([r["doubao"] for r in rows])
            m = c - d
            fire = c >= gate
            fire_m = fire & (m >= margin)
            print(
                f"{pool_name:8s} {len(rows):5d} {np.percentile(c, 50):9.3f} {np.percentile(c, 90):9.3f} "
                f"{np.percentile(d, 50):7.3f} {np.percentile(d, 90):7.3f} "
                f"{np.percentile(m, 10):8.3f} {np.percentile(m, 50):8.3f} "
                f"{100 * float(fire.mean()):6.1f}% {100 * float(fire_m.mean()):6.1f}%"
            )
    print("\nReading: on `qwen` the margin column must stay high (the candidate keeps its")
    print("own marks); on `doubao` fire+m must sit near zero (the candidate stays off")
    print("doubao marks). If fire is high on `doubao` and fire+m is not, the rival margin")
    print("is load-bearing for registration; if both are high, the mark cannot be")
    print("registered on this front-end at all.")


def _parse_ladder(raw: str) -> tuple[float, ...] | None:
    if not raw:
        return None
    return tuple(float(x) for x in raw.split(","))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, help="cohort USCC from vendor_cohort_harvest.py")
    ap.add_argument("--asset", required=True, help="silhouette asset name, e.g. qwen_alpha.png")
    ap.add_argument("--name", default="", help="label for output files (defaults to the asset stem)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--scale-basis", choices=("short", "width"), default="short")
    ap.add_argument("--sheets", action="store_true")
    ap.add_argument(
        "--fit-geometry",
        action="store_true",
        help="fit the basis + fraction + template aspect the mark scales with",
    )
    ap.add_argument("--width-frac", type=float, default=None, help="fitted alpha_width_frac (default: inherit doubao)")
    ap.add_argument(
        "--height-frac", type=float, default=None, help="fitted alpha_height_frac (default: inherit doubao)"
    )
    ap.add_argument("--ladder", default="", help="comma scale rungs, e.g. 0.8,1.0,1.25,1.6 (default: shipped 3)")
    ap.add_argument("--gate", type=float, default=0.45, help="candidate gate for the crossfire fire rates")
    ap.add_argument("--box-width-frac", type=float, default=None, help="fitted locate width_frac")
    ap.add_argument("--box-height-frac", type=float, default=None, help="fitted locate height_frac")
    ap.add_argument("--margin-x-frac", type=float, default=None, help="fitted locate margin_x_frac")
    ap.add_argument("--margin-bottom-frac", type=float, default=None, help="fitted locate margin_bottom_frac")
    ap.add_argument(
        "--crossfire",
        action="store_true",
        help="score the candidate AND doubao on the cohort, the doubao/jimeng pools and the clean arm",
    )
    a = ap.parse_args()
    name = a.name or a.asset.split("_")[0]
    ladder = _parse_ladder(a.ladder)
    overrides: dict[str, Any] = {}
    for arg, field in (
        (a.width_frac, "alpha_width_frac"),
        (a.height_frac, "alpha_height_frac"),
        (a.box_width_frac, "width_frac"),
        (a.box_height_frac, "height_frac"),
        (a.margin_x_frac, "margin_x_frac"),
        (a.margin_bottom_frac, "margin_bottom_frac"),
    ):
        if arg is not None:
            overrides[field] = arg
    if ladder is not None:
        overrides["ladder"] = ladder

    pos_paths, neg_paths = load_sets(a.cohort)
    if a.fit_geometry:
        print(f"cohort {a.cohort}: {len(pos_paths)} candidates")
        fit_geometry(pos_paths, a.asset, a.workers, paths_name=name)
        return

    if a.crossfire:
        cand = build_config(a.asset, name, a.scale_basis, overrides)
        pools = {
            "qwen": pos_paths,
            "doubao": _fired_pool("doubao", 400),
            "jimeng": _fired_pool("jimeng", 300),
            "clean": neg_paths,
        }
        print("pools: " + ", ".join(f"{k}={len(v)}" for k, v in pools.items()))
        crossfire(pools, cand, a.workers, a.gate)
        return

    print(f"cohort {a.cohort}: {len(pos_paths)} candidates | clean: {len(neg_paths)} hand-labelled")
    print(f"scale_basis={a.scale_basis} overrides={overrides}")
    pos = run(pos_paths, a.asset, name, a.workers, a.scale_basis, overrides)
    neg = run(neg_paths, a.asset, name, a.workers, a.scale_basis, overrides)
    OUT.write_text(
        "\n".join(json.dumps({**r, "arm": arm}) for arm, rows in (("cohort", pos), ("clean", neg)) for r in rows),
        encoding="utf-8",
    )
    report(pos, neg, name)
    if a.sheets:
        print(f"\nsheets -> {SHEET_DIR}")
        sheets(pos, name)


if __name__ == "__main__":
    main()
