"""Partition China-AIGC carriers into VENDOR COHORTS by their TC260 producer code.

THE PROBLEM THIS SOLVES
  Coverage of uncovered vendors is the largest remaining detection lever
  (`docs/verification-plan.md`, "Where detection work should go next"), and it is
  blocked on EVIDENCE: nothing may be registered off a single frame, and the
  previous session found exactly one confirmed positive each for `千问` and `百度`.
  Harvesting more by PIXELS is circular -- a detector is what we are trying to
  build -- and the generic shared-tail probe is too weak to label with (0.407 on a
  bold positive against a clean p99 of 0.298; see `cjk_tail_probe.py`).

THE KEY
  The TC260 label is not anonymous. Its `ContentProducer` field carries the
  producer's Chinese Unified Social Credit Code (USCC), e.g.
  ``001191110102MACQD9K64010000`` -> USCC ``91110102MACQD9K640``, which names a
  specific legal entity. So the metadata partitions carriers into per-ENTITY
  cohorts without looking at a single pixel. A cohort is a LABEL: once one frame
  in it is eyeballed, every frame in it is a labelled example of that vendor's
  mark. That is what turns "one confirmed positive" into "30+ per vendor".

  CLAUDE.md's "the generic TC260 label names no specific vendor" is about the
  label MARKER (the bare presence of `TC260:AIGC`), which indeed names nobody.
  The producer FIELD inside the block is a different thing and does name one.

  Caveat kept in view: the code names the SIGNING ENTITY, which is not always the
  consumer brand (an aggregator or a cloud host signs for several apps, and one
  vendor can hold several codes). So a cohort is a strong grouping key and a
  hypothesis about the brand -- the brand itself is settled by reading the crop,
  which is what `--sheets` is for.

WHAT IT COSTS
  Metadata only. The expensive pixel pass is NOT re-run: which detectors fired is
  joined from `_visible_positives.jsonl` (the completed full-corpus artifact), per
  the standing rule against relaunching finished sweeps to re-check them.

DATA SAFETY
  Corpus images are real user uploads: read-only, local analysis, gitignored
  output. Contact sheets stay under `data/spaces/`; nothing here is committed.

    uv run python scripts/vendor_cohort_harvest.py
    uv run python scripts/vendor_cohort_harvest.py --report-only --sheets 12
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data" / "spaces" / "originals"
OUT = REPO / "data" / "spaces" / "_vendor_cohorts.jsonl"
FIRED = REPO / "data" / "spaces" / "_visible_positives.jsonl"
SHEET_DIR = REPO / "data" / "spaces" / "_vendor_cohort_sheets"

# A producer code is `001` + `1` + USCC(18) + a 5-digit app/product suffix, so two
# codes sharing the USCC are the same legal entity registering different products.
# Slicing is defensive: anything not matching the layout is grouped by its raw value.
_USCC_START, _USCC_END = 4, 22


def uscc_of(code: str) -> str:
    """The 18-char Unified Social Credit Code embedded in a TC260 producer code."""
    if len(code) >= _USCC_END and code[:3] == "001":
        return code[_USCC_START:_USCC_END]
    return code


def _one(path_str: str) -> dict[str, Any] | None:
    from remove_ai_watermarks.metadata import aigc_label

    try:
        label = aigc_label(Path(path_str))
    except Exception:
        return None
    if not label:
        return None
    producer = str(label.get("ContentProducer") or "")
    return {
        "path": path_str,
        "producer": producer,
        "uscc": uscc_of(producer),
        "propagator": str(label.get("ContentPropagator") or ""),
        "service_provider": str(label.get("ServiceProvider") or ""),
    }


def load_fired() -> dict[str, list[str]]:
    """path -> detector keys that fired, from the completed full-corpus artifact."""
    if not FIRED.exists():
        print(f"WARNING: {FIRED.name} missing; cohorts will show no detector state")
        return {}
    out: dict[str, list[str]] = {}
    for line in FIRED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["path"]] = rec.get("keys") or []
    return out


def scan(limit: int, workers: int, out_path: Path) -> list[dict[str, Any]]:
    pool = sorted(glob.glob(str(CORPUS / "*" / "*")))
    if limit:
        pool = pool[:limit]
    print(f"scanning {len(pool)} corpus files for TC260 labels  workers={workers}", flush=True)
    rows: list[dict[str, Any]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh, ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, p) for p in pool]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception:  # noqa: S112 -- one bad file must not kill the scan
                continue
            if rec is not None:
                fh.write(json.dumps(rec) + "\n")
                rows.append(rec)
            if i % 5000 == 0:
                fh.flush()
                print(f"  {i}/{len(pool)}  carriers={len(rows)}", flush=True)
    return rows


def report(rows: list[dict[str, Any]], fired: dict[str, list[str]], min_size: int) -> None:
    by_uscc: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_uscc.setdefault(r["uscc"], []).append(r)

    print(f"\n{'=' * 92}\nVENDOR COHORTS  ({len(rows)} TC260 carriers, {len(by_uscc)} distinct entities)\n{'=' * 92}")
    print("\n`fires` = share of the cohort where SOME registered detector fires.")
    print("A large cohort with a low fire rate is an uncovered vendor -- the harvest target.\n")
    print(f"{'entity (USCC)':22s} {'n':>6s} {'fires':>7s}  {'detectors seen':38s} {'products':>8s}")
    print("-" * 92)

    cohorts = sorted(by_uscc.items(), key=lambda kv: -len(kv[1]))
    for uscc, members in cohorts:
        if len(members) < min_size:
            continue
        keys: Counter[str] = Counter()
        hit = 0
        for m in members:
            ks = fired.get(m["path"], [])
            if ks:
                hit += 1
            keys.update(ks)
        seen = ", ".join(f"{k}:{c}" for k, c in keys.most_common(4)) or "-- none --"
        products = len({m["producer"] for m in members})
        print(f"{uscc:22s} {len(members):6d} {100 * hit / len(members):6.1f}%  {seen:38s} {products:8d}")

    small = sum(1 for _, m in cohorts if len(m) < min_size)
    if small:
        print(f"\n({small} cohorts below --min-size {min_size} not shown)")


def _bands(img: Any, width: int, band: int) -> list[Any]:
    """Full-width top and bottom bands, scaled to a readable common width.

    An unregistered vendor's placement is unknown, so cropping the bottom-RIGHT
    corner (where the marks we already cover happen to sit) would beg the
    question. Full-width bands catch any horizontal position, and the two bands
    together cover every corner the standard's implementers actually use.
    """
    import cv2

    h = img.shape[0]
    strip = max(24, int(h * band))
    out = []
    for piece in (img[:strip], img[h - strip :]):
        scale = width / max(1, piece.shape[1])
        out.append(cv2.resize(piece, (width, max(12, int(piece.shape[0] * scale))), interpolation=cv2.INTER_AREA))
    return out


def sheets(rows: list[dict[str, Any]], fired: dict[str, list[str]], per: int, min_size: int) -> None:
    """Top/bottom bands per uncovered cohort, so the vendor and mark can be read off."""
    import cv2
    import numpy as np

    from remove_ai_watermarks.image_io import imread

    by_uscc: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_uscc.setdefault(r["uscc"], []).append(r)

    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting contact sheets -> {SHEET_DIR}")

    for uscc, members in sorted(by_uscc.items(), key=lambda kv: -len(kv[1])):
        if len(members) < min_size:
            continue
        quiet = [m for m in members if not fired.get(m["path"])]
        if not quiet:
            continue
        width = 900
        tiles: list[Any] = []
        for m in quiet[:per]:
            img = imread(m["path"])
            if img is None:
                continue
            for b in _bands(img, width, 0.10):
                tiles.append(b)
                tiles.append(np.full((3, width, 3), 60, np.uint8))
        if tiles:
            dest = SHEET_DIR / f"{uscc}_n{len(members)}_quiet{len(quiet)}.png"
            cv2.imwrite(str(dest), np.vstack(tiles))
            print(f"  {dest.name}  ({len(tiles) // 4} frames)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap files scanned (0 = whole corpus)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--min-size", type=int, default=5, help="hide cohorts smaller than this")
    ap.add_argument("--sheets", type=int, default=0, help="crops per cohort contact sheet")
    a = ap.parse_args()

    if a.report_only:
        rows = [json.loads(x) for x in a.out.read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        rows = scan(a.limit, a.workers, a.out)

    fired = load_fired()
    report(rows, fired, a.min_size)
    if a.sheets:
        sheets(rows, fired, a.sheets, a.min_size)


if __name__ == "__main__":
    main()
