"""Build an UNBIASED random sample for measuring visible-mark RECALL.

Every earlier labeling round sampled where detectors FIRED, so images that every
detector missed were absent by construction and recall was unmeasurable. This round
samples at random within a provenance class and shows the labeler the corners where
a mark can physically be, so a MISSED mark is visible as such.

Design decisions that matter:

* SAMPLING FRAME is per provenance class, not the whole corpus. Recall is only
  meaningful against a denominator where the mark CAN occur: TC260 carriers for the
  ByteDance marks and the pill, Google-C2PA for the sparkle. Reporting one blended
  recall over all uploads would mostly measure how often each vendor appears.
* NATIVE RESOLUTION crops, never a downscaled whole image: a 220px preview destroys a
  faint mark (measured in an earlier round), which would inflate the miss count with
  the labeler's own blindness rather than the detector's.
* BOTH corners per image (top-left pill, bottom-right wordmark/strip/sparkle), so one
  pass adjudicates every registered mark instead of one mark per crop.
* The detector's verdict is NOT shown and is not in the sheet order -- the manifest
  holds it and must not be opened until labeling ends.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from remove_ai_watermarks.image_io import imread

CELL_W = 300
COLS, ROWS = 4, 3
PER = COLS * ROWS


def corner_strip(img: NDArray[Any]) -> NDArray[Any] | None:
    """Top-left and bottom-right corners stacked, at (near) native scale.

    Crop size is a fraction of the SHORT side because that is what marks scale with
    (China's GB 45438-2025 sizes the mandated label off the shortest side).
    """
    h, w = img.shape[:2]
    short = min(h, w)
    cw, ch = int(short * 0.46), int(short * 0.17)
    cw, ch = min(cw, w), min(ch, h)
    tl = img[0:ch, 0:cw]
    br = img[h - ch : h, w - cw : w]
    strip = np.vstack([tl, np.full((6, cw, 3), 90, np.uint8), br])
    s = min(1.0, CELL_W / strip.shape[1])  # never UPSCALE past native
    if s < 1.0:
        strip = cv2.resize(strip, (CELL_W, max(1, int(strip.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return strip


def unique_content_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per file digest, independently of detector observations."""
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        with Path(row["path"]).open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        unique.setdefault(digest, row)
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path, help="JSONL corpus scan produced by the visible evaluation harness")
    parser.add_argument("output", type=Path, help="Directory for contact sheets and the blinded manifest")
    parser.add_argument("--tc260", type=int, default=160, help="Number of TC260 carriers to sample")
    parser.add_argument("--google", type=int, default=80, help="Number of Google-provenance carriers to sample")
    args = parser.parse_args()
    scan = args.scan
    out = args.output
    n_tc260 = args.tc260
    n_google = args.google
    out.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(line) for line in scan.open() if '"marks"' in line]
    uniq = unique_content_rows(recs)

    tc = [r for r in uniq if r["cls"] == "tc260"]
    goog = [r for r in uniq if r["cls"] == "neg" and "Google" in r.get("platform", "")]
    rng = random.Random(2026)  # noqa: S311 -- sampling, not cryptography
    rng.shuffle(tc)
    rng.shuffle(goog)
    picked = [("tc260", r) for r in tc[:n_tc260]] + [("google", r) for r in goog[:n_google]]
    rng.shuffle(picked)

    manifest, cells = [], []
    for cls, r in picked:
        img = imread(r["path"])
        if img is None:
            continue
        strip = corner_strip(img)
        if strip is None:
            continue
        cells.append(strip)
        manifest.append(
            {
                "idx": len(cells) - 1,
                "uid": r["uid"],
                "cls": cls,
                "path": r["path"],
                "fired": "|".join(sorted(k for k, m in r["marks"].items() if m["strict"])),
                **{f"ncc_{k}": m["conf"] for k, m in r["marks"].items()},
            }
        )

    cell_h = max(c.shape[0] for c in cells)
    for si in range(0, len(cells), PER):
        chunk = cells[si : si + PER]
        sheet = np.full((ROWS * (cell_h + 26), COLS * (CELL_W + 8), 3), 40, np.uint8)
        for i, c in enumerate(chunk):
            rr, cc = divmod(i, COLS)
            y0, x0 = rr * (cell_h + 26) + 22, cc * (CELL_W + 8) + 4
            sheet[y0 : y0 + c.shape[0], x0 : x0 + c.shape[1]] = c
            cv2.putText(sheet, f"{si + i:03d}", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(out / f"r{si // PER:02d}.png"), sheet)

    with (out / "MANIFEST_DO_NOT_OPEN.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    print(f"cells={len(cells)}  sheets={(len(cells) + PER - 1) // PER} -> {out}")
    print("each cell = TOP-LEFT corner above, BOTTOM-RIGHT corner below, near native scale")


if __name__ == "__main__":
    main()
