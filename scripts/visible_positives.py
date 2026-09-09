"""Parallel detection pass: list every corpus image carrying a known visible mark.

Why this exists separately from `visible_removal_audit.py`: that audit is single-process,
so repeated full-dataset sweeps waste time. Its
expensive half is DETECTION, and detection does not depend on the fill backend. Splitting
it out means detecting once in parallel and then feeding the positives to the audit via
its `--paths-file` seam, over a few thousand images instead of forty thousand.

CRASH TOLERANCE IS NOT OPTIONAL AT THIS SCALE
  cv2/libpng can crash natively on malformed images. A plain `ProcessPoolExecutor.map`
  over a large dataset can deadlock: the worker dies without a Python traceback and the parent
  waits forever on a result that never arrives (observed 2026-07-19 -- 26 min of work lost
  because results were only written at the end). So this script:
    * writes every result to JSONL as it arrives -- a kill never costs more than a batch;
    * is resumable -- an interrupted run skips what it already recorded;
    * runs each image in a fresh interpreter with a per-image timeout and bounded
      concurrency; a poisoned file is recorded without retrying it in the parent.

Treat input datasets as sensitive and read-only, and keep output gitignored.

    uv run python scripts/visible_positives.py --jobs 6
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from _isolated_image_workers import run_batch

# The package's own format set. An inlined copy here silently skipped .heif, which
# CLAUDE.md documents as supported.
from remove_ai_watermarks._internal.constants import SUPPORTED_FORMATS as _EXTS

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / ".local-eval" / "originals"
OUT = REPO / ".local-eval" / "visible-positives.jsonl"
PATHS = REPO / ".local-eval" / "visible-positives.txt"


def _one(path: str) -> dict[str, object]:
    from remove_ai_watermarks.image_io import imread
    from remove_ai_watermarks.watermark_registry import detect_marks

    try:
        img = imread(path)
        if img is None:
            return {"path": path, "keys": [], "status": "unreadable"}
        return {"path": path, "keys": [d.key for d in detect_marks(img) if d.detected], "status": "ok"}
    except Exception as e:
        return {"path": path, "keys": [], "status": f"error:{type(e).__name__}"}


def _run_batch(batch: list[str], jobs: int, timeout: int) -> list[dict[str, object]]:
    """Bound each image in an independent interpreter, including native failures."""
    return run_batch(Path(__file__), batch, jobs=jobs, timeout=timeout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=600, help="seconds per image before terminating its isolated worker")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--restart", action="store_true")
    a = ap.parse_args()

    files = sorted(p for p in glob.glob(str(CORPUS / "*" / "*")) if Path(p).suffix.lower() in _EXTS)
    if a.limit:
        files = files[: a.limit]

    done: set[str] = set()
    if a.restart and a.out.exists():
        a.out.unlink()  # --restart must TRUNCATE; the file is reopened in append mode below
    if a.out.exists() and not a.restart:
        with open(a.out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["path"])
                except Exception:  # noqa: S112 -- tolerate a torn last line
                    continue
    todo = [p for p in files if p not in done]
    print(f"images {len(files)}  already done {len(done)}  to do {len(todo)}  jobs {a.jobs}", flush=True)

    counts: collections.Counter[str] = collections.Counter()
    status: collections.Counter[str] = collections.Counter()
    seen = 0
    with open(a.out, "a", encoding="utf-8") as fh:
        for start in range(0, len(todo), a.batch):
            for rec in _run_batch(todo[start : start + a.batch], a.jobs, a.timeout):
                fh.write(json.dumps(rec) + "\n")
                status[str(rec["status"])] += 1
                for k in rec["keys"]:  # type: ignore[union-attr]
                    counts[str(k)] += 1
                seen += 1
            fh.flush()
            print(f"  {seen}/{len(todo)}  {dict(counts)}  {dict(status)}", flush=True)

    # Rebuild the paths file from the FULL record set, not just this run's slice.
    hits = []
    with open(a.out, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:  # noqa: S112
                continue
            if r.get("keys"):
                hits.append(r["path"])
    # Derive the paths file from --out so a trial run with a scratch --out cannot
    # overwrite the shared list a full sweep produced.
    paths_out = a.out.with_suffix(".txt")
    paths_out.write_text("\n".join(sorted(set(hits))) + "\n", encoding="utf-8")
    print(f"\npositives: {len(set(hits))} images")
    for k, v in counts.most_common():
        print(f"   {v:6d}  {k}")
    print(f"statuses: {dict(status)}\npaths -> {paths_out}\nrecords -> {a.out}")


if __name__ == "__main__":
    main()
