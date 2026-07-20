"""Measure the Jimeng pill on the PRODUCT path, not the raw detector.

WHY THIS EXISTS
  `visible_removal_audit.py` calls `get_mark(key).remove` directly, which bypasses
  `_keep_pill`. For the pill that is the wrong path and it reads as a disaster: on a
  300-image sample the raw path was "detector still fires after removal" 68-75% of the
  time. The product does not do that -- `--mark auto` gates the pill hard (never on
  Doubao; unrestricted only when the bottom-right wordmark fired; otherwise only on a
  flat footprint). Measured through the gate on the same sample, 7 of 80 raw detections
  survived to removal and 6 of those were corroborated real.

  That sample was too small to state a precision (95% CI 49-97%). This script runs the
  gated path over EVERY pill positive in the corpus so the interval is usable.

THE CORROBORATION PROXY AND ITS BIAS
  "Real pill" here means: TC260 metadata names Jimeng, OR the bottom-right "★ 即梦AI"
  wordmark is detected. Both are independent of the pill detector, which is the point.
  But the proxy MISSES a real pill on a metadata-stripped screenshot with no visible
  wordmark -- so measured precision is a LOWER BOUND, not a point estimate. Do not quote
  it as if it were exact.

Corpus images are user uploads: read-only, local analysis, gitignored output.

    uv run python scripts/pill_gate_audit.py --jobs 7
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO = Path(__file__).resolve().parents[1]
POSITIVES = REPO / "data" / "spaces" / "_visible_positives.jsonl"
OUT = REPO / "data" / "spaces" / "_pill_gate_audit.jsonl"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def _one(path_str: str) -> dict[str, object]:
    from remove_ai_watermarks.api import visible_provenance
    from remove_ai_watermarks.image_io import imread
    from remove_ai_watermarks.pill_engine import PillEngine
    from remove_ai_watermarks.watermark_registry import detect_marks, get_mark, remove_auto_marks

    rec: dict[str, object] = {"path": Path(path_str).name, "status": "ok"}
    try:
        img = imread(path_str)
        if img is None:
            return {**rec, "status": "unreadable"}
        prov = visible_provenance(Path(path_str))
        dets = {d.key: d for d in detect_marks(img) if d.detected}
        wordmark = "jimeng" in dets
        tc260 = "jimeng" in prov
        eng = PillEngine()
        _, labels = remove_auto_marks(img, sensitivity="auto", provenance=prov, backend="cv2")
        # Match the EXACT pill label. A substring test on "AI生成" also matches Doubao's
        # label ("Doubao 豆包AI生成 text"), which counted Doubao removals as pill removals
        # and inflated the pill's measured precision.
        pill_label = get_mark("jimeng_pill").label
        removed = any(str(x) == pill_label for x in labels)
        # The gate reads the ARBITRATED keys, not strict detection: jimeng can be accepted
        # on its relaxed (provenance-confirmed) arm, which `detect_marks` never shows. Take
        # the arm from what was actually removed, or every such case lands in "other".
        jimeng_decided = any(str(x) == get_mark("jimeng").label for x in labels)
        return {
            **rec,
            "removed_by_product": bool(removed),
            "corroborated": bool(wordmark or tc260),
            "wordmark": bool(wordmark),
            "jimeng_decided": bool(jimeng_decided),
            "tc260": bool(tc260),
            "doubao_fired": "doubao" in dets,
            "footprint_flat": bool(eng.footprint_is_flat(img)),
        }
    except Exception as e:
        return {**rec, "status": f"error:{type(e).__name__}"}


def _batch(paths: list[str], jobs: int, timeout: int) -> list[dict[str, object]]:
    try:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(_one, p) for p in paths]
            return [f.result() for f in as_completed(futs, timeout=timeout)]
    except (FutureTimeout, BrokenProcessPool, OSError, RuntimeError):
        out = []
        for p in paths:
            try:
                out.append(_one(p))
            except BaseException:  # a native crash costs one file, not the sweep
                out.append({"path": Path(p).name, "status": "crashed"})
        return out


def report(rows: list[dict[str, object]]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        print("no usable rows")
        return
    removed = [r for r in ok if r["removed_by_product"]]
    corro = [r for r in ok if r["corroborated"]]
    tp = [r for r in removed if r["corroborated"]]
    missed = [r for r in corro if not r["removed_by_product"]]

    print(f"\n{'=' * 76}\nJIMENG PILL ON THE PRODUCT PATH   n={len(ok)} raw detections\n{'=' * 76}")
    print(f"raw detections corroborated as real : {len(corro):5d}  ({100 * len(corro) / len(ok):.1f}%)")
    print(f"the gate lets through to removal    : {len(removed):5d}  ({100 * len(removed) / len(ok):.1f}%)")
    if removed:
        lo, hi = wilson(len(tp), len(removed))
        pct = 100 * len(tp) / len(removed)
        print(f"  of those, corroborated real       : {len(tp):5d}  -> precision {pct:.1f}% (95% CI {lo:.1f}-{hi:.1f})")
    if corro:
        lo, hi = wilson(len(tp), len(corro))
        pct = 100 * len(tp) / len(corro)
        head = f"corroborated pills removed          : {len(tp):5d}/{len(corro)}"
        print(f"{head}  -> recall {pct:.1f}% (95% CI {lo:.1f}-{hi:.1f})")
    print(f"corroborated pills the gate suppressed: {len(missed):5d}")
    print("\nprecision is a LOWER BOUND: the corroboration proxy cannot see a real pill on a")
    print("metadata-stripped image whose wordmark is absent or missed.\n")

    print("which gate arm let it through:")
    arms: collections.Counter[str] = collections.Counter()
    for r in removed:
        if r.get("jimeng_decided"):
            arms["jimeng wordmark accepted -> unrestricted"] += 1
        elif r["tc260"]:
            arms["tc260 metadata + flat footprint"] += 1
        else:
            arms["other (unexpected -- the gate has no third arm)"] += 1
    for k, v in arms.most_common():
        print(f"   {v:5d}  {k}")

    print("\nwhy the rest were suppressed:")
    sup: collections.Counter[str] = collections.Counter()
    for r in ok:
        if r["removed_by_product"]:
            continue
        if r["doubao_fired"]:
            sup["doubao fired (pill never rides on it)"] += 1
        elif not r["corroborated"]:
            sup["no corroboration (no wordmark, no TC260)"] += 1
        elif not r["footprint_flat"]:
            sup["corroborated but footprint too textured"] += 1
        else:
            sup["other"] += 1
    for k, v in sup.most_common():
        print(f"   {v:5d}  {k}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--restart", action="store_true")
    a = ap.parse_args()

    paths = []
    with open(POSITIVES, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:  # noqa: S112
                continue
            if "jimeng_pill" in (r.get("keys") or []):
                paths.append(r["path"])
    paths.sort()
    if a.limit:
        paths = paths[: a.limit]

    done: set[str] = set()
    if a.restart and a.out.exists():
        a.out.unlink()  # --restart must TRUNCATE; appending kept the old rows in the report
    if a.out.exists() and not a.restart:
        with open(a.out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["path"])
                except Exception:  # noqa: S112
                    continue
    todo = [p for p in paths if Path(p).name not in done]
    print(f"pill positives {len(paths)}  done {len(done)}  to do {len(todo)}  jobs {a.jobs}", flush=True)

    rows: list[dict[str, object]] = []
    with open(a.out, "a", encoding="utf-8") as fh:
        for i in range(0, len(todo), a.batch):
            for rec in _batch(todo[i : i + a.batch], a.jobs, a.timeout):
                fh.write(json.dumps(rec) + "\n")
                rows.append(rec)
            fh.flush()
            print(f"  {len(rows)}/{len(todo)}", flush=True)

    with open(a.out, encoding="utf-8") as fh:
        allrows = []
        for line in fh:
            try:
                allrows.append(json.loads(line))
            except Exception:  # noqa: S112
                continue
    report(allrows)
    print(f"records -> {a.out}")


if __name__ == "__main__":
    main()
