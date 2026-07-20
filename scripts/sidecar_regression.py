"""Tier A1: diff today's `identify` against the verdicts recorded in the corpus sidecars.

`data/spaces/identify/<day>/<uid>.json` holds the verdict a past run produced for
`data/spaces/originals/<day>/<uid>_src.<ext>`. Re-running identify and diffing turns the
corpus into a ~39k-image behavioral regression suite that needs no new labelling.

WHAT A DIFF MEANS -- READ THIS BEFORE PANICKING
  The sidecars were written by OLDER versions, so an intended improvement shows up as a
  diff exactly like a regression does. The output is therefore CLASSIFIED, not pass/fail:

    lost_ai        verdict was AI, now is not          <- the alarm; almost always a bug
    lost_signal    a signal family stopped firing      <- the alarm
    new_ai         verdict was not AI, now is          <- usually an improvement
    new_signal     a signal family started firing      <- usually an improvement
    platform       attribution changed
    confidence     confidence level changed
    unchanged      identical on every compared axis

  Only `lost_*` is a regression by default. Everything else needs a look before the
  baseline is moved.

WHY FAMILIES, NOT RAW STRINGS
  Watermark descriptions are human-readable prose and their WORDING has changed between
  versions ("C2PA Content Credentials (OpenAI)" vs "... (OpenAI, Truepic)"). Diffing raw
  strings would report every rewording as a lost+new signal pair and bury the real
  regressions. Signals are normalized to families (c2pa, synthid, visible_sparkle, ...)
  so the comparison tracks BEHAVIOR, not phrasing.

DATA SAFETY
  Corpus images are user uploads: read-only, local analysis. Output goes to a gitignored
  path under data/spaces/ and records uids, never image content.

    uv run python scripts/sidecar_regression.py --sample 500     # representative trial
    uv run python scripts/sidecar_regression.py                  # full corpus, resumable

Use `--sample` for a trial, never `--limit`: the sidecar list is sorted by day, so
`--limit N` reads only the EARLIEST day and its result does not generalize (the first
trial run of this script took 200 files that were all 2026-05-29).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO = Path(__file__).resolve().parents[1]
IDENTIFY_DIR = REPO / "data" / "spaces" / "identify"
ORIGINALS = REPO / "data" / "spaces" / "originals"
OUT = REPO / "data" / "spaces" / "_sidecar_regression.jsonl"

# Map a watermark description to a stable behavior family. Order matters: the first
# matching pattern wins, so put the specific tokens above the generic ones.
_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("synthid", ("synthid",)),
    ("visible_sparkle", ("sparkle",)),
    ("visible_doubao", ("豆包", "doubao")),
    ("visible_jimeng", ("即梦", "jimeng")),
    ("visible_samsung", ("galaxy ai", "generati dall")),
    ("aigc_tc260", ("aigc", "tc260")),
    ("iptc", ("iptc", "digitalsourcetype", "made with ai", "made-with-ai")),
    ("c2pa", ("c2pa", "content credentials")),
    ("trustmark", ("trustmark",)),
    ("open_invisible", ("open invisible", "dwt", "stable diffusion xl")),
    ("xai_signature", ("xai", "grok signature")),
    ("exif_generator", ("exif", "software tag", "png text")),
)


def family_of(description: str) -> str:
    low = description.lower()
    for name, tokens in _FAMILIES:
        if any(t in low for t in tokens):
            return name
    return "other"


def families(descriptions: list[str]) -> set[str]:
    return {family_of(d) for d in descriptions or []}


def compare(sidecar: dict, report: object) -> dict:
    """Classify the difference between a recorded verdict and a fresh one."""
    old_ai, new_ai = sidecar.get("is_ai_generated"), getattr(report, "is_ai_generated", None)
    old_fam = families(sidecar.get("watermarks") or [])
    new_fam = families(list(getattr(report, "watermarks", []) or []))
    old_plat, new_plat = sidecar.get("platform"), getattr(report, "platform", None)
    old_conf, new_conf = sidecar.get("confidence"), getattr(report, "confidence", None)

    classes: list[str] = []
    # Treat only a real True->not-True transition as lost. None means "unknown", and the
    # library never asserts False, so None->None is not a change.
    if bool(old_ai) and not bool(new_ai):
        classes.append("lost_ai")
    if not bool(old_ai) and bool(new_ai):
        classes.append("new_ai")
    if old_fam - new_fam:
        classes.append("lost_signal")
    if new_fam - old_fam:
        classes.append("new_signal")
    if old_plat != new_plat:
        classes.append("platform")
    if old_conf != new_conf:
        classes.append("confidence")

    return {
        "classes": classes or ["unchanged"],
        "old_ai": old_ai,
        "new_ai": new_ai,
        "lost_families": sorted(old_fam - new_fam),
        "new_families": sorted(new_fam - old_fam),
        "old_platform": old_plat,
        "new_platform": new_plat,
        "old_confidence": old_conf,
        "new_confidence": new_conf,
    }


def _one(sidecar_path: str) -> dict:
    uid = os.path.basename(sidecar_path)[:-5]
    day = os.path.basename(os.path.dirname(sidecar_path))
    try:
        with open(sidecar_path, encoding="utf-8") as fh:
            sidecar = json.load(fh)
    except Exception as e:  # a corrupt sidecar must not kill the sweep
        return {"uid": uid, "day": day, "status": "sidecar_unreadable", "error": str(e)[:200]}

    src = sidecar.get("src") or ""
    image = ORIGINALS / day / src
    if not src or not image.exists():
        found = glob.glob(str(ORIGINALS / day / f"{uid}*"))
        if not found:
            return {"uid": uid, "day": day, "status": "image_missing"}
        image = Path(found[0])

    try:
        from remove_ai_watermarks.identify import identify

        report = identify(image)
    except Exception as e:  # record and continue; a crash IS a finding
        return {"uid": uid, "day": day, "status": "identify_raised", "error": f"{type(e).__name__}: {e}"[:300]}

    return {"uid": uid, "day": day, "status": "ok", **compare(sidecar, report)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only N sidecars (trial run)")
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help="randomly sample N across ALL days -- use this for a trial, not --limit: "
        "the sidecar list is sorted, so --limit takes one day and is not representative",
    )
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--restart", action="store_true", help="ignore existing output and start over")
    a = ap.parse_args()

    sidecars = sorted(glob.glob(str(IDENTIFY_DIR / "*" / "*.json")))
    if a.sample:
        random.Random(7).shuffle(sidecars)  # noqa: S311 -- deterministic sampling, not crypto
        sidecars = sorted(sidecars[: a.sample])
    elif a.limit:
        sidecars = sidecars[: a.limit]
    if not sidecars:
        raise SystemExit(f"no sidecars under {IDENTIFY_DIR}")

    # Resume: a full sweep is ~1 h, so never redo work an interrupted run already did.
    done: set[str] = set()
    if a.restart and a.out.exists():
        a.out.unlink()  # --restart must TRUNCATE; the file is reopened in append mode below
    if a.out.exists() and not a.restart:
        with open(a.out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["uid"])
                except Exception:  # noqa: S112 -- tolerate a torn last line from an interrupted run
                    continue
    todo = [s for s in sidecars if os.path.basename(s)[:-5] not in done]
    print(f"sidecars {len(sidecars)}  already done {len(done)}  to do {len(todo)}  workers {a.workers}")

    counts: collections.Counter[str] = collections.Counter()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fh, ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures = {ex.submit(_one, s): s for s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["status"] != "ok":
                counts[rec["status"]] += 1
            else:
                for c in rec["classes"]:
                    counts[c] += 1
            if i % 500 == 0:
                fh.flush()
                print(f"  {i}/{len(todo)}  " + "  ".join(f"{k}={v}" for k, v in counts.most_common(6)), flush=True)

    print(f"\n{'=' * 70}\nSIDECAR REGRESSION  processed={len(todo)}\n{'=' * 70}")
    for k, v in counts.most_common():
        flag = "  <-- REGRESSION" if k.startswith("lost_") or k in ("identify_raised",) else ""
        print(f"  {k:22s} {v:7d}{flag}")
    print(f"\nfull records: {a.out}")


if __name__ == "__main__":
    main()
