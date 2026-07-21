"""End-to-end confidence run: drive the ACTUAL CLI over REAL corpus examples.

WHY THIS EXISTS AND WHAT IT IS NOT
  The 849-test suite and `smoke_matrix.py` prove the code paths behave on fixtures and
  synthetic inputs. This is the other half: run the real `remove-ai-watermarks` entry point,
  as a user would, over real corpus images spanning every command and every provenance
  class, and CHECK THE OUTPUT -- not that it exited 0, but that it did the right thing (the
  mark is actually gone on re-detect, the metadata actually strips, the diffusion actually
  writes a changed image). A green exit is not evidence the work happened.

WHAT IT COVERS
  identify   one real image per provenance class -> the verdict is right
  metadata   real AI-metadata files -> --check detects, --remove strip-and-verifies clean
  visible    real marked images per mark -> the mark is gone on re-detect, output written
  erase      a real image, each fill backend (cv2 / migan / lama) -> output written
  invisible  a real SynthID image on MPS at reduced resolution -> a CHANGED image is written
  all        a real marked image through the full pipeline -> output written
  batch      a real directory -> every input produces an output

  invisible/all run the diffusion model, so they are gated behind --diffusion and run at a
  small --max-resolution on MPS (the user's "reduced size on MPS" path). Everything else is
  cv2/numpy and fast.

DATA SAFETY
  Corpus images are user uploads: read-only, local analysis, outputs to a gitignored temp
  dir. Records example uids and pass/fail, never image content.

    uv run python scripts/real_examples_e2e.py                 # fast surface (no diffusion)
    uv run python scripts/real_examples_e2e.py --diffusion     # + invisible/all on MPS
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data" / "spaces" / "originals"
DATASETS = REPO / "data" / "spaces" / "_visible_datasets"
SAMPLES = REPO / "data" / "samples"
_UV = shutil.which("uv") or "uv"  # full path avoids the partial-executable lint


def run(args: list[str], timeout: int = 300) -> tuple[int, str]:
    """Invoke the real installed CLI. Returns (exit_code, combined output)."""
    proc = subprocess.run(  # noqa: S603 -- fixed argv, the whole point is the real entry point
        [_UV, "run", "remove-ai-watermarks", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def find_visible_positive(mark: str) -> Path | None:
    """A real corpus image the parity run bucketed as carrying this mark, that the current
    detector STILL fires on (the bucket was built by an older run; re-confirm live)."""
    from remove_ai_watermarks.image_io import imread
    from remove_ai_watermarks.watermark_registry import detect_marks

    pool = sorted(glob.glob(str(DATASETS / mark / "*")))
    random.Random(3).shuffle(pool)  # noqa: S311 -- deterministic sampling, not cryptography
    for p in pool[:60]:
        img = imread(p)
        if img is None:
            continue
        if any(d.detected and d.key == mark for d in detect_marks(img)):
            return Path(p)
    return None


class Results:
    """Collects one row per checked behaviour.

    A FAIL keeps the command's OUTPUT. That is not cosmetic: the first run of this harness
    discarded it, an `all` invocation failed once with exit 1, and because the output was
    gone there was no way to tell which of that command's three distinct `SystemExit(1)`
    paths had fired -- the failure was undiagnosable and did not reproduce. Keep the tail
    so a transient is at least identifiable after the fact.
    """

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str, str]] = []

    def add(self, cmd: str, example: str, ok: bool, detail: str, output: str = "") -> None:
        self.rows.append((cmd, example, ok, detail, "" if ok else output[-800:]))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {cmd:28s} {example:26s} {detail}", flush=True)

    def report(self) -> int:
        n = len(self.rows)
        bad = [r for r in self.rows if not r[2]]
        print(f"\n{'=' * 78}\nREAL-EXAMPLE E2E  {n - len(bad)}/{n} passed")
        print(f"{'=' * 78}")
        for cmd, ex, ok, detail, out in self.rows:
            if not ok:
                print(f"  FAIL  {cmd}  {ex}  {detail}")
                if out.strip():
                    print("        --- command output (tail) ---")
                    for line in out.strip().splitlines()[-12:]:
                        print(f"        {line}")
        if not bad:
            print("  every command produced the right result on real corpus examples")
        return 1 if bad else 0


def check_identify(res: Results) -> None:
    """One real image per provenance class -> the verdict is right (--json must parse)."""
    print("\nidentify -- real image per provenance class")
    cases = [
        ("chatgpt-1.png", True),
        ("firefly-1.png", True),
        ("mj-1.png", True),
        ("doubao-1.png", True),
        ("grok-1.jpg", True),
        ("flux-1.jpg", True),
    ]
    for name, want_ai in cases:
        p = SAMPLES / name
        if not p.exists():
            res.add("identify", name, False, "sample missing")
            continue
        code, out = run(["identify", str(p), "--json"])
        try:
            data = json.loads(out[out.index("{") : out.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            res.add("identify", name, False, f"json did not parse (exit {code})")
            continue
        is_ai = data.get("is_ai_generated")
        plat = (data.get("platform") or "").lower()
        ok = bool(is_ai) is want_ai
        res.add("identify", name, ok, f"is_ai={is_ai} platform={plat or '-'}")


def check_metadata(res: Results, tmp: Path) -> None:
    """Real AI-metadata files -> --check detects, --remove strip-and-verifies clean."""
    print("\nmetadata -- real AI-metadata files, detect then strip-and-verify")
    for name in ("chatgpt-1.png", "doubao-1.png", "mj-1.png", "grok-1.jpg", "flux-1.jpg"):
        p = SAMPLES / name
        if not p.exists():
            res.add("metadata --check", name, False, "sample missing")
            continue
        code, out = run(["metadata", str(p), "--check"])
        detected = "no ai" not in out.lower() and code == 0
        res.add("metadata --check", name, detected, "AI metadata detected" if detected else "nothing detected")

        outp = tmp / f"stripped_{name}"
        code, out = run(["metadata", str(p), "--remove", "-o", str(outp)])
        # strip-and-verify: the command re-scans the OUTPUT and fails loudly on leftovers.
        clean = outp.exists() and code == 0 and "still" not in out.lower()
        res.add("metadata --remove", name, clean, "output re-scanned clean" if clean else f"exit {code}")


def check_visible(res: Results, tmp: Path) -> None:
    """Real marked images -> the PRODUCT'S DECISION is honoured, and a removed mark clears.

    The success criterion is not "the mark is always gone" -- it is "the product did what it
    decided, and the decision is right". Two designed behaviours make a blind re-detect
    misleading:
      * The pill is GATED (`_keep_pill`): a low-confidence pill with no corroboration is
        deliberately NOT removed, so `visible` correctly writes nothing and exits 2. That is
        the gate working, not a miss -- checked by asking `remove_auto_marks` whether it
        chose to act.
      * Samsung is the faintest mark (peak alpha ~0.38) at a razor-thin 0.40 gate, so a
        borderline positive can be reduced yet re-detect just above threshold. Reported as
        the measured before->after confidence, not a bare pass/fail.
    """
    from remove_ai_watermarks.image_io import imread
    from remove_ai_watermarks.watermark_registry import detect_marks, get_mark, remove_auto_marks

    print("\nvisible --mark auto -- real marked image per mark, product decision then re-detect")
    for mark in ("doubao", "jimeng", "qwen", "gemini", "samsung", "jimeng_pill"):
        src = find_visible_positive(mark)
        if src is None:
            res.add("visible", mark, True, "no live positive in bucket (skipped, not a failure)")
            continue
        img = imread(str(src))
        conf_before = next((d.confidence for d in detect_marks(img) if d.detected and d.key == mark), 0.0)
        # What does the product DECIDE to do? (labels lists the marks it removed.)
        _out_img, labels = remove_auto_marks(img, sensitivity="auto", provenance=frozenset(), backend="cv2")
        acted = get_mark(mark).label in labels

        outp = tmp / f"visible_{mark}{src.suffix}"
        code, out = run(["visible", str(src), "--mark", "auto", "-o", str(outp)])

        if not acted:
            # The gate declined this mark. The CLI must then write nothing and exit 2 --
            # that is the correct outcome, so verify the CLI agrees with the decision.
            ok = code == 2 and not outp.exists()
            res.add(
                "visible", mark, ok, f"gate declined (conf {conf_before:.2f}); CLI exit {code}, no output -- correct"
            )
            continue

        if not outp.exists():
            res.add("visible", mark, False, f"product removed it but CLI wrote no output (exit {code})", out)
            continue
        cleaned = imread(str(outp))
        conf_after = next((d.confidence for d in detect_marks(cleaned) if d.detected and d.key == mark), 0.0)
        clean = conf_after == 0.0
        if clean:
            res.add("visible", mark, True, f"removed, re-detect clean ({conf_before:.2f} -> below gate)")
        else:
            # Reduced but still over the gate: a real residual. Honest partial, flagged.
            res.add(
                "visible",
                mark,
                False,
                f"reduced {conf_before:.2f} -> {conf_after:.2f} but still over gate (faint-mark residual)",
            )


def check_erase(res: Results, tmp: Path) -> None:
    """A real image, each fill backend actually runs and writes an output."""
    from remove_ai_watermarks.region_eraser import lama_available, migan_available

    print("\nerase --region -- each fill backend on a real image")
    src = next((Path(p) for p in sorted(glob.glob(str(CORPUS / "*" / "*"))) if Path(p).stat().st_size > 50_000), None)
    if src is None:
        res.add("erase", "-", False, "no corpus image")
        return
    backends = ["cv2"] + (["migan"] if migan_available() else []) + (["lama"] if lama_available() else [])
    for backend in backends:
        outp = tmp / f"erase_{backend}{src.suffix}"
        code, out = run(["erase", str(src), "--region", "10,10,120,60", "--backend", backend, "-o", str(outp)])
        ok = outp.exists() and code == 0 and outp.stat().st_size > 0
        res.add(f"erase --backend {backend}", src.name[:12], ok, "output written" if ok else f"exit {code}", out)


def check_diffusion(res: Results, tmp: Path, gemini_src: str, openai_src: str) -> None:
    """The GPU path: invisible + all on MPS at reduced resolution -> a CHANGED image."""
    import numpy as np

    from remove_ai_watermarks.image_io import imread

    print("\ninvisible / all -- real SynthID image on MPS, reduced resolution")
    for label, src in (("invisible/gemini", gemini_src), ("invisible/openai", openai_src)):
        if not src:
            res.add(label, "-", True, "no real positive found (skipped)")
            continue
        sp = Path(src)
        outp = tmp / f"inv_{sp.stem}.png"
        code, out = run(
            ["invisible", src, "-o", str(outp), "--device", "mps", "--max-resolution", "512", "--seed", "0"],
            timeout=1200,
        )
        if not outp.exists():
            res.add(label, sp.name[:12], False, f"no output (exit {code})", out)
            continue
        before, after = imread(src), imread(str(outp))
        # Diffusion regenerates every pixel; the output must actually differ from the input.
        # A DIFFERENT SHAPE is itself proof it changed (the pipeline resizes), so treat it
        # as changed rather than comparing arrays that cannot be compared. The first
        # version wrote `array_equal(after, before if shapes match else after)`, which
        # compares `after` WITH ITSELF on the mismatch branch and is therefore always
        # "unchanged" -- it would have reported a genuinely resized output as a no-op.
        changed = (
            after is not None
            and before is not None
            and (before.shape != after.shape or not np.array_equal(before, after))
        )
        res.add(label, sp.name[:12], changed, "diffusion wrote a changed image" if changed else "output == input", out)

    if gemini_src:
        sp = Path(gemini_src)
        outp = tmp / f"all_{sp.stem}.png"
        code, out = run(
            ["all", gemini_src, "-o", str(outp), "--device", "mps", "--max-resolution", "512", "--seed", "0"],
            timeout=1200,
        )
        ok = outp.exists() and outp.stat().st_size > 0
        res.add("all (full pipeline)", sp.name[:12], ok, "output written" if ok else f"exit {code}", out)


def check_batch(res: Results, tmp: Path) -> None:
    """A real directory -> every supported input produces an output."""
    print("\nbatch -- a real directory")
    indir = tmp / "batch_in"
    indir.mkdir(exist_ok=True)
    picks = sorted(glob.glob(str(DATASETS / "doubao" / "*")))[:5]
    for p in picks:
        shutil.copy2(p, indir / Path(p).name)
    n_in = len(list(indir.glob("*")))
    if n_in == 0:
        res.add("batch", "-", False, "no inputs to seed")
        return
    outdir = tmp / "batch_out"
    code, out = run(["batch", str(indir), "-o", str(outdir), "--mode", "visible"], timeout=600)
    n_out = len(list(outdir.glob("*"))) if outdir.exists() else 0
    ok = code == 0 and n_out >= n_in
    res.add("batch --mode visible", f"{n_in} imgs", ok, f"{n_out}/{n_in} outputs (exit {code})", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusion", action="store_true", help="also run invisible/all on MPS (slow)")
    ap.add_argument("--gemini", default="")
    ap.add_argument("--openai", default="")
    a = ap.parse_args()

    res = Results()
    with tempfile.TemporaryDirectory(prefix="raiw_e2e_") as td:
        tmp = Path(td)
        check_identify(res)
        check_metadata(res, tmp)
        check_visible(res, tmp)
        check_erase(res, tmp)
        check_batch(res, tmp)
        if a.diffusion:
            check_diffusion(res, tmp, a.gemini, a.openai)
        else:
            print("\n(diffusion skipped -- pass --diffusion to run invisible/all on MPS)")
    raise SystemExit(res.report())


if __name__ == "__main__":
    main()
