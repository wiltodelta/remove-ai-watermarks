"""Tier B4: peak RSS and wall time per fill backend across input sizes.

WHY THIS EXISTS
  The docs describe the backends in CAPABILITY prose -- "MI-GAN is the memory-tight pick",
  "big-LaMa does not fit a minimal droplet", "~0.6-0.9 GB regardless of upload size". Those
  numbers were measured once, informally, and are now load-bearing for a real deployment
  decision about which backend each hosting tier can afford. This measures them.

WHAT IT MEASURES
  For each (backend, input size): peak RSS of a FRESH process doing exactly one erase, and
  the wall time. A fresh subprocess per measurement is the point -- peak RSS inside a
  long-lived process is contaminated by whatever ran before it, and the question here is
  what a per-request worker actually needs.

  The mask is a fixed, small corner region at every size, because the claim under test is
  that the learned backends crop around the mask and so their memory is bounded by the MARK
  size, not the image size. If that holds, the curve is flat in input size; if it does not,
  it climbs and the droplet sizing is wrong.

READING IT
  RSS is the peak resident set of the whole worker, which includes the interpreter, numpy,
  cv2 and (for the learned backends) onnxruntime plus the model. That is the honest number
  for sizing a container -- not the model tensor alone.

DATA SAFETY
  Generates its own synthetic inputs. Reads nothing from the corpus, writes nothing tracked.

    uv run python scripts/resource_ceilings.py                  # cv2 + whatever is installed
    uv run python scripts/resource_ceilings.py --max-mp 25      # push to 25 MP
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO = Path(__file__).resolve().parents[1]
_UV = shutil.which("uv") or "uv"

# (label, width, height) -- 1 MP up to 25 MP, the range a phone photo upload spans.
SIZES = (("1MP", 1000, 1000), ("4MP", 2000, 2000), ("12MP", 4000, 3000), ("25MP", 5000, 5000))

# The child process: build an image, erase one small corner region, report peak RSS.
# Kept as a string so each measurement is a genuinely fresh interpreter.
_CHILD = """
import json, resource, sys, time
import numpy as np
from remove_ai_watermarks.region_eraser import erase

w, h, backend = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(0)
# Textured, not flat: a flat image compresses and can let a backend shortcut work.
img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
box = (w - 320, h - 90, 300, 70)   # a mark-sized corner region at every input size
t0 = time.monotonic()
try:
    out = erase(img, boxes=[box], backend=backend)
    # Assert the fill actually RAN. A wrong call signature or a silently-declining
    # backend would otherwise report the process's numpy footprint as if it were the
    # backend's cost -- the first version of this script did exactly that on all 12
    # cells and the numbers looked plausible.
    # Compare ONLY the mask box. A full-frame `(out != img).any()` allocates a boolean
    # temp the size of the image BEFORE getrusage is read -- 75 MB at 25 MP, ~17% of the
    # cv2 figure, and it GROWS with the input, so the harness would partly manufacture
    # the very "cv2 scales with input size" conclusion it is measuring.
    bx, by, bw_, bh_ = box
    changed = int((out[by:by+bh_, bx:bx+bw_] != img[by:by+bh_, bx:bx+bw_]).any())
    err = "" if changed else "NO-OP: backend did not modify the masked region"
except Exception as e:
    err = f"{type(e).__name__}: {e}"[:200]
elapsed = time.monotonic() - t0
peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print("RESULT" + json.dumps({"peak_kb": peak_kb, "sec": round(elapsed, 2), "error": err}))
"""


def peak_rss_mb(peak_kb: int) -> float:
    """ru_maxrss is BYTES on macOS and KILOBYTES on Linux -- normalize to MB.

    Getting this wrong silently reports 1024x off, which would look like a dramatic
    finding rather than a unit bug.
    """
    return peak_kb / (1024 * 1024) if sys.platform == "darwin" else peak_kb / 1024


def measure(backend: str, w: int, h: int, timeout: int = 900) -> dict[str, object] | None:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(_CHILD)
        child = fh.name
    try:
        p = subprocess.run(  # noqa: S603 -- fixed argv, our own child script
            [_UV, "run", "python", child, str(w), str(h), backend],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"TIMEOUT >{timeout}s", "sec": float(timeout), "peak_mb": float("nan")}
    finally:
        Path(child).unlink(missing_ok=True)

    line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT")), None)
    if line is None:
        return {"error": (p.stderr or p.stdout)[-200:], "sec": 0.0, "peak_mb": float("nan")}
    data = json.loads(line[len("RESULT") :])
    return {"error": data["error"], "sec": data["sec"], "peak_mb": round(peak_rss_mb(data["peak_kb"]), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mp", type=int, default=25, help="skip sizes above this megapixel count")
    a = ap.parse_args()

    from remove_ai_watermarks.region_eraser import lama_available, migan_available

    backends = ["cv2"] + (["migan"] if migan_available() else []) + (["lama"] if lama_available() else [])
    sizes = [s for s in SIZES if (s[1] * s[2]) / 1e6 <= a.max_mp + 0.5]
    print(f"backends: {backends}\nsizes: {[s[0] for s in sizes]}")
    print("\nOne FRESH process per cell; the mask is a fixed small corner at every size.")
    print("If the learned backends really crop around the mask, RSS stays flat in input size.\n")

    print(f"{'backend':8s} {'size':6s} {'peak RSS':>11s} {'wall':>8s}  note")
    rows: list[tuple[str, str, dict[str, object]]] = []
    for backend in backends:
        for label, w, h in sizes:
            r = measure(backend, w, h)
            if r is None:
                continue
            rows.append((backend, label, r))
            note = str(r["error"])[:48] or "ok"
            print(f"{backend:8s} {label:6s} {r['peak_mb']:8} MB {r['sec']:7}s  {note}", flush=True)

    print(f"\n{'=' * 72}\nRESOURCE CEILINGS\n{'=' * 72}")
    for backend in backends:
        cells = [(lbl, r) for b, lbl, r in rows if b == backend and not r["error"]]
        if not cells:
            continue
        peaks = [float(r["peak_mb"]) for _, r in cells]  # type: ignore[arg-type]
        lo, hi = min(peaks), max(peaks)
        growth = "flat in input size" if hi <= lo * 1.6 else f"GROWS {hi / max(lo, 0.1):.1f}x with input size"
        print(f"  {backend:6s} peak {lo:.0f}-{hi:.0f} MB across {cells[0][0]}..{cells[-1][0]}  -- {growth}")
    print("\nRSS is the whole worker (interpreter + numpy + cv2 + any model), i.e. the")
    print("number to size a container with, not the model tensor alone.")


if __name__ == "__main__":
    main()
