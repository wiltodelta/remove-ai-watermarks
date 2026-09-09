"""Tier E: does the real CLI fail GRACEFULLY on adversarial and degenerate inputs?

WHAT "PASS" MEANS HERE
  Not "it succeeded" -- most of these inputs SHOULD be rejected. A pass is a graceful
  outcome: a clear message, a sane exit code, and **no unhandled traceback**, within the
  timeout. The failure modes this is hunting are the ones a user actually hits and that no
  unit test covers, because unit tests feed well-formed fixtures:

    * an unhandled traceback  -- the library crashed instead of reporting
    * a hang                  -- worse than a crash for a batch caller
    * a silent success on garbage -- it "processed" a corrupt file and wrote something

  A non-zero exit with a readable error is a PASS. A Python traceback is a FAIL even when
  the exit code looks tidy.

WHY THESE INPUTS
  The suite covers truncated files, mismatched extensions, Unicode filenames, and
  concurrent access because a wrapping service may run
  concurrent jobs against one path. Decompression bombs and absurd geometry are the cheap
  denial-of-service shapes any tool taking untrusted files must survive.

DATA SAFETY
  Builds its own inputs (synthetic, or truncated copies of committed fixtures) inside a
  temp dir. Reads corpus images read-only for the one large-input case. Writes nothing
  tracked.

    uv run python scripts/robustness_suite.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "data" / "fixtures" / "provenance"
_UV = shutil.which("uv") or "uv"

# A traceback in the output means the failure escaped the error handling, whatever the
# exit code says.
_CRASH_MARKERS = ("Traceback (most recent call last)", "Fatal Python error", "Segmentation fault")


class Results:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str, str]] = []

    def add(self, case: str, cmd: str, ok: bool, detail: str, output: str = "") -> None:
        self.rows.append((case, cmd, ok, detail, "" if ok else output[-600:]))
        print(f"  [{'PASS' if ok else 'FAIL'}] {case:34s} {cmd:10s} {detail}", flush=True)

    def report(self) -> int:
        bad = [r for r in self.rows if not r[2]]
        print(f"\n{'=' * 78}\nROBUSTNESS (Tier E)  {len(self.rows) - len(bad)}/{len(self.rows)} graceful")
        print(f"{'=' * 78}")
        for case, cmd, ok, detail, out in self.rows:
            if not ok:
                print(f"  FAIL  {case}  ({cmd})  {detail}")
                for line in out.strip().splitlines()[-10:]:
                    print(f"        {line}")
        if not bad:
            print("  every adversarial input was handled without a crash or a hang")
        return 1 if bad else 0


def run(args: list[str], timeout: int = 120) -> tuple[int, str, bool]:
    """Returns (exit_code, output, timed_out)."""
    try:
        p = subprocess.run(  # noqa: S603 -- fixed argv, driving our own CLI on purpose
            [_UV, "run", "remove-ai-watermarks", *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (-1, "", True)
    return (p.returncode, p.stdout + p.stderr, False)


def graceful(res: Results, case: str, cmd: str, args: list[str], timeout: int = 120) -> None:
    """Run one adversarial case and score it on gracefulness, not on success."""
    code, out, timed_out = run(args, timeout=timeout)
    if timed_out:
        res.add(case, cmd, False, f"HUNG (> {timeout}s)", out)
        return
    if code not in (0, 1, 2):
        res.add(case, cmd, False, f"unexpected process exit {code}", out)
        return
    crashed = next((m for m in _CRASH_MARKERS if m in out), None)
    if crashed:
        res.add(case, cmd, False, f"unhandled {crashed} (exit {code})", out)
        return
    res.add(case, cmd, True, f"handled cleanly, exit {code}")


def make_inputs(tmp: Path) -> dict[str, Path]:
    """Build the adversarial corpus from representative malformed inputs."""
    import numpy as np

    from remove_ai_watermarks.image_io import imwrite

    made: dict[str, Path] = {}

    # A well-formed baseline, so a failure elsewhere is attributable to the input.
    good = tmp / "good.png"
    imwrite(good, np.full((600, 800, 3), 128, np.uint8))
    made["good"] = good

    # Truncated PNG cut mid-stream.
    src = SAMPLES / "chatgpt-1.png"
    if src.exists():
        raw = src.read_bytes()
        trunc = tmp / "truncated.png"
        trunc.write_bytes(raw[: len(raw) // 3])
        made["truncated"] = trunc

    # Corrupt: correct magic bytes, garbage body.
    corrupt = tmp / "corrupt.png"
    corrupt.write_bytes(b"\x89PNG\r\n\x1a\n" + os.urandom(4096))
    made["corrupt"] = corrupt

    # Zero-byte file with a valid extension.
    empty = tmp / "empty.png"
    empty.write_bytes(b"")
    made["empty"] = empty

    # Not an image at all, but named like one.
    text = tmp / "actually_text.jpg"
    text.write_bytes(b"this is not an image, it is a text file pretending\n" * 50)
    made["not_an_image"] = text

    # Degenerate geometry: a 1x1, and a 1-pixel-tall sliver (the shape that once faulted
    # cv2's GaussianBlur natively on Windows).
    imwrite(tmp / "tiny.png", np.full((1, 1, 3), 200, np.uint8))
    made["tiny_1x1"] = tmp / "tiny.png"
    imwrite(tmp / "sliver.png", np.full((1, 4000, 3), 200, np.uint8))
    made["sliver_1x4000"] = tmp / "sliver.png"

    # Decompression bomb: a tiny file that decodes to a huge canvas. Hand-built so the
    # on-disk size stays trivial while the declared dimensions are enormous.
    bomb = tmp / "bomb.png"
    bomb.write_bytes(_png_bomb(16000, 16000))
    made["decompression_bomb"] = bomb

    # Unicode + RTL filenames (issue #17 was Unicode-safe IO).
    uni = tmp / "изображение-测试-🎨.png"
    shutil.copy2(good, uni)
    made["unicode_filename"] = uni
    rtl = tmp / "صورة-اختبار.png"
    shutil.copy2(good, rtl)
    made["rtl_filename"] = rtl

    # Mismatched extension: PNG content named .jpg.
    mismatch = tmp / "png_named_jpg.jpg"
    shutil.copy2(good, mismatch)
    made["mismatched_extension"] = mismatch

    return made


def _png_bomb(w: int, h: int) -> bytes:
    """A valid PNG header declaring a huge canvas over highly-compressible data."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return len(data).to_bytes(4, "big") + tag + data + zlib.crc32(tag + data).to_bytes(4, "big")

    ihdr = w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])  # 8-bit RGB
    raw = b"".join(b"\x00" + b"\x00" * (w * 3) for _ in range(h))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def check_bad_inputs(res: Results, tmp: Path, inputs: dict[str, Path]) -> None:
    print("\nmalformed and degenerate inputs -- every command must refuse, not crash")
    out = tmp / "out.png"
    for case in ("truncated", "corrupt", "empty", "not_an_image", "tiny_1x1", "sliver_1x4000"):
        src = inputs.get(case)
        if src is None:
            continue
        graceful(res, case, "identify", ["identify", str(src)])
        graceful(res, case, "visible", ["visible", str(src), "-o", str(out)])
        graceful(res, case, "metadata", ["metadata", str(src), "--check"])


def check_bomb(res: Results, tmp: Path, inputs: dict[str, Path]) -> None:
    print("\ndecompression bomb -- must not exhaust memory or hang")
    bomb = inputs.get("decompression_bomb")
    if bomb is None:
        return
    size_kb = bomb.stat().st_size / 1024
    print(f"  (bomb is {size_kb:.0f} KB on disk, declares 16000x16000)")
    graceful(res, "decompression_bomb", "identify", ["identify", str(bomb)], timeout=180)
    graceful(res, "decompression_bomb", "visible", ["visible", str(bomb), "-o", str(tmp / "b.png")], timeout=180)


def check_filenames(res: Results, tmp: Path, inputs: dict[str, Path]) -> None:
    """Unicode/RTL paths must round-trip on BOTH read and write (issue #17)."""
    print("\nunicode / RTL / mismatched-extension paths")
    for case in ("unicode_filename", "rtl_filename", "mismatched_extension"):
        src = inputs.get(case)
        if src is None:
            continue
        # Write to a Unicode OUTPUT path too -- the read side alone does not prove the IO.
        outp = tmp / f"вывод-{case}-📤.png"
        code, out, timed = run(["identify", str(src)])
        if timed or any(m in out for m in _CRASH_MARKERS):
            res.add(case, "identify", False, "crashed or hung", out)
        else:
            res.add(case, "identify", True, f"read fine, exit {code}")
        code, out, timed = run(["erase", str(src), "--region", "5,5,50,30", "-o", str(outp)])
        wrote = outp.exists() and outp.stat().st_size > 0
        crash = any(m in out for m in _CRASH_MARKERS)
        res.add(case, "erase", wrote and not crash and not timed, f"unicode output written={wrote} exit={code}", out)


def check_output_paths(res: Results, tmp: Path, inputs: dict[str, Path]) -> None:
    print("\nhostile output paths")
    good = inputs["good"]

    # A nested output dir that does not exist yet -- the CLI should create it.
    nested = tmp / "a" / "b" / "c" / "out.png"
    code, out, timed = run(["erase", str(good), "--region", "5,5,40,20", "-o", str(nested)])
    crash = any(m in out for m in _CRASH_MARKERS)
    res.add(
        "nonexistent_nested_outdir",
        "erase",
        nested.exists() and not crash and not timed,
        f"created={nested.exists()} exit={code}",
        out,
    )

    # A READ-ONLY output directory -- must report, not traceback.
    ro = tmp / "readonly"
    ro.mkdir(exist_ok=True)
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        graceful(
            res, "readonly_output_dir", "erase", ["erase", str(good), "--region", "5,5,40,20", "-o", str(ro / "x.png")]
        )
    finally:
        ro.chmod(stat.S_IRWXU)  # restore so the temp dir can be cleaned up

    # A directory where a file is expected.
    graceful(res, "directory_as_input", "identify", ["identify", str(tmp)])

    # A path that simply is not there.
    graceful(res, "missing_input", "identify", ["identify", str(tmp / "nope.png")])


def check_concurrency(res: Results, tmp: Path, inputs: dict[str, Path]) -> None:
    """A wrapping service will run jobs in parallel against one input path."""
    print("\nconcurrent runs against a single input")
    good = inputs["good"]

    def one(i: int) -> tuple[int, str, bool]:
        return run(["erase", str(good), "--region", "5,5,40,20", "-o", str(tmp / f"conc_{i}.png")])

    with ThreadPoolExecutor(max_workers=4) as ex:
        outs = list(ex.map(one, range(4)))
    crashed = [o for o in outs if any(m in o[1] for m in _CRASH_MARKERS) or o[2]]
    all_written = all((tmp / f"conc_{i}.png").exists() for i in range(4))
    res.add(
        "4x concurrent on one input",
        "erase",
        not crashed and all_written,
        f"{sum(1 for i in range(4) if (tmp / f'conc_{i}.png').exists())}/4 outputs, {len(crashed)} crashed",
        "".join(o[1] for o in crashed),
    )


def check_batch_silent_loss(res: Results, tmp: Path, inputs: dict[str, Path]) -> None:
    """The nastiest shape: NO output files AND a success exit code.

    `graceful()` cannot see this class -- it scores exit code and traceback markers, and a
    run that writes nothing while exiting 0 has neither. A regression case showed that
    `batch --mode visible` into a read-only directory could write no files and exit 0, so a
    wrapping service would treat an empty output directory as a completed run. Any check
    for a silent no-op must assert on the ARTIFACTS, not on the status.
    """
    print("\nbatch into a read-only output dir -- must NOT exit 0 with nothing written")
    indir = tmp / "loss_in"
    indir.mkdir(exist_ok=True)
    for i in range(2):
        shutil.copy2(inputs["good"], indir / f"img{i}.png")
    ro = tmp / "loss_out"
    ro.mkdir(exist_ok=True)
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        code, out, timed = run(["batch", str(indir), "-o", str(ro)], timeout=180)
        written = len(list(ro.glob("*")))
    finally:
        ro.chmod(stat.S_IRWXU)
    ok = not timed and not (code == 0 and written == 0)
    res.add(
        "batch_readonly_outdir", "batch", ok, f"exit {code}, {written}/2 written (exit 0 + 0 files = data loss)", out
    )


def check_batch_edges(res: Results, tmp: Path) -> None:
    print("\nbatch edge cases")
    empty_dir = tmp / "empty_dir"
    empty_dir.mkdir(exist_ok=True)
    graceful(res, "batch_on_empty_dir", "batch", ["batch", str(empty_dir), "-o", str(tmp / "bo")])

    # A directory of junk: nothing decodable, must not crash the whole run.
    junk = tmp / "junk_dir"
    junk.mkdir(exist_ok=True)
    (junk / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + os.urandom(500))
    (junk / "b.jpg").write_bytes(b"not an image at all")
    graceful(res, "batch_on_undecodable_dir", "batch", ["batch", str(junk), "-o", str(tmp / "jo")], timeout=180)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    res = Results()
    with tempfile.TemporaryDirectory(prefix="raiw_robust_") as td:
        tmp = Path(td)
        print("building adversarial inputs...")
        inputs = make_inputs(tmp)
        print(f"built {len(inputs)} inputs\n")
        check_bad_inputs(res, tmp, inputs)
        check_bomb(res, tmp, inputs)
        check_filenames(res, tmp, inputs)
        check_output_paths(res, tmp, inputs)
        check_concurrency(res, tmp, inputs)
        check_batch_silent_loss(res, tmp, inputs)
        check_batch_edges(res, tmp)
    raise SystemExit(res.report())


if __name__ == "__main__":
    main()
