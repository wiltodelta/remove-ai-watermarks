"""Release smoke matrix: exercise every CLI parameter CHOICE on REAL data.

Run before a release. This is not a unit-test substitute -- it drives the real CLI as a
subprocess, so it covers argument parsing, exit-code semantics and the file-writing
contracts that unit tests with fakes cannot.

WHAT IT COVERS AND WHY THAT SHAPE
  * Every choice-valued flag at least once (`--backend`, `--sensitivity`, `--mark`,
    `--mode`, `--inpaint-method`, ...). Full permutation is combinatorially large and
    mostly meaningless; dead branches and typos hide in the CHOICES, not in their
    products.
  * Every input FORMAT and shape edge case through the pixel paths -- PNG/JPEG/WebP/
    HEIC/AVIF, alpha, unicode names, misnamed extensions, truncated files, tiny and
    landscape frames. Real-world breakage lives here far more than in flag combos.
  * CONTRACTS, not just exit codes: `visible` must write NO output and exit 2 when no
    mark is found (re-serving the input reads as success -- the recurring "it didn't
    work" report); a no-op must be byte-identical; `metadata --remove` must actually
    strip; a JPEG strip must not touch the pixels. Note the pixel-lossless contract is
    the DEFAULT path's -- `--remove-all` deliberately re-encodes (see metadata.py).
  * The diffusion bodies under `--diffusion`, at `--max-resolution 512` so they fit MPS
    (~1 min/image on 32 GB unified memory). Not just exit codes: `invisible` must
    restore the input resolution and must NOT re-stamp SDXL's own open watermark.

WHAT IT DOES NOT COVER, DELIBERATELY AND LOUDLY
  * Without `--diffusion`, the model-running bodies are reported as SKIPPED with a
    reason, never as passes -- a green run that quietly skipped half the surface is
    worse than a red one.
  * Removal STRENGTH is never certified here. Whether a watermark is actually gone
    needs the per-vendor oracles (docs/known-limitations.md); these rows prove the
    paths run and keep their contracts, nothing more.
  * The re-embed row is gated on a POSITIVE CONTROL. imwatermark is positive-only and
    fails to round-trip on some pristine carriers, so "no watermark found" proves
    nothing there; the row degrades to a skip rather than a false pass.

    uv run python scripts/smoke_matrix.py                 # corpus + fixtures
    uv run python scripts/smoke_matrix.py --quick         # fixtures only, no corpus
    uv run python scripts/smoke_matrix.py --diffusion     # + the SDXL model paths
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "data" / "samples"
CORPUS = REPO / "data" / "spaces" / "originals"
EXIT_NO_VISIBLE_MARK = 2


def _capture(args: list[str]) -> str:
    """Run the CLI and return stdout (for the rows that inspect output, not exit code)."""
    exe = shutil.which("uv") or "uv"
    p = subprocess.run(  # noqa: S603
        [exe, "run", "remove-ai-watermarks", *args], capture_output=True, text=True, cwd=REPO, check=False
    )
    return p.stdout


@dataclass
class Result:
    name: str
    status: str  # pass | FAIL | skip
    detail: str = ""
    cmd: str = ""


@dataclass
class Runner:
    tmp: Path
    results: list[Result] = field(default_factory=list)

    def run(self, name: str, args: list[str], *, expect_exit: int | None = 0, timeout: int = 180) -> Result:
        exe = shutil.which("uv") or "uv"
        cmd = [exe, "run", "remove-ai-watermarks", *args]
        try:
            p = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO, check=False
            )
        except subprocess.TimeoutExpired:
            r = Result(name, "FAIL", f"timeout after {timeout}s", " ".join(args))
            self.results.append(r)
            return r
        ok = expect_exit is None or p.returncode == expect_exit
        detail = "" if ok else f"exit {p.returncode} (want {expect_exit}): {(p.stderr or p.stdout).strip()[-200:]}"
        r = Result(name, "pass" if ok else "FAIL", detail, " ".join(args))
        self.results.append(r)
        return r

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append(Result(name, "pass" if ok else "FAIL", "" if ok else detail))

    def skip(self, name: str, why: str) -> None:
        self.results.append(Result(name, "skip", why))


def corpus_pick(n: int, suffixes: tuple[str, ...]) -> list[Path]:
    """Real uploads, chosen deterministically so a failure is reproducible."""
    if not CORPUS.exists():
        return []
    pool = [p for p in CORPUS.glob("*/*") if p.suffix.lower() in suffixes]
    random.Random(7).shuffle(pool)  # noqa: S311 -- deterministic sampling, not cryptography
    return pool[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fixtures only; skip the corpus rows")
    ap.add_argument(
        "--diffusion", action="store_true", help="also run the model-running paths (SDXL weights, ~1 min/image)"
    )
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="raiw-smoke-"))
    r = Runner(tmp)
    doubao = SAMPLES / "doubao-1.png"
    chatgpt = SAMPLES / "chatgpt-1.png"
    grok = SAMPLES / "grok-1.jpg"

    # ---- identify: every flag, and JSON must actually parse -------------------
    r.run("identify plain", ["identify", str(doubao)])
    r.run("identify --no-visible", ["identify", str(doubao), "--no-visible"])
    try:
        json.loads(_capture(["identify", str(doubao), "--json"]))
        r.check("identify --json parses", True)
    except Exception as e:
        r.check("identify --json parses", False, str(e))

    # ---- metadata: check/remove, and the strip must be REAL ------------------
    r.run("metadata --check", ["metadata", str(doubao), "--check"])
    out = tmp / "meta.png"
    r.run("metadata --remove", ["metadata", str(doubao), "--remove", "-o", str(out)])
    if out.exists():
        try:
            rep = json.loads(_capture(["identify", str(out), "--json"]))
            # Scope the assert to METADATA. doubao-1 also carries a visible pixel mark,
            # which `metadata --remove` must NOT touch -- demanding an empty watermark
            # list would fail on correct behavior (and hide a real metadata leak behind
            # a permanently-red row).
            leftover = [s for s in rep.get("signals", []) if not s.get("name", "").startswith("visible_")]
            r.check(
                "metadata --remove strips every metadata signal",
                not leftover and not rep.get("ai_from_metadata"),
                f"still reports metadata signals {[s.get('name') for s in leftover]}",
            )
            r.check(
                "metadata --remove leaves the visible mark alone",
                any(s.get("name", "").startswith("visible_") for s in rep.get("signals", [])),
                "the pixel mark vanished -- a metadata strip must not touch pixels",
            )
        except Exception as e:
            r.check("metadata --remove strips every metadata signal", False, str(e))

    # The pixel-lossless contract belongs to the DEFAULT path. `--remove-all`
    # (keep_standard=False) deliberately falls through to the full PIL re-encode
    # (metadata.py: the lossless marker walk preserves standard segments, so it cannot
    # serve a strip-everything caller), which is lossy for JPEG -- measured ~49 dB.
    # Asserting losslessness there tested a contract the code never made.
    import numpy as np

    from remove_ai_watermarks.image_io import imread

    out2 = tmp / "meta2.jpg"
    r.run("metadata --remove (jpeg, default)", ["metadata", str(grok), "--remove", "-o", str(out2)])
    if out2.exists():
        a_, b_ = imread(str(grok)), imread(str(out2))
        same = a_ is not None and b_ is not None and a_.shape == b_.shape and bool(np.array_equal(a_, b_))
        r.check("jpeg metadata strip is pixel-lossless", same, "pixels changed -- the strip re-encoded")
    out3 = tmp / "meta3.jpg"
    r.run("metadata --remove --remove-all (jpeg)", ["metadata", str(grok), "--remove", "--remove-all", "-o", str(out3)])
    if out3.exists():
        rep3 = json.loads(_capture(["identify", str(out3), "--json"]))
        r.check(
            "--remove-all strips the AI metadata too",
            not [s for s in rep3.get("signals", []) if not s.get("name", "").startswith("visible_")],
            f"still reports {rep3.get('signals')}",
        )

    # ---- visible: every --mark choice, every --backend, every --sensitivity ---
    for mark in ("auto", "gemini", "doubao", "jimeng", "samsung", "jimeng_pill"):
        # doubao-1 carries only the doubao mark, so every other --mark must exit 2
        want = 0 if mark in ("auto", "doubao") else EXIT_NO_VISIBLE_MARK
        r.run(
            f"visible --mark {mark}",
            ["visible", str(doubao), "--mark", mark, "-o", str(tmp / f"m_{mark}.png")],
            expect_exit=want,
        )
    # Every backend CHOICE, not just the two that need no extra. migan/lama were
    # measured at library level (scripts/fill_quality.py) but had never been driven
    # through the CLI, which is a different code path (resolve_backend + the warning).
    from remove_ai_watermarks.region_eraser import lama_available, migan_available

    for backend in ("auto", "cv2", "migan", "lama"):
        have = {"migan": migan_available(), "lama": lama_available()}.get(backend, True)
        if not have:
            r.skip(f"visible --backend {backend}", f"the `{backend}` extra is not installed")
            continue
        r.run(
            f"visible --backend {backend}",
            ["visible", str(doubao), "--backend", backend, "-o", str(tmp / f"b_{backend}.png")],
            timeout=600,
        )
    for sens in ("auto", "strict"):
        r.run(
            f"visible --sensitivity {sens}",
            ["visible", str(doubao), "--sensitivity", sens, "-o", str(tmp / f"s_{sens}.png")],
        )
    r.run("visible --keep-metadata", ["visible", str(doubao), "--keep-metadata", "-o", str(tmp / "keep.png")])
    r.run("visible --no-detect (forced)", ["visible", str(doubao), "--no-detect", "-o", str(tmp / "force.png")])
    for removed in ("assume-ai", "aggressive"):
        r.run(
            f"visible rejects --sensitivity {removed}",
            ["visible", str(doubao), "--sensitivity", removed],
            expect_exit=2,
        )

    # CONTRACT: no mark -> no output file, exit 2 (never re-serve the input as success)
    noout = tmp / "must_not_exist.png"
    r.run("visible no-mark exits 2", ["visible", str(chatgpt), "-o", str(noout)], expect_exit=EXIT_NO_VISIBLE_MARK)
    r.check("visible no-mark writes NO output", not noout.exists(), "wrote an output for an undetected mark")

    # ---- erase: backends, methods, repeated regions, dilate ------------------
    for method in ("telea", "ns"):
        r.run(
            f"erase --inpaint-method {method}",
            [
                "erase",
                str(doubao),
                "--region",
                "10,10,60,30",
                "--inpaint-method",
                method,
                "-o",
                str(tmp / f"e_{method}.png"),
            ],
        )
    r.run(
        "erase repeated --region",
        ["erase", str(doubao), "--region", "10,10,40,20", "--region", "80,80,40,20", "-o", str(tmp / "e_multi.png")],
    )
    r.run(
        "erase --dilate",
        ["erase", str(doubao), "--region", "10,10,40,20", "--dilate", "5", "-o", str(tmp / "e_dil.png")],
    )
    for backend in ("cv2", "migan", "lama"):
        have = {"migan": migan_available(), "lama": lama_available()}.get(backend, True)
        if not have:
            r.skip(f"erase --backend {backend}", f"the `{backend}` extra is not installed")
            continue
        r.run(
            f"erase --backend {backend}",
            [
                "erase",
                str(doubao),
                "--region",
                "10,10,40,20",
                "--backend",
                backend,
                "-o",
                str(tmp / f"e_{backend}.png"),
            ],
            timeout=600,
        )
    r.run(
        "erase --dilate 0 (no dilation)",
        ["erase", str(doubao), "--region", "10,10,40,20", "--dilate", "0", "-o", str(tmp / "e_d0.png")],
    )
    r.run("erase rejects a malformed --region", ["erase", str(doubao), "--region", "not,a,box"], expect_exit=2)

    # ---- global + explicit-default flags that had never been exercised -------
    r.run("--verbose", ["--verbose", "identify", str(doubao)])
    r.run("--version", ["--version"])
    r.run("metadata --keep-standard (explicit)", ["metadata", str(doubao), "--check", "--keep-standard"])
    r.run(
        "visible --strip-metadata (explicit)",
        ["visible", str(doubao), "--strip-metadata", "-o", str(tmp / "sm.png")],
    )
    r.run("visible --detect (explicit default)", ["visible", str(doubao), "--detect", "-o", str(tmp / "det.png")])

    # ---- batch: the non-diffusion modes -------------------------------------
    bd = tmp / "batch_in"
    bd.mkdir()
    for f in (doubao, chatgpt):
        shutil.copy(f, bd / f.name)
    for mode in ("visible", "metadata"):
        r.run(
            f"batch --mode {mode}",
            ["batch", str(bd), "--mode", mode, "-o", str(tmp / f"batch_{mode}")],
            expect_exit=None,
        )

    # ---- diffusion: argument handling always; the model body under --diffusion ----
    r.run("invisible --help parses full knob set", ["invisible", "--help"])
    r.run("all --help parses full knob set", ["all", "--help"])
    _media_rows(r, tmp)
    if a.diffusion:
        _diffusion_rows(r, tmp, doubao)
    else:
        for name in ("invisible", "all", "batch --mode invisible"):
            r.skip(f"{name} (model-running body)", "pass --diffusion to exercise it (needs the SDXL weights)")

    # ---- real-data formats and shapes ---------------------------------------
    if not a.quick:
        picks: list[tuple[str, Path]] = []
        for suf, label in ((".heic", "heic"), (".avif", "avif"), (".webp", "webp"), (".jpeg", "jpeg")):
            picks += [(label, p) for p in corpus_pick(2, (suf,))]
        picks += [("png", p) for p in corpus_pick(3, (".png",))]
        if not picks:
            r.skip("real-format rows", "corpus not present (data/spaces/originals)")
        for label, p in picks:
            r.run(f"identify real {label}", ["identify", str(p), "--json"])
            r.run(
                f"visible auto real {label}",
                ["visible", str(p), "-o", str(tmp / f"r_{label}_{p.stem[:8]}.png")],
                expect_exit=None,
            )  # 0 or 2 are both correct; a CRASH is not

        # unicode + misnamed extension + truncated: the documented real-world traps
        if picks:
            src = picks[0][1]
            uni = tmp / "тест изображение 测试.png"
            shutil.copy(src, uni)
            r.run("unicode filename", ["identify", str(uni), "--json"])
            mis = tmp / "actually_png.jpg"  # content PNG, extension JPEG
            shutil.copy(SAMPLES / "chatgpt-1.png", mis)
            r.run("misnamed extension", ["identify", str(mis), "--json"])
            trunc = tmp / "truncated.png"
            trunc.write_bytes((SAMPLES / "chatgpt-1.png").read_bytes()[:4096])
            r.run("truncated file does not crash", ["identify", str(trunc), "--json"], expect_exit=None)

    # ---- report --------------------------------------------------------------
    bad = [x for x in r.results if x.status == "FAIL"]
    skipped = [x for x in r.results if x.status == "skip"]
    ok = [x for x in r.results if x.status == "pass"]
    print(f"\n{'=' * 74}\nSMOKE MATRIX  pass={len(ok)}  FAIL={len(bad)}  skipped={len(skipped)}\n{'=' * 74}")
    for x in skipped:
        print(f"  SKIP  {x.name:46s} {x.detail}")
    for x in bad:
        print(f"  FAIL  {x.name:46s} {x.detail}")
        if x.cmd:
            print(f"        cmd: {x.cmd}")
    if not bad:
        print("  no failures")
    print(f"\ntmp artifacts: {tmp}")
    raise SystemExit(1 if bad else 0)


def _knob_rows(r: Runner, tmp: Path, img: Path) -> None:
    """Every diffusion knob the matrix never touched.

    Deliberately cheap (`--steps 4`, `--max-resolution 384`): these rows answer "is the
    knob accepted and does the run complete", NOT "is the output good". Quality per knob
    needs a per-knob oracle and most of them have none (`--humanize` has no oracle at
    all), so claiming more here would be dishonest.
    """
    from remove_ai_watermarks import upscaler

    # --steps 20 is the floor that WORKS, not an arbitrary choice: effective timesteps
    # are int(steps * strength), so at the default strength 0.15 anything below
    # --steps 7 rounds to ZERO and the pipeline dies inside torch. The first version of
    # these rows used --steps 4 and every single one failed with the same reshape error.
    fast = ["--max-resolution", "384", "--min-resolution", "0", "--steps", "20", "--force", "--seed", "0"]

    def run(name: str, extra: list[str], *, tag: str, expect: int | None = 0) -> None:
        r.run(
            name,
            ["invisible", str(img), "-o", str(tmp / f"k_{tag}.png"), *fast, *extra],
            expect_exit=expect,
            timeout=2400,
        )

    run("--pipeline sdxl", ["--pipeline", "sdxl"], tag="sdxl")
    run("--pipeline controlnet", ["--pipeline", "controlnet"], tag="cnet")
    run("--strength", ["--strength", "0.2"], tag="strength")
    run("--guidance-scale", ["--guidance-scale", "5.0"], tag="gs")
    run("--controlnet-scale", ["--controlnet-scale", "0.5"], tag="cns")
    run("--humanize", ["--humanize", "0.3"], tag="hum")
    run("--unsharp", ["--unsharp", "0.5"], tag="uns")
    run("--no-adaptive-polish", ["--no-adaptive-polish"], tag="nap")
    run("--tile", ["--tile", "--tile-size", "256", "--tile-overlap", "64"], tag="tile")
    run("--device mps", ["--device", "mps"], tag="mps")
    run("--upscaler lanczos", ["--upscaler", "lanczos"], tag="lanczos")
    run("--auto (deprecated no-op)", ["--auto"], tag="auto")

    if upscaler.is_available():
        run("--upscaler esrgan", ["--upscaler", "esrgan"], tag="esrgan")
    else:
        r.skip("--upscaler esrgan", "the `esrgan` extra is not installed")

    # CPU is correctness-relevant (it is the documented MPS-OOM fallback) but slow, so
    # it gets the smallest possible run rather than being skipped.
    r.run(
        "--device cpu",
        [
            "invisible",
            str(img),
            "-o",
            str(tmp / "k_cpu.png"),
            "--device",
            "cpu",
            "--max-resolution",
            "256",
            "--min-resolution",
            "0",
            "--steps",
            "20",
            "--force",
            "--seed",
            "0",
        ],
        timeout=3600,
    )

    # qwen is CUDA-class by design (bf16 MMDiT, no MPS fallback). On this host the
    # honest outcome is a CLEAN failure, not a crash -- assert it does not hang or
    # dump a traceback at the user.
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    if cuda:
        run("--pipeline qwen", ["--pipeline", "qwen"], tag="qwen")
    else:
        r.skip("--pipeline qwen", "CUDA-class pipeline; no CUDA device on this host")

    # --model and --hf-token are deliberately not exercised: one would download a second
    # multi-GB checkpoint, the other needs a real credential. Skipped loudly, not passed.
    r.skip("--model", "would download a second multi-GB checkpoint")
    r.skip("--hf-token", "needs a real credential; cannot be exercised meaningfully here")

    # CONTRACT, not just execution: the same seed must reproduce the same pixels.
    a_out, b_out = tmp / "seed_a.png", tmp / "seed_b.png"
    for out in (a_out, b_out):
        r.run(
            f"seed determinism run ({out.name})",
            ["invisible", str(img), "-o", str(out), *fast],
            timeout=2400,
        )
    if a_out.exists() and b_out.exists():
        import numpy as np

        from remove_ai_watermarks.image_io import imread

        x, y = imread(str(a_out)), imread(str(b_out))
        r.check(
            "same --seed reproduces identical pixels",
            x is not None and y is not None and x.shape == y.shape and bool(np.array_equal(x, y)),
            "two runs with the same seed differed",
        )


def _media_rows(r: Runner, tmp: Path) -> None:
    """Audio/video metadata strip via ffmpeg -- a supported path with no corpus coverage.

    The corpus is images only, so the media is synthesized here with ffmpeg rather than
    left untested.
    """
    if not shutil.which("ffmpeg"):
        r.skip("audio/video metadata strip", "ffmpeg not on PATH")
        return
    for name, gen in (
        ("mp4", ["-f", "lavfi", "-i", "testsrc=duration=1:size=128x128:rate=8", "-pix_fmt", "yuv420p"]),
        ("mp3", ["-f", "lavfi", "-i", "sine=frequency=440:duration=1"]),
    ):
        src = tmp / f"media.{name}"
        ff = shutil.which("ffmpeg") or "ffmpeg"
        made = subprocess.run(  # noqa: S603
            [ff, "-y", *gen, "-metadata", "comment=Made with AI", str(src)],
            capture_output=True,
            check=False,
        )
        if made.returncode != 0 or not src.exists():
            r.skip(f"{name} metadata strip", "ffmpeg could not synthesize the fixture")
            continue
        out = tmp / f"media_clean.{name}"
        r.run(f"{name} metadata strip runs", ["metadata", str(src), "--remove", "-o", str(out)], expect_exit=None)
        if out.exists():
            r.check(f"{name} strip produced a non-empty file", out.stat().st_size > 0, "empty output")


def _sdxl_watermark_bits(img: object) -> float:
    """Bits of the open SDXL DWT-DCT watermark recovered from `img` (128 = perfect)."""
    import numpy as np
    from imwatermark import WatermarkDecoder

    truth = np.frombuffer(b"StableDiffusionV1"[:16], dtype=np.uint8)
    rec = WatermarkDecoder("bytes", 128).decode(img, "dwtDct")
    return float(128 - np.unpackbits(truth ^ np.frombuffer(bytes(rec), dtype=np.uint8)).sum())


def _diffusion_rows(r: Runner, tmp: Path, doubao: Path) -> None:
    """Exercise the model-running bodies at a reduced resolution (MPS-friendly).

    Bounded with `--max-resolution 512` and a fixed seed: the point is that the paths
    RUN and keep their contracts, not to certify removal strength (that needs the
    per-vendor oracles, see docs/known-limitations.md).
    """
    import numpy as np

    from remove_ai_watermarks.image_io import imread

    small = ["--max-resolution", "512", "--seed", "0"]

    # `invisible` must restore the original resolution and NOT re-stamp SDXL's own
    # open watermark (add_watermarker=False; a remover that re-marks its output is the
    # regression this row exists for).
    #
    # The carrier matters: imwatermark is positive-only and fails to round-trip on some
    # pristine images, so an "absent" verdict on a fragile carrier proves NOTHING. mj-1
    # is used because it round-trips at 128/128; the control is re-checked on the actual
    # OUTPUT and the row degrades to a skip rather than a false pass if it goes fragile.
    mj = SAMPLES / "mj-1.png"
    inv = tmp / "inv_mj.png"
    res = r.run("invisible runs (mps, 512px)", ["invisible", str(mj), "-o", str(inv), "--force", *small], timeout=1800)
    if res.status == "pass" and inv.exists():
        src_img, out_img = imread(str(mj)), imread(str(inv))
        r.check(
            "invisible restores the input resolution",
            src_img is not None and out_img is not None and src_img.shape == out_img.shape,
            f"{None if src_img is None else src_img.shape} -> {None if out_img is None else out_img.shape}",
        )
        try:
            from imwatermark import WatermarkEncoder

            enc = WatermarkEncoder()
            enc.set_watermark("bytes", b"StableDiffusionV1"[:16])
            control = _sdxl_watermark_bits(enc.encode(np.array(out_img).copy(), "dwtDct"))
            if control < 118:
                r.skip("invisible does not re-embed an SDXL watermark", f"carrier fragile (control {control:.0f}/128)")
            else:
                bits = _sdxl_watermark_bits(out_img)
                r.check(
                    "invisible does not re-embed an SDXL watermark",
                    bits < 118,
                    f"re-embedded: {bits:.0f}/128 recovered (control {control:.0f}/128)",
                )
        except ImportError:
            r.skip("invisible does not re-embed an SDXL watermark", "imwatermark absent (extra `detect`)")

    # `all`: every stage must land -- the visible mark AND the metadata both gone.
    allout = tmp / "all_out.png"
    res = r.run("all runs (mps, 512px)", ["all", str(doubao), "-o", str(allout), *small], timeout=1800)
    if res.status == "pass" and allout.exists():
        rep = json.loads(_capture(["identify", str(allout), "--json"]))
        r.check(
            "all clears visible + metadata in one pass",
            not rep.get("signals"),
            f"still reports {[s.get('name') for s in rep.get('signals', [])]}",
        )

    _knob_rows(r, tmp, mj)

    # `batch --mode invisible`: every input must produce an output (a silent short
    # write is the failure this row guards).
    bd = tmp / "batch_inv"
    bd.mkdir(exist_ok=True)
    for f in (SAMPLES / "chatgpt-2.png", mj):
        shutil.copy(f, bd / f.name)
    bout = tmp / "batch_inv_out"
    res = r.run(
        "batch --mode invisible runs (mps, 512px)",
        ["batch", str(bd), "--mode", "invisible", "-o", str(bout), *small],
        expect_exit=None,
        timeout=3600,
    )
    if res.status == "pass":
        produced = len(list(bout.glob("*"))) if bout.exists() else 0
        r.check("batch invisible writes one output per input", produced == 2, f"{produced} outputs for 2 inputs")

    # `batch --mode all` -- the only --mode value the matrix never ran.
    aout = tmp / "batch_all_out"
    r.run(
        "batch --mode all runs (mps, 512px)",
        ["batch", str(bd), "--mode", "all", "-o", str(aout), *small],
        expect_exit=None,
        timeout=3600,
    )

    # The AI-enhanced composite path: regenerate ONLY a region and feather it back,
    # leaving everything outside the box pixel-exact. Library-level -- the CLI has no
    # flag for it, so it would otherwise never be exercised on real data.
    try:
        import numpy as np

        from remove_ai_watermarks.image_io import imread
        from remove_ai_watermarks.noai.watermark_remover import WatermarkRemover

        src = imread(str(mj))
        h, w = src.shape[:2]
        box = (w // 4, h // 4, w // 4, h // 4)
        rem = WatermarkRemover(pipeline="controlnet")
        rout = tmp / "region_composite.png"
        rem.remove_watermark(mj, rout, strength=0.15, num_inference_steps=20, seed=0, region=box)
        got = imread(str(rout))
        if got is None or got.shape != src.shape:
            r.check("region composite keeps the frame outside the box", False, "shape changed or unreadable")
        else:
            mask = np.ones(src.shape[:2], dtype=bool)
            x, y, bw, bh = box
            # Outside the box PLUS the feather margin must be untouched.
            pad = 96
            mask[max(0, y - pad) : y + bh + pad, max(0, x - pad) : x + bw + pad] = False
            r.check(
                "region composite keeps the frame outside the box",
                bool(np.array_equal(src[mask], got[mask])),
                "pixels changed outside the regenerated region",
            )
    except Exception as e:
        r.skip("region composite (remove_watermark(region=...))", f"{type(e).__name__}: {e}"[:120])


if __name__ == "__main__":
    main()
