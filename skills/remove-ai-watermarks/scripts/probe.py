#!/usr/bin/env python3
"""Report whether this machine can run remove-ai-watermarks.

Stdlib only. JSON on stdout, indented with --pretty.
Works on Windows, macOS, and Linux. No interactive prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any

_GPU_MARKERS = ("GPU ", "gpu:")
# Below this, the CLI predates flags and behavior this skill documents. Raise it in
# the same change as any reference that names a flag the previous release lacked:
# 0.37.0 is here because `classify` and `verify-openai-synthid` arrived with it
# (`--pipeline auto` needed 0.36.0; chroma-zimage and the doubao video mark 0.35.0;
# the `microsoft` mark 0.34.0; `--vendor` 0.32.0), and a stale floor told an agent
# its CLI was current right before it typed a value that build rejects.
# tests/test_agent_skill.py pins it against the package version, so it can never
# advertise a release that does not exist.
MIN_CLI_VERSION = (0, 37, 0)
# Markers a build without the pixel stack prints. The first is the current CLI's own
# install hint; the rest are what older builds emit before the guard existed, and the
# Homebrew formula ships exactly such a build.
_NO_PIXELS_MARKERS = (
    "visible-mark dependencies are not installed",
    "No module named 'cv2'",
    "No module named 'numpy'",
)
_NO_INVISIBLE_MARKER = "invisible-removal dependencies are not installed"
# The capability checks run the CLI on a file with this name. A caller reading a trace
# of CLI invocations uses it to tell the probe's own calls from the ones an agent chose.
PROBE_IMAGE_NAME = "probe.png"


def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(argv: list[str], timeout: float = 8.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(  # noqa: S603
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1, "", ""
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _probe_python() -> dict[str, Any]:
    info = sys.version_info
    return {
        "executable": sys.executable,
        "version": f"{info.major}.{info.minor}.{info.micro}",
        "note": "Interpreter running this probe, not the CLI runtime.",
    }


def _parse_version(text: str) -> tuple[int, ...] | None:
    """The dotted release out of ``remove-ai-watermarks, version X.Y.Z``."""
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    return tuple(int(part) for part in match.groups()) if match else None


def _cli() -> dict[str, Any]:
    path = _which("remove-ai-watermarks")
    if path is None:
        return {"found": False, "path": None, "version": None, "outdated": None}
    code, stdout, stderr = _run([path, "--version"])
    text = (stdout or stderr).strip()
    parsed = _parse_version(text)
    return {
        "found": code == 0,
        "path": path,
        "version": text or None,
        # None means the version could not be read, which is not the same as current.
        "outdated": None if parsed is None else parsed < MIN_CLI_VERSION,
    }


def _blank_png(path: Path) -> None:
    """Write a 64x64 grey PNG with zlib and struct alone, no Pillow, no cv2."""
    width = height = 64
    raw = b"".join(b"\x00" + bytes([128]) * (width * 3) for _ in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _run_on_blank(cli_path: str, subcommand: str) -> tuple[int, str]:
    """Run one subcommand on a blank image this probe writes, and return (code, output).

    Both capability checks need the same invocation, so they share one shape: writing
    the PNG twice and spelling the flags twice is how the two would drift apart.
    """
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / PROBE_IMAGE_NAME
        _blank_png(source)
        code, stdout, stderr = _run(
            [cli_path, subcommand, str(source), "-o", str(Path(tmp) / "probe_clean.png")],
            timeout=60.0,
        )
    return code, f"{stdout}\n{stderr}"


def _pixel_stack(cli_path: str) -> str:
    """Whether the INSTALLED build can actually run a visible-mark command.

    A found binary says nothing about this. The default package ships without the
    pixel extra, so ``cli.found`` reported "ok" for a Homebrew install whose very
    first ``visible`` died on a missing cv2. So ask the real seam: run ``visible``
    on a blank image the probe writes itself. Exit 2 is the documented "no mark
    here", which only a working pixel stack can reach.
    """
    code, blob = _run_on_blank(cli_path, "visible")
    if code == 2:
        return "ok"
    if any(marker in blob for marker in _NO_PIXELS_MARKERS):
        return "missing"
    return "ok" if code == 0 else "unknown"


def _invisible_stack(cli_path: str) -> str:
    """Whether the diffusion extra is installed, WITHOUT downloading a model.

    ``all`` behaves in two opposite ways on a machine without CUDA, and the
    difference is this stack, not the GPU: without the extra it skips the invisible
    stage and still writes a visible+metadata result, and with the extra it fails at
    device construction and writes nothing. Advising "do not run" in both cases threw
    away the one command that does useful work on a Mac.

    ``invisible`` checks the extra first and detection second, so a blank image with
    no signal exits 2 before anything loads -- 0.23s here, no download.
    """
    code, blob = _run_on_blank(cli_path, "invisible")
    if _NO_INVISIBLE_MARKER in blob:
        return "missing"
    return "installed" if code == 2 else "unknown"


def _has_cuda() -> bool:
    exe = _which("nvidia-smi")
    if exe is None:
        return False
    code, stdout, stderr = _run([exe, "-L"])
    blob = f"{stdout}\n{stderr}"
    return code == 0 and any(marker in blob for marker in _GPU_MARKERS)


def _installers() -> dict[str, str | None]:
    return {
        "uv": _which("uv"),
        "pipx": _which("pipx"),
        "pip": _which("pip"),
        "brew": _which("brew"),
    }


def _preferred_installer(installers: dict[str, str | None]) -> str | None:
    for name in ("uv", "pipx", "pip"):
        if installers[name]:
            return name
    return None


def _image_all(*, no_pixels: bool, cuda: bool, invisible: str) -> str:
    """What ``all`` actually does here, measured rather than inferred from the GPU."""
    if no_pixels:
        return "needs_visible_extra"
    if cuda:
        return "ok"
    if invisible == "missing":
        # Writes the visible+metadata result and exits 1 with the invisible stage
        # named as skipped. A partial file, not a failure.
        return "partial_writes_visible_and_metadata"
    return "do_not_run_writes_nothing"


def _advice(
    *,
    cli_found: bool,
    pixels: str,
    cuda: bool,
    ffmpeg: bool,
    installer: str | None,
    outdated: bool | None,
    invisible: str,
) -> dict[str, str]:
    host = "https://raiw.cc"
    if not cli_found:
        if installer is None:
            return {
                "next": "install_python_tooling",
                "detail": (
                    "No remove-ai-watermarks CLI and no uv/pipx/pip. "
                    f"Send the user to {host} or install Python 3.11-3.14 first."
                ),
            }
        return {
            "next": "install_visible",
            "detail": f"Install with {installer}. Default extra is visible.",
        }

    # A found CLI without the pixel stack can still read metadata, and NOTHING else.
    # Reporting these as "ok" is what sent an agent into a missing-cv2 crash on a
    # machine the probe had just called ready.
    no_pixels = pixels == "missing"
    advice = {
        "identify": "metadata_only_no_pixels" if no_pixels else "ok",
        "visible": "needs_visible_extra" if no_pixels else "ok",
        "metadata": "ok",
        "video_visible": "needs_visible_extra" if no_pixels else ("ok" if ffmpeg else "needs_ffmpeg"),
        "invisible_images": "unavailable_no_cuda" if not cuda else ("needs_visible_extra" if no_pixels else "ok"),
        "image_all": _image_all(no_pixels=no_pixels, cuda=cuda, invisible=invisible),
        "video_invisible": "ok_cpu_or_gpu" if ffmpeg else "needs_ffmpeg",
        "no_cuda_fallback": host,
        "pixel_stack": pixels,
        "invisible_stack": invisible,
    }
    if no_pixels:
        advice["next"] = "install_visible"
        advice["detail"] = (
            "The installed CLI has no pixel dependencies, so only metadata commands work. "
            f"Reinstall with the visible extra using {installer or 'uv, pipx or pip'}. "
            "A Homebrew install cannot carry extras."
        )
    elif pixels == "unknown":
        advice["detail"] = (
            "The pixel check gave an unrecognized result; treat visible commands as unverified "
            "and read the first real error instead of assuming a cause."
        )
    if outdated:
        advice["upgrade_cli"] = (
            f"Installed CLI is older than {'.'.join(str(part) for part in MIN_CLI_VERSION)}; "
            "flags this skill documents may be missing. Upgrade before reporting a flag as broken."
        )
    return advice


def build_report() -> dict[str, Any]:
    installers = _installers()
    installer = _preferred_installer(installers)
    cli = _cli()
    cuda = _has_cuda()
    ffmpeg = _which("ffmpeg") is not None
    pixels = _pixel_stack(str(cli["path"])) if cli["found"] else "unknown"
    invisible = _invisible_stack(str(cli["path"])) if cli["found"] else "unknown"
    return {
        "os": os.name,
        "platform": sys.platform,
        "probe_python": _probe_python(),
        "cli": cli,
        "cuda": cuda,
        "ffmpeg": ffmpeg,
        "pixel_stack": pixels,
        "invisible_stack": invisible,
        "installers": {name: path is not None for name, path in installers.items()},
        "preferred_installer": installer,
        "advice": _advice(
            cli_found=bool(cli["found"]),
            pixels=pixels,
            cuda=cuda,
            ffmpeg=ffmpeg,
            installer=installer,
            outdated=cli["outdated"],
            invisible=invisible,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe this machine for remove-ai-watermarks.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Indent the JSON on stdout.",
    )
    args = parser.parse_args(argv)
    report = build_report()
    dumped = json.dumps(report, indent=2 if args.pretty else None, sort_keys=True)
    sys.stdout.write(dumped + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
