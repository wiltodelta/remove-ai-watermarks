"""Lazy, offline FFmpeg resolution for optional video operations."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from remove_ai_watermarks.optional_deps import module_available

log = logging.getLogger(__name__)


def bundled_video_tools_available() -> bool:
    """Check the offline payload without extracting it or loading ML dependencies."""
    return module_available("portable_ffmpeg", "portable_ffmpeg_full", "filelock")


@lru_cache(maxsize=1)
def _bundled_tools() -> tuple[str, str] | None:
    if not bundled_video_tools_available():
        return None
    from filelock import FileLock

    # The upstream extractor serializes threads. Serialize separate CLI workers
    # too, so simultaneous batch jobs cannot replace each other's staging cache.
    cache_dir = Path.home() / ".cache" / "remove-ai-watermarks" / "ffmpeg"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache_dir / "extract.lock"), timeout=120):
        # Extraction can print progress. Isolate it so library/CLI JSON output
        # and other threads' stdout remain untouched.
        command = [
            sys.executable,
            "-c",
            "import contextlib, json, sys\n"
            "from importlib.resources import files\n"
            "from pathlib import Path\n"
            "import portable_ffmpeg.core as core\n"
            "core.CACHE_DIR = Path(sys.argv[1])\n"
            "files('portable_ffmpeg_full')\n"
            "with contextlib.redirect_stdout(sys.stderr):\n"
            "    tools = core.get_ffmpeg()\n"
            "sys.stdout.write(json.dumps([str(p) for p in tools]))",
            str(cache_dir),
        ]
        result = subprocess.run(  # noqa: S603 - Fixed code in this interpreter, without a shell.
            command, capture_output=True, text=True, check=False, timeout=120
        )
        log.info(
            "FFmpeg bundle initialization: command=%s status=%s stdout=%s stderr=%s",
            command,
            result.returncode,
            result.stdout,
            result.stderr,
        )
        if result.returncode:
            raise RuntimeError(f"FFmpeg bundle initialization failed: {result.stderr}")
        ffmpeg, ffprobe = json.loads(result.stdout)
    log.info("Using bundled video tools: ffmpeg=%s ffprobe=%s", ffmpeg, ffprobe)
    return str(ffmpeg), str(ffprobe)


def find_video_tool(name: Literal["ffmpeg", "ffprobe"]) -> str | None:
    """Prefer a complete system pair, then the installed offline binary bundle.

    Base installations may still use a single system tool. Never call the
    upstream downloader unless the offline payload is installed. PATH is unchanged.
    """
    system = (shutil.which("ffmpeg"), shutil.which("ffprobe"))
    index = 0 if name == "ffmpeg" else 1
    if all(system):
        return system[index]
    try:
        bundled = _bundled_tools()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        log.exception("Could not initialize the installed FFmpeg bundle")
        raise RuntimeError("Could not initialize bundled FFmpeg; reinstall remove-ai-watermarks[video]") from error
    return bundled[index] if bundled else system[index]


def require_video_tool(name: Literal["ffmpeg", "ffprobe"]) -> str:
    """Resolve a tool or explain which extra supplies both executables."""
    executable = find_video_tool(name)
    if executable is None:
        raise RuntimeError(f"Video processing requires {name}; install remove-ai-watermarks[video]")
    return executable
