"""Strict native audio creation declarations, separate from waveform marks."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)
_CREATION = re.compile(
    r"made with suno(?P<studio> studio)?; created="
    r"(?P<created>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z)"
    r"(?:; id=(?P<id>[0-9a-f-]{36}))?",
    re.IGNORECASE | re.ASCII,
)
_MAX_RESPONSE = 1024 * 1024


def creation_tool(value: str | bytes) -> str | None:
    """Recognize a complete machine declaration, never an incidental mention."""
    if isinstance(value, bytes):
        for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
            try:
                if tool := creation_tool(value.decode(encoding)):
                    return tool
            except UnicodeError:
                continue
        return None
    match = _CREATION.fullmatch(value.strip(" \x00\ufeff"))
    if match is None or (not match["studio"] and match["id"] is None):
        return None
    try:
        datetime.fromisoformat(match["created"].upper().replace("Z", "+00:00"))
        if match["id"] is not None and str(UUID(match["id"])) != match["id"].lower():
            return None
    except ValueError:
        return None
    return "Suno Studio" if match["studio"] else "Suno"


@lru_cache(maxsize=256)
def _probe(path: str, mtime_ns: int, size: int, ffprobe: str) -> str | None:
    del mtime_ns, size  # Both participate in the cache identity.
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type:stream_disposition=attached_pic:stream_tags=comment:format_tags=comment",
        "-of",
        "json",
        path,
    ]
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False, timeout=30)  # noqa: S603
        stdout.seek(0)
        body = stdout.read(_MAX_RESPONSE + 1)
        stderr.seek(0)
        diagnostic = stderr.read(_MAX_RESPONSE).decode("utf-8", "replace")
    log.info(
        "Audio creation probe: command=%s status=%s response_bytes=%s stderr=%s",
        command,
        result.returncode,
        len(body),
        diagnostic,
    )
    if result.returncode or len(body) > _MAX_RESPONSE:
        log.warning(
            "Audio creation probe was incomplete: command=%s status=%s stderr=%s",
            command,
            result.returncode,
            diagnostic,
        )
        return None
    try:
        document = json.loads(body)
        streams = document.get("streams", [])
        if not any(stream.get("codec_type") == "audio" for stream in streams):
            return None
        if any(
            stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic")
            for stream in streams
        ):
            return None
        tags = [
            document.get("format", {}).get("tags", {}),
            *(stream.get("tags", {}) for stream in streams if stream.get("codec_type") == "audio"),
        ]
        for group in tags:
            for key, value in group.items():
                if key.casefold() == "comment" and isinstance(value, str) and (tool := creation_tool(value)):
                    return tool
    except (ValueError, AttributeError, TypeError) as exc:
        log.warning("Malformed audio creation probe: command=%s response_bytes=%s error=%s", command, len(body), exc)
    return None


def audio_creation_tool(path: Path) -> str | None:
    """Read a declaration only from an audio container, ignoring attached covers.

    ffprobe supplies structured tag and stream boundaries. Missing ffprobe leaves
    this optional declaration unavailable; the explicit audio API requires it.
    No waveform or authenticity claim is made by a creation comment.
    """
    try:
        with path.open("rb") as stream:
            head = stream.read(12)
        if head[4:8] == b"ftyp":
            from remove_ai_watermarks._internal.isobmff import has_audio_creation_comment

            if not has_audio_creation_comment(path):
                return None
        elif not (
            head.startswith((b"ID3", b"fLaC", b"OggS"))
            or (head[:4] == b"RIFF" and head[8:12] == b"WAVE")
            or (len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)
        ):
            return None
        from remove_ai_watermarks import video_tools

        if (ffprobe := shutil.which("ffprobe") or video_tools.find_video_tool("ffprobe")) is None:
            return None
        stat = path.stat()
        return _probe(str(path.resolve()), stat.st_mtime_ns, stat.st_size, ffprobe)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("Audio creation inspection failed: path=%s error=%s", path, exc)
        return None
