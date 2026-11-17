"""Model Context Protocol server over stdio, a thin wrapper around this package's CLI.

``remove-ai-watermarks mcp`` speaks newline-delimited JSON-RPC 2.0 on stdin and
stdout with the initialize handshake. Each tool call runs one CLI command in a
child process of the same interpreter, so a tool returns exactly what that command
would: its exit code, its output file and its text. Nothing but protocol messages
reaches stdout, and a crash inside a command cannot corrupt the stream.

The server needs no dependency beyond the standard library, so the bare package
runs it through ``uvx remove-ai-watermarks mcp``. Commands that need an optional
extra fail with the CLI's own install hint, returned as a tool error.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import subprocess
import sys
from collections.abc import Callable
from functools import lru_cache, partial
from pathlib import Path
from typing import IO, Any, cast

from remove_ai_watermarks import __version__
from remove_ai_watermarks._internal.constants import (
    AUDIO_BANDWIDTHS_KBPS,
    AUDIO_EXTENSIONS,
    AUDIO_METHODS,
    AUDIO_SCHEMES,
    AUDIO_VOCODER_DEVICES,
    DEFAULT_VIDEO_MESSAGE_LONG_SIDE,
    VIDEO_MESSAGE_DEVICES,
    VIDEO_MESSAGE_METHODS,
    VIDEO_MESSAGE_SCHEMES,
)
from remove_ai_watermarks.integration import input_schema, plan_operation, run_operation
from remove_ai_watermarks.text import TEXT_EXTENSIONS
from remove_ai_watermarks.video import VIDEO_EXTENSIONS

log = logging.getLogger(__name__)

# Newest first. A client asking for an unlisted version gets the newest, and
# decides for itself whether it can continue.
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

INSTRUCTIONS = (
    "Identify and remove AI provenance marks from images, video, audio, and text the user generated "
    "or edited themselves: C2PA, IPTC, EXIF and TC260 metadata, visible AI-generator "
    "labels, and supported invisible watermarks. Do not use it on stock, agency, "
    "marketplace or other third-party marks that protect someone else's asset. A "
    "result of 'unknown' or exit code 2 means no supported signal was found, not "
    "that the file is clean. Paths are read on the machine running the server."
)

_MODES = ("metadata", "visible", "all")

TOOLS: list[dict[str, Any]] = [
    {
        "name": "identify",
        "title": "Identify AI provenance",
        "description": (
            "Report which AI provenance signals an image or video carries: metadata "
            "(C2PA, IPTC, EXIF, TC260), visible AI-generator labels, and supported local "
            "invisible-watermark decoders. Read-only. Video is detected by file suffix."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Image or video file to inspect."},
                "check_visible": {
                    "type": "boolean",
                    "default": True,
                    "description": "Run the pixel and visible-mark detectors. False reads metadata only.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "remove",
        "title": "Remove AI watermarks from one file",
        "description": (
            "Write a cleaned copy of an image or video the user generated or edited. "
            "mode=metadata strips AI metadata without touching pixels or re-encoding video; "
            "mode=visible also removes a known visible AI label; for an image, mode=all adds "
            "the invisible-watermark scrub where the installed extras and hardware allow it. "
            "The source is never overwritten; the default output is <name>_clean beside it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Image or video file to clean."},
                "mode": {"type": "string", "enum": list(_MODES), "default": "metadata"},
                "output": {"type": "string", "description": "Output file. Must differ from the source."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    },
    {
        "name": "batch",
        "title": "Remove AI watermarks from a directory",
        "description": (
            "Clean every supported image (media=image) or video (media=video) in a directory "
            "the user owns, writing results to a separate directory (default <dir>_clean). "
            "Modes as for the remove tool."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory of input files."},
                "mode": {"type": "string", "enum": list(_MODES), "default": "metadata"},
                "media": {"type": "string", "enum": ["image", "video", "audio"], "default": "image"},
                "output_dir": {"type": "string", "description": "Output directory. Must differ from the input."},
            },
            "required": ["directory"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    },
]

for _name, _description, _options in (
    (
        "audio_identify",
        "Inspect audio metadata and named compatible public-model evidence. Read-only.",
        {"check_invisible": {"type": "boolean", "default": True}},
    ),
    (
        "audio_invisible",
        "Process one audio track to WAV, FLAC, MP3, M4A, OGG, or Opus. Default publication requires "
        "a compatible source positive under the requested models and a fully negative saved output. "
        "Force permits explicitly unverified results.",
        {},
    ),
    (
        "video_audio",
        "Process one selected audio track (or all source-positive tracks with all_streams=true) inside a video, "
        "copying other encoded streams. Outputs MKV/MOV with float PCM or MP4 with selected-track AAC "
        "(target 192 kbps). "
        "Visual marks and untouched audio tracks remain unchanged. "
        "Default publication requires compatible source-positive and final-mux-negative checks "
        "for the requested models.",
        {"all_streams": {"type": "boolean", "default": False}},
    ),
):
    _options = {
        **_options,
        "scheme": {
            "type": "string",
            "enum": list(AUDIO_SCHEMES),
            "default": AUDIO_SCHEMES[0],
        },
    }
    if _name != "audio_identify":
        _options = {
            **_options,
            "output": {"type": "string"},
            "force": {"type": "boolean", "default": False},
            "verify": {"type": "boolean", "default": True},
            "method": {
                "type": "string",
                "enum": list(AUDIO_METHODS),
                "default": AUDIO_METHODS[0],
                "description": (
                    "Mimi, FocalCodec, DAC, and BigVGAN are explicit experimental full-track profiles, "
                    "not SynthID decoders."
                ),
            },
            "vocoder_device": {
                "type": "string",
                "enum": list(AUDIO_VOCODER_DEVICES),
                "default": AUDIO_VOCODER_DEVICES[0],
            },
            "bandwidth": {
                "type": "integer",
                "enum": list(AUDIO_BANDWIDTHS_KBPS),
                "default": AUDIO_BANDWIDTHS_KBPS[-1],
            },
        }
    TOOLS.append(
        {
            "name": _name,
            "description": _description,
            "inputSchema": {
                "type": "object",
                "required": ["path"],
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string"},
                    "stream": {"type": "integer", "minimum": 0, "default": 0},
                    **_options,
                },
            },
            "annotations": {
                "readOnlyHint": _name == "audio_identify",
                "destructiveHint": False,
                "openWorldHint": False,
            },
        }
    )

for _action in ("identify", "remove"):
    _properties: dict[str, Any] = {
        "path": {"type": "string"},
        "reference": {
            "type": "string",
            "description": "Independent encoder message, 64 hex digits. Never derive it from this input.",
        },
        "scheme": {
            "type": "string",
            "enum": list(VIDEO_MESSAGE_SCHEMES),
            "default": VIDEO_MESSAGE_SCHEMES[0],
        },
    }
    if _action == "remove":
        _properties.update(
            {
                "output": {"type": "string"},
                "method": {
                    "type": "string",
                    "enum": list(VIDEO_MESSAGE_METHODS),
                    "default": VIDEO_MESSAGE_METHODS[0],
                },
                "noise_std": {"type": "number", "minimum": 0, "maximum": 1},
                "long_side": {"type": "integer", "minimum": 8, "default": DEFAULT_VIDEO_MESSAGE_LONG_SIDE},
                "device": {
                    "type": "string",
                    "enum": list(VIDEO_MESSAGE_DEVICES),
                    "default": VIDEO_MESSAGE_DEVICES[0],
                },
                "force": {"type": "boolean", "default": False},
                "verify": {"type": "boolean", "default": True},
            }
        )
    TOOLS.append(
        {
            "name": f"video_message_{_action}",
            "description": "Experimental known-message correlation under pinned public VideoSeal or PixelSeal models. "
            "Requires an independently known encoder message. Does not identify AI origin or unknown marks. "
            + (
                "Read-only, checks every decoded frame."
                if _action == "identify"
                else "Writes a processed video after saved-file checks; copies audio and reports delivered quality."
            ),
            "inputSchema": {
                "type": "object",
                "properties": _properties,
                "required": ["path", "reference"],
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": _action == "identify", "destructiveHint": False, "openWorldHint": False},
        }
    )

_TOOL_SCHEMAS = {tool["name"]: tool["inputSchema"] for tool in TOOLS}

Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


class ToolInputError(ValueError):
    """A tool call whose arguments the server refuses before running anything."""


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one CLI command in a child process of this interpreter."""
    env = {**os.environ, "COLUMNS": "120", "NO_COLOR": "1", "TERM": "dumb"}
    command = [sys.executable, "-m", "remove_ai_watermarks.cli", *args]
    log.info("Running CLI command: %s", command)
    completed = subprocess.run(  # noqa: S603 -- fixed interpreter, argument list, no shell
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    log.info("CLI command exited %d: %s", completed.returncode, command)
    return completed


def _validated(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Check arguments against the tool's own input schema and fill its defaults."""
    schema = _TOOL_SCHEMAS[tool]
    properties: dict[str, dict[str, Any]] = schema["properties"]
    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        raise ToolInputError(f"unknown argument(s): {', '.join(unknown)}")
    values: dict[str, Any] = {}
    for key, spec in properties.items():
        if key not in arguments:
            if key in schema["required"]:
                raise ToolInputError(f"'{key}' is required")
            if "default" in spec:
                values[key] = spec["default"]
            continue
        value = arguments[key]
        if spec["type"] == "boolean" and not isinstance(value, bool):
            raise ToolInputError(f"'{key}' must be a boolean")
        if spec["type"] == "integer" and (type(value) is not int or value < spec.get("minimum", 0)):
            raise ToolInputError(f"'{key}' must be a nonnegative integer")
        if spec["type"] == "number" and (
            type(value) not in (int, float)
            or not spec.get("minimum", -math.inf) <= value <= spec.get("maximum", math.inf)
            or not math.isfinite(value)
        ):
            raise ToolInputError(f"'{key}' must be a finite number within its declared bounds")
        if spec["type"] == "string" and (not isinstance(value, str) or not value.strip()):
            raise ToolInputError(f"'{key}' must be a non-empty string")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolInputError(f"'{key}' must be one of {', '.join(str(item) for item in spec['enum'])}")
        values[key] = value
    return values


def _is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def _is_audio(path: str) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


def _is_text(path: str) -> bool:
    return Path(path).suffix.lower() in TEXT_EXTENSIONS


def _identify_command(values: dict[str, Any]) -> tuple[list[str], str | None]:
    path = values["path"]
    if _is_text(path):
        return ["text", "identify", "--json", "--", path], None
    if _is_audio(path):
        flags = [] if values["check_visible"] else ["--no-invisible"]
        return ["audio", "identify", "--json", *flags, "--", path], None
    prefix = ["video", "identify"] if _is_video(path) else ["identify"]
    return [*prefix, "--json", *([] if values["check_visible"] else ["--no-visible"]), "--", path], None


def _require_distinct_output(source: Path, output: str, *, message: str) -> None:
    out_path = Path(output)
    if out_path.resolve() == source.resolve() or (out_path.exists() and source.exists() and out_path.samefile(source)):
        raise ToolInputError(message)


def _remove_command(values: dict[str, Any]) -> tuple[list[str], str | None]:
    path, mode = values["path"], values["mode"]
    source = Path(path)
    output = values.get("output") or str(source.with_stem(source.stem + "_clean"))
    _require_distinct_output(
        source,
        output,
        message="'output' must differ from the source; the server never overwrites an input",
    )
    if _is_text(path) and mode == "visible":
        raise ToolInputError("text files do not have a visible pixel mode; use mode=metadata or mode=all")
    if _is_audio(path):
        if mode == "visible":
            raise ToolInputError("audio files do not have a visible pixel mode; use mode=metadata or mode=all")
        audio_cmd = ["audio", "metadata", "--remove"] if mode == "metadata" else ["audio", "all", "--invisible"]
        return [*audio_cmd, "-o", output, "--", path], output
    prefix = ["text"] if _is_text(path) else ["video"] if _is_video(path) else []
    command = ["metadata", "--remove"] if mode == "metadata" else [mode]
    return [*prefix, *command, "-o", output, "--", path], output


def _batch_command(values: dict[str, Any]) -> tuple[list[str], str | None]:
    directory = values["directory"]
    resolved = Path(directory).resolve()
    output_dir = values.get("output_dir") or str(resolved.with_name(resolved.name + "_clean"))
    if Path(output_dir).resolve() == resolved:
        raise ToolInputError("'output_dir' must differ from the input directory")
    if values["media"] == "audio":
        if values["mode"] == "visible":
            raise ToolInputError("audio batch does not have a visible pixel mode; use mode=metadata or mode=all")
        extra = ["--invisible"] if values["mode"] == "all" else []
        return ["audio", "batch", "--mode", values["mode"], *extra, "-o", output_dir, "--", directory], output_dir
    command = ["video", "batch"] if values["media"] == "video" else ["batch"]
    return [*command, "--mode", values["mode"], "-o", output_dir, "--", directory], output_dir


def _validate_output_and_verify_flags(source: Path, output: str, values: dict[str, Any]) -> list[str]:
    _require_distinct_output(source, output, message="'output' must differ from the source")
    if not values["verify"] and not values["force"]:
        raise ToolInputError("Disabling verification requires force=true")
    flags = ["-o", output]
    if values["force"]:
        flags.append("--force")
    if not values["verify"]:
        flags.append("--no-verify")
    return flags


def _audio_command(values: dict[str, Any], *, video: bool = False) -> tuple[list[str], str | None]:
    source = Path(values["path"])
    identify = "check_invisible" in values
    prefix = ["video", "audio"] if video else ["audio", "identify" if identify else "invisible"]
    command = [*prefix, "--json", "--stream", str(values["stream"])]
    output = None
    if identify:
        if not values["check_invisible"]:
            command.append("--no-invisible")
    else:
        suffix = "_audio_processed.mkv" if video else "_processed.wav"
        output = values.get("output") or str(source.with_name(source.stem + suffix))
        command.extend(_validate_output_and_verify_flags(source, output, values))
        if values["method"] != AUDIO_METHODS[0]:
            command.extend(["--method", values["method"]])
        if values["bandwidth"] != AUDIO_BANDWIDTHS_KBPS[-1]:
            command.extend(["--bandwidth", str(values["bandwidth"])])
        if values["vocoder_device"] != AUDIO_VOCODER_DEVICES[0]:
            command.extend(["--vocoder-device", values["vocoder_device"]])
    if values.get("all_streams"):
        command.append("--all-streams")
    if values["scheme"] != AUDIO_SCHEMES[0]:
        command.extend(["--scheme", values["scheme"]])
    return [*command, "--", values["path"]], output


def _video_message_command(values: dict[str, Any], *, remove: bool = False) -> tuple[list[str], str | None]:
    command = [
        "video",
        "message",
        "remove" if remove else "identify",
        "--json",
        "--reference",
        values["reference"],
        "--scheme",
        values["scheme"],
    ]
    output = None
    if remove:
        source = Path(values["path"])
        output = values.get("output") or str(source.with_name(source.stem + "_message_processed.mp4"))
        if "noise_std" in values and values["method"] != "vae":
            raise ToolInputError("noise_std requires method=vae")
        command.extend(_validate_output_and_verify_flags(source, output, values))
        command.extend(
            [
                "--method",
                values["method"],
                "--long-side",
                str(values["long_side"]),
                "--device",
                values["device"],
            ]
        )
        if "noise_std" in values:
            command.extend(["--noise-std", str(values["noise_std"])])
    return [*command, "--", values["path"]], output


_COMMANDS: dict[str, Callable[[dict[str, Any]], tuple[list[str], str | None]]] = {
    "identify": _identify_command,
    "remove": _remove_command,
    "batch": _batch_command,
    "audio_identify": _audio_command,
    "audio_invisible": _audio_command,
    "video_audio": partial(_audio_command, video=True),
    "video_message_identify": _video_message_command,
    "video_message_remove": partial(_video_message_command, remove=True),
}


@lru_cache(maxsize=1)
def _operations() -> dict[str, dict[str, Any]]:
    """Discover the installed command surface without importing Click into MCP."""
    completed = subprocess.run(
        [sys.executable, "-m", "remove_ai_watermarks.integration", "catalog"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {operation["id"]: operation for operation in json.loads(completed.stdout)}


def tools() -> list[dict[str, Any]]:
    """Keep convenient legacy tools and expose every installed operation."""
    return [
        *TOOLS,
        *(
            {
                "name": operation["id"],
                "title": operation["title"],
                "description": operation["description"],
                "inputSchema": input_schema(operation),
                "annotations": {"openWorldHint": False},
            }
            for operation in _operations().values()
        ),
    ]


def call_tool(name: str, arguments: dict[str, Any], runner: Runner = run_cli) -> dict[str, Any]:
    """Run one tool and return an MCP ``tools/call`` result."""
    if name not in _COMMANDS:
        operation = _operations()[name]
        try:
            args, output = plan_operation(operation, arguments)
            result = run_operation(operation, arguments)
        except (ValueError, OSError) as exc:
            return {"content": [{"type": "text", "text": f"Invalid arguments: {exc}"}], "isError": True}
        token = arguments.get("hf_token")
        if token:
            args = ["[redacted]" if arg == token else arg for arg in args]
        completed = subprocess.CompletedProcess(args, result["exit_code"], result["stdout"], result["stderr"])
        formatted = _tool_result(completed, args, result["output"])
        formatted["structuredContent"]["operation"] = name
        if not args:
            formatted["structuredContent"]["command"] = [
                sys.executable,
                "-m",
                "remove_ai_watermarks.integration",
                "execute",
                name,
            ]
        return formatted
    try:
        args, output = _COMMANDS[name](_validated(name, arguments))
    except ToolInputError as exc:
        return {"content": [{"type": "text", "text": f"Invalid arguments: {exc}"}], "isError": True}
    try:
        completed = runner(args)
    except OSError as exc:  # the child never started, so there is no exit code to relay
        log.warning("CLI command could not start: %s: %s", args, exc)
        return {"content": [{"type": "text", "text": f"The command could not start: {exc}"}], "isError": True}
    return _tool_result(completed, args, output)


def _tool_result(completed: subprocess.CompletedProcess[str], args: list[str], output: str | None) -> dict[str, Any]:
    """Preserve CLI exit semantics for legacy and catalog-backed operations."""
    code = completed.returncode
    text = completed.stdout.strip()
    stderr = completed.stderr.strip()[-4000:]
    structured: dict[str, Any] = {"command": ["remove-ai-watermarks", *args], "exit_code": code, "output": output}
    if ("--json" in args or not args) and text:
        with contextlib.suppress(json.JSONDecodeError):
            structured["report"] = json.loads(text)
            if isinstance(structured["report"], dict) and "output" in structured["report"]:
                structured["output"] = structured["report"]["output"]
    if "report" not in structured:
        structured["stdout"] = text
    if stderr:
        structured["stderr"] = stderr
    # The CLI's own exit 2 means no supported signal, or a batch directory with nothing
    # to process, both results. Click also exits 2 on
    # a usage error, such as a missing file, and prints its usage line on stderr.
    no_signal = code == 2 and "Usage:" not in stderr
    is_error = code != 0 and not no_signal
    summary = f"exit code {code}"
    if no_signal:
        summary += " (no supported signal found, or nothing to process; this is unknown, not proof of a clean file)"
    body = "\n\n".join(part for part in (summary, text, stderr if is_error else "") if part)
    return {"content": [{"type": "text", "text": body}], "structuredContent": structured, "isError": is_error}


def _error(request_id: object, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _object(value: object) -> dict[str, Any] | None:
    """A decoded JSON object, or ``None`` for any other JSON value."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def handle(raw: object, runner: Runner = run_cli) -> dict[str, Any] | None:
    """Answer one JSON-RPC message; ``None`` for notifications and client responses."""
    message = _object(raw)
    if message is None or message.get("jsonrpc") != "2.0":
        return _error(None, -32600, "Invalid Request")
    method = message.get("method")
    if "id" not in message:
        return None  # a notification or a response to nothing this server sent
    request_id: object = message["id"]
    if not isinstance(method, str):
        return _error(request_id, -32600, "Invalid Request")
    params = _object(message.get("params") or {})
    if params is None:
        return _error(request_id, -32602, "params must be an object")

    if method == "initialize":
        requested = params.get("protocolVersion")
        result: dict[str, Any] = {
            "protocolVersion": requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "remove-ai-watermarks", "title": "Remove AI Watermarks", "version": __version__},
            "instructions": INSTRUCTIONS,
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": tools()}
    elif method == "tools/call":
        name = params.get("name")
        arguments = _object(params.get("arguments") or {})
        if not isinstance(name, str) or (name not in _COMMANDS and name not in _operations()):
            return _error(request_id, -32602, f"Unknown tool: {name}")
        if arguments is None:
            return _error(request_id, -32602, "arguments must be an object")
        result = call_tool(name, arguments, runner)
    else:
        return _error(request_id, -32601, f"Method not found: {method}")
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def serve(stdin: IO[str] | None = None, stdout: IO[str] | None = None, runner: Runner = run_cli) -> None:
    """Serve newline-delimited JSON-RPC until stdin closes."""
    source: IO[str] = sys.stdin if stdin is None else stdin
    sink: IO[str] = sys.stdout if stdout is None else stdout
    for line in source:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response: dict[str, Any] | None = _error(None, -32700, "Parse error")
        else:
            response = handle(message, runner)
        if response is not None:
            sink.write(json.dumps(response, ensure_ascii=False) + "\n")
            sink.flush()
