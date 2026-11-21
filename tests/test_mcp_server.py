"""`remove-ai-watermarks mcp`: the protocol layer, the CLI mapping, and real clients."""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from remove_ai_watermarks import mcp_server
from remove_ai_watermarks.metadata import KLING_SEI_TAG

ROOT = Path(__file__).resolve().parents[1]
KLING_MP4 = ROOT / "data/fixtures/provenance/higgsfield-kling-3-0-turbo.mp4"
SERVER = [sys.executable, "-m", "remove_ai_watermarks.cli", "mcp"]


class _Recorder:
    """Stands in for the CLI child process and records each argument list."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.calls: list[list[str]] = []
        self._result = (returncode, stdout, stderr)

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        returncode, stdout, stderr = self._result
        return subprocess.CompletedProcess(args, returncode, stdout, stderr)


def test_audio_tools_map_tracks_and_explicit_verification_options():
    runner = _Recorder(stdout='{"verified_removed": null}')
    result = mcp_server.call_tool(
        "audio_invisible", {"path": "speech.wav", "stream": 1, "force": True, "verify": False}, runner
    )
    assert runner.calls == [
        [
            "audio",
            "invisible",
            "--json",
            "--stream",
            "1",
            "-o",
            "speech_processed.wav",
            "--force",
            "--no-verify",
            "--",
            "speech.wav",
        ]
    ]
    assert result["structuredContent"]["report"]["verified_removed"] is None
    runner = _Recorder(stdout="{}")
    mcp_server.call_tool("audio_identify", {"path": "speech.wav", "check_invisible": False}, runner)
    assert runner.calls == [["audio", "identify", "--json", "--stream", "0", "--no-invisible", "--", "speech.wav"]]


@pytest.mark.parametrize("stream", [-1, True, "0", 0.5])
def test_audio_tools_reject_invalid_track_ordinals(stream):
    runner = _Recorder()
    result = mcp_server.call_tool("audio_identify", {"path": "a.wav", "stream": stream}, runner)
    assert result["isError"] is True
    assert runner.calls == []


@pytest.mark.parametrize("tool", ["audio_identify", "audio_invisible", "video_audio"])
def test_audio_tools_preserve_explicit_silentcipher16_scope(tool):
    runner = _Recorder(stdout="{}")
    arguments = {"path": "clip.mkv" if tool == "video_audio" else "speech.wav", "scheme": "silentcipher16"}
    if tool != "audio_identify":
        arguments["output"] = "processed.mkv" if tool == "video_audio" else "processed.wav"
    result = mcp_server.call_tool(tool, arguments, runner)
    assert result.get("isError") is not True
    assert len(runner.calls) == 1
    command = runner.calls[0]
    assert command[command.index("--scheme") + 1] == "silentcipher16"


def test_audio_tool_refuses_source_output():
    runner = _Recorder()
    result = mcp_server.call_tool("audio_invisible", {"path": "a.wav", "output": "a.wav"}, runner)
    assert result["isError"] is True
    assert runner.calls == []


def test_audio_tool_skip_does_not_report_an_unwritten_output():
    result = mcp_server.call_tool(
        "audio_invisible", {"path": "a.wav"}, _Recorder(returncode=2, stdout='{"skipped": true, "output": null}')
    )
    assert result["isError"] is False
    assert result["structuredContent"]["output"] is None
    assert result["structuredContent"]["report"]["skipped"] is True


def test_video_audio_tool_maps_to_the_mux_command():
    runner = _Recorder(stdout="{}")
    mcp_server.call_tool("video_audio", {"path": "input.mp4", "stream": 1}, runner)
    assert runner.calls == [
        ["video", "audio", "--json", "--stream", "1", "-o", "input_audio_processed.mkv", "--", "input.mp4"]
    ]


def test_video_audio_MP4_output_reaches_legacy_and_catalog_routes():
    from remove_ai_watermarks.integration import operation_catalog, plan_operation

    runner = _Recorder(stdout="{}")
    mcp_server.call_tool("video_audio", {"path": "input.mov", "output": "processed.mp4", "stream": 1}, runner)
    assert runner.calls == [["video", "audio", "--json", "--stream", "1", "-o", "processed.mp4", "--", "input.mov"]]
    operation = next(item for item in operation_catalog() if item["id"] == "cli_video_audio")
    args, output = plan_operation(operation, {"source": "input.mov", "output": "processed.mp4", "stream": 1})
    assert output == "processed.mp4"
    output_flag = "-o" if "-o" in args else "--output"
    assert args[args.index(output_flag) + 1] == "processed.mp4"
    parameter = next(item for item in operation["parameters"] if item["name"] == "output")
    assert "MP4 with AAC" in parameter["schema"]["description"]


def test_neural_audio_options_are_forwarded_to_the_cli():
    runner = _Recorder(stdout="{}")
    result = mcp_server.call_tool("audio_invisible", {"path": "input.wav", "method": "encodec", "bandwidth": 6}, runner)
    assert result["isError"] is False
    assert runner.calls[0][-6:] == ["--method", "encodec", "--bandwidth", "6", "--", "input.wav"]


def test_bigvgan_audio_options_reach_legacy_and_catalog_mcp():
    runner = _Recorder(stdout="{}")
    result = mcp_server.call_tool(
        "audio_invisible", {"path": "input.wav", "method": "bigvgan", "vocoder_device": "mps"}, runner
    )
    assert result["isError"] is False
    assert runner.calls[0][-6:] == ["--method", "bigvgan", "--vocoder-device", "mps", "--", "input.wav"]
    from remove_ai_watermarks.integration import operation_catalog, plan_operation

    operation = next(item for item in operation_catalog() if item["id"] == "cli_video_audio")
    args, _ = plan_operation(operation, {"source": "input.mp4", "method": "bigvgan", "vocoder_device": "mps"})
    assert args[args.index("--method") + 1] == "bigvgan"
    assert args[args.index("--vocoder-device") + 1] == "mps"


def test_invalid_neural_audio_bandwidth_is_rejected_before_launch():
    runner = _Recorder()
    result = mcp_server.call_tool("audio_invisible", {"path": "input.wav", "bandwidth": 7}, runner)
    assert result["isError"] is True
    assert runner.calls == []


def _request(method: str, params: dict | None = None, request_id: int = 1) -> dict:
    message: dict = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def _assert_the_cli_accepts(args: list[str]) -> None:
    """Every subcommand and flag a tool builds must exist in the real Click tree."""
    import click

    from remove_ai_watermarks.cli import main

    command: click.Command = main
    position = 0
    while isinstance(command, click.Group):
        command = command.commands[args[position]]
        position += 1
    options = {option for param in command.params for option in (*param.opts, *param.secondary_opts)}
    flags = [arg for arg in args[position : args.index("--")] if arg.startswith("-")]
    assert set(flags) <= options, (args, sorted(set(flags) - options))


@pytest.mark.parametrize(
    ("requested", "negotiated"),
    [("2025-06-18", "2025-06-18"), ("2024-11-05", "2024-11-05"), ("2099-01-01", mcp_server.PROTOCOL_VERSIONS[0])],
)
def test_initialize_negotiates_a_supported_protocol_version(requested: str, negotiated: str) -> None:
    response = mcp_server.handle(_request("initialize", {"protocolVersion": requested, "capabilities": {}}))

    assert response is not None
    assert response["result"]["protocolVersion"] == negotiated
    assert response["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert "third-party" in response["result"]["instructions"]


def test_notifications_get_no_reply_and_unknown_methods_get_method_not_found() -> None:
    assert mcp_server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    # A modern client probes server/discover first and falls back to initialize on this error.
    response = mcp_server.handle(_request("server/discover"))
    assert response is not None
    assert response["error"]["code"] == -32601


def test_tools_list_exposes_convenience_and_all_catalog_tools() -> None:
    response = mcp_server.handle(_request("tools/list"))

    assert response is not None
    tools = response["result"]["tools"]
    legacy = [tool["name"] for tool in mcp_server.TOOLS]
    assert [tool["name"] for tool in tools[: len(legacy)]] == legacy
    assert {tool["name"] for tool in tools[len(legacy) :]} == set(mcp_server._operations())
    assert tools[0]["annotations"]["readOnlyHint"] is True
    for tool in tools:
        assert tool["inputSchema"]["type"] == "object"
        assert tool["inputSchema"]["additionalProperties"] is False


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"path": "a.png"}, ["identify", "--json", "--", "a.png"]),
        ({"path": "a.png", "check_visible": False}, ["identify", "--json", "--no-visible", "--", "a.png"]),
        ({"path": "clip.MP4"}, ["video", "identify", "--json", "--", "clip.MP4"]),
        ({"path": "note.md"}, ["text", "identify", "--json", "--", "note.md"]),
    ],
)
def test_identify_maps_to_the_cli(arguments: dict, expected: list[str]) -> None:
    runner = _Recorder(stdout='{"is_ai_generated": true}')

    result = mcp_server.call_tool("identify", arguments, runner)

    assert runner.calls == [expected]
    _assert_the_cli_accepts(expected)
    assert result["isError"] is False
    assert result["structuredContent"]["report"] == {"is_ai_generated": True}


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"path": "d/a.png"}, ["metadata", "--remove", "-o", str(Path("d/a_clean.png")), "--", "d/a.png"]),
        ({"path": "a.jpg", "mode": "visible"}, ["visible", "-o", "a_clean.jpg", "--", "a.jpg"]),
        ({"path": "a.png", "mode": "all", "output": "o.png"}, ["all", "-o", "o.png", "--", "a.png"]),
        ({"path": "c.mov"}, ["video", "metadata", "--remove", "-o", "c_clean.mov", "--", "c.mov"]),
        ({"path": "c.mp4", "mode": "visible"}, ["video", "visible", "-o", "c_clean.mp4", "--", "c.mp4"]),
        ({"path": "doc.md"}, ["text", "metadata", "--remove", "-o", "doc_clean.md", "--", "doc.md"]),
        ({"path": "doc.md", "mode": "all"}, ["text", "all", "-o", "doc_clean.md", "--", "doc.md"]),
    ],
)
def test_remove_maps_to_the_cli_and_always_names_an_output(arguments: dict, expected: list[str]) -> None:
    runner = _Recorder()

    result = mcp_server.call_tool("remove", arguments, runner)

    assert runner.calls == [expected]
    _assert_the_cli_accepts(expected)
    assert result["structuredContent"]["output"] == expected[expected.index("-o") + 1]


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"directory": "in"}, ["batch", "--mode", "metadata", "-o", str(Path("in_clean").resolve()), "--", "in"]),
        (
            {"directory": "in", "media": "video", "mode": "visible", "output_dir": "out"},
            ["video", "batch", "--mode", "visible", "-o", "out", "--", "in"],
        ),
    ],
)
def test_batch_maps_to_the_cli(arguments: dict, expected: list[str]) -> None:
    runner = _Recorder()

    result = mcp_server.call_tool("batch", arguments, runner)

    assert runner.calls == [expected]
    assert result["structuredContent"]["output"] == expected[expected.index("-o") + 1]
    _assert_the_cli_accepts(expected)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("remove", {"path": "a.png", "output": "./a.png"}),
        ("remove", {"path": "a.png", "mode": "invisible"}),
        ("remove", {"path": ""}),
        ("remove", {}),
        ("identify", {"path": "a.png", "check_visible": "no"}),
        ("identify", {"path": "a.png", "shell": "rm -rf /"}),
        ("batch", {"directory": "in", "output_dir": "in"}),
    ],
)
def test_invalid_arguments_are_refused_before_anything_runs(tool: str, arguments: dict) -> None:
    runner = _Recorder()

    result = mcp_server.call_tool(tool, arguments, runner)

    assert runner.calls == []
    assert result["isError"] is True


@pytest.mark.parametrize(
    ("returncode", "stderr", "is_error"),
    [
        (0, "", False),
        (2, "", False),
        (1, "Error: boom", True),
        # Click's own usage errors also exit 2; they are failures, not a no-signal verdict.
        (2, "Usage: cli metadata [OPTIONS] SOURCE\nError: Invalid value for 'SOURCE'", True),
    ],
)
def test_only_failures_are_tool_errors(returncode: int, stderr: str, is_error: bool) -> None:
    runner = _Recorder(returncode=returncode, stdout="table", stderr=stderr)

    result = mcp_server.call_tool("remove", {"path": "a.png"}, runner)

    assert result["isError"] is is_error
    assert result["structuredContent"]["exit_code"] == returncode
    assert ("not proof of a clean file" in result["content"][0]["text"]) is (returncode == 2 and not is_error)
    if is_error:
        assert stderr.splitlines()[-1] in result["content"][0]["text"]


def test_serve_answers_line_by_line_and_reports_parse_errors() -> None:
    stdin = io.StringIO(
        json.dumps(_request("ping", request_id=7))
        + "\n\nnot json\n"
        + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
        + "\n"
    )
    stdout = io.StringIO()

    mcp_server.serve(stdin, stdout, _Recorder())

    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert replies == [
        {"jsonrpc": "2.0", "id": 7, "result": {}},
        {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
    ]


def test_the_cli_subcommand_serves_real_commands_over_stdio(tmp_path: Path) -> None:
    """End to end through the installed CLI: every stdout line must be a protocol message."""
    output = tmp_path / "clean.mp4"
    messages = [
        _request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}}, 1),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        _request("tools/call", {"name": "identify", "arguments": {"path": str(KLING_MP4), "check_visible": False}}, 2),
        _request("tools/call", {"name": "remove", "arguments": {"path": str(KLING_MP4), "output": str(output)}}, 3),
    ]
    completed = subprocess.run(  # noqa: S603
        SERVER,
        input="".join(json.dumps(message) + "\n" for message in messages),
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )

    replies = {reply["id"]: reply for reply in map(json.loads, completed.stdout.splitlines())}
    assert sorted(replies) == [1, 2, 3]
    report = replies[2]["result"]["structuredContent"]["report"]
    assert report["platform"] == "Kuaishou Kling AI"
    assert "kling_sei" in report["metadata_markers"]
    assert replies[3]["result"]["isError"] is False
    assert KLING_SEI_TAG not in output.read_bytes()


def test_the_official_sdk_client_connects_and_calls_a_tool() -> None:
    """The reference client probes server/discover first, then falls back to initialize."""
    anyio = pytest.importorskip("anyio")
    mcp = pytest.importorskip("mcp")

    async def session() -> tuple[list[str], dict]:
        params = mcp.StdioServerParameters(command=SERVER[0], args=SERVER[1:], cwd=str(ROOT))
        async with mcp.Client(params) as client:
            tools = await client.list_tools()
            result = await client.call_tool("identify", {"path": str(KLING_MP4), "check_visible": False})
            return [tool.name for tool in tools.tools], result.structured_content

    names, structured = anyio.run(session)

    legacy = [tool["name"] for tool in mcp_server.TOOLS]
    assert names[: len(legacy)] == legacy
    assert set(names[len(legacy) :]) == set(mcp_server._operations())
    assert structured["report"]["platform"] == "Kuaishou Kling AI"


def test_registry_manifest_matches_the_package_and_the_readme_marker() -> None:
    """The registry launches `uvx <identifier>==<version> <packageArguments>` and verifies
    ownership by the README's `mcp-name:` marker, so each of those must agree."""
    from click.testing import CliRunner

    from remove_ai_watermarks import __version__
    from remove_ai_watermarks.cli import main

    manifest = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    (package,) = manifest["packages"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert manifest["version"] == package["version"] == __version__
    assert package["identifier"] == "remove-ai-watermarks"
    assert re.search(rf"mcp-name: {re.escape(manifest['name'])}(?=\s|-->|<)", readme)
    args = [argument["value"] for argument in package["packageArguments"]]
    assert CliRunner().invoke(main, [*args, "--help"]).exit_code == 0


def test_video_message_tools_preserve_known_reference_and_nonzero_reports():
    runner = _Recorder(returncode=2, stdout='{"skipped":true,"output":null}')
    result = mcp_server.call_tool("video_message_remove", {"path": "clip.mp4", "reference": "a5" * 32}, runner)
    assert runner.calls[0][:3] == ["video", "message", "remove"]
    assert runner.calls[0][-2:] == ["--", "clip.mp4"]
    assert runner.calls[0][runner.calls[0].index("--reference") + 1] == "a5" * 32
    assert result["structuredContent"]["output"] is None
    assert result["structuredContent"]["report"]["skipped"] is True
    assert result["isError"] is False


@pytest.mark.parametrize(
    "options",
    [
        {"noise_std": float("nan"), "method": "vae"},
        {"noise_std": float("inf"), "method": "vae"},
        {"noise_std": True, "method": "vae"},
        {"noise_std": -0.1, "method": "vae"},
        {"noise_std": 1.1, "method": "vae"},
        {"noise_std": 10**400, "method": "vae"},
        {"noise_std": 0.1},
        {"verify": False},
        {"long_side": 7},
        {"output": "clip.mp4"},
    ],
)
def test_video_message_invalid_options_never_launch_a_child(options):
    runner = _Recorder()
    result = mcp_server.call_tool(
        "video_message_remove", {"path": "clip.mp4", "reference": "a5" * 32, **options}, runner
    )
    assert result["isError"] is True
    assert not runner.calls


def test_legacy_audio_and_video_message_tools_stay_in_lockstep_with_cli_catalog_and_stdlib_only():
    from remove_ai_watermarks import audio, integration, video_messages

    assert tuple(audio.AUDIO_SCHEMES) == (
        "audioseal",
        "wavmark",
        "perth",
        "silentcipher",
        "silentcipher16",
        "all",
        "auto",
    )
    assert tuple(audio.AUDIO_METHODS) == (
        "auto",
        "spectral",
        "encodec",
        "combined",
        "bigvgan",
        "mimi",
        "focalcodec",
        "dac",
        "snac",
    )
    assert tuple(video_messages.VIDEO_MESSAGE_SCHEMES) == ("videoseal", "pixelseal", "chunkyseal")
    assert tuple(video_messages.VIDEO_MESSAGE_METHODS) == ("auto", "codec", "vae")

    catalog = {op["id"]: op for op in integration.operation_catalog()}
    pairs = [
        ("audio_identify", "cli_audio_identify", [("scheme", "scheme")]),
        (
            "audio_invisible",
            "cli_audio_invisible",
            [
                ("scheme", "scheme"),
                ("method", "method"),
                ("vocoder_device", "vocoder_device"),
                ("bandwidth", "bandwidth"),
            ],
        ),
        (
            "video_audio",
            "cli_video_audio",
            [
                ("scheme", "scheme"),
                ("method", "method"),
                ("vocoder_device", "vocoder_device"),
                ("bandwidth", "bandwidth"),
            ],
        ),
        ("video_message_identify", "cli_video_message_identify", [("scheme", "scheme")]),
        (
            "video_message_remove",
            "cli_video_message_remove",
            [("scheme", "scheme"), ("method", "method"), ("device", "device"), ("long_side", "long_side")],
        ),
    ]
    legacy_tools = {tool["name"]: tool["inputSchema"]["properties"] for tool in mcp_server.TOOLS}
    for legacy_name, catalog_id, param_pairs in pairs:
        legacy_props = legacy_tools[legacy_name]
        catalog_params = {param["name"]: param["schema"] for param in catalog[catalog_id]["parameters"]}
        for legacy_key, catalog_key in param_pairs:
            l_spec = legacy_props[legacy_key]
            c_spec = catalog_params[catalog_key]
            if "enum" in l_spec or "enum" in c_spec:
                assert [str(v) for v in l_spec["enum"]] == [str(v) for v in c_spec["enum"]], (legacy_name, legacy_key)
            if "default" in l_spec and "default" in c_spec:
                assert str(l_spec["default"]) == str(c_spec["default"]), (legacy_name, legacy_key)

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "for mod in ('numpy', 'scipy', 'torch', 'click', 'cv2', 'PIL', 'av', 'huggingface_hub'):\n"
            "    sys.modules[mod] = None\n"
            "import remove_ai_watermarks.mcp_server as m\n"
            "assert len(m.TOOLS) >= 8\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr


def test_mcp_convenience_tools_route_audio_and_video_audio_all_streams() -> None:
    runner = _Recorder(stdout='{"watermark_status": "not_detected"}')
    res_id = mcp_server.call_tool("identify", {"path": "song.mp3"}, runner)
    assert res_id["isError"] is False
    assert runner.calls[-1] == ["audio", "identify", "--json", "--", "song.mp3"]
    _assert_the_cli_accepts(runner.calls[-1])

    res_rem_meta = mcp_server.call_tool("remove", {"path": "song.mp3", "mode": "metadata"}, runner)
    assert res_rem_meta["isError"] is False
    assert runner.calls[-1] == ["audio", "metadata", "--remove", "-o", "song_clean.mp3", "--", "song.mp3"]
    _assert_the_cli_accepts(runner.calls[-1])

    res_rem_all = mcp_server.call_tool("remove", {"path": "song.m4a", "mode": "all"}, runner)
    assert res_rem_all["isError"] is False
    assert runner.calls[-1] == ["audio", "all", "--invisible", "-o", "song_clean.m4a", "--", "song.m4a"]
    _assert_the_cli_accepts(runner.calls[-1])

    res_rem_vis = mcp_server.call_tool("remove", {"path": "song.mp3", "mode": "visible"}, runner)
    assert res_rem_vis["isError"] is True

    res_batch = mcp_server.call_tool(
        "batch", {"directory": "tracks", "media": "audio", "mode": "all", "output_dir": "tracks_out"}, runner
    )
    assert res_batch["isError"] is False
    assert runner.calls[-1] == ["audio", "batch", "--mode", "all", "--invisible", "-o", "tracks_out", "--", "tracks"]
    _assert_the_cli_accepts(runner.calls[-1])

    res_va_all = mcp_server.call_tool(
        "video_audio",
        {"path": "clip.mov", "all_streams": True, "scheme": "auto"},
        runner,
    )
    assert res_va_all["isError"] is False
    assert "--all-streams" in runner.calls[-1]
    assert runner.calls[-1][runner.calls[-1].index("--scheme") + 1] == "auto"
    _assert_the_cli_accepts(runner.calls[-1])

    # Split audio permissions and check_audio identify flag
    res_id_vid_audio = mcp_server.call_tool("identify", {"path": "clip.mp4", "check_audio": True}, runner)
    assert res_id_vid_audio["isError"] is False
    assert "--audio" in runner.calls[-1]
    _assert_the_cli_accepts(runner.calls[-1])

    res_id_audio_skip = mcp_server.call_tool("identify", {"path": "song.wav", "check_audio": False}, runner)
    assert res_id_audio_skip["isError"] is False
    assert "--no-invisible" in runner.calls[-1]
    _assert_the_cli_accepts(runner.calls[-1])

    res_dac_perm = mcp_server.call_tool(
        "audio_invisible",
        {"path": "veo.wav", "method": "dac", "allow_unverified": True},
        runner,
    )
    assert res_dac_perm["isError"] is False
    assert "--allow-unverified" in runner.calls[-1]
    assert "--force" not in runner.calls[-1]
    _assert_the_cli_accepts(runner.calls[-1])

    # Wave 2/3 parity: SNAC method, top-level presets on remove/batch, and multimodal batch
    res_snac = mcp_server.call_tool(
        "audio_invisible",
        {"path": "track.wav", "method": "snac", "allow_unverified": True},
        runner,
    )
    assert res_snac["isError"] is False
    assert runner.calls[-1][runner.calls[-1].index("--method") + 1] == "snac"
    _assert_the_cli_accepts(runner.calls[-1])

    res_preset_rm = mcp_server.call_tool(
        "remove",
        {"path": "clip.mp4", "preset": "deep-clean", "output": "clip_clean.mp4"},
        runner,
    )
    assert res_preset_rm["isError"] is False
    assert runner.calls[-1] == ["all", "--preset", "deep-clean", "-o", "clip_clean.mp4", "--", "clip.mp4"]
    _assert_the_cli_accepts(runner.calls[-1])

    res_preset_batch = mcp_server.call_tool(
        "batch",
        {"directory": "mixed", "media": "all", "mode": "all", "preset": "balanced", "output_dir": "mixed_out"},
        runner,
    )
    assert res_preset_batch["isError"] is False
    assert runner.calls[-1] == [
        "batch",
        "--mode",
        "all",
        "--preset",
        "balanced",
        "-o",
        "mixed_out",
        "--",
        "mixed",
    ]
    _assert_the_cli_accepts(runner.calls[-1])

    res_bad_preset_batch = mcp_server.call_tool(
        "batch",
        {"directory": "tracks", "media": "audio", "mode": "all", "preset": "balanced", "output_dir": "tracks_out"},
        runner,
    )
    assert res_bad_preset_batch["isError"] is True
