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
    assert [tool["name"] for tool in tools[:3]] == ["identify", "remove", "batch"]
    assert {tool["name"] for tool in tools[3:]} == set(mcp_server._operations())
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

    assert names[:3] == ["identify", "remove", "batch"]
    assert set(names[3:]) == set(mcp_server._operations())
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
