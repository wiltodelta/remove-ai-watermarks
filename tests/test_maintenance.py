"""The maintenance gate must stop when the security scanner fails."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parents[1] / "maintain.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(os.name != "posix" or BASH is None, reason="The maintenance entry point requires Bash")


@pytest.mark.parametrize(
    ("scanner_output", "scanner_status"),
    [
        ("No vulnerabilities or maintenance issues detected", 0),
        ("Vulnerabilities detected!", 2),
        ("Scanner failed before producing a verdict", 1),
        ("No vulnerabilities or maintenance issues detected\nScanner teardown failed", 1),
        ("All dependencies appear safe\nScanner teardown failed", 1),
    ],
)
def test_security_exit_status_controls_the_gate(tmp_path, scanner_output, scanner_status):
    commands = tmp_path / "commands.log"
    for name, body in {
        "uv": 'printf "uv %s\\n" "$*" >> "$GATE_TEST_COMMANDS"\n',
        "uvx": (
            'printf "uvx %s\\n" "$*" >> "$GATE_TEST_COMMANDS"\n'
            'if [ "$1" = uv-secure ]; then\n'
            '    printf "%s\\n" "$GATE_TEST_OUTPUT"\n'
            '    exit "$GATE_TEST_STATUS"\n'
            "fi\n"
        ),
    }.items():
        executable = tmp_path / name
        executable.write_text("#!/bin/sh\n" + body)
        executable.chmod(0o755)

    assert BASH is not None
    result = subprocess.run(  # noqa: S603 -- fixed repository script, isolated command stubs
        [BASH, str(GATE)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", ""),
            "GATE_TEST_COMMANDS": str(commands),
            "GATE_TEST_OUTPUT": scanner_output,
            "GATE_TEST_STATUS": str(scanner_status),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    calls = commands.read_text().splitlines()
    assert calls[:3] == ["uv sync --all-extras", "uvx uv-outdated", "uvx uv-secure uv.lock"]
    assert scanner_output in result.stdout
    if scanner_status:
        assert result.returncode != 0
        assert len(calls) == 3
    else:
        assert result.returncode == 0
        assert calls[-1] == "uv run pytest -n auto"
