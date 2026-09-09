"""Execute release workflow scripts against local, controlled service replies."""

# These tests execute repository-owned workflow snippets against synthetic services.
# ruff: noqa: S603, S607

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _workflow(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def _step(name: str, job: str, title: str) -> dict:
    return next(s for s in _workflow(name)["jobs"][job]["steps"] if s.get("name") == title)


@pytest.mark.parametrize(
    ("dependency", "accepted"),
    [
        ("remove-ai-watermarks>=0.38.0", True),
        ("Remove_AI_Watermarks[visible]>=0.38.0,<1", True),
        ("other-package>=0.38.0", False),
        ("remove-ai-watermarks>=0.38.01", False),
        ("remove-ai-watermarks>=0.37.0", False),
        ("remove-ai-watermarks>=0.38.0,!=0.38.0", False),
    ],
)
def test_registry_check_parses_distribution_and_exact_floor(dependency: str, accepted: bool) -> None:
    script = _step("verify-release.yml", "verify", "Verify ComfyUI registry node requires this release")["run"]
    code = re.search(r'python3 -c "\n(.*?)\n"\)', script, re.S)
    assert code is not None
    result = subprocess.run(
        [sys.executable, "-c", code[1].replace('\\"', '"')],
        input=json.dumps([{"dependencies": [dependency], "version": "1", "status": "active"}]),
        text=True,
        capture_output=True,
        env={**os.environ, "VERSION": "0.38.0"},
    )
    assert (result.returncode == 0) == accepted, result.stderr


@pytest.mark.skipif(os.name != "posix" or shutil.which("bash") is None, reason="Executes an Ubuntu Bash workflow")
def test_distributed_version_survives_a_newer_pypi_release(tmp_path: Path) -> None:
    producer = _step("distribute.yml", "resolve", "Record distributed release")
    consumer = _step("verify-release.yml", "verify", "Resolve target version")
    env = {
        **os.environ,
        "VERSION": "0.37.0",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "EVENT_NAME": "workflow_run",
        "INPUT_VERSION": "",
        "DISTRIBUTION_RUN_ID": "123",
        "DISTRIBUTION_RUN_ATTEMPT": "2",
    }
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text("#!/bin/sh\ncat <<'JSON'\n" + json.dumps({"info": {"version": "0.99.0"}}) + "\nJSON\n")
    curl.chmod(0o755)
    env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
    subprocess.run(["bash", "-eu", "-c", producer["run"]], cwd=tmp_path, env=env, check=True)
    artifact = tmp_path / "distribution-identities" / "distributed-release-2"
    artifact.mkdir(parents=True)
    identity = artifact / "release-identity.json"
    identity.write_bytes((tmp_path / "release-identity.json").read_bytes())
    # A failed-job rerun has no new resolve artifact and must keep this identity.
    env["DISTRIBUTION_RUN_ATTEMPT"] = "3"
    future = tmp_path / "distribution-identities" / "distributed-release-4"
    future.mkdir()
    (future / "release-identity.json").write_text(
        json.dumps({"version": "0.99.0", "run_id": "123", "run_attempt": "4"})
    )

    result = subprocess.run(
        ["bash", "-eu", "-c", consumer["run"]], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "version=0.37.0" in (tmp_path / "output").read_text()
    uploads = [
        s for s in _workflow("distribute.yml")["jobs"]["resolve"]["steps"] if "upload-artifact@" in s.get("uses", "")
    ]
    downloads = [
        s
        for s in _workflow("verify-release.yml")["jobs"]["verify"]["steps"]
        if "download-artifact@" in s.get("uses", "")
    ]
    assert uploads
    assert downloads
    assert downloads[0]["with"]["run-id"] == "${{ github.event.workflow_run.id }}"
    assert "github.run_attempt" in uploads[0]["with"]["name"]
    assert downloads[0]["with"]["pattern"] == "distributed-release-*"
    assert _workflow("verify-release.yml")["permissions"]["actions"] == "read"
    for altered in (
        {"DISTRIBUTION_RUN_ATTEMPT": "1"},
        {"DISTRIBUTION_RUN_ID": "999"},
    ):
        assert (
            subprocess.run(
                ["bash", "-eu", "-c", consumer["run"]],
                cwd=tmp_path,
                env={**env, **altered},
                capture_output=True,
            ).returncode
            != 0
        )
    identity.write_text(json.dumps({"version": "0.38.0", "run_id": "123", "run_attempt": "3"}))
    assert (
        subprocess.run(
            ["bash", "-eu", "-c", consumer["run"]],
            cwd=tmp_path,
            env=env,
            capture_output=True,
        ).returncode
        != 0
    )
    identity.write_text(json.dumps({"version": "0.38.0", "run_id": "999", "run_attempt": "2"}))
    assert (
        subprocess.run(["bash", "-eu", "-c", consumer["run"]], cwd=tmp_path, env=env, capture_output=True).returncode
        != 0
    )


@pytest.mark.skipif(os.name != "posix" or shutil.which("bash") is None, reason="Executes an Ubuntu Bash workflow")
def test_distributed_version_reads_the_flat_single_artifact_layout(tmp_path: Path) -> None:
    producer = _step("distribute.yml", "resolve", "Record distributed release")
    consumer = _step("verify-release.yml", "verify", "Resolve target version")
    env = {
        **os.environ,
        "VERSION": "0.37.0",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "EVENT_NAME": "workflow_run",
        "INPUT_VERSION": "",
        "DISTRIBUTION_RUN_ID": "123",
        "DISTRIBUTION_RUN_ATTEMPT": "2",
    }
    subprocess.run(["bash", "-eu", "-c", producer["run"]], cwd=tmp_path, env=env, check=True)
    # download-artifact@v8 extracts the only pattern-matched artifact directly
    # into the target path, with no artifact-name directory.
    artifact = tmp_path / "distribution-identities"
    artifact.mkdir()
    (artifact / "release-identity.json").write_bytes((tmp_path / "release-identity.json").read_bytes())

    result = subprocess.run(
        ["bash", "-eu", "-c", consumer["run"]], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "version=0.37.0" in (tmp_path / "output").read_text()


@pytest.mark.skipif(os.name != "posix" or shutil.which("bash") is None, reason="Executes an Ubuntu Bash workflow")
def test_clawhub_uses_local_install(tmp_path: Path) -> None:
    script = _step("distribute.yml", "clawhub", "Publish the agent skill when its version moved")["run"]
    script = script.replace("${{ github.event.release.name || github.ref_name }}", "test")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # A fresh runner installs the executable locally; no global clawhub exists.
    npm = bin_dir / "npm"
    npm.write_text('#!/bin/sh\nmkdir -p node_modules/.bin\ncp "$CLAW_STUB" node_modules/.bin/clawhub\n')
    npm.chmod(0o755)
    stub = tmp_path / "claw-stub"
    stub.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n'
        'if [ "$1" = inspect ]; then cat skills/remove-ai-watermarks/SKILL.md; fi\n'
    )
    stub.chmod(0o755)
    skill = tmp_path / "skills/remove-ai-watermarks/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text('metadata:\n  version: "1.0.5"\n')
    env = {
        **os.environ,
        "PATH": str(bin_dir) + ":/usr/bin:/bin",
        "CLAWHUB_TOKEN": "synthetic",
        "CLAW_STUB": str(stub),
        "CALL_LOG": str(tmp_path / "calls"),
    }
    result = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "calls").read_text().splitlines() == [
        "login --token synthetic",
        "inspect remove-ai-watermarks --file SKILL.md",
    ]
