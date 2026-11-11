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


def test_comfyui_waiter_outlasts_the_node_suite_then_reads_the_registry() -> None:
    script = _step("distribute.yml", "comfyui", "Sync and publish the ComfyUI nodes")["run"]
    assert "seq 1 360" in script
    assert "api.comfy.org/nodes/remove-ai-watermarks/versions" in script


def test_space_pin_is_dispatched_before_the_factory_rebuild() -> None:
    script = _step("distribute.yml", "hf-space", "Bump the Space pin, then factory-rebuild")["run"]
    assert "pin-library.yml" in script
    assert "RAIW_HF_SPACE_TOKEN secret is not set" in script
    assert "restart_space" in script
    # Nested YAML indent inside python -c crashed 0.40.1's waiter with
    # IndentationError before it could see the dispatched pin run.
    waiter = re.search(r"python3 -c \"(.*?)\"", script)
    assert waiter is not None
    assert waiter.group(1).lstrip() == waiter.group(1)


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


@pytest.mark.skipif(os.name != "posix" or shutil.which("bash") is None, reason="Executes an Ubuntu Bash workflow")
@pytest.mark.parametrize(("status", "listed"), [("200", "true"), ("404", "false"), ("503", None)])
def test_mcp_registry_lookup_skips_listed_versions_and_fails_on_other_replies(
    tmp_path: Path, status: str, listed: str | None
) -> None:
    workflow = _workflow("publish-mcp-registry.yml")
    script = _step("publish-mcp-registry.yml", "publish", "Skip a version the registry already lists")["run"]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\nprintf {status}\n')
    curl.chmod(0o755)
    env = {
        **os.environ,
        **workflow["jobs"]["publish"]["env"],
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "VERSION": "0.46.0",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "CALL_LOG": str(tmp_path / "calls"),
    }
    result = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    # The server name carries a slash, so it must reach the registry percent-encoded.
    assert "/v0/servers/io.github.wiltodelta%2Fremove-ai-watermarks/versions/0.46.0" in (tmp_path / "calls").read_text()
    if listed is None:
        assert result.returncode != 0
        return
    assert result.returncode == 0, result.stderr
    assert f"listed={listed}" in (tmp_path / "output").read_text()


@pytest.mark.skipif(os.name != "posix" or shutil.which("bash") is None, reason="Executes an Ubuntu Bash workflow")
@pytest.mark.parametrize(
    ("release", "package", "accepted"),
    [("1.0.0", "1.0.0", True), ("1.0.1", "1.0.0", False), ("1.0.0", "1.0.1", False)],
)
def test_mcp_registry_publishes_only_a_server_json_that_matches_the_release(
    tmp_path: Path, release: str, package: str, accepted: bool
) -> None:
    workflow = _workflow("publish-mcp-registry.yml")
    script = _step("publish-mcp-registry.yml", "publish", "Check server.json matches the release")["run"]
    manifest = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    manifest["version"] = "1.0.0"
    # A stale package entry under a bumped server version must not publish either.
    manifest["packages"][0]["version"] = package
    (tmp_path / "server.json").write_text(json.dumps(manifest), encoding="utf-8")
    env = {**os.environ, **workflow["jobs"]["publish"]["env"], "VERSION": release}
    result = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == accepted, result.stderr


def test_mcp_registry_publishes_on_release_with_an_oidc_token() -> None:
    workflow = _workflow("publish-mcp-registry.yml")
    assert workflow["permissions"]["id-token"] == "write"
    # PyYAML reads the bare `on:` key as the boolean True.
    assert set(workflow[True]) == {"release", "workflow_dispatch"}


@pytest.mark.skipif(os.name != "posix" or shutil.which("bash") is None, reason="Executes an Ubuntu Bash workflow")
def test_homebrew_bump_synchronizes_all_published_extra_options(tmp_path: Path) -> None:
    script = _step("distribute.yml", "homebrew", "Bump the formula")["run"]
    (tmp_path / "Formula").mkdir()
    formula = tmp_path / "Formula" / "remove-ai-watermarks.rb"
    formula.write_text('url "old"\nsha256 "old"\n')
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "sync_remove_ai_watermarks.py").write_text(
        "import argparse,json\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "for name in ('formula','version','url','sha256'): p.add_argument('--'+name,required=True)\n"
        "a=p.parse_args()\n"
        "Path('sync-call.json').write_text(json.dumps(vars(a)))\n"
        "Path(a.formula).write_text('artifact and Provides-Extra synchronized\\n')\n"
    )
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    git = executable_dir / "git"
    git.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> git-calls.txt\n')
    git.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(executable_dir) + os.pathsep + os.environ["PATH"],
        "VERSION": "0.56.0",
        "URL": "https://files.pythonhosted.org/packages/synthetic/remove_ai_watermarks-0.56.0.tar.gz",
        "SHA": "a" * 64,
    }
    result = subprocess.run(["bash", "-eu", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    invocation = tmp_path / "sync-call.json"
    assert invocation.exists(), "Homebrew bump skipped the published extras inventory"
    assert json.loads(invocation.read_text()) == {
        "formula": "Formula/remove-ai-watermarks.rb",
        "version": env["VERSION"],
        "url": env["URL"],
        "sha256": env["SHA"],
    }
    assert formula.read_text() == "artifact and Provides-Extra synchronized\n"
    assert "push" in (tmp_path / "git-calls.txt").read_text().splitlines()
