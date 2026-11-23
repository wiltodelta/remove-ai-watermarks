"""The Codex plugin manifest, held to the checks a Codex plugin catalog runs on it."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / ".codex-plugin" / "plugin.json"
CLAUDE_PLUGIN = ROOT / "skills" / ".claude-plugin" / "plugin.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_carries_every_field_the_catalog_validator_requires() -> None:
    manifest = _manifest()
    for field in ("name", "version", "description", "repository", "license"):
        assert str(manifest.get(field, "")).strip(), field
    interface = manifest["interface"]
    for field in ("displayName", "shortDescription", "composerIcon"):
        assert str(interface.get(field, "")).strip(), field
    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", manifest["name"])
    assert re.fullmatch(r"\d+\.\d+\.\d+", manifest["version"])
    assert re.fullmatch(r"#[0-9A-Fa-f]{6}", interface["brandColor"])


def test_manifest_describes_the_same_plugin_as_the_claude_manifest() -> None:
    manifest = _manifest()
    claude = json.loads(CLAUDE_PLUGIN.read_text(encoding="utf-8"))
    for field in ("name", "version", "description", "repository", "license", "homepage"):
        assert manifest[field] == claude[field], field


def test_every_declared_path_stays_inside_the_repository() -> None:
    manifest = _manifest()
    interface = manifest["interface"]
    paths = [manifest["skills"], interface["composerIcon"], interface["logo"]]
    for declared in paths:
        assert declared.startswith("./"), declared
        resolved = (ROOT / declared).resolve()
        assert resolved.is_relative_to(ROOT), declared
        assert resolved.exists(), declared
    assert (ROOT / manifest["skills"] / "remove-ai-watermarks" / "SKILL.md").is_file()


def test_icon_is_a_small_self_contained_svg() -> None:
    icon = ROOT / _manifest()["interface"]["composerIcon"]
    content = icon.read_text(encoding="utf-8")
    assert icon.suffix == ".svg"
    assert icon.stat().st_size <= 50 * 1024
    for forbidden in ("data:image/", "base64", "TODO", "PLACEHOLDER"):
        assert forbidden not in content, forbidden


def test_no_tracked_code_matches_the_scanner_dynamic_execution_rule() -> None:
    """The catalog's scanner reports a high finding for any call spelled like
    Python's built-in evaluator in its code files, and that rule cannot tell it from
    the PyTorch module method of the same name. Its scan job is advisory, but the
    finding still lands on the listing PR, so the code switches modules to inference
    with the equivalent ``module.train(False)``."""
    tracked = subprocess.run(
        ["git", "ls-files", "*.py", "*.js", "*.mjs", "*.cjs", "*.ts", "*.tsx", "*.jsx"],  # noqa: S607
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert tracked
    rule = re.compile(r"\beval\s*\(")
    offenders = [
        name
        for name in tracked
        if (ROOT / name).is_file() and rule.search((ROOT / name).read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == []
