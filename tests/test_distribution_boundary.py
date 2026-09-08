"""Distribution-boundary regression tests."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _array_values(section: str, key: str) -> set[str]:
    match = re.search(rf"(?ms)^{key}\s*=\s*\[(.*?)^\]", section)
    assert match is not None
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def test_sdist_has_explicit_public_boundary() -> None:
    """Hatchling must publish only package source and required metadata."""
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    sdist_config = config.split("[tool.hatch.build.targets.sdist]", maxsplit=1)[1].split("\n[", maxsplit=1)[0]

    assert _array_values(sdist_config, "include") == {"/src", "/LICENSE", "/README.md", "/pyproject.toml"}
    assert {"/data", "/tmp", "/.sc"} <= _array_values(sdist_config, "exclude")


def test_release_guide_names_every_published_surface() -> None:
    """A release checklist must not silently forget a downstream surface."""
    guide = (ROOT / "docs" / "release-and-distribution.md").read_text(encoding="utf-8")

    for surface in ("PyPI", "Homebrew", "Hugging Face Space", "ComfyUI Registry"):
        assert surface in guide
    assert re.search(r"Conda is\s+not a supported publishing surface", guide)


def test_local_only_data_stays_ignored() -> None:
    """A fresh clone must carry the ignore rules for the local-only data trees.

    Both were covered only by `.git/info/exclude` or an untracked .gitignore
    inside the directory, neither of which a clone has, so nothing stopped a
    fresh checkout from committing them and nothing reported the gap. Read the
    entries rather than the file text: a first draft matched the raw string and
    a comment naming `data/research/README.md` satisfied it with the rule gone.
    """
    entries = {
        stripped
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    }
    assert {"data/spaces/", "data/research/"} <= entries


def test_scripts_carry_no_cloud_gpu_harness() -> None:
    """Research that needs a cloud GPU account is not kept in this repository.

    Twelve such harnesses moved out on 2026-09-07, and the import is what made
    them harnesses rather than offline analysis. The private identifiers they
    named are deliberately not listed here: a denylist written into a tracked
    file publishes the very strings it exists to keep out, so that half of the
    audit stays a read of the private rules.
    """
    offenders = [
        path.name
        for path in sorted((ROOT / "scripts").rglob("*.py"))
        if re.search(r"^\s*(?:import modal|from modal import)\b", path.read_text(encoding="utf-8"), re.MULTILINE)
    ]
    assert offenders == []
