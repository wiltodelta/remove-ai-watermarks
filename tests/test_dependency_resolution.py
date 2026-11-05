"""Exercise binary dependency resolution without downloading or installing code."""

from __future__ import annotations

import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
import yaml
from packaging.markers import Marker, default_environment
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]


def test_lock_does_not_mix_opencv_builds() -> None:
    packages = tomllib.loads((ROOT / "uv.lock").read_text())["package"]
    versions = {
        package["version"]
        for package in packages
        if package["name"] in {"opencv-python", "opencv-contrib-python", "opencv-python-headless"}
    }
    assert len(versions) == 1, "OpenCV distributions share cv2 and must use the same build"


# Use the shipping CI matrix rather than a second list of supported Python minors.
MATRIX = yaml.safe_load((ROOT / ".github/workflows/test.yml").read_text())["jobs"]["test"]["strategy"]["matrix"][
    "include"
]


@pytest.mark.parametrize("target", MATRIX, ids=lambda target: f"{target['os']}-py{target['python-version']}")
def test_c2pa_lock_has_a_wheel_for_every_ci_target(target: dict[str, str]) -> None:
    platform, machine = {
        "ubuntu-latest": ("linux", "x86_64"),
        "windows-latest": ("win32", "AMD64"),
        "macos-latest": ("darwin", "arm64"),
    }[target["os"]]
    version = target["python-version"]
    environment = {
        **default_environment(),
        "sys_platform": platform,
        "platform_machine": machine,
        "os_name": "nt" if platform == "win32" else "posix",
        "platform_system": {"linux": "Linux", "win32": "Windows", "darwin": "Darwin"}[platform],
        "python_version": version,
        "python_full_version": version + ".0",
    }
    packages = tomllib.loads((ROOT / "uv.lock").read_text())["package"]
    candidates = [
        package
        for package in packages
        if package["name"] == "c2pa-python"
        and any(
            Marker(marker).evaluate(environment)
            for marker in package.get("resolution-markers", ["python_version >= '0'"])
        )
    ]
    assert len(candidates) == 1, f"Expected one C2PA reader for {environment}"
    python = tuple(int(part) for part in version.split("."))
    wheel_tags = {
        tag
        for wheel in candidates[0].get("wheels", [])
        for tag in parse_wheel_filename(unquote(Path(urlsplit(wheel["url"]).path).name))[3]
    }
    platforms = {
        tag.platform
        for tag in wheel_tags
        if tag.platform == "any"
        or (
            platform == "linux"
            and tag.platform.startswith(("manylinux_", "linux_"))
            and tag.platform.endswith("_x86_64")
        )
        or (platform == "win32" and tag.platform == "win_amd64")
        or (
            platform == "darwin"
            and tag.platform.startswith("macosx_")
            and tag.platform.endswith(("_arm64", "_universal2"))
        )
    }
    tags = set(cpython_tags(python_version=python, platforms=sorted(platforms)))
    tags.update(compatible_tags(python_version=python, platforms=sorted(platforms)))
    assert wheel_tags & tags, f"C2PA {candidates[0]['version']} has no wheel for {target['os']} Python {version}"
