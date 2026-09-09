"""Agent-skill packaging stays valid for marketplace discovery."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "remove-ai-watermarks" / "SKILL.md"
PLUGIN = ROOT / "skills" / ".claude-plugin" / "plugin.json"
MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"
AGENT_DOC = ROOT / "docs" / "agent-skill.md"
INSTALL_DOC = ROOT / "docs" / "installation.md"
INSTALL_REFERENCE = ROOT / "skills" / "remove-ai-watermarks" / "references" / "install.md"

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _frontmatter(text: str) -> str:
    match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.S)
    assert match is not None, "SKILL.md must start with YAML frontmatter"
    return match.group(1)


def _scalar(frontmatter: str, key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*(.*)$", frontmatter)
    assert match is not None, f"missing frontmatter field {key}"
    value = match.group(1).strip()
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    return value


def test_skill_frontmatter_matches_agent_skills_spec() -> None:
    text = SKILL.read_text(encoding="utf-8")
    frontmatter = _frontmatter(text)
    name = _scalar(frontmatter, "name")
    description = _scalar(frontmatter, "description")

    assert name == SKILL.parent.name
    assert _NAME_RE.fullmatch(name)
    assert 1 <= len(name) <= 64
    assert 1 <= len(description) <= 1024
    raw_description = re.search(r"(?m)^description:\s*(.*)$", frontmatter)
    assert raw_description is not None
    raw_value = raw_description.group(1)
    if ": " in raw_value:
        assert raw_value[:1] in {"'", '"'}, "colons in description must be quoted"
    assert "stock-agency" in description
    assert "remove-ai-watermarks" in description


def test_plugin_and_marketplace_names_agree() -> None:
    plugin = json.loads(PLUGIN.read_text(encoding="utf-8"))
    marketplace = json.loads(MARKETPLACE.read_text(encoding="utf-8"))
    entries = marketplace["plugins"]

    assert plugin["name"] == "remove-ai-watermarks"
    assert marketplace["name"] == "remove-ai-watermarks"
    assert len(entries) == 1
    assert entries[0]["name"] == plugin["name"]
    assert entries[0]["source"] == "./skills"
    assert plugin["skills"] == "./"
    assert plugin["version"] == marketplace["metadata"]["version"]
    assert plugin["license"] == "Apache-2.0"


def test_skill_version_agrees_with_plugin_and_marketplace() -> None:
    text = SKILL.read_text(encoding="utf-8")
    frontmatter = _frontmatter(text)
    metadata_version = re.search(r"(?m)^  version:\s*(.*)$", frontmatter)
    assert metadata_version is not None, "frontmatter metadata.version is missing"
    skill_version = metadata_version.group(1).strip().strip("'\"")
    plugin = json.loads(PLUGIN.read_text(encoding="utf-8"))
    marketplace = json.loads(MARKETPLACE.read_text(encoding="utf-8"))
    assert skill_version == plugin["version"] == marketplace["metadata"]["version"]


def test_skill_stays_under_the_500_line_budget() -> None:
    lines = SKILL.read_text(encoding="utf-8").count("\n")
    assert lines <= 500, f"SKILL.md is {lines} lines; move detail into references/"


def test_probe_min_cli_version_never_exceeds_the_package_version() -> None:
    import tomllib

    sys.path.insert(0, str(ROOT / "skills" / "remove-ai-watermarks" / "scripts"))
    try:
        import probe

        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        released = tuple(int(part) for part in pyproject["project"]["version"].split(".")[:3])
        assert released >= probe.MIN_CLI_VERSION, (
            "probe.py MIN_CLI_VERSION demands a CLI newer than this package version; "
            "raise the package version or lower MIN_CLI_VERSION"
        )
    finally:
        sys.path.remove(str(ROOT / "skills" / "remove-ai-watermarks" / "scripts"))
        sys.modules.pop("probe", None)


@pytest.mark.parametrize(
    "relative",
    [
        "skills/remove-ai-watermarks/references/commands.md",
        "skills/remove-ai-watermarks/references/install.md",
        "skills/remove-ai-watermarks/scripts/probe.py",
        "docs/agent-skill.md",
    ],
)
def test_skill_reference_files_exist(relative: str) -> None:
    assert (ROOT / relative).is_file()


def test_probe_script_prints_json_plan(capsys: pytest.CaptureFixture[str]) -> None:
    """Driven in process against a machine with no CLI at all: spawning the real
    interpreter and then the real installed CLI cost 3.7s and made the assertion depend
    on whatever build happened to be on this machine's PATH."""
    probe = _probe_module()
    probe._which = lambda _name: None

    assert probe.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) >= {
        "advice",
        "cli",
        "cuda",
        "ffmpeg",
        "installers",
        "preferred_installer",
        "probe_python",
    }
    assert "invisible_images" in report["advice"] or "next" in report["advice"]


# ── Skill text against the live CLI ──
# The packaging tests above pin the frontmatter and the marketplace files, which is
# what a catalog validates. They say nothing about the CONTENT, so mark keys, flag
# choices, extras, and exit codes drifted from the CLI with a green suite. These
# tests read the skill's own prose and compare it to the click application, so a
# renamed mark or a dropped choice fails here instead of in a user's session.
COMMANDS = ROOT / "skills" / "remove-ai-watermarks" / "references" / "commands.md"
INSTALL = ROOT / "skills" / "remove-ai-watermarks" / "references" / "install.md"
PYPROJECT = ROOT / "pyproject.toml"

_PINNED_ABSENT_FLAGS = ("--model", "--steps", "--guidance-scale")
_DEVICE_COMMANDS = {("video", "invisible"), ("video", "all"), ("video", "batch")}


def _cli_commands() -> dict[tuple[str, ...], object]:
    """Every leaf command of the installed CLI, keyed by its invocation path."""
    import click

    from remove_ai_watermarks.cli import main

    found: dict[tuple[str, ...], object] = {}

    def walk(group: click.Group, path: tuple[str, ...] = ()) -> None:
        for name, command in group.commands.items():
            here = (*path, name)
            if isinstance(command, click.Group):
                walk(command, here)
            else:
                found[here] = command

    walk(main)
    return found


def _long_options(command: object) -> set[str]:
    return {opt for param in command.params for opt in param.opts if opt.startswith("--")}


def _choices(command: object, option: str) -> set[str]:
    for param in command.params:
        if option in param.opts:
            return set(param.type.choices)
    raise AssertionError(f"{option} is not an option of {command.name}")


def _documented(text: str, lead: str) -> set[str]:
    """Backticked tokens in the sentence that follows ``lead``, up to its period."""
    match = re.search(rf"{re.escape(lead)}(.*?)\.", text, flags=re.S)
    assert match is not None, f"skill text no longer states {lead!r}"
    return set(re.findall(r"`([^`]+)`", match.group(1)))


def test_documented_image_mark_keys_match_the_cli() -> None:
    documented = _documented(COMMANDS.read_text(encoding="utf-8"), "Image mark keys:")
    assert documented == _choices(_cli_commands()[("visible",)], "--mark") - {"auto"}


def test_documented_video_mark_keys_match_the_cli() -> None:
    documented = _documented(COMMANDS.read_text(encoding="utf-8"), "Video mark keys:")
    assert documented == _choices(_cli_commands()[("video", "visible")], "--mark") - {"auto"}


def test_documented_erase_backends_match_the_cli() -> None:
    documented = _documented(COMMANDS.read_text(encoding="utf-8"), "Backends:")
    assert documented == _choices(_cli_commands()[("erase",)], "--backend")


def test_documented_batch_modes_match_the_cli() -> None:
    documented = _documented(COMMANDS.read_text(encoding="utf-8"), "`batch` modes:")
    assert documented == _choices(_cli_commands()[("batch",)], "--mode")


def test_flags_the_skill_calls_absent_exist_on_no_command() -> None:
    for path, command in _cli_commands().items():
        present = _long_options(command).intersection(_PINNED_ABSENT_FLAGS)
        assert not present, f"{' '.join(path)} grew {sorted(present)}, which the skill forbids"


def test_device_exists_exactly_where_the_skill_says_it_does() -> None:
    with_device = {path for path, command in _cli_commands().items() if "--device" in _long_options(command)}
    assert with_device == _DEVICE_COMMANDS


def test_documented_exit_codes_match_the_constants() -> None:
    from remove_ai_watermarks.cli import EXIT_NO_INVISIBLE_SIGNAL, EXIT_NO_VISIBLE_MARK

    assert EXIT_NO_VISIBLE_MARK == EXIT_NO_INVISIBLE_SIGNAL
    table = re.search(r"## Exit codes(.*)", COMMANDS.read_text(encoding="utf-8"), flags=re.S)
    assert table is not None
    no_signal = re.search(r"\|\s*(\d+)\s*\|\s*No targeted signal", table.group(1))
    assert no_signal is not None, "the exit-code table no longer has a no-signal row"
    assert int(no_signal.group(1)) == EXIT_NO_VISIBLE_MARK


def test_extras_named_in_the_skill_exist_in_pyproject() -> None:
    import tomllib

    declared = set(tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["optional-dependencies"])
    rows = re.findall(r"(?m)^\|[^|\n]*\|([^|\n]*)\|\s*$", INSTALL.read_text(encoding="utf-8"))
    named = {
        part.strip()
        for row in rows
        for token in re.findall(r"`([^`]+)`", row)
        for part in token.split(",")
        if part.strip() and part.strip() != "Extra"
    }
    assert named, "the extra table no longer names any extra"
    assert named <= declared, f"skill names undeclared extras: {sorted(named - declared)}"


# ── A found CLI is not a working CLI ──
# The probe called any binary on PATH "ok", and the skill installed only when the
# binary was absent. Homebrew ships the default package, which has no pixel stack,
# so the advertised install path produced a CLI whose first `visible` died on a
# missing cv2 and whose `identify` silently dropped the visible detectors. These
# tests pin the three halves of that fix: a readable error, an honest report, and a
# probe that measures capability instead of presence.
def _probe_module() -> Any:
    import importlib.util

    path = ROOT / "skills" / "remove-ai-watermarks" / "scripts" / "probe.py"
    spec = importlib.util.spec_from_file_location("raiw_skill_probe", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "args",
    [
        ["visible", "SOURCE", "-o", "OUT"],
        ["erase", "SOURCE", "--region", "0,0,8,8", "-o", "OUT"],
        ["all", "SOURCE", "-o", "OUT"],
        ["batch", "DIR", "--mode", "visible", "-o", "OUT"],
        # The video commands reach the same broken build through their own guard, which
        # names the `video` extra rather than `visible`. Both are install hints and
        # neither may be a traceback, so they belong in one test with one rule.
        ["video", "visible", "CLIP", "-o", "OUT"],
        ["video", "all", "CLIP", "-o", "OUT"],
        ["video", "batch", "DIR", "-o", "OUT"],
    ],
    ids=["visible", "erase", "all", "batch", "video-visible", "video-all", "video-batch"],
)
def test_missing_pixel_stack_names_the_extra_instead_of_a_traceback(
    args: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    from remove_ai_watermarks._internal.watermark_profiles import VISIBLE_EXTRA
    from remove_ai_watermarks.cli import main

    source = tmp_path / "image.png"
    source.write_bytes((ROOT / "data" / "fixtures" / "provenance" / "chatgpt-1.png").read_bytes())
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    substitutions = {
        "SOURCE": str(source),
        "CLIP": str(clip),
        "DIR": str(tmp_path),
        "OUT": str(tmp_path / "out.mp4"),
    }
    resolved = [substitutions.get(arg, arg) for arg in args]

    monkeypatch.setitem(sys.modules, "cv2", None)
    result = CliRunner().invoke(main, resolved, catch_exceptions=True)

    assert result.exit_code == 1
    hints = {VISIBLE_EXTRA, "'remove-ai-watermarks[video]'"}
    assert any(hint in result.output for hint in hints), result.output
    assert not isinstance(result.exception, ImportError), "the traceback reached the user again"


def test_identify_says_when_the_visible_detectors_could_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    from remove_ai_watermarks.cli import main

    source = tmp_path / "image.png"
    source.write_bytes((ROOT / "data" / "calibration" / "gemini" / "gemini_black_2048.png").read_bytes())

    with_pixels = CliRunner().invoke(main, ["identify", str(source), "--json"], catch_exceptions=False)
    assert with_pixels.exit_code == 0
    seen = json.loads(with_pixels.output[with_pixels.output.index("{") :])
    assert not any("detectors did NOT run" in caveat for caveat in seen["caveats"])

    monkeypatch.setitem(sys.modules, "cv2", None)
    without = CliRunner().invoke(main, ["identify", str(source), "--json"], catch_exceptions=False)
    blind = json.loads(without.output[without.output.index("{") :])

    # Compared as SETS, not against a label. Pinning the wording made this fail the day
    # "Visible Gemini sparkle" became "Google Gemini visible watermark (sparkle; ...)",
    # which is a rename and not the regression this test is here for. What must hold is
    # that a blind scan loses findings and SAYS so.
    lost = set(seen["watermarks"]) - set(blind["watermarks"])
    assert lost, "the visible detector ran anyway, so this test proves nothing"
    assert any("detectors did NOT run" in caveat for caveat in blind["caveats"]), (
        f"a scan that could not see {sorted(lost)} reported like a scan that saw nothing"
    )


def _probe_with_fake_cli(
    probe: Any,
    responses: dict[str, tuple[int, str, str]],
    seen: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Drive `build_report` against a scripted CLI, keyed by subcommand."""

    def fake_run(argv: list[str], timeout: float = 8.0) -> tuple[int, str, str]:
        if seen is not None:
            seen.append(argv)
        return responses[argv[1]]

    probe._run = fake_run
    probe._which = lambda name: "/usr/local/bin/raiw" if name == "remove-ai-watermarks" else None
    return probe.build_report()


def _current_version_reply(probe: Any) -> tuple[int, str, str]:
    """A --version reply at the probe's own floor, so raising the floor cannot make a
    fixture quietly describe an outdated CLI and change what the advice means."""
    return (0, "remove-ai-watermarks, version " + ".".join(str(part) for part in probe.MIN_CLI_VERSION), "")


_NO_PIXELS_REPLY = (1, "", "Error: the visible-mark dependencies are not installed.")
_NO_MARK_REPLY = (2, "  No known visible watermark detected", "")


def test_probe_reads_capability_not_presence() -> None:
    probe = _probe_module()
    calls: list[list[str]] = []
    report = _probe_with_fake_cli(
        probe,
        {"--version": _current_version_reply(probe), "visible": _NO_PIXELS_REPLY, "invisible": _NO_PIXELS_REPLY},
        seen=calls,
    )

    assert report["cli"]["found"] is True
    assert report["pixel_stack"] == "missing"
    assert report["advice"]["visible"] == "needs_visible_extra"
    assert report["advice"]["identify"] == "metadata_only_no_pixels"
    assert report["advice"]["next"] == "install_visible"
    assert any(call[1:2] == ["visible"] for call in calls), "the probe never exercised the pixel path"


def test_probe_keeps_metadata_only_video_routes_without_pixels() -> None:
    probe = _probe_module()
    report = _probe_with_fake_cli(
        probe,
        {"--version": _current_version_reply(probe), "visible": _NO_PIXELS_REPLY, "invisible": _NO_PIXELS_REPLY},
    )

    assert report["advice"]["video_identify"] == "metadata_only_no_pixels"
    assert report["advice"]["video_metadata"] == "ok"
    assert report["advice"]["video_visible"] == "needs_video_extra"
    assert report["advice"]["video_invisible"] == "needs_video_and_diffusion_extras"
    assert "pixel_stack" not in report["advice"]
    assert "invisible_stack" not in report["advice"]


def test_skill_and_install_guide_name_pixel_free_video_routes() -> None:
    texts = [
        SKILL.read_text(encoding="utf-8"),
        INSTALL_REFERENCE.read_text(encoding="utf-8"),
        INSTALL_DOC.read_text(encoding="utf-8"),
    ]

    for command in (
        "`batch --mode metadata`",
        "`video metadata`",
        "`video identify --no-visible`",
        "`video batch --mode metadata`",
    ):
        assert all(command in text for text in texts)
    assert all("every `video` command" not in text for text in texts)
    assert all("the `video` commands stop" not in text for text in texts)


def test_metadata_only_routes_run_without_the_pixel_stack(tmp_path: Path, monkeypatch) -> None:
    from click.testing import CliRunner

    from remove_ai_watermarks.cli import main

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "input.png").write_bytes(
        (ROOT / "data" / "fixtures" / "provenance" / "ideogram-c2pa.png").read_bytes()
    )
    video = ROOT / "data" / "fixtures" / "visible" / "sora" / "example.mp4"
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "input.mp4").write_bytes(video.read_bytes())
    monkeypatch.setitem(sys.modules, "cv2", None)

    for args in (
        ["batch", str(image_dir), "--mode", "metadata", "-o", str(tmp_path / "clean-images")],
        ["video", "metadata", str(video), "--check"],
        ["video", "identify", str(video), "--no-visible"],
        ["video", "batch", str(video_dir), "--mode", "metadata", "-o", str(tmp_path / "clean-videos")],
    ):
        result = CliRunner().invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output


def test_openai_submission_names_the_partner_gate() -> None:
    text = " ".join(AGENT_DOC.read_text(encoding="utf-8").split())

    assert "Contact your OpenAI partner before submitting" in text
    assert "local execution" in text
    assert "arbitrary file access" in text
    assert "hardware access" in text


def test_probe_calls_a_working_install_ok() -> None:
    probe = _probe_module()
    report = _probe_with_fake_cli(
        probe,
        {"--version": _current_version_reply(probe), "visible": _NO_MARK_REPLY, "invisible": _NO_MARK_REPLY},
    )

    assert report["pixel_stack"] == "ok"
    assert report["advice"]["visible"] == "ok"
    assert "next" not in report["advice"]
    assert "upgrade_cli" not in report["advice"]


def test_probe_blank_png_is_decodable_by_the_cli(tmp_path: Path) -> None:
    """The capability probe is only meaningful if the CLI can read what it writes."""
    import cv2

    probe = _probe_module()
    source = tmp_path / "probe.png"
    probe._blank_png(source)

    assert cv2.imread(str(source)) is not None


def test_probe_minimum_version_is_a_release_that_exists() -> None:
    """Compared against `__version__`, which is the string `--version` renders and the
    probe then parses -- pyproject is the other side of a seam the probe never reads."""
    from remove_ai_watermarks import __version__

    probe = _probe_module()
    current = tuple(int(part) for part in __version__.split(".")[:3])

    assert current >= probe.MIN_CLI_VERSION, "the probe demands a version the project has not released"


def test_skill_version_agrees_across_all_three_manifests() -> None:
    """Three files carry this version, and a catalog reads a different one from each."""
    frontmatter = _frontmatter(SKILL.read_text(encoding="utf-8"))
    match = re.search(r'(?m)^\s+version:\s*"?([0-9][^"\n]*?)"?\s*$', frontmatter)
    assert match is not None, "SKILL.md frontmatter no longer carries metadata.version"
    plugin = json.loads(PLUGIN.read_text(encoding="utf-8"))
    marketplace = json.loads(MARKETPLACE.read_text(encoding="utf-8"))

    assert match.group(1) == plugin["version"] == marketplace["metadata"]["version"]


def test_a_bad_invocation_shares_the_no_signal_exit_code(tmp_path: Path) -> None:
    """Click's usage errors exit 2 as well, so the skill must not read 2 as a verdict."""
    from click.testing import CliRunner

    from remove_ai_watermarks.cli import EXIT_NO_VISIBLE_MARK, main

    result = CliRunner().invoke(main, ["visible", str(tmp_path / "absent.png")], catch_exceptions=False)

    assert result.exit_code == EXIT_NO_VISIBLE_MARK
    assert "Invalid value" in result.output
    text = SKILL.read_text(encoding="utf-8") + COMMANDS.read_text(encoding="utf-8")
    assert "Invalid value" in text, "the skill does not warn that a bad invocation also exits 2"


def test_an_unreadable_file_is_not_reported_as_a_missing_extra(tmp_path: Path) -> None:
    """Both silence the visible detectors; naming the wrong cause sends the wrong fix."""
    from remove_ai_watermarks.identify import identify

    broken = tmp_path / "broken.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

    caveats = identify(broken, check_visible=True, check_invisible=False).caveats

    assert any("could not be decoded" in caveat for caveat in caveats)
    assert not any("no pixel dependencies" in caveat for caveat in caveats)


def test_all_without_the_invisible_extra_is_reported_as_a_partial_not_a_refusal() -> None:
    """Measured: it skips the invisible stage, writes the file, and still exits 1."""
    probe = _probe_module()
    report = _probe_with_fake_cli(
        probe,
        {
            "--version": _current_version_reply(probe),
            "visible": _NO_MARK_REPLY,
            "invisible": (1, "", "Error: the invisible-removal dependencies are not installed."),
        },
    )

    assert report["invisible_stack"] == "missing"
    assert report["advice"]["image_all"] == "partial_writes_visible_and_metadata"


def test_all_with_the_invisible_extra_and_no_cuda_is_still_refused() -> None:
    """Measured: engine construction fails, nothing is written, exit 1."""
    probe = _probe_module()
    report = _probe_with_fake_cli(
        probe,
        {
            "--version": _current_version_reply(probe),
            "visible": _NO_MARK_REPLY,
            "invisible": (2, "  No invisible AI watermark detected", ""),
        },
    )

    assert report["invisible_stack"] == "installed"
    assert report["advice"]["image_all"] == "do_not_run_writes_nothing"


def test_the_skill_documents_both_all_outcomes() -> None:
    text = SKILL.read_text(encoding="utf-8")

    for state in ("partial_writes_visible_and_metadata", "do_not_run_writes_nothing"):
        assert state in text, f"the skill never tells the agent what {state} means"


def test_documented_vendor_cohorts_match_the_cli() -> None:
    """A cohort the skill does not know is a cohort no agent will ever select.

    `--vendor meta` arrived with the Meta Content Seal release and nothing failed:
    the parity tests compare what the skill DOES document, so a brand-new flag is
    invisible to them. Meta Muse Image carries no provenance at all, so that flag is
    the only route to cleaning one.
    """
    documented = _documented(COMMANDS.read_text(encoding="utf-8"), "Vendor cohorts:")
    commands = _cli_commands()

    assert documented == _choices(commands[("invisible",)], "--vendor")
    for path in (("all",), ("batch",)):
        assert _choices(commands[path], "--vendor") == documented, f"{path[0]} disagrees with invisible"


def test_documented_pipeline_profiles_match_the_cli() -> None:
    """`--pipeline` had no guard, so `chroma-zimage` and `auto` shipped while the skill
    still offered two engines. A profile the skill cannot name is one no agent selects,
    and `auto` is the routing the user most often wants."""
    documented = _documented(COMMANDS.read_text(encoding="utf-8"), "Pipeline profiles:")
    commands = _cli_commands()

    assert documented == _choices(commands[("invisible",)], "--pipeline")
    for path in (("all",), ("batch",)):
        assert _choices(commands[path], "--pipeline") == documented, f"{path[0]} disagrees with invisible"


def test_every_cli_command_is_named_somewhere_in_the_skill() -> None:
    """The flag guards only compare what the skill already documents, so a whole new
    COMMAND stayed invisible to them three times running. `classify` and
    `verify-openai-synthid` shipped while the skill listed neither, and the second one
    uploads the user's file to a third party. Close the set instead of the instances:
    every leaf command must be named, or the skill is silently out of date."""
    text = SKILL.read_text(encoding="utf-8") + COMMANDS.read_text(encoding="utf-8")
    missing = sorted(" ".join(path) for path in _cli_commands() if " ".join(path) not in text)

    assert not missing, f"the CLI grew commands the skill never mentions: {missing}"


@pytest.mark.parametrize(
    ("video", "invisible", "visible_advice", "invisible_advice"),
    [
        ("missing", "installed", "needs_video_extra", "needs_video_and_diffusion_extras"),
        ("ok", "missing", "ok", "needs_video_and_diffusion_extras"),
        ("ok", "installed", "ok", "ok_cpu_or_gpu"),
        ("unknown", "unknown", "unverified", "unverified"),
    ],
)
def test_probe_video_advice_uses_the_installed_video_runtime(
    video: str,
    invisible: str,
    visible_advice: str,
    invisible_advice: str,
) -> None:
    probe = _probe_module()
    probe._video_stacks = lambda _cli: {"pixels": video, "invisible": invisible}
    probe._which = lambda _name: "/synthetic/executable"
    probe._has_cuda = lambda: False
    probe._cli = lambda: {"found": True, "path": "/synthetic/cli", "outdated": False}
    probe._pixel_stack = lambda _cli: "ok"
    probe._invisible_stack = lambda _cli: "missing"
    report = probe.build_report()
    assert report["advice"]["video_visible"] == visible_advice
    assert report["advice"]["video_invisible"] == invisible_advice


@pytest.mark.parametrize("wrapper", [False, True])
def test_video_probe_runs_the_cli_interpreter_without_loading_models(tmp_path: Path, wrapper: bool) -> None:
    probe = _probe_module()
    cli = tmp_path / "remove-ai-watermarks"
    interpreter = (tmp_path / "venv" / "bin" / "python3").as_posix()
    cli.write_text(f"#!/bin/sh\n'''exec' '{interpreter}' \"$0\" \"$@\"\n' '''\n" if wrapper else f"#!'{interpreter}'\n")
    calls: list[list[str]] = []

    def run(argv: list[str], timeout: float = 8) -> tuple[int, str, str]:
        calls.append(argv)
        return 0, json.dumps({"pixels": "ok", "invisible": "installed"}), ""

    probe._run = run
    assert probe._video_stacks(str(cli)) == {"pixels": "ok", "invisible": "installed"}
    assert len(calls) == 1
    assert calls[0][:2] == [interpreter, "-c"]
    # Execute the exact child code against the real package's availability seams.
    # The loader is forbidden even when the optional stack is present.
    from types import SimpleNamespace
    from unittest.mock import patch

    from remove_ai_watermarks import video_invisible

    with (
        patch.dict(sys.modules, {"av": SimpleNamespace(), "diffusers": SimpleNamespace(AutoencoderKL=object)}),
        patch("remove_ai_watermarks.video._require_video_runtime"),
        patch.object(video_invisible, "is_available", return_value=True),
        patch.object(video_invisible, "load_video_vae_runtime", side_effect=AssertionError("model load")),
    ):
        exec(calls[0][2], {})  # noqa: S102 - Execute the probe's own child code at the real seam.


@pytest.mark.parametrize(("video_missing", "diffusion_missing"), [(True, False), (False, True), (False, False)])
def test_video_runtime_probe_reports_both_real_availability_seams(
    video_missing: bool,
    diffusion_missing: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    from remove_ai_watermarks import video, video_invisible

    probe = _probe_module()
    guard = RuntimeError("missing video") if video_missing else None
    with (
        patch.dict(sys.modules, {"av": SimpleNamespace(), "diffusers": SimpleNamespace(AutoencoderKL=object)}),
        patch.object(video, "_require_video_runtime", side_effect=guard) as checked_video,
        patch.object(video_invisible, "is_available", return_value=not diffusion_missing) as checked_diffusion,
        patch.object(video_invisible, "load_video_vae_runtime", side_effect=AssertionError("model load")),
    ):
        exec(probe._VIDEO_RUNTIME_CHECK, {})  # noqa: S102 - Execute repository-owned probe code.
    result = json.loads(capsys.readouterr().out)
    checked_video.assert_called_once()
    checked_diffusion.assert_called_once()
    assert result == {
        "pixels": "missing" if video_missing else "ok",
        "invisible": "missing" if diffusion_missing else "installed",
    }


def test_unrecognized_cli_launcher_is_unverified(tmp_path: Path) -> None:
    probe = _probe_module()
    cli = tmp_path / "launcher"
    cli.write_text('#!/bin/sh\nexec unrecognized-wrapper "$@"\n')
    probe._run = lambda *_a, **_kw: pytest.fail("unknown launcher must not run")
    assert probe._video_stacks(str(cli)) == {"pixels": "unknown", "invisible": "unknown"}
