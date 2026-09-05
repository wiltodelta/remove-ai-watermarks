"""The skill-eval harness, tested where it must not silently rot.

The evals themselves need a model and are not part of the gate. What IS testable
without one: that the recording stand-in answers like the real CLI, that the grader
turns a trace into the verdicts it claims, and that every case is internally
consistent. A stand-in that drifts from the CLI, or a grader that passes everything,
measures nothing while looking green.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "evaluations" / "skill" / "cases.json"
STUB = ROOT / "scripts" / "skill_eval_stub.py"


def _eval_module() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("raiw_skill_eval", ROOT / "scripts" / "skill_eval.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cases() -> list[dict[str, Any]]:
    return json.loads(CASES.read_text(encoding="utf-8"))["cases"]


def _run_stub(args: list[str], tmp_path: Path, scenario: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    return subprocess.run(  # noqa: S603
        [sys.executable, str(STUB), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        # Overlaid on the real environment, not replacing it: a hand-built env with a
        # POSIX PATH and nothing else cannot even start python on Windows, which needs
        # SystemRoot to load its own DLLs.
        env={
            **os.environ,
            "RAIW_SCENARIO": str(scenario_path),
            "RAIW_TRACE": str(tmp_path / "trace.jsonl"),
        },
    )


def test_every_case_scenario_covers_the_commands_it_expects() -> None:
    """A case whose scenario has no reply for a command it demands measures nothing."""
    for case in _cases():
        scripted = set(case["scenario"])
        for wanted in case["expect"].get("calls_include", []):
            assert wanted in scripted or "_default" in scripted, f"{case['id']} expects {wanted} with no reply"


def test_every_case_scenario_is_self_consistent() -> None:
    """identify reporting a visible mark that `visible` then cannot find sends any agent
    into a retry loop, and the run measures the case rather than the skill."""
    for case in _cases():
        identify = case["scenario"].get("identify", {}).get("stdout", "")
        visible_exit = case["scenario"].get("visible", {}).get("exit")
        if "Visible " in identify and visible_exit == 2:
            pytest.fail(f"{case['id']}: identify names a visible mark but visible reports none")


def _probe_module() -> Any:
    import importlib.util

    path = ROOT / "skills" / "remove-ai-watermarks" / "scripts" / "probe.py"
    spec = importlib.util.spec_from_file_location("raiw_skill_probe_evals", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_probe_names_the_file_the_grader_keys_on() -> None:
    """The grader separates probe-originated calls by that filename.

    Read from the module, not from its source text: pinning the literal line failed
    the day the two capability checks were merged into one helper, which broke nothing.
    """
    assert _probe_module().PROBE_IMAGE_NAME == "probe.png"


def test_stub_records_what_the_agent_chose(tmp_path: Path) -> None:
    scenario = {"visible": {"exit": 0, "stdout": "  Removed", "writes_output": True}}
    source = tmp_path / "in.png"
    source.write_bytes(b"pixels")

    result = _run_stub(
        ["visible", str(source), "-o", str(tmp_path / "out.png"), "--mark", "gemini"], tmp_path, scenario
    )

    assert result.returncode == 0
    record = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip())
    assert record["subcommand"] == "visible"
    assert record["output"] == str(tmp_path / "out.png")
    assert record["from_probe"] is False
    assert "--mark" in record["argv"]


def test_stub_output_differs_from_its_input(tmp_path: Path) -> None:
    """A removal that returns the source byte for byte reads as a broken CLI."""
    scenario = {"visible": {"exit": 0, "stdout": "  Removed", "writes_output": True}}
    source = tmp_path / "in.png"
    source.write_bytes(b"pixels")

    _run_stub(["visible", str(source), "-o", str(tmp_path / "out.png")], tmp_path, scenario)

    assert (tmp_path / "out.png").read_bytes() != source.read_bytes()


def test_stub_group_set_matches_the_real_cli() -> None:
    """The stub must stay import-free (it is copied onto PATH as the CLI), so its group
    list is a literal. Add a second group and every call under it is recorded as the
    group name alone, the scenario falls through to `_default`, and the grader reports
    a skill failure that is really a harness failure."""
    import importlib.util

    import click

    from remove_ai_watermarks.cli import main

    spec = importlib.util.spec_from_file_location("raiw_skill_eval_stub", STUB)
    assert spec is not None
    assert spec.loader is not None
    stub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub)

    groups = {name for name, command in main.commands.items() if isinstance(command, click.Group)}
    assert groups == stub._GROUPS


def test_stub_groups_video_subcommands_but_not_source_arguments(tmp_path: Path) -> None:
    scenario = {"video invisible": {"exit": 0, "stdout": "ok"}, "identify": {"exit": 0, "stdout": "ok"}}
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"mp4")

    _run_stub(["video", "invisible", str(source)], tmp_path, scenario)
    _run_stub(["identify", str(source)], tmp_path, scenario)

    recorded = [json.loads(line)["subcommand"] for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert recorded == ["video invisible", "identify"]


def test_stub_exit_codes_match_the_real_cli_for_a_missing_pixel_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scenario the no-pixel case scripts must be what the CLI really does."""
    case = next(c for c in _cases() if c["id"] == "no_pixel_stack_ru")
    scripted = case["scenario"]["visible"]

    from click.testing import CliRunner

    from remove_ai_watermarks.cli import main

    source = tmp_path / "image.png"
    source.write_bytes((ROOT / "data" / "fixtures" / "provenance" / "chatgpt-1.png").read_bytes())
    monkeypatch.setitem(sys.modules, "cv2", None)
    real = CliRunner().invoke(main, ["visible", str(source), "-o", str(tmp_path / "out.png")])

    assert real.exit_code == scripted["exit"]
    assert "visible-mark dependencies are not installed" in real.output
    assert "visible-mark dependencies are not installed" in scripted["stdout"]


def test_grader_fails_a_fabricated_output_file(tmp_path: Path) -> None:
    """The measured failure: a finished-looking report naming a file nobody wrote."""
    module = _eval_module()
    case = {"expect": {"skill_used": False}}
    transcript = {"bash": [], "skills": [], "answer": "Готово! Очищенный файл: my_image_cleaned.png"}

    checks = module.grade(case, transcript, [], {"gemini"}, tmp_path)

    assert checks["claimed_files_exist"] is False
    (tmp_path / "my_image_cleaned.png").write_bytes(b"x")
    assert module.grade(case, transcript, [], {"gemini"}, tmp_path)["claimed_files_exist"] is True


def test_grader_ignores_probe_originated_calls(tmp_path: Path) -> None:
    """The probe runs `visible` itself; crediting that as the agent's choice would pass
    a run in which the agent never removed anything."""
    module = _eval_module()
    case = {"expect": {"skill_used": False, "calls_include": ["visible"]}}
    trace = [{"subcommand": "visible", "argv": ["visible"], "output": None, "from_probe": True}]

    checks = module.grade(case, {"bash": [], "skills": [], "answer": ""}, trace, set(), tmp_path)

    assert checks["called:visible"] is False


def test_grader_flags_a_metadata_strip_that_overwrites_the_source(tmp_path: Path) -> None:
    module = _eval_module()
    case = {"expect": {"skill_used": False, "removal_calls_carry_output": True}}
    trace = [{"subcommand": "metadata", "argv": ["metadata", "a.png", "--remove"], "output": None, "from_probe": False}]

    checks = module.grade(case, {"bash": [], "skills": [], "answer": ""}, trace, set(), tmp_path)

    assert checks["source_never_overwritten"] is False
    trace[0]["argv"] = ["metadata", "a.png", "--check"]
    assert module.grade(case, {"bash": [], "skills": [], "answer": ""}, trace, set(), tmp_path)[
        "source_never_overwritten"
    ]


def test_grader_rejects_an_invented_mark_key(tmp_path: Path) -> None:
    """Measured on haiku: it reached for `--mark gemini-sparkle`, which does not exist."""
    module = _eval_module()
    case = {"expect": {"skill_used": False, "mark_keys_are_real": True}}
    trace = [
        {
            "subcommand": "visible",
            "argv": ["visible", "a.png", "--mark", "gemini-sparkle"],
            "output": "b.png",
            "from_probe": False,
        }
    ]

    checks = module.grade(case, {"bash": [], "skills": [], "answer": ""}, trace, {"gemini", "auto"}, tmp_path)

    assert checks["mark_keys_real"] is False


def test_parse_stream_reads_tool_calls_and_the_final_answer() -> None:
    module = _eval_module()
    stream = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "tool_use", "name": "Skill", "input": {"skill": "remove-ai-watermarks"}},
                            {"type": "tool_use", "name": "Bash", "input": {"command": "python3 probe.py"}},
                        ]
                    },
                }
            ),
            "not json at all",
            json.dumps({"type": "result", "result": "done"}),
        ]
    )

    parsed = module.parse_stream(stream)

    assert parsed["skills"] == ["remove-ai-watermarks"]
    assert parsed["bash"] == ["python3 probe.py"]
    assert parsed["answer"] == "done"


def test_the_sandbox_cli_copy_does_not_announce_itself(tmp_path: Path) -> None:
    """A model that recognizes the harness stops behaving like a user's agent."""
    module = _eval_module()
    module._sandbox_cli(tmp_path)
    text = (tmp_path / "remove-ai-watermarks").read_text(encoding="utf-8")

    assert not re.search(r"eval|stand-in|fake|mock", text, flags=re.IGNORECASE)
    if os.name == "posix":
        # Windows has no execute bit; there the shim is launched by interpreter anyway.
        assert (tmp_path / "remove-ai-watermarks").stat().st_mode & 0o111


def test_probe_markers_still_match_what_the_cli_prints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe classifies a machine by two strings the CLI prints. Reword either
    message and the probe silently reports `unknown` for a state it used to name."""
    from click.testing import CliRunner

    import remove_ai_watermarks.invisible_engine as invisible_engine
    from remove_ai_watermarks.cli import main

    probe = _probe_module()

    source = tmp_path / "image.png"
    source.write_bytes((ROOT / "data" / "fixtures" / "provenance" / "chatgpt-1.png").read_bytes())

    monkeypatch.setattr(invisible_engine, "is_available", lambda: False)
    no_invisible = CliRunner().invoke(main, ["invisible", str(source), "-o", str(tmp_path / "a.png")])
    assert probe._NO_INVISIBLE_MARKER in no_invisible.output

    monkeypatch.setitem(sys.modules, "cv2", None)
    no_pixels = CliRunner().invoke(main, ["visible", str(source), "-o", str(tmp_path / "b.png")])
    assert any(marker in no_pixels.output for marker in probe._NO_PIXELS_MARKERS), (
        "no marker the probe looks for appears in what the CLI prints"
    )
