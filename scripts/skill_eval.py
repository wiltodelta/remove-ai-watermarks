#!/usr/bin/env python3
"""Run the agent-skill evals: does an agent reach for this skill, and then obey it.

The deterministic suite proves the skill's PROSE matches the CLI. It cannot prove the
only thing a skill is for -- that a model picks it up on a real request and follows the
checklist. This harness measures that: it builds a throwaway project with the skill
installed and a recording stand-in for the CLI on PATH, runs a headless agent on one
user prompt, and grades the recorded command trace.

Grading is mechanical wherever possible. "Did it run `all` on a machine the probe said
writes nothing" is a fact in the trace; asking a model to judge its own transcript is
not. Only the answer-text checks are textual, and they are plain substrings.

Agents are nondeterministic, so a single run is an anecdote. ``--repeat`` runs each
case N times and the report is a pass RATE per check, never one verdict.

    python3 scripts/skill_eval.py --model haiku --repeat 3
    python3 scripts/skill_eval.py --case visible_ru --model sonnet --out /tmp/skill-eval.json

Nothing here runs in maintain.sh: it spends model tokens and needs the claude CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "remove-ai-watermarks"
STUB = Path(__file__).resolve().parent / "skill_eval_stub.py"
CASES = ROOT / "data" / "evaluations" / "skill" / "cases.json"
FIXTURE = ROOT / "data" / "calibration" / "gemini" / "gemini_black_2048.png"

# A model that recognizes the harness stops behaving like a user's agent: the first
# run ended with "оказалось, что это тестовое окружение" after the agent read the
# shim's own source. The sandbox copy therefore carries a neutral header. This is
# mock hygiene, not deception of a user.
_NEUTRAL_HEADER = '#!/usr/bin/env python3\n"""Console entry point for remove-ai-watermarks."""\n'
_INSTALLER_RE = re.compile(r"\b(uv tool install|pipx install|pip install|brew install)\b")
# Language-independent hallucination check. A weak model with no pixel stack reported a
# finished removal, complete with a size table for `my_image_cleaned.png`, having never
# run a removal command. Naming a file that is not on disk is that failure, mechanically.
_CLAIMED_FILE_RE = re.compile(r"[\w./-]+\.(?:png|jpg|jpeg|webp|mp4|mov)")


def _sandbox_cli(bin_dir: Path) -> None:
    source = STUB.read_text(encoding="utf-8")
    body = source.split('"""', 2)[2].lstrip("\n")
    target = bin_dir / "remove-ai-watermarks"
    target.write_text(_NEUTRAL_HEADER + body, encoding="utf-8")
    target.chmod(0o755)


def build_sandbox(case: dict[str, Any], root: Path) -> dict[str, Path]:
    skills_dir = root / ".claude" / "skills"
    bin_dir = root / "bin"
    skills_dir.mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    shutil.copytree(SKILL, skills_dir / SKILL.name)
    _sandbox_cli(bin_dir)
    shutil.copyfile(FIXTURE, root / "my_image.png")
    scenario = root / "scenario.json"
    scenario.write_text(json.dumps(case["scenario"]), encoding="utf-8")
    return {"root": root, "bin": bin_dir, "scenario": scenario, "trace": root / "trace.jsonl"}


def run_agent(case: dict[str, Any], paths: dict[str, Path], model: str, timeout: float) -> dict[str, Any]:
    env = dict(os.environ)
    env["PATH"] = f"{paths['bin']}{os.pathsep}{env['PATH']}"
    env["RAIW_SCENARIO"] = str(paths["scenario"])
    env["RAIW_TRACE"] = str(paths["trace"])
    claude = shutil.which("claude")
    if claude is None:
        raise RuntimeError("the claude CLI is not on PATH; the behavior evals need it")
    proc = subprocess.run(  # noqa: S603
        [
            claude,
            "-p",
            case["prompt"],
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "bypassPermissions",
            "--model",
            model,
        ],
        cwd=paths["root"],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    return parse_stream(proc.stdout)


def parse_stream(stdout: str) -> dict[str, Any]:
    """Tool calls and the final answer out of the stream-json transcript."""
    bash: list[str] = []
    skills: list[str] = []
    answer = ""
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") != "tool_use":
                    continue
                if block["name"] == "Bash":
                    bash.append(str(block.get("input", {}).get("command", "")))
                elif block["name"] == "Skill":
                    skills.append(str(block.get("input", {}).get("skill", "")))
        elif event.get("type") == "result":
            answer = str(event.get("result", ""))
    return {"bash": bash, "skills": skills, "answer": answer}


def read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def grade(
    case: dict[str, Any],
    transcript: dict[str, Any],
    trace: list[dict[str, Any]],
    mark_keys: set[str],
    root: Path,
) -> dict[str, bool]:
    expect = case["expect"]
    agent_calls = [record for record in trace if not record["from_probe"]]
    chosen = [record["subcommand"] for record in agent_calls]
    answer = transcript["answer"].lower()
    checks: dict[str, bool] = {}
    if expect.get("skill_used", True):
        checks["skill_used"] = SKILL.name in transcript["skills"]
    if expect.get("probe_ran"):
        checks["probe_ran"] = any("probe.py" in command for command in transcript["bash"])
    for wanted in expect.get("calls_include", []):
        checks[f"called:{wanted}"] = wanted in chosen
    for banned in expect.get("calls_exclude", []):
        checks[f"avoided:{banned}"] = banned not in chosen
    if expect.get("removal_calls_carry_output"):
        writers = [r for r in agent_calls if r["subcommand"] in {"visible", "erase", "all", "invisible", "metadata"}]
        checks["outputs_named"] = all(r["output"] for r in writers) if writers else True
        # The rule above is the skill's; THIS is the harm it exists to prevent. Image
        # `metadata` without -o overwrites the source, so a diagnostic run destroys the
        # user's original. --check is the read-only form and is always allowed.
        strips = [
            r
            for r in agent_calls
            if r["subcommand"] == "metadata" and "--check" not in r["argv"] and "--help" not in r["argv"]
        ]
        checks["source_never_overwritten"] = all(r["output"] for r in strips)
    if expect.get("mark_keys_are_real"):
        used = {
            record["argv"][record["argv"].index("--mark") + 1]
            for record in agent_calls
            if "--mark" in record["argv"] and record["argv"].index("--mark") + 1 < len(record["argv"])
        }
        checks["mark_keys_real"] = used <= mark_keys
    if expect.get("install_attempted"):
        checks["install_attempted"] = any(_INSTALLER_RE.search(command) for command in transcript["bash"])
    wanted_any = expect.get("answer_must_contain_any")
    if wanted_any:
        checks["answer_offers_a_fix"] = any(phrase.lower() in answer for phrase in wanted_any)
    for phrase in expect.get("answer_must_not_contain", []):
        checks[f"no_phrase:{phrase}"] = phrase.lower() not in answer
    claimed = {Path(name).name for name in _CLAIMED_FILE_RE.findall(transcript["answer"])}
    checks["claimed_files_exist"] = all((root / name).exists() for name in claimed)
    return checks


def mark_keys() -> set[str]:
    """The real image mark keys, read from the CLI this skill drives."""
    from remove_ai_watermarks import watermark_registry

    return {"auto", *watermark_registry.mark_keys()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--case", action="append", default=None, help="Case id; repeatable.")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=None, help="Write per-run results as JSON.")
    args = parser.parse_args(argv)

    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    if args.case:
        cases = [case for case in cases if case["id"] in set(args.case)]
    keys = mark_keys()

    runs: list[dict[str, Any]] = []
    for case in cases:
        for attempt in range(args.repeat):
            with tempfile.TemporaryDirectory(prefix="raiw-eval-") as tmp:
                paths = build_sandbox(case, Path(tmp))
                try:
                    transcript = run_agent(case, paths, args.model, args.timeout)
                except subprocess.TimeoutExpired:
                    runs.append({"case": case["id"], "attempt": attempt, "checks": {"completed": False}})
                    print(f"{case['id']} #{attempt}: TIMEOUT", flush=True)
                    continue
                trace = read_trace(paths["trace"])
                checks = grade(case, transcript, trace, keys, paths["root"])
                runs.append(
                    {
                        "case": case["id"],
                        "attempt": attempt,
                        "model": args.model,
                        "checks": checks,
                        "calls": [r["subcommand"] for r in trace if not r["from_probe"]],
                        "answer": transcript["answer"],
                    }
                )
                failed = [name for name, ok in checks.items() if not ok]
                verdict = "PASS" if not failed else "FAIL " + ",".join(failed)
                print(f"{case['id']} #{attempt}: {verdict}  ({len(checks)} checks)", flush=True)

    print("\n== pass rate per check ==")
    tally: dict[tuple[str, str], list[bool]] = {}
    for run in runs:
        for name, ok in run["checks"].items():
            tally.setdefault((run["case"], name), []).append(ok)
    for (case_id, name), results in sorted(tally.items()):
        passed = sum(results)
        flag = "" if passed == len(results) else "   <-- "
        print(f"{case_id:24} {name:28} {passed}/{len(results)}{flag}")

    if args.out:
        args.out.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
