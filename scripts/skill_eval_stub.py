#!/usr/bin/env python3
"""A recording stand-in for the ``remove-ai-watermarks`` CLI, for skill evals.

Every invocation is appended to ``$RAIW_TRACE`` as one JSON line, so a graded run
asks "which commands did the agent choose, in what order, with which flags" instead
of asking a model to describe its own behavior.

Responses come from the scenario at ``$RAIW_SCENARIO``: a JSON object keyed by
subcommand path (``visible``, ``video invisible``, ``--version``) holding ``exit``,
``stdout`` and optional ``writes_output``. The stand-in decides NOTHING on its own,
so a case that wants a machine without CUDA states that in its scenario rather than
relying on this file to imitate it.

The imitation is only worth as much as its fidelity to the real CLI, so
``tests/test_skill_evals.py`` drives both on the same arguments and requires the same
exit codes and markers wherever the real state is reproducible locally.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_FLAG_PREFIX = "-"
# The only command group; every other first word is already a leaf command, so the
# word after it is the SOURCE argument and must not join the command path.
_GROUPS = frozenset({"video"})


def _subcommand(argv: list[str]) -> str:
    """The command path the agent chose, ignoring arguments and flags."""
    if not argv:
        return ""
    first = argv[0]
    if first.startswith(_FLAG_PREFIX):
        return first
    if first in _GROUPS and len(argv) > 1 and not argv[1].startswith(_FLAG_PREFIX):
        return f"{first} {argv[1]}"
    return first


def _output_path(argv: list[str]) -> str | None:
    for flag in ("-o", "--output", "--output-dir"):
        if flag in argv:
            index = argv.index(flag)
            if index + 1 < len(argv):
                return argv[index + 1]
    return None


def main(argv: list[str]) -> int:
    scenario = json.loads(Path(os.environ["RAIW_SCENARIO"]).read_text(encoding="utf-8"))
    subcommand = _subcommand(argv)
    reply = scenario.get(subcommand) or scenario.get("_default") or {"exit": 1, "stdout": "unscripted command"}

    output = _output_path(argv)
    trace = os.environ.get("RAIW_TRACE")
    if trace:
        with Path(trace).open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "subcommand": subcommand,
                        "argv": argv,
                        "output": output,
                        # The published probe runs the CLI on a file it names probe.png,
                        # which is how its own calls stay separable from the ones the
                        # agent chose. The name is pinned by the suite.
                        "from_probe": any(Path(arg).name == "probe.png" for arg in argv),
                    }
                )
                + "\n"
            )

    sys.stdout.write(reply.get("stdout", "") + "\n")
    if reply.get("writes_output") and output:
        source = next((a for a in argv[1:] if not a.startswith(_FLAG_PREFIX) and Path(a).is_file()), None)
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if source:
            # A real removal changes pixels. Copying the source byte for byte made an
            # agent check md5, see the input unchanged, and spend the rest of the run
            # deciding the CLI was broken -- the harness measuring itself.
            out_path.write_bytes(Path(source).read_bytes() + b"\n")
        else:
            out_path.write_bytes(b"")
    return int(reply.get("exit", 0))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
