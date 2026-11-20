"""Add an Apple Foundation Models arm to a sealed-source comparison manifest."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from remove_ai_watermarks.text_api import _exact_sentence_slots  # noqa: E402
from scripts import benchmark_text_pairs as paired  # noqa: E402

log = logging.getLogger(__name__)


def _retained(source: str, candidate: str | None) -> list[str]:
    if candidate is None or source == candidate:
        return []
    candidate_sentences = {sentence.strip() for sentence in _exact_sentence_slots(candidate)[0]}
    return list(
        dict.fromkeys(
            value
            for sentence in _exact_sentence_slots(source)[0]
            if (value := sentence.strip()) and value in candidate_sentences
        )
    )


def _merge(manifest: dict[str, Any], arm: str, inputs: list[dict[str, str]], responses: Path) -> list[dict[str, Any]]:
    with responses.open(encoding="utf-8") as stream:
        rows = [json.loads(line, object_pairs_hook=paired._distinct_fields) for line in stream]
    if not all(isinstance(row, dict) for row in rows) or [row.get("id") for row in rows] != [x["id"] for x in inputs]:
        raise ValueError("Apple responses must match every input ID in generation order.")
    outcomes = {}
    for row in rows:
        candidate, error = row.get("candidate"), row.get("error")
        if not (
            (isinstance(candidate, str) and error is None) or (candidate is None and paired._nonempty_string(error))
        ):
            raise ValueError("Each Apple response must contain text or a refusal.")
        if not paired._nonempty_string(candidate):
            row["candidate"] = None
            row["error"] = error or "Empty model response."
        outcomes[row["id"]] = row["candidate"]

    manifest["arms"].append(arm)
    for pair in manifest["pairs"]:
        for side in paired.SIDES:
            trial = pair[side]
            source = trial["source"]
            candidate = outcomes[paired.text_digest(source)]
            trial["outputs"][arm] = candidate
            retained = _retained(source, candidate)
            if retained:
                trial.setdefault("retained_prose", {})[arm] = retained
    return rows


def generate(manifest_path: Path, config_path: Path, output: Path) -> None:
    """Run the local system model once per distinct source and keep every disposition."""
    arms, _, trials, manifest_digest = paired._manifest(manifest_path, minimum_arms=1)
    manifest, observed_digest = paired._read(manifest_path)
    if observed_digest != manifest_digest:
        raise ValueError("The input manifest changed while loading.")
    config, config_digest = paired._read(config_path)
    if not isinstance(config, dict):
        raise ValueError("Expected an Apple generation plan object.")
    arm, instructions, build, maximum = (
        config.get(key) for key in ("arm", "instructions", "macos_build", "maximum_response_tokens")
    )
    if (
        not all(paired._nonempty_string(value) for value in (arm, instructions, build))
        or arm in arms
        or type(maximum) is not int
        or maximum < 1
    ):
        raise ValueError("Provide a new arm, instructions, macOS build and positive response token limit.")
    if output.exists():
        raise FileExistsError(output)
    if platform.system() != "Darwin":
        raise RuntimeError("The Apple baseline requires macOS and the Foundation Models SDK.")
    if subprocess.check_output(["/usr/bin/sw_vers", "-buildVersion"], text=True).strip() != build:
        raise ValueError("The current macOS build differs from the generation plan.")
    swift = Path(__file__).resolve().with_suffix(".swift")
    sources = {paired.text_digest(trial.source): trial.source for trial in trials}
    inputs = [{"id": digest, "source": source} for digest, source in sorted(sources.items())]
    output.mkdir()
    job_path = output / "generation-input.json"
    job_digest = paired._write(
        job_path, {"instructions": instructions, "maximumResponseTokens": maximum, "cases": inputs}
    )
    paired._write(
        output / "settings.json",
        {
            "input_sha256": manifest_digest,
            "config_sha256": config_digest,
            "job_sha256": job_digest,
            "worker_sha256": paired._file_digest(Path(__file__)),
            "swift_source_sha256": paired._file_digest(swift),
            "scorer_sha256": paired._file_digest(Path(paired.__file__)),
            "platform": "SystemLanguageModel.default",
            "sampling": "greedy",
            "macos": subprocess.check_output(["/usr/bin/sw_vers"], text=True).strip(),
            "sdk": subprocess.check_output(["/usr/bin/xcrun", "--show-sdk-version"], text=True).strip(),
            "swift": subprocess.check_output(["/usr/bin/xcrun", "swiftc", "--version"], text=True).strip(),
        },
    )
    responses = output / "responses.jsonl"
    with (
        TemporaryDirectory(prefix="raiw-apple-benchmark-") as directory,
        (output / "runtime.log").open("xb") as diagnostics,
    ):
        executable = Path(directory) / "run-apple"
        subprocess.run(  # noqa: S603 - fixed compiler and repository source file, no shell
            ["/usr/bin/xcrun", "swiftc", str(swift), "-o", str(executable)], stderr=diagnostics, check=True
        )
        with responses.open("xb") as stream:
            completed = subprocess.run(  # noqa: S603 - helper compiled into our temporary directory
                [str(executable), str(job_path.resolve())], stdout=stream, stderr=diagnostics, check=False
            )
        if completed.returncode:
            raise RuntimeError(f"Apple generation exited {completed.returncode}. See {output / 'runtime.log'}.")

    attempts = _merge(manifest, arm, inputs, responses)
    paired._write(output / "attempts.json", attempts)
    paired._write(output / "trials.json", manifest)
    log.info(
        "Added %s Apple outputs and %s refusals.",
        sum(r["candidate"] is not None for r in attempts),
        sum(r["candidate"] is None for r in attempts),
    )


def main() -> None:
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("input", "config", "output-dir"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        generate(args.input, args.config, args.output_dir)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
