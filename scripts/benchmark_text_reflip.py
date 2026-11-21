"""Add a pinned local ReFlip baseline to a paired text manifest."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import benchmark_text_pairs as paired  # noqa: E402

log = logging.getLogger(__name__)
ARM = "reflip"
SOURCE_FILES = {
    "reflip/__init__.py",
    "reflip/facts.py",
    "reflip/facts_lexicon.json",
    "reflip/transforms/__init__.py",
    "reflip/transforms/llm.py",
    "reflip/transforms/rules.py",
    "reflip/transforms/unicode.py",
    "reflip/words.py",
}


def add_baseline(manifest_path: Path, config_path: Path, reference_root: Path, url: str, output: Path) -> None:
    """Generate the baseline in a new directory without changing the input manifest."""
    endpoint = url.rstrip("/")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or parsed.path != "/v1"
        or parsed.username is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Use an HTTP /v1 endpoint on a literal loopback address.")
    arms, _, trials, manifest_digest = paired._manifest(manifest_path, minimum_arms=1)
    manifest, digest = paired._read(manifest_path)
    if digest != manifest_digest:
        raise ValueError("The manifest changed while loading.")
    config, config_digest = paired._read(config_path)
    if not isinstance(config, dict):
        raise ValueError("Expected a ReFlip plan object.")
    method = config.get("method", "paraphrase")
    if method not in ("paraphrase", "infill"):
        raise ValueError("ReFlip method must be paraphrase or infill.")
    arm = ARM if method == "paraphrase" else "reflip_infill"
    if arm in arms:
        raise ValueError(f"The input already declares the {arm} arm.")
    expected_digest = config.get("model_digest")
    workers = config.get("workers")
    if (
        not isinstance(expected_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
        or type(workers) is not int
        or workers < 1
        or not paired._nonempty_string(config.get("revision"))
        or not paired._nonempty_string(config.get("ollama_version"))
    ):
        raise ValueError("Provide the ReFlip revision, Ollama version, model digest and positive worker count.")
    paired._verify_file_digests(reference_root, config.get("source_sha256"), SOURCE_FILES)
    if output.exists():
        raise FileExistsError(output)
    if "reflip" in sys.modules:
        raise RuntimeError("The ReFlip package is already loaded.")

    import requests

    sys.path.insert(0, str(reference_root.resolve()))
    from reflip.transforms import Options, llm

    rewrite = llm.paraphrase if method == "paraphrase" else llm.infill
    for module_name, module in sys.modules.copy().items():
        if module_name == "reflip" or module_name.startswith("reflip."):
            relative = Path(module.__file__).resolve().relative_to(reference_root.resolve())
            if relative.as_posix() not in config["source_sha256"]:
                raise RuntimeError(f"Unpinned ReFlip module loaded: {relative}.")
    options = Options(base_url=endpoint, workers=workers)
    with requests.Session() as session:
        session.trust_env = False
        server = endpoint.removesuffix("/v1")
        metadata = {}
        for resource in ("version", "tags"):
            response = session.get(f"{server}/api/{resource}", timeout=5, allow_redirects=False)
            if response.status_code != 200:
                raise RuntimeError(f"Ollama /api/{resource} returned HTTP {response.status_code}.")
            metadata[resource] = response.json()
    if metadata["version"].get("version") != config["ollama_version"] or not any(
        row.get("name") == options.model and row.get("digest") == expected_digest
        for row in metadata["tags"].get("models", [])
    ):
        raise ValueError("The running Ollama version or ReFlip model digest differs from the plan.")

    output.mkdir()
    paired._write(
        output / "settings.json",
        {
            "input_sha256": manifest_digest,
            "config": config,
            "config_sha256": config_digest,
            "worker_sha256": paired._file_digest(Path(__file__)),
            "scorer_sha256": paired._file_digest(Path(paired.__file__)),
            "options": {key: value for key, value in asdict(options).items() if key != "api_key"},
            "runtime": {
                "python": sys.version,
                "requests": version("requests"),
                "ollama": metadata["version"]["version"],
            },
            "method": method,
        },
    )
    original_chat = llm.Chat
    mutex = Lock()
    current = ("", "")
    finish_reasons: list[str | None] = []

    with (output / "transport.jsonl").open("x", encoding="utf-8") as transport:

        class ObservedChat(original_chat):
            def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 600.0) -> None:
                super().__init__(base_url, api_key, model, timeout)
                self.session.trust_env = False
                self.session.max_redirects = 0

            def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
                response = None
                error = None
                try:
                    response = super()._post(path, body)
                    return response
                except llm.ChatError as exc:
                    error = str(exc)
                    raise
                finally:
                    with mutex:
                        if isinstance(response, dict) and isinstance(response.get("choices"), list):
                            finish_reasons.extend(
                                choice.get("finish_reason")
                                for choice in response["choices"]
                                if isinstance(choice, dict)
                            )
                        row = {
                            "pair": current[0],
                            "side": current[1],
                            "payload": body,
                            "response": response,
                            "error": error,
                        }
                        transport.write(json.dumps(row, ensure_ascii=False) + "\n")
                        transport.flush()

        attempts = []
        outcomes = {}
        llm.Chat = ObservedChat
        try:
            for trial in trials:
                current = trial.pair, trial.side
                finish_reasons.clear()
                reason = None
                notes = {}
                try:
                    result = rewrite(trial.source, options)
                    candidate, notes = result.text, result.notes
                except (RuntimeError, ValueError) as error:
                    candidate, reason = None, str(error)
                if "length" in finish_reasons:
                    candidate, reason = None, "Length-limited completion."
                elif notes.get("blocks_failed"):
                    candidate, reason = None, "ReFlip could not rewrite every prose block."
                if candidate is not None and not paired._nonempty_string(candidate):
                    candidate, reason = None, "The baseline returned empty text."
                outcomes[current] = candidate
                attempts.append(
                    {
                        "pair": trial.pair,
                        "side": trial.side,
                        "finish_reasons": list(finish_reasons),
                        "notes": notes,
                        "refusal": reason,
                    }
                )
        finally:
            llm.Chat = original_chat
    for pair in manifest["pairs"]:
        for side in paired.SIDES:
            pair[side]["outputs"][arm] = outcomes[pair["id"], side]
    manifest["arms"].append(arm)
    paired._write(output / "attempts.json", attempts)
    paired._write(output / "trials.json", manifest)
    log.info("Ran ReFlip on %s marked and clean sources.", len(attempts))


def main() -> None:
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("input", "config", "reference-root", "output-dir"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        add_baseline(args.input, args.config, args.reference_root, args.url, args.output_dir)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
