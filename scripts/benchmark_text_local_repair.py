"""Produce paired trials for local sentence restoration and regeneration."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from remove_ai_watermarks import text, text_api  # noqa: E402
from scripts import benchmark_text_pairs as paired  # noqa: E402

log = logging.getLogger(__name__)
ARMS = ("restore", "strict_residual", "document_residual")
INITIAL = {"temperature": 0, "seed": 7291}
RESIDUAL = {"temperature": 0.4, "seed": 84237}
MAX_TOKENS = 2048
TIMEOUT = 185
SentenceSlots = tuple[list[str], list[str]]
Rewriter = Callable[[tuple[str, ...]], tuple[str, ...]]


def _prose(document: str) -> tuple[str, ...]:
    return tuple(span.text.strip() for span in text.split_protected_text_spans(document) if not span.locked)


def _aligned(original: list[SentenceSlots], revised: list[SentenceSlots]) -> list[tuple[int, int, str, str]]:
    return [
        (i, j, before, after)
        for i, (left, right) in enumerate(zip(original, revised, strict=True))
        for j, (before, after) in enumerate(zip(left[0], right[0], strict=True))
    ]


def _replace(chunks: list[SentenceSlots], updates: list[tuple[int, int, str]]) -> tuple[str, ...]:
    for i, j, sentence in updates:
        chunks[i][0][j] = sentence
    return tuple(
        sentences[0] + "".join(gap + sentence for gap, sentence in zip(gaps, sentences[1:], strict=True))
        for sentences, gaps in chunks
    )


def _validate(source: str, replacements: tuple[str, ...]) -> str:
    return text.remove_text_invisible(
        source, document_rewriter=lambda _: replacements, document_validator=text.verify_text_rewrite_facts
    ).cleaned_text


def _rewrite(source: str, initial: Rewriter, residual: Rewriter) -> tuple[dict, dict]:
    result = {"source": source, "outputs": dict.fromkeys(ARMS)}
    trace = {"calls": 0}
    verify = text.verify_text_rewrite_facts
    original = _prose(source)
    if not original:
        return result, {**trace, "refusal": "No rewriteable prose."}
    aligned = [text_api._exact_sentence_slots(prose) for prose in original]
    try:
        trace["calls"] += 1
        draft = initial(original)
        try:
            restored = _validate(source, draft)
        except RuntimeError as error:
            if str(error) != text._DOCUMENT_VETO:
                raise
            chunks = [text_api._exact_sentence_slots(prose) for prose in draft]
            edits = [(i, j, old) for i, j, old, new in _aligned(aligned, chunks) if not verify(old, new)]
            if not edits:
                raise RuntimeError("Document veto without individually rejected sentences.") from None
            restored = _validate(source, _replace(chunks, edits))
        chunks = [text_api._exact_sentence_slots(prose) for prose in _prose(restored)]
        retained = [(i, j, old) for i, j, old, new in _aligned(aligned, chunks) if old == new]
    except (RuntimeError, ValueError) as error:
        return result, {**trace, "refusal": str(error)}

    result["outputs"]["restore"] = restored
    if not retained:
        result["outputs"] = dict.fromkeys(ARMS, restored)
        return result, trace

    previous = tuple(old for _, _, old in retained)
    result["retained_prose"] = {"restore": list(previous)}
    try:
        trace["calls"] += 1
        revised = residual(previous)
        document = _validate(
            source, _replace(chunks, [(i, j, new) for (i, j, _), new in zip(retained, revised, strict=True)])
        )
        final = [text_api._exact_sentence_slots(prose) for prose in _prose(document)]
        if any(old == new for _, _, old, new in _aligned(aligned, final)):
            raise RuntimeError("Residual proposal preserved an original aligned sentence.")
    except (RuntimeError, ValueError) as error:
        return result, {**trace, "residual_refusal": str(error)}
    result["outputs"]["document_residual"] = document
    if all(verify(old, new) for old, new in zip(previous, revised, strict=True)):
        result["outputs"]["strict_residual"] = document
    return result, trace


def _client(url: str, model: str, settings: dict[str, int | float]) -> Rewriter:
    limit_field = "max_tokens"
    rewrite, _ = text_api.build_text_api_document_rewriters(
        base_url=url,
        model=model,
        rewrite_style="sentence",
        structured_output=True,
        disable_reasoning=True,
        max_completion_tokens=MAX_TOKENS,
        completion_token_field=limit_field,
        timeout=TIMEOUT,
        temperature=settings["temperature"],
        seed=int(settings["seed"]),
    )
    return rewrite


def generate(manifest: Path, output: Path, *, url: str, model: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}:
        raise ValueError("Use an HTTP endpoint on a literal loopback address.")
    arms, detectors, trials, digest = paired._manifest(manifest)
    if set(arms) != set(ARMS) or any(
        trial.retained_prose or any(value is not None for value in trial.outputs.values()) for trial in trials
    ):
        raise ValueError("Declare the three experiment arms with null outputs and no retained prose.")
    initial, residual = _client(url, model, INITIAL), _client(url, model, RESIDUAL)
    output.mkdir()
    provenance = {
        "input_sha256": digest,
        "sources": {
            Path(module.__file__).name: paired._file_digest(Path(module.__file__))
            for module in (text, text_api, paired)
        },
        "worker_sha256": paired._file_digest(Path(__file__)),
        "endpoint": url,
        "model": model,
        "first": INITIAL,
        "residual": RESIDUAL,
        "max_tokens": MAX_TOKENS,
        "timeout": TIMEOUT,
    }
    paired._write(output / "settings.json", provenance)
    pairs = {}
    records = []
    for trial in trials:
        result, trace = _rewrite(trial.source, initial, residual)
        pair = pairs.setdefault(trial.pair, {"id": trial.pair, "detector": trial.detector})
        pair[trial.side] = result
        records.append({"pair": trial.pair, "side": trial.side, **trace})
    paired._write(output / "attempts.json", records)
    paired._write(output / "trials.json", {"arms": list(ARMS), "detectors": detectors, "pairs": list(pairs.values())})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", required=True, help="Chat Completions base URL on a literal loopback address")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        generate(args.input, args.output_dir, url=args.url, model=args.model)
    except (OSError, RuntimeError, ValueError) as error:
        log.error("%s", error)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
