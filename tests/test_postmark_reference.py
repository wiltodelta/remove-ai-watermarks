"""Offline contracts for author PostMark replays."""

import argparse
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import benchmark_postmark_reference as adapter


def test_each_output_recomputes_its_own_detection_words():
    seen = []
    embedder = SimpleNamespace(get_words=lambda text: [text.lower()])

    def presence(text, words, threshold):
        seen.append((text, words, threshold))
        return 0.5

    scores = adapter.score_texts({"source": "Original", "after": "Changed"}, embedder, presence, 0.7)
    assert seen == [("Original", ["original"], 0.7), ("Changed", ["changed"], 0.7)]
    assert scores["after"] == {"words": ["changed"], "presence": 0.5}


@pytest.mark.parametrize("words", [[], None, "word", [""], [3]])
def test_empty_word_selection_is_a_failure(words):
    with pytest.raises(ValueError, match="words"):
        adapter.score_texts({"source": "Short"}, SimpleNamespace(get_words=lambda _: words), lambda *a, **k: 0, 0.7)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_presence_scores_are_failures(score):
    with pytest.raises(ValueError, match="presence"):
        adapter.score_texts(
            {"source": "Text"}, SimpleNamespace(get_words=lambda _: ["word"]), lambda *a, **k: score, 0.7
        )


def _inputs(path):
    path.mkdir()
    hashes = {}
    for name, text in [("source", "Source"), ("baseline", "Baseline"), ("after", "After")]:
        data = text.encode()
        (path / f"{name}.txt").write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = {"inputs_sha256": hashes, "generation": {"synthetic": True}, "retained_prose_spans": 0}
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_corrupted_input_fails_before_author_code_executes(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    _inputs(inputs)
    (inputs / "after.txt").write_text("Tampered")
    monkeypatch.setattr(adapter, "load_reference", lambda _: pytest.fail("unverified inputs reached author code"))
    with pytest.raises(ValueError, match="hash"):
        adapter.run(
            argparse.Namespace(
                input_dir=inputs, output_dir=tmp_path / "out", ratio=0.12, cosine_threshold=0.7, threshold=None
            )
        )


def test_numerical_replay_stays_uncalibrated_and_records_recomputed_words(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    _inputs(inputs)
    embedder = SimpleNamespace(get_words=lambda text: [text.lower()])
    monkeypatch.setattr(
        adapter, "load_reference", lambda _: (embedder, lambda *a, **k: 0.5, {"reference_revision": "test"})
    )
    args = argparse.Namespace(
        input_dir=inputs, output_dir=tmp_path / "out", ratio=0.12, cosine_threshold=0.7, threshold=None
    )
    report = adapter.run(args)
    assert report["status"] == "scored_uncalibrated"
    assert report["scores"]["after"]["words"] == ["after"]
    assert json.loads((args.output_dir / "benchmark.json").read_text()) == report
    with pytest.raises(FileExistsError):
        adapter.run(args)


@pytest.mark.parametrize(
    ("field", "declared", "observed"), [("ratio", 0.2, 0.12), ("table_sha256", "writer-key", "detector-key")]
)
def test_declared_generation_key_and_ratio_must_match_replay(tmp_path, monkeypatch, field, declared, observed):
    inputs = tmp_path / "inputs"
    _inputs(inputs)
    manifest_path = inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["generation"][field] = declared
    manifest_path.write_text(json.dumps(manifest))
    metadata = {"word_table_sha256": observed}
    monkeypatch.setattr(adapter, "load_reference", lambda _: (None, None, metadata))
    args = argparse.Namespace(
        input_dir=inputs, output_dir=tmp_path / "out", ratio=0.12, cosine_threshold=0.7, threshold=None
    )
    with pytest.raises(ValueError, match="generation"):
        adapter.run(args)
