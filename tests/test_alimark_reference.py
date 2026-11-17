"""Offline contracts for matched AliMark reference replays."""

import hashlib
import json

import pytest

from scripts.benchmark_alimark_reference import assess_scores, load_inputs


def test_uncalibrated_scores_never_become_a_removal_verdict():
    assert assess_scores(9.0, 0.0, -1.0, threshold=None) == "scored_uncalibrated"


@pytest.mark.parametrize(
    ("before", "baseline", "after", "retained", "expected"),
    [
        (1.0, 0.0, -1.0, 0, "invalid_positive_control"),
        (9.0, 5.0, -1.0, 0, "invalid_negative_control"),
        (9.0, 0.0, None, 0, "no_after_output"),
        (9.0, 0.0, 5.0, 0, "retained"),
        (9.0, 0.0, -1.0, 2, "partial_unverified"),
        (9.0, 0.0, -1.0, 0, "below_declared_threshold"),
    ],
)
def test_declared_threshold_preserves_control_and_partial_accounting(before, baseline, after, retained, expected):
    assert assess_scores(before, baseline, after, threshold=4.0, retained_prose_spans=retained) == expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scores_are_runtime_errors(bad):
    with pytest.raises(ValueError, match="finite"):
        assess_scores(9.0, 0.0, bad, threshold=None)


def test_frozen_input_hashes_are_checked_before_loading_models(tmp_path):
    source = "  An original synthetic sentence.\r\n"
    baseline = "A separate unmarked control."
    for name, text in [("source", source), ("baseline", baseline)]:
        (tmp_path / f"{name}.txt").write_text(text)
    manifest = {
        "inputs_sha256": {
            "source": hashlib.sha256(source.encode()).hexdigest(),
            "baseline": hashlib.sha256(baseline.encode()).hexdigest(),
        },
        "retained_prose_spans": 0,
        "generation": {"reference_revision": "research-reference"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    loaded, observed = load_inputs(tmp_path)
    assert loaded == {"source": source, "baseline": baseline}
    assert observed == manifest
    (tmp_path / "source.txt").write_text("Tampered source.")
    with pytest.raises(ValueError, match="hash"):
        load_inputs(tmp_path)


def test_relative_encoder_path_is_resolved_before_entering_reference_checkout(tmp_path, monkeypatch):
    import argparse
    import importlib
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from scripts import benchmark_alimark_reference as adapter

    reference = tmp_path / "reference"
    reference.mkdir()
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    observed = []

    def original_download(*_args, **_kwargs):
        return None

    nltk = SimpleNamespace(data=SimpleNamespace(path=[], find=lambda _: None), download=original_download)

    def stop_at_model_load(config, *, load_llm):
        observed.append((config.watermark_embedder, load_llm))
        raise RuntimeError("stop after recording the encoder path")

    original_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "nltk":
            return nltk
        if name == "watermark.alimark":
            return SimpleNamespace(AliMark=stop_at_model_load)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(adapter, "load_inputs", lambda _: ({"source": "Synthetic source."}, {}))
    args = argparse.Namespace(
        reference_root=reference,
        embedder=Path("encoder"),
        dependency_dir=None,
        input_dir=tmp_path,
        output_dir=tmp_path / "output",
        nltk_data=tmp_path / "nltk",
        device="cpu",
        block_size=4,
    )
    with pytest.raises(RuntimeError, match="stop after recording"):
        adapter.run(args)
    assert observed == [(str(encoder), False)]
    assert Path.cwd() == tmp_path
    assert nltk.download is original_download
