"""Prevent incompatible token maps or missing row signals becoming coverage."""

import json

import numpy as np
import pytest

from scripts import benchmark_automark_reference as adapter


def _assets(tmp_path):
    candidate = {
        "model": {"vocab": {"word": 0, "next": 1}},
        "decoder": {"type": "ByteLevel"},
        "pre_tokenizer": {"type": "ByteLevel"},
        "added_tokens": [{"id": 2, "content": "<stop>"}],
    }
    reference = {**candidate, "added_tokens": [*candidate["added_tokens"], {"id": 3, "content": "<audio>"}]}
    for name, data in [("candidate", candidate), ("reference", reference)]:
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    np.savez(tmp_path / "groups.npz", groups=np.arange(4), eligible=np.array([True, True, False, False]))
    np.savez(tmp_path / "canon.npz", vocab_size=4, sense=np.arange(4))
    return [
        tmp_path / "candidate.json",
        tmp_path / "reference.json",
        tmp_path / "groups.npz",
        tmp_path / "canon.npz",
        4,
    ]


def test_unshared_special_ids_are_explicitly_forbidden(tmp_path):
    proof = adapter.validate_assets(*_assets(tmp_path))
    assert proof["forbidden_token_ids"] == [3]
    assert proof["ordinary_mapping_equal"] is True


@pytest.mark.parametrize("field", ["model", "decoder", "pre_tokenizer"])
def test_changed_text_tokenization_is_not_compatible(tmp_path, field):
    args = _assets(tmp_path)
    data = json.loads(args[0].read_text())
    data[field] = {"changed": True}
    args[0].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="tokenizer"):
        adapter.validate_assets(*args)


@pytest.mark.parametrize("fault", ["size", "eligible", "canonical", "out_of_range"])
def test_asset_mismatches_are_not_silently_accepted(tmp_path, fault):
    args = _assets(tmp_path)
    if fault == "size":
        np.savez(args[2], groups=np.arange(3), eligible=np.array([True, True, False]))
    elif fault == "eligible":
        np.savez(args[2], groups=np.arange(4), eligible=np.ones(4, dtype=bool))
    elif fault == "canonical":
        np.savez(args[3], vocab_size=4, sense=np.array([0, 1, 2, 2]))
    else:
        np.savez(args[2], groups=np.array([0, 1, 2, 4]), eligible=np.zeros(4, dtype=bool))
    with pytest.raises(ValueError, match="asset"):
        adapter.validate_assets(*args)


@pytest.mark.parametrize(
    ("source", "baseline", "after", "expected"),
    [
        (0.2, 0.8, 0.7, "invalid_positive_control"),
        (0.001, 0.001, 0.5, "invalid_negative_control"),
        (0.001, 0.8, None, "no_after_output"),
        (0.001, 0.8, 0.001, "retained"),
        (0.001, 0.8, 0.7, "not_detected_at_reference_alpha"),
    ],
)
def test_pvalue_controls_and_refusals_remain_distinct(source, baseline, after, expected):
    assert adapter.assess(source, baseline, after, 0.01) == expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_pvalues_fail(bad):
    with pytest.raises(ValueError, match="p-value"):
        adapter.assess(bad, 0.8, 0.7, 0.01)


@pytest.mark.parametrize("fault", ["missing", "shape", "nan"])
def test_missing_or_corrupt_auxiliary_signals_fail(fault):
    lp = np.array([0.0, -1.0, -2.0])
    ids = np.tile(np.arange(8), (3, 1)).astype(float)
    rows = np.tile(np.linspace(-1.0, -8.0, 8), (3, 1))
    if fault == "missing":
        lp[:] = 0
        rows[:] = 0
    elif fault == "shape":
        rows = rows[:2]
    else:
        rows[1, 1] = np.nan
    with pytest.raises(ValueError, match="signal"):
        adapter.validate_auxiliary([0, 1, 2], lp, ids, rows, 10)


@pytest.mark.parametrize("fault", ["hash", "key", "scheme", "sampling", "post_sampling"])
def test_frozen_configuration_mismatches_fail_before_loading_author_code(tmp_path, monkeypatch, fault):
    import argparse
    import hashlib

    hashes = {}
    for name in ["source", "baseline"]:
        data = name.encode()
        (tmp_path / f"{name}.txt").write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    payload = {"seed": 1, "logit_processor_config": {"watermark_class": "RotationFirst"}}
    config = tmp_path / "watermark.json"
    config.write_text(json.dumps(payload))
    generation = {
        "watermark_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "key": 1,
        "scheme": "RotationFirst",
        "sampling_after_processor": True,
        "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
    }
    if fault == "hash":
        config.write_text(config.read_text() + "\n")
    elif fault == "key":
        generation["key"] = 2
    elif fault == "scheme":
        generation["scheme"] = "ConcordMark"
    elif fault == "sampling":
        generation["sampling"]["temperature"] = 0.7
    else:
        generation["sampling_after_processor"] = False
    (tmp_path / "manifest.json").write_text(json.dumps({"inputs_sha256": hashes, "generation": generation}))
    monkeypatch.setattr(adapter, "load_reference", lambda *_a: pytest.fail("invalid metadata reached author code"))
    with pytest.raises(ValueError, match="AutoMark"):
        adapter.run(argparse.Namespace(input_dir=tmp_path, output_dir=tmp_path / "output", alpha=0.01))
