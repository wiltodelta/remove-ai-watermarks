"""Offline contracts for the published DIPPER-paper NumPy detector adapter."""

import hashlib
from types import SimpleNamespace

import pytest

from scripts import benchmark_dipper_reference as adapter


def contract():
    return {
        "generation": {
            "detector_variant": "dipper_paper_numpy",
            "reference_revision": adapter.REFERENCE_REVISION,
            "reference_utils_sha256": adapter.REFERENCE_SHA256,
            "vocab_sha256": adapter.VOCAB_SHA256,
            "tokenization_sha256": adapter.TOKENIZATION_SHA256,
            "watermark_fraction": 0.5,
            "vocab_size": 50257,
            "context_sha256": hashlib.sha256(b"Prefix").hexdigest(),
        }
    }


@pytest.mark.parametrize(
    "field",
    [
        "detector_variant",
        "reference_revision",
        "reference_utils_sha256",
        "vocab_sha256",
        "tokenization_sha256",
        "watermark_fraction",
        "vocab_size",
        "context_sha256",
    ],
)
def test_reference_contract_rejects_drift(field):
    manifest = contract()
    manifest["generation"][field] = "wrong"
    with pytest.raises(ValueError, match=field):
        adapter.validate_contract(manifest, b"Prefix")


def test_valid_contract_and_empty_prefix():
    adapter.validate_contract(contract(), b"Prefix")
    with pytest.raises(ValueError, match="nonempty"):
        adapter.validate_contract(contract(), b"  ")


def test_reference_hash_before_execution(tmp_path):
    source = tmp_path / "reference.py"
    source.write_text("raise RuntimeError('Executed foreign reference')")
    with pytest.raises(ValueError, match="hash mismatch"):
        adapter.load_reference(source)


def test_unvalidated_seed_byteorder_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter.sys, "byteorder", "big")
    with pytest.raises(ValueError, match="little-endian"):
        adapter.load_reference(tmp_path / "not_loaded.py")


def test_same_size_different_token_mapping_is_rejected():
    with pytest.raises(ValueError, match="token mapping"):
        adapter.validate_tokenizer(SimpleNamespace(vocab_size=50257, get_vocab=lambda: {"wrong": 0}))


def test_scoring_uses_prefix_and_separate_completion_tokens():
    calls = []

    def tokenizer(text, add_special_tokens):
        calls.append((text, add_special_tokens))
        return {"input_ids": [8, 9] if text == "Prefix" else [10, 11, 12]}

    def detect(ids, cache, fraction, vocab):
        assert ids == [9, 10, 11, 12]
        assert cache == {}
        assert fraction == 0.5
        assert vocab == 50257
        return 3.0, True

    assert adapter.score_texts({"source": "Completion"}, "Prefix", tokenizer, {"watermark_detect": detect}) == {
        "source": {"z": 3.0, "scored_transitions": 3}
    }
    assert calls == [("Prefix", True), ("Completion", False)]


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), None, "1", []])
def test_invalid_reference_scores_fail(value):
    with pytest.raises(ValueError, match="invalid score"):
        adapter.score_texts(
            {"source": "Text"},
            "Prefix",
            lambda *a, **k: {"input_ids": [1]},
            {"watermark_detect": lambda *a: (value, True)},
        )


@pytest.mark.parametrize(
    ("source", "baseline", "after", "retained", "status"),
    [
        (4, 0, 0, 0, "invalid_positive_control"),
        (8, 4.1, 0, 0, "invalid_negative_control"),
        (8, 0, None, 0, "no_after_output"),
        (8, 0, 4.1, 0, "retained"),
        (8, 4, 4, 0, "not_detected_at_reference_threshold"),
        (8, 0, 2, 1, "partial_unverified"),
    ],
)
def test_author_strict_threshold_and_control_priority(source, baseline, after, retained, status):
    scores = {"source": {"z": source}, "baseline": {"z": baseline}}
    if after is not None:
        scores["after"] = {"z": after}
    assert adapter.assess(scores, retained) == status


def test_same_vocab_different_tokenization_is_rejected(monkeypatch):
    import json

    vocab = {"test": 0}
    monkeypatch.setattr(
        adapter,
        "VOCAB_SHA256",
        hashlib.sha256(json.dumps(vocab, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    )
    tokenizer = SimpleNamespace(
        vocab_size=50257, get_vocab=lambda: vocab, backend_tokenizer=SimpleNamespace(to_str=lambda: "{}")
    )
    with pytest.raises(ValueError, match="structure"):
        adapter.validate_tokenizer(tokenizer)


@pytest.mark.parametrize(
    ("context_ids", "completion_ids", "message"),
    [
        ([], [1], "context tokenization"),
        ([1], [], "no completion tokens"),
        ([1], [50257], "invalid token IDs"),
        ([1], [True], "invalid token IDs"),
    ],
)
def test_empty_or_out_of_range_tokenization_fails(context_ids, completion_ids, message):
    def tokenizer(text, **kwargs):
        return {"input_ids": context_ids if text == "Prefix" else completion_ids}

    with pytest.raises(ValueError, match=message):
        adapter.score_texts({"source": "Text"}, "Prefix", tokenizer, {})


def test_reference_function_inventory_is_required(tmp_path, monkeypatch):
    source = tmp_path / "reference.py"
    source.write_text("def watermark_detect(): pass\n")
    monkeypatch.setattr(adapter, "REFERENCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="inventory"):
        adapter.load_reference(source)
