"""Lexical assets need exact tokenizer semantics, not just matching array lengths."""

import copy
import json

import numpy as np
import pytest

from scripts import benchmark_automark_submission as adapter


@pytest.fixture
def lexical(tmp_path):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    payload = {
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]},
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": None,
        "decoder": None,
        "post_processor": None,
        "padding": None,
        "truncation": None,
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(payload))
    payload["model"]["merges"] = [["a", "b"]]
    (tokenizer / "tokenizer.json").write_text(json.dumps(payload))
    source = tmp_path / "source"
    assets = source / "src/lm_wm_tools/watermarks/assets"
    assets.mkdir(parents=True)
    metadata = {
        "tokenizer": "reference-model",
        "tokenizer_revision": "frozen",
        "vocabulary_size": 3,
        "groups": 2,
        "merged_tokens": 1,
        "eligible_tokens": 2,
    }
    (assets / "llama3_wordnet_groups.json").write_text(json.dumps(metadata))
    np.savez(assets / "llama3_wordnet_groups.npz", groups=np.array([0, 0, 2]), eligible=np.array([True, True, False]))
    reference = {"model": "reference-model", "revision": "frozen", "tokenizer_sha256": adapter.sha(reference_path)}
    return source, tokenizer, reference_path, reference


def test_bpe_merge_serialization_changes_do_not_change_the_tokenizer(lexical):
    proof = adapter.validate_lexical_assets(*lexical, 3)
    assert proof["canonical_tokenizer_equal"]
    assert proof["counts"] == {"groups": 2, "merged_tokens": 1, "eligible_tokens": 2}


def test_canonicalization_preserves_input_and_merge_rank():
    payload = {"model": {"type": "BPE", "merges": ["a b", "ab c"]}}
    before = copy.deepcopy(payload)
    first = adapter.canonical_tokenizer(payload)
    assert payload == before
    assert first["model"]["merges"] == [["a", "b"], ["ab", "c"]]
    payload["model"]["merges"].reverse()
    assert adapter.canonical_tokenizer(payload) != first


@pytest.mark.parametrize("merge", [["a"], ["a", "b", "c"], "a b c", ["a", 1], ["", "b"]])
def test_invalid_merge_rows_are_not_normalized_away(merge):
    with pytest.raises(ValueError, match="BPE"):
        adapter.canonical_tokenizer({"model": {"type": "BPE", "merges": [merge]}})


@pytest.mark.parametrize(
    "field", ["vocab", "merges", "normalizer", "pre_tokenizer", "decoder", "added_tokens", "post_processor"]
)
def test_same_size_but_different_tokenizer_semantics_are_rejected(lexical, field):
    source, tokenizer, reference_path, reference = lexical
    path = tokenizer / "tokenizer.json"
    payload = json.loads(path.read_text())
    if field == "vocab":
        payload["model"]["vocab"]["a"], payload["model"]["vocab"]["b"] = 1, 0
    elif field == "merges":
        payload["model"]["merges"] = [["b", "a"]]
    else:
        payload[field] = [] if field == "added_tokens" else {"changed": True}
        if field == "added_tokens":
            payload[field].append({"id": 3, "content": "different"})
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="semantics"):
        adapter.validate_lexical_assets(source, tokenizer, reference_path, reference, 3)


@pytest.mark.parametrize(
    "change",
    [
        "range",
        "shape",
        "dtype",
        "idempotence",
        "eligible_dtype",
        "counts",
        "identity",
        "identity_model",
        "vocabulary_metadata",
        "reference_hash",
    ],
)
def test_incompatible_lexical_arrays_and_metadata_are_rejected(lexical, change):
    source, tokenizer, reference_path, reference = lexical
    assets = source / "src/lm_wm_tools/watermarks/assets"
    if change in {"range", "shape", "dtype", "idempotence", "eligible_dtype"}:
        groups = {
            "range": [0, 0, 3],
            "shape": [0, 0],
            "dtype": [0.0, 0.0, 2.0],
            "idempotence": [1, 2, 2],
            "eligible_dtype": [0, 0, 2],
        }[change]
        eligible = np.array([1, 1, 0]) if change == "eligible_dtype" else np.array([True, True, False])
        np.savez(assets / "llama3_wordnet_groups.npz", groups=np.array(groups), eligible=eligible)
    elif change in {"counts", "identity", "identity_model", "vocabulary_metadata"}:
        path = assets / "llama3_wordnet_groups.json"
        payload = json.loads(path.read_text())
        field = {
            "counts": "groups",
            "identity": "tokenizer_revision",
            "identity_model": "tokenizer",
            "vocabulary_metadata": "vocabulary_size",
        }[change]
        payload[field] = 3 if change == "counts" else (4 if change == "vocabulary_metadata" else "wrong")
        path.write_text(json.dumps(payload))
    else:
        reference["tokenizer_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=r"tokenizer|Lexical"):
        adapter.validate_lexical_assets(source, tokenizer, reference_path, reference, 3)


@pytest.mark.parametrize(
    ("identifier", "watermark_class"), [("candidate-014", "CascadeCopula"), ("candidate-015", "FreshRace")]
)
def test_supported_ids_do_not_accept_a_different_class(identifier, watermark_class):
    proof = {
        "submission": {
            "id": identifier,
            "watermark_class": watermark_class,
            "source_map_sha256": "map",
            "configuration_sha256": "config",
        },
        "reference_revision": "revision",
    }
    generation = {
        "submission_id": identifier,
        "reference_revision": "revision",
        "source_map_sha256": "map",
        "configuration_sha256": "config",
        "lexical_assets": {},
        "sampling_after_processor": True,
        "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
        "key": 45123,
        "vocab_size": 128256,
    }
    with pytest.raises(ValueError, match="exact submission"):
        adapter.validate_provenance(generation, proof)


@pytest.mark.parametrize("failure", ["missing_reference", "stale_proof"])
@pytest.mark.parametrize("family", ["lexical", "freshrace"])
def test_replay_checks_lexical_provenance_before_importing_author_code(tmp_path, monkeypatch, failure, family):
    from types import SimpleNamespace

    from scripts import benchmark_alimark_reference as inputs_adapter

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.yaml").write_text("{}")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "watermark.json").write_text(json.dumps({"device": "cpu", "seed": 1}))
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (tokenizer / name).write_text(json.dumps({"vocab_size": 3}))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"tokenizer_references": {"llama3_wordnet": {}}}))
    generation = {
        "vocab_size": 3,
        "key": 1,
        "tokenizer_files_sha256": {path.name: adapter.sha(path) for path in tokenizer.iterdir()},
        "watermark_config_sha256": adapter.sha(inputs / "watermark.json"),
        family + "_assets": {"frozen": True},
    }
    monkeypatch.setattr(inputs_adapter, "load_inputs", lambda _: ({}, {"generation": generation}))
    monkeypatch.setattr(adapter, "verify_submission", lambda *_: {"submission": {"configuration_path": "config.yaml"}})
    monkeypatch.setattr(adapter, "validate_provenance", lambda *_: None)
    monkeypatch.setattr(adapter, "validate_lexical_assets", lambda *_: {"frozen": False})
    monkeypatch.setattr(adapter, "validate_freshrace_assets", lambda *_: {"frozen": False})
    monkeypatch.setattr(
        adapter, "require_fresh_namespace", lambda: pytest.fail("Author import reached before proof checks")
    )
    args = SimpleNamespace(
        input_dir=inputs,
        output_dir=tmp_path / "output",
        source_store=tmp_path / "store",
        submission_dir=source,
        inventory=inventory,
        submission_id="candidate-002" if family == "lexical" else "candidate-015",
        tokenizer_dir=tokenizer,
        reference_tokenizer=None if failure == "missing_reference" else tmp_path / "reference.json",
    )
    expected = "pinned reference tokenizer" if failure == "missing_reference" else "provenance mismatch"
    with pytest.raises(ValueError, match=expected):
        adapter.run(args)
    assert not args.output_dir.exists()
