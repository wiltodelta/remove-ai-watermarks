"""Exact FreshRace source states need frozen asset and tokenizer provenance."""

import json

import numpy as np
import pytest

from scripts import benchmark_automark_submission as adapter


@pytest.fixture
def fresh(tmp_path):
    source = tmp_path / "source"
    assets = source / "src/lm_wm_tools/watermarks/assets"
    assets.mkdir(parents=True)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    payload = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}, "added_tokens": []}
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(payload))
    (tokenizer / "tokenizer.json").write_text(json.dumps(payload))
    np.save(assets / "freshrace_common_pairs.npy", np.array([[100, 1], [200, 2]], dtype="uint64"))
    np.save(assets / "freshrace_predictable.npy", np.array([2, 4], dtype="uint64"))
    for name in ["adaptive", "lexical"]:
        arrays = {"rare": np.array([True, False, True]), "common": np.array([3, 5], dtype="uint64")}
        if name == "lexical":
            arrays["groups"] = np.array([0, 0, 2])
        np.savez(assets / f"freshrace_{name}.npz", **arrays)
    return source, tokenizer, reference, {"tokenizer_sha256": adapter.sha(reference)}


def test_exact_fresh_assets_record_the_missing_embedded_tokenizer_metadata(fresh):
    proof = adapter.validate_freshrace_assets(*fresh, 3)
    assert proof["canonical_tokenizer_equal"]
    assert proof["embedded_tokenizer_metadata"] is None
    root = fresh[0] / "src/lm_wm_tools/watermarks/assets"
    assert proof["assets_sha256"] == {path.name: adapter.sha(path) for path in root.iterdir()}
    assert set(proof["assets_sha256"]) == {
        "freshrace_common_pairs.npy",
        "freshrace_predictable.npy",
        "freshrace_adaptive.npz",
        "freshrace_lexical.npz",
    }


@pytest.mark.parametrize("change", ["pair_shape", "pair_dtype", "pair_id", "predictable_shape", "predictable_dtype"])
def test_invalid_fresh_tables_are_rejected(fresh, change):
    source, *_ = fresh
    assets = source / "src/lm_wm_tools/watermarks/assets"
    if change.startswith("pair"):
        values = np.array([[100, 1], [200, 2]], dtype="uint64")
        if change == "pair_shape":
            values = values.reshape(-1)
        elif change == "pair_dtype":
            values = values.astype("int64")
        else:
            values[0, 1] = 3
        path = assets / "freshrace_common_pairs.npy"
    else:
        values = (
            np.array([[2, 4]], dtype="uint64") if change == "predictable_shape" else np.array([2, 4], dtype="int64")
        )
        path = assets / "freshrace_predictable.npy"
    np.save(path, values)
    with pytest.raises(ValueError, match="FreshRace"):
        adapter.validate_freshrace_assets(*fresh, 3)


@pytest.mark.parametrize(
    "change",
    [
        "keys",
        "rare_shape",
        "rare_dtype",
        "common_shape",
        "common_dtype",
        "common_empty",
        "common_order",
        "groups_shape",
        "groups_dtype",
        "groups_negative",
        "groups_range",
        "groups_idempotence",
    ],
)
def test_invalid_fresh_lexical_arrays_are_rejected(fresh, change):
    source, *_ = fresh
    path = source / "src/lm_wm_tools/watermarks/assets/freshrace_lexical.npz"
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    if change == "keys":
        arrays["unknown"] = np.array([1])
    elif change.startswith("rare"):
        arrays["rare"] = np.array([True]) if change == "rare_shape" else np.array([0, 1, 0])
    elif change.startswith("common"):
        arrays["common"] = {
            "common_shape": np.array([[3, 5]], dtype="uint64"),
            "common_dtype": np.array([3, 5], dtype="int64"),
            "common_empty": np.array([], dtype="uint64"),
            "common_order": np.array([5, 3], dtype="uint64"),
        }[change]
    else:
        arrays["groups"] = {
            "groups_shape": np.array([0]),
            "groups_dtype": np.array([0.0, 0.0, 2.0]),
            "groups_negative": np.array([0, 1, -1]),
            "groups_range": np.array([0, 0, 3]),
            "groups_idempotence": np.array([1, 0, 2]),
        }[change]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="FreshRace"):
        adapter.validate_freshrace_assets(*fresh, 3)


@pytest.mark.parametrize("change", ["reference_hash", "semantics", "vocabulary"])
def test_fresh_tables_do_not_qualify_a_different_tokenizer(fresh, change):
    source, tokenizer, reference, info = fresh
    if change == "reference_hash":
        info["tokenizer_sha256"] = "0" * 64
    elif change == "semantics":
        payload = json.loads((tokenizer / "tokenizer.json").read_text())
        payload["model"]["vocab"]["a"], payload["model"]["vocab"]["b"] = 1, 0
        (tokenizer / "tokenizer.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"tokenizer|vocabulary"):
        adapter.validate_freshrace_assets(source, tokenizer, reference, info, 4 if change == "vocabulary" else 3)


def _provenance():
    identifier = "candidate-015"
    proof = {
        "submission": {
            "id": identifier,
            "watermark_class": adapter.FRESHRACE_SUBMISSIONS[identifier],
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
        "freshrace_assets": {},
        "sampling_after_processor": True,
        "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
        "key": 45123,
        "vocab_size": 3,
    }
    return generation, proof


def test_fresh_provenance_does_not_require_the_different_wordnet_asset_family():
    generation, proof = _provenance()
    adapter.validate_provenance(generation, proof)


@pytest.mark.parametrize("bad_proof", [None, [], "frozen"])
def test_fresh_provenance_requires_its_own_asset_record(bad_proof):
    generation, proof = _provenance()
    generation["freshrace_assets"] = bad_proof
    with pytest.raises(ValueError, match=r"FreshRace.*asset provenance"):
        adapter.validate_provenance(generation, proof)
