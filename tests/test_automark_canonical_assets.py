"""Canonical states must not run with a foreign tokenizer or identity fallback."""

import json

import numpy as np
import pytest

from scripts import benchmark_automark_submission as adapter

MODES = ("space", "lowsp", "lower", "alnum", "stem", "sense")


@pytest.fixture
def canonical(tmp_path):
    source = tmp_path / "source"
    root = source / "src/lm_wm_tools/watermarks/assets"
    root.mkdir(parents=True)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    payload = {
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, "c": 3}, "merges": ["a b"]},
        "added_tokens": [],
    }
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(payload))
    (tokenizer / "tokenizer.json").write_text(json.dumps(payload))
    arrays = {mode: np.array([0, 1, 1, 3], dtype="int32") for mode in MODES}
    arrays["sense"] = np.array([0, 0, 2, 2], dtype="int32")
    arrays["vocab_size"] = np.array(4, dtype="int64")
    path = root / "canon_maps.npz"
    np.savez(path, **arrays)
    return source, tokenizer, reference, {"tokenizer_sha256": adapter.sha(reference)}, path


@pytest.mark.parametrize(("mode", "changed"), [("alnum", 1), ("sense", 2)])
def test_selected_map_and_declared_reference_are_bound(canonical, mode, changed):
    *args, path = canonical
    proof = adapter.validate_canonical_assets(*args, 4, mode=mode)
    assert proof["mode"] == mode
    assert proof["changed_token_ids"] == changed
    assert proof["vocab_size"] == 4
    assert proof["canonical_tokenizer_equal"] is True
    assert proof["embedded_tokenizer_metadata"] is None
    assert proof["reference_tokenizer_sha256"] == adapter.sha(args[2])
    assert proof["tokenizer_sha256"] == adapter.sha(args[1] / "tokenizer.json")
    assert proof["canonical_asset_sha256"] == adapter.sha(path)


@pytest.mark.parametrize("mode", ["alnum", "sense"])
@pytest.mark.parametrize(
    "fault",
    [
        "keys",
        "missing_mode",
        "missing_size",
        "size_shape",
        "size_float",
        "size_bool",
        "size_value",
        "map_shape",
        "map_float",
        "map_bool",
        "negative",
        "outside",
        "nonidempotent",
        "not_minimal",
        "identity",
    ],
)
def test_malformed_or_inactive_canonical_maps_fail(canonical, mode, fault):
    *args, path = canonical
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if fault == "keys":
        arrays["unknown"] = np.array(1)
    elif fault == "missing_mode":
        arrays.pop(mode)
    elif fault == "missing_size":
        arrays.pop("vocab_size")
    elif fault == "size_shape":
        arrays["vocab_size"] = np.array([4])
    elif fault == "size_float":
        arrays["vocab_size"] = np.array(4.0)
    elif fault == "size_bool":
        arrays["vocab_size"] = np.array(True)
    elif fault == "size_value":
        arrays["vocab_size"] = np.array(5)
    elif fault == "map_shape":
        arrays[mode] = arrays[mode].reshape(2, 2)
    elif fault == "map_float":
        arrays[mode] = arrays[mode].astype(float)
    elif fault == "map_bool":
        arrays[mode] = arrays[mode].astype(bool)
    elif fault == "negative":
        arrays[mode][3] = -1
    elif fault == "outside":
        arrays[mode][3] = 4
    elif fault == "nonidempotent":
        arrays[mode] = np.array([0, 0, 1, 3])
    elif fault == "not_minimal":
        arrays[mode] = np.array([0, 2, 2, 3])
    elif fault == "identity":
        arrays[mode] = np.arange(4)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=r"[Cc]anonical"):
        adapter.validate_canonical_assets(*args, 4, mode=mode)


@pytest.mark.parametrize("fault", ["missing", "corrupt", "symlink"])
def test_unavailable_assets_do_not_degrade_to_exact_indices(canonical, fault):
    *args, path = canonical
    if fault == "corrupt":
        path.write_bytes(b"not an archive")
    elif fault == "missing":
        path.unlink()
    else:
        target = path.with_name("elsewhere.npz")
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises(ValueError, match=r"[Cc]anonical"):
        adapter.validate_canonical_assets(*args, 4, mode="alnum")


@pytest.mark.parametrize("fault", ["hash", "decoder", "vocab", "merges", "special"])
def test_foreign_reference_or_tokenizer_is_rejected(canonical, fault):
    source, tokenizer, reference, proof, _path = canonical
    if fault == "hash":
        proof["tokenizer_sha256"] = "0" * 64
    else:
        path = tokenizer / "tokenizer.json"
        payload = json.loads(path.read_text())
        if fault == "decoder":
            payload["decoder"] = {"type": "ByteLevel"}
        elif fault == "vocab":
            payload["model"]["vocab"]["a"], payload["model"]["vocab"]["b"] = 1, 0
        elif fault == "merges":
            payload["model"]["merges"] = ["b a"]
        else:
            payload["added_tokens"] = [{"id": 4, "content": "<stop>"}]
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"[Tt]okenizer"):
        adapter.validate_canonical_assets(source, tokenizer, reference, proof, 4, mode="alnum")


@pytest.mark.parametrize("size", [True, 4.0, -1, 0, 3])
def test_invalid_or_mismatched_model_vocabulary_fails(canonical, size):
    *args, _path = canonical
    with pytest.raises(ValueError, match=r"[Cc]anonical"):
        adapter.validate_canonical_assets(*args, size, mode="alnum")


@pytest.mark.parametrize("mode", ["none", "lower", None, []])
def test_undeclared_modes_fail_before_detection(canonical, mode):
    *args, _path = canonical
    with pytest.raises(ValueError, match=r"[Cc]anonical"):
        adapter.validate_canonical_assets(*args, 4, mode=mode)


def test_noncontiguous_reference_ids_fail(canonical):
    source, tokenizer, reference, proof, _path = canonical
    payload = json.loads(reference.read_text())
    payload["model"]["vocab"]["c"] = 5
    reference.write_text(json.dumps(payload))
    (tokenizer / "tokenizer.json").write_text(json.dumps(payload))
    proof["tokenizer_sha256"] = adapter.sha(reference)
    with pytest.raises(ValueError, match=r"[Cc]anonical"):
        adapter.validate_canonical_assets(source, tokenizer, reference, proof, 4, mode="alnum")


def test_merge_serialization_normalization_is_still_allowed(canonical):
    *args, _path = canonical
    path = args[1] / "tokenizer.json"
    payload = json.loads(path.read_text())
    payload["model"]["merges"] = [["a", "b"]]
    path.write_text(json.dumps(payload))
    assert adapter.validate_canonical_assets(*args, 4, mode="alnum")["canonical_tokenizer_equal"] is True


def test_object_array_is_rejected_without_unpickling(canonical):
    from pathlib import Path

    *args, path = canonical
    marker = path.with_name("unpickled-marker")

    class Probe:
        def __reduce__(self):
            return Path.touch, (marker,)

    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["alnum"] = np.array([Probe()] * 4, dtype=object)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=r"[Cc]anonical"):
        adapter.validate_canonical_assets(*args, 4, mode="alnum")
    assert not marker.exists()


@pytest.mark.parametrize(("mode", "accepted"), [("alnum", True), ("sense", False)])
def test_pre_sense_published_archive_schema(canonical, mode, accepted):
    *args, path = canonical
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != "sense"}
    np.savez(path, **arrays)
    if accepted:
        assert adapter.validate_canonical_assets(*args, 4, mode=mode)["changed_token_ids"] == 1
    else:
        with pytest.raises(ValueError, match=r"[Cc]anonical"):
            adapter.validate_canonical_assets(*args, 4, mode=mode)
