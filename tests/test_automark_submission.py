"""Exact source-state and frozen-generation checks for AutoMark submissions."""

import json
import sys
from types import SimpleNamespace

import pytest

from scripts import benchmark_automark_submission as adapter


@pytest.fixture
def submission(tmp_path):
    store = tmp_path / "store"
    source = tmp_path / "source"
    (store / "files").mkdir(parents=True)
    (store / "objects").mkdir()
    source.mkdir()
    mapping = {}
    for name, content in {"config.yaml": b"config", "src/module.py": b"code"}.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        digest = adapter.sha(path)
        mapping[name] = digest
        (store / "objects" / digest).write_bytes(content)
    (store / "manifest.json").write_text(json.dumps({"submissions": [{"job_id": "candidate-001", "file_count": 2}]}))
    mapping_path = store / "files/candidate-001.json"
    mapping_path.write_text(json.dumps(mapping))
    row = {
        "id": "candidate-001",
        "watermark_class": "CascadeCopula",
        "configuration_path": "config.yaml",
        "configuration_sha256": mapping["config.yaml"],
        "source_map_sha256": adapter.sha(mapping_path),
    }
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "manifest_sha256": adapter.sha(store / "manifest.json"),
                "reference_revision": "frozen-reference",
                "submissions": [row],
            }
        )
    )
    return store, source, inventory


def test_verification_checks_every_object_and_source_file(submission):
    proof = adapter.verify_submission(*submission, "candidate-001")
    assert proof["reference_revision"] == "frozen-reference"
    assert set(proof["files_sha256"]) == {"config.yaml", "src/module.py"}


@pytest.mark.parametrize("target", ["source", "object", "mapping", "manifest", "extra", "missing", "config_hash"])
def test_changed_or_extra_material_is_not_executed(submission, target):
    store, source, inventory = submission
    if target == "source":
        (source / "src/module.py").write_bytes(b"changed")
    elif target == "object":
        (store / "objects" / adapter.sha(source / "src/module.py")).write_bytes(b"changed")
    elif target == "mapping":
        with (store / "files/candidate-001.json").open("a") as stream:
            stream.write(" ")
    elif target == "manifest":
        with (store / "manifest.json").open("a") as stream:
            stream.write(" ")
    elif target == "extra":
        (source / "extra.py").write_bytes(b"code")
    elif target == "missing":
        (source / "src/module.py").unlink()
    else:
        payload = json.loads(inventory.read_text())
        payload["submissions"][0]["configuration_sha256"] = "0" * 64
        inventory.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"hash|missing|extra"):
        adapter.verify_submission(*submission, "candidate-001")


@pytest.fixture
def provenance(submission):
    proof = adapter.verify_submission(*submission, "candidate-001")
    row = proof["submission"]
    generation = {
        "submission_id": row["id"],
        "reference_revision": proof["reference_revision"],
        "source_map_sha256": row["source_map_sha256"],
        "configuration_sha256": row["configuration_sha256"],
        "sampling_after_processor": True,
        "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
        "key": 45123,
        "vocab_size": 151936,
    }
    return generation, proof


def test_exact_generation_provenance_is_accepted(provenance):
    adapter.validate_provenance(*provenance)


@pytest.mark.parametrize("identifier", [f"candidate-{i:03}" for i in range(37, 41)])
def test_exact_tallymark_source_states_need_no_lexical_assets(provenance, identifier):
    generation, proof = provenance
    generation["submission_id"] = identifier
    proof["submission"].update(
        id=identifier,
        watermark_class="lm_wm_tools.watermarks.logits_processors.tallymark.TallyMarkWatermark",
    )
    adapter.validate_provenance(generation, proof)
    proof["submission"]["watermark_class"] = "TallyMarkWatermark"
    with pytest.raises(ValueError, match="Unsupported exact submission/class pair"):
        adapter.validate_provenance(generation, proof)


@pytest.mark.parametrize(
    ("identifier", "module", "class_name"),
    [
        ("candidate-041", "choirmark", "ChoirMarkWatermark"),
        ("candidate-042", "canonmark", "CanonMarkWatermark"),
        ("candidate-043", "laddermark", "LadderMarkWatermark"),
        ("candidate-044", "preludemark", "PreludeMarkWatermark"),
        ("candidate-045", "consortmark", "ConsortMarkWatermark"),
        ("candidate-046", "sestetmark", "SestetMarkWatermark"),
        ("candidate-047", "motetmark", "MotetMarkWatermark"),
        ("candidate-048", "codamark", "CodaMarkWatermark"),
    ],
)
def test_exact_unweighted_multivoice_states_need_no_lexical_assets(provenance, identifier, module, class_name):
    generation, proof = provenance
    generation["submission_id"] = identifier
    proof["submission"].update(
        id=identifier, watermark_class=f"lm_wm_tools.watermarks.logits_processors.{module}.{class_name}"
    )
    adapter.validate_provenance(generation, proof)
    proof["submission"]["watermark_class"] = class_name
    with pytest.raises(ValueError, match="Unsupported exact submission/class pair"):
        adapter.validate_provenance(generation, proof)


@pytest.mark.parametrize("identifier", ["candidate-026", "candidate-049", "candidate-050"])
def test_weighted_source_states_are_not_enabled_by_plain_tallymark_support(provenance, identifier):
    generation, proof = provenance
    generation["submission_id"] = identifier
    proof["submission"].update(id=identifier, watermark_class="unsupported-weighted-method")
    with pytest.raises(ValueError, match="Unsupported exact submission/class pair"):
        adapter.validate_provenance(generation, proof)


@pytest.mark.parametrize("signals", [[], ["entropies"], ["surprisals"], ["row_weights"]])
def test_plain_replay_never_omits_required_author_signals(signals):
    config = SimpleNamespace(required_auxiliary_signals=lambda: signals)
    if signals:
        with pytest.raises(ValueError, match="requires auxiliary signals"):
            adapter.require_plain_detection(config)
    else:
        adapter.require_plain_detection(config)


def test_replay_rejects_required_signals_before_constructing_detector(tmp_path, provenance, monkeypatch):
    from scripts import benchmark_alimark_reference

    generation, proof = provenance
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    generation["tokenizer_files_sha256"] = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (tokenizer / name).write_text(json.dumps({"vocab_size": generation["vocab_size"]}))
        generation["tokenizer_files_sha256"][name] = adapter.sha(tokenizer / name)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "watermark.json").write_text(json.dumps({"device": "cpu", "seed": generation["key"]}))
    generation["watermark_config_sha256"] = adapter.sha(inputs / "watermark.json")
    source = tmp_path / "replay-source"
    source.mkdir()
    (source / proof["submission"]["configuration_path"]).write_text("{}")
    monkeypatch.setattr(adapter, "verify_submission", lambda *args: proof)
    monkeypatch.setattr(
        benchmark_alimark_reference,
        "load_inputs",
        lambda *args: ({"source": "source", "baseline": "baseline"}, {"generation": generation}),
    )

    def unexpected_detector(*args):
        pytest.fail("A detector must not run without its required signals")

    config = SimpleNamespace(required_auxiliary_signals=lambda: ["entropies"], get_detector=unexpected_detector)
    author = SimpleNamespace(WatermarkConfig=SimpleNamespace(from_payload=lambda payload: config))
    real_import = adapter.importlib.import_module
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda name: author if name == "lm_wm_tools.configs.main_config" else real_import(name),
    )
    args = SimpleNamespace(
        input_dir=inputs,
        output_dir=tmp_path / "output",
        source_store=tmp_path,
        submission_dir=source,
        inventory=tmp_path / "inventory.json",
        submission_id="candidate-001",
        tokenizer_dir=tokenizer,
    )
    with pytest.raises(ValueError, match="requires auxiliary signals"):
        adapter.run(args)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("source_map_sha256", "changed"),
        ("configuration_sha256", "changed"),
        ("reference_revision", "changed"),
        ("sampling_after_processor", False),
        ("sampling_after_processor", 1),
        ("key", True),
        ("key", -1),
        ("vocab_size", True),
        ("vocab_size", 2**20),
    ],
)
def test_provenance_rejects_drift_and_truthy_flags(provenance, key, value):
    generation, proof = provenance
    generation[key] = value
    with pytest.raises(ValueError, match=r"generation|Research|vocabulary|Sampling"):
        adapter.validate_provenance(generation, proof)


def test_sampler_temperature_is_not_a_boolean_or_a_different_distribution(provenance):
    generation, proof = provenance
    for value in (True, 0.7):
        generation["sampling"]["temperature"] = value
        with pytest.raises(ValueError, match="sampling"):
            adapter.validate_provenance(generation, proof)


def test_a_different_submission_requires_its_own_asset_provenance(provenance):
    generation, proof = provenance
    proof["submission"]["id"] = "candidate-002"
    generation["submission_id"] = "candidate-002"
    with pytest.raises(ValueError, match="asset provenance"):
        adapter.validate_provenance(generation, proof)


@pytest.mark.parametrize("change", ["eos", "roundtrip", "unsupported", "short"])
def test_generation_completion_and_scored_length_are_required(change):
    generation = {
        name + "_generation": {
            "ended_with_eos": True,
            "token_roundtrip_exact": True,
            "unsupported_output_ids": [],
            "text": name,
            "text_ids": list(range(64)),
        }
        for name in ("source", "baseline")
    }
    scores = {name: {"raw": {"mask": [True] * 64, "tokens": list(range(64))}} for name in ("source", "baseline")}
    assert adapter.generation_is_qualified(generation, scores, {"source": "source", "baseline": "baseline"})
    if change == "eos":
        generation["source_generation"]["ended_with_eos"] = 1
    elif change == "roundtrip":
        generation["baseline_generation"]["token_roundtrip_exact"] = False
    elif change == "unsupported":
        generation["source_generation"]["unsupported_output_ids"] = [151935]
    else:
        scores["source"]["raw"]["mask"] = [True] * 63 + [False]
    assert not adapter.generation_is_qualified(generation, scores, {"source": "source", "baseline": "baseline"})


def test_imported_reference_namespaces_are_never_silently_reused(monkeypatch):
    monkeypatch.setitem(sys.modules, "lm_wm_tools.unrelated", SimpleNamespace())
    with pytest.raises(RuntimeError, match="fresh process"):
        adapter.require_fresh_namespace()


def test_configuration_must_belong_to_the_verified_source_map(submission):
    store, source, inventory = submission
    outside = source.parent / "outside.yaml"
    outside.write_bytes((source / "config.yaml").read_bytes())
    data = json.loads(inventory.read_text())
    data["submissions"][0]["configuration_path"] = "../outside.yaml"
    inventory.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="configuration"):
        adapter.verify_submission(store, source, inventory, "candidate-001")


def test_file_count_is_checked_after_a_valid_manifest_hash(submission):
    store, source, inventory = submission
    data = json.loads((store / "manifest.json").read_text())
    data["submissions"][0]["file_count"] = 3
    (store / "manifest.json").write_text(json.dumps(data))
    data = json.loads(inventory.read_text())
    data["manifest_sha256"] = adapter.sha(store / "manifest.json")
    inventory.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="file count"):
        adapter.verify_submission(store, source, inventory, "candidate-001")


def test_undeclared_sampler_options_are_not_silently_accepted(provenance):
    generation, proof = provenance
    generation["sampling"]["min_p"] = 0.2
    with pytest.raises(ValueError, match="sampling"):
        adapter.validate_provenance(generation, proof)


def test_generation_flags_are_bound_to_the_actual_scored_text_and_tokens():
    generation = {
        name + "_generation": {
            "ended_with_eos": True,
            "token_roundtrip_exact": True,
            "unsupported_output_ids": [],
            "text": name,
            "text_ids": list(range(64)),
        }
        for name in ("source", "baseline")
    }
    scores = {name: {"raw": {"mask": [True] * 64, "tokens": list(range(64))}} for name in ("source", "baseline")}
    texts = {"source": "source", "baseline": "baseline"}
    assert adapter.generation_is_qualified(generation, scores, texts)
    generation["source_generation"]["text"] = "a different source"
    assert not adapter.generation_is_qualified(generation, scores, texts)
    generation["source_generation"]["text"] = "source"
    generation["baseline_generation"]["text_ids"] = list(range(65))
    assert not adapter.generation_is_qualified(generation, scores, texts)


def test_shared_json_conversion_preserves_the_archived_helper_name():
    from scripts import benchmark_automark_reference as selected

    assert selected.json_value({"scores": [float("nan"), 0.5]}) == {"scores": [None, 0.5]}
    assert selected._json_value is selected.json_value


@pytest.mark.parametrize(("retained", "expected"), [(0, "not_detected_at_reference_alpha"), (1, "partial_unverified")])
def test_a_negative_whole_score_cannot_qualify_retained_prose(retained, expected):
    generation = {
        name + "_generation": {
            "ended_with_eos": True,
            "token_roundtrip_exact": True,
            "unsupported_output_ids": [],
            "text": name,
            "text_ids": list(range(64)),
        }
        for name in ("source", "baseline")
    }
    scores = {
        name: {"pvalue": 0.001 if name == "source" else 1.0, "raw": {"mask": [True] * 64, "tokens": list(range(64))}}
        for name in ("source", "baseline", "after")
    }
    texts = {"source": "source", "baseline": "baseline", "after": "after"}
    assert adapter.assess_submission(scores, generation, texts, retained, 0.01) == expected
    generation["source_generation"]["ended_with_eos"] = False
    assert adapter.assess_submission(scores, generation, texts, retained, 0.01) == "unqualified_generation"
