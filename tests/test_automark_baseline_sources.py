"""Declared AutoMark baselines need complete original-tree provenance."""

import hashlib
import json
import shutil

import pytest

from scripts import benchmark_automark_submission as adapter


@pytest.fixture
def baseline(tmp_path):
    reference = tmp_path / "reference"
    store = reference / "agent-submissions"
    store.mkdir(parents=True)
    original = reference / "lm-wm-library/base"
    original.mkdir(parents=True)
    (original / "src").mkdir()
    (original / "src/module.py").write_bytes(b"code")
    (original / "config.yaml").write_bytes(b"config")
    source = tmp_path / "materialized"
    shutil.copytree(original, source)
    hashes = {str(p.relative_to(original)): adapter.sha(p) for p in original.rglob("*") if p.is_file()}
    row = {
        "id": "baseline-AAR-context3",
        "kind": "baseline",
        "source_project": "../lm-wm-library/base",
        "configuration_path": "config.yaml",
        "configuration_sha256": hashes["config.yaml"],
        "source_tree_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "watermark_class": "AAR",
    }
    entry = {"job_id": row["id"], "kind": "baseline", "source_project": row["source_project"], "config": "config.yaml"}
    (store / "manifest.json").write_text(json.dumps({"submissions": [entry]}))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "manifest_sha256": adapter.sha(store / "manifest.json"),
                "reference_revision": "pinned",
                "submissions": [row],
            }
        )
    )
    return store, original, source, inventory


def test_complete_original_and_materialized_baseline_trees_are_verified(baseline):
    store, _original, source, inventory = baseline
    proof = adapter.verify_submission(store, source, inventory, "baseline-AAR-context3")
    assert proof["reference_revision"] == "pinned"
    assert set(proof["files_sha256"]) == {"config.yaml", "src/module.py"}
    assert proof["submission"]["kind"] == "baseline"


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "original",
        "both_trees",
        "extra",
        "missing",
        "config_hash",
        "config_declaration",
        "project_declaration",
        "tree_anchor",
        "manifest",
        "source_link",
        "original_link",
        "source_root_link",
        "original_root_link",
        "empty_dir_link",
        "escape",
        "kind",
        "duplicate_entry",
        "missing_entry",
    ],
)
def test_changed_or_undeclared_baseline_material_is_rejected(baseline, tmp_path, change):
    store, original, source, inventory = baseline
    catalog = json.loads(inventory.read_text())
    row = catalog["submissions"][0]
    if change in {"source", "both_trees"}:
        (source / "src/module.py").write_bytes(b"changed")
    if change in {"original", "both_trees"}:
        (original / "src/module.py").write_bytes(b"changed")
    if change == "extra":
        (source / "extra.py").write_bytes(b"extra")
    elif change == "missing":
        (source / "src/module.py").unlink()
    elif change == "config_hash":
        row["configuration_sha256"] = "0" * 64
    elif change == "config_declaration":
        row.update(configuration_path="src/module.py", configuration_sha256=adapter.sha(source / "src/module.py"))
    elif change == "project_declaration":
        shutil.copytree(original, store.parent / "other")
        row["source_project"] = "../other"
    elif change == "tree_anchor":
        row["source_tree_sha256"] = "0" * 64
    elif change == "manifest":
        with (store / "manifest.json").open("a") as stream:
            stream.write(" ")
    elif change in {"source_link", "original_link"}:
        target = tmp_path / "module-target.py"
        target.write_bytes(b"code")
        path = (source if change == "source_link" else original) / "src/module.py"
        path.unlink()
        path.symlink_to(target)
    elif change in {"source_root_link", "original_root_link"}:
        path = source if change == "source_root_link" else original
        real = path.with_name(path.name + "-real")
        path.rename(real)
        path.symlink_to(real, target_is_directory=True)
    elif change == "empty_dir_link":
        target = tmp_path / "empty"
        target.mkdir()
        (source / "extra-dir").symlink_to(target, target_is_directory=True)
    elif change == "escape":
        shutil.copytree(original, tmp_path / "outside")
        manifest = json.loads((store / "manifest.json").read_text())
        row["source_project"] = manifest["submissions"][0]["source_project"] = "../../outside"
        (store / "manifest.json").write_text(json.dumps(manifest))
        catalog["manifest_sha256"] = adapter.sha(store / "manifest.json")
    elif change == "kind":
        row["kind"] = "candidate"
    elif change in {"duplicate_entry", "missing_entry"}:
        manifest = json.loads((store / "manifest.json").read_text())
        manifest["submissions"] = manifest["submissions"] * 2 if change == "duplicate_entry" else []
        (store / "manifest.json").write_text(json.dumps(manifest))
        catalog["manifest_sha256"] = adapter.sha(store / "manifest.json")
    inventory.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match=r"(?i)baseline|submission"):
        adapter.verify_submission(store, source, inventory, "baseline-AAR-context3")


@pytest.mark.parametrize(
    ("identifier", "class_name"), [("baseline-AAR-context3", "AAR"), ("baseline-SynthID-context3", "SynthID")]
)
def test_baseline_generation_requires_its_tree_anchor_and_exact_class(baseline, identifier, class_name):
    store, _original, source, inventory = baseline
    proof = adapter.verify_submission(store, source, inventory, "baseline-AAR-context3")
    proof["submission"].update(id=identifier, watermark_class=class_name)
    row = proof["submission"]
    generation = {
        "submission_id": identifier,
        "reference_revision": "pinned",
        "source_tree_sha256": row["source_tree_sha256"],
        "configuration_sha256": row["configuration_sha256"],
        "sampling_after_processor": True,
        "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
        "key": 45123,
        "vocab_size": 151936,
    }
    adapter.validate_provenance(generation, proof)
    for change in ("missing_tree", "wrong_tree", "map_only", "both_proofs", "wrong_class"):
        bad = dict(generation)
        other = {**proof, "submission": dict(row)}
        if change == "missing_tree":
            bad.pop("source_tree_sha256")
        elif change == "wrong_tree":
            bad["source_tree_sha256"] = "changed"
        elif change == "map_only":
            bad["source_map_sha256"] = bad.pop("source_tree_sha256")
        elif change == "both_proofs":
            bad["source_map_sha256"] = bad["source_tree_sha256"]
        else:
            other["submission"]["watermark_class"] = "WrongClass"
        with pytest.raises(ValueError, match=r"generation|Unsupported exact"):
            adapter.validate_provenance(bad, other)


def test_textseal_baseline_is_not_enabled_without_its_entropy_pipeline(baseline):
    store, _original, source, inventory = baseline
    proof = adapter.verify_submission(store, source, inventory, "baseline-AAR-context3")
    proof["submission"].update(id="baseline-TextSeal", watermark_class="TextSeal")
    row = proof["submission"]
    generation = {
        "submission_id": row["id"],
        "reference_revision": "pinned",
        "source_tree_sha256": row["source_tree_sha256"],
        "configuration_sha256": row["configuration_sha256"],
        "sampling_after_processor": True,
        "sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
        "key": 45123,
        "vocab_size": 151936,
    }
    with pytest.raises(ValueError, match="Unsupported exact submission/class pair"):
        adapter.validate_provenance(generation, proof)
