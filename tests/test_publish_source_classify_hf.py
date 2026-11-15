"""Local staging checks for the source-classifier Hub publisher."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import publish_source_classify_hf as publish  # noqa: E402


def test_hub_ids_match_the_runtime_loader() -> None:
    # The workflow runs this script with --no-project, so it cannot import the package constants.
    from remove_ai_watermarks.source_classify import CONFIG_FILE, MODEL_FILE, WEIGHTS_REPO

    assert publish.HUB_REPO == WEIGHTS_REPO
    assert publish.CONFIG_FILE == CONFIG_FILE
    assert publish.MODEL_FILE == MODEL_FILE
    assert "wiltodelta/raiw-photo-classify" in publish.RETIRED_HUB_REPOS


def test_stage_release_copies_public_artifacts_and_synthesizes_config(tmp_path: Path) -> None:
    from remove_ai_watermarks.source_classify import (
        ALL_PROVIDER_CLASSES,
        FEATURE_DIM_V4,
        MODEL_FILE,
        MODEL_SCHEMA,
        MODEL_SHA256,
    )

    source = tmp_path / "source"
    source.mkdir()
    (source / publish.MODEL_FILE).write_bytes(b"weights")
    destination = tmp_path / "stage"
    publish.stage_release(destination, source)
    assert sorted(path.name for path in destination.iterdir()) == sorted(publish.PUBLIC_FILES)

    config = json.loads((destination / publish.CONFIG_FILE).read_text(encoding="utf-8"))
    assert config["model_type"] == "raiw-source-classify"
    assert config["schema"] == MODEL_SCHEMA
    assert config["artifact_file"] == MODEL_FILE
    assert config["artifact_sha256"] == MODEL_SHA256
    assert config["feature_dimension"] == FEATURE_DIM_V4
    assert config["providers"] == list(ALL_PROVIDER_CLASSES)
    assert config["abstain_label"] == "unknown"


def test_stage_release_requires_the_model(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=re.escape(publish.MODEL_FILE)):
        publish.stage_release(tmp_path / "stage", tmp_path)


def test_card_matches_the_metrics_manifest() -> None:
    from remove_ai_watermarks.source_classify import MODEL_FILE, MODEL_SCHEMA, MODEL_SHA256

    metrics = json.loads((publish.CARD_DIR / "metrics.json").read_text(encoding="utf-8"))
    card = (publish.CARD_DIR / "README.md").read_text(encoding="utf-8")
    assert metrics["schema"] == MODEL_SCHEMA
    assert metrics["artifact_file"] == MODEL_FILE
    assert metrics["artifact_sha256"] == MODEL_SHA256
    assert MODEL_FILE in card
    assert metrics["artifact_sha256"] in card


def test_publish_uploads_allowlist_and_deletes_retired_hub_repos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import types

    import huggingface_hub

    created: list[tuple[str, str, bool, bool]] = []
    uploads: list[dict[str, object]] = []
    deleted: list[tuple[str, str, bool]] = []

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            assert token == "tok"

        def create_repo(self, repo_id: str, *, repo_type: str, exist_ok: bool, private: bool) -> None:
            created.append((repo_id, repo_type, exist_ok, private))

        def upload_folder(self, **kwargs: object) -> object:
            uploads.append(kwargs)
            return types.SimpleNamespace(oid="abc123")

        def delete_repo(self, repo_id: str, *, repo_type: str, missing_ok: bool) -> None:
            deleted.append((repo_id, repo_type, missing_ok))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    rev = publish.publish(tmp_path, token="tok", message="msg")
    assert rev == "abc123"
    assert created == [(publish.HUB_REPO, "model", True, False)]
    assert len(uploads) == 1
    assert uploads[0]["allow_patterns"] == list(publish.PUBLIC_FILES)
    assert ("wiltodelta/raiw-photo-classify", "model", True) in deleted


def test_stage_release_includes_standalone_scripts_and_colab_notebook() -> None:
    assert {"infer.py", "benchmark.py", "demo.ipynb"} <= set(publish.TRACKED_CARD_FILES)
    assert all((publish.CARD_DIR / name).is_file() for name in publish.TRACKED_CARD_FILES)


def _load_module(name: str, path: Path) -> object:
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_standalone_infer_script_inspects_weights_and_classifies_images(tmp_path: Path) -> None:
    from typing import Any

    import numpy as np
    from PIL import Image

    sc_mod: Any = _load_module("raiw_test_source_classify", Path(__file__).with_name("test_source_classify.py"))
    infer_mod: Any = _load_module("raiw_hf_infer", publish.CARD_DIR / "infer.py")

    weights = sc_mod._write_synthetic_v5_npz(
        tmp_path / "synthetic_v5.npz",
        det_winner="openai",
        attr_winner="openai",
        det_score=-2.0,
        det_threshold=2.0,
    )
    summary = infer_mod.inspect_weights(weights)
    assert summary["schema"] == "multi-provider-source-v5"
    assert summary["feature_dim"] == 512
    assert len(summary["providers"]) == 10
    assert summary["abstain_label"] == "unknown"
    assert summary["watermark_truth"] == "unknown"
    assert "openai" in summary["version_subfamilies"]

    sample_img = tmp_path / "gradient.png"
    y, x = np.ogrid[:320, :320]
    arr = np.stack(
        (
            (x * 3 + y * 5) % 256,
            np.broadcast_to((x * 7) % 256, (320, 320)),
            np.broadcast_to((y * 11) % 256, (320, 320)),
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(sample_img)

    exit_inspect = infer_mod.main(["--inspect-weights", "--weights", str(weights), "--json"])
    assert exit_inspect == 0

    rows = infer_mod.classify_targets([sample_img], weights_path=weights)
    assert len(rows) == 1
    assert rows[0]["label"] == "unknown"
    assert rows[0]["reason"] == "abstained"
    assert rows[0]["watermark_truth"] == "unknown"


def test_demo_notebook_is_valid_nbformat_and_compiles_cleanly() -> None:
    nb_path = publish.CARD_DIR / "demo.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list)
    assert len(nb["cells"]) >= 4

    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) >= 3
    for idx, cell in enumerate(code_cells):
        assert cell.get("outputs") == []
        source_lines = [line for line in cell["source"] if not line.lstrip().startswith(("%", "!"))]
        compile("".join(source_lines), f"demo.ipynb#cell-{idx}", "exec")


def test_standalone_benchmark_script_summarizes_and_formats_metrics() -> None:
    from typing import Any

    bench_mod: Any = _load_module("raiw_hf_benchmark", publish.CARD_DIR / "benchmark.py")
    samples = [
        bench_mod.SampleItem(
            sha256="a" * 64,
            category="ui_screenshot",
            expected_provider=None,
            container_format="PNG",
            width=1024,
            height=768,
            raw_bytes=b"",
        ),
        bench_mod.SampleItem(
            sha256="b" * 64,
            category="ai_flux",
            expected_provider="flux",
            container_format="JPEG",
            width=1024,
            height=1024,
            raw_bytes=b"",
        ),
    ]
    preds = [
        {"pred_ai": False, "pred_provider": None},
        {"pred_ai": True, "pred_provider": "flux"},
    ]
    summary = bench_mod._summarize_predictions(
        model_id="wiltodelta/raiw-source-classify",
        revision="deadbeef",
        runtime="NumPy + OpenCV (CPU)",
        weights_mb=15.52,
        license_id="apache-2.0",
        median_cpu_ms=500.0,
        samples=samples,
        preds=preds,
    )
    assert summary["non_ai_fpr_pct"] == 0.0
    assert summary["ai_recall_pct"] == 100.0
    assert summary["precision_pct"] == 100.0
    assert summary["provider_attribution_pct"] == 100.0
    md = bench_mod.format_markdown_table([summary])
    assert "`wiltodelta/raiw-source-classify`" in md
    assert "0.0% (0/1)" in md
