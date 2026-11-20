"""Tests for `scripts/sync_ai_detector_leaderboard.py`."""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import sync_ai_detector_leaderboard as sync_lb  # noqa: E402


def test_tracked_leaderboard_matches_metrics_manifest_and_has_no_drift_numbers() -> None:
    metrics = json.loads((sync_lb.CARD_DIR / "metrics.json").read_text(encoding="utf-8"))
    lb = json.loads((sync_lb.LEADERBOARD_DIR / sync_lb.LEADERBOARD_JSON_REL).read_text(encoding="utf-8"))
    sync_lb.verify_leaderboard_parity(lb, metrics)

    from remove_ai_watermarks.source_classify import WEIGHTS_REVISION

    assert lb["results"][0]["revision"] == WEIGHTS_REVISION
    index_html = (sync_lb.LEADERBOARD_DIR / "index.html").read_text(encoding="utf-8")
    assert "259,420" not in index_html


def test_verify_leaderboard_parity_rejects_stale_raiw_metrics() -> None:
    metrics = json.loads((sync_lb.CARD_DIR / "metrics.json").read_text(encoding="utf-8"))
    lb = json.loads((sync_lb.LEADERBOARD_DIR / sync_lb.LEADERBOARD_JSON_REL).read_text(encoding="utf-8"))
    lb["results"][0]["arena_ai"]["recall_pct"] = 87.0

    with pytest.raises(ValueError, match=r"arena_ai\.recall_pct"):
        sync_lb.verify_leaderboard_parity(lb, metrics)


def test_extract_pinned_baselines_from_source_finds_all_nineteen_competitors() -> None:
    baselines = sync_lb._extract_pinned_baselines_from_source()
    assert len(baselines) == 20
    assert "abacusai/Smaug-AI-Image-Detector" in baselines
    assert "OwensLab/commfor-model-384" in baselines
    assert "Bombek1/ai-image-detector-siglip-dinov2" in baselines
    assert all(10 <= len(rev) <= 40 for rev in baselines.values())


def test_publish_dry_run_rolls_back_without_mutating_files_or_calling_hub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import huggingface_hub

    class ForbiddenApi:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("HfApi write client must not be constructed in --dry-run")

    lb_copy = tmp_path / "leaderboard"
    shutil.copytree(sync_lb.LEADERBOARD_DIR, lb_copy)
    before_bytes = (lb_copy / sync_lb.LEADERBOARD_JSON_REL).read_bytes()

    monkeypatch.setattr(sync_lb, "LEADERBOARD_DIR", lb_copy)
    monkeypatch.setattr(
        sync_lb,
        "query_raiw_hub_timestamps",
        lambda **_kw: ("2026-11-19T06:46:33+00:00", "2026-11-20T01:00:00+00:00", 1795070793.0),
    )
    monkeypatch.setattr(huggingface_hub, "HfApi", ForbiddenApi)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    rc = sync_lb.main(["--refresh-hub-dates", "--publish", "--dry-run"])
    assert rc == 0
    assert (lb_copy / sync_lb.LEADERBOARD_JSON_REL).read_bytes() == before_bytes


def test_publish_uploads_exact_space_allowlist_and_deletes_orphan_css(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import huggingface_hub

    created: list[tuple[str, str, str | None, bool]] = []
    uploads: list[dict[str, object]] = []

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            assert token == "secret-tok"

        def create_repo(
            self,
            repo_id: str,
            *,
            repo_type: str,
            space_sdk: str | None = None,
            exist_ok: bool,
            private: bool,
        ) -> None:
            assert exist_ok is True
            created.append((repo_id, repo_type, space_sdk, private))

        def upload_folder(self, **kwargs: object) -> object:
            uploads.append(kwargs)
            return types.SimpleNamespace(oid="deadbeef" * 5)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    stage = tmp_path / "stage"
    sync_lb.stage_leaderboard_space(stage)
    oid = sync_lb.publish_leaderboard_space(stage, token="secret-tok", message="sync v8")

    assert oid == "deadbeef" * 5
    assert created == [(sync_lb.LEADERBOARD_SPACE_REPO, "space", "static", False)]
    assert len(uploads) == 1
    assert uploads[0]["allow_patterns"] == list(sync_lb.SPACE_TRACKED_FILES)
    assert uploads[0]["delete_patterns"] == list(sync_lb.SPACE_RETIRED_FILES)


def test_check_external_arenas_flags_stale_owned_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    stale_lb = json.loads((sync_lb.LEADERBOARD_DIR / sync_lb.LEADERBOARD_JSON_REL).read_text(encoding="utf-8"))
    stale_lb["results"][0]["revision"] = "46b331cb8f0831e6ced6eb3a7f7b55db3f91867e"

    baselines = sync_lb._extract_pinned_baselines_from_source()

    class FakeApi:
        def repo_info(self, *, repo_id: str, repo_type: str) -> object:
            if repo_type == "dataset":
                assert repo_id == sync_lb.DFBENCH_DATASET_REPO
                return types.SimpleNamespace(sha=sync_lb.DFBENCH_DATASET_REVISION)
            assert repo_type == "model"
            return types.SimpleNamespace(sha=baselines[repo_id])

    def fake_get_text(url: str, *, accept: str = "") -> tuple[int, str]:
        del accept
        if "ai-image-detector-leaderboard/resolve/main/data/leaderboard.json" in url:
            return 200, json.dumps(stale_lb)
        if "huggingface.co/blog/" in url:
            return 200, "<html>multi-provider-source-v7.npz 4faef51eea48f819</html>"
        if "DFBench/Leaderboard-Image-2025/resolve/main/leaderboard.csv" in url:
            return 200, 'Rank,Model,Accuracy\n1,"Resemble.ai",85.0685\n'
        if "dataset_versions.md" in url:
            return 200, "| **0.1** | Feb 2026 |"
        if url.endswith("/issues/1/comments"):
            return 200, json.dumps([{"body": "Updated run for multi-provider-source-v8.npz"}])
        if "api.github.com/repos/AI-Detect-Arena" in url:
            return 200, json.dumps(
                {
                    "html_url": "https://github.com/AI-Detect-Arena/benchmark-dataset/issues/1",
                    "state": "open",
                    "comments": 1,
                    "body": "multi-provider-source-v7.npz",
                }
            )
        return 200, "<html>AI Detector Arena</html>"

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(sync_lb, "_http_get_text", fake_get_text)

    report = sync_lb.check_external_arenas()
    assert report["owned_surfaces_up_to_date"] is False
    assert set(report["owned_stale_surfaces"]) == {"hf_leaderboard_space", "hf_blog_article"}
    assert report["surfaces"]["ai_detector_arena"]["latest_dataset_version"] == "0.1"
    assert report["surfaces"]["ai_detector_arena"]["issue_matches_current_artifact"] is True
    assert report["surfaces"]["ai_detector_arena"]["issue_latest_artifact"] == "multi-provider-source-v8.npz"
    assert report["surfaces"]["dfbench_image_2025"]["entry_count"] == 1
    assert report["surfaces"]["dfbench_image_2025"]["dataset_matches_pinned_sha"] is True
    assert report["surfaces"]["upstream_hf_baselines"]["drifted_count"] == 0
    md = sync_lb.format_arena_markdown_summary(report)
    assert "STALE / ERROR" in md
    assert "NEEDS EDIT" in md
    assert "DFBench/Leaderboard-Image-2025" in md


def _make_stub_weights(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import test_source_classify as sc_test

    sc_test.install_stub_ort_if_missing(monkeypatch)
    return sc_test._write_synthetic_v7_npz(
        tmp_path / "synthetic_v7.npz",
        det_winner="openai",
        attr_winner="openai",
        det_score=3.0,
        det_threshold=1.0,
        syn_threshold=-1.0,
    )


def test_export_aidetectarena_submission_writes_csv_and_honors_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    weights = _make_stub_weights(monkeypatch, tmp_path)

    ds_dir = tmp_path / "arena"
    (ds_dir / "metadata").mkdir(parents=True)
    (ds_dir / "images").mkdir(parents=True)
    arr = np.zeros((320, 320, 3), dtype=np.uint8)
    arr[:, :, 0] = 120
    Image.fromarray(arr, mode="RGB").save(ds_dir / "images" / "one.png")
    (ds_dir / "metadata" / "images_metadata.csv").write_text(
        "id,filename,is_ai,generator,category\nai_001,images/one.png,true,Flux,art\n",
        encoding="utf-8",
    )

    dry_out = tmp_path / "dry_out"
    dry_summary = sync_lb.export_aidetectarena_submission(ds_dir, dry_out, weights_path=weights, dry_run=True)
    assert dry_summary["total_images"] == 1
    assert not dry_out.exists()

    real_out = tmp_path / "real_out"
    real_summary = sync_lb.export_aidetectarena_submission(ds_dir, real_out, weights_path=weights, dry_run=False)
    assert real_summary["total_images"] == 1
    csv_lines = (real_out / "predictions.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == "image_id,prediction,confidence"
    assert csv_lines[1].startswith("ai_001,ai,")
    assert (real_out / "issue_update.md").is_file()


def test_binary_roc_auc_and_openfake_provider_mapping() -> None:
    assert sync_lb._binary_roc_auc([False, False, True, True], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert sync_lb._binary_roc_auc([False, True, False, True], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert sync_lb._binary_roc_auc([True, True], [0.2, 0.9]) is None
    assert sync_lb._map_openfake_provider("flux.2-klein-9b") == "flux"
    assert sync_lb._map_openfake_provider("gpt-image-1.5") == "openai"
    assert sync_lb._map_openfake_provider("midjourney-7") == "midjourney"
    assert sync_lb._map_openfake_provider("z-image-turbo") is None


def test_export_dfbench_submission_supports_tiff_numeric_sort_and_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    weights = _make_stub_weights(monkeypatch, tmp_path)

    ds_dir = tmp_path / "dfbench" / "dataset"
    ds_dir.mkdir(parents=True)
    arr = np.full((320, 320, 3), 140, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(ds_dir / "10.tiff")
    Image.fromarray(arr, mode="RGB").save(ds_dir / "2.webp")

    dry_out = tmp_path / "df_dry"
    dry_res = sync_lb.export_dfbench_submission(ds_dir.parent, dry_out, weights_path=weights, dry_run=True)
    assert dry_res["total_images"] == 2
    assert not dry_out.exists()

    real_out = tmp_path / "df_real"
    real_res = sync_lb.export_dfbench_submission(ds_dir.parent, real_out, weights_path=weights, dry_run=False)
    assert real_res["by_extension"] == {"webp": 1, "tiff": 1}
    lines = (real_out / "wiltodelta_raiw-source-classify.csv").read_text(encoding="utf-8").splitlines()
    assert lines == ["filename,label", "2.webp,fake", "10.tiff,fake"]
