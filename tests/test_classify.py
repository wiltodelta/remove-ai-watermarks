"""Pixel classify gate: Model 1 then gated Model 2, no downloads, no identify hook."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from remove_ai_watermarks.classify import (
    MLP_THRESHOLD,
    PROVIDER_MARGIN,
    RIDGE_THRESHOLD,
    WEIGHTS_REPO,
    WEIGHTS_REVISION,
    classify_from_scores,
    classify_pixels,
    detector_level,
    label_for,
    provider_from_scores,
)
from remove_ai_watermarks.cli import main


def _scores(
    *,
    openai: float = 0.0,
    google: float = 0.0,
    tc260: float = 0.0,
    meta: float = 0.0,
    no_ai: float = 0.0,
) -> dict[str, float]:
    return {
        "openai": openai,
        "google": google,
        "tc260": tc260,
        "meta_muse_image": meta,
        "no_ai": no_ai,
    }


def test_hub_snapshot_is_the_freeze_revision() -> None:
    assert WEIGHTS_REPO == "wiltodelta/raiw-models"
    assert WEIGHTS_REVISION == "c0ac82b6f1ae9fc0b92c467562282e9422da6da6"


def test_shipped_operating_point_matches_the_runtime_defaults() -> None:
    payload = json.loads(Path("docs/photo-classify-hf/operating-point.json").read_text())
    assert payload["model1"]["mlp_threshold"] == MLP_THRESHOLD
    assert payload["model1"]["ridge_threshold"] == RIDGE_THRESHOLD
    assert payload["model2"]["margin"] == PROVIDER_MARGIN
    assert payload["model2"]["runs_only_after"] == "definitely"
    assert payload["model2"]["classes"] == ["openai", "google", "tc260", "muse-image", "no_ai"]
    assert payload["model2"]["class_kind"]["openai"] == "provider"
    assert payload["model2"]["class_kind"]["google"] == "provider"
    assert payload["model2"]["class_kind"]["tc260"] == "label-standard"
    assert payload["model2"]["class_kind"]["muse-image"] == "model"
    assert payload["model2"]["checkpoint_keys"]["muse-image"] == "meta_muse_image"


def test_definitely_is_the_and_of_ridge_and_mlp() -> None:
    assert detector_level(1.0, 10.0) == "definitely"
    assert detector_level(0.0, 10.0) == "possibly"
    assert detector_level(1.0, 0.0) == "possibly"
    assert detector_level(0.0, 0.0) == "likely_human"


def test_only_definitely_is_ai() -> None:
    assert label_for("definitely") == "ai"
    assert label_for("possibly") == "unknown"
    assert label_for("likely_human") == "human"


def test_provider_requires_margin_over_no_ai() -> None:
    assert provider_from_scores(_scores(openai=1.0, no_ai=0.5)) == "openai"
    assert provider_from_scores(_scores(openai=0.5, no_ai=0.4)) is None
    assert provider_from_scores(_scores(google=2.0, openai=1.5, no_ai=0.0)) == "google"
    assert provider_from_scores(_scores(meta=1.0, no_ai=0.0)) == "muse-image"
    assert provider_from_scores(_scores(tc260=1.0, no_ai=0.0)) == "tc260"


def test_definitely_plus_openai_is_ai_openai() -> None:
    result = classify_from_scores(1.0, 10.0, _scores(openai=1.0, no_ai=0.0))
    assert result.label == "ai"
    assert result.domain == "photo"
    assert result.detector == "definitely"
    assert result.provider == "openai"
    assert result.to_dict()["provider"] == "openai"


def test_definitely_without_provider_scores_stays_ai_with_no_provider() -> None:
    result = classify_from_scores(1.0, 10.0, None)
    assert result.label == "ai"
    assert result.provider is None


def test_definitely_plus_no_ai_head_has_no_provider() -> None:
    result = classify_from_scores(1.0, 10.0, _scores(no_ai=2.0, openai=1.0))
    assert result.label == "ai"
    assert result.provider is None


def test_possibly_does_not_run_provider() -> None:
    result = classify_from_scores(1.0, 0.0, _scores(openai=9.0, no_ai=0.0))
    assert result.label == "unknown"
    assert result.detector == "possibly"
    assert result.provider is None


def test_likely_human_does_not_run_provider() -> None:
    result = classify_from_scores(0.0, 0.0, _scores(openai=9.0, no_ai=0.0))
    assert result.label == "human"
    assert result.detector == "likely_human"
    assert result.provider is None


@pytest.mark.parametrize("caller", ["identify", "has_invisible_target"])
def test_provenance_paths_do_not_import_classify(tmp_path: Path, caller: str) -> None:
    path = tmp_path / "plain.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path)
    sys.modules.pop("remove_ai_watermarks.classify", None)
    from remove_ai_watermarks.identify import has_invisible_target, identify

    if caller == "identify":
        report = identify(path, check_visible=False, check_invisible=False)
        assert report.is_ai_generated is None
    else:
        assert has_invisible_target(path) is False
    loaded = [name for name in sys.modules if name.startswith("remove_ai_watermarks.classify")]
    assert loaded == []


def _plain_png(tmp_path: Path) -> Path:
    path = tmp_path / "plain.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path)
    return path


def test_124d_is_not_extracted_unless_definitely(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []
    runtime = SimpleNamespace(
        ridge_mean=np.zeros(768),
        ridge_scale=np.ones(768),
        ridge_weights=np.zeros(768),
        ridge_threshold=RIDGE_THRESHOLD,
        mlp_threshold=MLP_THRESHOLD,
        provider_margin=PROVIDER_MARGIN,
    )
    monkeypatch.setattr("remove_ai_watermarks.classify.is_available", lambda: True)
    monkeypatch.setattr("remove_ai_watermarks.classify._get_runtime", lambda device: runtime)
    monkeypatch.setattr("remove_ai_watermarks.classify._embed", lambda *args, **kwargs: np.zeros(768))
    monkeypatch.setattr("remove_ai_watermarks.classify._mlp_score", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        "remove_ai_watermarks.classify._forensic",
        lambda image: calls.append(1) or np.zeros(124),
    )
    result = classify_pixels(_plain_png(tmp_path))
    assert result.label == "human"
    assert calls == []


def test_weight_files_may_sit_in_run1(tmp_path: Path) -> None:
    from remove_ai_watermarks.classify import _find_weight

    nested = tmp_path / "run1"
    nested.mkdir()
    target = nested / "detector.pt"
    target.write_bytes(b"x")
    assert _find_weight(tmp_path, "detector.pt") == target
    assert _find_weight(tmp_path, "clip-l-ft.pt") is None


def test_operating_point_prefers_the_sidecar(tmp_path: Path) -> None:
    from remove_ai_watermarks.classify import _operating_point

    (tmp_path / "operating-point.json").write_text(
        json.dumps({"model1": {"mlp_threshold": 1.25}, "model2": {"margin": 0.5}})
    )
    assert _operating_point(tmp_path) == (1.25, 0.5)
    assert _operating_point(tmp_path / "missing") == (MLP_THRESHOLD, PROVIDER_MARGIN)


def test_missing_extra_raises_the_install_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import remove_ai_watermarks.classify as classify_mod

    monkeypatch.setattr(classify_mod, "is_available", lambda: False)
    monkeypatch.setattr(
        classify_mod,
        "_get_runtime",
        lambda device: (_ for _ in ()).throw(AssertionError("must not load weights")),
    )
    path = tmp_path / "plain.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path)
    with pytest.raises(RuntimeError, match=r"remove-ai-watermarks\[classify\]"):
        classify_mod.classify_pixels(path)


def test_cli_classify_without_extra_prints_the_install_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import remove_ai_watermarks.classify as classify_mod

    monkeypatch.setattr(classify_mod, "is_available", lambda: False)
    path = tmp_path / "plain.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path)
    result = CliRunner().invoke(main, ["classify", str(path)])
    assert result.exit_code == 1
    assert "remove-ai-watermarks[classify]" in result.output
    assert "pip install" in result.output
