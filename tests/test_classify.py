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
    RECEIPT_GATE_FILE,
    RECEIPT_GATE_THRESHOLD,
    RIDGE_THRESHOLD,
    WEIGHTS_ALLOW_PATTERNS,
    WEIGHTS_REPO,
    WEIGHTS_REVISION,
    classify_from_scores,
    classify_pixels,
    detector_level,
    label_for,
    provider_from_scores,
    receipt_gate_score,
)
from remove_ai_watermarks.cli import main


def _scores(
    *,
    openai: float = 0.0,
    google: float = 0.0,
    tc260: float = 0.0,
    meta: float = 0.0,
    no_ai: float = 0.0,
    bytedance: float | None = None,
) -> dict[str, float]:
    scores = {
        "openai": openai,
        "google": google,
        "tc260": tc260,
        "meta_muse_image": meta,
        "no_ai": no_ai,
    }
    if bytedance is not None:
        scores["bytedance"] = bytedance
    return scores


def test_hub_snapshot_is_the_freeze_revision() -> None:
    assert WEIGHTS_REPO == "wiltodelta/raiw-photo-classify"
    assert WEIGHTS_REVISION == "4c2763766dd1c8c212d64e01e1cc3f166243c47d"


def test_runtime_requests_the_exported_allow_patterns(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exported pre-cache list and the runtime request must stay one list.

    An offline deploy pre-caches WEIGHTS_ALLOW_PATTERNS from the Hub; the
    runtime then resolves the same snapshot offline. If the runtime's request
    grows a file the export misses, every offline deployment breaks on a
    four-fifths-present cache, so pin the seam in both directions.
    """
    from remove_ai_watermarks.classify import _WEIGHT_FILES, _weights_dir

    captured: dict[str, object] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        captured.append(kwargs)
        return "/cached/snapshot"

    # huggingface_hub ships with the classify extra, not with the dev sync,
    # so stub the module instead of importing it.
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download))
    monkeypatch.delenv("RAIW_CLASSIFY_WEIGHTS", raising=False)

    assert _weights_dir() == Path("/cached/snapshot")
    assert captured[0]["repo_id"] == WEIGHTS_REPO
    assert captured[0]["revision"] == WEIGHTS_REVISION
    assert captured[0]["allow_patterns"] == list(WEIGHTS_ALLOW_PATTERNS)
    assert set(_WEIGHT_FILES) <= set(WEIGHTS_ALLOW_PATTERNS)


def test_shipped_operating_point_matches_the_runtime_defaults() -> None:
    payload = json.loads(Path("docs/photo-classify-hf/operating-point.json").read_text())
    assert payload["model1"]["mlp_threshold"] == MLP_THRESHOLD
    assert payload["model1"]["ridge_threshold"] == RIDGE_THRESHOLD
    assert payload["model2"]["margin"] == PROVIDER_MARGIN
    assert payload["model2"]["runs_only_after"] == "definitely"
    assert payload["model2"]["classes"] == ["openai", "google", "bytedance", "tc260", "muse-image", "no_ai"]
    assert payload["model2"]["class_kind"]["openai"] == "provider"
    assert payload["model2"]["class_kind"]["google"] == "provider"
    assert payload["model2"]["class_kind"]["bytedance"] == "provider"
    assert payload["model2"]["class_kind"]["tc260"] == "abstain-veto"
    assert payload["model2"]["class_kind"]["muse-image"] == "model"
    assert payload["model2"]["public_values"] == ["openai", "google", "bytedance", "muse-image", None]
    assert payload["model2"]["checkpoint_keys"]["muse-image"] == "meta_muse_image"
    assert payload["receipt_gate"]["threshold"] == RECEIPT_GATE_THRESHOLD
    assert payload["receipt_gate"]["asset"] == RECEIPT_GATE_FILE
    assert payload["receipt_gate"]["runs_only_after"] == "definitely"
    assert payload["receipt_gate"]["on_hit"] == "unknown"


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


def test_bytedance_names_the_shared_bytedance_renderer_lineage() -> None:
    """Doubao and Jimeng render with one ByteDance model family, so the
    generator class is the union; the value names the lineage."""
    assert provider_from_scores(_scores(bytedance=1.0, no_ai=0.0)) == "bytedance"
    assert provider_from_scores(_scores(bytedance=3.0, openai=1.0, no_ai=0.0)) == "bytedance"
    assert provider_from_scores(_scores(bytedance=1.0, openai=2.0, no_ai=0.0)) == "openai"


def test_older_snapshots_without_bytedance_keep_working() -> None:
    """provider.pt predating bytedance has five heads; the candidate set
    comes from the LOADED scores, so such snapshots run unchanged and
    their would-be bytedance rows fall to the tc260 veto."""
    legacy = _scores(tc260=2.0, openai=1.0, no_ai=0.0)
    assert "bytedance" not in legacy
    assert provider_from_scores(legacy) is None
    assert provider_from_scores(_scores(openai=3.0, no_ai=0.0)) == "openai"


def test_tc260_wins_abstain_instead_of_naming_a_provider() -> None:
    """tc260 is not one class of anything: it covers China's generator
    ecosystem, providers that are peers of openai/google/meta. No mixed
    head can honestly name that group, so its argmax win publishes None; a
    row where a real provider outscores it keeps that provider."""
    assert provider_from_scores(_scores(tc260=1.0, no_ai=0.0)) is None
    assert provider_from_scores(_scores(tc260=3.0, openai=1.0, no_ai=0.0)) is None
    assert provider_from_scores(_scores(tc260=1.0, openai=2.0, no_ai=0.0)) == "openai"


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


def test_shipped_receipt_gate_asset_matches_the_pinned_threshold() -> None:
    payload = np.load(Path("src/remove_ai_watermarks/assets") / RECEIPT_GATE_FILE)
    assert float(payload["threshold"]) == RECEIPT_GATE_THRESHOLD
    assert payload["w"].shape == (768,)
    assert payload["mu"].shape == (768,)
    assert payload["sd"].shape == (768,)


def test_receipt_gate_downgrades_definitely_to_unknown() -> None:
    result = classify_from_scores(1.0, 10.0, _scores(openai=9.0, no_ai=0.0), receipt_score=RECEIPT_GATE_THRESHOLD + 1.0)
    assert result.label == "unknown"
    assert result.detector == "definitely"
    assert result.provider is None


def test_receipt_gate_below_threshold_keeps_ai_and_provider() -> None:
    result = classify_from_scores(
        1.0,
        10.0,
        _scores(openai=1.0, no_ai=0.0),
        receipt_score=RECEIPT_GATE_THRESHOLD - 0.01,
    )
    assert result.label == "ai"
    assert result.provider == "openai"


@pytest.mark.parametrize(
    ("ridge", "mlp", "expected"),
    [(1.0, 0.0, "unknown"), (0.0, 0.0, "human")],
)
def test_receipt_gate_never_touches_non_definitely(ridge: float, mlp: float, expected: str) -> None:
    result = classify_from_scores(ridge, mlp, None, receipt_score=99.0)
    assert result.label == expected


def test_receipt_gate_score_is_deterministic_and_bounded() -> None:
    v = np.full(768, 1.0 / np.sqrt(768), dtype=np.float64)
    first = receipt_gate_score(v)
    assert first == receipt_gate_score(v)
    assert -50.0 < first < 50.0


def test_receipt_gate_prefers_the_weights_directory(tmp_path: Path) -> None:
    from remove_ai_watermarks.classify import _load_receipt_gate, _receipt_gate_cache

    source = np.load(Path("src/remove_ai_watermarks/assets") / RECEIPT_GATE_FILE)
    weights = tmp_path / "weights"
    weights.mkdir()
    np.savez(
        weights / RECEIPT_GATE_FILE,
        w=source["w"],
        b=source["b"],
        mu=source["mu"],
        sd=source["sd"],
        threshold=99.0,
    )
    _receipt_gate_cache.clear()
    try:
        gate = _load_receipt_gate(weights)
        assert gate["threshold"] == 99.0
        mid = (RECEIPT_GATE_THRESHOLD + 99.0) / 2
        result = classify_from_scores(1.0, 10.0, None, receipt_score=mid, receipt_threshold=gate["threshold"])
        assert result.label == "ai"
        assert classify_from_scores(1.0, 10.0, None, receipt_score=mid).label == "unknown"
    finally:
        _receipt_gate_cache.clear()


def test_receipt_gate_stable_name_beats_the_legacy_spelling(tmp_path: Path) -> None:
    """The model-side artifact (stable name) owns the gate: same directory,
    stable spelling wins, and head plus threshold come from that ONE file.
    """
    from remove_ai_watermarks.classify import (
        RECEIPT_GATE_STABLE_FILE,
        _load_receipt_gate,
        _receipt_gate_cache,
        receipt_gate_score,
    )

    source = np.load(Path("src/remove_ai_watermarks/assets") / RECEIPT_GATE_FILE)
    weights = tmp_path / "weights"
    weights.mkdir()
    np.savez(weights / RECEIPT_GATE_FILE, **{k: source[k] for k in ("w", "b", "mu", "sd")}, threshold=99.0)
    np.savez(
        weights / RECEIPT_GATE_STABLE_FILE,
        w=np.full(768, 0.5),
        b=0.0,
        mu=np.zeros(768),
        sd=np.ones(768),
        threshold=-99.0,
    )
    _receipt_gate_cache.clear()
    try:
        gate = _load_receipt_gate(weights)
        assert gate["threshold"] == -99.0
        probe = np.full(768, 1.0 / np.sqrt(768), dtype=np.float64)
        # The score must come from the SAME stable-name artifact: with its
        # uniform weights the score is exactly 0.5*sqrt(768), not the legacy
        # package head's value.
        assert receipt_gate_score(probe, gate) == pytest.approx(0.5 * np.sqrt(768))
    finally:
        _receipt_gate_cache.clear()


def test_receipt_threshold_defaults_to_the_artifact_value() -> None:
    """classify_from_scores without an explicit threshold resolves the
    operating point from the loaded gate artifact (legacy package fallback
    here), not from a code constant."""
    from remove_ai_watermarks.classify import _load_receipt_gate, _receipt_gate_cache

    _receipt_gate_cache.clear()
    try:
        artifact_threshold = _load_receipt_gate(None)["threshold"]
        just_above = classify_from_scores(1.0, 10.0, None, receipt_score=artifact_threshold + 1e-9)
        just_below = classify_from_scores(1.0, 10.0, None, receipt_score=artifact_threshold - 1e-9)
        assert just_above.label == "unknown"
        assert just_below.label == "ai"
    finally:
        _receipt_gate_cache.clear()


def test_receipt_gate_falls_back_to_the_package_asset() -> None:
    from remove_ai_watermarks.classify import _load_receipt_gate, _receipt_gate_cache

    _receipt_gate_cache.clear()
    try:
        assert _load_receipt_gate(None)["threshold"] == RECEIPT_GATE_THRESHOLD
        assert _load_receipt_gate(Path("/nonexistent-weights"))["threshold"] == RECEIPT_GATE_THRESHOLD
    finally:
        _receipt_gate_cache.clear()


def _patch_definitely_runtime(monkeypatch: pytest.MonkeyPatch, *, forensic_calls: list[int]) -> None:
    """Patch classify internals so a plain PNG scores DEFINITELY on Model 1."""
    runtime = SimpleNamespace(
        ridge_mean=np.zeros(768),
        ridge_scale=np.ones(768),
        ridge_weights=np.full(768, 0.01),
        ridge_threshold=RIDGE_THRESHOLD,
        mlp_threshold=MLP_THRESHOLD,
        provider_margin=PROVIDER_MARGIN,
        weights_dir=None,
    )
    monkeypatch.setattr("remove_ai_watermarks.classify.is_available", lambda: True)
    monkeypatch.setattr("remove_ai_watermarks.classify._get_runtime", lambda device: runtime)
    monkeypatch.setattr("remove_ai_watermarks.classify._embed", lambda *args, **kwargs: np.full(768, 100.0))
    monkeypatch.setattr("remove_ai_watermarks.classify._mlp_score", lambda *args, **kwargs: 10.0)
    monkeypatch.setattr(
        "remove_ai_watermarks.classify._forensic",
        lambda image: forensic_calls.append(1) or np.zeros(124),
    )


def test_receipt_gate_hit_skips_the_124d_forensics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []
    _patch_definitely_runtime(monkeypatch, forensic_calls=calls)
    monkeypatch.setattr(
        "remove_ai_watermarks.classify.receipt_gate_score", lambda *args, **kwargs: RECEIPT_GATE_THRESHOLD + 5.0
    )
    result = classify_pixels(_plain_png(tmp_path))
    assert result.label == "unknown"
    assert result.detector == "definitely"
    assert calls == []


def test_receipt_gate_miss_still_runs_the_124d_forensics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []
    _patch_definitely_runtime(monkeypatch, forensic_calls=calls)
    monkeypatch.setattr(
        "remove_ai_watermarks.classify._provider_scores",
        lambda runtime, forensic: _scores(),
    )
    monkeypatch.setattr(
        "remove_ai_watermarks.classify.receipt_gate_score", lambda *args, **kwargs: RECEIPT_GATE_THRESHOLD - 5.0
    )
    result = classify_pixels(_plain_png(tmp_path))
    assert result.label == "ai"
    assert calls == [1]


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
        weights_dir=None,
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
