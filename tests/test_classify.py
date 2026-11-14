"""Pixel classify gate: Model 1 + receipt gate + 10-provider Model 2, no downloads, no identify hook."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from remove_ai_watermarks import source_classify
from remove_ai_watermarks.classify import (
    CLIP_ONNX_FILE,
    MLP_THRESHOLD,
    PROVIDER_CLASSES,
    PROVIDER_FILE,
    PROVIDER_SHA256,
    RECEIPT_GATE_FILE,
    RECEIPT_GATE_THRESHOLD,
    RIDGE_THRESHOLD,
    WEIGHTS_ALLOW_PATTERNS,
    WEIGHTS_REPO,
    WEIGHTS_REVISION,
    PixelClassification,
    classify_from_scores,
    classify_pixels,
    detector_level,
    label_for,
    provider_from_source,
    receipt_gate_score,
)
from remove_ai_watermarks.cli import main
from remove_ai_watermarks.source_classify import ALL_PROVIDER_CLASSES, SourceClassification


def _source_result(label: str = "unknown") -> SourceClassification:
    reason = "abstained" if label == "unknown" else "classified"
    conf = 0.25 if label == "unknown" else 0.90
    return SourceClassification(
        label=label,  # type: ignore[arg-type]
        version_subfamily=f"{label}:v1" if label != "unknown" else None,
        fine_subfamily=f"{label}:v1" if label != "unknown" else None,
        confidence=conf,
        reason=reason,  # type: ignore[arg-type]
        margins=dict.fromkeys(ALL_PROVIDER_CLASSES, 0.0),
        thresholds=dict.fromkeys(ALL_PROVIDER_CLASSES, 1.0),
        pooled_scores=dict.fromkeys(ALL_PROVIDER_CLASSES, 0.0),
    )


def test_hub_snapshot_and_provider_artifact_match_source_classify() -> None:
    assert WEIGHTS_REPO == "wiltodelta/raiw-photo-classify"
    assert len(WEIGHTS_REVISION) == 40
    assert PROVIDER_FILE == source_classify.MODEL_FILE == "multi-provider-source-v6.npz"
    assert PROVIDER_SHA256 == source_classify.MODEL_SHA256
    assert PROVIDER_CLASSES == ALL_PROVIDER_CLASSES


def test_runtime_requests_only_the_selected_backend_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Online downloads stay narrow while the exported pre-cache covers both.

    An offline deploy pre-caches WEIGHTS_ALLOW_PATTERNS from the Hub; the
    runtime requests a subset of that snapshot for its selected backend.
    """
    from remove_ai_watermarks.classify import _WEIGHT_FILES, _weights_dir

    captured: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        captured.append(kwargs)
        return "/cached/snapshot"

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download))
    monkeypatch.delenv("RAIW_CLASSIFY_WEIGHTS", raising=False)

    assert _weights_dir() == Path("/cached/snapshot")
    assert captured[0]["repo_id"] == WEIGHTS_REPO
    assert captured[0]["revision"] == WEIGHTS_REVISION
    torch_patterns = captured[0]["allow_patterns"]
    assert "clip-l-ft.pt" in torch_patterns
    assert PROVIDER_FILE in torch_patterns
    assert "provider.pt" not in torch_patterns
    assert CLIP_ONNX_FILE not in torch_patterns
    assert set(_WEIGHT_FILES) <= set(WEIGHTS_ALLOW_PATTERNS)
    assert CLIP_ONNX_FILE in WEIGHTS_ALLOW_PATTERNS

    captured.clear()
    assert _weights_dir("onnx") == Path("/cached/snapshot")
    onnx_patterns = captured[0]["allow_patterns"]
    assert CLIP_ONNX_FILE in onnx_patterns
    assert PROVIDER_FILE in onnx_patterns
    assert "clip-l-ft.pt" not in onnx_patterns
    assert set(torch_patterns) | set(onnx_patterns) == set(WEIGHTS_ALLOW_PATTERNS)


def test_backend_requires_its_own_model_file(tmp_path: Path) -> None:
    from remove_ai_watermarks.classify import DETECTOR_FILE, PROBE_FILE, _require_files

    for name in (CLIP_ONNX_FILE, PROBE_FILE, DETECTOR_FILE, PROVIDER_FILE):
        (tmp_path / name).touch()

    _require_files(tmp_path, backend="onnx")
    with pytest.raises(RuntimeError, match=r"clip-l-ft\.pt"):
        _require_files(tmp_path, backend="torch")


def test_onnx_backend_is_cpu_only() -> None:
    import torch

    from remove_ai_watermarks.classify import _resolve_backend

    assert _resolve_backend("onnx", torch.device("cpu")) == "onnx"
    with pytest.raises(ValueError, match=r"ONNX.*CPU"):
        _resolve_backend("onnx", torch.device("cuda"))


def test_onnx_backend_names_its_install_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    from remove_ai_watermarks import classify as classify_mod
    from remove_ai_watermarks import optional_deps

    monkeypatch.setattr(classify_mod, "is_available", lambda: True)
    monkeypatch.setattr(optional_deps, "module_available", lambda *_names: False)

    with pytest.raises(RuntimeError, match=r"remove-ai-watermarks\[classify-onnx\]"):
        classify_mod.classify_pixels(Path("unused.png"), backend="onnx")


def test_embed_dispatches_to_the_selected_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from remove_ai_watermarks import classify as classify_mod
    from remove_ai_watermarks._internal import clip_l_ft

    calls: list[str] = []
    monkeypatch.setattr(clip_l_ft, "embed_image", lambda *_args: calls.append("torch") or "torch-vector")
    monkeypatch.setattr(clip_l_ft, "embed_image_onnx", lambda *_args: calls.append("onnx") or "onnx-vector")
    runtime = SimpleNamespace(backend="onnx", clip_model=object(), processor=object(), device=object())

    assert classify_mod._embed(runtime, object()) == "onnx-vector"
    runtime.backend = "torch"
    assert classify_mod._embed(runtime, object()) == "torch-vector"
    assert calls == ["onnx", "torch"]


def test_shipped_operating_point_matches_the_runtime_defaults() -> None:
    payload = json.loads(Path("docs/photo-classify-hf/operating-point.json").read_text())
    card = Path("docs/photo-classify-hf/README.md").read_text()
    assert payload["model1"]["mlp_threshold"] == MLP_THRESHOLD
    assert payload["model1"]["ridge_threshold"] == RIDGE_THRESHOLD
    assert payload["model2"]["artifact"] == PROVIDER_FILE
    assert payload["model2"]["artifact_sha256"] == PROVIDER_SHA256
    assert payload["model2"]["feature_width"] == 512
    assert payload["model2"]["runs_only_after"] == "definitely"
    assert payload["model2"]["classes"] == list(PROVIDER_CLASSES)
    assert payload["model2"]["public_values"] == [*PROVIDER_CLASSES, None]
    assert payload["receipt_gate"]["threshold"] == RECEIPT_GATE_THRESHOLD
    assert payload["receipt_gate"]["asset"] == RECEIPT_GATE_FILE
    assert payload["receipt_gate"]["runs_only_after"] == "definitely"
    assert payload["receipt_gate"]["on_hit"] == "unknown"
    assert PROVIDER_FILE in payload["files"]
    assert PROVIDER_FILE in card
    assert PROVIDER_SHA256 in card


def test_definitely_is_the_and_of_ridge_and_mlp() -> None:
    assert detector_level(1.0, 10.0) == "definitely"
    assert detector_level(0.0, 10.0) == "possibly"
    assert detector_level(1.0, 0.0) == "possibly"
    assert detector_level(0.0, 0.0) == "likely_human"


def test_only_definitely_is_ai() -> None:
    assert label_for("definitely") == "ai"
    assert label_for("possibly") == "unknown"
    assert label_for("likely_human") == "human"


@pytest.mark.parametrize("provider_name", PROVIDER_CLASSES)
def test_provider_from_source_accepts_all_ten_providers(provider_name: str) -> None:
    assert provider_from_source(_source_result(provider_name)) == provider_name
    assert provider_from_source(provider_name) == provider_name  # type: ignore[arg-type]


def test_provider_from_source_abstains_and_rejects_unknown_names() -> None:
    assert provider_from_source(None) is None
    assert provider_from_source("unknown") is None
    assert provider_from_source(_source_result("unknown")) is None
    with pytest.raises(ValueError, match="unsupported pixel provider"):
        provider_from_source("muse-image")  # type: ignore[arg-type]


def test_definitely_plus_provider_publishes_ai_and_provider() -> None:
    for provider_name in ("openai", "flux", "meta", "firefly", "xai"):
        result = classify_from_scores(1.0, 10.0, _source_result(provider_name))
        assert result.label == "ai"
        assert result.domain == "photo"
        assert result.detector == "definitely"
        assert result.provider == provider_name
        assert result.to_dict()["provider"] == provider_name


def test_definitely_without_provider_or_abstained_stays_ai_with_no_provider() -> None:
    assert classify_from_scores(1.0, 10.0, None) == PixelClassification(
        label="ai", domain="photo", detector="definitely", provider=None
    )
    assert classify_from_scores(1.0, 10.0, _source_result("unknown")) == PixelClassification(
        label="ai", domain="photo", detector="definitely", provider=None
    )


def test_possibly_does_not_run_provider() -> None:
    result = classify_from_scores(1.0, 0.0, _source_result("openai"))
    assert result.label == "unknown"
    assert result.detector == "possibly"
    assert result.provider is None


def test_likely_human_does_not_run_provider() -> None:
    result = classify_from_scores(0.0, 0.0, _source_result("openai"))
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
    result = classify_from_scores(
        1.0,
        10.0,
        _source_result("openai"),
        receipt_score=RECEIPT_GATE_THRESHOLD + 1.0,
    )
    assert result.label == "unknown"
    assert result.detector == "definitely"
    assert result.provider is None


def test_receipt_gate_below_threshold_keeps_ai_and_provider() -> None:
    result = classify_from_scores(
        1.0,
        10.0,
        _source_result("flux"),
        receipt_score=RECEIPT_GATE_THRESHOLD - 0.01,
    )
    assert result.label == "ai"
    assert result.provider == "flux"


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
        assert receipt_gate_score(probe, gate) == pytest.approx(0.5 * np.sqrt(768))
    finally:
        _receipt_gate_cache.clear()


def test_receipt_threshold_defaults_to_the_artifact_value() -> None:
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


def _patch_definitely_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model2_calls: list[int] | None = None,
    provider_path: Path = Path("/unused/multi-provider-source-v6.npz"),
    weights_override: str | None = "/override",
) -> SimpleNamespace:
    """Patch classify internals so a plain PNG scores DEFINITELY on Model 1."""
    runtime = SimpleNamespace(
        ridge_mean=np.zeros(768),
        ridge_scale=np.ones(768),
        ridge_weights=np.full(768, 0.01),
        ridge_threshold=RIDGE_THRESHOLD,
        mlp_threshold=MLP_THRESHOLD,
        provider_path=provider_path,
        weights_dir=None,
        weights_override=weights_override,
        provider_verified=False,
    )
    monkeypatch.setattr("remove_ai_watermarks.classify.is_available", lambda: True)
    monkeypatch.setattr("remove_ai_watermarks.classify._get_runtime", lambda device, backend="torch": runtime)
    monkeypatch.setattr("remove_ai_watermarks.classify._embed", lambda *args, **kwargs: np.full(768, 100.0))
    monkeypatch.setattr("remove_ai_watermarks.classify._mlp_score", lambda *args, **kwargs: 10.0)
    if model2_calls is not None:
        monkeypatch.setattr(
            "remove_ai_watermarks.classify._run_model2",
            lambda rt, image, **kwargs: model2_calls.append(1) or _source_result("openai"),
        )
    return runtime


def test_receipt_gate_hit_skips_the_512d_forensics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []
    _patch_definitely_runtime(monkeypatch, model2_calls=calls)
    monkeypatch.setattr(
        "remove_ai_watermarks.classify.receipt_gate_score", lambda *args, **kwargs: RECEIPT_GATE_THRESHOLD + 5.0
    )
    result = classify_pixels(_plain_png(tmp_path))
    assert result.label == "unknown"
    assert result.detector == "definitely"
    assert calls == []


def test_receipt_gate_miss_runs_v6_model2_and_attributes_ten_class_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.test_source_classify import _write_synthetic_v5_npz

    v6_path = _write_synthetic_v5_npz(
        tmp_path / PROVIDER_FILE,
        det_winner="flux",
        attr_winner="ideogram",
        det_score=5.0,
        det_threshold=1.0,
    )
    calls: list[int] = []
    runtime = _patch_definitely_runtime(
        monkeypatch,
        provider_path=v6_path,
        weights_override=None,
    )
    monkeypatch.setattr(
        "remove_ai_watermarks.classify.PROVIDER_SHA256",
        hashlib.sha256(v6_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        "remove_ai_watermarks.source_classify.extract_512d_v4_features",
        lambda image: calls.append(1) or np.zeros(512),
    )
    monkeypatch.setattr(
        "remove_ai_watermarks.classify.receipt_gate_score", lambda *args, **kwargs: RECEIPT_GATE_THRESHOLD - 5.0
    )
    result = classify_pixels(_plain_png(tmp_path, side=256))
    assert result.label == "ai"
    assert result.provider == "ideogram"
    assert calls == [1]
    assert runtime.provider_verified is True


def test_small_image_below_256px_abstains_provider_while_keeping_ai_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = _patch_definitely_runtime(monkeypatch, weights_override=None)
    monkeypatch.setattr(
        "remove_ai_watermarks.classify.receipt_gate_score", lambda *args, **kwargs: RECEIPT_GATE_THRESHOLD - 5.0
    )
    # _plain_png is 32x32 (< 256x256), so _run_model2 returns None before hashing or loading v6.npz.
    result = classify_pixels(_plain_png(tmp_path))
    assert result.label == "ai"
    assert result.detector == "definitely"
    assert result.provider is None
    assert runtime.provider_verified is False


def test_verify_sha256_enforces_hub_provider_digest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / PROVIDER_FILE
    target.write_bytes(b"corrupted")
    _patch_definitely_runtime(monkeypatch, provider_path=target, weights_override=None)
    monkeypatch.setattr(
        "remove_ai_watermarks.classify.receipt_gate_score", lambda *args, **kwargs: RECEIPT_GATE_THRESHOLD - 5.0
    )
    with pytest.raises(RuntimeError, match=r"SHA-256 hash mismatch for multi-provider-source-v6\.npz"):
        classify_pixels(_plain_png(tmp_path, side=256))


@pytest.mark.parametrize("caller", ["identify", "has_invisible_target"])
def test_provenance_paths_do_not_import_classify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caller: str) -> None:
    path = tmp_path / "plain.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path)
    monkeypatch.delitem(sys.modules, "remove_ai_watermarks.classify", raising=False)
    monkeypatch.delitem(sys.modules, "remove_ai_watermarks.source_classify", raising=False)
    from remove_ai_watermarks.identify import has_invisible_target, identify

    if caller == "identify":
        report = identify(path, check_visible=False, check_invisible=False)
        assert report.is_ai_generated is None
    else:
        assert has_invisible_target(path) is False
    loaded = [
        name
        for name in sys.modules
        if name.startswith(("remove_ai_watermarks.classify", "remove_ai_watermarks.source_classify"))
    ]
    assert loaded == []


def _plain_png(tmp_path: Path, *, side: int = 32) -> Path:
    path = tmp_path / f"plain-{side}.png"
    Image.fromarray(np.zeros((side, side, 3), dtype=np.uint8)).save(path)
    return path


def test_512d_is_not_extracted_unless_definitely(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []
    runtime = SimpleNamespace(
        ridge_mean=np.zeros(768),
        ridge_scale=np.ones(768),
        ridge_weights=np.zeros(768),
        ridge_threshold=RIDGE_THRESHOLD,
        mlp_threshold=MLP_THRESHOLD,
        provider_path=Path("/unused/multi-provider-source-v6.npz"),
        weights_dir=None,
    )
    monkeypatch.setattr("remove_ai_watermarks.classify.is_available", lambda: True)
    monkeypatch.setattr("remove_ai_watermarks.classify._get_runtime", lambda device, backend="torch": runtime)
    monkeypatch.setattr("remove_ai_watermarks.classify._embed", lambda *args, **kwargs: np.zeros(768))
    monkeypatch.setattr("remove_ai_watermarks.classify._mlp_score", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        "remove_ai_watermarks.classify._run_model2",
        lambda rt, image, **kwargs: calls.append(1) or None,
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

    (tmp_path / "operating-point.json").write_text(json.dumps({"model1": {"mlp_threshold": 1.25}}))
    assert _operating_point(tmp_path) == 1.25
    assert _operating_point(tmp_path / "missing") == MLP_THRESHOLD


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


def test_cli_classify_json_and_text_emit_expanded_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import remove_ai_watermarks.classify as classify_mod

    monkeypatch.setattr(classify_mod, "is_available", lambda: True)
    monkeypatch.setattr(
        classify_mod,
        "classify_pixels",
        lambda _p: PixelClassification(label="ai", domain="photo", detector="definitely", provider="flux"),
    )
    path = _plain_png(tmp_path)
    json_res = CliRunner().invoke(main, ["classify", str(path), "--json"])
    assert json_res.exit_code == 0
    assert json.loads(json_res.output) == {
        "label": "ai",
        "domain": "photo",
        "detector": "definitely",
        "provider": "flux",
    }

    text_res = CliRunner().invoke(main, ["classify", str(path)])
    assert text_res.exit_code == 0
    assert "Provider: flux" in text_res.output
