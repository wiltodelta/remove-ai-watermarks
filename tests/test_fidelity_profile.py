"""Runtime fidelity routing, local acceptance and compatible legacy fallback."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from remove_ai_watermarks._internal import qwen_alignment_pipeline as profiles
from remove_ai_watermarks._internal import text_restoration
from remove_ai_watermarks._internal import two_stage_pipeline as shared
from remove_ai_watermarks._internal.perceptual_metric import LocalPerceptualMetric
from remove_ai_watermarks._internal.watermark_profiles import (
    PROFILE_CHOICES,
    QWEN21_ZIMAGE_PROFILE,
    QWEN_ZIMAGE_PROFILE,
    resolve_strength,
)
from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover
from remove_ai_watermarks.invisible_engine import InvisibleEngine


@pytest.fixture
def removal_runtime(monkeypatch):
    """Exercise model-free routing without requiring the optional GPU stack."""
    from remove_ai_watermarks._internal import watermark_remover

    monkeypatch.setattr(watermark_remover, "_HAS_REMOVAL_MODULES", True)


def pipeline(**kwargs):
    return profiles.Qwen21FidelityZImagePipeline(device="cuda", torch_dtype=torch.bfloat16, **kwargs)


@pytest.mark.parametrize(
    ("tile", "manifest", "size", "reason"),
    [
        (True, None, (64, 64), "tiled_input"),
        (False, SimpleNamespace(lines=[]), (64, 64), "reviewed_text"),
        (False, None, (63, 80), "small_input"),
    ],
)
def test_uncalibrated_paths_retain_the_exact_completed_baseline(tile, manifest, size, reason):
    selector = Mock(side_effect=AssertionError("Uncalibrated path ran OCR"))
    metric = Mock(side_effect=AssertionError("Uncalibrated path ran LPIPS"))
    recipe = pipeline(line_detector=selector, perceptual_metric=metric)
    source = Image.new("RGB", size, "white")
    baseline = Image.new("RGB", size, "gray")
    context = recipe._prepare_regeneration(source, tile=tile, text_manifest=manifest)
    final = recipe._finish_regeneration(source, baseline, [], strength=0.35, seed=0, context=context)
    assert final is baseline
    assert recipe.last_fidelity_skip_reason == reason
    selector.assert_not_called()
    metric.assert_not_called()


def test_empty_text_geometry_still_uses_the_calibrated_fidelity_path(monkeypatch):
    recipe = pipeline(perceptual_metric=lambda *_: 0.0)
    source = Image.new("RGB", (64, 64))
    baseline = Image.new("RGB", source.size)
    expected = Image.new("RGB", source.size, "white")
    finish = Mock(return_value=expected)
    monkeypatch.setattr(profiles.Qwen21AlignmentZImagePipeline, "_finish_regeneration", finish)
    context = recipe._prepare_regeneration(source, tile=False, text_manifest=None)
    assert recipe._finish_regeneration(source, baseline, [], strength=0.2, seed=0, context=context) is expected
    finish.assert_called_once()


def test_acceptance_failure_retains_baseline_and_clears_partial_diagnostics(monkeypatch):
    recipe = pipeline(perceptual_metric=lambda *_: 0.0)
    source = Image.new("RGB", (64, 64))
    baseline = source.copy()
    context = recipe._prepare_regeneration(source, tile=False, text_manifest=None)
    monkeypatch.setattr(
        profiles.Qwen21AlignmentZImagePipeline, "_finish_regeneration", Mock(side_effect=RuntimeError("metric failed"))
    )
    assert recipe._finish_regeneration(source, baseline, [], strength=0.2, seed=0, context=context) is baseline
    assert recipe.last_fidelity_skip_reason == "quality_stage_error"
    assert recipe.last_quality is None
    assert recipe.last_text_quality is None
    assert recipe.last_color_quality is None


def test_reviewed_text_still_reaches_the_shared_vae_donor_and_renderer(monkeypatch):
    recipe = pipeline(perceptual_metric=Mock(side_effect=AssertionError("Reviewed path ran fidelity")))
    source = Image.new("RGB", (64, 64), "white")
    regenerated = Image.new("RGB", source.size, "gray")
    donor = Image.new("RGB", source.size, "blue")
    expected = Image.new("RGB", source.size, "red")
    manifest = SimpleNamespace(lines=("verified-line",))
    monkeypatch.setattr(recipe, "_require_cuda", lambda: None)
    monkeypatch.setattr(recipe, "_run_global", lambda *_: regenerated)
    monkeypatch.setattr(recipe, "_vae_roundtrip", lambda *_: donor)
    monkeypatch.setattr(shared, "detect_faces", lambda *_: [])
    restore = Mock(return_value=expected)
    monkeypatch.setattr(text_restoration, "restore_verified_text", restore)
    assert recipe.run(source, strength=0.35, seed=0, text_manifest=manifest) is expected
    restore.assert_called_once_with(source, regenerated, donor, manifest.lines)


@pytest.mark.usefixtures("removal_runtime")
def test_runtime_registration_constructs_the_real_fidelity_class_without_loading_models():
    assert QWEN21_ZIMAGE_PROFILE in PROFILE_CHOICES
    engine = InvisibleEngine(device="cuda", pipeline=QWEN21_ZIMAGE_PROFILE)

    def selector(_image):
        return ((1, 1, 30, 20),)

    engine.set_text_region_selector(selector)
    recipe = engine._remover._load_qwen_zimage_pipeline()
    assert isinstance(recipe, profiles.Qwen21FidelityZImagePipeline)
    assert isinstance(recipe.perceptual_metric, LocalPerceptualMetric)
    assert recipe.perceptual_metric._model is None
    assert recipe.line_detector is selector
    engine.set_text_region_selector(None)
    assert recipe.line_detector is None
    engine.set_fidelity_enabled(False)
    assert recipe.fidelity_enabled is False
    engine.set_fidelity_enabled(True)
    assert recipe.fidelity_enabled is True
    assert engine._remover._load_qwen_zimage_pipeline() is recipe


def test_disabled_fidelity_retains_baseline_without_calling_the_text_selector():
    selector = Mock(side_effect=AssertionError("Disabled profile ran OCR"))
    recipe = pipeline(line_detector=selector, perceptual_metric=lambda *_: 0.0, fidelity_enabled=False)
    source = Image.new("RGB", (64, 64))
    baseline = source.copy()
    context = recipe._prepare_regeneration(source, tile=False, text_manifest=None)
    assert recipe._finish_regeneration(source, baseline, [], strength=0.35, seed=0, context=context) is baseline
    assert recipe.last_fidelity_skip_reason == "disabled"
    selector.assert_not_called()


@pytest.mark.usefixtures("removal_runtime")
def test_disabled_remover_retains_the_legacy_strength_policy(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64)).save(source)
    remover = WatermarkRemover(device="cuda", pipeline=QWEN21_ZIMAGE_PROFILE)
    remover.set_fidelity_enabled(False)
    run = Mock(return_value=Image.new("RGB", (64, 64)))
    monkeypatch.setattr(remover, "_load_qwen_zimage_pipeline", lambda: SimpleNamespace(run=run))
    monkeypatch.setattr(remover, "_write_output", Mock())
    remover.remove_watermark(source, vendor="google")
    assert run.call_args.kwargs["strength"] == 0.35


@pytest.mark.usefixtures("removal_runtime")
def test_fidelity_activation_rejects_non_boolean_values():
    engine = InvisibleEngine(device="cuda", pipeline=QWEN21_ZIMAGE_PROFILE)
    with pytest.raises(TypeError, match="boolean"):
        engine.set_fidelity_enabled(1)


@pytest.mark.usefixtures("removal_runtime")
def test_other_model_does_not_silently_accept_a_text_selector():
    engine = InvisibleEngine(device="cuda", pipeline=QWEN_ZIMAGE_PROFILE)
    with pytest.raises(ValueError, match="requires"):
        engine.set_text_region_selector(lambda _image: ())


def test_perceptual_model_is_lazy_cached_and_receives_normalized_owned_rgb(monkeypatch):
    import sys

    model = Mock()
    model.float.return_value = model
    model.cpu.return_value = model
    model.train.return_value = model
    model.return_value = torch.tensor(0.25)
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace(LPIPS=factory))
    metric = LocalPerceptualMetric()
    factory.assert_not_called()
    source = Image.new("L", (2048, 1024), 0)
    candidate = Image.new("RGB", source.size, "white")
    assert metric(source, candidate) == 0.25
    assert metric(source, candidate) == 0.25
    factory.assert_called_once_with(net="alex", verbose=False)
    model.train.assert_called_once_with(False)
    reference_tensor, candidate_tensor = model.call_args.args
    assert tuple(reference_tensor.shape) == (1, 3, 512, 1024)
    assert torch.all(reference_tensor == -1)
    assert torch.all(candidate_tensor == 1)
    assert source.mode == "L"
    assert source.size == (2048, 1024)
    assert np.all(np.asarray(candidate) == 255)


@pytest.mark.parametrize(
    ("tile", "manifest", "size", "expected"),
    [
        (True, None, (64, 64), 0.35),
        (False, object(), (64, 64), 0.35),
        (False, None, (32, 80), 0.35),
        (False, None, (64, 64), 0.45),
    ],
)
@pytest.mark.usefixtures("removal_runtime")
def test_remover_uses_legacy_google_strength_only_for_legacy_fallback(
    tmp_path, monkeypatch, tile, manifest, size, expected
):
    source = tmp_path / "source.png"
    Image.new("RGB", size, "white").save(source)
    remover = WatermarkRemover(device="cuda", pipeline=QWEN21_ZIMAGE_PROFILE)
    run = Mock(return_value=Image.new("RGB", size))
    monkeypatch.setattr(remover, "_load_qwen_zimage_pipeline", lambda: SimpleNamespace(run=run))
    monkeypatch.setattr(remover, "_write_output", Mock())
    remover.remove_watermark(source, vendor="google", tile=tile, text_manifest=manifest)
    assert run.call_args.kwargs["strength"] == expected
    assert run.call_args.kwargs["tile"] == tile
    assert run.call_args.kwargs["text_manifest"] is manifest
    assert resolve_strength(None, "google", QWEN21_ZIMAGE_PROFILE, size=size) == 0.45
