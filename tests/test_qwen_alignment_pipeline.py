"""Exercise shared lifecycle, local metric isolation, and exact fallback."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from remove_ai_watermarks._internal import image_alignment as geometry
from remove_ai_watermarks._internal import qwen_alignment_pipeline as aligned
from remove_ai_watermarks._internal import qwen_zimage_pipeline as kernels
from remove_ai_watermarks._internal import two_stage_pipeline as shared
from remove_ai_watermarks._internal.chroma_zimage_pipeline import ChromaZImagePipeline
from remove_ai_watermarks._internal.qwen_zimage_pipeline import Qwen21ZImagePipeline, QwenZImagePipeline
from remove_ai_watermarks._internal.sdxl_zimage_pipeline import SdxlZImagePipeline

OLD_PROFILES = (QwenZImagePipeline, Qwen21ZImagePipeline, SdxlZImagePipeline, ChromaZImagePipeline)


@pytest.fixture
def recipe(monkeypatch):
    state = SimpleNamespace(events=[], scores=[0.5, 0.4], mutate=False)
    state.source = Image.new("RGB", (64, 64), (180, 180, 180))
    base = np.full((64, 64, 3), 200, np.uint8)
    base[20:30, 20:30] = 220
    state.baseline = Image.fromarray(base)
    state.candidate = np.full_like(base, 190)
    state.candidate[20:30, 20:30] = 220

    def metric(reference, candidate):
        state.events.append("metric")
        assert reference.getpixel((0, 0)) == (180, 180, 180)
        if state.mutate:
            reference.putpixel((0, 0), (255, 255, 255))
            candidate.putpixel((0, 0), (255, 255, 255))
        return state.scores.pop(0)

    pipeline = aligned.Qwen21AlignmentZImagePipeline(device="cuda", torch_dtype="bf16", perceptual_metric=metric)
    monkeypatch.setattr(pipeline, "_require_cuda", lambda: None)

    def global_pass(image, strength, seed):
        state.events.append(("global", strength, seed))
        return Image.new("RGB", image.size, (200, 200, 200))

    def faces(*_args, **_kwargs):
        state.events.append("faces")
        return state.baseline

    def align(reference, donor, boxes, *, resampling):
        state.events.append("align")
        assert resampling == "lanczos"
        assert boxes == ((20, 20, 30, 30),)
        assert np.array_equal(donor, base)
        return state.candidate.copy(), np.zeros((64, 64, 2), np.float32)

    monkeypatch.setattr(pipeline, "_run_global", global_pass)
    monkeypatch.setattr(shared, "detect_faces", lambda _image: [(20, 20, 30, 30)])
    monkeypatch.setattr(pipeline, "_sam_masks", lambda *_: [])
    monkeypatch.setattr(pipeline, "_run_faces", faces)
    monkeypatch.setattr(geometry, "align_regenerated", align)
    monkeypatch.setattr(geometry, "local_error_weights", lambda *_: np.ones((64, 64), np.float32))
    monkeypatch.setattr(
        geometry, "attenuate_regenerated", lambda *_: (state.candidate.copy(), np.zeros((64, 64, 2), np.float32))
    )
    return pipeline, state


def test_profile_is_a_direct_sibling_with_shared_lifecycle():
    cls = aligned.Qwen21AlignmentZImagePipeline
    assert cls.__bases__ == (shared.TwoStageZImagePipeline,)
    assert not any(issubclass(cls, profile) for profile in OLD_PROFILES)
    for name in ("run", "preload", "_run_faces", "_load_zimage", "_load_sam", "_face_stage_dtype"):
        assert inspect.getattr_static(cls, name) is inspect.getattr_static(shared.TwoStageZImagePipeline, name)


def test_unbound_kernels_are_the_existing_qwen21_implementations():
    assert kernels.qwen21_run_global is Qwen21ZImagePipeline._run_global
    assert kernels.qwen21_load_global is Qwen21ZImagePipeline._load_global
    assert kernels.qwen21_vram_config is Qwen21ZImagePipeline._qwen_vram_config
    assert kernels.qwen21_vae_roundtrip is Qwen21ZImagePipeline._vae_roundtrip


def test_global_delegation_forwards_profile_input_strength_and_seed(monkeypatch):
    pipeline = aligned.Qwen21AlignmentZImagePipeline(
        device="cuda", torch_dtype="bf16", perceptual_metric=lambda _source, _candidate: 0.0
    )
    source = Image.new("RGB", (64, 64))
    sentinel = Image.new("RGB", (64, 64), (50, 50, 50))
    calls = []

    def run(owner, image, strength, seed):
        calls.append((owner, image, strength, seed))
        return sentinel

    monkeypatch.setattr(aligned, "qwen21_run_global", run)
    assert pipeline._run_global(source, 0.2, 17) is sentinel
    assert calls == [(pipeline, source, 0.2, 17)]


def test_alignment_follows_completed_faces_with_one_global_pass(recipe):
    pipeline, state = recipe
    result = pipeline.run(state.source, strength=0.35, seed=0)
    assert state.events == [("global", 0.35, 0), "faces", "align", "metric", "metric"]
    assert np.array_equal(np.asarray(result), state.candidate)
    assert pipeline.last_quality["selected"]
    assert pipeline.last_quality["reason"] == "both_metrics_improved"
    assert np.array_equal(np.asarray(result)[20:30, 20:30], np.asarray(state.baseline)[20:30, 20:30])


@pytest.mark.parametrize("scores", [[0.4, 0.5], [0.4, 0.4]])
def test_perceptual_regression_or_tie_returns_the_exact_baseline_object(recipe, scores):
    pipeline, state = recipe
    state.scores = scores
    result = pipeline.run(state.source, strength=0.35, seed=0)
    assert result is state.baseline
    assert not pipeline.last_quality["selected"]
    assert pipeline.last_quality["reason"] == "perceptual_error_did_not_improve"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True, "invalid", None])
def test_invalid_metric_cannot_select_a_candidate(recipe, value):
    pipeline, state = recipe
    state.scores = [0.5, value]
    assert pipeline.run(state.source, strength=0.35, seed=0) is state.baseline
    assert pipeline.last_quality["reason"] == "invalid_perceptual_metric"


def test_pixel_regression_skips_perceptual_scoring(recipe):
    pipeline, state = recipe
    state.candidate[:] = 240
    state.candidate[20:30, 20:30] = 220
    assert pipeline.run(state.source, strength=0.35, seed=0) is state.baseline
    assert "metric" not in state.events
    assert pipeline.last_quality["reason"] == "pixel_error_did_not_improve"


def test_metric_gets_owned_copies_instead_of_live_source_or_donors(recipe):
    pipeline, state = recipe
    state.mutate = True
    result = pipeline.run(state.source, strength=0.35, seed=0)
    assert state.source.getpixel((0, 0)) == (180, 180, 180)
    assert state.baseline.getpixel((0, 0)) == (200, 200, 200)
    assert result.getpixel((0, 0)) == (190, 190, 190)


def test_request_failure_does_not_retain_previous_quality_selection(recipe, monkeypatch):
    pipeline, state = recipe
    pipeline.run(state.source, strength=0.35, seed=0)
    assert pipeline.last_quality["selected"]

    def fail(*_args, **_kwargs):
        raise RuntimeError("geometry failed")

    monkeypatch.setattr(geometry, "align_regenerated", fail)
    with pytest.raises(RuntimeError, match="geometry failed"):
        pipeline.run(state.source, strength=0.35, seed=1)
    assert pipeline.last_quality is None


def test_protected_face_postcondition_is_independent_of_geometry_helper(recipe):
    pipeline, state = recipe
    state.candidate[20:30, 20:30] = 0
    with pytest.raises(RuntimeError, match="protected baseline pixels"):
        pipeline.run(state.source, strength=0.35, seed=0)
    assert "metric" not in state.events


@pytest.mark.parametrize("options", [{"tile": True}, {"text_manifest": object()}])
def test_uncalibrated_modes_fail_before_inference(recipe, options):
    pipeline, state = recipe
    with pytest.raises(ValueError, match="experimental alignment profile"):
        pipeline.run(state.source, strength=0.35, seed=0, **options)
    assert not state.events


def test_small_input_fails_before_inference(recipe):
    pipeline, state = recipe
    with pytest.raises(ValueError, match="at least 64 pixels"):
        pipeline.run(Image.new("RGB", (32, 64)), strength=0.35, seed=0)
    assert not state.events


@pytest.mark.parametrize("cls", OLD_PROFILES)
@pytest.mark.parametrize("with_faces", [False, True])
def test_existing_profiles_keep_the_exact_unextended_result(cls, with_faces, monkeypatch):
    pipeline = cls(device="cuda", torch_dtype="bf16")
    baseline = Image.new("RGB", (64, 64), (90, 90, 90))
    face_result = Image.new("RGB", (64, 64), (110, 110, 110))
    monkeypatch.setattr(pipeline, "_require_cuda", lambda: None)
    monkeypatch.setattr(pipeline, "_run_global", lambda *_: baseline)
    monkeypatch.setattr(shared, "detect_faces", lambda _: [(10, 10, 20, 20)] if with_faces else [])
    monkeypatch.setattr(pipeline, "_sam_masks", lambda *_: [])
    monkeypatch.setattr(pipeline, "_run_faces", lambda *_args, **_kwargs: face_result)
    assert pipeline.run(Image.new("RGB", (64, 64)), strength=0.35, seed=0) is (face_result if with_faces else baseline)


@pytest.fixture
def text_recipe(recipe, monkeypatch):
    from remove_ai_watermarks._internal import text_shapes

    pipeline, state = recipe
    state.proposals = [[1, 1, 12, 12]]
    state.glyph = state.candidate.copy()
    state.glyph[1:12, 1:12] = 185
    state.scores = [0.5, 0.4, 0.3]
    state.frozen_boxes = None
    pipeline.line_detector = lambda _image: state.proposals

    def restore(source, donor, boxes, *, protected_boxes):
        state.frozen_boxes = boxes
        assert protected_boxes == ((20, 20, 30, 30),)
        return state.glyph.copy(), np.zeros((64, 64), np.uint8), []

    monkeypatch.setattr(text_shapes, "restore_contour_glyphs", restore)
    return pipeline, state


def test_integrated_text_improves_the_selected_alignment_with_one_global_pass(text_recipe):
    pipeline, state = text_recipe
    result = pipeline.run(state.source, strength=0.2, seed=0)
    assert np.array_equal(np.asarray(result), state.glyph)
    assert pipeline.last_text_quality["selected"]
    assert pipeline.last_text_quality["candidate_mse"] < pipeline.last_text_quality["parent_mse"]
    assert state.events.count(("global", 0.2, 0)) == 1
    assert state.events.count("metric") == 3


@pytest.mark.parametrize("score", [0.45, 0.5, float("nan"), -1.0, True])
def test_text_metric_regression_or_invalid_retains_selected_alignment(text_recipe, score):
    pipeline, state = text_recipe
    state.scores[-1] = score
    result = pipeline.run(state.source, strength=0.2, seed=0)
    assert np.array_equal(np.asarray(result), state.candidate)
    assert not pipeline.last_text_quality["selected"]


def test_text_metric_tie_with_parent_is_allowed_when_baseline_improves(text_recipe):
    pipeline, state = text_recipe
    state.scores[-1] = 0.4
    assert np.array_equal(np.asarray(pipeline.run(state.source, strength=0.2, seed=0)), state.glyph)


def test_text_pixel_regression_retains_parent_without_extra_metric(text_recipe):
    pipeline, state = text_recipe
    state.glyph[1:12, 1:12] = 240
    result = pipeline.run(state.source, strength=0.2, seed=0)
    assert np.array_equal(np.asarray(result), state.candidate)
    assert state.events.count("metric") == 2
    assert pipeline.last_text_quality["reason"] == "pixel_error_did_not_improve"


def test_unchanged_text_avoids_extra_metric(text_recipe):
    pipeline, state = text_recipe
    state.glyph = state.candidate.copy()
    pipeline.run(state.source, strength=0.2, seed=0)
    assert state.events.count("metric") == 2
    assert pipeline.last_text_quality["reason"] == "unchanged"


def test_text_face_postcondition_is_independent_of_renderer(text_recipe):
    pipeline, state = text_recipe
    state.glyph[20:30, 20:30] = 0
    with pytest.raises(RuntimeError, match="Text reconstruction changed protected"):
        pipeline.run(state.source, strength=0.2, seed=0)


def test_text_geometry_is_frozen_before_inference_and_callback_gets_a_copy(text_recipe, monkeypatch):
    pipeline, state = text_recipe

    def detect(image):
        image.putpixel((0, 0), (0, 0, 0))
        return state.proposals

    pipeline.line_detector = detect
    original = pipeline._run_global

    def global_pass(*args):
        state.proposals[0][:] = [2, 2, 13, 13]
        return original(*args)

    monkeypatch.setattr(pipeline, "_run_global", global_pass)
    pipeline.run(state.source, strength=0.2, seed=0)
    assert state.frozen_boxes == ((1, 1, 12, 12),)
    assert state.source.getpixel((0, 0)) == (180, 180, 180)


@pytest.mark.parametrize("box", [[True, 1, 12, 12], [1.0, 1, 12, 12], [1, 1, 65, 12], [1, 1, 1, 12], [1, 1, 12]])
def test_invalid_text_geometry_fails_before_inference(recipe, box):
    pipeline, state = recipe
    pipeline.line_detector = lambda _image: [box]
    with pytest.raises(ValueError, match="Text box"):
        pipeline.run(state.source, strength=0.2, seed=0)
    assert not state.events


def test_text_metric_mutation_cannot_modify_source_parent_or_selected_pixels(text_recipe):
    pipeline, state = text_recipe
    state.mutate = True
    result = pipeline.run(state.source, strength=0.2, seed=0)
    assert state.source.getpixel((0, 0)) == (180, 180, 180)
    assert result.getpixel((0, 0)) == (190, 190, 190)
    assert state.baseline.getpixel((0, 0)) == (200, 200, 200)


def test_text_can_improve_when_alignment_is_rejected_for_pixel_error(text_recipe):
    pipeline, state = text_recipe
    state.candidate[:] = 240
    state.candidate[20:30, 20:30] = 220
    state.scores = [0.5, 0.3]
    result = pipeline.run(state.source, strength=0.2, seed=0)
    assert np.array_equal(np.asarray(result), state.glyph)
    assert not pipeline.last_quality["selected"]
    assert pipeline.last_text_quality["selected"]


def test_request_failure_clears_previous_text_decision(text_recipe):
    pipeline, state = text_recipe
    pipeline.run(state.source, strength=0.2, seed=0)
    assert pipeline.last_text_quality["selected"]
    pipeline.line_detector = lambda _image: [(1, 1, 999, 999)]
    with pytest.raises(ValueError, match="Text box outside image"):
        pipeline.run(state.source, strength=0.2, seed=0)
    assert pipeline.last_text_quality is None


@pytest.fixture
def color_recipe(recipe, monkeypatch):
    from remove_ai_watermarks._internal import text_shapes

    pipeline, state = recipe
    state.color = state.candidate.copy()
    state.color[1:12, 1:12] = 185
    state.mask = np.zeros((64, 64), np.uint8)
    state.mask[1:12, 1:12] = 255

    def neutral(*_args, **_kwargs):
        pipeline.last_text_quality = {
            "diagnostics": [{"box": (1, 1, 12, 12), "reason": "colored or low contrast glyphs", "color_eligible": True}]
        }
        return Image.fromarray(state.candidate)

    def restore(source, donor, boxes, *, protected_boxes, allow_colored):
        assert allow_colored
        assert boxes == ((1, 1, 12, 12),)
        return state.color.copy(), state.mask.copy(), []

    monkeypatch.setattr(pipeline, "_finish_neutral_text", neutral)
    monkeypatch.setattr(text_shapes, "restore_contour_glyphs", restore)
    return pipeline, state


def test_color_stage_selects_a_fidelity_improvement(color_recipe):
    pipeline, state = color_recipe
    result = pipeline.run(state.source, strength=0.2, seed=0)
    assert np.array_equal(np.asarray(result), state.color)
    assert pipeline.last_color_quality["selected"]


@pytest.mark.parametrize("scores", [[0.4, 0.5], [0.4, 0.4], [0.4, float("nan")], [0.4, True]])
def test_color_stage_rejects_regression_ties_and_invalid_metrics(color_recipe, scores):
    pipeline, state = color_recipe
    state.scores = scores
    result = pipeline.run(state.source, strength=0.2, seed=0)
    assert np.array_equal(np.asarray(result), state.candidate)
    assert not pipeline.last_color_quality["selected"]


def test_color_stage_rejects_pixel_error_before_metric_calls(color_recipe):
    pipeline, state = color_recipe
    state.color[1:12, 1:12] = 240
    result = pipeline.run(state.source, strength=0.2, seed=0)
    assert np.array_equal(np.asarray(result), state.candidate)
    assert "metric" not in state.events


def test_color_stage_independently_guards_pixels_outside_support(color_recipe):
    pipeline, state = color_recipe
    state.color[0, 0] = 185
    with pytest.raises(RuntimeError, match="outside its support"):
        pipeline.run(state.source, strength=0.2, seed=0)


def test_color_stage_independently_guards_protected_regions(color_recipe):
    pipeline, state = color_recipe
    state.color[20:30, 20:30] = 185
    state.mask[20:30, 20:30] = 255
    with pytest.raises(RuntimeError, match="protected baseline pixels"):
        pipeline.run(state.source, strength=0.2, seed=0)


def test_color_stage_metric_copies_preserve_source_and_returned_pixels(color_recipe):
    pipeline, state = color_recipe
    state.mutate = True
    result = pipeline.run(state.source, strength=0.2, seed=0)
    assert state.source.getpixel((0, 0)) == (180, 180, 180)
    assert result.getpixel((0, 0)) == (190, 190, 190)


def test_color_dispatch_does_not_depend_on_diagnostic_wording(color_recipe, monkeypatch):
    pipeline, state = color_recipe
    original = pipeline._finish_neutral_text

    def neutral(*args, **kwargs):
        parent = original(*args, **kwargs)
        pipeline.last_text_quality["diagnostics"][0]["reason"] = "A human-readable explanation"
        return parent

    monkeypatch.setattr(pipeline, "_finish_neutral_text", neutral)
    result = pipeline.run(state.source, strength=0.2, seed=0)
    assert np.array_equal(np.asarray(result), state.color)
    assert pipeline.last_color_quality["selected"]


def test_color_dispatch_requires_explicit_eligibility(color_recipe, monkeypatch):
    pipeline, state = color_recipe
    original = pipeline._finish_neutral_text

    def neutral(*args, **kwargs):
        parent = original(*args, **kwargs)
        pipeline.last_text_quality["diagnostics"][0]["color_eligible"] = False
        return parent

    monkeypatch.setattr(pipeline, "_finish_neutral_text", neutral)
    result = pipeline.run(state.source, strength=0.2, seed=0)
    assert np.array_equal(np.asarray(result), state.candidate)
    assert pipeline.last_color_quality is None
