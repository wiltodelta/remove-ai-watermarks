"""Tests for the Qwen 2512 + Z-Image SynthID removal profile."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image


def _mock_watermark_runtime_deps(monkeypatch):
    """Bypass optional GPU imports while testing Qwen Z-Image routing."""
    from remove_ai_watermarks._internal import watermark_remover

    fake_torch = MagicMock()
    fake_torch.float16 = object()
    fake_torch.float32 = object()
    monkeypatch.setattr(watermark_remover, "torch", fake_torch)
    monkeypatch.setattr(watermark_remover, "_HAS_TORCH", False)
    monkeypatch.setattr(watermark_remover, "is_watermark_removal_available", lambda: True)


def test_resolution_adaptive_denoise_preserves_calibrated_values():
    from remove_ai_watermarks._internal.two_stage_pipeline import resolution_adaptive_denoise

    assert resolution_adaptive_denoise(600, 500, adaptive_level=6) == pytest.approx(0.084)
    assert resolution_adaptive_denoise(2000, 1850, adaptive_level=6) == pytest.approx(0.154)


def test_largest_face_denoise_preserves_calibrated_values():
    from remove_ai_watermarks._internal.two_stage_pipeline import largest_face_denoise

    image_size = (1000, 1000)
    assert largest_face_denoise([(0, 0, 300, 100)], image_size) == pytest.approx(0.10)
    assert largest_face_denoise([(0, 0, 150, 100)], image_size) == pytest.approx(0.05)
    assert largest_face_denoise([(0, 0, 900, 900)], image_size) == pytest.approx(0.28)


def test_global_kwargs_use_lightning_and_diffsynth_controlnet_shape():
    from remove_ai_watermarks._internal.qwen_zimage_pipeline import build_global_kwargs

    image = Image.new("RGB", (1122, 1402))
    kwargs = build_global_kwargs(image, strength=0.11, seed=7, controlnet_input="CONTROL")

    assert kwargs["input_image"].size == (1120, 1392)
    assert kwargs["blockwise_controlnet_inputs"] == ["CONTROL"]
    assert kwargs["denoising_strength"] == 0.11
    assert kwargs["num_inference_steps"] == 4
    assert kwargs["cfg_scale"] == 1.0
    assert kwargs["seed"] == 7
    assert kwargs["width"] == 1120
    assert kwargs["height"] == 1392
    assert kwargs["exponential_shift_mu"] == pytest.approx(math.log(3.0))
    assert kwargs["prompt"] == "ultra clear and smoothe skin, spotless skin"
    assert kwargs["negative_prompt"] == "moles, freckes, high detail skin"


def test_face_kwargs_use_project_zimage_settings():
    from remove_ai_watermarks._internal.two_stage_pipeline import build_face_kwargs

    crop = Image.new("RGB", (713, 941))
    kwargs = build_face_kwargs(crop, strength=0.17, seed=9)

    assert kwargs["input_image"].size == (704, 928)
    assert kwargs["denoising_strength"] == 0.17
    assert kwargs["num_inference_steps"] == 8
    assert kwargs["cfg_scale"] == 1.0
    assert kwargs["seed"] == 9
    assert kwargs["width"] == 704
    assert kwargs["height"] == 928
    assert kwargs["prompt"] == ""
    assert kwargs["negative_prompt"] == "blurry, ugly, bad quality,"


def test_canny_control_image_is_three_channel_and_detects_an_edge():
    from remove_ai_watermarks._internal.two_stage_pipeline import build_canny_control_image

    source = np.zeros((64, 80, 3), dtype=np.uint8)
    source[:, 40:] = 255
    result = np.asarray(build_canny_control_image(Image.fromarray(source)))

    assert result.shape == (64, 80, 3)
    assert np.array_equal(result[:, :, 0], result[:, :, 1])
    assert np.array_equal(result[:, :, 1], result[:, :, 2])
    import cv2

    expected = cv2.Canny(cv2.cvtColor(source, cv2.COLOR_RGB2GRAY), 13, 64)
    assert np.array_equal(result[:, :, 0], expected)


def test_face_crop_geometry_preserves_calibrated_values():
    from remove_ai_watermarks._internal.qwen_zimage_pipeline import QwenZImagePipeline
    from remove_ai_watermarks._internal.two_stage_pipeline import _expanded_box

    assert _expanded_box((100, 100, 200, 200), (500, 500)) == (25, 25, 275, 275)
    assert QwenZImagePipeline._detail_size((500, 400), (100, 80)) == (1024, 816)


def test_yunet_download_targets_verified_lfs_artifact():
    from remove_ai_watermarks._internal.two_stage_pipeline import (
        YUNET_MODEL_SHA256,
        YUNET_MODEL_URL,
        YUNET_SCORE_THRESHOLD,
    )

    assert YUNET_MODEL_URL.startswith("https://media.githubusercontent.com/media/opencv/opencv_zoo/")
    assert YUNET_MODEL_SHA256 == "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    assert pytest.approx(0.5) == YUNET_SCORE_THRESHOLD


def test_global_stack_is_resident_on_a_large_card_and_streams_on_a_small_one():
    """The mandatory Qwen stack must not stream from disk on a card that can hold it.

    DiffSynth offloads by dropping weights to the meta device and re-reading every
    parameter through its DiskMap, so the streaming config costs a full model reload
    on each stage transition. Benchmark in docs/module-internals.md, "CPU offload".
    """
    import torch

    from remove_ai_watermarks._internal.qwen_zimage_pipeline import (
        QwenZImagePipeline,
        resolve_global_model_residency,
    )

    assert resolve_global_model_residency(None, total_memory_gib=79.2) is True
    assert resolve_global_model_residency(None, total_memory_gib=39.5) is False
    assert resolve_global_model_residency(False, total_memory_gib=79.2) is False
    assert resolve_global_model_residency(True, total_memory_gib=39.5) is True

    large = QwenZImagePipeline(
        device="cuda",
        torch_dtype=torch.bfloat16,
        keep_global_models_on_device=True,
    )._qwen_vram_config()
    # No "disk" anywhere: DiffSynth latches disk_offload from offload_dtype once, so
    # leaving it in would keep the meta-drop even with every device set to cuda.
    assert "disk" not in large.values()
    assert large["offload_device"] == "cuda"
    assert large["onload_device"] == "cuda"
    assert large["computation_dtype"] is torch.bfloat16

    small = QwenZImagePipeline(
        device="cuda",
        torch_dtype=torch.bfloat16,
        keep_global_models_on_device=False,
    )._qwen_vram_config()
    assert small["offload_dtype"] == "disk"
    assert small["offload_device"] == "disk"
    assert small["onload_device"] == "cpu"


@pytest.mark.parametrize(
    ("cpu_offload", "expected"),
    [(True, False), (False, None)],
)
def test_cpu_offload_forces_both_stacks_to_stream(monkeypatch, cpu_offload, expected):
    """``cpu_offload`` is the caller's escape hatch and must cover the global stack too.

    Without the global flag it silenced only the face stack, so a caller asking for
    low VRAM still got the larger global stack pinned.
    """
    from remove_ai_watermarks._internal import qwen_zimage_pipeline as pipeline_module
    from remove_ai_watermarks._internal import watermark_remover as module

    captured: dict[str, object] = {}

    class Recorder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    # `_load_qwen_zimage_pipeline` imports the class inside the function body, so the
    # patch has to land on the defining module rather than on watermark_remover.
    monkeypatch.setattr(pipeline_module, "QwenZImagePipeline", Recorder)

    remover = module.WatermarkRemover.__new__(module.WatermarkRemover)
    remover.model_profile = "qwen-zimage"
    remover.device = "cuda"
    remover.torch_dtype = None
    remover.hf_token = None
    remover._progress_callback = None
    remover.controlnet_conditioning_scale = 1.0
    remover.cpu_offload = cpu_offload
    remover._qwen_zimage_pipeline = None

    remover._load_qwen_zimage_pipeline()

    assert captured["keep_global_models_on_device"] is expected
    assert captured["keep_face_models_on_device"] is expected


def test_face_stage_loads_in_its_own_dtype_when_the_global_stage_differs(monkeypatch, tmp_path):
    """A subclass that changes the pipeline dtype must not change the face stage's.

    ``sdxl-zimage`` is constructed fp16 for its global model. That dtype used to reach
    the inherited ``_load_zimage``, which builds its modules bf16 from
    ``_zimage_vram_config``, so Z-Image got fp16 latents into bf16 convolutions and
    every image containing a face died in the VAE. Zero-face inputs never enter the
    face stage, so the profile looked healthy right up to the first portrait.

    Asserts the dtype the loaders actually RECEIVE. Comparing the accessor against the
    config it is derived from would restate the implementation and pass for any
    consistently-wrong value.
    """
    import torch

    # The loaders this asserts on are monkeypatched, not called, but the modules still
    # have to be importable to be patched -- and CI's base job installs the library
    # without the qwen-zimage extra.
    transformers = pytest.importorskip("transformers")
    z_image = pytest.importorskip("diffsynth.pipelines.z_image")

    from remove_ai_watermarks._internal.sdxl_zimage_pipeline import SdxlZImagePipeline

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    captured: dict[str, object] = {}

    def fake_zimage(**kwargs):
        captured["zimage"] = kwargs["torch_dtype"]
        return MagicMock(units=[])

    def fake_sam(_model_id, **kwargs):
        captured["sam"] = kwargs["torch_dtype"]
        return MagicMock()

    monkeypatch.setattr(z_image.ZImagePipeline, "from_pretrained", staticmethod(fake_zimage))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: MagicMock()))
    monkeypatch.setattr(transformers.AutoModelForMaskGeneration, "from_pretrained", staticmethod(fake_sam))

    pipeline = SdxlZImagePipeline(device="cuda", torch_dtype=torch.float16)
    pipeline._load_zimage()
    pipeline._load_sam()

    assert pipeline.torch_dtype == torch.float16, "the global stage keeps its own dtype"
    # Z-Image is the one that crashed; SAM never did, but it read the same wrong field.
    assert captured["zimage"] == torch.bfloat16
    assert captured["sam"] == torch.bfloat16


def test_resident_face_models_disable_vram_offload():
    from remove_ai_watermarks._internal.qwen_zimage_pipeline import QwenZImagePipeline
    from remove_ai_watermarks._internal.two_stage_pipeline import (
        _pin_vram_managed_models,
        resolve_face_model_residency,
    )

    config = QwenZImagePipeline._zimage_vram_config()

    assert config["offload_device"] == "cpu"
    assert config["onload_device"] == "cpu"
    assert config["preparing_device"] == "cuda"
    assert config["computation_device"] == "cuda"
    assert resolve_face_model_residency(None, total_memory_gib=79.2) is True
    assert resolve_face_model_residency(None, total_memory_gib=39.5) is False
    assert resolve_face_model_residency(False, total_memory_gib=79.2) is False
    assert resolve_face_model_residency(True, total_memory_gib=39.5) is True

    class ManagedModule:
        offload_dtype = "bf16"
        offload_device = "cpu"
        onload_dtype = "bf16"
        onload_device = "cpu"
        preparing_dtype = "bf16"
        preparing_device = "cuda"
        computation_dtype = "bf16"
        computation_device = "cuda"

        def modules(self):
            return [self]

    class Pipe:
        text_encoder = ManagedModule()
        dit = ManagedModule()
        vae_encoder = ManagedModule()
        vae_decoder = ManagedModule()

        def __init__(self):
            self.loaded = None

        def load_models_to_device(self, names):
            self.loaded = names

    pipe = Pipe()
    _pin_vram_managed_models(pipe)

    assert pipe.loaded == ["text_encoder", "dit", "vae_encoder", "vae_decoder"]
    assert pipe.dit.offload_device == "cuda"
    assert pipe.dit.onload_device == "cuda"
    assert pipe.dit.preparing_device == "cuda"


def test_static_prompt_cache_reuses_embeddings_without_caching_image_edits():
    from remove_ai_watermarks._internal.two_stage_pipeline import _cache_static_prompt_embeddings

    class PromptUnit:
        output_params = ("prompt_embeds",)

        def __init__(self):
            self.calls = 0

        def process(self, _pipe, prompt, edit_image=None):
            self.calls += 1
            return {"prompt_embeds": [object()], "prompt": prompt, "edit_image": edit_image}

    unit = PromptUnit()
    pipe = MagicMock()
    pipe.units = [unit]

    assert _cache_static_prompt_embeddings(pipe, ("prompt_embeds",)) is True
    first = unit.process(pipe, "constant")
    second = unit.process(pipe, "constant")
    different = unit.process(pipe, "different")
    edited_first = unit.process(pipe, "constant", edit_image=object())
    edited_second = unit.process(pipe, "constant", edit_image=object())

    assert first is second
    assert first is not different
    assert edited_first is not edited_second
    assert unit.calls == 4


def test_prompt_cache_path_is_keyed_by_version_model_outputs_and_prompt(monkeypatch, tmp_path):
    """A model, prompt or format change must not read a stale embedding."""
    from remove_ai_watermarks._internal import two_stage_pipeline as ts

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    baseline = ts._prompt_cache_path("model/a", ("prompt_emb",), "text")

    assert baseline.parent == tmp_path / "remove-ai-watermarks" / "prompt-embeddings"
    assert baseline == ts._prompt_cache_path("model/a", ("prompt_emb",), "text")
    assert baseline != ts._prompt_cache_path("model/b", ("prompt_emb",), "text")
    assert baseline != ts._prompt_cache_path("model/a", ("prompt_embeds",), "text")
    assert baseline != ts._prompt_cache_path("model/a", ("prompt_emb",), "other")

    monkeypatch.setattr(ts, "_PROMPT_CACHE_VERSION", ts._PROMPT_CACHE_VERSION + 1)
    assert baseline != ts._prompt_cache_path("model/a", ("prompt_emb",), "text")


def test_model_cache_dir_prefers_the_persistent_hugging_face_root(monkeypatch, tmp_path):
    """A scale-to-zero runner only mounts HF_HOME, so it must win over XDG."""
    from remove_ai_watermarks._internal.two_stage_pipeline import _model_cache_dir

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("HF_HOME", raising=False)
    assert _model_cache_dir() == tmp_path / "xdg" / "remove-ai-watermarks"

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert _model_cache_dir() == tmp_path / "hf" / "remove-ai-watermarks"


def test_stored_prompt_embedding_round_trips_without_casting_the_mask(monkeypatch, tmp_path):
    """The mask rides in the same payload and is integer; casting it corrupts the prompt."""
    import torch

    from remove_ai_watermarks._internal import two_stage_pipeline as ts
    from remove_ai_watermarks._internal.qwen_zimage_pipeline import _QWEN_PROMPT_OUTPUTS

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    path = ts._prompt_cache_path("model/a", _QWEN_PROMPT_OUTPUTS, "text")
    payload = {
        "prompt_emb": torch.ones((1, 2, 3), dtype=torch.float32),
        "prompt_emb_mask": torch.ones((1, 2), dtype=torch.int64),
    }
    ts._store_prompt_payload(path, payload)
    restored = ts._load_prompt_payload(path, "cpu", torch.bfloat16)

    assert restored["prompt_emb"].dtype == torch.bfloat16
    assert restored["prompt_emb_mask"].dtype == torch.int64
    assert torch.equal(restored["prompt_emb"].float(), payload["prompt_emb"])


def test_persisted_prompt_cache_lets_a_second_pipeline_skip_the_text_encoder(monkeypatch, tmp_path):
    """The whole point: container two must not call the encoder container one ran."""
    import torch

    from remove_ai_watermarks._internal import two_stage_pipeline as ts

    monkeypatch.setenv("HF_HOME", str(tmp_path))

    class PromptUnit:
        output_params = ts._ZIMAGE_PROMPT_OUTPUTS

        def __init__(self):
            self.calls = 0

        def process(self, _pipe, prompt, edit_image=None):
            self.calls += 1
            return {"prompt_embeds": [torch.ones((2, 2), dtype=torch.float32)]}

    def build():
        unit = PromptUnit()
        pipe = MagicMock(units=[unit], device="cpu", torch_dtype=torch.float32)
        return unit, pipe

    first_unit, first_pipe = build()
    ts._cache_static_prompt_embeddings(first_pipe, ts._ZIMAGE_PROMPT_OUTPUTS, model_id="model/a", require_cache=False)
    first_unit.process(first_pipe, ts._FACE_PROMPT)
    assert first_unit.calls == 1

    second_unit, second_pipe = build()
    ts._cache_static_prompt_embeddings(second_pipe, ts._ZIMAGE_PROMPT_OUTPUTS, model_id="model/a", require_cache=True)
    restored = second_unit.process(second_pipe, ts._FACE_PROMPT)

    assert second_unit.calls == 0
    assert torch.equal(restored["prompt_embeds"][0], torch.ones((2, 2)))


def test_a_missing_cache_fails_loudly_once_the_text_encoder_was_left_out(monkeypatch, tmp_path):
    """Silently calling an absent text encoder would surface as an opaque crash."""
    import torch

    from remove_ai_watermarks._internal import two_stage_pipeline as ts
    from remove_ai_watermarks._internal.qwen_zimage_pipeline import _QWEN_PROMPT_OUTPUTS

    monkeypatch.setenv("HF_HOME", str(tmp_path))

    class PromptUnit:
        output_params = _QWEN_PROMPT_OUTPUTS

        def process(self, _pipe, prompt, edit_image=None):
            raise AssertionError("the text encoder is not loaded")

    unit = PromptUnit()
    pipe = MagicMock(units=[unit], device="cpu", torch_dtype=torch.float32)
    ts._cache_static_prompt_embeddings(pipe, _QWEN_PROMPT_OUTPUTS, model_id="model/a", require_cache=True)

    with pytest.raises(RuntimeError, match="disappeared"):
        unit.process(pipe, "never cached")


def test_sam_pixels_match_model_dtype_without_casting_boxes():
    import torch

    from remove_ai_watermarks._internal.two_stage_pipeline import _prepare_sam_inputs

    class Inputs(dict[str, torch.Tensor]):
        def to(self, device: str):
            return Inputs({name: value.to(device) for name, value in self.items()})

    inputs = Inputs(
        {
            "pixel_values": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
            "input_boxes": torch.zeros((1, 1, 4), dtype=torch.float32),
        }
    )

    prepared = _prepare_sam_inputs(inputs, "cpu", torch.bfloat16)

    assert prepared["pixel_values"].dtype == torch.bfloat16
    assert prepared["input_boxes"].dtype == torch.float32


def test_sam_prompts_match_impact_center_and_clip_masks_to_boxes():
    from remove_ai_watermarks._internal.two_stage_pipeline import (
        _clip_sam_masks_to_boxes,
        _sam_point_prompts,
    )

    boxes = [(2, 3, 8, 9), (10, 4, 16, 12)]
    points, labels = _sam_point_prompts(boxes)
    masks = [np.full((14, 18), 255, dtype=np.uint8) for _box in boxes]

    clipped = _clip_sam_masks_to_boxes(masks, boxes, (18, 14))

    assert points == [[[[5.0, 6.0]], [[13.0, 8.0]]]]
    assert labels == [[[1], [1]]]
    assert np.count_nonzero(clipped[0]) == 36
    assert np.count_nonzero(clipped[1]) == 48
    assert clipped[0][2, 2] == 0
    assert clipped[0][3, 2] == 255


def test_sam_proposal_selection_matches_impact_sub_threshold():
    from remove_ai_watermarks._internal.two_stage_pipeline import _select_sam_masks

    masks = np.zeros((2, 3, 8, 8), dtype=np.float32)
    masks[0, 0, 1:3, 1:3] = 1.0
    masks[0, 1, 4:7, 4:7] = 1.0
    masks[0, 2, :, :] = 1.0
    masks[1, 0, :, :] = 1.0
    masks[1, 1, 1:5, 1:5] = 1.0
    masks[1, 2, 2:4, 2:5] = 1.0
    scores = np.asarray(
        [
            [0.95, 0.94, 0.50],
            [0.50, 0.70, 0.90],
        ],
        dtype=np.float32,
    )

    selected = _select_sam_masks(masks, scores)

    # The first face unions both proposals over 0.93. The second has none over
    # 0.93, so it falls back to its single highest-IoU proposal.
    assert np.count_nonzero(selected[0]) == 13
    assert np.count_nonzero(selected[1]) == 6
    assert selected[0][6, 6] == 255
    assert selected[1][1, 1] == 0


def test_sam_bfloat16_outputs_convert_to_numpy_float32():
    import torch

    from remove_ai_watermarks._internal.two_stage_pipeline import _sam_outputs_to_numpy

    masks = torch.ones((1, 2, 3, 4, 4), dtype=torch.bfloat16)
    scores = torch.tensor([[[0.95, 0.75, 0.50], [0.99, 0.80, 0.60]]], dtype=torch.bfloat16)

    mask_array, score_array = _sam_outputs_to_numpy(masks, scores)

    assert mask_array.dtype == np.float32
    assert score_array.dtype == np.float32
    assert score_array[0, 0, 0] == pytest.approx(0.94921875)


def test_face_composite_preserves_every_pixel_outside_mask():
    from remove_ai_watermarks._internal.two_stage_pipeline import composite_face

    base = np.full((32, 32, 3), 10, dtype=np.uint8)
    detail = np.full((32, 32, 3), 240, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[12:20, 12:20] = 255

    result = composite_face(base, detail, mask, feather=0)

    assert np.array_equal(result[:12], base[:12])
    assert np.array_equal(result[:, :12], base[:, :12])
    assert np.all(result[12:20, 12:20] == 240)


def test_profile_defaults_to_four_global_steps_and_a_fixed_seed():
    """The step count belongs to the stage, not to a caller-settable profile knob."""
    from remove_ai_watermarks._internal.qwen_zimage_pipeline import GLOBAL_STEPS
    from remove_ai_watermarks._internal.watermark_profiles import normalize_profile, resolve_seed

    assert normalize_profile("qwen-zimage") == "qwen-zimage"
    assert GLOBAL_STEPS == 4
    assert resolve_seed(None) == 0
    assert resolve_seed(17) == 17


def test_cli_exposes_qwen_zimage_profile():
    from remove_ai_watermarks.cli import _PIPELINE_CHOICES

    assert "qwen-zimage" in _PIPELINE_CHOICES


def test_cli_qwen_zimage_keeps_profile_postprocess_default(tmp_image_path, monkeypatch):
    from remove_ai_watermarks import cli

    mock_engine = MagicMock()
    mock_engine.remove_watermark.return_value = tmp_image_path
    monkeypatch.setattr("remove_ai_watermarks.invisible_engine.is_available", lambda: True)
    monkeypatch.setattr("remove_ai_watermarks.invisible_engine.InvisibleEngine", MagicMock(return_value=mock_engine))

    result = CliRunner().invoke(
        cli.main,
        ["invisible", str(tmp_image_path), "--pipeline", "qwen-zimage", "--force"],
    )

    assert result.exit_code == 0, result.output
    # Both defaults are the profile's, resolved once by the library rather than
    # pre-resolved here: the CLI passes them through unset so a library caller on the
    # same profile gets the same answer.
    assert mock_engine.remove_watermark.call_args.kwargs["adaptive_polish"] is None
    assert mock_engine.remove_watermark.call_args.kwargs["seed"] is None

    result = CliRunner().invoke(
        cli.main,
        [
            "invisible",
            str(tmp_image_path),
            "--pipeline",
            "qwen-zimage",
            "--adaptive-polish",
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    assert mock_engine.remove_watermark.call_args.kwargs["adaptive_polish"] is True


def test_cli_forwards_verified_text_manifest(tmp_image_path, tmp_path, monkeypatch):
    from remove_ai_watermarks import cli

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    mock_engine = MagicMock()
    mock_engine.remove_watermark.return_value = tmp_image_path
    monkeypatch.setattr("remove_ai_watermarks.invisible_engine.is_available", lambda: True)
    monkeypatch.setattr("remove_ai_watermarks.invisible_engine.InvisibleEngine", MagicMock(return_value=mock_engine))

    result = CliRunner().invoke(
        cli.main,
        ["invisible", str(tmp_image_path), "--text-manifest", str(manifest), "--force"],
    )

    assert result.exit_code == 0, result.output
    assert mock_engine.remove_watermark.call_args.kwargs["text_manifest"] == manifest


def test_cli_reports_verified_text_manifest_errors(tmp_image_path, tmp_path, monkeypatch):
    from remove_ai_watermarks import cli

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    mock_engine = MagicMock()
    mock_engine.remove_watermark.side_effect = ValueError("manifest pixels do not match")
    monkeypatch.setattr("remove_ai_watermarks.invisible_engine.is_available", lambda: True)
    monkeypatch.setattr("remove_ai_watermarks.invisible_engine.InvisibleEngine", MagicMock(return_value=mock_engine))

    result = CliRunner().invoke(
        cli.main,
        ["invisible", str(tmp_image_path), "--text-manifest", str(manifest), "--force"],
    )

    assert result.exit_code == 1
    assert "manifest pixels do not match" in result.output


def test_no_face_path_still_runs_verified_text_restoration(monkeypatch):
    from remove_ai_watermarks._internal import text_restoration, two_stage_pipeline
    from remove_ai_watermarks._internal.qwen_zimage_pipeline import QwenZImagePipeline
    from remove_ai_watermarks._internal.text_restoration import VerifiedTextLine, VerifiedTextManifest

    pipeline = object.__new__(QwenZImagePipeline)
    pipeline.device = "cuda"
    pipeline.progress_callback = None
    source = Image.new("RGB", (32, 32), (10, 20, 30))
    donor = Image.new("RGB", (32, 32), (40, 50, 60))
    global_result = Image.new("RGB", (32, 32), (70, 80, 90))
    anchor = Image.new("RGB", (32, 32), (100, 110, 120))
    restored = Image.new("RGB", (32, 32), (130, 140, 150))
    pipeline._vae_roundtrip = MagicMock(return_value=donor)
    pipeline._run_global = MagicMock(return_value=global_result)
    monkeypatch.setattr(two_stage_pipeline, "detect_faces", lambda _image: [])
    blend = MagicMock(return_value=anchor)
    restore = MagicMock(return_value=restored)
    monkeypatch.setattr(text_restoration, "blend_fidelity_anchor", blend)
    monkeypatch.setattr(text_restoration, "restore_verified_text", restore)
    manifest = VerifiedTextManifest(
        "0" * 64,
        32,
        32,
        (VerifiedTextLine((4, 4, 20, 16), "Exact", "alphabetic"),),
    )

    result = pipeline.run(source, strength=0.1, seed=0, text_manifest=manifest)

    assert result is restored
    # Off by default since 0.27.1 (leak finding, docs/text-protection-research.md):
    # no whole-frame donor blend; the raw global result feeds restoration.
    blend.assert_not_called()
    restore.assert_called_once_with(source, global_result, donor, manifest.lines)

    result = pipeline.run(source, strength=0.1, seed=0, text_manifest=manifest, fidelity_anchor=True)

    assert result is restored
    blend.assert_called_once_with(global_result, donor)
    restore.assert_called_with(source, anchor, donor, manifest.lines)


def test_tiled_verified_text_is_rejected_before_model_work():
    from remove_ai_watermarks._internal.qwen_zimage_pipeline import QwenZImagePipeline
    from remove_ai_watermarks._internal.text_restoration import VerifiedTextLine, VerifiedTextManifest

    pipeline = object.__new__(QwenZImagePipeline)
    pipeline.device = "cuda"
    pipeline.progress_callback = None
    source = Image.new("RGB", (96, 80), (10, 20, 30))
    manifest = VerifiedTextManifest(
        "0" * 64,
        96,
        80,
        (VerifiedTextLine((4, 4, 20, 16), "Exact", "alphabetic"),),
    )

    with pytest.raises(ValueError, match="not calibrated with tiled diffusion"):
        pipeline.run(
            source,
            strength=0.1,
            seed=0,
            tile=True,
            tile_size=64,
            tile_overlap=16,
            text_manifest=manifest,
        )


def test_watermark_remover_dispatches_to_full_pipeline(tmp_path, monkeypatch):
    from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover

    _mock_watermark_runtime_deps(monkeypatch)
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    Image.new("RGB", (64, 48), (20, 30, 40)).save(source)

    runtime = MagicMock()
    runtime.run.return_value = Image.new("RGB", (64, 48), (50, 60, 70))
    remover = WatermarkRemover(device="cuda", pipeline="qwen-zimage")
    monkeypatch.setattr(remover, "_load_qwen_zimage_pipeline", lambda: runtime)

    remover.remove_watermark(
        source,
        output,
    )

    runtime.run.assert_called_once()
    _, kwargs = runtime.run.call_args
    assert kwargs["strength"] == pytest.approx(0.084)
    assert kwargs["seed"] == 0
    assert kwargs["text_manifest"] is None
    assert output.exists()


def test_watermark_remover_dispatches_qwen_tiling_to_full_pipeline(tmp_path, monkeypatch):
    from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover

    _mock_watermark_runtime_deps(monkeypatch)
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    Image.new("RGB", (96, 80), (20, 30, 40)).save(source)

    runtime = MagicMock()
    runtime.run.return_value = Image.new("RGB", (96, 80), (50, 60, 70))
    remover = WatermarkRemover(device="cuda", pipeline="qwen-zimage")
    monkeypatch.setattr(remover, "_load_qwen_zimage_pipeline", lambda: runtime)

    remover.remove_watermark(
        source,
        output,
        seed=0,
        tile=True,
        tile_size=64,
        tile_overlap=16,
    )

    runtime.run.assert_called_once()
    _, kwargs = runtime.run.call_args
    assert kwargs["seed"] == 0
    assert kwargs["tile"] is True
    assert kwargs["tile_size"] == 64
    assert kwargs["tile_overlap"] == 16
    assert output.exists()


def test_watermark_remover_rejects_tiled_verified_text_before_pipeline(tmp_path, monkeypatch):
    from remove_ai_watermarks._internal.text_restoration import VerifiedTextLine, VerifiedTextManifest
    from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover

    _mock_watermark_runtime_deps(monkeypatch)
    source = tmp_path / "source.png"
    Image.new("RGB", (96, 80), (20, 30, 40)).save(source)
    manifest = VerifiedTextManifest(
        "0" * 64,
        96,
        80,
        (VerifiedTextLine((4, 4, 20, 16), "Exact", "alphabetic"),),
    )
    remover = WatermarkRemover(device="cuda", pipeline="qwen-zimage")
    runtime = MagicMock()
    monkeypatch.setattr(remover, "_load_qwen_zimage_pipeline", lambda: runtime)

    with pytest.raises(ValueError, match="not calibrated with tiled diffusion"):
        remover.remove_watermark(source, text_manifest=manifest, tile=True)

    runtime.run.assert_not_called()


def test_qwen_tiling_runs_global_tiles_then_one_full_frame_face_stage(monkeypatch):
    from remove_ai_watermarks._internal.qwen_zimage_pipeline import QwenZImagePipeline
    from remove_ai_watermarks._internal.tiling import plan_tiles
    from remove_ai_watermarks._internal.two_stage_pipeline import resolution_adaptive_denoise

    image = Image.new("RGB", (1500, 1500), (20, 30, 40))
    runtime = QwenZImagePipeline(device="cuda", torch_dtype="bf16")
    monkeypatch.setattr(runtime, "_require_cuda", lambda: None)

    global_calls = []

    def fake_global(tile, strength, seed):
        global_calls.append((tile.size, strength, seed))
        return tile

    face_stage = MagicMock(return_value=image)
    monkeypatch.setattr(runtime, "_run_global", fake_global)
    monkeypatch.setattr(runtime, "_run_faces", face_stage)
    monkeypatch.setattr(
        "remove_ai_watermarks._internal.two_stage_pipeline.detect_faces",
        lambda _image: [(100, 100, 300, 300)],
    )
    monkeypatch.setattr(runtime, "_sam_masks", lambda _image, _boxes: [np.ones((1500, 1500), dtype=np.uint8)])

    result = runtime.run(
        image,
        strength=None,
        seed=0,
        tile=True,
        tile_size=1024,
        tile_overlap=128,
    )

    expected_tiles = plan_tiles(1500, 1500, 1024, 128)
    assert len(global_calls) == len(expected_tiles) == 4
    assert all(size == (1024, 1024) for size, _strength, _seed in global_calls)
    assert all(strength == pytest.approx(resolution_adaptive_denoise(1500, 1500)) for _, strength, _ in global_calls)
    assert all(seed == 0 for _, _, seed in global_calls)
    face_stage.assert_called_once()
    assert face_stage.call_args.kwargs["strength"] == pytest.approx(0.0296296296)
    assert face_stage.call_args.args[0] is image
    assert face_stage.call_args.args[1].size == image.size
    assert result.size == image.size


def test_global_only_preload_skips_face_models(monkeypatch):
    from remove_ai_watermarks._internal.qwen_zimage_pipeline import QwenZImagePipeline

    runtime = QwenZImagePipeline(device="cuda", torch_dtype="bf16")
    global_stage = MagicMock()
    zimage = MagicMock()
    sam = MagicMock()
    yunet = MagicMock()
    monkeypatch.setattr(runtime, "_load_global", global_stage)
    monkeypatch.setattr(runtime, "_load_zimage", zimage)
    monkeypatch.setattr(runtime, "_load_sam", sam)
    monkeypatch.setattr(
        "remove_ai_watermarks._internal.two_stage_pipeline._yunet_model_path",
        yunet,
    )

    runtime.preload(global_only=True)

    global_stage.assert_called_once_with()
    zimage.assert_not_called()
    sam.assert_not_called()
    yunet.assert_called_once_with()


def test_full_preload_still_loads_face_models(monkeypatch):
    from remove_ai_watermarks._internal.qwen_zimage_pipeline import QwenZImagePipeline

    runtime = QwenZImagePipeline(device="cuda", torch_dtype="bf16")
    global_stage = MagicMock()
    zimage = MagicMock()
    sam = MagicMock()
    yunet = MagicMock()
    monkeypatch.setattr(runtime, "_load_global", global_stage)
    monkeypatch.setattr(runtime, "_load_zimage", zimage)
    monkeypatch.setattr(runtime, "_load_sam", sam)
    monkeypatch.setattr(
        "remove_ai_watermarks._internal.two_stage_pipeline._yunet_model_path",
        yunet,
    )

    runtime.preload()

    global_stage.assert_called_once_with()
    zimage.assert_called_once_with()
    sam.assert_called_once_with()
    yunet.assert_called_once_with()


def test_watermark_remover_forwards_global_only_preload(monkeypatch):
    from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover

    runtime = MagicMock()
    remover = WatermarkRemover.__new__(WatermarkRemover)
    remover.model_profile = "qwen-zimage"
    monkeypatch.setattr(remover, "_load_qwen_zimage_pipeline", lambda: runtime)

    remover.preload(global_only=True)

    runtime.preload.assert_called_once_with(global_only=True)


def test_the_fixed_graph_offers_no_runtime_knob_to_reject(tmp_path, monkeypatch):
    """model_id, steps and CFG are not parameters at any layer.

    They used to be accepted and then rejected, which put the failure several frames
    below the caller and made the surface advertise choices the pinned stack cannot
    honor. TypeError from the signature is the earlier, clearer answer -- and it is
    what keeps a wrapper from threading a value that would silently do nothing.
    """
    from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover

    _mock_watermark_runtime_deps(monkeypatch)
    with pytest.raises(TypeError):
        WatermarkRemover(model_id="custom/model", device="cuda", pipeline="qwen-zimage")  # type: ignore[call-arg]

    source = tmp_path / "source.png"
    Image.new("RGB", (64, 48)).save(source)
    remover = WatermarkRemover(device="cuda", pipeline="qwen-zimage")
    with pytest.raises(TypeError):
        remover.remove_watermark(source, guidance_scale=2.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        remover.remove_watermark(source, num_inference_steps=8)  # type: ignore[call-arg]


def test_invisible_engine_passes_the_seed_but_never_a_step_count(tmp_image_path, tmp_path):
    from remove_ai_watermarks.invisible_engine import InvisibleEngine

    engine = InvisibleEngine.__new__(InvisibleEngine)
    engine._progress_callback = None
    engine._remover = MagicMock(model_profile="qwen-zimage")
    engine._remover.remove_watermark.return_value = tmp_path / "clean.png"

    engine.remove_watermark(tmp_image_path, tmp_path / "clean.png")

    kwargs = engine._remover.remove_watermark.call_args.kwargs
    assert kwargs["seed"] == 0
    assert "num_inference_steps" not in kwargs
    assert "guidance_scale" not in kwargs


def test_sdxl_zimage_strength_is_vendor_adaptive_and_leaves_other_profiles_alone():
    """An SDXL global pass needs more strength than Qwen, so it gets its own policy."""
    from remove_ai_watermarks._internal.watermark_profiles import (
        QWEN_ZIMAGE_GOOGLE_STRENGTH,
        QWEN_ZIMAGE_OPENAI_STRENGTH,
        SDXL_ZIMAGE_GEMINI_STRENGTH,
        SDXL_ZIMAGE_OPENAI_STRENGTH,
        resolve_strength,
    )

    assert resolve_strength(None, "openai", "sdxl-zimage") == pytest.approx(SDXL_ZIMAGE_OPENAI_STRENGTH)
    assert resolve_strength(None, "google", "sdxl-zimage") == pytest.approx(SDXL_ZIMAGE_GEMINI_STRENGTH)
    # Unknown provenance takes the stricter of the two.
    assert resolve_strength(None, None, "sdxl-zimage") == pytest.approx(SDXL_ZIMAGE_GEMINI_STRENGTH)
    # An explicit value still wins. qwen-zimage's unknown cohort keeps the curve,
    # while measured providers take their flat operating points.
    assert resolve_strength(0.4, "google", "sdxl-zimage") == pytest.approx(0.4)
    assert resolve_strength(None, "openai", "qwen-zimage", size=(2000, 1850)) == pytest.approx(
        QWEN_ZIMAGE_OPENAI_STRENGTH
    )
    assert resolve_strength(None, "google", "qwen-zimage", size=(2000, 1850)) == pytest.approx(
        QWEN_ZIMAGE_GOOGLE_STRENGTH
    )


def test_sdxl_zimage_shares_the_fixed_seed_contract():
    from remove_ai_watermarks._internal.watermark_profiles import normalize_profile, resolve_seed

    assert normalize_profile("sdxl_zimage") == "sdxl-zimage"
    assert resolve_seed(None) == 0


def test_sdxl_requested_steps_compensate_for_the_diffusers_truncation():
    """Diffusers truncates the step COUNT where DiffSynth truncates the sigma range.

    Asking Diffusers for four steps at strength 0.15 runs int(4 * 0.15) = 0 and
    returns a bare VAE round-trip, so the request has to be scaled up instead.
    """
    from remove_ai_watermarks._internal.sdxl_zimage_pipeline import requested_steps

    for strength in (0.15, 0.25, 0.0896):
        steps = requested_steps(4, strength)
        assert int(steps * strength) >= 4
        # Naively asking for four would have under-spent every time, and at the
        # strengths this profile actually uses it would have run nothing at all.
        assert int(4 * strength) < 4
    assert int(4 * 0.15) == 0


def test_sdxl_zimage_floors_to_its_own_latent_grid():
    """SDXL aligns to 8 pixels where Qwen aligns to 16."""
    from remove_ai_watermarks._internal.sdxl_zimage_pipeline import sdxl_target_size
    from remove_ai_watermarks._internal.two_stage_pipeline import _target_size

    assert sdxl_target_size(1122, 1402) == (1120, 1400)
    assert _target_size(1122, 1402) == (1120, 1392)
    assert sdxl_target_size(3, 3) == (8, 8)


def test_sdxl_zimage_inherits_the_face_stage_rather_than_copying_it():
    """The face stage must not be able to diverge between profiles.

    Both profiles are siblings of the shared two-stage base, not a chain: an SDXL or
    Flux profile must never inherit from the Qwen profile, and the shared stages live
    on the base so a change there reaches every profile at once.
    """
    from remove_ai_watermarks._internal.qwen_zimage_pipeline import QwenZImagePipeline
    from remove_ai_watermarks._internal.sdxl_zimage_pipeline import SdxlZImagePipeline
    from remove_ai_watermarks._internal.two_stage_pipeline import TwoStageZImagePipeline

    assert issubclass(SdxlZImagePipeline, TwoStageZImagePipeline)
    assert issubclass(QwenZImagePipeline, TwoStageZImagePipeline)
    assert QwenZImagePipeline not in SdxlZImagePipeline.__mro__
    for shared in ("_run_faces", "_sam_masks", "_load_zimage", "_load_sam", "run", "preload"):
        base_method = getattr(TwoStageZImagePipeline, shared)
        assert getattr(SdxlZImagePipeline, shared) is base_method
        assert getattr(QwenZImagePipeline, shared) is base_method
    # Only the global stage and what it needs may differ.
    assert SdxlZImagePipeline._run_global is not QwenZImagePipeline._run_global
    assert SdxlZImagePipeline._load_global is not QwenZImagePipeline._load_global


def test_chroma_zimage_strength_uses_measured_flat_floors():
    """The chroma floors come from the four-cohort oracle calibration; they must not
    drift from their derivation in watermark_profiles."""
    from remove_ai_watermarks._internal.watermark_profiles import (
        CHROMA_ZIMAGE_GOOGLE_STRENGTH,
        CHROMA_ZIMAGE_META_STRENGTH,
        CHROMA_ZIMAGE_MICROSOFT_STRENGTH,
        CHROMA_ZIMAGE_OPENAI_STRENGTH,
        CHROMA_ZIMAGE_UNKNOWN_STRENGTH,
        normalize_profile,
        resolve_strength,
    )

    assert normalize_profile("chroma_zimage") == "chroma-zimage"
    assert resolve_strength(None, "openai", "chroma-zimage") == pytest.approx(CHROMA_ZIMAGE_OPENAI_STRENGTH)
    assert pytest.approx(0.20) == CHROMA_ZIMAGE_OPENAI_STRENGTH
    assert resolve_strength(None, "microsoft", "chroma-zimage") == pytest.approx(CHROMA_ZIMAGE_MICROSOFT_STRENGTH)
    assert pytest.approx(0.125) == CHROMA_ZIMAGE_MICROSOFT_STRENGTH
    assert resolve_strength(None, "google", "chroma-zimage") == pytest.approx(CHROMA_ZIMAGE_GOOGLE_STRENGTH)
    assert pytest.approx(0.40) == CHROMA_ZIMAGE_GOOGLE_STRENGTH
    assert resolve_strength(None, "meta", "chroma-zimage") == pytest.approx(CHROMA_ZIMAGE_META_STRENGTH)
    assert pytest.approx(0.17) == CHROMA_ZIMAGE_META_STRENGTH
    # Unknown follows the strictest measured cohort, as with the other profiles.
    assert resolve_strength(None, None, "chroma-zimage") == pytest.approx(CHROMA_ZIMAGE_UNKNOWN_STRENGTH)
    assert CHROMA_ZIMAGE_UNKNOWN_STRENGTH == CHROMA_ZIMAGE_GOOGLE_STRENGTH
    # An explicit strength still wins, and other profiles are untouched.
    assert resolve_strength(0.2, "google", "chroma-zimage") == pytest.approx(0.2)
    assert resolve_strength(None, "openai", "qwen-zimage", size=(2000, 1850)) == pytest.approx(0.07675)


def test_chroma_zimage_inherits_the_shared_stages_and_only_swaps_the_global():
    """Like sdxl-zimage: a sibling of the base, never a chain, with the shared
    stages inherited verbatim."""
    from remove_ai_watermarks._internal.chroma_zimage_pipeline import ChromaZImagePipeline
    from remove_ai_watermarks._internal.qwen_zimage_pipeline import QwenZImagePipeline
    from remove_ai_watermarks._internal.two_stage_pipeline import TwoStageZImagePipeline

    assert issubclass(ChromaZImagePipeline, TwoStageZImagePipeline)
    assert QwenZImagePipeline not in ChromaZImagePipeline.__mro__
    for shared in ("_run_faces", "_sam_masks", "_load_zimage", "_load_sam", "run", "preload"):
        base_method = getattr(TwoStageZImagePipeline, shared)
        assert getattr(ChromaZImagePipeline, shared) is base_method
    assert ChromaZImagePipeline._run_global is not QwenZImagePipeline._run_global
    assert ChromaZImagePipeline._load_global is not QwenZImagePipeline._load_global
    assert ChromaZImagePipeline.profile_name == "chroma-zimage"


@pytest.mark.parametrize(
    ("strength", "requested", "effective"),
    [(0.09, 45, 5), (0.20, 20, 4), (0.125, 32, 4), (0.17, 24, 5), (0.40, 10, 4)],
)
def test_chroma_preserves_calibrated_requests_and_actual_scheduler_steps(strength, requested, effective):
    from types import SimpleNamespace

    diffusers = pytest.importorskip("diffusers")
    from remove_ai_watermarks._internal.chroma_zimage_pipeline import CHROMA_NOMINAL_STEPS, requested_steps

    steps = requested_steps(CHROMA_NOMINAL_STEPS, strength)
    assert steps == requested
    pipeline = SimpleNamespace(scheduler=SimpleNamespace(timesteps=list(range(steps)), order=1))
    timesteps, count = diffusers.ChromaImg2ImgPipeline.get_timesteps(pipeline, steps, strength, "cpu")
    assert count == effective
    assert len(timesteps) == effective


def test_chroma_zimage_uses_its_own_calibrated_prompts():
    """The floors were measured with the neutral faithful prompt, not the shared
    canny-stage prompt; swapping them silently moves the oracle boundaries."""
    from remove_ai_watermarks._internal import chroma_zimage_pipeline as chroma_mod
    from remove_ai_watermarks._internal.two_stage_pipeline import _GLOBAL_PROMPT

    assert chroma_mod.CHROMA_PROMPT != _GLOBAL_PROMPT
    assert chroma_mod.CHROMA_PROMPT == "high quality, sharp, detailed, faithful to the original"
    assert chroma_mod.CHROMA_GUIDANCE == 5.0
    assert chroma_mod.CHROMA_NOMINAL_STEPS == 4


def test_chroma_global_stage_calls_diffusers_with_the_calibrated_shape(monkeypatch):
    """The calibrated call shape must not drift: prompt, guidance, width/height
    on the /16 grid, requested-step count, and no Canny conditioning. A silent
    parameter change moves the oracle floors without any test seeing it."""
    # CI's base job installs the library without the qwen-zimage extra.
    diffusers = pytest.importorskip("diffusers")
    import torch

    from remove_ai_watermarks._internal.chroma_zimage_pipeline import (
        CHROMA_GUIDANCE,
        ChromaZImagePipeline,
    )

    calls: dict[str, dict] = {}

    def fake_chroma(*_args, **kwargs):
        calls["chroma"] = kwargs
        from PIL import Image

        return type("R", (), {"images": [Image.new("RGB", (kwargs["width"], kwargs["height"]))]})()

    class FakeChromaPipe:
        text_encoder = object()  # truthy: the encoder is loaded

        def to(self, _device):
            return self

        @staticmethod
        def encode_prompt(*args, **kwargs):
            import torch

            n = 8  # sequence length stub
            embed = torch.zeros(1, n, 64)  # prompt embeds stub
            neg_embed = torch.zeros(1, n, 64)
            mask = torch.ones(1, n, dtype=torch.long)
            neg_mask = torch.ones(1, n, dtype=torch.long)
            return embed, None, mask, neg_embed, None, neg_mask

        @staticmethod
        def __call__(*args, **kwargs):
            return fake_chroma(*args, **kwargs)

    monkeypatch.setattr(
        diffusers.ChromaImg2ImgPipeline,
        "from_pretrained",
        staticmethod(lambda *a, **k: FakeChromaPipe()),
    )

    # _run_global creates a torch.Generator(device="cuda"), which needs a real
    # CUDA library; patch the module's torch reference to yield a CPU-safe stub.
    class FakeGenerator:
        def manual_seed(self, _seed):
            return self

    monkeypatch.setattr("torch.Generator", lambda device=None: FakeGenerator())

    pipeline = ChromaZImagePipeline(device="cpu", torch_dtype=torch.bfloat16)
    monkeypatch.setattr(type(pipeline), "_require_cuda", lambda self: None)
    result = pipeline._run_global(Image.new("RGB", (1122, 1402)), 0.09, 0)

    assert result.size == (1122, 1402)
    call = calls["chroma"]
    # The calibrated prompt is cached on first encode; every subsequent call
    # passes prompt_embeds (not the raw prompt string), skipping the T5 inference.
    assert "prompt" not in call, "cached path must not send the raw prompt"
    assert "prompt_embeds" in call, "cached path must pass pre-computed embeddings"
    assert "negative_prompt_embeds" in call
    # The negative prompt is baked into the cached embeddings too.
    assert call["guidance_scale"] == CHROMA_GUIDANCE
    # The /16 latent grid, not the raw input size.
    assert call["width"] == 1120
    assert call["height"] == 1392
    # Effective-step compensation: ceil(4 / 0.09) = 45 requested.
    assert call["num_inference_steps"] == 45
    assert call["strength"] == 0.09
    # No Canny conditioning: the calibrated path is a plain strength pass.
    assert "control_image" not in call
    assert "controlnet_conditioning_scale" not in call


def test_chroma_zimage_floors_to_its_own_latent_grid():
    """Chroma1 floors to /16, the same grid as Qwen but unlike SDXL's /8."""
    from remove_ai_watermarks._internal.chroma_zimage_pipeline import chroma_target_size
    from remove_ai_watermarks._internal.sdxl_zimage_pipeline import sdxl_target_size

    assert chroma_target_size(1122, 1402) == (1120, 1392)
    assert sdxl_target_size(1122, 1402) == (1120, 1400)
    assert chroma_target_size(3, 3) == (16, 16)


def test_watermark_remover_accepts_text_manifest_for_chroma_but_not_sdxl(tmp_path, monkeypatch):
    """Chroma owns a VAE donor; SDXL still fails at the public boundary."""
    from remove_ai_watermarks._internal.text_restoration import VerifiedTextLine, VerifiedTextManifest
    from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover

    _mock_watermark_runtime_deps(monkeypatch)
    source = tmp_path / "source.png"
    Image.new("RGB", (96, 80)).save(source)
    manifest = VerifiedTextManifest("0" * 64, 96, 80, (VerifiedTextLine((4, 4, 20, 16), "x", "alphabetic"),))

    chroma = WatermarkRemover(device="cuda", pipeline="chroma-zimage")
    chroma_runtime = MagicMock()
    chroma_runtime.run.return_value = Image.new("RGB", (96, 80))
    monkeypatch.setattr(chroma, "_load_qwen_zimage_pipeline", lambda: chroma_runtime)

    chroma.remove_watermark(source, text_manifest=manifest)

    assert chroma_runtime.run.call_count == 1
    assert chroma_runtime.run.call_args.kwargs["text_manifest"] is manifest
    with pytest.raises(ValueError, match=r"fidelity anchor.*qwen-zimage"):
        chroma.remove_watermark(source, text_manifest=manifest, fidelity_anchor=True)
    assert chroma_runtime.run.call_count == 1

    sdxl = WatermarkRemover(device="cuda", pipeline="sdxl-zimage")
    with pytest.raises(ValueError, match="not supported by the sdxl-zimage profile"):
        sdxl.remove_watermark(source, text_manifest=manifest)


def test_watermark_remover_dispatches_to_chroma_pipeline(monkeypatch):
    """The remover must construct ChromaZImagePipeline, not fall through to qwen."""
    from remove_ai_watermarks._internal import chroma_zimage_pipeline as chroma_module
    from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover

    captured: dict[str, object] = {}

    class Recorder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(chroma_module, "ChromaZImagePipeline", Recorder)

    remover = WatermarkRemover.__new__(WatermarkRemover)
    remover.model_profile = "chroma-zimage"
    remover.device = "cuda"
    remover.torch_dtype = None
    remover.hf_token = None
    remover._progress_callback = None
    remover.controlnet_conditioning_scale = 1.0
    remover.cpu_offload = False
    remover._qwen_zimage_pipeline = None

    remover._load_qwen_zimage_pipeline()

    assert captured["device"] == "cuda"
    assert captured["hf_token"] is None


def test_chroma_adaptive_polish_defaults_off():
    """Like qwen-zimage: Chroma1's output already matches the input's detail
    level, so polishing would add grain to an already-faithful output."""
    from remove_ai_watermarks._internal.watermark_profiles import resolve_adaptive_polish

    assert resolve_adaptive_polish(None, "chroma-zimage") is False
    assert resolve_adaptive_polish(True, "chroma-zimage") is True


def test_chroma_vae_roundtrip_uses_deterministic_mode_and_restores_native_size(monkeypatch):
    """The Chroma donor pads without distorting geometry and never samples."""
    import torch

    from remove_ai_watermarks._internal.chroma_zimage_pipeline import ChromaZImagePipeline

    seen: dict[str, object] = {}
    latent = torch.ones(1, 4, 4, 6)

    class LatentDistribution:
        def mode(self):
            seen["mode_calls"] = int(seen.get("mode_calls", 0)) + 1
            return latent

        def sample(self, *_args, **_kwargs):
            raise AssertionError("A verified-text donor must not sample VAE latents")

    class FakeVae:
        @staticmethod
        def encode(tensor):
            seen["encoded_shape"] = tuple(tensor.shape)
            return type("Encoded", (), {"latent_dist": LatentDistribution()})()

        @staticmethod
        def decode(received, *, return_dict):
            assert received is latent
            assert return_dict is False
            return (torch.zeros(1, 3, 32, 48),)

    class FakeProcessor:
        @staticmethod
        def preprocess(image, *, height, width):
            seen["preprocess"] = (image.size, height, width)
            return torch.zeros(1, 3, height, width)

        @staticmethod
        def postprocess(decoded, *, output_type):
            assert output_type == "pil"
            return [Image.new("RGB", (decoded.shape[3], decoded.shape[2]))]

    pipe = type("FakePipe", (), {"vae": FakeVae(), "image_processor": FakeProcessor()})()
    pipeline = ChromaZImagePipeline(device="cpu", torch_dtype=torch.float32)
    monkeypatch.setattr(pipeline, "_load_global", lambda: pipe)

    result = pipeline._vae_roundtrip(Image.new("RGB", (35, 19)))

    assert seen["preprocess"] == ((48, 32), 32, 48)
    assert seen["encoded_shape"] == (1, 3, 32, 48)
    assert seen["mode_calls"] == 1
    assert result.size == (35, 19)


def test_chroma_google_face_content_gets_the_lower_adaptive_floor():
    """Face content on Google SynthID clears at 0.125, not 0.40.

    The four-fixture calibration showed a clean face-count split: both zero-face
    text cards need 0.25, both face fixtures clear at 0.12. The adaptive arm
    saves face identity (0.611 vs 0.279 cosine) and 6.4 dB PSNR on face content.
    """
    from remove_ai_watermarks._internal.watermark_profiles import (
        CHROMA_ZIMAGE_GOOGLE_FACE_STRENGTH,
        CHROMA_ZIMAGE_GOOGLE_STRENGTH,
        resolve_strength,
    )

    assert pytest.approx(0.125) == CHROMA_ZIMAGE_GOOGLE_FACE_STRENGTH
    # Face content gets the lower adaptive floor.
    assert resolve_strength(None, "google", "chroma-zimage", face_count=17) == pytest.approx(0.125)
    assert resolve_strength(None, "google", "chroma-zimage", face_count=1) == pytest.approx(0.125)
    # Zero-face content gets the flat floor (text/graphic cohort).
    assert resolve_strength(None, "google", "chroma-zimage", face_count=0) == pytest.approx(0.40)
    assert pytest.approx(0.40) == CHROMA_ZIMAGE_GOOGLE_STRENGTH
    # No face_count hint: conservative (flat floor).
    assert resolve_strength(None, "google", "chroma-zimage") == pytest.approx(0.40)
    # Other vendors ignore face_count: the split is Google-SynthID-specific.
    assert resolve_strength(None, "openai", "chroma-zimage", face_count=5) == pytest.approx(0.20)
    assert resolve_strength(None, "meta", "chroma-zimage", face_count=5) == pytest.approx(0.17)
    # An explicit strength still wins over the adaptive arm.
    assert resolve_strength(0.20, "google", "chroma-zimage", face_count=17) == pytest.approx(0.20)
    # Other profiles are untouched by the face_count parameter.
    assert resolve_strength(None, "google", "qwen-zimage", size=(2000, 1850), face_count=17) == pytest.approx(0.27)


def test_auto_profile_routes_to_the_measured_engine_per_vendor():
    """--pipeline auto picks the engine the four-cohort calibration supports:
    chroma-zimage for Microsoft, qwen-zimage for OpenAI/Google/Meta/unknown."""
    from remove_ai_watermarks._internal.watermark_profiles import (
        CHROMA_ZIMAGE_PROFILE,
        QWEN_ZIMAGE_PROFILE,
        normalize_profile,
        resolve_auto_profile,
    )

    assert normalize_profile("auto") == "auto"
    assert resolve_auto_profile("openai") == QWEN_ZIMAGE_PROFILE
    assert resolve_auto_profile("microsoft") == CHROMA_ZIMAGE_PROFILE
    assert resolve_auto_profile("google") == QWEN_ZIMAGE_PROFILE
    assert resolve_auto_profile("meta") == QWEN_ZIMAGE_PROFILE
    assert resolve_auto_profile(None) == QWEN_ZIMAGE_PROFILE
    assert resolve_auto_profile("unknown") == QWEN_ZIMAGE_PROFILE


def test_auto_profile_strength_uses_the_resolved_engine_floors(tmp_path, monkeypatch):
    """Auto resolves OpenAI to Qwen and uses its oracle-verified 0.07675 floor."""
    from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover

    _mock_watermark_runtime_deps(monkeypatch)
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 48)).save(source)

    remover = WatermarkRemover(device="cuda", pipeline="auto")
    assert remover.model_profile == "auto"

    runtime = MagicMock()
    runtime.run.return_value = Image.new("RGB", (64, 48), (50, 60, 70))
    monkeypatch.setattr(remover, "_load_qwen_zimage_pipeline", lambda: runtime)

    remover.remove_watermark(source, vendor="openai")

    assert remover.model_profile == "qwen-zimage"
    _, kwargs = runtime.run.call_args
    assert kwargs["strength"] == pytest.approx(0.07675)


def test_auto_profile_text_manifest_uses_the_measured_qwen_engine_once(tmp_path, monkeypatch):
    """A manifest does not override the researched OpenAI-to-Qwen decision."""
    from remove_ai_watermarks._internal.text_restoration import VerifiedTextLine, VerifiedTextManifest
    from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover

    _mock_watermark_runtime_deps(monkeypatch)
    source = tmp_path / "source.png"
    Image.new("RGB", (96, 80)).save(source)
    manifest = VerifiedTextManifest("0" * 64, 96, 80, (VerifiedTextLine((4, 4, 20, 16), "x", "alphabetic"),))

    remover = WatermarkRemover(device="cuda", pipeline="auto")
    runtime = MagicMock()
    runtime.run.return_value = Image.new("RGB", (96, 80))
    monkeypatch.setattr(remover, "_load_qwen_zimage_pipeline", lambda: runtime)

    remover.remove_watermark(source, vendor="openai", text_manifest=manifest)

    assert remover.model_profile == "qwen-zimage"
    assert runtime.run.call_count == 1
    assert runtime.run.call_args.kwargs["text_manifest"] is manifest
