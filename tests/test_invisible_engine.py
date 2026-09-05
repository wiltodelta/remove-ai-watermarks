"""Tests for the invisible watermark engine (unit tests, no GPU required)."""

from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from remove_ai_watermarks.invisible_engine import (
    InvisibleEngine,
    _apply_postprocessing,
    _target_size,
    is_available,
)


class TestIsAvailable:
    """Tests for dependency checking."""

    def test_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)

    def test_the_module_list_includes_the_face_stage_runtime(self):
        """diffsynth is part of the answer, not an optional upgrade.

        Both profiles repair faces with the DiffSynth Z-Image stage, so a
        torch+diffusers-only environment used to clear this gate and then die there.
        The discriminating guard -- that this gate and the remover's precondition
        BOTH close when any one module is missing -- lives in
        ``test_platform.py::TestAvailability``; comparing ``is_available()`` to a
        tuple derived from itself passes on every host and proves nothing.
        """
        from remove_ai_watermarks._internal.watermark_profiles import REMOVAL_MODULES

        assert "diffsynth" in REMOVAL_MODULES


class TestInvisibleEngineInit:
    """Tests for InvisibleEngine construction (no GPU required)."""

    def test_preload_forwards_global_only(self):
        engine = object.__new__(InvisibleEngine)
        engine._remover = SimpleNamespace(preload=lambda **kwargs: setattr(engine, "_preload_kwargs", kwargs))

        engine.preload(global_only=True)

        assert engine._preload_kwargs == {"global_only": True}


class TestVerifiedTextMode:
    """The experimental mode must fail before loading models on unmeasured inputs."""

    @staticmethod
    def _engine(profile: str = "qwen-zimage") -> InvisibleEngine:
        engine = object.__new__(InvisibleEngine)
        engine._progress_callback = None
        engine._remover = SimpleNamespace(model_profile=profile)
        return engine

    def test_rejects_incompatible_pipeline_options(self, tmp_path):
        import pytest

        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        cases = (
            ("sdxl-zimage", {}, "not supported by the sdxl-zimage profile"),
            ("qwen-zimage", {"tile": True}, "not calibrated with --tile"),
            ("qwen-zimage", {"max_resolution": 1024}, "max-resolution 0"),
            ("qwen-zimage", {"humanize": 1.0}, "humanize=0"),
            ("qwen-zimage", {"adaptive_polish": True}, "polish disabled"),
        )
        for profile, kwargs, message in cases:
            with pytest.raises(ValueError, match=message):
                self._engine(profile).remove_watermark(
                    tmp_path / "unused.png",
                    text_manifest=manifest,
                    **kwargs,
                )

    def test_rejects_fidelity_anchor_without_manifest(self, tmp_path):
        import pytest

        with pytest.raises(ValueError, match="fidelity_anchor requires a text manifest"):
            self._engine().remove_watermark(
                tmp_path / "unused.png",
                fidelity_anchor=True,
            )

    def test_loads_and_forwards_verified_manifest(self, tmp_path, monkeypatch):
        import json

        from remove_ai_watermarks import region_eraser
        from remove_ai_watermarks._internal.text_restoration import source_pixel_sha256

        source = tmp_path / "source.png"
        output = tmp_path / "output.png"
        image = Image.new("RGB", (48, 32), (10, 20, 30))
        image.save(source)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "verified": True,
                    "source_pixel_sha256": source_pixel_sha256(image),
                    "width": 48,
                    "height": 32,
                    "lines": [{"box": [8, 8, 40, 24], "text": "Exact", "script": "alphabetic"}],
                }
            ),
            encoding="utf-8",
        )
        seen = {}

        def fake_remove(**kwargs):
            seen.update(kwargs)
            Image.open(kwargs["image_path"]).save(kwargs["output_path"])
            return kwargs["output_path"]

        # The public engine must not reject auto before WatermarkRemover resolves
        # the measured per-vendor profile.
        engine = self._engine("auto")
        engine._remover.remove_watermark = fake_remove
        monkeypatch.setattr(region_eraser, "lama_available", lambda: True)

        engine.remove_watermark(source, output, text_manifest=manifest)

        assert seen["text_manifest"].lines[0].text == "Exact"
        # Leak-safe default since 0.27.1: the global 15% donor blend is OFF unless
        # explicitly requested (measured to return detector-visible OpenAI SynthID
        # on poster-scale manifests; see docs/text-protection-research.md).
        assert seen["fidelity_anchor"] is False

        engine.remove_watermark(source, output, text_manifest=manifest, fidelity_anchor=True)

        assert seen["fidelity_anchor"] is True


class TestNativeOutputSize:
    """Model-side latent-grid rounding must not change the public output size."""

    def test_no_polish_restores_native_non_multiple_of_eight_size(self, tmp_path):
        engine = object.__new__(InvisibleEngine)

        def _remove_watermark(image_path, output_path=None, **_kwargs):
            out = output_path or image_path.with_stem(image_path.stem + "_clean")
            # Model-side latent-grid rounding: 18px becomes 16px.
            Image.open(image_path).crop((0, 0, 24, 16)).save(out)
            return out

        engine._remover = SimpleNamespace(remove_watermark=_remove_watermark, model_profile="qwen-zimage")
        engine._progress_callback = None
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        Image.new("RGB", (24, 18), (128, 128, 128)).save(src)

        engine.remove_watermark(src, out, adaptive_polish=False)

        assert Image.open(out).size == (24, 18)


class TestTargetSize:
    """Regression guard for the native-resolution decision (issues #10 / #15).

    max_resolution=0 must NOT downscale -- the forced downscale->upscale
    round-trip was the quality loss in #10, and downscaling at all let SynthID
    survive in #15 (the native SDXL pass at strength ~0.05 is what defeats it).
    """

    def test_native_default_no_downscale(self):
        # The default (0) means native resolution: no resize, regardless of size.
        assert _target_size(4096, 4096, 0) is None
        assert _target_size(123, 456, 0) is None

    def test_the_engine_default_is_itself_a_valid_input(self):
        # Read the default off the signature rather than restating 0: a caller forwards
        # whatever ``remove_watermark`` declares, and ``None`` there would reach the
        # ``max_resolution > 0`` test below as a TypeError instead of "no cap".
        import inspect

        default = inspect.signature(InvisibleEngine.remove_watermark).parameters["max_resolution"].default
        assert _target_size(1024, 768, default) is None

    def test_negative_cap_treated_as_native(self):
        assert _target_size(4096, 4096, -1) is None

    def test_cap_below_long_side_downscales(self):
        # 2000x1000, cap 1024 -> long side scaled to 1024, aspect preserved.
        assert _target_size(2000, 1000, 1024) == (1024, 512)

    def test_cap_uses_long_side_for_portrait(self):
        # Portrait: height is the long side, so it drives the ratio.
        assert _target_size(1000, 2000, 1024) == (512, 1024)

    def test_cap_at_or_above_long_side_no_downscale(self):
        # Already within the cap (and exactly equal) -> no resize.
        assert _target_size(800, 600, 1024) is None
        assert _target_size(1024, 768, 1024) is None

    def test_integer_truncation_matches_pil_call_site(self):
        # 1254x1254 (the gpt-image sample) capped at 1000: int(1254*1000/1254)=1000.
        assert _target_size(1254, 1254, 1000) == (1000, 1000)
        # Non-divisible ratio truncates toward zero like int() at the call site.
        assert _target_size(1000, 333, 500) == (500, 166)

    def test_extreme_aspect_ratio_clamps_short_side_to_one(self):
        # 5000x3 capped at 1024: int(3 * 1024/5000) = 0 would crash resize();
        # the short side must clamp to 1, never 0.
        assert _target_size(5000, 3, 1024) == (1024, 1)
        assert _target_size(3, 5000, 1024) == (1, 1024)

    def test_a_small_input_is_left_at_native_size(self):
        """No minimum-resolution floor: only the cap can move geometry."""
        assert _target_size(381, 512, 0) is None
        assert _target_size(381, 512, 4096) is None


class TestEngineConstructsWithoutAModelId:
    """Plain construction must reach the remover, and must not name a model.

    The engine used to take a ``model_id`` and substitute the SDXL default for None.
    Once the remover tightened its "you may not override the fixed stack" check to
    ``is not None``, that substitution made EVERY construction raise -- and no test
    saw it, because the library tests build WatermarkRemover directly while the engine
    tests mock it. A deployed Modal worker caught it instead. The parameter is gone on
    both sides now, so guard the property that broke: a default construction reaches
    the remover, carrying no model at all.
    """

    def test_default_construction_names_no_model(self):
        from unittest.mock import patch

        import remove_ai_watermarks.invisible_engine as engine_module

        with patch("remove_ai_watermarks._internal.watermark_remover.WatermarkRemover") as remover:
            engine_module.InvisibleEngine(pipeline="qwen-zimage")
        assert remover.call_count == 1
        assert "model_id" not in remover.call_args.kwargs

    def test_a_model_id_is_not_accepted(self):
        import pytest

        import remove_ai_watermarks.invisible_engine as engine_module

        with pytest.raises(TypeError):
            engine_module.InvisibleEngine(model_id="org/custom", pipeline="qwen-zimage")  # type: ignore[call-arg]


class TestEngineResolvesThePolishPerProfile:
    """The engine, not the CLI, turns an unset adaptive_polish into the profile default.

    This is the change that stopped a library caller and a CLI caller on one profile
    from producing different pixels, and it had no test: rebinding
    ``resolve_adaptive_polish`` to ``bool(value)`` -- exactly the pre-commit behavior --
    left the whole suite green.
    """

    @staticmethod
    def _engine(profile: str):
        from unittest.mock import MagicMock

        engine = object.__new__(InvisibleEngine)
        engine._progress_callback = None
        engine._remover = MagicMock(model_profile=profile)
        return engine

    def _polish_used(self, profile: str, requested, tmp_path, monkeypatch) -> bool:
        seen: list[bool] = []
        monkeypatch.setattr(
            "remove_ai_watermarks.humanizer.adaptive_polish",
            lambda out, ref, seed=None, on_skip=None: (seen.append(True), out)[1],
        )
        src = tmp_path / f"{profile}_{requested}.png"
        Image.new("RGB", (32, 32), (90, 120, 150)).save(src)
        engine = self._engine(profile)
        engine._remover.remove_watermark.side_effect = lambda **kw: (
            Image.open(kw["image_path"]).save(kw["output_path"]),
            kw["output_path"],
        )[1]
        engine.remove_watermark(src, tmp_path / f"out_{profile}_{requested}.png", adaptive_polish=requested)
        return bool(seen)

    def test_unset_follows_the_profile_not_the_signature_default(self, tmp_path, monkeypatch):
        assert self._polish_used("qwen-zimage", None, tmp_path, monkeypatch) is False
        assert self._polish_used("sdxl-zimage", None, tmp_path, monkeypatch) is True

    def test_an_explicit_value_still_wins_on_both_profiles(self, tmp_path, monkeypatch):
        assert self._polish_used("qwen-zimage", True, tmp_path, monkeypatch) is True
        assert self._polish_used("sdxl-zimage", False, tmp_path, monkeypatch) is False


class TestPostprocessingOrder:
    """The stage order is a contract: grain last, so the polish still composes."""

    @staticmethod
    def _record(monkeypatch) -> list[str]:
        seen: list[str] = []
        monkeypatch.setattr(
            "remove_ai_watermarks.humanizer.unsharp_mask",
            lambda img, amount=0.5, sigma=1.0: (seen.append("unsharp"), img)[1],
        )
        monkeypatch.setattr(
            "remove_ai_watermarks.humanizer.adaptive_polish",
            lambda img, ref, seed=None, on_skip=None: (seen.append("polish"), img)[1],
        )
        monkeypatch.setattr(
            "remove_ai_watermarks.humanizer.apply_analog_humanizer",
            lambda img, grain_intensity=4.0, chromatic_shift=1: (seen.append("humanize"), img)[1],
        )
        return seen

    def test_the_polish_runs_before_the_grain(self, monkeypatch):
        import numpy as np

        seen = self._record(monkeypatch)
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        _apply_postprocessing(
            img,
            lambda: img,
            humanize=6.0,
            unsharp=0.5,
            adaptive_polish=True,
            orig_size=(16, 16),
            seed=0,
        )
        # Grain last: humanize first is what silenced the polish (it reads grain as
        # detail the image already has and self-limits to nothing).
        assert seen == ["unsharp", "polish", "humanize"]

    def test_the_reference_is_only_decoded_when_the_polish_needs_it(self, monkeypatch):
        import numpy as np

        self._record(monkeypatch)
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        calls: list[int] = []

        def reference():
            calls.append(1)
            return img

        _apply_postprocessing(
            img, reference, humanize=6.0, unsharp=0.5, adaptive_polish=False, orig_size=(16, 16), seed=0
        )
        assert calls == []
        _apply_postprocessing(
            img, reference, humanize=0.0, unsharp=0.0, adaptive_polish=True, orig_size=(16, 16), seed=0
        )
        assert calls == [1]

    def test_the_resize_happens_before_every_filter(self, monkeypatch):
        import numpy as np

        seen = self._record(monkeypatch)
        sizes: list[tuple[int, int]] = []
        monkeypatch.setattr(
            "remove_ai_watermarks.humanizer.unsharp_mask",
            lambda img, amount=0.5, sigma=1.0: (seen.append("unsharp"), sizes.append(img.shape[:2]), img)[2],
        )
        out = _apply_postprocessing(
            np.zeros((8, 8, 3), dtype=np.uint8),
            lambda: np.zeros((16, 16, 3), dtype=np.uint8),
            humanize=0.0,
            unsharp=0.5,
            adaptive_polish=False,
            orig_size=(16, 16),
            seed=0,
        )
        # Grain and sharpening belong at output resolution, not smeared by a later upscale.
        assert sizes == [(16, 16)]
        assert out.shape[:2] == (16, 16)
