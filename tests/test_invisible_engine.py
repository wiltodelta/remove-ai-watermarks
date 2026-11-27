"""Tests for the invisible watermark engine (unit tests, no GPU required)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
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
        engine._remover = SimpleNamespace(configured_profile=profile, model_profile=profile)
        return engine

    def test_rejects_incompatible_pipeline_options(self, tmp_path):
        import pytest

        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        cases = (
            ("qwen-zimage", {"tile": True}, "not calibrated with tile=True"),
            ("qwen-zimage", {"max_resolution": 1024}, "max_resolution=0"),
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

        engine.remove_watermark(source, output, text_manifest=manifest, vendor="google")

        assert seen["text_manifest"].lines[0].text == "Exact"
        assert seen["vendor"] == "google"
        assert "fidelity_anchor" not in seen


class TestNativeOutputSize:
    """Model-side latent-grid rounding must not change the public output size."""

    def test_no_polish_restores_native_non_multiple_of_eight_size(self, tmp_path):
        engine = object.__new__(InvisibleEngine)

        def _remove_watermark(image_path, output_path=None, **_kwargs):
            out = output_path or image_path.with_stem(image_path.stem + "_clean")
            # Model-side latent-grid rounding: 18px becomes 16px.
            Image.open(image_path).crop((0, 0, 24, 16)).save(out)
            return out

        engine._remover = SimpleNamespace(
            remove_watermark=_remove_watermark,
            configured_profile="qwen-zimage",
            model_profile="qwen-zimage",
        )
        engine._progress_callback = None
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        Image.new("RGB", (24, 18), (128, 128, 128)).save(src)

        engine.remove_watermark(src, out, adaptive_polish=False)

        assert Image.open(out).size == (24, 18)


class TestTransparency:
    """Exercise real image IO and postprocessing around an RGB-only model stand-in."""

    @staticmethod
    def _engine():
        engine = object.__new__(InvisibleEngine)
        engine._progress_callback = None

        def remove_watermark(image_path, output_path, **_kwargs):
            with Image.open(image_path) as image:
                assert image.mode == "RGB"
                width, height = image.size
            # Model-side grid rounding and changed pixels, not a copy of the input.
            Image.new("RGB", (width // 8 * 8, height // 8 * 8), (40, 120, 200)).save(output_path)
            return output_path

        engine._remover = SimpleNamespace(configured_profile="qwen-zimage", remove_watermark=remove_watermark)
        return engine

    @staticmethod
    def _source(mode, size):
        image = Image.new("RGB", size, (180, 90, 30))
        alpha = np.resize(np.array([0, 64, 128, 255], dtype=np.uint8), (size[1], size[0]))
        if mode == "P":
            image = Image.fromarray(alpha).convert("P")
            image.info["transparency"] = bytes(range(256))
        elif mode == "RGB-key":
            image.paste((0, 0, 0), (0, 0, size[0] // 2, size[1]))
            image.info["transparency"] = (0, 0, 0)
        else:
            image = image.convert(mode)
            image.putalpha(Image.fromarray(alpha))
        return image

    @pytest.mark.parametrize("mode", ["RGBA", "LA", "P", "RGB-key"])
    @pytest.mark.parametrize(("size", "max_resolution"), [((32, 24), 0), ((35, 27), 0), ((32, 24), 16)])
    def test_preserves_exact_alpha_after_model_and_size_restoration(self, tmp_path, mode, size, max_resolution):
        source, output = tmp_path / "source.png", tmp_path / "output.png"
        self._source(mode, size).save(source)
        with Image.open(source) as image:
            expected = np.asarray(image.convert("RGBA"))[:, :, 3].copy()

        self._engine().remove_watermark(source, output, max_resolution=max_resolution, adaptive_polish=False)

        with Image.open(output) as image:
            assert image.mode == "RGBA"
            assert image.size == size
            actual = np.asarray(image)
        np.testing.assert_array_equal(actual[:, :, 3], expected)
        # Retain the model's RGB result rather than restoring the source image.
        np.testing.assert_array_equal(actual[:, :, :3], np.broadcast_to([40, 120, 200], actual[:, :, :3].shape))

    @pytest.mark.parametrize("orientation", [2, 6, 7])
    def test_in_place_output_keeps_alpha_in_display_orientation(self, tmp_path, orientation):
        from PIL import ImageOps

        source = tmp_path / "source.png"
        image = self._source("RGBA", (32, 24))
        image.getexif()[274] = orientation
        image.save(source)
        with Image.open(source) as opened:
            expected = np.asarray(ImageOps.exif_transpose(opened).getchannel("A")).copy()

        self._engine().remove_watermark(source, adaptive_polish=False)

        with Image.open(source) as result:
            assert result.mode == "RGBA"
            assert result.getexif().get(274, 1) == 1
            np.testing.assert_array_equal(np.asarray(result.getchannel("A")), expected)

    def test_webp_keeps_alpha_after_postprocessing(self, tmp_path):
        source, output = tmp_path / "source.png", tmp_path / "output.webp"
        image = self._source("RGBA", (32, 24))
        image.save(source)

        self._engine().remove_watermark(source, output, unsharp=0.5, adaptive_polish=True, humanize=1.0)

        with Image.open(output) as result:
            assert result.mode == "RGBA"
            np.testing.assert_array_equal(np.asarray(result.getchannel("A")), np.asarray(image.getchannel("A")))

    @pytest.mark.parametrize(("source_mode", "output_suffix"), [("RGB", ".png"), ("RGBA", ".jpg")])
    def test_rgb_and_non_alpha_output_remain_rgb(self, tmp_path, source_mode, output_suffix):
        source, output = tmp_path / "source.png", tmp_path / ("output" + output_suffix)
        Image.new(source_mode, (32, 24)).save(source)

        self._engine().remove_watermark(source, output, adaptive_polish=False)

        with Image.open(output) as result:
            assert result.mode == "RGB"

    @pytest.mark.parametrize("command", ["invisible", "batch"])
    def test_cli_preserves_alpha_through_the_real_engine(self, tmp_path, monkeypatch, command):
        from click.testing import CliRunner

        from remove_ai_watermarks import invisible_engine
        from remove_ai_watermarks.cli import main

        source = tmp_path / "source.png"
        image = self._source("RGBA", (32, 24))
        image.save(source)
        engine = self._engine()
        monkeypatch.setattr(invisible_engine, "InvisibleEngine", lambda **_kwargs: engine)
        monkeypatch.setattr(invisible_engine, "is_available", lambda: True)
        if command == "batch":
            output_dir = tmp_path / "results"
            args = ["batch", str(tmp_path), "--mode", "invisible", "-o", str(output_dir), "--force"]
            output = output_dir / "source.png"
        else:
            output = tmp_path / "output.png"
            args = ["invisible", str(source), "-o", str(output), "--force"]

        result = CliRunner().invoke(main, args)

        assert result.exit_code == 0, result.output
        with Image.open(output) as processed:
            assert processed.mode == "RGBA"
            np.testing.assert_array_equal(np.asarray(processed.getchannel("A")), np.asarray(image.getchannel("A")))


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
    tests mock it. A deployed GPU worker caught it instead. The parameter is gone on
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
        engine._remover = MagicMock(configured_profile=profile, model_profile=profile)
        return engine

    def _polish_used(self, profile: str, requested, tmp_path, monkeypatch, *, vendor: str | None = None) -> bool:
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
        engine.remove_watermark(
            src,
            tmp_path / f"out_{profile}_{requested}.png",
            adaptive_polish=requested,
            vendor=vendor,
        )
        return bool(seen)

    def test_unset_follows_the_profile_not_the_signature_default(self, tmp_path, monkeypatch):
        assert self._polish_used("qwen-zimage", None, tmp_path, monkeypatch) is False
        assert self._polish_used("sdxl-zimage", None, tmp_path, monkeypatch) is True

    def test_auto_google_uses_the_resolved_sdxl_default(self, tmp_path, monkeypatch):
        assert self._polish_used("auto", None, tmp_path, monkeypatch, vendor="google") is True

    def test_an_explicit_value_still_wins_on_both_profiles(self, tmp_path, monkeypatch):
        assert self._polish_used("qwen-zimage", True, tmp_path, monkeypatch) is True
        assert self._polish_used("sdxl-zimage", False, tmp_path, monkeypatch) is False


class TestPostprocessingOrder:
    """The stage order is a contract: grain last, so the polish still composes."""

    @staticmethod
    def _record(monkeypatch) -> list[str]:
        seen: list[str] = []

        def record(stage):
            def stub(img, *_args, **_kwargs):
                seen.append(stage)
                return img

            return stub

        monkeypatch.setattr(
            "remove_ai_watermarks.humanizer.unsharp_mask",
            record("unsharp"),
        )
        monkeypatch.setattr(
            "remove_ai_watermarks.humanizer.adaptive_polish",
            record("polish"),
        )
        monkeypatch.setattr(
            "remove_ai_watermarks.humanizer.apply_analog_humanizer",
            record("humanize"),
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

        def record_unsharp(img, *_args, **_kwargs):
            seen.append("unsharp")
            sizes.append(img.shape[:2])
            return img

        monkeypatch.setattr(
            "remove_ai_watermarks.humanizer.unsharp_mask",
            record_unsharp,
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
