"""Tests for device detection, profile resolution, and platform-specific paths.

Invisible-watermark removal is CUDA-only, so the device tests here assert a binary
answer and a clean refusal rather than a fallback ladder.
"""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from remove_ai_watermarks._internal.utils import get_image_format, is_supported_format
from remove_ai_watermarks._internal.watermark_profiles import (
    PROFILE_CHOICES,
    QWEN_ZIMAGE_GOOGLE_STRENGTH,
    QWEN_ZIMAGE_META_STRENGTH,
    QWEN_ZIMAGE_OPENAI_STRENGTH,
    REMOVAL_MODULES,
    SDXL_ZIMAGE_GEMINI_STRENGTH,
    SDXL_ZIMAGE_OPENAI_STRENGTH,
    SDXL_ZIMAGE_UNKNOWN_STRENGTH,
    normalize_profile,
    resolve_strength,
    strength_default_help,
)
from remove_ai_watermarks._internal.watermark_remover import get_device, is_watermark_removal_available

# ── Device detection ────────────────────────────────────────────────


class TestDeviceDetection:
    """get_device() is binary: CUDA, or the "cpu" that names its absence."""

    def test_answer_is_cuda_or_cpu(self):
        """No mps/xpu answer exists. Both would travel one frame to the same refusal.

        Reporting them anyway cost a device probe each and let a caller believe the
        library had an Apple-silicon or Intel-GPU path that it does not.
        """
        assert get_device() in ("cpu", "cuda")

    @patch("remove_ai_watermarks._internal.watermark_remover._HAS_TORCH", False)
    def test_no_torch_returns_cpu(self):
        assert get_device() == "cpu"

    def test_working_cuda_is_selected_and_probed(self):
        """A reported CUDA device is smoke-tested before it is returned.

        torch.cuda.is_available() can be True on a build whose CUDA backend then
        raises on the first real op; without the probe that surfaced much later.
        """
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        with patch("remove_ai_watermarks._internal.watermark_remover.torch", fake_torch):
            assert get_device() == "cuda"
        fake_torch.tensor.assert_called_with([1.0], device="cuda")

    def test_broken_cuda_backend_falls_back_to_cpu(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.tensor.side_effect = RuntimeError("no kernel image")
        with patch("remove_ai_watermarks._internal.watermark_remover.torch", fake_torch):
            assert get_device() == "cpu"

    def test_non_cuda_devices_are_refused_at_construction(self):
        """CUDA is a precondition of the object, not of the run.

        Both remaining profiles raise on any other device, so accepting cpu/mps/xpu
        here only defers a guaranteed failure to model-load time - several layers down,
        after the dependency check and the pipeline import, under a message naming
        whichever profile the internal pipeline happens to be.

        Deliberately NOT gated on the diffusion stack. It used to be, and since no CI
        job installs diffusers or diffsynth (the dev extra pulls torch only, via
        invisible-watermark) the guard skipped in every environment it ran in --
        including the maintainer's. The refusal fires before any torch attribute is
        touched, so faking the dependency probe is enough to reach it.
        """
        from remove_ai_watermarks._internal import watermark_remover as module

        with patch.object(module, "is_watermark_removal_available", return_value=True):
            for device in ("cpu", "mps", "xpu"):
                with pytest.raises(ValueError, match="CUDA-only"):
                    module.WatermarkRemover(device=device)

    def test_the_refusal_names_the_resolved_device_not_a_bare_none(self):
        """``device=None`` on a CUDA-less host must report "cpu", not "None".

        The message used to interpolate the raw argument, so the common auto-detect
        path told the user that ``'None'`` cannot run the removal.
        """
        from remove_ai_watermarks._internal import watermark_remover as module

        with (
            patch.object(module, "is_watermark_removal_available", return_value=True),
            patch.object(module, "get_device", return_value="cpu"),
            pytest.raises(ValueError, match="'cpu' cannot run it"),
        ):
            module.WatermarkRemover(device=None)

    def test_a_cuda_remover_picks_the_profile_dtype(self):
        """The dtype half still needs real torch, so it keeps its skip."""
        if not is_watermark_removal_available():
            pytest.skip("the qwen-zimage extra is not installed")
        import torch

        from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover

        remover = WatermarkRemover(device="cuda")
        assert remover.device == "cuda"
        assert remover.torch_dtype == torch.bfloat16
        assert WatermarkRemover(device="cuda", pipeline="sdxl-zimage").torch_dtype == torch.float16


class TestModelProfiles:
    """Only the two CUDA-only two-stage profiles remain."""

    def test_canonical_profiles_unchanged(self):
        assert normalize_profile("qwen-zimage") == "qwen-zimage"
        assert normalize_profile("sdxl-zimage") == "sdxl-zimage"

    def test_underscore_spellings_resolve(self):
        assert normalize_profile("qwen_zimage") == "qwen-zimage"
        assert normalize_profile("  SDXL_ZImage ") == "sdxl-zimage"

    def test_retired_names_no_longer_resolve_to_a_profile(self):
        """default/sdxl/controlnet/qwen were removed, not aliased onward.

        Silently mapping them at the alias layer would route an old script into a
        profile it never asked for; the remover raises on the unknown name instead.
        """
        for retired in ("default", "sdxl", "controlnet", "qwen"):
            assert normalize_profile(retired) not in PROFILE_CHOICES


class TestResolveAdaptivePolish:
    """The polish default is per-profile data, not a CLI parameter-source inference."""

    def test_unset_follows_the_profile(self):
        from remove_ai_watermarks._internal.watermark_profiles import resolve_adaptive_polish

        # qwen-zimage already matches the input's detail level, so polishing it only
        # moves the output away from upstream. An SDXL global pass leaves the softer
        # result the polish exists for.
        assert resolve_adaptive_polish(None, "qwen-zimage") is False
        assert resolve_adaptive_polish(None, "sdxl-zimage") is True
        assert resolve_adaptive_polish(None, "qwen_zimage") is False

    def test_an_explicit_choice_always_wins(self):
        from remove_ai_watermarks._internal.watermark_profiles import resolve_adaptive_polish

        assert resolve_adaptive_polish(True, "qwen-zimage") is True
        assert resolve_adaptive_polish(False, "sdxl-zimage") is False


class TestNoReembeddedWatermark:
    """F2 regression: the SDXL global stage must disable the diffusers watermarker.

    diffusers stamps an open "Stable Diffusion XL" DWT-DCT watermark onto every SDXL
    output whenever ``invisible-watermark`` is installed. A watermark REMOVER that left
    it on would replace one detectable AI watermark (SynthID) with another -- the
    cleaned output re-reads as AI. The ControlNet sub-model load must NOT receive the
    kwarg, since it is not a pipeline and does not accept it.

    Only sdxl-zimage carries an SDXL pipeline now; qwen-zimage's global stage is
    DiffSynth, which has no such watermarker.
    """

    def test_sdxl_global_stage_disables_watermarker(self, monkeypatch: pytest.MonkeyPatch):
        if not is_watermark_removal_available():
            pytest.skip("the qwen-zimage extra is not installed")
        import diffusers

        from remove_ai_watermarks._internal.sdxl_zimage_pipeline import SdxlZImagePipeline

        calls: dict[str, dict] = {}

        def record(name):
            def fake(*_args, **kwargs):
                calls[name] = kwargs
                return MagicMock()

            return fake

        monkeypatch.setattr(diffusers.ControlNetModel, "from_pretrained", record("controlnet"))
        monkeypatch.setattr(diffusers.AutoencoderKL, "from_pretrained", record("vae"))
        monkeypatch.setattr(diffusers.StableDiffusionXLControlNetImg2ImgPipeline, "from_pretrained", record("pipeline"))
        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: "lora.safetensors")
        # from_config would otherwise resolve the mock's config as a repo id.
        monkeypatch.setattr(diffusers.EulerDiscreteScheduler, "from_config", lambda *a, **k: MagicMock())

        pipeline = SdxlZImagePipeline(device="cuda", torch_dtype=None)
        monkeypatch.setattr(type(pipeline), "_require_cuda", lambda self: None)
        pipeline._load_global()

        assert calls["pipeline"].get("add_watermarker") is False
        assert "add_watermarker" not in calls["controlnet"]


# Minimal WebP stub whose XMP chunk carries the IPTC trainedAlgorithmicMedia
# tag exactly as Muse outputs place it (built inline so the test has no binary
# fixture dependency).
_XMP_PAYLOAD = (
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax#">'
    b'<rdf:Description rdf:about="" xmlns:iptcExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" '
    b'iptcExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/>'
    b"</rdf:RDF></x:xmpmeta>"
)


def _webp_stub(xmp: bytes | None) -> bytes:
    def chunk(cid: bytes, data: bytes) -> bytes:
        return cid + struct.pack("<I", len(data)) + data + (b"\x00" if len(data) % 2 else b"")

    body = b"WEBP" + chunk(b"VP8L", b"\x00" * 20)
    if xmp is not None:
        body += chunk(b"XMP ", xmp)
    return b"RIFF" + struct.pack("<I", len(body)) + body


_MUSE_WEBP_WITH_IPTC_TAG = _webp_stub(_XMP_PAYLOAD)


class TestResolveStrength:
    """resolve_strength owns the qwen-zimage and sdxl-zimage policies."""

    def test_qwen_zimage_answers_from_the_resolution_curve(self):
        """The function is total: it owns both policies rather than returning None.

        qwen-zimage picks unknown content's strength from image area, so it takes the
        size. Returning None would push that branch onto every caller and leave one of
        the two strength policies living outside this module. Measured providers take
        flat corpus-derived operating points instead.
        """
        assert resolve_strength(None, "openai", "qwen-zimage", size=(2000, 1850)) == pytest.approx(
            QWEN_ZIMAGE_OPENAI_STRENGTH
        )
        assert resolve_strength(None, None, "qwen-zimage", size=(600, 500)) == pytest.approx(0.084)
        assert resolve_strength(None, "google", "qwen-zimage", size=(600, 500)) == QWEN_ZIMAGE_GOOGLE_STRENGTH
        # The floor holds at every size, not only above the curve's top rung.
        assert resolve_strength(None, "google", "qwen-zimage", size=(2000, 1850)) == QWEN_ZIMAGE_GOOGLE_STRENGTH

    def test_qwen_zimage_without_a_size_fails_loudly(self):
        """A missing size must not silently fall back to some vendor value."""
        with pytest.raises(ValueError, match="size is required"):
            resolve_strength(None, "openai", "qwen-zimage")

    @pytest.mark.parametrize(("vendor", "expected"), [("microsoft", 0.15), ("openai", 0.07675), ("meta", 0.1)])
    def test_qwen_zimage_measured_vendors_use_flat_cross_source_margins(self, vendor, expected):
        """Measured providers must not fall below their operating points on small files."""
        assert resolve_strength(None, vendor, "qwen-zimage", size=(600, 500)) == pytest.approx(expected)
        assert resolve_strength(None, vendor, "qwen-zimage", size=(4000, 3000)) == pytest.approx(expected)

    def test_meta_floor_is_size_independent_and_sdxl_falls_to_unknown(self):
        """The Meta floor bypasses the area curve at every size (Content Seal removal
        was bracketed on 2.56 MP generations and derived as a flat cross-source
        margin, like the other measured cohorts), while sdxl-zimage has no measured
        Meta rung and must fall to the conservative unknown value, not invent one."""
        for size in ((600, 500), (1600, 1600), (1920, 1280), (4000, 3000)):
            assert resolve_strength(None, "meta", "qwen-zimage", size=size) == pytest.approx(QWEN_ZIMAGE_META_STRENGTH)
        assert resolve_strength(None, "meta", "sdxl-zimage") == SDXL_ZIMAGE_UNKNOWN_STRENGTH

    def test_vendor_for_strength_routes_standalone_iptc_to_meta(self, tmp_path):
        """Auto mode: a file whose only provenance is the AI IPTC tag routes to the
        meta cohort (Muse carries no C2PA; the tag is its fallback companion), while
        a file without the tag stays on the resolution curve and a C2PA issuer still
        wins over the tag."""
        from remove_ai_watermarks._internal.watermark_profiles import vendor_for_strength

        tagged = tmp_path / "tagged.webp"
        tagged.write_bytes(_MUSE_WEBP_WITH_IPTC_TAG)
        assert vendor_for_strength(tagged) == "meta"

        stripped = tmp_path / "stripped.webp"
        stripped.write_bytes(b"RIFF\x24\x00\x00\x00WEBPVP8 \x18\x00\x00\x00" + b"\x00" * 16)
        assert vendor_for_strength(stripped) is None

    def test_sdxl_zimage_uses_its_flat_vendor_ladder(self):

        assert SDXL_ZIMAGE_OPENAI_STRENGTH == 0.15
        assert SDXL_ZIMAGE_GEMINI_STRENGTH == 0.25
        assert SDXL_ZIMAGE_UNKNOWN_STRENGTH == SDXL_ZIMAGE_GEMINI_STRENGTH
        assert resolve_strength(None, "openai", "sdxl-zimage") == SDXL_ZIMAGE_OPENAI_STRENGTH
        assert resolve_strength(None, "google", "sdxl-zimage") == SDXL_ZIMAGE_GEMINI_STRENGTH
        # An unrecognized issuer takes the stricter Gemini value, not the OpenAI one.
        assert resolve_strength(None, "adobe", "sdxl-zimage") == SDXL_ZIMAGE_UNKNOWN_STRENGTH
        assert resolve_strength(None, None, "sdxl-zimage") == SDXL_ZIMAGE_UNKNOWN_STRENGTH

    def test_strength_default_help_derives_from_constants(self):

        h = strength_default_help()
        assert str(SDXL_ZIMAGE_OPENAI_STRENGTH) in h
        assert str(SDXL_ZIMAGE_GEMINI_STRENGTH) in h

    def test_explicit_value_overrides_vendor(self):

        assert resolve_strength(0.3, "openai", "sdxl-zimage") == 0.3
        assert resolve_strength(0.3, None, "qwen-zimage") == 0.3

    def test_explicit_zero_is_respected_not_treated_as_unset(self):
        # 0.0 is falsy but explicit -- it must not fall through to the vendor default
        # (the old `strength or DEFAULT` bug would have). Range validation lives in
        # remove_watermark, not here.

        assert resolve_strength(0.0, "google", "sdxl-zimage") == 0.0
        assert resolve_strength(0.0, None, "qwen-zimage") == 0.0


class TestVendorForStrength:
    """Normalize supported invisible-watermark provenance to a removal profile."""

    @staticmethod
    def _patch(value):
        return patch("remove_ai_watermarks.metadata.synthid_source", return_value=value)

    def test_openai(self):
        from remove_ai_watermarks._internal.watermark_profiles import vendor_for_strength

        with self._patch("OpenAI"):
            assert vendor_for_strength(Path("x.png")) == "openai"

    def test_google(self):
        from remove_ai_watermarks._internal.watermark_profiles import vendor_for_strength

        with self._patch("Google"):
            assert vendor_for_strength(Path("x.png")) == "google"

    @pytest.mark.parametrize(("integrity", "expected"), [("valid", "microsoft"), ("invalid", None)])
    def test_only_valid_microsoft_invismark_selects_the_removal_floor(self, integrity, expected):
        from remove_ai_watermarks._internal.watermark_profiles import vendor_for_strength

        info = {
            "soft_binding_vendors": ["Microsoft InvisMark"],
            "c2pa_integrity": integrity,
            "c2pa_signature": "valid",
            "c2pa_signer_validity": "valid",
        }
        with (
            self._patch(None),
            patch("remove_ai_watermarks._internal.c2pa.extract_c2pa_info", return_value=info),
        ):
            assert vendor_for_strength(Path("x.png")) == expected

    def test_both_issuers_google_wins(self):
        # The more-robust watermark wins -> safer (higher) strength.
        from remove_ai_watermarks._internal.watermark_profiles import vendor_for_strength

        with self._patch("OpenAI, Google"):
            assert vendor_for_strength(Path("x.png")) == "google"

    def test_none_when_no_synthid_source(self):
        from remove_ai_watermarks._internal.watermark_profiles import vendor_for_strength

        with self._patch(None):
            assert vendor_for_strength(Path("x.png")) is None

    def test_unreadable_metadata_is_none(self):
        from remove_ai_watermarks._internal.watermark_profiles import vendor_for_strength

        with patch("remove_ai_watermarks.metadata.synthid_source", side_effect=OSError):
            assert vendor_for_strength(Path("x.png")) is None


# ── Format utilities ────────────────────────────────────────────────


class TestFormatUtils:
    """Tests for utils.py format helpers."""

    def test_supported_png(self, tmp_path):
        assert is_supported_format(tmp_path / "test.png")

    def test_supported_jpg(self, tmp_path):
        assert is_supported_format(tmp_path / "test.jpg")

    def test_supported_jpeg(self, tmp_path):
        assert is_supported_format(tmp_path / "test.jpeg")

    def test_supported_webp(self, tmp_path):
        assert is_supported_format(tmp_path / "test.webp")

    def test_unsupported_bmp(self, tmp_path):
        assert not is_supported_format(tmp_path / "test.bmp")

    def test_unsupported_gif(self, tmp_path):
        assert not is_supported_format(tmp_path / "test.gif")

    def test_get_format_png(self, tmp_path):
        assert get_image_format(tmp_path / "x.png") == "PNG"

    def test_get_format_jpg(self, tmp_path):
        assert get_image_format(tmp_path / "x.jpg") == "JPEG"

    def test_get_format_jpeg(self, tmp_path):
        assert get_image_format(tmp_path / "x.jpeg") == "JPEG"

    def test_get_format_webp_defaults_png(self, tmp_path):
        # .webp falls through to PNG in current implementation
        assert get_image_format(tmp_path / "x.webp") == "PNG"


# ── Availability checks ────────────────────────────────────────────


class TestAvailability:
    """The CLI gate and the remover precondition must answer from the same module list.

    Both used to hardcode (torch, diffusers) while the code moved to REMOVAL_MODULES,
    which includes diffsynth. In a torch+diffusers-only environment the assertions were
    then simply wrong -- and, worse, comparing each gate against a tuple copied from
    itself can never catch the two disagreeing, which is the drift that let the CLI pass
    an environment the run then died in.
    """

    def test_both_gates_agree_and_read_the_shared_module_list(self):
        import importlib.util

        from remove_ai_watermarks.invisible_engine import is_available

        assert "diffsynth" in REMOVAL_MODULES
        expected = all(importlib.util.find_spec(m) is not None for m in REMOVAL_MODULES)
        assert is_watermark_removal_available() is expected
        assert is_available() is expected

    def test_a_missing_module_closes_both_gates(self, monkeypatch: pytest.MonkeyPatch):
        """Discriminating, not vacuous: it must FAIL if either gate stops requiring one.

        Comparing the live answer to a tuple derived from the same constant passes on
        any host -- with the full stack (True == True) and with none of it
        (False == False). Simulate each module's absence instead.
        """
        import remove_ai_watermarks.invisible_engine as engine_module
        from remove_ai_watermarks._internal import watermark_remover as remover_module

        for missing in REMOVAL_MODULES:
            present = {name: name != missing for name in REMOVAL_MODULES}
            monkeypatch.setattr(
                "remove_ai_watermarks.optional_deps.module_available",
                lambda *names, _p=present: all(_p.get(n, True) for n in names),
            )
            # The remover probes at import time, so drive its cached flags directly.
            monkeypatch.setattr(remover_module, "_HAS_TORCH", missing != "torch")
            monkeypatch.setattr(remover_module, "_HAS_REMOVAL_MODULES", missing == "torch")
            assert engine_module.is_available() is False, f"engine gate ignores a missing {missing}"
            assert remover_module.is_watermark_removal_available() is False, f"remover gate ignores a missing {missing}"


# ── Platform-specific path handling ─────────────────────────────────


class TestPlatformPaths:
    """Verify path handling works on current platform."""

    def test_pathlib_works_for_assets(self):
        from pathlib import Path

        asset_dir = Path(__file__).parent.parent / "src" / "remove_ai_watermarks" / "assets"
        assert (asset_dir / "gemini_bg_48.png").exists()
        assert (asset_dir / "gemini_bg_96.png").exists()
        assert (asset_dir / "qwen_symbol_alpha.png").exists()

    def test_asset_loading_works(self):
        """Verify embedded assets load correctly (critical for packaging)."""
        from remove_ai_watermarks.gemini_engine import GeminiEngine

        engine = GeminiEngine()
        # If we get here without error, asset loading works
        assert engine._alpha_small.shape == (48, 48)
        assert engine._alpha_large.shape == (96, 96)

        from remove_ai_watermarks.qwen_engine import QwenEngine

        assert QwenEngine()._symbol_template.shape == (53, 53)
