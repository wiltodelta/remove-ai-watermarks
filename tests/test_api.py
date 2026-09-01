"""High-level convenience API (remove_visible / visible_provenance) and the lazy
top-level re-exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import remove_ai_watermarks as raiw
from remove_ai_watermarks import api

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "provenance"
DOUBAO = SAMPLES / "doubao-1.png"
CHATGPT = SAMPLES / "chatgpt-1.png"


class TestTopLevelExports:
    def test_lazy_reexports_resolve(self):
        from remove_ai_watermarks import openai_provenance

        assert raiw.remove_visible is api.remove_visible
        assert raiw.visible_provenance is api.visible_provenance
        assert raiw.verify_openai_synthid is openai_provenance.verify_openai_synthid
        assert raiw.OpenAIProvenanceError is openai_provenance.OpenAIProvenanceError
        assert raiw.OpenAISynthIDDetection is openai_provenance.OpenAISynthIDDetection

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            _ = raiw.does_not_exist

    def test_local_synthid_detector_is_not_a_package_export(self):
        with pytest.raises(AttributeError):
            _ = raiw.detect_synthid
        with pytest.raises(AttributeError):
            _ = raiw.SynthIDDetection

    def test_bare_import_is_light(self):
        # importing the package must not pull the heavy cv2/torch stack (PEP 562 lazy).
        # Checked in a FRESH interpreter -- another test in this process may already
        # have imported cv2, so an in-process sys.modules check would be flaky.
        import subprocess
        import sys

        code = "import remove_ai_watermarks, sys; print(int(any(m in sys.modules for m in ('cv2','torch'))))"
        out = subprocess.run(  # noqa: S603  -- fixed sys.executable + literal code, no untrusted input
            [sys.executable, "-c", code], check=True, capture_output=True, text=True
        )
        assert out.stdout.strip() == "0", f"bare import pulled a heavy module: {out.stdout!r}"


class TestRemoveVisibleArray:
    def test_array_no_mark_is_noop_copy(self):
        arr = np.zeros((256, 256, 3), np.uint8)
        result, removed = raiw.remove_visible(arr, backend="cv2")
        assert removed == []
        assert result.shape == arr.shape
        assert np.array_equal(result, arr)

    def test_array_accepts_knobs(self):
        arr = np.zeros((256, 256, 3), np.uint8)
        result, removed = raiw.remove_visible(arr, sensitivity="strict", backend="cv2")
        assert removed == []
        assert result.shape == arr.shape

    def test_bad_source_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Could not read image"):
            raiw.remove_visible(tmp_path / "nope.png")


@pytest.mark.skipif(not DOUBAO.exists(), reason="doubao sample not present")
class TestRemoveVisiblePath:
    def test_path_removes_and_writes(self, tmp_path):
        out = tmp_path / "clean.png"
        result, removed = raiw.remove_visible(DOUBAO, out, backend="cv2")
        assert out.exists()
        assert any("Doubao" in lbl for lbl in removed)
        assert result.shape[2] == 3

    def test_path_no_output_returns_without_writing(self, tmp_path):
        # output=None returns the array but writes nothing
        result, _ = raiw.remove_visible(DOUBAO, backend="cv2")
        assert result.ndim == 3


class TestNoOpPreservesOriginal:
    def test_no_mark_copies_original_bytes(self, tmp_path):
        # A clean image (no mark) same-format-out must be copied VERBATIM, not
        # re-encoded -- so a no-op never degrades the original ("work with originals").
        import filecmp

        from PIL import Image

        src = tmp_path / "clean.jpg"
        Image.fromarray(np.full((40, 40, 3), 120, np.uint8), "RGB").save(src, quality=90)
        out = tmp_path / "clean_out.jpg"
        _, removed = raiw.remove_visible(str(src), str(out), sensitivity="strict", backend="cv2")
        assert removed == []
        assert filecmp.cmp(str(src), str(out), shallow=False)  # byte-identical


class TestVisibleProvenance:
    @pytest.mark.skipif(not DOUBAO.exists(), reason="doubao sample not present")
    def test_doubao_tc260_maps_to_the_producer_it_names(self):
        """The TC260 producer identifies the vendor, so only Doubao is relaxed.

        This used to relax Doubao AND Jimeng on every China-AIGC image, because the
        label alone does not say which vendor made it. Its ``ContentProducer`` does.
        """
        prov = raiw.visible_provenance(DOUBAO)
        assert "doubao" in prov
        assert "jimeng" not in prov

    def test_unmapped_tc260_producer_falls_back_to_the_bytedance_pair(self, monkeypatch, tmp_path):
        """An unrecognized producer must not lose the relaxation entirely: the label is
        still evidence that some China-AIGC vendor made the image."""
        from types import SimpleNamespace

        from remove_ai_watermarks import identify, metadata

        monkeypatch.setattr(
            identify,
            "identify",
            lambda *a, **k: SimpleNamespace(platform=None, signals=[SimpleNamespace(name="aigc")]),
        )
        monkeypatch.setattr(metadata, "aigc_label", lambda _p: {"ContentProducer": "0011999999999999999999999"})
        assert raiw.visible_provenance(tmp_path / "x.png") == frozenset({"doubao", "jimeng"})

    def test_known_tc260_producer_names_a_single_vendor(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        from remove_ai_watermarks import identify, metadata

        monkeypatch.setattr(
            identify,
            "identify",
            lambda *a, **k: SimpleNamespace(platform=None, signals=[SimpleNamespace(name="aigc")]),
        )
        # 001 + 1 + USCC(18) + 5-digit product suffix, the Qwen entity.
        monkeypatch.setattr(metadata, "aigc_label", lambda _p: {"ContentProducer": "001191440101MA9Y9T4H7A00001"})
        assert raiw.visible_provenance(tmp_path / "x.png") == frozenset({"qwen"})

    def test_every_mapped_producer_names_a_registered_mark(self):
        """The table drives the arbiter's provenance set, so a typo'd key would relax
        nothing and fail silently. Checked here, not at import: importing the registry
        from _internal.constants would drag it into every metadata-only path."""
        from remove_ai_watermarks._internal.constants import TC260_FALLBACK_VENDORS
        from remove_ai_watermarks.watermark_registry import known_marks, mark_keys, tc260_producer_vendors

        keys = set(mark_keys())
        assert set(tc260_producer_vendors().values()) <= keys
        assert keys >= TC260_FALLBACK_VENDORS
        # Every TC260 mark should name its producer, or it silently falls back to the
        # ByteDance pair on an image carrying that mark.
        unmapped = {m.key for m in known_marks() if m.label_regime == "tc260" and not m.tc260_producer_codes}
        assert unmapped == {"jimeng_pill"}, f"TC260 marks with no producer code: {unmapped}"

    @pytest.mark.skipif(not CHATGPT.exists(), reason="chatgpt sample not present")
    def test_openai_image_has_no_visible_vendor(self):
        # OpenAI C2PA is not one of the visible-mark vendors -> empty provenance
        assert raiw.visible_provenance(CHATGPT) == frozenset()

    def test_unreadable_path_is_empty(self, tmp_path):
        assert raiw.visible_provenance(tmp_path / "missing.png") == frozenset()

    def test_uses_report_signals_for_falsy_metadata_values(self, monkeypatch, tmp_path):
        """An empty TC260 object and Samsung genAIType=0 are still present signals.

        The report has already normalized those values, so the public API must not
        re-read the file and accidentally discard them by truthiness.
        """
        from types import SimpleNamespace

        from remove_ai_watermarks import identify

        report = SimpleNamespace(
            platform=None,
            signals=[SimpleNamespace(name="aigc"), SimpleNamespace(name="samsung_genai")],
        )
        monkeypatch.setattr(identify, "identify", lambda *args, **kwargs: report)

        assert raiw.visible_provenance(tmp_path / "synthetic.png") == frozenset({"doubao", "jimeng", "samsung"})


class TestRemoveVisibleOutputPath:
    """Output-path robustness: in-place clean (#3) and a missing output dir (#4)."""

    def _write_clean(self, p: Path) -> None:
        from remove_ai_watermarks import image_io

        image_io.imwrite(str(p), np.full((128, 128, 3), 200, np.uint8))

    def test_inplace_clean_no_crash(self, tmp_path: Path):
        p = tmp_path / "clean.png"
        self._write_clean(p)
        _, removed = raiw.remove_visible(str(p), str(p), backend="cv2")
        assert removed == []
        assert p.exists()

    def test_creates_missing_output_dir(self, tmp_path: Path):
        src = tmp_path / "in.png"
        self._write_clean(src)
        out = tmp_path / "sub" / "out.png"
        raiw.remove_visible(str(src), str(out), backend="cv2")
        assert out.exists()


class TestInvisibleOptionsMirrorTheEngine:
    """``InvisibleOptions`` forwards to ``InvisibleEngine`` and promises every NAME and
    default mirrors it. Compare the signatures field by field rather than pinning the
    values we happen to know about, so the next field added on one side and not the
    other fails here. The comparison deliberately has no exception table: a field that
    needs one is a field that belongs somewhere else, which is why ``force`` is a
    parameter of ``remove_all``. See the incident record in ``docs/module-internals.md``."""

    def test_every_default_matches_the_engine(self):
        import dataclasses
        import inspect

        from remove_ai_watermarks.invisible_engine import InvisibleEngine

        engine = {
            name: p.default
            for method in (InvisibleEngine.__init__, InvisibleEngine.remove_watermark)
            for name, p in inspect.signature(method).parameters.items()
            if p.default is not inspect.Parameter.empty
        }
        options = {f.name: f.default for f in dataclasses.fields(api.InvisibleOptions)}

        assert options == {name: engine.get(name, "<not an engine parameter>") for name in options}

    @pytest.mark.skipif(not CHATGPT.exists(), reason="sample image not present")
    def test_every_field_arrives_at_the_engine_with_the_caller_s_value(self, monkeypatch, tmp_path):
        """A defaults comparison is not a forwarding test. `_run_invisible` hands each
        field to one of TWO engine callables by hand, and a hardcoded literal there is
        invisible to the mirror above -- `controlnet_conditioning_scale` shipped that way
        and the whole suite stayed green. Drive the real seam with every field set OFF
        its default and assert the caller's value arrives, whichever callable takes it."""
        import dataclasses

        from remove_ai_watermarks import invisible_engine

        # Every value differs from the default, so a hardcoded default cannot pass.
        opts = api.InvisibleOptions(
            strength=0.42,
            pipeline="sdxl-zimage",
            seed=7,
            hf_token="token",
            humanize=0.3,
            unsharp=0.2,
            adaptive_polish=True,
            max_resolution=1536,
            controlnet_conditioning_scale=0.65,
            cpu_offload=True,
            tile=True,
            tile_size=768,
            tile_overlap=64,
            text_manifest=tmp_path / "verified-lines.json",
        )
        seen: dict[str, object] = {}

        class FakeEngine:
            def __init__(self, **kwargs):
                seen.update(kwargs)

            def remove_watermark(self, **kwargs):
                seen.update(kwargs)

        monkeypatch.setattr(invisible_engine, "is_available", lambda: True)
        monkeypatch.setattr(invisible_engine, "InvisibleEngine", FakeEngine)
        api._run_invisible(
            CHATGPT,
            CHATGPT,
            tmp_path / "out.png",
            opts,
            None,
            lambda _stage, _detail: None,
            api._SourceEvidence(CHATGPT),
            True,
        )

        missing = {f.name: getattr(opts, f.name) for f in dataclasses.fields(opts) if f.name not in seen}
        assert not missing, f"never forwarded to the engine: {missing}"
        wrong = {
            f.name: (getattr(opts, f.name), seen[f.name])
            for f in dataclasses.fields(opts)
            if seen[f.name] != getattr(opts, f.name)
        }
        assert not wrong, f"forwarded a value the caller did not pass (want, got): {wrong}"


class TestRemoveAllLibrary:
    """The three-stage pipeline is a library function, not CLI-only.

    It used to live only in ``cli.py``, written twice (once for ``all``, once for
    ``batch``) with divergent behavior, so no library caller could reach it.
    """

    def _patched_engine(self, monkeypatch, calls: list):
        from remove_ai_watermarks import invisible_engine

        class FakeEngine:
            def remove_watermark(self, *args, **kwargs):
                calls.append(kwargs.get("image_path") or args[0])

        monkeypatch.setattr(invisible_engine, "is_available", lambda: True)
        return FakeEngine()

    @pytest.mark.skipif(not DOUBAO.exists(), reason="doubao sample not present")
    def test_runs_all_three_stages_and_reports_each(self, monkeypatch, tmp_path):
        from remove_ai_watermarks import api

        calls: list = []
        engine = self._patched_engine(monkeypatch, calls)
        monkeypatch.setattr(api._SourceEvidence, "has_invisible_target", lambda _self: True)
        out = tmp_path / "clean.png"
        events: list[tuple[str, str]] = []

        result = api.remove_all(DOUBAO, out, backend="cv2", engine=engine, progress=lambda s, d: events.append((s, d)))

        assert out.exists()
        assert result.invisible == "removed"
        assert result.visible_label is not None  # the Doubao mark fired
        assert calls, "the invisible engine was never invoked"
        # Progress is (stage, stable-token), never prose the caller has to parse back.
        assert any(stage == "visible" for stage, _ in events)
        assert ("invisible", "removed") in events
        assert ("metadata", "stripped") in events

    @pytest.mark.skipif(not DOUBAO.exists(), reason="doubao sample not present")
    def test_no_signal_skips_the_scrub_but_still_writes(self, monkeypatch, tmp_path):
        from remove_ai_watermarks import api

        calls: list = []
        engine = self._patched_engine(monkeypatch, calls)
        monkeypatch.setattr(api._SourceEvidence, "has_invisible_target", lambda _self: False)
        out = tmp_path / "clean.png"

        result = api.remove_all(DOUBAO, out, backend="cv2", engine=engine)

        assert result.invisible == "no-signal"
        assert not calls
        assert out.exists()  # a deliberate skip is still a successful run

    @pytest.mark.skipif(not DOUBAO.exists(), reason="doubao sample not present")
    def test_missing_gpu_extra_is_reported_not_raised(self, monkeypatch, tmp_path):
        from remove_ai_watermarks import api, invisible_engine

        monkeypatch.setattr(invisible_engine, "is_available", lambda: False)
        out = tmp_path / "clean.png"

        result = api.remove_all(DOUBAO, out, backend="cv2")

        assert result.invisible == "unavailable"
        assert out.exists()  # it LOOKS processed -- which is why the caller must warn

    @pytest.mark.skipif(not DOUBAO.exists(), reason="doubao sample not present")
    def test_incomplete_strip_leaves_no_output_file(self, monkeypatch, tmp_path):
        """The contract the CLI depends on: an AI-readable output plus a non-zero exit
        is worse than no output at all, so the raise happens BEFORE the final write."""
        from remove_ai_watermarks import api, invisible_engine, metadata

        monkeypatch.setattr(invisible_engine, "is_available", lambda: False)
        monkeypatch.setattr(metadata, "strip_and_verify", lambda src, dst: (dst, {"c2pa"}))
        out = tmp_path / "clean.png"

        with pytest.raises(api.MetadataStripIncomplete, match="c2pa"):
            api.remove_all(DOUBAO, out, backend="cv2")
        assert not out.exists()

    def test_unreadable_source_raises_valueerror(self, tmp_path):
        from remove_ai_watermarks import api

        with pytest.raises(ValueError, match="Could not read image"):
            api.remove_all(tmp_path / "nope.png", tmp_path / "out.png")

    @pytest.mark.skipif(not DOUBAO.exists(), reason="doubao sample not present")
    def test_stages_through_the_system_temp_dir_not_the_output_dir(self, monkeypatch, tmp_path):
        """Staging next to the output would defeat the point: the user must not see a
        partial file there during a long model download."""
        from remove_ai_watermarks import api, invisible_engine

        monkeypatch.setattr(invisible_engine, "is_available", lambda: False)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        seen: list[set[str]] = []

        real = api.remove_all

        def spy(*args, **kwargs):
            result = real(*args, **kwargs)
            seen.append({p.name for p in out_dir.iterdir()})
            return result

        spy(DOUBAO, out_dir / "clean.png", backend="cv2")
        assert seen == [{"clean.png"}], "an intermediate was left in the output directory"


class TestRemoveBatchLibrary:
    @pytest.mark.skipif(not DOUBAO.exists(), reason="doubao sample not present")
    def test_visible_mode_writes_every_image(self, tmp_path):
        import shutil

        from remove_ai_watermarks import api

        src = tmp_path / "in"
        src.mkdir()
        for i in range(3):
            shutil.copyfile(DOUBAO, src / f"img{i}.png")
        out = tmp_path / "out"

        summary = api.remove_batch(src, out, mode="visible", backend="cv2")

        assert summary.processed == 3
        assert summary.failed == 0
        assert sorted(p.name for p in out.iterdir()) == ["img0.png", "img1.png", "img2.png"]

    def test_one_bad_file_does_not_abandon_the_rest(self, tmp_path):
        from remove_ai_watermarks import api

        src = tmp_path / "in"
        src.mkdir()
        (src / "broken.png").write_bytes(b"not a png at all")
        image = np.full((64, 64, 3), 120, np.uint8)
        raiw.remove_visible(image)  # sanity: the registry is importable here
        from remove_ai_watermarks import image_io

        image_io.imwrite(src / "good.png", image)
        out = tmp_path / "out"

        summary = api.remove_batch(src, out, mode="visible", backend="cv2")

        assert summary.processed == 1
        assert summary.failed == 1
        assert [p.name for p, _ in summary.errors] == ["broken.png"]
        assert (out / "good.png").exists()

    def test_a_failed_write_is_counted_not_swallowed(self, tmp_path, monkeypatch):
        """Tier E: a read-only output directory once produced zero files and exit 0."""
        from remove_ai_watermarks import api, image_io

        src = tmp_path / "in"
        src.mkdir()
        image_io.imwrite(src / "a.png", np.full((64, 64, 3), 120, np.uint8))
        monkeypatch.setattr(image_io, "write_bgr_with_alpha", lambda *a, **k: False)

        summary = api.remove_batch(src, tmp_path / "out", mode="visible", backend="cv2")

        assert summary.processed == 0
        assert summary.failed == 1

    @pytest.mark.parametrize("mode", ["all", "invisible"])
    @pytest.mark.parametrize(("force", "expected"), [(True, "removed"), (False, "no-signal")])
    def test_force_reaches_the_scrub_gate_in_every_scrubbing_mode(self, monkeypatch, tmp_path, mode, force, expected):
        """``force`` reaches the gate through a DIFFERENT seam per mode: ``all`` re-enters
        ``remove_all``, ``invisible`` calls ``_run_invisible`` directly. Only the second
        was guarded, and pinning the first to False passed the whole suite while every
        output kept its watermark and the run still reported the files as processed."""
        from remove_ai_watermarks import api, image_io, invisible_engine

        src = tmp_path / "in"
        src.mkdir()
        for i in range(2):
            image_io.imwrite(src / f"img{i}.png", np.full((64, 64, 3), 120, np.uint8))
        scrubbed: list[Path] = []

        class FakeEngine:
            def remove_watermark(self, **kwargs):
                scrubbed.append(kwargs["image_path"])
                image_io.imwrite(kwargs["output_path"], np.full((64, 64, 3), 120, np.uint8))

        monkeypatch.setattr(invisible_engine, "is_available", lambda: True)
        events: list[tuple[str, str]] = []

        summary = api.remove_batch(
            src,
            tmp_path / "out",
            mode=mode,
            backend="cv2",
            force=force,
            engine=FakeEngine(),
            progress=lambda _p, stage, detail: events.append((stage, detail)),
        )

        assert summary.processed == 2
        assert ("invisible", expected) in events
        assert len(scrubbed) == (2 if force else 0)


class TestSourceEvidenceHolder:
    """One metadata extraction per source file, per call.

    ``remove_all`` asks the same file two provenance questions (which vendor is
    confirmed, and is there an invisible target); both start from the same extraction.
    """

    @pytest.mark.skipif(not DOUBAO.exists(), reason="doubao sample not present")
    def test_remove_all_extracts_evidence_once(self, monkeypatch, tmp_path):
        import shutil

        from remove_ai_watermarks import api, identify, invisible_engine

        class Fake:
            def remove_watermark(self, *args, **kwargs):
                pass

        source = tmp_path / "in.png"
        shutil.copyfile(DOUBAO, source)
        calls: list[int] = []
        real = identify.extract_provenance_evidence
        monkeypatch.setattr(identify, "extract_provenance_evidence", lambda p: (calls.append(1), real(p))[1])
        monkeypatch.setattr(invisible_engine, "is_available", lambda: True)

        api.remove_all(source, tmp_path / "out.png", backend="cv2", engine=Fake())
        assert len(calls) == 1

    @pytest.mark.skipif(not DOUBAO.exists(), reason="doubao sample not present")
    def test_holder_agrees_with_the_standalone_functions(self):
        from remove_ai_watermarks import api, identify

        holder = api._SourceEvidence(DOUBAO)
        assert holder.visible_provenance() == api.visible_provenance(DOUBAO)
        assert holder.has_invisible_target() == identify.has_invisible_target(DOUBAO)

    def test_holder_preserves_invalid_c2pa_removal_hint(self, tampered_chatgpt_png):
        from remove_ai_watermarks import api, identify

        holder = api._SourceEvidence(tampered_chatgpt_png)

        assert identify.identify(tampered_chatgpt_png, check_visible=False).ai_from_metadata is False
        assert holder.has_invisible_target() is True
        assert holder.has_invisible_target() == identify.has_invisible_target(tampered_chatgpt_png)

    def test_extraction_failure_fails_safe_in_both_directions(self, monkeypatch, tmp_path):
        """No provenance means no relaxation; an unknown invisible target means SCRUB.
        Leaving a watermark on a paid removal is worse than over-regenerating."""
        from remove_ai_watermarks import api, identify

        def boom(_path):
            raise OSError("extract exploded")

        monkeypatch.setattr(identify, "extract_provenance_evidence", boom)
        holder = api._SourceEvidence(tmp_path / "x.png")
        assert holder.visible_provenance() == frozenset()
        assert holder.has_invisible_target() is True

    @pytest.mark.skipif(not DOUBAO.exists(), reason="doubao sample not present")
    def test_a_verdict_failure_also_fails_safe(self, monkeypatch):
        """The suppress must span the VERDICT and the mapping, not just the extraction:
        a raise here used to escape as a traceback where the old code returned empty."""
        from remove_ai_watermarks import api, identify

        def boom(*args, **kwargs):
            raise RuntimeError("verdict exploded")

        monkeypatch.setattr(identify, "identify_from_evidence", boom)
        holder = api._SourceEvidence(DOUBAO)
        assert holder.visible_provenance() == frozenset()
        assert holder.has_invisible_target() is True


class TestBatchProgressIsStructured:
    """`remove_batch` emits exactly one terminal event per image, in every mode.

    Regression: the CLI advanced its progress bar by string-matching a stage line that
    only ``mode="all"`` ever emitted, so a `visible` or `metadata` batch sat at 0% for
    the whole run and jumped to 100% at the end.
    """

    def _run(self, tmp_path, mode: str) -> list[tuple[str, str, str]]:
        from remove_ai_watermarks import api, image_io

        src = tmp_path / "in"
        src.mkdir()
        for i in range(3):
            image_io.imwrite(src / f"img{i}.png", np.full((64, 64, 3), 120, np.uint8))
        events: list[tuple[str, str, str]] = []
        api.remove_batch(
            src,
            tmp_path / "out",
            mode=mode,  # type: ignore[arg-type]
            backend="cv2",
            progress=lambda p, stage, detail: events.append((p.name, stage, detail)),
        )
        return events

    @pytest.mark.parametrize("mode", ["visible", "metadata"])
    def test_one_terminal_event_per_image(self, tmp_path, mode):
        events = self._run(tmp_path, mode)
        terminal = [name for name, stage, _ in events if stage in ("done", "failed")]
        assert sorted(terminal) == ["img0.png", "img1.png", "img2.png"]

    def test_progress_is_a_token_not_prose(self, tmp_path):
        """The CLI keys console text off these tokens; free text would break it."""
        events = self._run(tmp_path, "visible")
        assert {stage for _, stage, _ in events} <= {"visible", "invisible", "metadata", "done", "failed"}
