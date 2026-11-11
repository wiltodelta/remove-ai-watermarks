"""Every command fails closed: an unreadable input or a skipped step never exits 0.

Found 2026-10-18 by a sweep that ran each command over random bytes, an empty file
and a truncated copy of a tracked fixture. The pixel commands already refused all
three. The metadata-only paths did not: they scan raw bytes, a scan of bytes that
are not an image finds nothing, and "nothing" printed as "No AI metadata found"
with exit 0, the same answer a real file without metadata gets. `metadata
--remove` then copied the garbage through and reported "AI metadata stripped".
A truncated MP4 without its `moov` box got the same treatment from the video
metadata commands. The rule pinned here: the exit code of a scan that could not
read its input is never the exit code of a scan that read it and found nothing.

Absent OPTIONAL detectors are not in this set. `identify` on the default install,
which has no pixel stack, still answers with exit 0 and a caveat naming what did
not run (decided 2026-10-18); see `test_optional_deps.py`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from remove_ai_watermarks import image_io, metadata
from remove_ai_watermarks._internal.constants import SUPPORTED_FORMATS
from remove_ai_watermarks.cli import main
from remove_ai_watermarks.video import VIDEO_EXTENSIONS

ROOT = Path(__file__).resolve().parents[1]
C2PA_PNG = ROOT / "data/fixtures/provenance/ideogram-c2pa.png"
MP4 = ROOT / "data/fixtures/provenance/runway-gen45-video.mp4"


def _unreadable_images(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    garbage = tmp_path / "garbage.png"
    garbage.write_bytes(rng.integers(0, 256, 4000, dtype=np.uint8).tobytes())
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(C2PA_PNG.read_bytes()[:300])
    return {"garbage": garbage, "empty": empty, "truncated": truncated}


@pytest.fixture
def clean_png(tmp_path: Path) -> Path:
    path = tmp_path / "clean.png"
    gradient = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    assert image_io.imwrite(path, np.dstack([gradient, gradient.T, gradient]))
    return path


@pytest.fixture
def truncated_mp4(tmp_path: Path) -> Path:
    """The ftyp box survives, so the suffix sniff passes; the box walk does not."""
    path = tmp_path / "truncated.mp4"
    path.write_bytes(MP4.read_bytes()[:3000])
    return path


@pytest.fixture
def cuda_refused(monkeypatch):
    """The invisible engine is installed but cannot start, as on a host without CUDA."""
    import remove_ai_watermarks.invisible_engine as invisible_engine

    def refuse(*_args, **_kwargs):
        raise ValueError("Invisible-watermark removal is CUDA-only")

    monkeypatch.setattr(invisible_engine, "is_available", lambda: True)
    monkeypatch.setattr(invisible_engine, "InvisibleEngine", refuse)


def _invoke(args: list[str]):
    return CliRunner().invoke(main, args)


class TestUnreadableImageInputs:
    @pytest.mark.parametrize("kind", ["garbage", "empty", "truncated"])
    @pytest.mark.parametrize("flags", [[], ["--no-visible"], ["--json"]])
    def test_identify_exits_nonzero(self, tmp_path, kind, flags):
        result = _invoke(["identify", *flags, str(_unreadable_images(tmp_path)[kind])])
        assert result.exit_code == 1, result.output
        assert isinstance(result.exception, SystemExit)

    def test_identify_keeps_exit_zero_on_a_readable_file(self, clean_png):
        result = _invoke(["identify", str(clean_png)])
        assert result.exit_code == 0, result.output

    @pytest.mark.parametrize("kind", ["garbage", "empty", "truncated"])
    def test_metadata_check_does_not_report_no_metadata(self, tmp_path, kind):
        result = _invoke(["metadata", "--check", str(_unreadable_images(tmp_path)[kind])])
        assert result.exit_code == 1, result.output
        assert "No AI metadata found" not in result.output
        assert "Nothing was scanned" in result.output, "the probe runs before any scan"

    @pytest.mark.parametrize("kind", ["garbage", "empty", "truncated"])
    def test_metadata_remove_writes_nothing_and_exits_nonzero(self, tmp_path, kind):
        out = tmp_path / "out.png"
        result = _invoke(["metadata", "--remove", str(_unreadable_images(tmp_path)[kind]), "-o", str(out)])
        assert result.exit_code == 1, result.output
        assert "stripped" not in result.output
        assert not out.exists()

    def test_metadata_remove_keeps_working_on_a_readable_file(self, clean_png, tmp_path):
        result = _invoke(["metadata", "--remove", str(clean_png), "-o", str(tmp_path / "out.png")])
        assert result.exit_code == 0, result.output

    def test_batch_metadata_counts_an_unreadable_file_as_failed(self, tmp_path, clean_png):
        indir = tmp_path / "in"
        indir.mkdir()
        shutil.copy(clean_png, indir / "clean.png")
        shutil.copy(_unreadable_images(tmp_path)["garbage"], indir / "garbage.png")
        result = _invoke(["batch", str(indir), "--mode", "metadata", "-o", str(tmp_path / "out")])
        assert result.exit_code == 1, result.output
        assert "1 processed" in result.output
        assert "1 errors" in result.output


class TestTruncatedVideo:
    @pytest.mark.parametrize(
        "args",
        [
            ["video", "identify", "--no-visible"],
            ["video", "metadata", "--check"],
            ["video", "metadata", "--remove"],
            ["metadata", "--check"],
        ],
    )
    def test_metadata_paths_exit_nonzero(self, truncated_mp4, args):
        result = _invoke([*args, str(truncated_mp4)])
        assert result.exit_code == 1, result.output
        assert "No AI metadata found" not in result.output
        assert "stripped" not in result.output

    def test_video_batch_metadata_counts_it_as_failed(self, tmp_path, truncated_mp4):
        indir = tmp_path / "in"
        indir.mkdir()
        shutil.copy(truncated_mp4, indir / "truncated.mp4")
        result = _invoke(["video", "batch", str(indir), "--mode", "metadata", "-o", str(tmp_path / "out")])
        assert result.exit_code == 1, result.output
        assert "0 processed, 1 failed" in result.output


class TestReadabilityProbe:
    """The probe must pass every tracked fixture, or the fix trades one lie for another."""

    @pytest.mark.parametrize(
        "path",
        sorted(p for p in (ROOT / "data/fixtures").rglob("*") if p.is_file() and p.suffix.lower() != ".md"),
        ids=lambda p: str(p.relative_to(ROOT / "data/fixtures")),
    )
    def test_every_tracked_fixture_is_readable(self, path):
        if path.suffix.lower() not in SUPPORTED_FORMATS | VIDEO_EXTENSIONS:
            pytest.skip("not a container the metadata commands accept")
        assert metadata.unreadable_reason(path) is None

    def test_unreadable_inputs_name_a_reason(self, tmp_path, truncated_mp4):
        for path in [*_unreadable_images(tmp_path).values(), truncated_mp4]:
            assert metadata.unreadable_reason(path), path

    def test_a_png_cut_inside_its_pixel_data_is_unreadable(self, tmp_path):
        """The header parses, so only a full decode sees the missing IDAT bytes."""
        path = tmp_path / "cut.png"
        data = (ROOT / "data/fixtures/provenance/chatgpt-1.png").read_bytes()
        path.write_bytes(data[: len(data) * 3 // 5])
        assert metadata.unreadable_reason(path)

    def test_a_trailing_partial_box_is_unreadable(self, tmp_path):
        """Cut inside the LAST box, so ftyp and moov both survive and only the walk sees it."""
        data = MP4.read_bytes()
        path = tmp_path / "cut.mp4"
        path.write_bytes(data[:-100])
        assert metadata.unreadable_reason(path)

    def test_a_truncated_ebml_segment_is_unreadable(self, tmp_path):
        """A Segment that declares more bytes than the file holds, as a cut WebM does."""
        path = tmp_path / "x.webm"
        ebml_header = b"\x1a\x45\xdf\xa3\x80"  # empty EBML header element
        segment = b"\x18\x53\x80\x67\x88" + bytes(4)  # declares 8 payload bytes, holds 4
        path.write_bytes(ebml_header + segment)
        assert metadata.unreadable_reason(path)
        path.write_bytes(ebml_header + segment + bytes(4))
        assert metadata.unreadable_reason(path) is None

    def test_a_truncated_riff_is_unreadable(self, tmp_path):
        path = tmp_path / "x.wav"
        path.write_bytes(b"RIFF" + (100).to_bytes(4, "little") + b"WAVE" + bytes(20))
        assert metadata.unreadable_reason(path)
        path.write_bytes(b"RIFF" + (100).to_bytes(4, "little") + b"WAVE" + bytes(96))
        assert metadata.unreadable_reason(path) is None


class TestSkippedStepsAreFailures:
    @pytest.mark.parametrize(
        "args",
        [
            ["visible", "--mark", "doubao", "--no-detect"],
            ["erase", "--region", "2,2,10,10"],
        ],
    )
    def test_a_metadata_strip_that_left_markers_is_not_success(self, tmp_path, clean_png, monkeypatch, args):
        """Both commands used to swallow the strip and report only under --verbose."""
        monkeypatch.setattr(metadata, "get_ai_metadata", lambda _path: {"c2pa": "survived"})
        out = tmp_path / "out.png"
        result = _invoke([*args, str(clean_png), "-o", str(out)])
        assert result.exit_code == 1, result.output
        assert "survived" in result.output
        assert not out.exists(), "an output that kept its AI metadata was published"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_a_staged_output_keeps_the_default_permissions(self, tmp_path, clean_png):
        """The staged sibling is not pre-created, so it is not mkstemp's 0600."""
        out = tmp_path / "out.png"
        reference = tmp_path / "reference"
        reference.write_bytes(b"")
        result = _invoke(["erase", "--region", "2,2,10,10", str(clean_png), "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert out.stat().st_mode & 0o777 == reference.stat().st_mode & 0o777

    def test_the_library_visible_strip_is_verified(self, tmp_path, clean_png, monkeypatch):
        from remove_ai_watermarks.api import MetadataStripIncomplete, remove_visible

        monkeypatch.setattr(metadata, "get_ai_metadata", lambda _path: {"c2pa": "survived"})
        with pytest.raises(MetadataStripIncomplete):
            remove_visible(clean_png, tmp_path / "out.png")
        assert list(tmp_path.iterdir()) == [clean_png], "the staged output leaked"

    def test_the_invisible_output_strip_is_verified(self, tmp_path, monkeypatch):
        """The diffusion writer used to strip without re-scanning what it wrote."""
        from PIL import Image

        from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover

        monkeypatch.setattr(metadata, "get_ai_metadata", lambda _path: {"c2pa": "survived"})
        remover = object.__new__(WatermarkRemover)  # the writer needs no model state
        with pytest.raises(RuntimeError, match="survived"):
            remover._write_output(Image.new("RGB", (16, 16)), tmp_path / "out.png")
        assert list(tmp_path.iterdir()) == [], "an output that kept its AI metadata was published"

    @pytest.mark.parametrize("command", ["erase", "remove_visible", "invisible_writer"])
    def test_an_output_the_package_just_wrote_is_not_probed_again(self, tmp_path, clean_png, monkeypatch, command):
        """The readability probe decodes every pixel; on our own encode it is pure waste."""
        from PIL import Image

        from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover
        from remove_ai_watermarks.api import remove_visible

        probed: list[Path] = []
        real = metadata.unreadable_reason
        monkeypatch.setattr(metadata, "unreadable_reason", lambda path: probed.append(path) or real(path))
        out = tmp_path / "out.png"
        if command == "erase":
            result = _invoke(["erase", "--region", "2,2,10,10", str(clean_png), "-o", str(out)])
            assert result.exit_code == 0, result.output
        elif command == "remove_visible":
            remove_visible(clean_png, out, write_noop=True)
        else:
            object.__new__(WatermarkRemover)._write_output(Image.new("RGB", (16, 16)), out)
        assert out.exists()
        assert [p for p in probed if p.name != clean_png.name] == []

    def test_metadata_remove_reports_a_missing_ffmpeg_without_a_traceback(self, tmp_path, monkeypatch):
        from remove_ai_watermarks import video_tools

        monkeypatch.setattr(shutil, "which", lambda _name: None)
        monkeypatch.setattr(video_tools, "_bundled_tools", lambda: None)
        source = tmp_path / "x.webm"
        source.write_bytes(b"\x1a\x45\xdf\xa3\x80")  # an empty EBML header tiles the file
        result = _invoke(["metadata", "--remove", str(source), "-o", str(tmp_path / "out.webm")])
        assert result.exit_code == 1, result.output
        assert isinstance(result.exception, SystemExit), result.exception
        assert "ffmpeg" in result.output

    @pytest.mark.usefixtures("cuda_refused")
    def test_invisible_reports_an_engine_that_cannot_start(self, clean_png):
        result = _invoke(["invisible", "--force", str(clean_png)])
        assert result.exit_code == 1, result.output
        assert isinstance(result.exception, SystemExit), result.exception
        assert "CUDA-only" in result.output

    @pytest.mark.usefixtures("cuda_refused")
    def test_batch_reports_an_engine_that_cannot_start(self, tmp_path, clean_png):
        indir = tmp_path / "in"
        indir.mkdir()
        shutil.copy(clean_png, indir / "clean.png")
        result = _invoke(["batch", str(indir), "--mode", "invisible", "-o", str(tmp_path / "out")])
        assert result.exit_code == 1, result.output
        assert isinstance(result.exception, SystemExit), result.exception
        assert "CUDA-only" in result.output


class TestEmptyBatch:
    """Nothing processed is the no-signal code, not success (decided 2026-10-18)."""

    def test_image_batch(self, tmp_path):
        indir = tmp_path / "in"
        indir.mkdir()
        (indir / "notes.txt").write_text("not an image")
        result = _invoke(["batch", str(indir), "-o", str(tmp_path / "out")])
        assert result.exit_code == 2, result.output
        assert not (tmp_path / "out").exists()

    def test_video_batch(self, tmp_path):
        indir = tmp_path / "in"
        indir.mkdir()
        result = _invoke(["video", "batch", str(indir), "--mode", "metadata", "-o", str(tmp_path / "out")])
        assert result.exit_code == 2, result.output
        assert not (tmp_path / "out").exists()


def test_mcp_reports_a_child_that_could_not_start():
    from remove_ai_watermarks import mcp_server

    def broken(_args):
        raise OSError("exec format error")

    result = mcp_server.call_tool("identify", {"path": "x.png"}, runner=broken)
    assert result["isError"] is True
    assert "exec format error" in result["content"][0]["text"]


def test_mcp_does_not_call_an_empty_batch_a_scanned_file():
    """An empty batch directory exits 2 like no-signal, but nothing was read."""
    import subprocess

    from remove_ai_watermarks import mcp_server

    def empty_batch(args):
        return subprocess.CompletedProcess(args, 2, stdout="No supported images found", stderr="")

    result = mcp_server.call_tool("batch", {"directory": "in"}, runner=empty_batch)
    assert result["isError"] is False
    assert "nothing to process" in result["content"][0]["text"]
