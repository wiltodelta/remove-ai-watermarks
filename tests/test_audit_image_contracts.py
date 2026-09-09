"""Regression coverage for image output lifetimes and display fidelity."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageCms, ImageOps

from remove_ai_watermarks import api, image_io, invisible_engine, watermark_registry
from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover


def _source(path, orientation=1, alpha=False, size=(96, 64)):
    width, height = size
    pixels = np.zeros((height, width, 4 if alpha else 3), dtype=np.uint8)
    pixels[:, : width // 3, :3] = (240, 20, 30)
    pixels[: height // 2, width // 3 :, :3] = (10, 200, 40)
    if alpha:
        pixels[:, :, 3] = np.arange(width, dtype=np.uint8)[None, :] + 2 * np.arange(height, dtype=np.uint8)[:, None]
    image = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[274] = orientation
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    image.save(path, exif=exif, icc_profile=icc)
    with Image.open(path) as reopened:
        upright = np.array(ImageOps.exif_transpose(reopened))
    return icc, upright


def _engine(run=None):
    engine = object.__new__(invisible_engine.InvisibleEngine)
    engine._progress_callback = None
    remover = object.__new__(WatermarkRemover)
    remover.model_profile = "qwen-zimage"
    remover._auto = False
    remover._qwen_zimage_pipeline = SimpleNamespace(run=run or (lambda source, **kwargs: source))
    engine._remover = remover
    return engine


def _no_visible(monkeypatch):
    monkeypatch.setattr(
        watermark_registry,
        "remove_auto_marks_detailed",
        lambda image, **kwargs: watermark_registry.VisibleRemovalResult(image, ()),
    )


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("in_place", [False, True])
def test_invisible_batch_passthrough_uses_current_source(tmp_path, monkeypatch, available, in_place):
    source_dir = tmp_path / "in"
    source_dir.mkdir()
    out_dir = source_dir if in_place else tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    source = source_dir / "image.png"
    icc, expected = _source(source, orientation=3)
    original_bytes = source.read_bytes()
    destination = out_dir / source.name
    if not in_place:
        Image.new("RGB", (96, 64), "blue").save(destination)
    monkeypatch.setattr(invisible_engine, "is_available", lambda: available)
    monkeypatch.setattr(api._SourceEvidence, "has_invisible_target", lambda self: False)

    summary = api.remove_batch(source_dir, out_dir, mode="invisible")

    assert summary.processed == 1
    assert summary.failed == 0
    assert summary.items[0].invisible == ("no-signal" if available else "unavailable")
    if in_place:
        assert source.read_bytes() == original_bytes
    with Image.open(destination) as result:
        np.testing.assert_array_equal(np.array(ImageOps.exif_transpose(result)), expected)
        assert result.info["icc_profile"] == icc


@pytest.mark.parametrize("mode", ["visible", "all"])
@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 6, 8])
@pytest.mark.parametrize("run_invisible", [False, True])
def test_batch_display_tags_and_alpha_survive_repeated_in_place_writes(
    tmp_path, monkeypatch, mode, orientation, run_invisible
):
    source = tmp_path / "image.png"
    icc, expected = _source(source, orientation=orientation, alpha=True)
    _no_visible(monkeypatch)
    monkeypatch.setattr(invisible_engine, "is_available", lambda: True)
    calls = []

    def run(image, **kwargs):
        calls.append(image.size)
        return image

    engine = _engine(run)
    for _ in range(2):
        summary = api.remove_batch(tmp_path, tmp_path, mode=mode, force=run_invisible, engine=engine)
        assert summary.failed == 0, summary.errors
        with Image.open(source) as result:
            assert result.info.get("icc_profile") == icc
            np.testing.assert_array_equal(np.array(ImageOps.exif_transpose(result)), expected)
    assert len(calls) == (2 if mode == "all" and run_invisible else 0)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("polish", [False, True])
def test_invisible_engine_overwrites_original_and_retains_display(tmp_path, explicit, polish):
    source = tmp_path / "source.png"
    icc, expected = _source(source, orientation=6)
    engine = _engine()
    output = engine.remove_watermark(source, source if explicit else None, adaptive_polish=polish)
    assert output == source
    assert output.exists()
    with Image.open(output) as result:
        assert result.info.get("icc_profile") == icc
        assert result.getexif().get(274, 1) == 1
        assert result.size == (64, 96)
        if not polish:
            np.testing.assert_array_equal(np.array(result), expected)


@pytest.mark.parametrize(("tile", "expected"), [(False, (512, 256)), (True, (2048, 1024))])
def test_tiling_bypasses_resolution_cap(tmp_path, tile, expected):
    source = tmp_path / "source.png"
    _source(source, size=(2048, 1024))
    seen = []

    def run(image, **kwargs):
        seen.append((image.size, kwargs["tile"]))
        return image

    _engine(run).remove_watermark(source, tmp_path / "out.png", tile=tile, max_resolution=512)
    assert seen == [(expected, tile)]


def test_invisible_postprocessing_write_failure_raises(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    _source(source)
    original_writer = image_io.imwrite

    def write(path, pixels, **kwargs):
        # The model's write succeeds, the final public postprocessing write fails.
        if isinstance(path, Path):
            return False
        return original_writer(path, pixels, **kwargs)

    monkeypatch.setattr(image_io, "imwrite", write)
    with pytest.raises(OSError, match="write"):
        _engine().remove_watermark(source, tmp_path / "out.png", adaptive_polish=True)


@pytest.mark.parametrize("orientation", [2, 3, 4, 5, 6, 7, 8])
def test_explicit_upright_writer_does_not_reapply_square_image_orientation(tmp_path, orientation):
    source = tmp_path / "source.png"
    icc, upright = _source(source, orientation=orientation, size=(64, 64))
    assert image_io.imwrite(source, upright[:, :, ::-1], display_tags_from=source, orientation_applied=True)
    with Image.open(source) as result:
        assert result.info.get("icc_profile") == icc
        np.testing.assert_array_equal(np.array(ImageOps.exif_transpose(result)), upright)


@pytest.mark.parametrize("failure", [None, "save", "model"])
def test_invisible_temporary_input_is_cleaned_after_success_and_failure(tmp_path, monkeypatch, failure):
    import tempfile

    source = tmp_path / "source.png"
    _source(source)
    original_bytes = source.read_bytes()
    temporary_paths = []
    original_mkstemp = tempfile.mkstemp

    def make_temp(**kwargs):
        fd, name = original_mkstemp(dir=tmp_path, **kwargs)
        temporary_paths.append(Path(name))
        return fd, name

    monkeypatch.setattr(tempfile, "mkstemp", make_temp)
    original_save = Image.Image.save

    def save(image, fp, **kwargs):
        if failure == "save" and Path(fp) in temporary_paths:
            raise OSError("synthetic temp save failure")
        return original_save(image, fp, **kwargs)

    monkeypatch.setattr(Image.Image, "save", save)

    def run(image, **kwargs):
        if failure == "model":
            raise OSError("synthetic model failure")
        return image

    engine = _engine(run)
    destination = tmp_path / "out.png"
    if failure:
        with pytest.raises(OSError, match="synthetic"):
            engine.remove_watermark(source, destination)
        assert not destination.exists()
    else:
        assert engine.remove_watermark(source, destination) == destination
        assert destination.exists()
    assert source.read_bytes() == original_bytes
    assert len(temporary_paths) == 1
    assert not temporary_paths[0].exists()


def test_invisible_batch_preserves_new_model_result(tmp_path, monkeypatch):
    source_dir = tmp_path / "in"
    source_dir.mkdir()
    source = source_dir / "image.png"
    _source(source)
    monkeypatch.setattr(invisible_engine, "is_available", lambda: True)
    engine = _engine(lambda image, **kwargs: Image.new("RGB", image.size, (20, 30, 220)))
    summary = api.remove_batch(source_dir, tmp_path / "out", mode="invisible", force=True, engine=engine)
    assert summary.failed == 0
    assert summary.items[0].invisible == "removed"
    with Image.open(tmp_path / "out" / source.name) as output:
        assert output.getpixel((0, 0)) == (20, 30, 220)


@pytest.mark.parametrize("orientation", [3, 6])
def test_single_visible_format_conversion_preserves_display_tags(tmp_path, monkeypatch, orientation):
    source = tmp_path / "source.png"
    icc, expected = _source(source, orientation=orientation, size=(64, 64))
    _no_visible(monkeypatch)
    destination = tmp_path / "out.webp"
    api.remove_visible(source, destination, strip_metadata=False)
    with Image.open(destination) as result:
        assert result.info.get("icc_profile") == icc
        np.testing.assert_array_equal(np.array(ImageOps.exif_transpose(result)), expected)
