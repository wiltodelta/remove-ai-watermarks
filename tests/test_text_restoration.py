"""Verified-text manifest and compositor tests without model downloads."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from remove_ai_watermarks._internal.text_restoration import (
    VerifiedTextLine,
    load_verified_text_manifest,
    restore_verified_text,
    source_pixel_sha256,
    source_silhouette_mask,
)


def _manifest(image: Image.Image) -> dict[str, object]:
    return {
        "schema_version": 1,
        "verified": True,
        "source_pixel_sha256": source_pixel_sha256(image),
        "width": image.width,
        "height": image.height,
        "lines": [
            {
                "box": [8, 8, 40, 24],
                "text": "Exact text",
                "script": "alphabetic",
                "angle": 0.0,
            }
        ],
    }


def _geometry_manifest(image: Image.Image) -> dict[str, object]:
    return {
        "schema_version": 2,
        "verified": True,
        "source_pixel_sha256": source_pixel_sha256(image),
        "width": image.width,
        "height": image.height,
        "lines": [{"box": [8, 8, 40, 24], "angle": 0.0}],
    }


def test_pixel_hash_ignores_container_metadata(tmp_path) -> None:
    image = Image.new("RGB", (48, 32), (10, 20, 30))
    plain = tmp_path / "plain.png"
    tagged = tmp_path / "tagged.png"
    image.save(plain)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("note", "different container bytes")
    image.save(tagged, pnginfo=metadata)

    with Image.open(plain) as left, Image.open(tagged) as right:
        assert plain.read_bytes() != tagged.read_bytes()
        assert source_pixel_sha256(left) == source_pixel_sha256(right)


def test_verified_manifest_is_bound_to_source_pixels(tmp_path) -> None:
    source = Image.new("RGB", (48, 32), (10, 20, 30))
    path = tmp_path / "lines.json"
    path.write_text(json.dumps(_manifest(source)), encoding="utf-8")

    loaded = load_verified_text_manifest(path, source)

    assert loaded.width == 48
    assert loaded.height == 32
    assert loaded.lines == (VerifiedTextLine((8, 8, 40, 24), "Exact text", "alphabetic", 0.0),)


def test_geometry_manifest_needs_no_transcription_or_script(tmp_path) -> None:
    source = Image.new("RGB", (48, 32), (10, 20, 30))
    path = tmp_path / "regions.json"
    path.write_text(json.dumps(_geometry_manifest(source)), encoding="utf-8")

    loaded = load_verified_text_manifest(path, source)

    assert loaded.lines == (VerifiedTextLine((8, 8, 40, 24), angle=0.0),)


@pytest.mark.parametrize("field", ["text", "script"])
def test_schema_one_still_requires_text_metadata(tmp_path, field) -> None:
    source = Image.new("RGB", (48, 32), (10, 20, 30))
    payload = _manifest(source)
    del payload["lines"][0][field]
    path = tmp_path / "lines.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_verified_text_manifest(path, source)


@pytest.mark.parametrize("schema_version", [True, 0, 3])
def test_manifest_rejects_unsupported_schema_versions(tmp_path, schema_version) -> None:
    source = Image.new("RGB", (48, 32), (10, 20, 30))
    payload = _geometry_manifest(source)
    payload["schema_version"] = schema_version
    path = tmp_path / "lines.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported text manifest schema"):
        load_verified_text_manifest(path, source)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"verified": False}, "verified=true"),
        ({"source_pixel_sha256": "0" * 64}, "does not match"),
        ({"width": 49}, "dimensions"),
        ({"lines": []}, "non-empty"),
    ],
)
def test_manifest_rejects_unverified_or_unbound_input(tmp_path, mutation, message) -> None:
    source = Image.new("RGB", (48, 32), (10, 20, 30))
    payload = _manifest(source)
    payload.update(mutation)
    path = tmp_path / "lines.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_verified_text_manifest(path, source)


def test_restoration_uses_lama_and_qwen_vae_core(monkeypatch) -> None:
    from remove_ai_watermarks import region_eraser

    source = np.full((40, 64, 3), 20, dtype=np.uint8)
    source[12:24, 12:44] = 235
    candidate = np.full_like(source, 30)
    candidate[12:24, 12:44] = 150
    donor = np.full_like(source, 40)
    donor[12:24, 12:44] = (210, 220, 230)
    calls: list[np.ndarray] = []

    def fake_erase(image_bgr, mask):
        calls.append(mask.copy())
        output = image_bgr.copy()
        output[mask > 0] = (30, 30, 30)
        return output

    monkeypatch.setattr(region_eraser, "lama_available", lambda: True)
    monkeypatch.setattr(region_eraser, "erase_lama", fake_erase)

    result = restore_verified_text(
        Image.fromarray(source),
        Image.fromarray(candidate),
        Image.fromarray(donor),
        (VerifiedTextLine((8, 8, 48, 28)),),
    )

    restored = np.asarray(result)
    assert calls
    assert np.all(restored[16, 20] == donor[16, 20])
    assert np.all(restored[0, 0] == candidate[0, 0])


def test_silhouette_includes_descender_below_the_detector_box() -> None:
    source = np.full((40, 50, 3), 240, dtype=np.uint8)
    source[10:22, 18:24] = 20
    source[22:27, 18:22] = 20  # tail of a y / Cyrillic u, under the box

    mask = source_silhouette_mask(source, (10, 10, 40, 22))

    assert mask[24, 20] == 255
    assert mask[16, 20] == 255


def test_silhouette_includes_glyph_edges_beside_the_detector_box() -> None:
    source = np.full((40, 70, 3), 240, dtype=np.uint8)
    source[10:22, 20:50] = 20
    source[14:18, 10:20] = 20  # leading flourish, left of the detector box
    source[14:18, 50:58] = 20  # punctuation or icon edge, right of the box
    source[14:18, 2:5] = 20  # separate decoration must stay outside the crop

    mask = source_silhouette_mask(source, (20, 10, 50, 22))

    assert mask[16, 12] == 255
    assert mask[16, 56] == 255
    assert mask[16, 3] == 0
