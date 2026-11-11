"""Check shape reconstruction, protected regions and local-error rejection."""

import cv2
import numpy as np
import pytest

from remove_ai_watermarks._internal import text_shapes as study

restore = study.restore_contour_glyphs


def scene():
    source = np.full((64, 80, 3), 244, np.uint8)
    source[18:42, 20:44] = 16
    source[23:37, 25:39] = 244
    candidate = np.full_like(source, 230)
    candidate[18:42, 24:44] = 50
    return source, candidate, [(18, 16, 46, 44)]


def test_shape_restores_hole_and_missing_stroke_using_candidate_color():
    source, candidate, boxes = scene()
    output, mask, diagnostics = restore(source, candidate, boxes)
    assert diagnostics[0]["accepted"]
    assert np.all(output[20, 21] == 50)
    assert np.all(output[30, 30] > 200)
    assert np.array_equal(output[mask == 0], candidate[mask == 0])
    assert not np.shares_memory(output, candidate)
    assert np.all(source[20, 21] == 16)


def test_complex_background_and_absent_proposals_do_not_edit_image():
    source, candidate, boxes = scene()
    checker = (np.indices(source.shape[:2]).sum(axis=0) % 2) * 64 + 180
    complex_source = np.repeat(checker[..., None], 3, axis=2).astype(np.uint8)
    complex_source[source[..., 0] == 16] = 16
    output, mask, diagnostic = restore(complex_source, candidate, boxes)
    assert not diagnostic[0]["accepted"]
    assert not np.any(mask)
    assert np.array_equal(output, candidate)
    output, mask, diagnostic = restore(source, candidate, [])
    assert np.array_equal(output, candidate)
    assert not np.any(mask)
    assert diagnostic == []


def test_subpixel_contour_reduces_diagonal_coverage_error():
    y, x = np.indices((40, 48), dtype=np.float32)
    coverage = np.clip(x + 0.5 - (0.6 * y + 12.25), 0, 1)
    contour = study._contour_coverage(coverage)
    binary = cv2.GaussianBlur((coverage >= 0.5).astype(np.float32), (3, 3), 0.45)
    assert np.sum((contour - coverage) ** 2) < np.sum((binary - coverage) ** 2)
    assert len(np.unique(contour)) > 8


def test_protected_face_intersecting_only_line_padding_skips_entire_line():
    source, candidate, boxes = scene()
    output, mask, diagnostics = study.restore_contour_glyphs(
        source, candidate, boxes, protected_boxes=((47, 16, 49, 44),)
    )
    assert np.array_equal(output, candidate)
    assert not np.any(mask)
    assert diagnostics[0]["reason"] == "protected region intersection"


def test_overlapping_group_cannot_hide_one_bad_line(monkeypatch):
    source = np.full((64, 120, 3), 100, np.uint8)
    candidate = np.zeros_like(source)
    proposal = candidate.copy()
    mask = np.zeros(source.shape[:2], np.uint8)
    boxes = ((10, 10, 45, 28), (35, 22, 60, 38), (85, 10, 105, 30))
    for a, b, c, d in boxes:
        proposal[b:d, a:c] = 100
        mask[b:d, a:c] = 255
    proposal[22:38, 35:60] = 255
    diagnostics = [{"box": box, "accepted": True} for box in boxes]
    monkeypatch.setattr(study, "_restore_glyphs", lambda *_, **_kwargs: (proposal, mask, diagnostics))
    output, accepted_mask, report = study.restore_contour_glyphs(source, candidate, boxes)
    assert np.array_equal(output[:45, :65], candidate[:45, :65])
    assert not np.any(accepted_mask[:45, :65])
    assert np.array_equal(output[10:30, 85:105], proposal[10:30, 85:105])
    assert not report[0]["accepted"]
    assert report[2]["accepted"]


def test_error_in_padded_fill_cannot_hide_behind_improved_text(monkeypatch):
    source = np.full((60, 60, 3), 100, np.uint8)
    candidate = np.zeros_like(source)
    proposal = candidate.copy()
    proposal[16:44, 16:44] = 255
    proposal[20:40, 20:40] = 100
    mask = np.zeros(source.shape[:2], np.uint8)
    mask[16:44, 16:44] = 255
    boxes = ((20, 20, 40, 40),)
    diagnostics = [{"box": boxes[0], "accepted": True}]
    monkeypatch.setattr(study, "_restore_glyphs", lambda *_, **_kwargs: (proposal, mask, diagnostics))
    output, accepted_mask, report = study.restore_contour_glyphs(source, candidate, boxes)
    assert np.array_equal(output, candidate)
    assert not np.any(accepted_mask)
    assert report[0]["quality_group"]["each_line_nonregression"] == [True]
    assert not report[0]["quality_group"]["full_support_nonregression"]


@pytest.mark.parametrize("flat_color", [16, 230])
def test_missing_candidate_contrast_is_not_invented(flat_color):
    source, candidate, boxes = scene()
    candidate[:] = flat_color
    output, mask, diagnostics = study.restore_contour_glyphs(source, candidate, boxes)
    assert np.array_equal(output, candidate)
    assert not np.any(mask)
    assert diagnostics[0]["reason"] == "candidate contrast unavailable"


@pytest.mark.parametrize(("scale", "accepted"), [(2, True), (9, True), (10, False)])
def test_large_neutral_line_eligibility_and_upper_bound(scale, accepted):
    source, candidate, boxes = scene()
    source = np.repeat(np.repeat(source, scale, axis=0), scale, axis=1)
    candidate = np.repeat(np.repeat(candidate, scale, axis=0), scale, axis=1)
    boxes = tuple(tuple(value * scale for value in box) for box in boxes)
    output, mask, diagnostics = restore(source, candidate, boxes)
    assert diagnostics[0]["accepted"] is accepted
    assert np.array_equal(output[mask == 0], candidate[mask == 0])
    if accepted:
        assert np.any(mask)
        assert np.sum((output.astype(float) - source) ** 2) < np.sum((candidate.astype(float) - source) ** 2)
    else:
        assert diagnostics[0]["reason"] == "large line"
        assert np.array_equal(output, candidate)


def textured_scene():
    y, x = np.indices((100, 160))
    bg = np.stack((35 + x * 0.8, 25 + x * 0.5, 20 + x * 0.2), axis=-1) + 8 * np.sin(y[..., None] / 4)
    source = np.rint(bg).clip(0, 255).astype(np.uint8)
    candidate = source.copy()
    for a, b, c, d in ((48, 20, 56, 75), (90, 20, 98, 75), (48, 43, 98, 51)):
        source[b:d, a:c] = (245, 245, 240)
    for a, b, c, d in ((52, 20, 56, 75), (93, 20, 101, 75), (52, 43, 101, 51)):
        candidate[b:d, a:c] = (225, 225, 220)
    return source, candidate, ((40, 10, 110, 85),)


def test_geometry_and_colors_come_from_separate_images():
    s, c, b = textured_scene()
    o, mask, d = study._restore_textured_glyphs(s, c, b)
    assert d[0]["accepted"]
    assert np.any(mask)
    assert np.array_equal(o[mask == 0], c[mask == 0])
    assert np.all(o[30, 50] == (225, 225, 220))
    assert np.all(s[30, 50] == (245, 245, 240))
    assert np.sum((o.astype(float) - s) ** 2) < np.sum((c.astype(float) - s) ** 2)


def test_protected_padded_line_is_preserved():
    s, c, b = textured_scene()
    o, mask, d = study._restore_textured_glyphs(s, c, b, protected_boxes=((37, 8, 41, 12),))
    assert np.array_equal(o, c)
    assert not np.any(mask)
    assert d[0]["reason"] == "protected region intersection"


def test_colored_foreground_is_rejected():
    s, c, b = textured_scene()
    s[np.all(s == (245, 245, 240), axis=2)] = (10, 245, 20)
    o, mask, d = study._restore_textured_glyphs(s, c, b)
    assert np.array_equal(o, c)
    assert not np.any(mask)
    assert not d[0]["accepted"]


def test_perfect_donor_cannot_be_degraded():
    s, _, b = textured_scene()
    o, mask, d = study._restore_textured_glyphs(s, s.copy(), b)
    assert np.array_equal(o, s)
    assert not np.any(mask)
    assert d[0]["reason"] == "group fidelity regression"


def test_texture_overlap_cannot_degrade_an_accepted_flat_line(monkeypatch):
    source = np.full((64, 80, 3), 100, np.uint8)
    candidate = np.zeros_like(source)
    boxes = ((10, 10, 45, 28), (35, 22, 60, 38))
    flat = candidate.copy()
    flat[10:28, 10:45] = 100
    flat_mask = np.zeros(source.shape[:2], np.uint8)
    flat_mask[10:28, 10:45] = 255
    proposal = flat.copy()
    proposal[22:38, 35:60] = 100
    proposal[22:28, 35:45] = 255
    texture_mask = np.zeros_like(flat_mask)
    texture_mask[22:38, 35:60] = 255
    flat_diagnostics = [{"box": boxes[0], "accepted": True}, {"box": boxes[1], "accepted": False}]
    texture_diagnostics = [{"box": boxes[0], "accepted": False}, {"box": boxes[1], "accepted": True}]
    monkeypatch.setattr(study, "restore_contour_glyphs", lambda *_, **__: (flat, flat_mask, flat_diagnostics))
    monkeypatch.setattr(
        study, "_restore_textured_glyphs", lambda *_, **__: (proposal, texture_mask, texture_diagnostics)
    )
    output, mask, diagnostics = study.restore_textured_glyphs(source, candidate, boxes)
    assert np.array_equal(output, flat)
    assert np.array_equal(mask, flat_mask)
    assert diagnostics[0]["accepted"]
    assert diagnostics[1]["reason"] == "combined fidelity regression"


def test_composed_textured_method_applies_the_spatial_stage():
    source, candidate, boxes = textured_scene()
    output, mask, diagnostics = study.restore_textured_glyphs(source, candidate, boxes)
    assert diagnostics[0]["accepted"]
    assert diagnostics[0]["combined_quality_group"]["accepted"]
    assert np.all(output[30, 50] == (225, 225, 220))
    assert np.array_equal(output[mask == 0], candidate[mask == 0])
    assert np.sum((output.astype(float) - source) ** 2) < np.sum((candidate.astype(float) - source) ** 2)


def test_chromatic_foreground_is_opt_in_and_preserves_support_boundaries():
    source, candidate, boxes = scene()
    source[source[..., 0] == 16] = (190, 20, 20)
    candidate[candidate[..., 0] == 50] = (170, 30, 30)
    unchanged, _mask, diagnostics = restore(source, candidate, boxes)
    assert not any(d["accepted"] for d in diagnostics)
    assert any(d.get("color_eligible", False) for d in diagnostics)
    assert np.array_equal(unchanged, candidate)
    output, mask, diagnostics = restore(source, candidate, boxes, allow_colored=True)
    assert any(d["accepted"] for d in diagnostics)
    assert np.array_equal(output[mask == 0], candidate[mask == 0])
    assert np.mean((source.astype(float) - output.astype(float)) ** 2) < np.mean(
        (source.astype(float) - candidate.astype(float)) ** 2
    )
