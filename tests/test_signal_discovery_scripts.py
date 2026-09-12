"""Contracts for the signal-discovery corpus harnesses."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import corpus_gap_scan as gap
import sidecar_regression as regression


def _report(**overrides):
    values = {
        "is_ai_generated": None,
        "platform": None,
        "signals": [],
        "integrity_clashes": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_gap_markers_are_case_insensitive():
    assert "ComfyUI" in gap._marker_hits(b"COMFYUI WORKFLOW")


def test_gap_markers_require_structural_provenance_evidence():
    noise = b"c2pa digitalSourceType Samsung Galaxy Doubao PhotoEditor_Re_Edit Signature: " + b"A" * 80
    assert gap._marker_hits(noise) == []

    tc260 = b'{"AIGC":{"label":"1","contentProducer":"synthetic"}}'
    assert "TC260 AIGC" in gap._marker_hits(tc260)
    assert "Samsung genAIType" in gap._marker_hits(b'PhotoEditor_Re_Edit_Data{"genAIType":1}')


def test_gap_candidates_cover_more_than_blind_unknowns():
    signal = SimpleNamespace(name="iptc")
    assert gap._candidate_classes(_report(signals=[signal]), []) == ["unattributed_signal"]
    assert gap._candidate_classes(_report(is_ai_generated=True), []) == ["unattributed_ai"]
    assert gap._candidate_classes(_report(), ["SynthID"]) == ["blind_marker"]


def test_since_filter_uses_corpus_date_directories(tmp_path):
    old = tmp_path / "2026-09-01" / "old.png"
    current = tmp_path / "2026-09-10" / "current.png"
    undated = tmp_path / "loose.png"
    for path in (old, current, undated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert gap._files(tmp_path, date(2026, 9, 10)) == [current]


def test_sidecar_regression_prefers_stable_signal_names():
    sidecar = {"signals": ["visible_qwen"], "watermarks": ["wording may change"]}
    assert regression.sidecar_families(sidecar) == {"visible_qwen"}


def test_current_report_uses_signal_names_not_watermark_prose():
    report = SimpleNamespace(
        signals=[SimpleNamespace(name="visible_liblib")],
        watermarks=["new wording not known to the legacy mapper"],
    )
    assert regression.report_families(report) == {"visible_liblib"}


def test_current_report_keeps_watermark_only_synthid_family():
    report = SimpleNamespace(signals=[], watermarks=["SynthID watermark (Google)"])
    assert regression.report_families(report) == {"synthid"}


def test_registry_drives_current_visible_families():
    from remove_ai_watermarks.watermark_registry import known_marks

    for mark in known_marks():
        expected = "visible_sparkle" if mark.key == "gemini" else f"visible_{mark.key}"
        assert regression.family_of(mark.label) == expected


def test_registry_keys_cover_legacy_visible_wording():
    assert regression.family_of("Qwen 千问 AI label") == "visible_qwen"
    assert regression.family_of("Microsoft AI badge") == "visible_microsoft"
