"""Contracts for the signal-discovery corpus harnesses."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import corpus_gap_scan as gap
import sidecar_regression as regression
import visible_eval
import visible_positives as visible


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


def test_gap_markers_cover_current_provenance_and_generator_inventory():
    from remove_ai_watermarks.metadata import KLING_SEI_TAG

    assert "c2pa.ai-disclosure" in gap._marker_hits(b"assertion c2pa.ai-disclosure payload")
    assert "Kling SEI tag" in gap._marker_hits(b"\x00\x00\x01\x06" + KLING_SEI_TAG)
    for token in (
        b"Grok Imagine",
        b"Amazon Bedrock",
        b"Higgsfield AI",
        b"recraft.ai",
        b"Luma AI",
        b"NovelAI",
        b"InvokeAI",
        b"Apple Image Playground",
        b"Byteplus",
    ):
        assert gap._marker_hits(token), f"expected gap._marker_hits to detect {token!r}"


def test_gap_markers_require_structural_provenance_evidence():
    noise = b"c2pa digitalSourceType Samsung Galaxy Doubao Dreamina PhotoEditor_Re_Edit Signature: " + b"A" * 80
    assert gap._marker_hits(noise) == []

    tc260 = b'{"AIGC":{"label":"1","contentProducer":"synthetic"}}'
    assert "TC260 AIGC" in gap._marker_hits(tc260)
    assert "App AIGC disclosure" in gap._marker_hits(b'{"product":"dreamina","exportType":"generation"}')
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


def test_gap_checkpoint_keeps_complete_rows_and_ignores_a_torn_tail(tmp_path):
    checkpoint = tmp_path / "report.csv.progress.jsonl"
    row = gap._base_row("2026-09-10/example.png", ".png", "1.2.3")
    checkpoint.write_text(json.dumps(row) + '\n{"path":"torn', encoding="utf-8")

    assert gap._read_checkpoint(checkpoint) == {row["path"]: row}

    gap._repair_checkpoint(checkpoint)
    second = gap._base_row("2026-09-10/second.png", ".png", "1.2.3")
    with checkpoint.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(second) + "\n")

    assert gap._read_checkpoint(checkpoint) == {row["path"]: row, second["path"]: second}


def test_visible_checkpoint_is_repaired_before_resume(tmp_path):
    checkpoint = tmp_path / "visible.jsonl"
    first = {"path": "/corpus/2026-09-10/first.png", "keys": [], "status": "ok"}
    checkpoint.write_text(json.dumps(first) + '\n{"path":"torn', encoding="utf-8")

    visible._repair_checkpoint(checkpoint)
    with checkpoint.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"path": "/corpus/2026-09-10/second.png", "keys": [], "status": "ok"}) + "\n")

    assert [row["path"] for row in visible._read_records(checkpoint)] == [
        "/corpus/2026-09-10/first.png",
        "/corpus/2026-09-10/second.png",
    ]


def test_visible_since_filter_uses_corpus_date_directories(tmp_path):
    old = tmp_path / "2026-09-01" / "old.png"
    current = tmp_path / "2026-09-10" / "current.png"
    nested = tmp_path / "2026-09-10" / "nested" / "current.webp"
    undated = tmp_path / "loose.png"
    for path in (old, current, nested, undated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert visible._files(tmp_path, date(2026, 9, 10)) == [current, nested]


def test_visible_discovery_candidates_include_unmapped_provenance_and_unbiased_quiet_rows():
    rows = [
        {"path": "known.png", "keys": ["qwen"], "status": "ok", "uscc": "unmapped-a"},
        {"path": "a.png", "keys": [], "status": "ok", "uscc": "unmapped-a"},
        {"path": "b.png", "keys": [], "status": "ok", "platform": "Google Gemini"},
        {"path": "c.png", "keys": [], "status": "ok"},
        {"path": "bad.png", "keys": [], "status": "unreadable", "uscc": "unmapped-b"},
    ]

    selected = visible._discovery_candidates(rows, per_cohort=2, random_quiet=10, seed=7)

    assert {row["path"] for row in selected} == {"a.png", "b.png", "c.png"}
    assert next(row for row in selected if row["path"] == "a.png")["discovery_strata"] == [
        "tc260:unmapped-a",
        "unbiased-quiet",
    ]


def test_visible_evaluation_inventory_follows_the_registry():
    from remove_ai_watermarks.watermark_registry import mark_keys

    assert tuple(mark_keys()) == visible_eval.MARKS


def test_gap_worker_error_is_a_complete_resumable_row(monkeypatch, tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"not an image")
    calls = []

    def fail(_path):
        calls.append(_path)
        raise ValueError("broken\ninput")

    monkeypatch.setattr(gap, "extract_provenance_evidence", fail)
    row = gap._scan_one((str(path), "bad.bin", "1.2.3"))

    assert set(row) == set(gap.REPORT_FIELDS)
    assert row["candidate_classes"] == "identify_error"
    assert row["error"] == "ValueError: broken input"
    assert calls == [path]


def test_gap_worker_records_c2pa_review_fields(monkeypatch, tmp_path):
    path = tmp_path / "signed.png"
    path.touch()
    evidence = SimpleNamespace(
        c2pa_info={
            "issuer": "Example signer",
            "claim_generator": "Example generator",
            "actions": "created",
            "source_type": "trainedAlgorithmicMedia (AI-generated)",
            "c2pa_validation_state": "Valid",
        },
        scan=b"c2pa",
    )
    monkeypatch.setattr(gap, "extract_provenance_evidence", lambda _path: evidence)
    monkeypatch.setattr(
        gap,
        "identify_from_evidence",
        lambda actual: (
            SimpleNamespace(
                is_ai_generated=True,
                platform="Example",
                confidence="high",
                watermarks=["C2PA"],
                signals=[SimpleNamespace(name="c2pa")],
                integrity_clashes=[],
            )
            if actual is evidence
            else None
        ),
    )

    row = gap._scan_one((str(path), "signed.png", "1.2.3"))

    assert row["c2pa_issuer"] == "Example signer"
    assert row["c2pa_claim_generator"] == "Example generator"
    assert row["c2pa_actions"] == "created"
    assert row["c2pa_source_type"] == "trainedAlgorithmicMedia (AI-generated)"
    assert row["c2pa_validation"] == "Valid"


def test_gap_summary_does_not_report_a_negative_hidden_count(capsys):
    row = gap._base_row("candidate.png", ".png", "1.2.3")
    row["candidate_classes"] = "blind_marker"

    gap._summarize([row])

    output = capsys.readouterr()
    assert "more candidate row" not in output.out + output.err


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


def test_legacy_watermark_strings_align_with_current_signal_names():
    sidecar = {
        "watermarks": [
            "China AIGC label (TC260 standard)",
            "IPTC AISystemUsed: Watermark Remover",
            "Open invisible watermark (DWT-DCT / RivaGAN; Stable Diffusion XL / Kandinsky 2.1)",
            "Generation parameters (prompt / seed / steps / sampler / CFG)",
            "Embedded generator tag: ByteDance Aweme AI",
            "Forensic watermark soft binding (Microsoft InvisMark; com.microsoft.invismark.1)",
        ]
    }
    report = SimpleNamespace(
        signals=[
            SimpleNamespace(name="aigc"),
            SimpleNamespace(name="iptc_ai_system"),
            SimpleNamespace(name="invisible_watermark"),
            SimpleNamespace(name="gen_params"),
            SimpleNamespace(name="exif_generator"),
            SimpleNamespace(name="soft_binding"),
            SimpleNamespace(name="invismark"),
        ],
        watermarks=sidecar["watermarks"],
    )
    assert regression.sidecar_families(sidecar) == regression.report_families(report)


def test_corpus_gap_scan_routes_video_through_identify_video_and_ignores_empty_dreamina_slot(
    tmp_path: Path,
) -> None:
    from remove_ai_watermarks.metadata import KLING_SEI_TAG

    ftyp = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    sei_mdat = (8 + len(KLING_SEI_TAG)).to_bytes(4, "big") + b"mdat" + KLING_SEI_TAG
    kling_mp4 = tmp_path / "kling.mp4"
    kling_mp4.write_bytes(ftyp + sei_mdat)

    row = gap._scan_one((str(kling_mp4), "kling.mp4", "test"))
    assert row["is_ai"] == "True"
    assert row["platform"] == "Kuaishou Kling AI"
    assert row["candidate_classes"] == ""

    # Non-AI CapCut mobile exports carry an empty ``DreaminaMetaInfo`` key slot
    # ``{"data":{}}`` without ``aigc_info``; only structural app AIGC disclosures fire.
    empty_slot = b'mdtaDreaminaMetaInfo\x00\x00\x00\x14data\x00\x00\x00\x01\x00\x00\x00\x00{"data":{}}'
    assert gap._marker_hits(empty_slot) == []
    dreamina_gen = b'{"product":"dreamina","exportType":"generation"}'
    assert "App AIGC disclosure" in gap._marker_hits(dreamina_gen)
