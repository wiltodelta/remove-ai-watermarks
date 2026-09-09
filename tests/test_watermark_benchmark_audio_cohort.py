"""Contracts for the audio benchmark cohort builder."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str, filename: str) -> object:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COHORT = _load("watermark_benchmark_audio_cohort", "watermark_benchmark_audio_cohort.py")
EXPERIMENT = _load("audioseal_experiment_cohort_test", "audioseal_experiment.py")
BENCHMARK = sys.modules.get("watermark_benchmark") or _load("watermark_benchmark", "watermark_benchmark.py")
sys.path.insert(0, str(SCRIPTS))
import audioseal_oracle  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not audioseal_oracle.available(),
    reason="audio cohort builder requires ffmpeg and the dev extra (audioseal, torch)",
)


def test_builder_refuses_existing_output_directory(tmp_path: Path) -> None:
    (tmp_path / "cohort").mkdir()

    with pytest.raises(FileExistsError):
        COHORT.build_cohort(tmp_path / "cohort", duration_s=0.2)


def test_builder_rejects_unknown_carriers_and_attacks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown carriers"):
        COHORT.build_cohort(tmp_path / "a", carriers=("silence",), duration_s=0.2)
    with pytest.raises(ValueError, match="unknown attacks"):
        COHORT.build_cohort(tmp_path / "b", attacks=("reverb",), duration_s=0.2)


def test_built_cohort_manifest_is_strict_and_complete(tmp_path: Path) -> None:
    manifest = COHORT.build_cohort(tmp_path / "cohort", duration_s=0.5)

    cases = BENCHMARK.load_manifest(manifest)
    assert len(cases) == 20
    by_case = {case.case_id: case for case in cases}
    assert by_case["tone_stack-removed"].state == "removed"
    assert by_case["tone_stack-removed"].expected == "not_detected"
    assert by_case["tone_stack-forged"].state == "forged"
    assert by_case["tone_stack-forged"].arm == "wrong_key"
    assert by_case["tone_stack-forged"].expected == "detected"
    assert by_case["tone_stack-clean"].arm == "matched_negative"
    assert by_case["tone_stack-clean"].expected == "not_detected"
    assert by_case["tone_stack-marked"].arm == "positive"
    assert by_case["tone_stack-mp3_128k"].state == "attacked"
    assert by_case["tone_stack-mp3_128k"].expected == "unresolved"
    assert by_case["white_noise_hard_negative-clean"].arm == "hard_negative"
    assert all(case.media_type == "audio" for case in cases)
    assert all(case.adapter == "audioseal" for case in cases)
    assert all(case.path.is_file() for case in cases)
    marked = by_case["tone_stack-marked"]
    assert marked.reference_path == by_case["tone_stack-clean"].path

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    embed_parameters = next(row for row in rows if row["case_id"] == "tone_stack-marked")
    assert embed_parameters["transform"]["parameters"]["message"] == EXPERIMENT.message_bits()


def test_carrier_artifacts_are_deterministic_across_rebuilds(tmp_path: Path) -> None:
    manifest = COHORT.build_cohort(tmp_path / "cohort", duration_s=0.5)

    cases = {case.case_id: case for case in BENCHMARK.load_manifest(manifest)}
    first = (tmp_path / "cohort" / "artifacts" / "tone_stack-clean.wav").read_bytes()
    second = EXPERIMENT.wav_pcm16_bytes(EXPERIMENT.synth_carrier("tone_stack", 0.5), 16_000)
    assert first == second
    rebuilt = COHORT.build_cohort(tmp_path / "cohort-2", duration_s=0.5)
    cases_2 = {case.case_id: case for case in BENCHMARK.load_manifest(rebuilt)}
    for case_id, case in cases.items():
        if case.state == "clean":
            assert case.sha256 == cases_2[case_id].sha256, case_id


def test_hard_negative_is_distinct_from_every_marked_carrier(tmp_path: Path) -> None:
    hard = COHORT.synth_carrier_hard_negative(0.5)
    carrier = EXPERIMENT.synth_carrier("white_noise", 0.5)

    assert hard.shape == carrier.shape
    assert not np.array_equal(hard, carrier)


@pytest.mark.skipif(shutil.which("say") is None, reason="speech carriers require the macOS say tool")
def test_speech_extension_adds_provenance_backed_rows(tmp_path: Path) -> None:
    manifest = COHORT.build_cohort(tmp_path / "cohort", duration_s=0.5, include_speech=True)

    cases = BENCHMARK.load_manifest(manifest)
    assert len(cases) == 35
    speech = {case.case_id: case for case in cases if case.pair_id.startswith("speech_")}
    assert len(speech) == 15
    assert {case.pair_id for case in speech.values()} == {
        "speech_Samantha",
        "speech_Daniel",
        "speech_Milena",
    }
    marked = next(case for case in speech.values() if case.state == "marked")
    assert marked.reference_path is not None
    digests = {case.sha256 for case in speech.values() if case.state == "clean"}
    assert len(digests) == 3, "aliased system voices would collapse clean-carrier digests"

    rows = {json.loads(line)["case_id"]: json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()}
    tts = rows["speech_Samantha-clean"]["transform"]["parameters"]["tts"]
    assert tts["tts"] == "say"
    assert tts["macos"]
    assert tts["voice"] == "Samantha"
    assert tts["text_sha256"]
