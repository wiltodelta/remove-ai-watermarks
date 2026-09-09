"""Contracts for the video benchmark cohort builder."""

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


COHORT = _load("watermark_benchmark_video_cohort", "watermark_benchmark_video_cohort.py")
BENCHMARK = sys.modules.get("watermark_benchmark") or _load("watermark_benchmark", "watermark_benchmark.py")
sys.path.insert(0, str(SCRIPTS))
import videoseal_oracle  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not videoseal_oracle.available(),
    reason="video cohort builder requires ffmpeg and the dev extra (torch)",
)


def test_builder_refuses_existing_output_directory(tmp_path: Path) -> None:
    (tmp_path / "cohort").mkdir()

    with pytest.raises(FileExistsError):
        COHORT.build_cohort(tmp_path / "cohort")


def test_builder_rejects_unknown_carriers_and_attacks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown carriers"):
        COHORT.build_cohort(tmp_path / "a", carriers=("black",))
    with pytest.raises(ValueError, match="unknown attacks"):
        COHORT.build_cohort(tmp_path / "b", attacks=("rotate",))


def test_carriers_are_deterministic_and_distinct() -> None:
    first = COHORT.synth_carrier("moving_gradient")
    second = COHORT.synth_carrier("moving_gradient")
    texture = COHORT.synth_carrier("moving_texture")
    hard = COHORT.synth_carrier("moving_texture", seed_offset=991)

    assert first.shape == (COHORT.FRAME_COUNT, COHORT.HEIGHT, COHORT.WIDTH, 3)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, texture)
    assert not np.array_equal(texture, hard)


def test_built_cohort_manifest_is_strict_and_complete(tmp_path: Path) -> None:
    manifest = COHORT.build_cohort(tmp_path / "cohort")

    cases = BENCHMARK.load_manifest(manifest)
    assert len(cases) == 15
    by_case = {case.case_id: case for case in cases}
    assert by_case["moving_gradient-removed"].state == "removed"
    assert by_case["moving_gradient-removed"].expected == "not_detected"
    assert by_case["moving_gradient-forged"].state == "forged"
    assert by_case["moving_gradient-forged"].arm == "wrong_key"
    assert by_case["moving_gradient-forged"].expected == "not_detected"
    assert by_case["moving_gradient-clean"].arm == "matched_negative"
    assert by_case["moving_gradient-marked"].arm == "positive"
    assert by_case["moving_gradient-marked"].reference_path == by_case["moving_gradient-clean"].path
    assert by_case["moving_gradient-h264_crf23"].state == "attacked"
    assert by_case["moving_gradient-h264_crf23"].expected == "unresolved"
    assert by_case["moving_texture_hard_negative-clean"].arm == "hard_negative"
    assert all(case.media_type == "video" for case in cases)
    assert all(case.adapter == "videoseal" for case in cases)
    assert all(case.path.is_file() for case in cases)

    rows = {json.loads(line)["case_id"]: json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()}
    marked = rows["moving_gradient-marked"]["transform"]["parameters"]
    assert marked["frames"] == COHORT.FRAME_COUNT
    assert marked["message_sha256"]
    assert rows["moving_gradient-h264_crf23"]["transform"]["parameters"]["crf"] == 23


def test_scale_and_fps_attacks_carry_no_reference(tmp_path: Path) -> None:
    manifest = COHORT.build_cohort(tmp_path / "cohort")

    cases = {case.case_id: case for case in BENCHMARK.load_manifest(manifest)}
    assert cases["moving_gradient-h264_crf23"].reference_path is not None
    assert cases["moving_gradient-scale_075"].reference_path is None
    assert cases["moving_gradient-fps_half"].reference_path is None
