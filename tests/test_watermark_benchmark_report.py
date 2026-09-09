"""Contracts for the watermark benchmark report builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "watermark_benchmark_report.py"
SPEC = importlib.util.spec_from_file_location("watermark_benchmark_report", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
BENCHMARK = sys.modules["watermark_benchmark"]


class _FakeAdapter:
    name = "dwt-dct"
    source_file = SCRIPT

    def __init__(self, status: str) -> None:
        self.status = status

    @property
    def available(self) -> bool:
        return self.status != "unavailable"

    def detect(self, _path: Path, _image: Any) -> Any:
        if self.status == "error":
            raise RuntimeError("failed")
        return BENCHMARK.DetectorOutcome(
            status=self.status,
            label="mark" if self.status == "detected" else None,
        )


def _image(path: Path, value: int) -> Path:
    Image.fromarray(np.full((16, 20, 3), value, dtype=np.uint8), "RGB").save(path)
    return path


def _row(
    artifact: Path,
    case_id: str,
    *,
    state: str = "marked",
    status: str = "detected",
    elapsed_ms: float | None = 2.0,
    transform: str = "embed",
    expected: str = "detected",
    transform_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    digest = BENCHMARK.sha256_file(artifact)
    case = BENCHMARK.BenchmarkCase(
        case_id=case_id,
        pair_id=case_id.split("--")[0],
        media_type="image",
        adapter="dwt-dct",
        arm="positive" if state != "clean" else "matched_negative",
        state=state,
        path=artifact,
        sha256=digest,
        reference_path=artifact,
        reference_sha256=digest,
        source_revision="cohort@abc",
        transform=BENCHMARK.Transform(
            name=transform,
            revision="recipe-v1",
            parameters=transform_parameters or {},
        ),
        seed=7,
        expected=expected,
    )
    times = iter((0, round((elapsed_ms or 0.0) * 1e6)))

    def clock_ns() -> int:
        return next(times)

    return BENCHMARK.evaluate_case(
        case,
        adapters={"dwt-dct": _FakeAdapter(status)},
        repository={"commit": "abc123", "dirty": False},
        clock_ns=clock_ns,
    )


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _group(summary: dict[str, Any], section: str, **identity: str) -> dict[str, Any]:
    return next(row for row in summary[section] if all(row[key] == value for key, value in identity.items()))


def test_summary_deduplicates_artifacts_and_separates_cold_from_warm(tmp_path: Path) -> None:
    clean_path = _image(tmp_path / "clean.png", 10)
    marked_path = _image(tmp_path / "marked.png", 20)
    removed_path = _image(tmp_path / "removed.png", 30)
    clean = _row(
        clean_path,
        "clean-sdxl",
        state="clean",
        status="not_detected",
        elapsed_ms=10.0,
        transform="synthetic-carrier",
        expected="not_detected",
    )
    repeated_clean = _row(
        clean_path,
        "clean-flux",
        state="clean",
        status="not_detected",
        elapsed_ms=1.0,
        transform="synthetic-carrier",
        expected="not_detected",
    )
    marked = _row(marked_path, "marked", elapsed_ms=2.0)
    removed = _row(
        removed_path,
        "removed",
        state="removed",
        status="not_detected",
        elapsed_ms=3.0,
        transform="remove",
        expected="unresolved",
    )
    first = _write(tmp_path / "run-1.jsonl", [clean, repeated_clean, marked, removed])
    second_rows = json.loads(json.dumps([clean, repeated_clean, marked, removed]))
    for row, elapsed in zip(second_rows, (20.0, 1.5, 2.5, 3.5), strict=True):
        row["detection"]["adapter_elapsed_ms"] = elapsed
    second = _write(tmp_path / "run-2.jsonl", second_rows)

    summary = MODULE.summarize(MODULE.load_results([first, second]))

    clean_group = _group(summary, "detection", adapter="dwt-dct", state="clean")
    assert clean_group == {
        "adapter": "dwt-dct",
        "state": "clean",
        "unique_artifacts": 1,
        "observations": 4,
        "detected": 0,
        "not_detected": 1,
        "unavailable": 0,
        "error": 0,
        "unstable": 0,
    }
    cold = _group(summary, "timing", adapter="dwt-dct", phase="cold")
    assert cold == {"adapter": "dwt-dct", "phase": "cold", "n": 2, "p50_ms": 10.0, "p90_ms": 20.0, "max_ms": 20.0}
    warm = _group(summary, "timing", adapter="dwt-dct", phase="warm")
    assert warm == {"adapter": "dwt-dct", "phase": "warm", "n": 6, "p50_ms": 2.0, "p90_ms": 3.5, "max_ms": 3.5}
    removal = _group(summary, "removal", adapter="dwt-dct")
    assert removal["unique_artifacts"] == 1
    assert removal["no_recognized_signal_after_removal"] == 1
    assert summary["runs"] == 2
    assert summary["observations"] == 8


def test_summary_reports_unstable_errors_unavailable_and_mismatches(tmp_path: Path) -> None:
    changing_path = _image(tmp_path / "changing.png", 10)
    error_path = _image(tmp_path / "error.png", 20)
    unavailable_path = _image(tmp_path / "unavailable.png", 30)
    first_case = _row(changing_path, "changing", status="detected")
    second_case = _row(
        changing_path,
        "changing",
        status="not_detected",
    )
    error = _row(error_path, "error", status="error", elapsed_ms=4.0)
    unavailable = _row(unavailable_path, "unavailable", status="unavailable", elapsed_ms=None)
    first = _write(tmp_path / "run-1.jsonl", [first_case, error, unavailable])
    second = _write(tmp_path / "run-2.jsonl", [second_case])

    summary = MODULE.summarize(MODULE.load_results([first, second]))

    marked = _group(summary, "detection", adapter="dwt-dct", state="marked")
    assert marked["unstable"] == 1
    assert marked["error"] == 1
    assert marked["unavailable"] == 1
    assert summary["unstable_cases"] == [
        {"case_id": "changing", "adapter": "dwt-dct", "statuses": ["detected", "not_detected"]}
    ]
    assert summary["expected_mismatches"] == [{"case_id": "changing", "adapter": "dwt-dct", "status": "not_detected"}]


def test_summary_rejects_case_identity_drift_between_runs(tmp_path: Path) -> None:
    first_artifact = _image(tmp_path / "first.png", 10)
    second_artifact = _image(tmp_path / "second.png", 20)
    first = _write(tmp_path / "run-1.jsonl", [_row(first_artifact, "same-case")])
    second = _write(tmp_path / "run-2.jsonl", [_row(second_artifact, "same-case")])

    with pytest.raises(ValueError, match="case identity drift"):
        MODULE.summarize(MODULE.load_results([first, second]))


def test_summary_preserves_real_cohort_provider_and_content_strata(tmp_path: Path) -> None:
    first_artifact = _image(tmp_path / "first.png", 10)
    second_artifact = _image(tmp_path / "second.png", 20)
    first = _row(
        first_artifact,
        "first",
        transform_parameters={
            "scheme": "sdxl",
            "source": {"provider": "openai", "content_stratum": "portrait"},
        },
    )
    second = _row(
        second_artifact,
        "second",
        status="not_detected",
        transform_parameters={
            "scheme": "sdxl",
            "source": {"provider": "meta", "content_stratum": "portrait"},
        },
    )
    source = _write(tmp_path / "run.jsonl", [first, second])

    summary = MODULE.summarize(MODULE.load_results([source]))

    openai = _group(
        summary,
        "detection_by_source_provider",
        adapter="dwt-dct",
        state="marked",
        provider="openai",
    )
    portrait = _group(
        summary,
        "detection_by_content_stratum",
        adapter="dwt-dct",
        state="marked",
        content_stratum="portrait",
    )
    profile = _group(
        summary,
        "detection_by_profile",
        adapter="dwt-dct",
        state="marked",
        profile="sdxl",
    )
    assert openai["detected"] == 1
    assert portrait["unique_artifacts"] == 2
    assert portrait["detected"] == 1
    assert portrait["not_detected"] == 1
    assert profile["unique_artifacts"] == 2


def test_summary_reports_attack_transitions_from_each_marked_pair(tmp_path: Path) -> None:
    marked_artifact = _image(tmp_path / "marked.png", 10)
    attacked_artifact = _image(tmp_path / "attacked.png", 20)
    marked = _row(marked_artifact, "pair--marked", status="not_detected")
    attacked = _row(
        attacked_artifact,
        "pair--jpeg",
        state="attacked",
        status="detected",
        transform="embed-then-jpeg",
        expected="unresolved",
    )
    source = _write(tmp_path / "run.jsonl", [marked, attacked])

    summary = MODULE.summarize(MODULE.load_results([source]))

    transitions = _group(
        summary,
        "attack_transitions",
        adapter="dwt-dct",
        transform="embed-then-jpeg",
    )
    assert transitions == {
        "adapter": "dwt-dct",
        "transform": "embed-then-jpeg",
        "pairs": 1,
        "retained": 0,
        "lost": 0,
        "gained": 1,
        "still_not_detected": 0,
        "unmeasured": 0,
    }


def test_loader_names_invalid_json_location(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    artifact = _image(tmp_path / "valid.png", 10)
    path.write_text(json.dumps(_row(artifact, "valid")) + "\n{broken}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"broken\.jsonl:2: invalid JSON"):
        MODULE.load_results([path])


def test_summary_rejects_a_false_erasure_certificate(tmp_path: Path) -> None:
    artifact = _image(tmp_path / "removed.png", 10)
    row = _row(
        artifact,
        "removed",
        state="removed",
        status="not_detected",
        expected="unresolved",
    )
    row["removal"]["certifies_erasure"] = True
    path = _write(tmp_path / "run.jsonl", [row])

    with pytest.raises(ValueError, match="certifies_erasure must be false"):
        MODULE.summarize(MODULE.load_results([path]))


def test_report_refuses_overwrite_and_preserves_interpretation_caveat(tmp_path: Path) -> None:
    artifact = _image(tmp_path / "marked.png", 10)
    source = _write(tmp_path / "run.jsonl", [_row(artifact, "marked")])
    summary = MODULE.summarize(MODULE.load_results([source]))
    report = MODULE.render_markdown(summary, title="Test report")
    output = tmp_path / "report.md"

    MODULE.write_report(output, report)

    assert output.read_text(encoding="utf-8").startswith("# Test report\n")
    assert "not evidence that a watermark is absent" in report
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.write_report(output, report)


def test_report_accepts_audio_rows_and_reports_snr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("audio rows decode through the system ffmpeg")
    import sys

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from audioseal_experiment import wav_pcm16_bytes

    samples = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    artifact = tmp_path / "marked.wav"
    reference = tmp_path / "clean.wav"
    artifact.write_bytes(wav_pcm16_bytes(samples, 16_000))
    reference.write_bytes(wav_pcm16_bytes(samples * 0.9, 16_000))

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_id": "audio-1",
                "pair_id": "audio-1",
                "media_type": "audio",
                "adapter": "audioseal",
                "arm": "positive",
                "state": "marked",
                "path": artifact.name,
                "sha256": sha(artifact),
                "reference_path": reference.name,
                "reference_sha256": sha(reference),
                "source_revision": "fixture@1",
                "transform": {"name": "embed", "revision": "v1", "parameters": {}},
                "seed": 1,
                "expected": "detected",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    case = BENCHMARK.load_manifest(manifest)[0]
    monkeypatch.setattr(
        BENCHMARK,
        "default_adapters",
        lambda: {"audioseal": _FakeAdapter("detected")},
    )
    record = BENCHMARK.evaluate_case(
        case,
        adapters=BENCHMARK.default_adapters(),
        repository={"commit": "abc", "dirty": False},
    )

    result_path = tmp_path / "run.jsonl"
    BENCHMARK.write_jsonl(result_path, [record])

    summary = MODULE.summarize(MODULE.load_results([result_path]))
    fidelity = next(row for row in summary["fidelity"] if row["adapter"] == "audioseal")

    assert fidelity["measured"] == 1
    assert fidelity["finite_snr"] == 1
    assert fidelity["p50_snr_db"] is not None
    assert fidelity["finite_psnr"] == 0

    markdown = MODULE.render_markdown(summary)
    assert "p50 SNR dB" in markdown
