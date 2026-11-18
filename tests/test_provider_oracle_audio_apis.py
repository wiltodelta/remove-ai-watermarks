"""Contract tests for ElevenLabs and Cartesia provider-oracle API adapters."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import httpx
import provider_oracle_audio_apis as audio_apis
import provider_oracles as oracles
import pytest
from click.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path


def _write_minimal_mp3(path: Path) -> Path:
    # Minimal valid 1-frame MPEG-1 Layer III silence frame so ffprobe/metadata paths accept it when needed.
    frame = bytes([0xFF, 0xFB, 0x90, 0x64]) + b"\x00" * 413
    path.write_bytes(frame * 8)
    return path


def test_elevenlabs_requires_explicit_acknowledgement(tmp_path: Path) -> None:
    audio_file = _write_minimal_mp3(tmp_path / "sample.mp3")
    with pytest.raises(ValueError, match="acknowledge_upload=True"):
        audio_apis.verify_elevenlabs_provenance(audio_file, acknowledge_upload=False, api_key="el-secret-123")


def test_elevenlabs_parses_detected_and_not_detected_and_redacts_secret_in_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file = _write_minimal_mp3(tmp_path / "sample.mp3")
    monkeypatch.setattr(
        audio_apis,
        "_sanitize_audio_upload",
        lambda src, dst: dst.write_bytes(src.read_bytes()) or True,
    )

    seen_headers: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        assert str(request.url) == "https://api.elevenlabs.io/v1/content-provenance/watermark"
        return httpx.Response(
            200,
            json={
                "is_detected": True,
                "details": {
                    "product": "tts",
                    "model": "Eleven Multilingual v2",
                    "generated_at": "2026-11-09T17:29:30.719673+00:00",
                    "unique_id": "FV6ptUbRJ5044rpWOOkt",
                },
            },
        )

    transport = httpx.MockTransport(handler)
    with caplog.at_level(logging.INFO):
        res = audio_apis.verify_elevenlabs_provenance(
            audio_file,
            acknowledge_upload=True,
            api_key="el-super-secret-key",
            transport=transport,
        )

    assert res.detected is True
    assert res.status == "detected"
    assert res.model == "Eleven Multilingual v2"
    assert res.product == "tts"
    assert res.unique_id == "FV6ptUbRJ5044rpWOOkt"
    assert res.ai_metadata_stripped is True
    assert res.samples_preserved is True
    assert seen_headers[0]["xi-api-key"] == "el-super-secret-key"
    assert "el-super-secret-key" not in caplog.text
    assert "Eleven Multilingual v2" in caplog.text

    def handler_neg(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"is_detected": False, "details": None})

    res_neg = audio_apis.verify_elevenlabs_provenance(
        audio_file,
        acknowledge_upload=True,
        api_key="el-super-secret-key",
        transport=httpx.MockTransport(handler_neg),
    )
    assert res_neg.detected is False
    assert res_neg.status == "not_detected"
    assert res_neg.model is None
    assert res_neg.unique_id is None


def test_elevenlabs_http_failure_never_leaks_api_key(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file = _write_minimal_mp3(tmp_path / "sample.mp3")
    monkeypatch.setattr(
        audio_apis,
        "_sanitize_audio_upload",
        lambda src, dst: dst.write_bytes(src.read_bytes()) or True,
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text='{"detail":{"status":"invalid_api_key","key":"el-leak-me"}}')

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(audio_apis.AudioProvenanceApiError) as exc_info,
    ):
        audio_apis.verify_elevenlabs_provenance(
            audio_file,
            acknowledge_upload=True,
            api_key="el-leak-me",
            transport=httpx.MockTransport(handler),
        )
    assert exc_info.value.status_code == 401
    assert "el-leak-me" not in str(exc_info.value)
    assert "el-leak-me" not in caplog.text


def test_cartesia_parses_detected_and_not_detected_with_version_header(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file = _write_minimal_mp3(tmp_path / "sonic.mp3")
    monkeypatch.setattr(
        audio_apis,
        "_sanitize_audio_upload",
        lambda src, dst: dst.write_bytes(src.read_bytes()) or True,
    )
    seen_headers: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        assert str(request.url) == "https://api.cartesia.ai/v1/watermark/detect"
        return httpx.Response(200, json={"detected": False})

    with caplog.at_level(logging.INFO):
        res = audio_apis.verify_cartesia_watermark(
            audio_file,
            acknowledge_upload=True,
            api_key="cartesia-secret-token",
            transport=httpx.MockTransport(handler),
        )

    assert res.detected is False
    assert res.status == "not_detected"
    assert seen_headers[0]["x-api-key"] == "cartesia-secret-token"
    assert seen_headers[0]["cartesia-version"] == audio_apis.CARTESIA_API_VERSION
    assert "cartesia-secret-token" not in caplog.text


def test_cli_check_elevenlabs_and_check_cartesia(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file = _write_minimal_mp3(tmp_path / "clip.mp3")
    env_file = tmp_path / ".env"
    env_file.write_text("ELEVENLABS_API_KEY=el-cli-key\nCARTESIA_API_KEY=ca-cli-key\n", encoding="utf-8")

    monkeypatch.setattr(
        oracles,
        "verify_elevenlabs",
        lambda path, *, acknowledge_upload, api_key, strip_metadata: audio_apis.ElevenLabsProvenanceDetection(
            status="not_detected",
            product=None,
            model=None,
            generated_at=None,
            unique_id=None,
            raw_response={"is_detected": False, "details": None},
            ai_metadata_stripped=strip_metadata,
            samples_preserved=True,
        ),
    )
    monkeypatch.setattr(
        oracles,
        "verify_cartesia",
        lambda path, *, acknowledge_upload, api_key, strip_metadata: audio_apis.CartesiaWatermarkDetection(
            status="detected",
            raw_response={"detected": True},
            ai_metadata_stripped=strip_metadata,
            samples_preserved=True,
        ),
    )

    runner = CliRunner()
    missing_ack = runner.invoke(oracles.cli, ["check-elevenlabs", str(audio_file), "--env-file", str(env_file)])
    assert missing_ack.exit_code != 0
    assert "--acknowledge-upload" in missing_ack.output

    el_ok = runner.invoke(
        oracles.cli,
        ["check-elevenlabs", str(audio_file), "--acknowledge-upload", "--env-file", str(env_file)],
    )
    assert el_ok.exit_code == 0, el_ok.output
    el_payload = json.loads(el_ok.output)
    assert el_payload["surface"] == "elevenlabs-api"
    assert el_payload["status"] == "not_detected"

    ca_ok = runner.invoke(
        oracles.cli,
        ["check-cartesia", str(audio_file), "--acknowledge-upload", "--env-file", str(env_file)],
    )
    assert ca_ok.exit_code == 0, ca_ok.output
    ca_payload = json.loads(ca_ok.output)
    assert ca_payload["surface"] == "cartesia-api"
    assert ca_payload["status"] == "detected"


def test_resemble_verifies_remote_sha256_polls_and_maps_perth_and_synthid_verdicts(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    audio_file = _write_minimal_mp3(tmp_path / "perth.mp3")
    file_bytes = audio_file.read_bytes()
    presigned_uri = "https://cdn.example.com/oracle/perth.mp3?X-Amz-Signature=presigned-secret-sig"

    # 1. Rejects non-HTTPS URI and mismatched remote SHA-256 before calling Resemble
    with pytest.raises(ValueError, match="HTTPS"):
        audio_apis.verify_resemble_watermark(
            audio_file,
            content_uri="http://cdn.example.com/perth.mp3",
            acknowledge_upload=True,
            api_key="resemble-secret-token",
        )

    resemble_called = False

    def mismatch_handler(request: httpx.Request) -> httpx.Response:
        nonlocal resemble_called
        if request.url.host == "app.resemble.ai":
            resemble_called = True
        return httpx.Response(200, content=b"wrong-bytes")

    with pytest.raises(ValueError, match="do not match"):
        audio_apis.verify_resemble_watermark(
            audio_file,
            content_uri=presigned_uri,
            acknowledge_upload=True,
            api_key="resemble-secret-token",
            transport=httpx.MockTransport(mismatch_handler),
        )
    assert resemble_called is False

    # 2. Polls pending -> completed, maps present + synthid=False, and scrubs token + query signature from logs
    calls: list[tuple[str, str, dict[str, str]]] = []

    def poll_handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url), dict(request.headers)))
        if request.url.host == "cdn.example.com":
            return httpx.Response(200, content=file_bytes)
        if request.method == "POST" and str(request.url) == "https://app.resemble.ai/api/v2/watermark/detect":
            body = json.loads(request.content.decode("utf-8"))
            assert body == {"url": presigned_uri}
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "item": {
                        "uuid": "job_123",
                        "media_type": "audio",
                        "operation_type": "detect",
                        "status": "pending",
                        "metrics": None,
                    },
                },
            )
        assert request.method == "GET"
        assert str(request.url) == "https://app.resemble.ai/api/v2/watermark/detect/job_123/result"
        return httpx.Response(
            200,
            json={
                "success": True,
                "item": {
                    "uuid": "job_123",
                    "media_type": "audio",
                    "operation_type": "detect",
                    "status": "completed",
                    "source_media_url": presigned_uri,
                    "metrics": {
                        "overall_status": "present",
                        "detected_model_versions": ["perth_v1"],
                        "coverage_complete": True,
                        "synthid": False,
                    },
                },
            },
        )

    with caplog.at_level(logging.INFO):
        res_pos = audio_apis.verify_resemble_watermark(
            audio_file,
            content_uri=presigned_uri,
            acknowledge_upload=True,
            api_key="resemble-secret-token",
            transport=httpx.MockTransport(poll_handler),
            sleeper=lambda _s: None,
        )
    assert res_pos.detected is True
    assert res_pos.status == "detected"
    assert res_pos.overall_status == "present"
    assert res_pos.synthid_status == "not_detected"
    assert res_pos.coverage_complete is True
    assert res_pos.detected_model_versions == ("perth_v1",)
    assert calls[1][2]["authorization"] == "Bearer resemble-secret-token"
    assert calls[1][2]["prefer"] == "wait"
    assert "resemble-secret-token" not in caplog.text
    assert "presigned-secret-sig" not in caplog.text

    # 3. Maps absent + coverage_complete=True -> not_detected, and absent + coverage_complete=False -> indeterminate
    def make_completed_handler(metrics: dict[str, object]) -> httpx.MockTransport:
        def _h(request: httpx.Request) -> httpx.Response:
            if request.url.host == "cdn.example.com":
                return httpx.Response(200, content=file_bytes)
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "item": {
                        "uuid": "job_456",
                        "media_type": "audio",
                        "status": "completed",
                        "metrics": metrics,
                    },
                },
            )

        return httpx.MockTransport(_h)

    res_neg = audio_apis.verify_resemble_watermark(
        audio_file,
        content_uri="https://cdn.example.com/oracle/perth.mp3",
        acknowledge_upload=True,
        api_key="resemble-secret-token",
        transport=make_completed_handler(
            {"overall_status": "absent", "detected_model_versions": [], "coverage_complete": True}
        ),
    )
    assert res_neg.status == "not_detected"
    assert res_neg.synthid_status == "unavailable"

    res_incomplete = audio_apis.verify_resemble_watermark(
        audio_file,
        content_uri="https://cdn.example.com/oracle/perth.mp3",
        acknowledge_upload=True,
        api_key="resemble-secret-token",
        transport=make_completed_handler(
            {"overall_status": "absent", "detected_model_versions": [], "coverage_complete": False, "synthid": True}
        ),
    )
    assert res_incomplete.status == "indeterminate"
    assert res_incomplete.synthid_status == "detected"


def test_cli_check_resemble(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio_file = _write_minimal_mp3(tmp_path / "clip.mp3")
    env_file = tmp_path / ".env"
    env_file.write_text("RESEMBLE_API_KEY=rs-cli-key\n", encoding="utf-8")

    monkeypatch.setattr(
        oracles,
        "verify_resemble",
        lambda path, *, content_uri, custom_message, acknowledge_upload, api_key: audio_apis.ResembleWatermarkDetection(
            status="not_detected",
            overall_status="absent",
            synthid_status="unavailable",
            job_uuid="job_cli_1",
            media_type="audio",
            coverage_complete=True,
            detected_model_versions=(),
            raw_response={"success": True},
        ),
    )
    runner = CliRunner()
    res = runner.invoke(
        oracles.cli,
        [
            "check-resemble",
            str(audio_file),
            "--content-uri",
            "https://cdn.example.com/clip.mp3",
            "--acknowledge-upload",
            "--env-file",
            str(env_file),
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["surface"] == "resemble-api"
    assert payload["status"] == "not_detected"
    assert payload["synthid_status"] == "unavailable"
