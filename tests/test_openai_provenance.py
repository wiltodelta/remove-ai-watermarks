"""Contract tests for metadata-independent official OpenAI SynthID verification."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image

from remove_ai_watermarks import openai_provenance as provenance

if TYPE_CHECKING:
    from pathlib import Path


class _Checks:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[tuple[str, bytes, str]] = []

    def create(self, *, file: tuple[str, Any, str], timeout: float) -> Any:
        assert timeout == provenance.REQUEST_TIMEOUT_SECONDS
        filename, stream, media_type = file
        self.calls.append((filename, stream.read(), media_type))
        return self.response


def _client(response: Any) -> tuple[Any, _Checks]:
    checks = _Checks(response)
    return SimpleNamespace(content_provenance_checks=checks), checks


def _response(*, synthid: str, c2pa: str = "not_detected") -> dict[str, Any]:
    return {
        "object": "content_provenance_check",
        "created_at": 1_778_000_000,
        "results": [
            {
                "type": "c2pa",
                "outcome": c2pa,
                "validation_state": "trusted" if c2pa == "detected" else "not_present",
                "issuer": "OpenAI OpCo, LLC" if c2pa == "detected" else None,
                "model": "metadata-model" if c2pa == "detected" else None,
                "generated_at": "2026-07-27T18:34:12Z" if c2pa == "detected" else None,
            },
            {
                "type": "synthid",
                "outcome": synthid,
                "model": "pixel-model" if synthid == "detected" else None,
                "generated_at": "2026-07-28T18:34:12Z" if synthid == "detected" else None,
            },
        ],
    }


def _verify(image_path: Path, *, client: Any | None = None) -> provenance.OpenAISynthIDDetection:
    return provenance.verify_openai_synthid(image_path, acknowledge_upload=True, client=client)


def test_upload_requires_explicit_library_acknowledgement(tmp_clean_png: Path) -> None:
    with pytest.raises(ValueError, match="acknowledge_upload=True"):
        provenance.verify_openai_synthid(tmp_clean_png)


def test_c2pa_only_response_is_not_a_synthid_detection(tmp_png_with_ai_metadata: Path) -> None:
    client, checks = _client(_response(synthid="not_detected", c2pa="detected"))

    result = _verify(tmp_png_with_ai_metadata, client=client)

    assert result.status == "not_detected"
    assert result.model is None
    assert result.generated_at is None
    assert result.ai_metadata_stripped is True
    assert result.pixels_preserved is True
    assert len(checks.calls) == 1
    filename, uploaded, media_type = checks.calls[0]
    assert filename == "upload.png"
    assert media_type == "image/png"
    with Image.open(io.BytesIO(uploaded)) as image:
        image.load()
        assert image.convert("RGBA").getpixel((0, 0)) == (128, 128, 128, 255)
        assert "parameters" not in image.info
        assert "prompt" not in image.info


def test_detected_result_uses_only_synthid_fields(tmp_clean_png: Path) -> None:
    client, _checks = _client(_response(synthid="detected", c2pa="not_detected"))

    result = _verify(tmp_clean_png, client=client)

    assert result.detected is True
    assert result.model == "pixel-model"
    assert result.generated_at == "2026-07-28T18:34:12Z"
    assert result.api_created_at == 1_778_000_000
    assert "c2pa" not in result.to_dict()
    assert result.to_dict()["metadata_used_for_verdict"] is False
    assert result.to_dict()["provider_scope"] == "openai"


def test_sdk_model_response_is_normalized(tmp_clean_png: Path) -> None:
    class SDKModel:
        def model_dump(self, *, mode: str) -> dict[str, Any]:
            assert mode == "json"
            return _response(synthid="detected")

    client, _checks = _client(SDKModel())

    result = _verify(tmp_clean_png, client=client)

    assert result.status == "detected"


def test_unexpected_response_object_is_an_error(tmp_clean_png: Path) -> None:
    response = _response(synthid="detected")
    response["object"] = "future_response"
    client, _checks = _client(response)

    with pytest.raises(RuntimeError, match="unexpected 'object'"):
        _verify(tmp_clean_png, client=client)


@pytest.mark.parametrize("entry", [None, {"outcome": "detected"}, {"type": 3, "outcome": "detected"}])
def test_malformed_result_entry_is_an_error(tmp_clean_png: Path, entry: Any) -> None:
    client, _checks = _client(
        {
            "object": "content_provenance_check",
            "results": [entry],
        }
    )

    with pytest.raises(RuntimeError, match=r"invalid result entry|valid type"):
        _verify(tmp_clean_png, client=client)


@pytest.mark.parametrize(
    ("image_format", "suffix", "media_type"),
    [("PNG", ".png", "image/png"), ("JPEG", ".jpg", "image/jpeg"), ("WEBP", ".webp", "image/webp")],
)
def test_all_documented_image_formats_preserve_decoded_pixels(
    tmp_path: Path,
    image_format: str,
    suffix: str,
    media_type: str,
) -> None:
    source = tmp_path / f"source{suffix}"
    image = Image.new("RGB", (19, 17))
    image.putdata([((x * 13) % 256, (x * 29) % 256, (x * 47) % 256) for x in range(19 * 17)])
    image.save(source, format=image_format, quality=91)
    with Image.open(source) as decoded:
        expected = decoded.convert("RGBA").tobytes()
    client, checks = _client(_response(synthid="not_detected"))

    _verify(source, client=client)

    filename, uploaded, actual_media_type = checks.calls[0]
    assert filename == f"upload{suffix}"
    assert actual_media_type == media_type
    with Image.open(io.BytesIO(uploaded)) as decoded:
        assert decoded.convert("RGBA").tobytes() == expected


@pytest.mark.parametrize("results", [[], [{"type": "c2pa", "outcome": "detected"}]])
def test_missing_synthid_result_is_an_error(tmp_clean_png: Path, results: list[dict[str, str]]) -> None:
    client, _checks = _client({"object": "content_provenance_check", "results": results})

    with pytest.raises(RuntimeError, match="0 SynthID results"):
        _verify(tmp_clean_png, client=client)


def test_duplicate_synthid_results_are_an_error(tmp_clean_png: Path) -> None:
    client, _checks = _client(
        {
            "object": "content_provenance_check",
            "results": [
                {"type": "synthid", "outcome": "detected"},
                {"type": "synthid", "outcome": "not_detected"},
            ],
        }
    )

    with pytest.raises(RuntimeError, match="2 SynthID results"):
        _verify(tmp_clean_png, client=client)


def test_pixel_mutation_aborts_before_remote_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_clean_png: Path,
) -> None:
    from remove_ai_watermarks import metadata

    client, checks = _client(_response(synthid="detected"))

    def mutate(source: Path, output: Path, *, keep_standard: bool) -> tuple[Path, dict[str, str]]:
        assert keep_standard is True
        with Image.open(source) as image:
            changed = image.convert("RGB")
            changed.putpixel((0, 0), (0, 0, 0))
            changed.save(output)
        return output, {}

    monkeypatch.setattr(metadata, "strip_and_verify", mutate)

    with pytest.raises(RuntimeError, match="changed the decoded pixels"):
        _verify(tmp_clean_png, client=client)
    assert checks.calls == []


def test_surviving_ai_metadata_aborts_before_remote_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_clean_png: Path,
) -> None:
    from remove_ai_watermarks import metadata

    client, checks = _client(_response(synthid="detected"))

    def survive(source: Path, output: Path, *, keep_standard: bool) -> tuple[Path, dict[str, str]]:
        assert keep_standard is True
        output.write_bytes(source.read_bytes())
        return output, {"C2PA": "present"}

    monkeypatch.setattr(metadata, "strip_and_verify", survive)

    with pytest.raises(RuntimeError, match="metadata survived"):
        _verify(tmp_clean_png, client=client)
    assert checks.calls == []


def test_unsupported_image_format_is_rejected_before_remote_request(tmp_path: Path) -> None:
    source = tmp_path / "image.bmp"
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(source)
    client, checks = _client(_response(synthid="detected"))

    with pytest.raises(ValueError, match="supports JPEG, PNG, WEBP"):
        _verify(source, client=client)
    assert checks.calls == []


def test_upload_limit_is_checked_after_sanitizing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_clean_png: Path,
) -> None:
    client, checks = _client(_response(synthid="detected"))
    monkeypatch.setattr(provenance, "MAX_UPLOAD_BYTES", 1)

    with pytest.raises(ValueError, match="50 MiB"):
        _verify(tmp_clean_png, client=client)
    assert checks.calls == []


def test_upload_limit_allows_exact_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_clean_png: Path,
) -> None:
    from remove_ai_watermarks import metadata

    client, checks = _client(_response(synthid="not_detected"))

    def copy_clean(source: Path, output: Path, *, keep_standard: bool) -> tuple[Path, dict[str, str]]:
        assert keep_standard is True
        output.write_bytes(source.read_bytes())
        return output, {}

    monkeypatch.setattr(metadata, "strip_and_verify", copy_clean)
    monkeypatch.setattr(provenance, "MAX_UPLOAD_BYTES", tmp_clean_png.stat().st_size)

    result = _verify(tmp_clean_png, client=client)

    assert result.status == "not_detected"
    assert len(checks.calls) == 1


def test_missing_optional_sdk_has_install_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_clean_png: Path,
) -> None:
    monkeypatch.setattr(provenance, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match=r"remove-ai-watermarks\[verify\]"):
        _verify(tmp_clean_png)


def test_client_configuration_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_clean_png: Path,
) -> None:
    def fail(**_kwargs: Any) -> None:
        raise ValueError("OPENAI_API_KEY is missing")

    monkeypatch.setattr(provenance, "is_available", lambda: True)
    monkeypatch.setattr(provenance.importlib, "import_module", lambda _name: SimpleNamespace(OpenAI=fail))

    with pytest.raises(RuntimeError, match=r"could not initialize.*OPENAI_API_KEY"):
        _verify(tmp_clean_png)


def test_default_client_bounds_one_acknowledged_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    expected = SimpleNamespace(content_provenance_checks=object())

    def factory(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(provenance, "is_available", lambda: True)
    monkeypatch.setattr(provenance.importlib, "import_module", lambda _name: SimpleNamespace(OpenAI=factory))

    assert provenance._default_client() is expected
    assert calls == [
        {
            "timeout": provenance.REQUEST_TIMEOUT_SECONDS,
            "max_retries": 0,
        }
    ]


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (400, "rejected"),
        (401, "authentication failed"),
        (403, "not permitted"),
        (404, "does not have"),
        (429, "rate limit"),
        (500, "temporary server error"),
    ],
)
def test_documented_api_errors_are_actionable(
    tmp_clean_png: Path,
    status_code: int,
    message: str,
) -> None:
    class APIError(Exception):
        pass

    error = APIError("details")
    error.status_code = status_code  # type: ignore[attr-defined]

    class FailingChecks:
        def create(self, *, file: tuple[str, Any, str], timeout: float) -> None:
            assert timeout == provenance.REQUEST_TIMEOUT_SECONDS
            raise error

    client = SimpleNamespace(content_provenance_checks=FailingChecks())

    with pytest.raises(RuntimeError, match=message):
        _verify(tmp_clean_png, client=client)


@pytest.mark.parametrize(
    ("error_name", "message"),
    [("APITimeoutError", "timed out"), ("APIConnectionError", "could not be reached")],
)
def test_transport_errors_are_actionable(
    tmp_clean_png: Path,
    error_name: str,
    message: str,
) -> None:
    error_type = type(error_name, (Exception,), {})

    class FailingChecks:
        def create(self, *, file: tuple[str, Any, str], timeout: float) -> None:
            assert timeout == provenance.REQUEST_TIMEOUT_SECONDS
            raise error_type("details")

    client = SimpleNamespace(content_provenance_checks=FailingChecks())

    with pytest.raises(RuntimeError, match=message):
        _verify(tmp_clean_png, client=client)


def test_rate_limit_error_preserves_retry_context(tmp_clean_png: Path) -> None:
    class RateLimitError(Exception):
        status_code = 429
        code = "rate_limit_exceeded"
        request_id = "req_test"
        response = SimpleNamespace(headers={"retry-after": "7"})

    class FailingChecks:
        def create(self, *, file: tuple[str, Any, str], timeout: float) -> None:
            assert timeout == provenance.REQUEST_TIMEOUT_SECONDS
            raise RateLimitError("details")

    client = SimpleNamespace(content_provenance_checks=FailingChecks())

    with pytest.raises(provenance.OpenAIProvenanceError, match="Retry-After: 7") as raised:
        _verify(tmp_clean_png, client=client)
    assert raised.value.status_code == 429
    assert raised.value.error_code == "rate_limit_exceeded"
    assert raised.value.request_id == "req_test"
    assert raised.value.retry_after == "7"
    assert raised.value.retryable is True


def test_client_error_is_not_marked_retryable(tmp_clean_png: Path) -> None:
    class BadRequestError(Exception):
        status_code = 400
        code = "invalid_image"

    class FailingChecks:
        def create(self, *, file: tuple[str, Any, str], timeout: float) -> None:
            assert timeout == provenance.REQUEST_TIMEOUT_SECONDS
            raise BadRequestError("details")

    client = SimpleNamespace(content_provenance_checks=FailingChecks())

    with pytest.raises(provenance.OpenAIProvenanceError) as raised:
        _verify(tmp_clean_png, client=client)
    assert raised.value.status_code == 400
    assert raised.value.error_code == "invalid_image"
    assert raised.value.retryable is False


def test_keyboard_interrupt_is_not_wrapped_or_retried(tmp_clean_png: Path) -> None:
    class InterruptingChecks:
        calls = 0

        def create(self, *, file: tuple[str, Any, str], timeout: float) -> None:
            assert timeout == provenance.REQUEST_TIMEOUT_SECONDS
            self.calls += 1
            raise KeyboardInterrupt

    checks = InterruptingChecks()
    client = SimpleNamespace(content_provenance_checks=checks)

    with pytest.raises(KeyboardInterrupt):
        _verify(tmp_clean_png, client=client)
    assert checks.calls == 1
