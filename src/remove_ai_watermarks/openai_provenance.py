"""Official OpenAI SynthID verification with metadata-independent input.

The Content Provenance API returns C2PA and SynthID outcomes independently.
This module removes AI provenance metadata before upload, proves that the
decoded RGBA raster did not change, and then consumes only the SynthID result.
It is intentionally separate from :func:`identify`: calling it uploads one
sanitized raster to OpenAI and therefore always requires an explicit user
action.

The OpenAI SDK is optional. Imports remain lazy so local and metadata-only
paths do not acquire a network client dependency.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

log = logging.getLogger(__name__)

OpenAISynthIDStatus = Literal["detected", "not_detected"]

DETECTOR_ID = "openai-content-provenance-synthid-v1"
INSTALL_HINT = "install the verification extra: uv add 'remove-ai-watermarks[verify]'"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 120.0
# One acknowledgement authorizes one upload. The SDK otherwise retries some
# failures by default, which can transmit the same media more than once.
MAX_AUTOMATIC_RETRIES = 0
_FORMAT_DETAILS = {
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
}


@dataclass(frozen=True)
class OpenAISynthIDDetection:
    """One official OpenAI pixel-watermark verdict."""

    status: OpenAISynthIDStatus
    model: str | None
    generated_at: str | None
    api_created_at: int | None
    detector: str = DETECTOR_ID
    ai_metadata_stripped: bool = True
    pixels_preserved: bool = True
    signal_family: str = "synthid"
    provider_scope: str = "openai"
    backend: str = "official-openai-api"
    metadata_used_for_verdict: bool = False

    @property
    def detected(self) -> bool:
        """Whether the official verifier recognized an OpenAI SynthID signal."""
        return self.status == "detected"

    def to_dict(self) -> dict[str, str | int | bool | None]:
        """Return a JSON-safe result without a local path or C2PA outcome."""
        return {
            "status": self.status,
            "model": self.model,
            "generated_at": self.generated_at,
            "api_created_at": self.api_created_at,
            "detector": self.detector,
            "ai_metadata_stripped": self.ai_metadata_stripped,
            "pixels_preserved": self.pixels_preserved,
            "signal_family": self.signal_family,
            "provider_scope": self.provider_scope,
            "backend": self.backend,
            "metadata_used_for_verdict": self.metadata_used_for_verdict,
        }


class OpenAIProvenanceError(RuntimeError):
    """An official verification failure, with retry and support context."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        retry_after: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.request_id = request_id
        self.retry_after = retry_after
        self.retryable = retryable


def is_available() -> bool:
    """True when the optional OpenAI SDK is installed."""
    from remove_ai_watermarks.optional_deps import module_available

    return module_available("openai")


def _pixel_fingerprint(path: Path) -> tuple[str, str]:
    """Return the PIL format and a bounded-memory hash of decoded RGBA pixels."""
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        image_format = image.format
        if image_format not in _FORMAT_DETAILS:
            supported = ", ".join(sorted(_FORMAT_DETAILS))
            actual = image_format or "unknown"
            raise ValueError(f"OpenAI SynthID verification supports {supported} images; got {actual}")

        digest = hashlib.sha256()
        digest.update(f"{image.width}x{image.height}:RGBA\0".encode())
        # Hash bands instead of materializing a second full-image byte string.
        for top in range(0, image.height, 128):
            bottom = min(top + 128, image.height)
            digest.update(image.crop((0, top, image.width, bottom)).convert("RGBA").tobytes())
    return image_format, digest.hexdigest()


def _response_mapping(response: Any) -> Mapping[str, Any]:
    """Normalize an SDK model or test double to the documented response mapping."""
    if isinstance(response, Mapping):
        return cast("Mapping[str, Any]", response)
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return cast("Mapping[str, Any]", dumped)
    raise RuntimeError("OpenAI Content Provenance returned an unexpected response type")


def _optional_string(entry: Mapping[str, Any], field: str) -> str | None:
    value = entry.get(field)
    if value is None or isinstance(value, str):
        return value
    raise RuntimeError(f"OpenAI SynthID result has an invalid {field!r} field")


def _parse_synthid_result(payload: Mapping[str, Any]) -> OpenAISynthIDDetection:
    """Read exactly one SynthID entry and deliberately ignore C2PA entries."""
    if payload.get("object") != "content_provenance_check":
        raise RuntimeError("OpenAI Content Provenance response has an unexpected 'object' field")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError("OpenAI Content Provenance response has no results list")
    results = cast("list[Any]", raw_results)
    synthid_entries: list[Mapping[str, Any]] = []
    for raw_entry in results:
        if not isinstance(raw_entry, Mapping):
            raise RuntimeError("OpenAI Content Provenance response has an invalid result entry")
        entry = cast("Mapping[str, Any]", raw_entry)
        result_type = entry.get("type")
        if not isinstance(result_type, str):
            raise RuntimeError("OpenAI Content Provenance response has a result without a valid type")
        if result_type == "synthid":
            synthid_entries.append(entry)
    if len(synthid_entries) != 1:
        raise RuntimeError(f"OpenAI Content Provenance returned {len(synthid_entries)} SynthID results; expected one")

    synthid = synthid_entries[0]
    outcome = synthid.get("outcome")
    if outcome not in ("detected", "not_detected"):
        raise RuntimeError(f"OpenAI SynthID result has an unsupported outcome: {outcome!r}")
    created_at = payload.get("created_at")
    if created_at is not None and (not isinstance(created_at, int) or isinstance(created_at, bool)):
        raise RuntimeError("OpenAI Content Provenance response has an invalid 'created_at' field")
    return OpenAISynthIDDetection(
        status=outcome,
        model=_optional_string(synthid, "model"),
        generated_at=_optional_string(synthid, "generated_at"),
        api_created_at=created_at,
    )


def _default_client() -> Any:
    if not is_available():
        raise RuntimeError(f"OpenAI SynthID verification needs the OpenAI SDK; {INSTALL_HINT}")
    openai_module = importlib.import_module("openai")
    client_factory = cast("Callable[..., Any]", openai_module.OpenAI)
    try:
        client = client_factory(
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=MAX_AUTOMATIC_RETRIES,
        )
    except Exception as exc:
        raise RuntimeError(f"could not initialize the OpenAI client: {exc}") from exc
    if not hasattr(client, "content_provenance_checks"):
        raise RuntimeError(f"OpenAI SynthID verification needs openai>=2.52.0; {INSTALL_HINT}")
    return client


def _string_attribute(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _response_header(exc: Exception, name: str) -> str | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        return _string_attribute(headers.get(name))
    except (AttributeError, TypeError):
        return None


def _request_error(exc: Exception) -> OpenAIProvenanceError:
    raw_status_code = getattr(exc, "status_code", None)
    status_code = (
        raw_status_code if isinstance(raw_status_code, int) and not isinstance(raw_status_code, bool) else None
    )
    error_code = _string_attribute(getattr(exc, "code", None))
    if error_code is None:
        body = getattr(exc, "body", None)
        # The SDK types the exception body as Any, so narrowing it leaves a
        # Mapping with unknown parameters. The same cast the module already uses
        # for response payloads keeps the gate clean here.
        if isinstance(body, Mapping):
            error_code = _string_attribute(cast("Mapping[str, Any]", body).get("code"))
    request_id = _string_attribute(getattr(exc, "request_id", None)) or _response_header(exc, "x-request-id")
    retry_after = _response_header(exc, "retry-after")
    error_name = type(exc).__name__
    if error_name == "APITimeoutError":
        detail = "the OpenAI Content Provenance request timed out"
    elif error_name == "APIConnectionError":
        detail = "the OpenAI Content Provenance API could not be reached"
    elif status_code == 400:
        detail = "OpenAI rejected the image as malformed, unsupported, or blocked"
    elif status_code == 401:
        detail = "OpenAI Content Provenance authentication failed"
    elif status_code == 403:
        detail = "the OpenAI project is not permitted to use Content Provenance"
    elif status_code == 404:
        detail = "the OpenAI organization does not have Content Provenance API access"
    elif status_code == 429:
        detail = "the OpenAI Content Provenance API rate limit was exceeded"
    elif status_code is not None and status_code >= 500:
        detail = "the OpenAI Content Provenance API returned a temporary server error"
    else:
        detail = f"OpenAI Content Provenance request failed: {exc}"
    if retry_after is not None:
        detail += f"; Retry-After: {retry_after}"
    if error_code is not None:
        detail += f"; error code: {error_code}"
    if request_id is not None:
        detail += f"; request id: {request_id}"
    retryable = (
        error_name in ("APITimeoutError", "APIConnectionError")
        or status_code == 429
        or (status_code is not None and status_code >= 500)
    )
    return OpenAIProvenanceError(
        detail,
        status_code=status_code,
        error_code=error_code,
        request_id=request_id,
        retry_after=retry_after,
        retryable=retryable,
    )


def verify_openai_synthid(
    image_path: str | Path,
    *,
    acknowledge_upload: bool = False,
    client: Any | None = None,
) -> OpenAISynthIDDetection:
    """Verify OpenAI SynthID after stripping AI metadata without changing pixels.

    This function performs one remote request and uploads a temporary sanitized
    copy of the image. It never uses C2PA as a fallback and never interprets a
    negative result as proof that the image is human-created.
    """
    if not acknowledge_upload:
        raise ValueError(
            "OpenAI SynthID verification uploads a temporary pixel-identical copy; "
            "pass acknowledge_upload=True to continue"
        )
    source = Path(image_path)
    source_format, source_fingerprint = _pixel_fingerprint(source)
    media_type, suffix = _FORMAT_DETAILS[source_format]

    with tempfile.TemporaryDirectory(prefix="remove-ai-watermarks-openai-") as directory:
        sanitized = Path(directory) / f"upload{suffix}"
        from remove_ai_watermarks.metadata import strip_and_verify

        stripped, remaining = strip_and_verify(source, sanitized, keep_standard=True)
        if remaining:
            fields = ", ".join(sorted(remaining))
            raise RuntimeError(f"refusing upload because AI provenance metadata survived stripping: {fields}")
        stripped_format, stripped_fingerprint = _pixel_fingerprint(stripped)
        if stripped_format != source_format or stripped_fingerprint != source_fingerprint:
            raise RuntimeError("refusing upload because metadata stripping changed the decoded pixels")
        upload_bytes = stripped.stat().st_size
        if upload_bytes > MAX_UPLOAD_BYTES:
            raise ValueError("sanitized image exceeds the OpenAI Content Provenance 50 MiB upload limit")

        api_client = client if client is not None else _default_client()
        if not hasattr(api_client, "content_provenance_checks"):
            raise RuntimeError("OpenAI client does not expose content_provenance_checks; openai>=2.52.0 is required")
        request_context = {
            "endpoint": "/v1/content_provenance_checks",
            "filename": sanitized.name,
            "media_type": media_type,
            "bytes": upload_bytes,
            "automatic_retries": MAX_AUTOMATIC_RETRIES,
            "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        }
        log.info("OpenAI Content Provenance request: %s", json.dumps(request_context, sort_keys=True))
        started_at = time.monotonic()
        try:
            with stripped.open("rb") as upload:
                response = api_client.content_provenance_checks.create(
                    file=(sanitized.name, upload, media_type),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
        except Exception as exc:
            failure_context = {
                **request_context,
                "duration_ms": round((time.monotonic() - started_at) * 1000),
                "error_code": _string_attribute(getattr(exc, "code", None)),
                "request_id": _string_attribute(getattr(exc, "request_id", None))
                or _response_header(exc, "x-request-id"),
                "status_code": getattr(exc, "status_code", None),
            }
            log.exception(
                "OpenAI Content Provenance request failed: %s",
                json.dumps(failure_context, default=str, sort_keys=True),
            )
            raise _request_error(exc) from exc

        payload = _response_mapping(response)
        response_context = {
            "duration_ms": round((time.monotonic() - started_at) * 1000),
            "payload": payload,
            "request_id": _string_attribute(getattr(response, "_request_id", None)),
        }
        log.info("OpenAI Content Provenance response: %s", json.dumps(response_context, default=str, sort_keys=True))
        return _parse_synthid_result(payload)
