"""Direct ElevenLabs, Cartesia, and Resemble watermark-oracle API adapters."""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlparse, urlunparse

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

ELEVENLABS_ENDPOINT = "https://api.elevenlabs.io/v1/content-provenance/watermark"
ELEVENLABS_DETECTOR_ID = "elevenlabs-content-provenance-v1"
ELEVENLABS_MAX_BYTES = 100 * 1024 * 1024

CARTESIA_ENDPOINT = "https://api.cartesia.ai/v1/watermark/detect"
CARTESIA_API_VERSION = "2026-08-14"
CARTESIA_DETECTOR_ID = "cartesia-watermark-detect-2026-08-14"
CARTESIA_MAX_BYTES = 32 * 1024 * 1024

RESEMBLE_DETECT_ENDPOINT = "https://app.resemble.ai/api/v2/watermark/detect"
RESEMBLE_DETECTOR_ID = "resemble-watermark-detect-v2"
RESEMBLE_MAX_BYTES = 200 * 1024 * 1024
RESEMBLE_POLL_INTERVAL_SECONDS = 2.0
RESEMBLE_MAX_POLL_ATTEMPTS = 30

REQUEST_TIMEOUT_SECONDS = 120.0

WatermarkStatus = Literal["detected", "not_detected"]
ResembleWatermarkStatus = Literal["detected", "not_detected", "indeterminate"]
ResembleSynthIdStatus = Literal["detected", "not_detected", "unavailable"]

_RESEMBLE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".aac",
        ".avif",
        ".flac",
        ".jpeg",
        ".jpg",
        ".m4a",
        ".m4v",
        ".mov",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpga",
        ".oga",
        ".ogg",
        ".opus",
        ".png",
        ".wav",
        ".webm",
        ".webp",
    }
)

_AUDIO_MIMES: dict[str, str] = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".oga": "audio/ogg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "video/webm",
}


class AudioProvenanceApiError(RuntimeError):
    """Official audio-provenance API failure with scrubbed diagnostic context."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class _AudioDetectionBase:
    status: ResembleWatermarkStatus

    @property
    def detected(self) -> bool:
        return self.status == "detected"

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for field in fields(self):
            val = getattr(self, field.name)
            payload[field.name] = list(val) if isinstance(val, tuple) else val
        return payload


@dataclass(frozen=True)
class ElevenLabsProvenanceDetection(_AudioDetectionBase):
    """One official ElevenLabs Content Provenance watermark verdict."""

    status: WatermarkStatus
    product: str | None
    model: str | None
    generated_at: str | None
    unique_id: str | None
    raw_response: dict[str, object]
    ai_metadata_stripped: bool = True
    samples_preserved: bool = True
    detector: str = ELEVENLABS_DETECTOR_ID
    signal_family: str = "elevenlabs_watermark"
    provider_scope: str = "elevenlabs"
    backend: str = "official-elevenlabs-api"
    metadata_used_for_verdict: bool = False


@dataclass(frozen=True)
class CartesiaWatermarkDetection(_AudioDetectionBase):
    """One official Cartesia Sonic watermark detection verdict."""

    status: WatermarkStatus
    raw_response: dict[str, object]
    ai_metadata_stripped: bool = True
    samples_preserved: bool = True
    detector: str = CARTESIA_DETECTOR_ID
    signal_family: str = "sonic_watermark"
    provider_scope: str = "cartesia"
    backend: str = "official-cartesia-api"
    metadata_used_for_verdict: bool = False


@dataclass(frozen=True)
class ResembleWatermarkDetection(_AudioDetectionBase):
    """One official Resemble Perth watermark and auxiliary SynthID verdict."""

    overall_status: str
    synthid_status: ResembleSynthIdStatus
    job_uuid: str
    media_type: str | None
    coverage_complete: bool | None
    detected_model_versions: tuple[str, ...]
    raw_response: dict[str, object]
    detector: str = RESEMBLE_DETECTOR_ID
    signal_family: str = "perth"
    provider_scope: str = "resemble"
    backend: str = "official-resemble-api"
    metadata_used_for_verdict: bool = False


def _scrub(text: str, secret: str) -> str:
    return text.replace(secret, "[REDACTED]") if secret else text


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pcm_digest(path: Path) -> str:
    """Hash raw decoded float32 samples without synthetic Xing/edit-list delay trimming."""
    from remove_ai_watermarks._internal import audio_io

    cmd = [
        audio_io.require_tool("ffmpeg"),
        "-nostdin",
        "-v",
        "error",
        "-flags2",
        "+skip_manual",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    digest = hashlib.sha256()
    with tempfile.TemporaryFile(mode="w+b") as stderr:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr)  # noqa: S603
        try:
            if proc.stdout is None:
                raise RuntimeError("ffmpeg stdout pipe failed to initialize")
            while chunk := proc.stdout.read(65536):
                digest.update(chunk)
            audio_io.finish_process(proc, stderr, cmd)
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            audio_io.stop_process(proc)
    return digest.hexdigest()


def _sanitize_audio_upload(source: Path, target: Path) -> None:
    """Strip AI container metadata into TARGET and verify decoded PCM invariance."""
    from remove_ai_watermarks._internal.audio_quality import sha256_file
    from remove_ai_watermarks.metadata import strip_and_verify

    stripped, remaining = strip_and_verify(source, target, keep_standard=True)
    if remaining:
        fields_left = ", ".join(sorted(remaining))
        raise RuntimeError(f"refusing audio oracle upload because AI metadata survived stripping: {fields_left}")
    if stripped.stat().st_size == source.stat().st_size and sha256_file(stripped) == sha256_file(source):
        return
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_stripped = pool.submit(_pcm_digest, stripped)
        f_source = pool.submit(_pcm_digest, source)
        if f_stripped.result() != f_source.result():
            raise RuntimeError("refusing audio oracle upload because metadata stripping altered decoded audio samples")


def _prepare_upload_bytes(
    path: Path,
    *,
    max_bytes: int,
    strip_metadata: bool,
) -> tuple[bytes, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"oracle source is not a file: {resolved}")
    suffix = resolved.suffix.lower()
    mime = _AUDIO_MIMES.get(suffix)
    if mime is None:
        raise ValueError(f"unsupported audio/media extension for API oracle: {suffix or '<none>'}")
    if not strip_metadata:
        raw_size = resolved.stat().st_size
        if raw_size <= 0 or raw_size > max_bytes:
            raise ValueError(f"upload size {raw_size} bytes is outside allowed range 1..{max_bytes}")
        payload = resolved.read_bytes()
    else:
        with tempfile.TemporaryDirectory(prefix="raiw-audio-oracle-") as tmp_dir:
            staged = Path(tmp_dir) / f"sanitized{suffix}"
            _sanitize_audio_upload(resolved, staged)
            payload = staged.read_bytes()
    if len(payload) <= 0 or len(payload) > max_bytes:
        raise ValueError(f"upload size {len(payload)} bytes is outside allowed range 1..{max_bytes}")
    return payload, mime


def _execute_json_request(
    send: Callable[[], httpx.Response],
    *,
    provider_label: str,
    context_url: str,
    scrub: Callable[[str], str],
) -> dict[str, object]:
    try:
        response = send()
    except httpx.HTTPError as exc:
        scrubbed_err = scrub(str(exc))
        log.error("%s transport error at %s: %s", provider_label, context_url, scrubbed_err)
        raise AudioProvenanceApiError(f"{provider_label} request failed: {scrubbed_err}") from None

    scrubbed_body = scrub(response.text)
    if response.status_code >= 400:
        log.warning(
            "%s returned HTTP %d from %s: %s",
            provider_label,
            response.status_code,
            context_url,
            scrubbed_body,
        )
        raise AudioProvenanceApiError(
            f"{provider_label} returned HTTP {response.status_code}: {scrubbed_body}",
            status_code=response.status_code,
        )

    log.info("%s HTTP %d response (%s): %s", provider_label, response.status_code, context_url, scrubbed_body)
    try:
        parsed: Any = response.json()
    except ValueError:
        log.error("%s returned non-JSON body: %s", provider_label, scrubbed_body)
        raise AudioProvenanceApiError(
            f"{provider_label} returned non-JSON response",
            status_code=response.status_code,
        ) from None
    if not isinstance(parsed, dict):
        raise AudioProvenanceApiError(
            f"{provider_label} returned unexpected JSON payload type",
            status_code=response.status_code,
        )
    return cast("dict[str, object]", parsed)


def _post_multipart(
    *,
    provider_label: str,
    endpoint: str,
    headers: dict[str, str],
    secret: str,
    filename: str,
    payload: bytes,
    mime: str,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
) -> dict[str, object]:
    upload_sha = _sha256_bytes(payload)
    log.info(
        "POST %s (%s) file=%s sha256=%s bytes=%d mime=%s",
        endpoint,
        provider_label,
        filename,
        upload_sha,
        len(payload),
        mime,
    )
    with httpx.Client(
        timeout=timeout_seconds,
        follow_redirects=False,
        trust_env=False,
        transport=transport,
    ) as client:
        return _execute_json_request(
            lambda: client.post(endpoint, headers=headers, files={"file": (filename, payload, mime)}),
            provider_label=provider_label,
            context_url=endpoint,
            scrub=lambda text: _scrub(text, secret),
        )


def _submit_audio_oracle_upload(
    path: Path,
    *,
    provider_short: str,
    provider_label: str,
    endpoint: str,
    max_bytes: int,
    acknowledge_upload: bool,
    api_key: str,
    auth_header: str,
    extra_headers: dict[str, str] | None = None,
    strip_metadata: bool,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
) -> dict[str, object]:
    if not acknowledge_upload:
        raise ValueError(f"acknowledge_upload=True is required before transmitting audio to {provider_short}")
    cleaned_key = api_key.strip()
    if not cleaned_key:
        raise ValueError(f"{provider_short} API key must not be empty")
    payload, mime = _prepare_upload_bytes(
        path,
        max_bytes=max_bytes,
        strip_metadata=strip_metadata,
    )
    headers = {auth_header: cleaned_key, "Accept": "application/json", **(extra_headers or {})}
    return _post_multipart(
        provider_label=provider_label,
        endpoint=endpoint,
        headers=headers,
        secret=cleaned_key,
        filename=f"upload{path.suffix.lower()}",
        payload=payload,
        mime=mime,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )


def _optional_str(mapping: dict[str, Any], key: str) -> str | None:
    val = mapping.get(key)
    return val if isinstance(val, str) else None


def verify_elevenlabs_provenance(
    path: Path,
    *,
    acknowledge_upload: bool,
    api_key: str,
    strip_metadata: bool = True,
    timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> ElevenLabsProvenanceDetection:
    """Submit one audio file to the official ElevenLabs Content Provenance API."""
    raw = _submit_audio_oracle_upload(
        path,
        provider_short="ElevenLabs",
        provider_label="ElevenLabs Content Provenance API",
        endpoint=ELEVENLABS_ENDPOINT,
        max_bytes=ELEVENLABS_MAX_BYTES,
        acknowledge_upload=acknowledge_upload,
        api_key=api_key,
        auth_header="xi-api-key",
        strip_metadata=strip_metadata,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )
    is_detected = raw.get("is_detected")
    if not isinstance(is_detected, bool):
        raise AudioProvenanceApiError("ElevenLabs response missing boolean is_detected field")
    details_raw = raw.get("details")
    details = cast("dict[str, Any]", details_raw) if isinstance(details_raw, dict) else {}
    return ElevenLabsProvenanceDetection(
        status="detected" if is_detected else "not_detected",
        product=_optional_str(details, "product"),
        model=_optional_str(details, "model"),
        generated_at=_optional_str(details, "generated_at"),
        unique_id=_optional_str(details, "unique_id"),
        raw_response=raw,
        ai_metadata_stripped=strip_metadata,
        metadata_used_for_verdict=not strip_metadata,
    )


def verify_cartesia_watermark(
    path: Path,
    *,
    acknowledge_upload: bool,
    api_key: str,
    strip_metadata: bool = True,
    timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> CartesiaWatermarkDetection:
    """Submit one audio/video file to the official Cartesia Watermark Detection API."""
    raw = _submit_audio_oracle_upload(
        path,
        provider_short="Cartesia",
        provider_label="Cartesia Watermark Detection API",
        endpoint=CARTESIA_ENDPOINT,
        max_bytes=CARTESIA_MAX_BYTES,
        acknowledge_upload=acknowledge_upload,
        api_key=api_key,
        auth_header="X-API-Key",
        extra_headers={"Cartesia-Version": CARTESIA_API_VERSION},
        strip_metadata=strip_metadata,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )
    detected = raw.get("detected")
    if not isinstance(detected, bool):
        raise AudioProvenanceApiError("Cartesia response missing boolean detected field")
    return CartesiaWatermarkDetection(
        status="detected" if detected else "not_detected",
        raw_response=raw,
        ai_metadata_stripped=strip_metadata,
        metadata_used_for_verdict=not strip_metadata,
    )


def _validate_resemble_content_uri(content_uri: str) -> tuple[str, str]:
    parsed = urlparse(content_uri)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("content_uri must be an HTTPS URL without credentials or fragment")
    safe_uri = urlunparse(parsed._replace(query="<redacted>")) if parsed.query else content_uri
    return content_uri, safe_uri


def _scrub_resemble(text: str, *, secret: str, content_uri: str, safe_uri: str) -> str:
    scrubbed = _scrub(text, secret)
    if content_uri != safe_uri:
        scrubbed = scrubbed.replace(content_uri, safe_uri)
    return scrubbed


@contextmanager
def _redact_httpx_uri_logs(content_uri: str, safe_uri: str) -> Any:
    if content_uri == safe_uri:
        yield
        return

    class _UriFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            rendered = record.getMessage()
            if content_uri in rendered:
                record.msg = rendered.replace(content_uri, safe_uri)
                record.args = None
            return True

    httpx_logger = logging.getLogger("httpx")
    uri_filter = _UriFilter()
    httpx_logger.addFilter(uri_filter)
    try:
        yield
    finally:
        httpx_logger.removeFilter(uri_filter)


def _verify_remote_sha256(
    client: httpx.Client,
    *,
    content_uri: str,
    safe_uri: str,
    secret: str,
    expected_sha256: str,
) -> None:
    digest = hashlib.sha256()
    total = 0
    try:
        with client.stream("GET", content_uri) as response:
            if response.status_code >= 400:
                raise AudioProvenanceApiError(
                    f"Resemble content_uri preflight returned HTTP {response.status_code} for {safe_uri}",
                    status_code=response.status_code,
                )
            for chunk in response.iter_bytes(65536):
                total += len(chunk)
                if total > RESEMBLE_MAX_BYTES:
                    raise ValueError(f"content_uri exceeds Resemble 200 MiB limit ({RESEMBLE_MAX_BYTES} bytes)")
                digest.update(chunk)
    except httpx.HTTPError as exc:
        err = _scrub_resemble(str(exc), secret=secret, content_uri=content_uri, safe_uri=safe_uri)
        log.error("Resemble content_uri preflight failed for %s: %s", safe_uri, err)
        raise AudioProvenanceApiError(f"Resemble content_uri preflight failed: {err}") from None
    actual_sha256 = digest.hexdigest()
    log.info("Resemble content_uri preflight: uri=%s bytes=%d sha256=%s", safe_uri, total, actual_sha256)
    if actual_sha256 != expected_sha256:
        raise ValueError("content_uri bytes do not match the local source")


def _resemble_json_request(
    client: httpx.Client,
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    json_body: dict[str, object] | None,
    secret: str,
    content_uri: str,
    safe_uri: str,
) -> dict[str, object]:
    return _execute_json_request(
        lambda: client.request(method, url, headers=headers, json=json_body),
        provider_label="Resemble API",
        context_url=f"{method} {url}",
        scrub=lambda text: _scrub_resemble(text, secret=secret, content_uri=content_uri, safe_uri=safe_uri),
    )


def verify_resemble_watermark(
    path: Path,
    *,
    content_uri: str,
    acknowledge_upload: bool,
    api_key: str,
    custom_message: str | None = None,
    timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    poll_interval_seconds: float = RESEMBLE_POLL_INTERVAL_SECONDS,
    max_poll_attempts: int = RESEMBLE_MAX_POLL_ATTEMPTS,
    transport: httpx.BaseTransport | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> ResembleWatermarkDetection:
    """Verify one local file against Resemble's official Watermark Detection API via a matching HTTPS URI."""
    from remove_ai_watermarks._internal.audio_quality import sha256_file

    if not acknowledge_upload:
        raise ValueError("acknowledge_upload=True is required before submitting a URI to Resemble")
    cleaned_key = api_key.strip()
    if not cleaned_key:
        raise ValueError("Resemble API key must not be empty")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"oracle source is not a file: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix not in _RESEMBLE_EXTENSIONS:
        raise ValueError(f"unsupported media extension for Resemble oracle: {suffix or '<none>'}")
    raw_size = resolved.stat().st_size
    if raw_size <= 0 or raw_size > RESEMBLE_MAX_BYTES:
        raise ValueError(f"source size {raw_size} bytes is outside allowed range 1..{RESEMBLE_MAX_BYTES}")
    if custom_message is not None and not (1 <= len(custom_message) <= 64):
        raise ValueError("custom_message must be between 1 and 64 characters")
    validated_uri, safe_uri = _validate_resemble_content_uri(content_uri)
    local_sha = sha256_file(resolved)

    with (
        _redact_httpx_uri_logs(validated_uri, safe_uri),
        httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        ) as client,
    ):
        _verify_remote_sha256(
            client,
            content_uri=validated_uri,
            safe_uri=safe_uri,
            secret=cleaned_key,
            expected_sha256=local_sha,
        )
        body: dict[str, object] = {"url": validated_uri}
        if custom_message is not None:
            body["custom_message"] = custom_message
        log.info(
            "POST %s (Resemble Watermark Detection API) uri=%s sha256=%s",
            RESEMBLE_DETECT_ENDPOINT,
            safe_uri,
            local_sha,
        )
        raw = _resemble_json_request(
            client,
            method="POST",
            url=RESEMBLE_DETECT_ENDPOINT,
            headers={
                "Authorization": f"Bearer {cleaned_key}",
                "Accept": "application/json",
                "Prefer": "wait",
            },
            json_body=body,
            secret=cleaned_key,
            content_uri=validated_uri,
            safe_uri=safe_uri,
        )
        if raw.get("success") is not True or not isinstance(raw.get("item"), dict):
            raise AudioProvenanceApiError("Resemble API response did not return a valid detect item")
        item = cast("dict[str, Any]", raw["item"])
        job_uuid = _optional_str(item, "uuid")
        if job_uuid is None or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_uuid) is None:
            raise AudioProvenanceApiError("Resemble API response returned an invalid job uuid")

        polls = 0
        while item.get("status") in {"pending", "processing"}:
            if polls >= max_poll_attempts:
                raise AudioProvenanceApiError(f"Resemble watermark job {job_uuid} timed out while polling")
            polls += 1
            sleeper(poll_interval_seconds)
            poll_url = f"{RESEMBLE_DETECT_ENDPOINT}/{job_uuid}/result"
            raw = _resemble_json_request(
                client,
                method="GET",
                url=poll_url,
                headers={"Authorization": f"Bearer {cleaned_key}", "Accept": "application/json"},
                json_body=None,
                secret=cleaned_key,
                content_uri=validated_uri,
                safe_uri=safe_uri,
            )
            if raw.get("success") is not True or not isinstance(raw.get("item"), dict):
                raise AudioProvenanceApiError("Resemble poll response did not return a valid detect item")
            item = cast("dict[str, Any]", raw["item"])

    if item.get("status") == "failed":
        raise AudioProvenanceApiError(f"Resemble watermark job {job_uuid} failed")
    if item.get("status") != "completed" or not isinstance(item.get("metrics"), dict):
        raise AudioProvenanceApiError(f"Resemble watermark job {job_uuid} ended without metrics")
    metrics = cast("dict[str, Any]", item["metrics"])
    overall_status = _optional_str(metrics, "overall_status")
    coverage_raw = metrics.get("coverage_complete")
    if coverage_raw is not None and not isinstance(coverage_raw, bool):
        raise AudioProvenanceApiError("Resemble response has invalid coverage_complete field")
    if overall_status in {"present", "degraded"}:
        status: ResembleWatermarkStatus = "detected"
    elif overall_status == "absent":
        status = "indeterminate" if coverage_raw is False else "not_detected"
    elif overall_status == "inconclusive":
        status = "indeterminate"
    else:
        raise AudioProvenanceApiError(f"Resemble response has unsupported overall_status: {overall_status!r}")

    synthid_raw = metrics.get("synthid")
    if synthid_raw is None:
        synthid_status: ResembleSynthIdStatus = "unavailable"
    elif isinstance(synthid_raw, bool):
        synthid_status = "detected" if synthid_raw else "not_detected"
    else:
        raise AudioProvenanceApiError("Resemble response has invalid synthid field")

    versions_raw = metrics.get("detected_model_versions")
    versions = (
        tuple(v for v in cast("list[object]", versions_raw) if isinstance(v, str))
        if isinstance(versions_raw, list)
        else ()
    )
    if isinstance(item.get("source_media_url"), str) and validated_uri != safe_uri:
        item = {**item, "source_media_url": safe_uri}
        raw = {**raw, "item": item}
    return ResembleWatermarkDetection(
        status=status,
        overall_status=overall_status,
        synthid_status=synthid_status,
        job_uuid=job_uuid,
        media_type=_optional_str(item, "media_type"),
        coverage_complete=coverage_raw,
        detected_model_versions=versions,
        raw_response=raw,
    )
