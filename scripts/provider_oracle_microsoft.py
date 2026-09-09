"""Direct Azure Content Provenance Detection adapter for oracle experiments."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import unquote, urlparse

import httpx

log = logging.getLogger(__name__)

API_VERSION = "2026-07-01-preview"
DETECTOR_ID = "azure-content-provenance-2026-07-01-preview"
MAX_MEDIA_BYTES = 100 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 120.0
POLL_INTERVAL_SECONDS = 5.0
TERMINAL_OPERATION_STATES = frozenset({"Succeeded", "Failed", "Canceled"})

MicrosoftWatermarkStatus = Literal["detected", "not_detected"]
MicrosoftProvenanceStatus = Literal["present", "absent"]


@dataclass(frozen=True)
class MicrosoftMarker:
    """One marker returned by the official Microsoft detector."""

    type: Literal["Watermark", "C2PA"]
    provider: str | None
    model_name: str | None
    timestamp: str | None

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-safe marker using the provider's field names."""
        return {
            "type": self.type,
            "provider": self.provider,
            "modelName": self.model_name,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class MicrosoftProvenanceDetection:
    """One official Microsoft watermark and C2PA verdict."""

    status: MicrosoftWatermarkStatus
    provenance_status: MicrosoftProvenanceStatus
    operation_id: str
    markers: tuple[MicrosoftMarker, ...]
    raw_response: dict[str, object]
    detector: str = DETECTOR_ID
    signal_family: str = "invismark"
    provider_scope: str = "microsoft"
    backend: str = "official-microsoft-api"
    metadata_used_for_verdict: bool = False

    @property
    def detected(self) -> bool:
        """Whether Microsoft found one of its supported watermark signals."""
        return self.status == "detected"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe result while preserving the provider response."""
        return {
            "status": self.status,
            "provenance_status": self.provenance_status,
            "operation_id": self.operation_id,
            "markers": [marker.to_dict() for marker in self.markers],
            "raw_response": self.raw_response,
            "detector": self.detector,
            "signal_family": self.signal_family,
            "provider_scope": self.provider_scope,
            "backend": self.backend,
            "metadata_used_for_verdict": self.metadata_used_for_verdict,
        }


def _required_string(mapping: Mapping[str, Any], field: str, *, context: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{context} has an invalid {field!r} field")
    return value


def _optional_string(mapping: Mapping[str, Any], field: str, *, context: str) -> str | None:
    value = mapping.get(field)
    if value is None or isinstance(value, str):
        return value
    raise RuntimeError(f"{context} has an invalid {field!r} field")


def _json_mapping(response: httpx.Response, *, context: str) -> dict[str, object]:
    try:
        payload: object = response.json()
    except ValueError as error:
        raise RuntimeError(f"{context} returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} returned a non-object JSON payload")
    return cast("dict[str, object]", payload)


def _parse_marker(raw_marker: object) -> MicrosoftMarker:
    if not isinstance(raw_marker, Mapping):
        raise RuntimeError("Microsoft Content Provenance returned an invalid marker")
    marker = cast("Mapping[str, Any]", raw_marker)
    marker_type = marker.get("type")
    if marker_type not in {"Watermark", "C2PA"}:
        raise RuntimeError(f"Microsoft Content Provenance returned an unsupported marker type: {marker_type!r}")
    return MicrosoftMarker(
        type=cast("Literal['Watermark', 'C2PA']", marker_type),
        provider=_optional_string(marker, "provider", context="Microsoft marker"),
        model_name=_optional_string(marker, "modelName", context="Microsoft marker"),
        timestamp=_optional_string(marker, "timestamp", context="Microsoft marker"),
    )


def parse_detection(payload: Mapping[str, Any]) -> MicrosoftProvenanceDetection:
    """Parse a terminal Microsoft operation without conflating C2PA and watermark."""
    operation_id = _required_string(payload, "id", context="Microsoft Content Provenance response")
    if payload.get("status") != "Succeeded":
        raise RuntimeError(f"Microsoft Content Provenance operation did not succeed: {payload.get('status')!r}")
    raw_result = payload.get("result")
    if not isinstance(raw_result, Mapping):
        raise RuntimeError("Microsoft Content Provenance response has no result object")
    result = cast("Mapping[str, Any]", raw_result)
    outcome = result.get("outcome")
    if outcome == "NoProvenanceDetected":
        markers: tuple[MicrosoftMarker, ...] = ()
    elif outcome == "ProvenanceDetected":
        raw_markers = result.get("results")
        if not isinstance(raw_markers, list) or not raw_markers:
            raise RuntimeError("Microsoft Content Provenance reported provenance without markers")
        markers = tuple(_parse_marker(marker) for marker in cast("list[object]", raw_markers))
    else:
        raise RuntimeError(f"Microsoft Content Provenance returned an unsupported outcome: {outcome!r}")
    return MicrosoftProvenanceDetection(
        status="detected" if any(marker.type == "Watermark" for marker in markers) else "not_detected",
        provenance_status="present" if any(marker.type == "C2PA" for marker in markers) else "absent",
        operation_id=operation_id,
        markers=markers,
        raw_response=dict(cast("Mapping[str, object]", payload)),
    )


def _azure_cli_value(arguments: list[str], *, value_label: str) -> str:
    """Read one secret from Azure CLI without logging it."""
    executable = shutil.which("az")
    if executable is None:
        raise RuntimeError("Microsoft API verification needs Azure CLI; install it and run `az login`")
    completed = subprocess.run(  # noqa: S603
        [executable, *arguments], check=False, capture_output=True, text=True
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value:
        detail = completed.stderr.strip() or f"Azure CLI returned no {value_label}"
        raise RuntimeError(f"could not obtain {value_label}: {detail}")
    return value


def azure_content_safety_key(subscription_id: str, resource_group: str, account_name: str) -> str:
    """Read the Content Safety resource key into memory through Azure CLI."""
    return _azure_cli_value(
        [
            "cognitiveservices",
            "account",
            "keys",
            "list",
            "--name",
            account_name,
            "--resource-group",
            resource_group,
            "--subscription",
            subscription_id,
            "--query",
            "key1",
            "--output",
            "tsv",
        ],
        value_label="Azure Content Safety resource key",
    )


def _validate_https_url(value: str, *, label: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{label} must be an HTTPS URL without credentials, query, or fragment")
    return value


def _azure_blob_parts(content_uri: str) -> tuple[str, str, str]:
    """Parse the exact private Azure Blob URL accepted by the live API."""
    parsed = urlparse(_validate_https_url(content_uri, label="content_uri"))
    hostname = parsed.hostname or ""
    match = re.fullmatch(r"([a-z0-9]{3,24})\.blob\.core\.windows\.net", hostname)
    path_parts = parsed.path.removeprefix("/").split("/", maxsplit=1)
    if match is None or len(path_parts) != 2 or not all(path_parts):
        raise ValueError("content_uri must identify one blob on an Azure Blob Storage HTTPS endpoint")
    return match.group(1), unquote(path_parts[0]), unquote(path_parts[1])


def _local_sha256(path: Path) -> str:
    """Hash a local file without loading the complete media into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def azure_blob_sha256(subscription_id: str, storage_resource_group: str, content_uri: str) -> str:
    """Download one private Azure blob through CLI and hash its exact bytes."""
    account_name, container_name, blob_name = _azure_blob_parts(content_uri)
    storage_key = _azure_cli_value(
        [
            "storage",
            "account",
            "keys",
            "list",
            "--account-name",
            account_name,
            "--resource-group",
            storage_resource_group,
            "--subscription",
            subscription_id,
            "--query",
            "[0].value",
            "--output",
            "tsv",
        ],
        value_label="Azure Storage resource key",
    )
    executable = shutil.which("az")
    if executable is None:
        raise RuntimeError("Microsoft API verification needs Azure CLI; install it and run `az login`")
    with tempfile.TemporaryDirectory(prefix="microsoft-oracle-blob-") as directory:
        download = Path(directory) / "content"
        command = [
            executable,
            "storage",
            "blob",
            "download",
            "--account-name",
            account_name,
            "--container-name",
            container_name,
            "--name",
            blob_name,
            "--file",
            str(download),
            "--auth-mode",
            "key",
            "--subscription",
            subscription_id,
            "--no-progress",
            "--only-show-errors",
            "--output",
            "none",
        ]
        child_environment = os.environ.copy()
        child_environment["AZURE_STORAGE_KEY"] = storage_key
        completed = subprocess.run(  # noqa: S603
            command, check=False, capture_output=True, text=True, env=child_environment
        )
        if completed.returncode != 0 or not download.is_file():
            detail = completed.stderr.strip() or "Azure CLI did not download the blob"
            raise RuntimeError(f"could not read the Microsoft oracle blob: {detail}")
        if download.stat().st_size > MAX_MEDIA_BYTES:
            raise ValueError("Microsoft Content Provenance media exceeds the 100 MiB limit")
        digest = _local_sha256(download)
    log.info("Microsoft private blob preflight: uri=%s sha256=%s", content_uri, digest)
    return digest


def _poll_url(endpoint: str, operation_id: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F-]{36}", operation_id) is None:
        raise RuntimeError("Microsoft Content Provenance returned an invalid operation id")
    return f"{endpoint.rstrip('/')}/contentsafety/provenance/operations/{operation_id}?api-version={API_VERSION}"


def verify_microsoft_provenance(
    source: Path,
    *,
    content_uri: str,
    endpoint: str,
    subscription_id: str,
    resource_group: str,
    account_name: str,
    storage_resource_group: str,
    acknowledge_upload: bool = False,
    client: httpx.Client | None = None,
    key_loader: Callable[[str, str, str], str] = azure_content_safety_key,
    blob_hash_loader: Callable[[str, str, str], str] = azure_blob_sha256,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> MicrosoftProvenanceDetection:
    """Run one hash-bound Microsoft provenance operation with no submit retry."""
    if not acknowledge_upload:
        raise ValueError("Microsoft verification requires acknowledge_upload=True")
    if not source.is_file():
        raise ValueError(f"Microsoft verification source is not a file: {source}")
    if source.suffix.lower() not in {".jpeg", ".jpg", ".png", ".webp"}:
        raise ValueError("the current Microsoft API adapter supports JPEG, PNG, and WEBP images")
    content_uri = _validate_https_url(content_uri, label="content_uri")
    _azure_blob_parts(content_uri)
    endpoint = _validate_https_url(endpoint, label="endpoint").rstrip("/")
    configuration = {
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "account_name": account_name,
        "storage_resource_group": storage_resource_group,
    }
    for label, value in configuration.items():
        if not value.strip():
            raise ValueError(f"{label} must not be empty")

    if blob_hash_loader(subscription_id, storage_resource_group, content_uri) != _local_sha256(source):
        raise ValueError("content_uri bytes do not match the local source")
    resource_key = key_loader(subscription_id, resource_group, account_name)
    request_headers = {"Ocp-Apim-Subscription-Key": resource_key}
    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True, trust_env=False)
    try:
        submit_url = f"{endpoint}/contentsafety/provenance:detect?api-version={API_VERSION}"
        request_body = {"content": {"uri": content_uri}}
        log.info(
            "Microsoft Content Provenance request: POST %s headers={'Ocp-Apim-Subscription-Key': '<redacted>'} body=%s",
            submit_url,
            request_body,
        )
        submit = client.post(submit_url, headers=request_headers, json=request_body)
        log.info(
            "Microsoft Content Provenance response: status=%s headers=%s body=%s",
            submit.status_code,
            dict(submit.headers),
            submit.text,
        )
        if submit.status_code != 202:
            raise RuntimeError(f"Microsoft Content Provenance submit returned HTTP {submit.status_code}: {submit.text}")
        operation_id = _required_string(
            _json_mapping(submit, context="Microsoft Content Provenance submit"),
            "id",
            context="Microsoft Content Provenance submit",
        )
        poll_url = _poll_url(endpoint, operation_id)
        deadline = monotonic() + REQUEST_TIMEOUT_SECONDS
        while True:
            log.info(
                "Microsoft Content Provenance request: GET %s headers={'Ocp-Apim-Subscription-Key': '<redacted>'}",
                poll_url,
            )
            poll = client.get(poll_url, headers=request_headers)
            log.info(
                "Microsoft Content Provenance response: status=%s headers=%s body=%s",
                poll.status_code,
                dict(poll.headers),
                poll.text,
            )
            if poll.status_code != 200:
                raise RuntimeError(f"Microsoft Content Provenance poll returned HTTP {poll.status_code}: {poll.text}")
            payload = _json_mapping(poll, context="Microsoft Content Provenance poll")
            state = payload.get("status")
            if state in TERMINAL_OPERATION_STATES:
                return parse_detection(payload)
            if state not in {"NotStarted", "Running"}:
                raise RuntimeError(f"Microsoft Content Provenance returned an unsupported status: {state!r}")
            if monotonic() >= deadline:
                raise RuntimeError("Microsoft Content Provenance operation timed out")
            sleeper(POLL_INTERVAL_SECONDS)
    finally:
        if owns_client:
            client.close()
