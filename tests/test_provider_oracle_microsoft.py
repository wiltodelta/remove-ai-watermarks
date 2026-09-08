"""Contract tests for the direct Microsoft Content Provenance adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import httpx
import provider_oracle_microsoft as microsoft
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

CONTENT_URI = "https://storageacct.blob.core.windows.net/oracle-inputs/sample.png"
ENDPOINT = "https://content-safety.example.test"
OPERATION_ID = "12345678-1234-1234-1234-123456789abc"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_acknowledgement_is_required_before_media_or_key_access(tmp_clean_png: Path) -> None:
    with pytest.raises(ValueError, match="acknowledge_upload=True"):
        microsoft.verify_microsoft_provenance(
            tmp_clean_png,
            content_uri=CONTENT_URI,
            endpoint=ENDPOINT,
            subscription_id="subscription",
            resource_group="resource-group",
            account_name="content-safety",
            storage_resource_group="storage-resource-group",
        )


def test_remote_bytes_must_match_before_token_or_submit(tmp_clean_png: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"different bytes")

    def key_loader(_subscription_id: str, _resource_group: str, _account_name: str) -> str:
        pytest.fail("key must not be loaded for mismatched media")

    def blob_hash_loader(_subscription_id: str, _resource_group: str, _content_uri: str) -> str:
        return "different-hash"

    with _client(handler) as client, pytest.raises(ValueError, match="do not match"):
        microsoft.verify_microsoft_provenance(
            tmp_clean_png,
            content_uri=CONTENT_URI,
            endpoint=ENDPOINT,
            subscription_id="subscription",
            resource_group="resource-group",
            account_name="content-safety",
            storage_resource_group="storage-resource-group",
            acknowledge_upload=True,
            client=client,
            key_loader=key_loader,
            blob_hash_loader=blob_hash_loader,
        )

    assert requests == []


def test_private_blob_preflight_keeps_storage_key_out_of_process_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "test-storage-secret"

    def cli_value(_arguments: list[str], *, value_label: str) -> str:
        assert value_label == "Azure Storage resource key"
        return secret

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        assert secret not in command
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["AZURE_STORAGE_KEY"] == secret
        download = Path(command[command.index("--file") + 1])
        download.write_bytes(b"private blob bytes")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(microsoft, "_azure_cli_value", cli_value)
    monkeypatch.setattr(microsoft.shutil, "which", lambda _name: "/usr/bin/az")
    monkeypatch.setattr(microsoft.subprocess, "run", run)

    digest = microsoft.azure_blob_sha256("subscription", "resource-group", CONTENT_URI)

    assert digest == hashlib.sha256(b"private blob bytes").hexdigest()


@pytest.mark.parametrize(
    ("result", "expected_status", "expected_provenance"),
    [
        ({"outcome": "NoProvenanceDetected"}, "not_detected", "absent"),
        (
            {
                "outcome": "ProvenanceDetected",
                "results": [{"type": "C2PA", "provider": "Microsoft"}],
            },
            "not_detected",
            "present",
        ),
        (
            {
                "outcome": "ProvenanceDetected",
                "results": [
                    {
                        "type": "Watermark",
                        "provider": "Microsoft",
                        "modelName": "Image Creator",
                        "timestamp": "2026-09-08T00:00:00Z",
                    }
                ],
            },
            "detected",
            "absent",
        ),
    ],
)
def test_direct_check_separates_watermark_from_c2pa(
    tmp_clean_png: Path,
    result: dict[str, object],
    expected_status: str,
    expected_provenance: str,
) -> None:
    source_hash = hashlib.sha256(tmp_clean_png.read_bytes()).hexdigest()
    requests: list[httpx.Request] = []
    poll_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        requests.append(request)
        if request.method == "POST":
            assert json.loads(request.content) == {"content": {"uri": CONTENT_URI}}
            return httpx.Response(202, json={"id": OPERATION_ID})
        poll_count += 1
        state = "Running" if poll_count == 1 else "Succeeded"
        payload: dict[str, object] = {"id": OPERATION_ID, "status": state}
        if state == "Succeeded":
            payload["result"] = result
        return httpx.Response(200, json=payload)

    def key_loader(_subscription_id: str, _resource_group: str, _account_name: str) -> str:
        return "test-key"

    def blob_hash_loader(_subscription_id: str, _resource_group: str, _content_uri: str) -> str:
        return source_hash

    def sleeper(_seconds: float) -> None:
        return None

    with _client(handler) as client:
        detection = microsoft.verify_microsoft_provenance(
            tmp_clean_png,
            content_uri=CONTENT_URI,
            endpoint=ENDPOINT,
            subscription_id="subscription",
            resource_group="resource-group",
            account_name="content-safety",
            storage_resource_group="storage-resource-group",
            acknowledge_upload=True,
            client=client,
            key_loader=key_loader,
            blob_hash_loader=blob_hash_loader,
            sleeper=sleeper,
        )

    assert detection.status == expected_status
    assert detection.provenance_status == expected_provenance
    assert detection.metadata_used_for_verdict is False
    assert [request.method for request in requests] == ["POST", "GET", "GET"]
    for request in requests:
        assert request.headers["Ocp-Apim-Subscription-Key"] == "test-key"


@pytest.mark.parametrize(
    "payload",
    [
        {"id": OPERATION_ID, "status": "Failed"},
        {"id": OPERATION_ID, "status": "Succeeded", "result": {"outcome": "Unknown"}},
        {
            "id": OPERATION_ID,
            "status": "Succeeded",
            "result": {"outcome": "ProvenanceDetected", "results": [{"type": "Other"}]},
        },
    ],
)
def test_parser_rejects_failed_or_unknown_provider_results(payload: dict[str, object]) -> None:
    with pytest.raises(RuntimeError):
        microsoft.parse_detection(payload)


@pytest.mark.parametrize(
    ("content_uri", "endpoint"),
    [
        ("http://media.example.test/sample.png", ENDPOINT),
        ("https://user:password@media.example.test/sample.png", ENDPOINT),
        ("https://media.example.test/sample.png?token=secret", ENDPOINT),
        ("https://media.example.test/sample.png", ENDPOINT),
        (CONTENT_URI, "http://content-safety.example.test"),
    ],
)
def test_urls_require_https_without_credentials(
    tmp_clean_png: Path,
    content_uri: str,
    endpoint: str,
) -> None:
    with pytest.raises(ValueError, match=r"HTTPS URL|Azure Blob Storage"):
        microsoft.verify_microsoft_provenance(
            tmp_clean_png,
            content_uri=content_uri,
            endpoint=endpoint,
            subscription_id="subscription",
            resource_group="resource-group",
            account_name="content-safety",
            storage_resource_group="storage-resource-group",
            acknowledge_upload=True,
        )


def test_poll_timeout_uses_injected_clock_without_waiting(tmp_clean_png: Path) -> None:
    source_hash = hashlib.sha256(tmp_clean_png.read_bytes()).hexdigest()
    clock_values = iter([0.0, 121.0])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"id": OPERATION_ID})
        return httpx.Response(200, json={"id": OPERATION_ID, "status": "Running"})

    def key_loader(_subscription_id: str, _resource_group: str, _account_name: str) -> str:
        return "test-key"

    def blob_hash_loader(_subscription_id: str, _resource_group: str, _content_uri: str) -> str:
        return source_hash

    def sleeper(_seconds: float) -> None:
        pytest.fail("timeout must occur before sleeping")

    with _client(handler) as client, pytest.raises(RuntimeError, match="timed out"):
        microsoft.verify_microsoft_provenance(
            tmp_clean_png,
            content_uri=CONTENT_URI,
            endpoint=ENDPOINT,
            subscription_id="subscription",
            resource_group="resource-group",
            account_name="content-safety",
            storage_resource_group="storage-resource-group",
            acknowledge_upload=True,
            client=client,
            key_loader=key_loader,
            blob_hash_loader=blob_hash_loader,
            sleeper=sleeper,
            monotonic=lambda: next(clock_values),
        )
