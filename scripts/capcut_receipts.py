"""Offline checks of captured CapCut supported-signal receipts, not a decoder."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode, urlsplit


def receipt_verdict(receipt: Any, artifact: dict[str, Any]) -> str:
    """Bind a settled response to one captured upload of the expected video.

    These are auditable local records, not signed attestations from CapCut.
    Empty signal details never identify a watermark family.
    """
    try:
        return _receipt_verdict(receipt, artifact)
    except (KeyError, TypeError, ValueError, AttributeError):
        return "unverified"


def _receipt_verdict(receipt: dict[str, Any], artifact: dict[str, Any]) -> str:
    if receipt.get("error") or receipt.get("http_status") != 200:
        return "unverified"
    capture = receipt["capture"]
    requests, responses = capture["requests"], capture["responses"]
    if capture["errors"] or capture["fileUploads"] != 1 or not requests or len(requests) != len(responses):
        return "unverified"
    filename, size, digest = receipt["filename"], artifact["size_bytes"], artifact["sha256"]
    if (
        receipt["sha256"] != digest
        or receipt["size_bytes"] != size
        or receipt["media_type"] != "video"
        or not isinstance(filename, str)
        or not filename
    ):
        return "unverified"
    job = None
    final: dict[str, Any] | None = None
    for index, (request, response) in enumerate(zip(requests, responses, strict=True)):
        url = request["url"]
        if not isinstance(url, str):
            return "unverified"
        endpoint = urlsplit(url)
        if request["method"] != "POST" or endpoint.path != "/open/v1/detect":
            return "unverified"
        if endpoint.netloc and (endpoint.scheme, endpoint.netloc) != ("https", "ai-generated-detect.capcut.com"):
            return "unverified"
        if index == 0:
            fields = request["fields"]
            selected = fields["file"]
            if (
                request["body_kind"] != "multipart"
                or fields["media_type"] != "video"
                or request["selected_file_sha256"] != digest
                or (selected["sha256"], selected["filename"], selected["size"], selected["type"])
                != (digest, filename, size, "video/mp4")
            ):
                return "unverified"
        elif request["body_kind"] != "job-only-poll" or not job or request["body_text"] != urlencode({"job_id": job}):
            return "unverified"
        body = json.loads(response["response_text"])
        if index < len(responses) - 1:
            if response["http_status"] != 202 or body["status"] != "pending":
                return "unverified"
            next_job = body["job_id"]
            if not isinstance(next_job, str) or not next_job or (job is not None and job != next_job):
                return "unverified"
            job = next_job
        else:
            if response["http_status"] != 200 or body["status"] != "completed" or body.get("error_code"):
                return "unverified"
            final = body
    if not isinstance(final, dict):
        return "unverified"
    exchange = receipt["exchange"]
    if (
        exchange["job_id"] != job
        or exchange["final"]["status"] != 200
        or json.loads(exchange["final"]["text"]) != final
        or receipt["response"] != final
        or receipt.get("owned_tab_left_open_for_pending_job")
    ):
        return "unverified"
    info, display = final["file_info"], final["display"]
    if (info["filename"], info["size_bytes"], info["media_type"]) != (filename, size, "video"):
        return "unverified"
    detected = final["signal_detected"]
    if type(detected) is not bool:
        return "unverified"
    state = display["state"]
    verdict = "unverified"
    if detected and isinstance(state, str) and state.startswith("detected"):
        verdict = "detected"
    elif not detected and state == "not_detected":
        verdict = "not_detected"
    return verdict if receipt["verdict"] == verdict else "unverified"
