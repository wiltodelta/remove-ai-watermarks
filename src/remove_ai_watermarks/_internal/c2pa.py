"""C2PA inspection through the official reader with a bounded PNG fallback."""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import re
import struct
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from remove_ai_watermarks._internal.constants import (
    C2PA_ACTIONS,
    C2PA_AI_TOOLS,
    C2PA_AI_VENDORS,
    C2PA_CHUNK_TYPE,
    C2PA_ISSUERS,
    C2PA_SIGNATURES,
    C2PA_SOFT_BINDING_REGISTRY,
    C2PA_SOFT_BINDINGS,
    PNG_SIGNATURE,
    C2paSoftBindingAlgorithm,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import BinaryIO

_C2paReader: Any = None
_C2paError: Any = None
with contextlib.suppress(Exception):
    from c2pa import C2paError as _C2paError  # pyright: ignore[reportMissingTypeStubs]
    from c2pa import Reader as _C2paReader  # pyright: ignore[reportMissingTypeStubs]

_C2PA_READER_AVAILABLE = _C2paReader is not None
_PNG_HEADER = struct.Struct(">I4s")

_CONTENT_BINDING_MATCHES = (
    "assertion.dataHash.match",
    "assertion.bmffHash.match",
    "assertion.boxesHash.match",
    "assertion.collectionHash.match",
    "assertion.multiAssetHash.match",
)
_CONTENT_BINDING_FAILURE_MARKERS = (
    "Hash.incorrectFileCount",
    "Hash.malformed",
    "Hash.mismatch",
    "Hash.missingPart",
    "assertion.hashedURI.mismatch",
    "claim.hardBindings.missing",
    "hashedUri.mismatch",
    "ingredient.manifest.mismatch",
)
_SIGNATURE_FAILURES = frozenset(
    {
        "claimSignature.mismatch",
        "claimSignature.missing",
        "claimSignature.outsideValidity",
    }
)
# A credential the issuer itself has disowned, disqualifying like a hash mismatch.
# ``signingCredential.expired`` is deliberately absent: an expired certificate does not
# imply the signed bytes changed, and a signature actually made outside validity already
# arrives as ``claimSignature.outsideValidity`` above.
_CREDENTIAL_FAILURES = frozenset(
    {
        "signingCredential.invalid",
        "signingCredential.ocsp.revoked",
    }
)


@dataclass(frozen=True)
class _PngChunk:
    payload: bytes
    serialized: bytes


def reader_available() -> bool:
    """Return whether the official C2PA reader loaded successfully."""
    return _C2PA_READER_AVAILABLE


def _manifest_json_uncached(path: str, *, strict: bool = False) -> str | None:
    """The manifest store as JSON, or None when this file has no readable manifest.

    Two outcomes are routine and stay at debug: a file with no manifest (``try_create``
    returns None) and a container the reader does not support. ANY other failure is
    logged at warning, because the caller cannot tell the difference from the return
    value and the consequence is severe: the verdict silently falls back to the raw
    byte scan and can lose a high-confidence signal. The log line preserves the
    diagnostic context needed to investigate an intermittent reader failure.
    """
    try:
        reader = _C2paReader.try_create(path)
    except _C2paError.NotSupported as error:
        logger.debug("C2PA reader does not support %s: %s", path, error)
        return None
    except Exception as error:
        logger.warning("C2PA reader failed to open %s: %s: %s", path, type(error).__name__, error)
        if strict:
            raise
        return None
    if reader is None:
        return None
    try:
        with reader:
            return cast("str", reader.json())
    except Exception as error:
        # The reader opened the file, so a manifest is there; failing to serialize it
        # is never routine.
        logger.warning("C2PA reader could not serialize %s: %s: %s", path, type(error).__name__, error)
        if strict:
            raise
        return None


@functools.lru_cache(maxsize=8)
def _manifest_json_cached(path: str, _mtime_ns: int, *, strict: bool = False) -> str | None:
    return _manifest_json_uncached(path, strict=strict)


def read_manifest_store_json(image_path: Path, *, strict: bool = False) -> str | None:
    """Read the manifest store, optionally raising unexpected reader failures.

    Strict reads have a separate cache key: a tolerant failure cached as ``None``
    must never become evidence that a later strict collection completed.
    """
    if not reader_available():
        return None
    path = str(image_path)
    try:
        mtime_ns = image_path.stat().st_mtime_ns
    except OSError:
        return _manifest_json_uncached(path, strict=strict)
    return _manifest_json_cached(path, mtime_ns, strict=strict)


def _find_c2pa_chunk(path: Path) -> _PngChunk | None:
    """Return the first recognizable C2PA chunk without loading the whole PNG."""
    try:
        stream = path.open("rb")
    except OSError:
        return None
    with stream:
        if stream.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            return None
        file_size = stream.seek(0, 2)
        stream.seek(len(PNG_SIGNATURE))
        while True:
            header = stream.read(_PNG_HEADER.size)
            if len(header) != _PNG_HEADER.size:
                return None
            length, kind = _PNG_HEADER.unpack(header)
            if length + 4 > file_size - stream.tell():
                return None
            if kind == C2PA_CHUNK_TYPE:
                payload = stream.read(length)
                crc = stream.read(4)
                if _looks_like_c2pa(payload):
                    return _PngChunk(payload, header + payload + crc)
            else:
                stream.seek(length + 4, 1)
            if kind == b"IEND":
                return None


def _is_well_formed_png(path: Path) -> bool:
    """Validate PNG chunk bounds with seeks rather than payload allocations."""
    try:
        stream = path.open("rb")
    except OSError:
        return False
    with stream:
        if stream.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            return False
        file_size = stream.seek(0, 2)
        stream.seek(len(PNG_SIGNATURE))
        while True:
            header = stream.read(_PNG_HEADER.size)
            if len(header) != _PNG_HEADER.size:
                return False
            length, kind = _PNG_HEADER.unpack(header)
            if length + 4 > file_size - stream.tell():
                return False
            stream.seek(length + 4, 1)
            if kind == b"IEND":
                return True


def _copy_bytes(source: BinaryIO, target: BinaryIO, byte_count: int) -> None:
    """Copy exactly one bounded chunk without allocating its complete payload."""
    remaining = byte_count
    while remaining:
        block = source.read(min(remaining, 1024 * 1024))
        if not block:
            raise OSError("PNG changed while it was being copied")
        target.write(block)
        remaining -= len(block)


def _looks_like_c2pa(payload: bytes) -> bool:
    lowered = payload.lower()
    return any(signature in payload for signature in C2PA_SIGNATURES) or b"c2pa" in lowered or b"jumb" in lowered


def extract_c2pa_chunk(image_path: Path) -> bytes | None:
    """Return the first complete C2PA PNG chunk, including header and CRC."""
    if image_path.suffix.casefold() != ".png":
        return None
    chunk = _find_c2pa_chunk(image_path)
    return None if chunk is None else chunk.serialized


def has_c2pa_metadata(image_path: Path) -> bool:
    """Return whether a validly bounded PNG contains a recognizable C2PA chunk."""
    return extract_c2pa_chunk(Path(image_path)) is not None


def _active_manifest(store: dict[str, Any]) -> dict[str, Any]:
    manifests = store.get("manifests")
    if not isinstance(manifests, dict):
        return {}
    typed_manifests = cast("dict[object, object]", manifests)
    active = typed_manifests.get(store.get("active_manifest"))
    return cast("dict[str, Any]", active) if isinstance(active, dict) else {}


def _manifest_chain(store: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the active manifest followed by credential-usable reachable ingredients."""
    manifests = store.get("manifests")
    active_label = store.get("active_manifest")
    if not isinstance(manifests, dict) or not isinstance(active_label, str):
        return []
    typed_manifests = cast("dict[object, object]", manifests)
    pending = deque([active_label])
    seen: set[str] = set()
    chain: list[dict[str, Any]] = []
    while pending:
        label = pending.popleft()
        if label in seen:
            continue
        seen.add(label)
        manifest_value = typed_manifests.get(label)
        if not isinstance(manifest_value, dict):
            continue
        manifest = cast("dict[str, Any]", manifest_value)
        chain.append(manifest)
        ingredients = manifest.get("ingredients")
        if not isinstance(ingredients, list):
            continue
        for ingredient_value in cast("list[object]", ingredients):
            if not isinstance(ingredient_value, dict):
                continue
            ingredient = cast("dict[object, object]", ingredient_value)
            child = ingredient.get("active_manifest")
            if (
                isinstance(child, str)
                and child not in seen
                and not _store_has_invalid_credential(cast("dict[str, Any]", ingredient))
            ):
                pending.append(child)
    return chain


def _status_codes(store: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Return ordered status codes for this store's active credential."""
    successes: list[str] = []
    informationals: list[str] = []
    failures: list[str] = []

    def extend(status_set: object) -> None:
        if not isinstance(status_set, dict):
            return
        mapping = cast("dict[object, object]", status_set)
        for key, target in (
            ("success", successes),
            ("informational", informationals),
            ("failure", failures),
        ):
            values = mapping.get(key)
            if not isinstance(values, list):
                continue
            for value in cast("list[object]", values):
                if not isinstance(value, dict):
                    continue
                code = cast("dict[object, object]", value).get("code")
                if isinstance(code, str) and code:
                    target.append(code)

    validation_results = store.get("validation_results")
    if isinstance(validation_results, dict):
        results = cast("dict[object, object]", validation_results)
        extend(results.get("activeManifest"))

    if not (successes or informationals or failures):
        # Older readers expose only the flat non-success list. It still carries the
        # exact codes needed to reject a mismatched binding or invalid credential.
        values = store.get("validation_status")
        if isinstance(values, list):
            for value in cast("list[object]", values):
                if not isinstance(value, dict):
                    continue
                code = cast("dict[object, object]", value).get("code")
                if isinstance(code, str) and code:
                    failures.append(code)
    return successes, informationals, failures


def _store_has_invalid_credential(store: dict[str, Any]) -> bool:
    """Return whether this store's own active credential failed validation.

    Deliberately routed through the same codes -> dimensions -> disqualified path the
    report uses, rather than re-testing the raw codes here. When this walk classified
    failures on its own, adding one rule meant editing it and
    :func:`c2pa_info_has_invalid_credential` in lockstep, and the ingredient
    reachability walk was free to drift away from the verdict it feeds.
    """
    return c2pa_info_has_invalid_credential(_validation_fields(store))


def c2pa_info_has_invalid_credential(info: dict[str, Any]) -> bool:
    """Return whether parsed C2PA info reports a failed binding, signature, or credential."""
    return (
        info.get("c2pa_integrity") == "invalid"
        or info.get("c2pa_signature") == "invalid"
        or info.get("c2pa_signer_validity") == "invalid"
    )


def c2pa_info_has_invismark(info: dict[str, Any]) -> bool:
    """Return whether parsed C2PA info declares Microsoft InvisMark."""
    soft_bindings = info.get("soft_binding_vendors")
    return isinstance(soft_bindings, list) and "Microsoft InvisMark" in soft_bindings


def c2pa_info_has_removal_hint(info: dict[str, Any]) -> bool:
    """Return whether a C2PA AI or watermark claim should keep removal fail-safe."""
    return bool(info.get("ai_source_kind") or info.get("synthid_watermark") or c2pa_info_has_invismark(info))


def _has_suffix(codes: list[str], suffixes: tuple[str, ...]) -> bool:
    return any(code.endswith(suffix) for code in codes for suffix in suffixes)


def _validation_fields(store: dict[str, Any]) -> dict[str, Any]:
    successes, informationals, failures = _status_codes(store)
    all_codes = list(dict.fromkeys([*successes, *informationals, *failures]))

    disqualifying = [
        code
        for code in failures
        if any(marker in code for marker in _CONTENT_BINDING_FAILURE_MARKERS)
        or code in _SIGNATURE_FAILURES
        or code in _CREDENTIAL_FAILURES
    ]
    binding_failed = any(marker in code for code in disqualifying for marker in _CONTENT_BINDING_FAILURE_MARKERS)
    binding_matched = _has_suffix(successes, _CONTENT_BINDING_MATCHES)
    signature_failed = any(code in _SIGNATURE_FAILURES for code in disqualifying)
    signature_validated = "claimSignature.validated" in successes

    if binding_failed:
        integrity = "invalid"
    elif binding_matched:
        integrity = "valid"
    else:
        integrity = "unknown"
    if signature_failed:
        signature = "invalid"
    elif signature_validated:
        signature = "valid"
    else:
        signature = "unknown"

    if "signingCredential.trusted" in successes:
        signer_trust = "trusted"
    elif "signingCredential.untrusted" in failures:
        signer_trust = "untrusted"
    else:
        signer_trust = "unknown"

    if any(code in _CREDENTIAL_FAILURES for code in disqualifying):
        signer_validity = "invalid"
    elif "signingCredential.expired" in failures:
        signer_validity = "expired"
    elif "claimSignature.insideValidity" in successes:
        signer_validity = "valid"
    else:
        signer_validity = "unknown"

    state = store.get("validation_state")
    return {
        "c2pa_validation_source": "reader",
        "c2pa_validation_state": state if isinstance(state, str) and state else "unknown",
        "c2pa_integrity": integrity,
        "c2pa_signature": signature,
        "c2pa_signer_trust": signer_trust,
        "c2pa_signer_validity": signer_validity,
        "c2pa_validation_codes": all_codes,
        # The subset of failures that actually drove a dimension to invalid. Callers
        # rendering "why" must use this rather than re-classifying the full code list,
        # or the reason shown and the verdict reached stop being the same rule.
        "c2pa_failed_codes": disqualifying,
    }


def _claim_generator_from_store(store: dict[str, Any]) -> str | None:
    active = _active_manifest(store)
    direct = active.get("claim_generator")
    if isinstance(direct, str) and direct.isprintable() and direct:
        return direct
    candidates = active.get("claim_generator_info")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        candidate = cast("dict[object, object]", candidates[0])
        name = candidate.get("name")
        if isinstance(name, str) and name.isprintable() and name:
            return name
    return None


def _structured_manifest_fields(store: dict[str, Any]) -> dict[str, Any]:
    """Extract provenance only from the active manifest and its ingredient graph."""
    chain = _manifest_chain(store)
    if not chain:
        return {}

    info: dict[str, Any] = {}
    issuers: list[str] = []
    tools: list[str] = []
    actions: list[str] = []
    source_types: list[str] = []
    soft_binding_algorithms: list[str] = []
    soft_binding_values: list[str] = []
    # Raw signer/generator identity strings, used to scope SynthID evidence to
    # the vendor that actually asserted the manifest. A vendor token appearing
    # anywhere else in the chain (e.g. Microsoft Designer's "Azure OpenAI
    # ImageGen" softwareAgent) is a service name, not that vendor's provenance.
    identity_strings: list[str] = []
    claim_generator_asserts_ai = False

    def add_tool_matches(value: str, *, asserts_ai: bool = False) -> None:
        nonlocal claim_generator_asserts_ai
        matches = _ordered_matches(value.encode(), C2PA_AI_TOOLS)
        tools.extend(matches)
        if asserts_ai and matches:
            claim_generator_asserts_ai = True

    for manifest in chain:
        signature_value = manifest.get("signature_info")
        if isinstance(signature_value, dict):
            signature = cast("dict[object, object]", signature_value)
            for key in ("issuer", "common_name", "certificate_issuer"):
                value = signature.get(key)
                if isinstance(value, str):
                    issuers.extend(_ordered_matches(value.encode(), C2PA_ISSUERS))
                    identity_strings.append(value)

        direct_generator = manifest.get("claim_generator")
        if isinstance(direct_generator, str):
            identity_strings.append(direct_generator)
            add_tool_matches(direct_generator, asserts_ai=True)

        candidates = manifest.get("claim_generator_info")
        if isinstance(candidates, list):
            for candidate_value in cast("list[object]", candidates):
                if not isinstance(candidate_value, dict):
                    continue
                name = cast("dict[object, object]", candidate_value).get("name")
                if isinstance(name, str):
                    add_tool_matches(name, asserts_ai=True)
                    identity_strings.append(name)

        assertions = manifest.get("assertions")
        if not isinstance(assertions, list):
            continue
        for assertion_value in cast("list[object]", assertions):
            if not isinstance(assertion_value, dict):
                continue
            assertion = cast("dict[object, object]", assertion_value)
            label = assertion.get("label")
            data = assertion.get("data")
            if isinstance(label, str) and label.startswith("c2pa.soft-binding") and isinstance(data, dict):
                soft_binding = cast("dict[object, object]", data)
                algorithm = soft_binding.get("alg")
                if isinstance(algorithm, str):
                    soft_binding_algorithms.append(algorithm)
                    blocks = soft_binding.get("blocks")
                    if isinstance(blocks, list):
                        for block_value in cast("list[object]", blocks):
                            if not isinstance(block_value, dict):
                                continue
                            value = cast("dict[object, object]", block_value).get("value")
                            if isinstance(value, str) and value.isprintable() and len(value) <= 256:
                                soft_binding_values.append(value)
            if not (isinstance(label, str) and label.startswith("c2pa.actions") and isinstance(data, dict)):
                continue
            action_values = cast("dict[object, object]", data).get("actions")
            if not isinstance(action_values, list):
                continue
            for action_value in cast("list[object]", action_values):
                if not isinstance(action_value, dict):
                    continue
                action = cast("dict[object, object]", action_value)
                action_name = action.get("action")
                if isinstance(action_name, str):
                    actions.append(C2PA_ACTIONS.get(action_name.encode(), action_name.removeprefix("c2pa.")))
                source_type = action.get("digitalSourceType")
                if isinstance(source_type, str):
                    source_types.append(source_type)
                software_agent = action.get("softwareAgent")
                if isinstance(software_agent, dict):
                    name = cast("dict[object, object]", software_agent).get("name")
                    if isinstance(name, str):
                        add_tool_matches(name)

    issuers = list(dict.fromkeys(issuers))
    tools = list(dict.fromkeys(tools))
    actions = list(dict.fromkeys(actions))
    if issuers:
        info["issuer"] = ", ".join(issuers)
    if tools:
        info["ai_tool"] = ", ".join(tools)
    if actions:
        info["actions"] = ", ".join(actions)
    if claim_generator_asserts_ai:
        info["c2pa_identity_ai"] = True

    generated = any(
        "trainedAlgorithmicMedia" in value and "compositeWithTrainedAlgorithmicMedia" not in value
        for value in source_types
    )
    enhanced = any("compositeWithTrainedAlgorithmicMedia" in value for value in source_types)
    procedural = any("algorithmicMedia" in value for value in source_types)
    if generated:
        info.update(source_type="trainedAlgorithmicMedia (AI-generated)", ai_source_kind="generated")
    elif enhanced:
        info.update(source_type="compositeWithTrainedAlgorithmicMedia (AI-enhanced)", ai_source_kind="enhanced")
    elif procedural:
        info["source_type"] = "algorithmicMedia"

    has_watermark_action = any(action.startswith("watermarked") for action in actions)
    if has_watermark_action:
        info["watermarked"] = True
    soft_binding_algorithms = list(dict.fromkeys(soft_binding_algorithms))
    algorithms = "\n".join(soft_binding_algorithms).encode()
    soft_binding_entries = soft_binding_registry_entries_in(algorithms)
    soft_binding_blocks_synthid = any(entry.kind == "watermark" for entry in soft_binding_entries) or len(
        soft_binding_entries
    ) < len(soft_binding_algorithms)
    if info.get("ai_source_kind") and not soft_binding_blocks_synthid:
        # Evidence scope: only the signer/generator identity strings above, never
        # the whole chain - a vendor named inside another vendor's manifest (the
        # Designer case) must not turn into that vendor's SynthID provenance. A
        # registered fingerprint does not suppress an independently established
        # SynthID watermark; a watermark or unknown algorithm does.
        identity_bytes = json.dumps(identity_strings, ensure_ascii=False).encode()
        synthid = synthid_evidence_vendors_in(identity_bytes, has_watermark_action=has_watermark_action)
        if synthid:
            info["synthid_vendors"] = synthid
            info["synthid_watermark"] = synthid_verdict(", ".join(synthid))

    if soft_binding_algorithms:
        soft_bindings = soft_binding_labels(soft_binding_entries)
        if soft_bindings:
            info["soft_binding_vendors"] = soft_bindings
            info["soft_binding"] = ", ".join(soft_bindings)
        info["soft_binding_algorithm"] = ", ".join(soft_binding_algorithms)
    if soft_binding_values:
        info["soft_binding_value"] = ", ".join(dict.fromkeys(soft_binding_values))
    return info


def synthid_verdict(vendors: str) -> str:
    """Describe why supported provenance establishes a SynthID watermark."""
    return f"present according to {vendors} provenance"


def synthid_evidence_vendors_in(buffer: bytes, *, has_watermark_action: bool | None = None) -> list[str]:
    """List issuers whose provenance establishes SynthID for this asset.

    Google applies SynthID to all media generated by its tools, so its AI C2PA
    provenance is sufficient. OpenAI C2PA predates OpenAI's SynthID rollout;
    current manifests distinguish the watermarked generation with the explicit
    ``c2pa.watermarked.*`` action. A legacy OpenAI issuer token alone therefore
    remains provenance evidence, but not SynthID evidence.
    """
    if has_watermark_action is None:
        has_watermark_action = b"c2pa.watermarked" in buffer
    return sorted(
        {
            vendor.org
            for vendor in C2PA_AI_VENDORS
            if vendor.synthid
            and vendor.issuer in buffer
            and (has_watermark_action or not vendor.synthid_requires_watermark_action)
        }
    )


def soft_binding_vendors_in(buffer: bytes) -> list[str]:
    """List the soft-binding algorithms named in manifest bytes."""
    return soft_binding_labels(soft_binding_registry_entries_in(buffer))


def soft_binding_registry_entries_in(buffer: bytes) -> tuple[C2paSoftBindingAlgorithm, ...]:
    """Return exact registered soft-binding entries named in manifest bytes."""
    return tuple(
        entry
        for entry in C2PA_SOFT_BINDING_REGISTRY
        if re.search(
            rb"(?<![A-Za-z0-9.-])" + re.escape(entry.algorithm.encode()) + rb"(?![A-Za-z0-9.-])",
            buffer,
        )
    )


def soft_binding_labels(entries: Iterable[C2paSoftBindingAlgorithm], *, kind: str | None = None) -> list[str]:
    """Return stable display labels for matching registry entries."""
    return sorted(
        {C2PA_SOFT_BINDINGS[entry.algorithm.encode()] for entry in entries if kind is None or entry.kind == kind}
    )


def _ordered_matches(buffer: bytes, registry: dict[bytes, str]) -> list[str]:
    return list(dict.fromkeys(label for token, label in registry.items() if token in buffer))


def _populate_registry_fields(buffer: bytes, info: dict[str, Any]) -> bool:
    issuers = _ordered_matches(buffer, C2PA_ISSUERS)
    tools = _ordered_matches(buffer, C2PA_AI_TOOLS)
    actions = _ordered_matches(buffer, C2PA_ACTIONS)
    if issuers:
        info["issuer"] = ", ".join(issuers)
    if tools:
        info["ai_tool"] = ", ".join(tools)
    if actions:
        info["actions"] = ", ".join(actions)

    ai_source = False
    if b"trainedAlgorithmicMedia" in buffer:
        info.update(source_type="trainedAlgorithmicMedia (AI-generated)", ai_source_kind="generated")
        ai_source = True
    elif b"compositeWithTrainedAlgorithmicMedia" in buffer:
        info.update(source_type="compositeWithTrainedAlgorithmicMedia (AI-enhanced)", ai_source_kind="enhanced")
        ai_source = True
    elif b"algorithmicMedia" in buffer:
        info["source_type"] = "algorithmicMedia"

    if b"c2pa.watermarked" in buffer:
        info["watermarked"] = True
    soft_binding_entries = soft_binding_registry_entries_in(buffer)
    soft_bindings = soft_binding_labels(soft_binding_entries)
    synthid = (
        []
        if any(entry.kind == "watermark" for entry in soft_binding_entries)
        else synthid_evidence_vendors_in(buffer, has_watermark_action=info.get("watermarked", False))
    )
    if ai_source and synthid:
        info["synthid_vendors"] = synthid
        info["synthid_watermark"] = synthid_verdict(", ".join(synthid))

    if soft_bindings:
        info["soft_binding_vendors"] = soft_bindings
        info["soft_binding"] = ", ".join(soft_bindings)
    return ai_source


def _base_info(byte_count: int, *, fallback: bool = False) -> dict[str, Any]:
    container = "C2PA manifest" if fallback else "C2PA manifest store"
    info: dict[str, Any] = {
        "has_c2pa": True,
        "type": "C2PA (Coalition for Content Provenance and Authenticity)",
        "c2pa_manifest": f"{container} ({byte_count} bytes)",
    }
    if fallback:
        info.update(
            c2pa_validation_source="fallback",
            c2pa_validation_state="unknown",
            c2pa_integrity="unknown",
            c2pa_signature="unknown",
            c2pa_signer_trust="unknown",
            c2pa_signer_validity="unknown",
            c2pa_validation_codes=[],
            c2pa_failed_codes=[],
        )
    return info


def _info_from_store(store: dict[str, Any], encoded: bytes) -> dict[str, Any]:
    info = _base_info(len(encoded))
    info.update(_structured_manifest_fields(store))
    info.update(_validation_fields(store))
    generator = _claim_generator_from_store(store)
    if generator is not None:
        info["claim_generator"] = generator
    signature_value = _active_manifest(store).get("signature_info")
    if isinstance(signature_value, dict):
        signature = cast("dict[object, object]", signature_value)
        timestamp = signature.get("time")
        if timestamp:
            info["timestamp"] = str(timestamp)
    return info


def c2pa_info_from_manifest_store(store: str | dict[str, Any]) -> dict[str, Any]:
    """Normalize a manifest store supplied as JSON text or a decoded object."""
    try:
        raw_decoded: object = store if isinstance(store, dict) else json.loads(store)
        decoded = cast("dict[str, Any]", raw_decoded) if isinstance(raw_decoded, dict) else None
        if not isinstance(decoded, dict) or not decoded or decoded.get("error"):
            return {}
        encoded = json.dumps(decoded, ensure_ascii=False).encode() if isinstance(store, dict) else store.encode()
    except (TypeError, ValueError):
        return {}
    return _info_from_store(decoded, encoded)


def cbor_text_after(payload: bytes, key: bytes) -> str | None:
    """Decode a definite-length CBOR text value immediately following ``key``."""
    key_end = payload.find(key)
    if key_end < 0:
        return None
    cursor = key_end + len(key)
    if cursor >= len(payload):
        return None
    initial = payload[cursor]
    if 0x60 <= initial <= 0x77:
        length, cursor = initial & 0x1F, cursor + 1
    elif initial == 0x78 and cursor + 1 < len(payload):
        length, cursor = payload[cursor + 1], cursor + 2
    elif initial == 0x79 and cursor + 2 < len(payload):
        length = int.from_bytes(payload[cursor + 1 : cursor + 3], "big")
        cursor += 3
    else:
        return None
    raw = payload[cursor : cursor + length]
    if len(raw) != length:
        return None
    try:
        return raw.decode()
    except UnicodeDecodeError:
        return raw.decode("latin1", errors="replace")


def _parse_c2pa_chunk(payload: bytes, info: dict[str, Any]) -> None:
    info.update(_base_info(len(payload), fallback=True))
    _populate_registry_fields(payload, info)
    for key, output_key in ((b"name", "claim_generator"), (b"specVersion", "c2pa_spec")):
        value = cbor_text_after(payload, key)
        if value and value.isprintable():
            info[output_key] = value
    timestamps = [item.decode() for item in re.findall(rb"\d{14}Z", payload)]
    if timestamps:
        info["timestamp"] = timestamps[0]
        if len(timestamps) > 1:
            info["timestamps"] = timestamps[:3]


def _extract_c2pa_info_png(image_path: Path) -> dict[str, Any]:
    if image_path.suffix.casefold() != ".png":
        return {}
    chunk = _find_c2pa_chunk(image_path)
    if chunk is None:
        return {}
    info: dict[str, Any] = {}
    _parse_c2pa_chunk(chunk.payload, info)
    return info


def _extract_c2pa_info_impl(image_path: Path) -> dict[str, Any]:
    store = read_manifest_store_json(Path(image_path))
    if store is not None:
        return c2pa_info_from_manifest_store(store)
    return _extract_c2pa_info_png(Path(image_path))


@functools.lru_cache(maxsize=4)
def _extract_c2pa_info_cached(path_str: str, _mtime_ns: int, _size: int, _reader: bool) -> dict[str, Any]:
    """Cache shim: every argument after the path is key-only.

    ``_reader`` is in the key because the answer genuinely depends on it -- with the
    official reader the manifest comes back as a store, without it from the hand-rolled
    PNG chunk parser, and the two produce different ``c2pa_manifest`` labels. Keying on
    the file alone handed a reader-path result to a caller that had disabled the reader.
    """
    return _extract_c2pa_info_impl(Path(path_str))


def extract_c2pa_info(image_path: Path) -> dict[str, Any]:
    """Return normalized C2PA evidence from the official reader or PNG fallback.

    Memoized on ``(path, mtime_ns, size)``: one ``identify`` reaches this twice (once
    directly, once inside ``get_ai_metadata``) and each call re-runs the Rust manifest
    reader and re-parses its JSON. Size joins mtime in the key because this package
    rewrites files in place, and an in-place rewrite can land inside one mtime tick.
    """
    try:
        stat = image_path.stat()
    except OSError:
        # No stat (a pipe, or a race): read uncached rather than fail.
        return _extract_c2pa_info_impl(image_path)
    cached = _extract_c2pa_info_cached(str(image_path), stat.st_mtime_ns, stat.st_size, _C2PA_READER_AVAILABLE)
    # Deep-ish copy: values are scalars plus a few lists, and a caller mutating one of
    # those lists would otherwise poison every later reader of the same file.
    return {key: list(cast("list[Any]", value)) if isinstance(value, list) else value for key, value in cached.items()}


def inject_c2pa_chunk(target_path: Path, output_path: Path, c2pa_chunk: bytes) -> None:
    """Replace any C2PA chunks in a PNG and insert ``c2pa_chunk`` before IDAT."""
    if target_path.suffix.casefold() != ".png" or output_path.suffix.casefold() != ".png":
        raise ValueError("C2PA chunk injection is only supported for PNG files")
    if not _is_well_formed_png(target_path):
        raise ValueError("Target is not a well-formed PNG file")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("rb") as source, output_path.open("wb") as target:
        target.write(source.read(len(PNG_SIGNATURE)))
        inserted = False
        while True:
            header = source.read(_PNG_HEADER.size)
            length, kind = _PNG_HEADER.unpack(header)
            if kind == b"IDAT" and not inserted:
                target.write(c2pa_chunk)
                inserted = True
            if kind == C2PA_CHUNK_TYPE:
                source.seek(length + 4, 1)
            else:
                target.write(header)
                _copy_bytes(source, target, length + 4)
            if kind == b"IEND":
                break
