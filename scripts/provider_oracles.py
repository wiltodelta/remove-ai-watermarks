#!/usr/bin/env python3
"""Prepare, run, record, and verify development-only provider oracle checks.

The provider services are not interchangeable. This tool gives their manual and
programmatic workflows one local, hash-bound record format without treating one
provider's negative result as evidence about another provider's watermark.

It never submits a remote request unless ``check-openai``, ``check-microsoft``,
or ``run-web`` is invoked with the explicit upload acknowledgement. Manual batches are written
outside the repository and remain unsubmitted until a maintainer uses the named
provider surface and records its verbatim response.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, cast

import click
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLOTS_PATH = REPOSITORY_ROOT / ".oracle-slots.json"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

log = logging.getLogger(__name__)

FORMAT_VERSION = 2
WATERMARK_RESULTS = {"detected", "not_detected", "indeterminate", "refused", "unreachable"}
PROVENANCE_RESULTS = {"present", "absent", "indeterminate", "unavailable"}

_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}
_VIDEO_SUFFIXES = {".mp4"}
_AUDIO_SUFFIXES = {".mp3", ".wav"}


@dataclass(frozen=True)
class OracleSurface:
    """Stable routing information for one provider-oracle surface."""

    key: str
    provider_key: str
    provider: str
    signal_family: str
    surface: str
    automation: Literal["api", "manual", "playwright", "real_browser"]
    url: str
    media_types: tuple[str, ...]
    project_evidence: tuple[str, ...]
    prompt: str | None
    clean_result_note: str
    account_slots: bool
    network_slots: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe catalog row."""
        return asdict(self)


class BatchRow(TypedDict):
    """One immutable media identity in a prepared batch."""

    artifact_id: str
    index: int
    source_path: str
    source_sha256: str
    upload_path: str
    upload_sha256: str
    media_type: str
    metadata_stripped: bool
    pixels_preserved: bool | None
    bytes_preserved: bool
    pixel_sha256: str | None
    width: int | None
    height: int | None
    format: str


class Preparation(TypedDict):
    """Integrity facts created while preparing one upload."""

    metadata_stripped: bool
    pixels_preserved: bool | None
    bytes_preserved: bool
    pixel_sha256: str | None
    width: int | None
    height: int | None
    format: str


class ResultRow(TypedDict):
    """One mutable external-oracle response bound to an immutable upload."""

    artifact_id: str
    upload_sha256: str
    watermark_result: str | None
    provenance_result: str | None
    raw_response: str | None
    checked_at: str | None


class ExecutionSlot(TypedDict):
    """A non-secret account, browser, and network routing label."""

    name: str
    surface: str
    account_label: str | None
    browser_profile: str | None
    google_account_index: NotRequired[int | None]
    network_label: str
    proxy_url_env: NotRequired[str | None]
    api_key_env: NotRequired[str | None]


class ManifestDocument(TypedDict):
    """Validated top-level immutable manifest fields used by this tool."""

    format_version: int
    status: str
    created_at: str
    provider: str
    surface: str
    oracle: dict[str, object]
    slot: ExecutionSlot | None
    row_count: int
    rows: list[BatchRow]


class ResultsDocument(TypedDict):
    """Validated top-level mutable result fields used by this tool."""

    format_version: int
    manifest_sha256: str
    provider: str
    surface: str
    rows: list[ResultRow]


SURFACES: dict[str, OracleSurface] = {
    "gemini-web": OracleSurface(
        key="gemini-web",
        provider_key="google",
        provider="Google",
        signal_family="synthid",
        surface="Gemini app verification",
        automation="real_browser",
        url="https://gemini.google.com/",
        media_types=("image", "video", "audio"),
        project_evidence=("image", "video"),
        prompt="Was this image/video/audio created or edited by Google AI?",
        clean_result_note="Use not_detected only for an explicit SynthID-negative result; preserve unclear results.",
        account_slots=True,
        network_slots=True,
    ),
    "meta-web": OracleSurface(
        key="meta-web",
        provider_key="meta",
        provider="Meta",
        signal_family="content_seal",
        surface="Meta AI identification",
        automation="playwright",
        url="https://www.meta.ai/identification/",
        media_types=("image", "video", "audio"),
        project_evidence=("image",),
        prompt=None,
        clean_result_note=(
            'Record the settled result after "Upload another file" appears; do not reuse stale page text.'
        ),
        account_slots=False,
        network_slots=True,
    ),
    "microsoft-api": OracleSurface(
        key="microsoft-api",
        provider_key="microsoft",
        provider="Microsoft",
        signal_family="invismark",
        surface="Azure Content Provenance Detection API",
        automation="api",
        url=(
            "https://learn.microsoft.com/en-us/rest/api/contentsafety/"
            "content-provenance-operations/detect?view=rest-contentsafety-2026-07-01-preview"
        ),
        media_types=("image", "video", "audio"),
        project_evidence=("image",),
        prompt=None,
        clean_result_note="Record the Watermark result separately from C2PA; the API reads Microsoft signals only.",
        account_slots=True,
        network_slots=True,
    ),
    "microsoft-web": OracleSurface(
        key="microsoft-web",
        provider_key="microsoft",
        provider="Microsoft",
        signal_family="invismark",
        surface="Content Provenance Detection",
        automation="playwright",
        url="https://ai.azure.com/nextgen/validate",
        media_types=("image", "video", "audio"),
        project_evidence=("image",),
        prompt=None,
        clean_result_note=(
            "The public surface renders a clean result as Inconclusive. Record Watermark separately from C2PA."
        ),
        account_slots=False,
        network_slots=True,
    ),
    "openai-api": OracleSurface(
        key="openai-api",
        provider_key="openai",
        provider="OpenAI",
        signal_family="synthid",
        surface="Content Provenance API",
        automation="api",
        url="https://developers.openai.com/api/reference/python/resources/content_provenance_checks/methods/create",
        media_types=("image", "audio"),
        project_evidence=("image",),
        prompt=None,
        clean_result_note=(
            "A not_detected result means no supported OpenAI SynthID was found; it is not a general AI verdict."
        ),
        account_slots=True,
        network_slots=True,
    ),
    "openai-web": OracleSurface(
        key="openai-web",
        provider_key="openai",
        provider="OpenAI",
        signal_family="synthid",
        surface="Content Provenance web verifier",
        automation="playwright",
        url="https://openai.com/verify",
        media_types=("image", "audio"),
        project_evidence=("image",),
        prompt=None,
        clean_result_note=(
            "Read SynthID separately from Content Credentials and preserve throttled or unclear responses."
        ),
        account_slots=False,
        network_slots=True,
    ),
}

PROVIDER_KEYS = frozenset(surface.provider_key for surface in SURFACES.values())


def _surface_order(surface: OracleSurface) -> tuple[bool, str]:
    """Sort API surfaces before Web surfaces, then stabilize by key."""
    return surface.automation != "api", surface.key


def provider_plan(provider_key: str) -> list[OracleSurface]:
    """Return one provider's surfaces in the API-first operator order."""
    if provider_key not in PROVIDER_KEYS:
        raise ValueError(f"unknown provider oracle: {provider_key}")
    surfaces = [surface for surface in SURFACES.values() if surface.provider_key == provider_key]
    return sorted(surfaces, key=_surface_order)


def ordered_surfaces() -> list[OracleSurface]:
    """Return the catalog with every API surface before every Web surface."""
    return sorted(SURFACES.values(), key=_surface_order)


def _inside(path: Path, parent: Path) -> bool:
    """Return whether resolved PATH is inside resolved PARENT."""
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    """Hash PATH without loading the complete media file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_fingerprint(path: Path) -> tuple[str, int, int, str]:
    """Return a bounded-memory RGBA pixel hash, width, height, and format."""
    with Image.open(path) as image:
        image.load()
        image_format = image.format
        if image_format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError(f"unsupported image format: {image_format or 'unknown'}")
        digest = hashlib.sha256()
        digest.update(f"{image.width}x{image.height}:RGBA\0".encode())
        for top in range(0, image.height, 128):
            bottom = min(top + 128, image.height)
            digest.update(image.crop((0, top, image.width, bottom)).convert("RGBA").tobytes())
        return digest.hexdigest(), image.width, image.height, image_format


def _media_type(path: Path) -> str:
    """Classify the conservative media formats used by the project oracles."""
    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in _VIDEO_SUFFIXES:
        return "video"
    if suffix in _AUDIO_SUFFIXES:
        return "audio"
    raise ValueError(f"unsupported oracle media extension: {suffix or '<none>'}")


def _aware_timestamp(value: str) -> datetime:
    """Parse a timezone-aware ISO-8601 timestamp."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("checked_at must be a timezone-aware ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("checked_at must be a timezone-aware ISO-8601 timestamp")
    return parsed


def _utc_now() -> str:
    """Return the current UTC timestamp in the manifest's canonical spelling."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_slots(path: Path) -> dict[str, ExecutionSlot]:
    """Load non-secret execution slots and reject ambiguous routing metadata."""
    raw_document: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_document, dict):
        raise ValueError("slot configuration must be a JSON object")
    document = cast("dict[str, object]", raw_document)
    if document.get("format_version") != 1:
        raise ValueError("unsupported slot configuration format version")
    raw_slots = document.get("slots")
    if not isinstance(raw_slots, list):
        raise ValueError("slot configuration must contain a slots array")
    slot_items = cast("list[object]", raw_slots)
    allowed_fields = {
        "name",
        "surface",
        "account_label",
        "browser_profile",
        "google_account_index",
        "network_label",
        "proxy_url_env",
        "api_key_env",
    }
    slots: dict[str, ExecutionSlot] = {}
    for index, raw_slot in enumerate(slot_items):
        if not isinstance(raw_slot, dict):
            raise ValueError(f"slot {index} must be a JSON object")
        slot = cast("dict[str, object]", raw_slot)
        unknown = set(slot) - allowed_fields
        if unknown:
            raise ValueError(f"slot {index} has unsupported fields: {', '.join(sorted(unknown))}")
        name = slot.get("name")
        surface_key = slot.get("surface")
        network_label = slot.get("network_label")
        account_label = slot.get("account_label")
        browser_profile = slot.get("browser_profile")
        google_account_index = slot.get("google_account_index")
        proxy_url_env = slot.get("proxy_url_env")
        api_key_env = slot.get("api_key_env")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"slot {index} needs a nonempty name")
        if not isinstance(surface_key, str) or surface_key not in SURFACES:
            raise ValueError(f"slot {name} names an unknown surface")
        if not isinstance(network_label, str) or not network_label.strip():
            raise ValueError(f"slot {name} needs a nonempty network_label")
        if account_label is not None and (not isinstance(account_label, str) or not account_label.strip()):
            raise ValueError(f"slot {name} has an invalid account_label")
        if browser_profile is not None and (not isinstance(browser_profile, str) or not browser_profile.strip()):
            raise ValueError(f"slot {name} has an invalid browser_profile")
        if google_account_index is not None and (
            isinstance(google_account_index, bool)
            or not isinstance(google_account_index, int)
            or google_account_index < 0
        ):
            raise ValueError(f"slot {name} has an invalid google_account_index")
        if proxy_url_env is not None and (
            not isinstance(proxy_url_env, str) or re.fullmatch(r"[A-Z][A-Z0-9_]*", proxy_url_env) is None
        ):
            raise ValueError(f"slot {name} has an invalid proxy_url_env")
        if api_key_env is not None and (
            not isinstance(api_key_env, str) or re.fullmatch(r"[A-Z][A-Z0-9_]*", api_key_env) is None
        ):
            raise ValueError(f"slot {name} has an invalid api_key_env")
        if SURFACES[surface_key].account_slots and account_label is None:
            raise ValueError(f"surface {surface_key} requires an account_label")
        if account_label is not None and not SURFACES[surface_key].account_slots:
            raise ValueError(f"surface {surface_key} does not use account slots")
        if surface_key == "gemini-web" and browser_profile is None:
            raise ValueError("surface gemini-web requires a browser_profile")
        if surface_key == "gemini-web" and google_account_index is None:
            raise ValueError("surface gemini-web requires a google_account_index")
        if surface_key != "gemini-web" and google_account_index is not None:
            raise ValueError("only gemini-web uses google_account_index")
        if SURFACES[surface_key].automation == "api" and browser_profile is not None:
            raise ValueError(f"API surface {surface_key} does not use a browser_profile")
        if surface_key != "gemini-web" and browser_profile is not None:
            raise ValueError("only gemini-web uses a real browser_profile")
        if surface_key == "gemini-web" and proxy_url_env is not None:
            raise ValueError("surface gemini-web does not use proxy_url_env")
        if proxy_url_env is not None and surface_key not in {"openai-web", "microsoft-web", "meta-web"}:
            raise ValueError(f"surface {surface_key} does not use proxy_url_env")
        if surface_key == "openai-api" and api_key_env is None:
            raise ValueError("surface openai-api requires an api_key_env")
        if surface_key != "openai-api" and api_key_env is not None:
            raise ValueError("only openai-api uses api_key_env")
        if name in slots:
            raise ValueError(f"duplicate slot name: {name}")
        slots[name] = {
            "name": name,
            "surface": surface_key,
            "account_label": account_label,
            "browser_profile": browser_profile,
            "google_account_index": google_account_index,
            "network_label": network_label,
            "proxy_url_env": proxy_url_env,
            "api_key_env": api_key_env,
        }
    return slots


def resolve_slot(surface: str, *, slots_path: Path | None, slot_name: str | None) -> ExecutionSlot | None:
    """Resolve one explicitly selected slot without automatic fallback."""
    if slot_name is None:
        if slots_path is not None:
            raise ValueError("--slots requires --slot")
        return None
    if slots_path is None:
        slots_path = DEFAULT_SLOTS_PATH
    slots = load_slots(slots_path)
    if slot_name not in slots:
        raise ValueError(f"unknown execution slot: {slot_name}")
    slot = slots[slot_name]
    if slot["surface"] != surface:
        raise ValueError(f"slot {slot_name} belongs to {slot['surface']}, not {surface}")
    return slot


def openai_api_key(
    slot: ExecutionSlot | None,
    *,
    env_file: Path,
    environ: Mapping[str, str] = os.environ,
) -> str:
    """Resolve one selected OpenAI key without logging or mutating the environment."""
    variable = "OPENAI_API_KEY" if slot is None else slot.get("api_key_env")
    if variable is None:
        raise ValueError("the selected openai-api slot has no api_key_env")
    value = secret_environment(env_file, environ=environ).get(variable)
    if value is not None:
        return value
    raise ValueError(f"OpenAI API key variable {variable} is missing from the environment and {env_file}")


def secret_environment(
    env_file: Path,
    *,
    environ: Mapping[str, str] = os.environ,
) -> dict[str, str]:
    """Read nonempty dotenv secrets with nonempty process values taking precedence."""
    values: dict[str, str] = {}
    if env_file.is_file():
        from dotenv import dotenv_values

        values.update(
            (name, value.strip())
            for name, value in dotenv_values(env_file).items()
            if isinstance(value, str) and value.strip()
        )
    values.update((name, value.strip()) for name, value in environ.items() if value.strip())
    return values


def microsoft_api_configuration(
    *,
    env_file: Path,
    environ: Mapping[str, str] = os.environ,
) -> tuple[str, str, str, str, str]:
    """Resolve the Azure resources used by Microsoft checks."""
    values = secret_environment(env_file, environ=environ)
    names = (
        "AZURE_CONTENT_SAFETY_ENDPOINT",
        "AZURE_CONTENT_SAFETY_SUBSCRIPTION_ID",
        "AZURE_CONTENT_SAFETY_RESOURCE_GROUP",
        "AZURE_CONTENT_SAFETY_ACCOUNT_NAME",
        "AZURE_ORACLE_STORAGE_RESOURCE_GROUP",
    )
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"{', '.join(missing)} missing from the environment and {env_file}")
    return cast("tuple[str, str, str, str, str]", tuple(values[name] for name in names))


def _write_json_new(path: Path, payload: object) -> None:
    """Write a JSON document while refusing to replace an existing artifact."""
    if path.exists():
        raise ValueError(f"refusing to overwrite oracle artifact: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _replace_json(path: Path, payload: object) -> None:
    """Atomically replace a mutable result document."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _prepare_image(source: Path, upload: Path) -> Preparation:
    """Strip AI metadata into UPLOAD and prove decoded pixels are unchanged."""
    source_pixels, width, height, source_format = _pixel_fingerprint(source)
    from remove_ai_watermarks.metadata import strip_and_verify

    stripped, remaining = strip_and_verify(source, upload, keep_standard=True)
    if remaining:
        fields = ", ".join(sorted(remaining))
        raise RuntimeError(f"refusing oracle batch because AI metadata survived stripping: {fields}")
    upload_pixels, upload_width, upload_height, upload_format = _pixel_fingerprint(stripped)
    if (upload_pixels, upload_width, upload_height, upload_format) != (
        source_pixels,
        width,
        height,
        source_format,
    ):
        raise RuntimeError("refusing oracle batch because metadata stripping changed the decoded pixels")
    return {
        "metadata_stripped": True,
        "pixels_preserved": True,
        "bytes_preserved": _sha256(source) == _sha256(upload),
        "pixel_sha256": source_pixels,
        "width": width,
        "height": height,
        "format": source_format.lower(),
    }


def _prepare_binary(source: Path, upload: Path) -> Preparation:
    """Copy video or audio bytes exactly without making a metadata claim."""
    shutil.copyfile(source, upload)
    return {
        "metadata_stripped": False,
        "pixels_preserved": None,
        "bytes_preserved": True,
        "pixel_sha256": None,
        "width": None,
        "height": None,
        "format": source.suffix.lower().removeprefix("."),
    }


def prepare_batch(
    surface: str,
    sources: list[Path],
    *,
    output_dir: Path,
    repository_root: Path = REPOSITORY_ROOT,
    slot: ExecutionSlot | None = None,
) -> Path:
    """Create a hash-bound, unsubmitted oracle batch outside the repository."""
    if surface not in SURFACES:
        raise ValueError(f"unknown provider oracle surface: {surface}")
    if slot is not None and slot["surface"] != surface:
        raise ValueError(f"slot {slot['name']} belongs to {slot['surface']}, not {surface}")
    if not sources:
        raise ValueError("at least one source is required")
    if _inside(output_dir, repository_root):
        raise ValueError("oracle batches must be written outside the repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must not already contain files")
    output_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir = output_dir / "uploads"
    uploads_dir.mkdir()

    oracle = SURFACES[surface]
    rows: list[BatchRow] = []
    source_hashes: set[str] = set()
    for index, raw_source in enumerate(sources):
        source = raw_source.resolve()
        if not source.is_file():
            raise ValueError(f"oracle source is not a file: {source}")
        media_type = _media_type(source)
        if media_type not in oracle.media_types:
            raise ValueError(f"{surface} tooling does not support {media_type} inputs")
        source_sha256 = _sha256(source)
        if source_sha256 in source_hashes:
            raise ValueError(f"duplicate source bytes: {source}")
        source_hashes.add(source_sha256)
        upload = uploads_dir / f"{index:03d}-{source_sha256[:16]}{source.suffix.lower()}"
        preparation = _prepare_image(source, upload) if media_type == "image" else _prepare_binary(source, upload)
        upload_sha256 = _sha256(upload)
        row: BatchRow = {
            "artifact_id": f"{index:03d}-{upload_sha256[:16]}",
            "index": index,
            "source_path": str(source),
            "source_sha256": source_sha256,
            "upload_path": str(upload.relative_to(output_dir)),
            "upload_sha256": upload_sha256,
            "media_type": media_type,
            **preparation,
        }
        rows.append(row)

    manifest: ManifestDocument = {
        "format_version": FORMAT_VERSION,
        "status": "unsubmitted",
        "created_at": _utc_now(),
        "provider": oracle.provider_key,
        "surface": surface,
        "oracle": oracle.to_dict(),
        "slot": slot,
        "row_count": len(rows),
        "rows": rows,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json_new(manifest_path, manifest)
    manifest_sha256 = _sha256(manifest_path)
    digest_path = output_dir / "manifest.sha256"
    digest_path.write_text(f"{manifest_sha256}  manifest.json\n", encoding="utf-8")
    results: ResultsDocument = {
        "format_version": FORMAT_VERSION,
        "manifest_sha256": manifest_sha256,
        "provider": oracle.provider_key,
        "surface": surface,
        "rows": [
            {
                "artifact_id": row["artifact_id"],
                "upload_sha256": row["upload_sha256"],
                "watermark_result": None,
                "provenance_result": None,
                "raw_response": None,
                "checked_at": None,
            }
            for row in rows
        ],
    }
    _write_json_new(output_dir / "results.json", results)
    verify_batch(manifest_path)
    return manifest_path


def load_batch(manifest_path: Path) -> tuple[ManifestDocument, ResultsDocument]:
    """Load and structurally validate one immutable manifest and result file."""
    batch_root = manifest_path.parent
    digest_parts = (batch_root / "manifest.sha256").read_text(encoding="utf-8").split()
    if not digest_parts or _sha256(manifest_path) != digest_parts[0]:
        raise ValueError("manifest hash mismatch")
    raw_manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_results: object = json.loads((batch_root / "results.json").read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict) or not isinstance(raw_results, dict):
        raise ValueError("oracle manifest and results must be JSON objects")
    manifest = cast("dict[str, object]", raw_manifest)
    results = cast("dict[str, object]", raw_results)
    if manifest.get("format_version") != FORMAT_VERSION or results.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported oracle batch format version")
    if manifest.get("provider") not in PROVIDER_KEYS:
        raise ValueError("manifest names an unknown provider")
    surface = manifest.get("surface")
    if not isinstance(surface, str) or surface not in SURFACES:
        raise ValueError("manifest names an unknown oracle surface")
    if SURFACES[surface].provider_key != manifest["provider"]:
        raise ValueError("manifest provider does not match its oracle surface")
    if (
        results.get("provider") != manifest["provider"]
        or results.get("surface") != surface
        or results.get("manifest_sha256") != digest_parts[0]
    ):
        raise ValueError("results do not identify this oracle manifest")
    raw_manifest_rows = manifest.get("rows")
    raw_result_rows = results.get("rows")
    if not isinstance(raw_manifest_rows, list):
        raise ValueError("manifest rows must be a JSON array")
    if not isinstance(raw_result_rows, list):
        raise ValueError("result rows must be a JSON array")
    manifest_rows = cast("list[object]", raw_manifest_rows)
    result_rows = cast("list[object]", raw_result_rows)
    if len(manifest_rows) != manifest.get("row_count"):
        raise ValueError("manifest row count mismatch")
    if len(result_rows) != len(manifest_rows):
        raise ValueError("results row count mismatch")
    if not all(isinstance(row, dict) for row in manifest_rows):
        raise ValueError("manifest rows must be JSON objects")
    if not all(isinstance(row, dict) for row in result_rows):
        raise ValueError("result rows must be JSON objects")
    return cast("ManifestDocument", manifest), cast("ResultsDocument", results)


def verify_batch(manifest_path: Path, *, require_complete: bool = False) -> dict[str, object]:
    """Validate batch hashes, result identities, and recorded verdicts."""
    manifest_path = manifest_path.resolve()
    if _inside(manifest_path.parent, REPOSITORY_ROOT):
        raise ValueError("oracle batches must remain outside the repository")
    manifest, results = load_batch(manifest_path)
    result_counts: Counter[str] = Counter()
    recorded = 0
    for expected, result in zip(manifest["rows"], results["rows"], strict=True):
        source = Path(expected["source_path"])
        upload = manifest_path.parent / expected["upload_path"]
        if _sha256(source) != expected["source_sha256"]:
            raise ValueError(f"source hash mismatch for {expected['artifact_id']}")
        if _sha256(upload) != expected["upload_sha256"]:
            raise ValueError(f"upload hash mismatch for {expected['artifact_id']}")
        identity = (result.get("artifact_id"), result.get("upload_sha256"))
        expected_identity = (expected["artifact_id"], expected["upload_sha256"])
        if identity != expected_identity:
            raise ValueError("result identity differs from the immutable manifest")
        watermark_result = result.get("watermark_result")
        if watermark_result is None:
            if any(result.get(field) is not None for field in ("provenance_result", "raw_response", "checked_at")):
                raise ValueError("partial oracle result row")
            continue
        if watermark_result not in WATERMARK_RESULTS:
            raise ValueError(f"invalid watermark result: {watermark_result!r}")
        if result.get("provenance_result") not in PROVENANCE_RESULTS:
            raise ValueError("invalid or missing provenance result")
        raw_response = result.get("raw_response")
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise ValueError("raw_response must preserve the nonempty verbatim oracle result")
        checked_at = result.get("checked_at")
        if not isinstance(checked_at, str):
            raise ValueError("checked_at must be a timezone-aware ISO-8601 timestamp")
        _aware_timestamp(checked_at)
        recorded += 1
        result_counts[watermark_result] += 1

    complete = recorded == len(manifest["rows"])
    if require_complete and not complete:
        raise ValueError(f"oracle batch is incomplete: {recorded}/{len(manifest['rows'])} results recorded")
    return {
        "provider": manifest["provider"],
        "surface": manifest["surface"],
        "total": len(manifest["rows"]),
        "recorded": recorded,
        "complete": complete,
        "watermark_results": dict(sorted(result_counts.items())),
    }


def record_result(
    manifest_path: Path,
    *,
    artifact_id: str,
    watermark_result: str,
    provenance_result: str,
    raw_response: str,
    checked_at: str,
) -> None:
    """Record one verbatim manual result while refusing to overwrite evidence."""
    verify_batch(manifest_path)
    if watermark_result not in WATERMARK_RESULTS:
        raise ValueError(f"invalid watermark result: {watermark_result!r}")
    if provenance_result not in PROVENANCE_RESULTS:
        raise ValueError(f"invalid provenance result: {provenance_result!r}")
    if not raw_response.strip():
        raise ValueError("raw_response must not be empty")
    _aware_timestamp(checked_at)
    _manifest, results = load_batch(manifest_path.resolve())
    matches = [row for row in results["rows"] if row.get("artifact_id") == artifact_id]
    if len(matches) != 1:
        raise ValueError(f"artifact_id must identify exactly one row: {artifact_id}")
    row = matches[0]
    if row.get("watermark_result") is not None:
        raise ValueError(f"artifact {artifact_id} already has a recorded result")
    row.update(
        {
            "watermark_result": watermark_result,
            "provenance_result": provenance_result,
            "raw_response": raw_response,
            "checked_at": checked_at,
        }
    )
    _replace_json(manifest_path.parent / "results.json", results)
    verify_batch(manifest_path)


def verify_openai(path: Path, *, acknowledge_upload: bool, api_key: str | None) -> object:
    """Call the hardened package adapter, kept as a seam for contract tests."""
    from remove_ai_watermarks.openai_provenance import verify_openai_synthid

    return verify_openai_synthid(path, acknowledge_upload=acknowledge_upload, api_key=api_key)


def verify_microsoft(
    path: Path,
    *,
    content_uri: str,
    endpoint: str,
    subscription_id: str,
    resource_group: str,
    account_name: str,
    storage_resource_group: str,
    acknowledge_upload: bool,
) -> object:
    """Call the direct Microsoft adapter, kept as a seam for contract tests."""
    from provider_oracle_microsoft import verify_microsoft_provenance

    return verify_microsoft_provenance(
        path,
        content_uri=content_uri,
        endpoint=endpoint,
        subscription_id=subscription_id,
        resource_group=resource_group,
        account_name=account_name,
        storage_resource_group=storage_resource_group,
        acknowledge_upload=acknowledge_upload,
    )


@click.group()
def cli() -> None:
    """Work with every provider watermark oracle used by the project."""


@cli.command("list")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def list_oracles(as_json: bool) -> None:
    """List provider scope, supported media, and verification surfaces."""
    definitions = ordered_surfaces()
    if as_json:
        click.echo(json.dumps([definition.to_dict() for definition in definitions], indent=2, sort_keys=True))
        return
    for definition in definitions:
        media = ",".join(definition.media_types)
        click.echo(f"{definition.key}\t{definition.automation}\t{media}\t{definition.surface}\t{definition.url}")


@cli.command()
@click.argument("provider", type=click.Choice(sorted(PROVIDER_KEYS)))
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def plan(provider: str, as_json: bool) -> None:
    """Show the API-first surface order for one provider."""
    definitions = provider_plan(provider)
    rows = []
    for index, definition in enumerate(definitions):
        row = definition.to_dict()
        row["preference"] = "primary" if index == 0 else "fallback"
        rows.append(row)
    if as_json:
        click.echo(json.dumps(rows, indent=2, sort_keys=True))
        return
    for row in rows:
        click.echo(f"{row['preference']}\t{row['key']}\t{row['automation']}\t{row['surface']}")
    if len(rows) > 1:
        click.echo("Web fallback requires a separate operator decision; failures do not trigger it automatically.")


@cli.command()
@click.argument("surface", type=click.Choice(sorted(SURFACES)))
@click.option("--slots", "slots_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--slot", "slot_name")
def guide(surface: str, slots_path: Path | None, slot_name: str | None) -> None:
    """Print the exact provider-specific workflow."""
    oracle = SURFACES[surface]
    slot = resolve_slot(surface, slots_path=slots_path, slot_name=slot_name)
    click.echo(f"Provider: {oracle.provider}")
    click.echo(f"Surface key: {oracle.key}")
    click.echo(f"Signal: {oracle.signal_family}")
    click.echo(f"Surface: {oracle.surface}")
    click.echo(f"URL: {oracle.url}")
    click.echo(f"Media: {', '.join(oracle.media_types)}")
    if oracle.prompt:
        click.echo(f"Prompt: {oracle.prompt}")
    click.echo(f"Result rule: {oracle.clean_result_note}")
    if slot is not None:
        google_account_index = slot.get("google_account_index")
        proxy_url_env = slot.get("proxy_url_env")
        api_key_env = slot.get("api_key_env")
        click.echo(f"Execution slot: {slot['name']}")
        click.echo(f"Account: {slot['account_label'] or 'not recorded'}")
        click.echo(f"Browser profile: {slot['browser_profile'] or 'not recorded'}")
        if google_account_index is not None:
            click.echo(f"Google account index: {google_account_index}")
        click.echo(f"Network: {slot['network_label']}")
        if proxy_url_env is not None:
            click.echo(f"Proxy environment variable: {proxy_url_env}")
        if api_key_env is not None:
            click.echo(f"OpenAI API key variable: {api_key_env}")
    if oracle.automation == "playwright":
        click.echo("Prepare a batch, then run it with run-web and an explicit upload acknowledgement.")
    elif oracle.automation == "real_browser":
        click.echo("Prepare a batch, then invoke the provider-oracles skill to drive the authenticated Chrome session.")
    elif surface == "microsoft-api":
        click.echo("Run check-microsoft with an HTTPS media URI whose bytes exactly match the local source.")
    else:
        click.echo("Prepare files first, submit one file at a time, and preserve the settled response verbatim.")


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def slots(config: Path, as_json: bool) -> None:
    """List validated account/browser/network execution slots."""
    loaded = load_slots(config)
    ordered = [loaded[name] for name in sorted(loaded)]
    if as_json:
        click.echo(json.dumps(ordered, indent=2, sort_keys=True))
        return
    for slot in ordered:
        google_account_index = slot.get("google_account_index")
        google_index_text = str(google_account_index) if google_account_index is not None else "-"
        click.echo(
            f"{slot['name']}\t{slot['surface']}\t{slot['account_label'] or '-'}\t"
            f"{slot['browser_profile'] or '-'}\t{google_index_text}\t"
            f"{slot['network_label']}\t{slot.get('proxy_url_env') or '-'}\t"
            f"{slot.get('api_key_env') or '-'}"
        )


@cli.command()
@click.argument("surface", type=click.Choice(sorted(SURFACES)))
@click.argument("sources", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--slots", "slots_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--slot", "slot_name")
def prepare(
    surface: str,
    sources: tuple[Path, ...],
    output: Path,
    slots_path: Path | None,
    slot_name: str | None,
) -> None:
    """Prepare an immutable provider-oracle batch outside the repository."""
    slot = resolve_slot(surface, slots_path=slots_path, slot_name=slot_name)
    manifest = prepare_batch(surface, list(sources), output_dir=output, slot=slot)
    click.echo(str(manifest))


@cli.command()
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--artifact-id", required=True)
@click.option("--watermark-result", required=True, type=click.Choice(sorted(WATERMARK_RESULTS)))
@click.option("--provenance-result", required=True, type=click.Choice(sorted(PROVENANCE_RESULTS)))
@click.option("--raw-response", required=True, help="Settled provider response, copied verbatim.")
@click.option("--checked-at", required=True, help="Timezone-aware ISO-8601 timestamp.")
def record(
    manifest: Path,
    artifact_id: str,
    watermark_result: str,
    provenance_result: str,
    raw_response: str,
    checked_at: str,
) -> None:
    """Record one manual result in the batch's mutable results file."""
    record_result(
        manifest,
        artifact_id=artifact_id,
        watermark_result=watermark_result,
        provenance_result=provenance_result,
        raw_response=raw_response,
        checked_at=checked_at,
    )
    click.echo(json.dumps(verify_batch(manifest), sort_keys=True))


@cli.command()
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--require-complete", is_flag=True, help="Fail when any row has no result.")
def verify(manifest: Path, require_complete: bool) -> None:
    """Verify immutable media and the recorded result document."""
    click.echo(json.dumps(verify_batch(manifest, require_complete=require_complete), sort_keys=True))


@cli.command("run-web")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--acknowledge-uploads",
    is_flag=True,
    help="Acknowledge uploads of every unrecorded manifest row to its named provider.",
)
@click.option("--headed/--headless", default=True, help="Show the isolated Playwright browser window.")
@click.option("--timeout", "timeout_seconds", type=click.FloatRange(min=1), default=120.0, show_default=True)
@click.option(
    "--env-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path(".env"),
    show_default=True,
    help="Read the selected proxy URL variable from this dotenv file.",
)
def run_web(
    manifest: Path,
    acknowledge_uploads: bool,
    headed: bool,
    timeout_seconds: float,
    env_file: Path,
) -> None:
    """Submit a prepared non-Google Web batch through isolated Playwright."""
    from provider_oracle_web import run_web_batch

    report = run_web_batch(
        manifest,
        acknowledge_uploads=acknowledge_uploads,
        headed=headed,
        timeout_seconds=timeout_seconds,
        env_file=env_file,
    )
    click.echo(json.dumps(report, sort_keys=True))


@cli.command("check-openai")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--acknowledge-upload",
    is_flag=True,
    help="Acknowledge exactly one metadata-stripped upload to OpenAI.",
)
@click.option("--slots", "slots_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--slot", "slot_name")
@click.option(
    "--env-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path(".env"),
    show_default=True,
    help="Read the selected OpenAI API key variable from this dotenv file.",
)
def check_openai(
    source: Path,
    acknowledge_upload: bool,
    slots_path: Path | None,
    slot_name: str | None,
    env_file: Path,
) -> None:
    """Run one official OpenAI SynthID check through the hardened adapter."""
    if not acknowledge_upload:
        raise click.UsageError("pass --acknowledge-upload to authorize exactly one OpenAI request")
    slot = resolve_slot("openai-api", slots_path=slots_path, slot_name=slot_name)
    api_key = openai_api_key(slot, env_file=env_file)
    result = verify_openai(source, acknowledge_upload=True, api_key=api_key)
    to_dict = getattr(result, "to_dict", None)
    if not callable(to_dict):
        raise RuntimeError("OpenAI oracle returned an unexpected result object")
    raw_payload: object = to_dict()
    if not isinstance(raw_payload, dict):
        raise RuntimeError("OpenAI oracle returned an unexpected result payload")
    payload = cast("dict[str, object]", raw_payload).copy()
    payload["surface"] = "openai-api"
    payload["execution_slot"] = slot
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@cli.command("check-microsoft")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--content-uri", required=True, help="Private Azure Blob URI whose bytes exactly match SOURCE.")
@click.option(
    "--acknowledge-upload",
    is_flag=True,
    help="Acknowledge exactly one Microsoft request that fetches the named URI.",
)
@click.option("--slots", "slots_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--slot", "slot_name")
@click.option(
    "--env-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path(".env"),
    show_default=True,
    help="Read the Azure Content Safety and Blob resource identities from this dotenv file.",
)
def check_microsoft(
    source: Path,
    content_uri: str,
    acknowledge_upload: bool,
    slots_path: Path | None,
    slot_name: str | None,
    env_file: Path,
) -> None:
    """Run one official Microsoft watermark and C2PA check."""
    if not acknowledge_upload:
        raise click.UsageError("pass --acknowledge-upload to authorize exactly one Microsoft request")
    slot = resolve_slot("microsoft-api", slots_path=slots_path, slot_name=slot_name)
    endpoint, subscription_id, resource_group, account_name, storage_resource_group = microsoft_api_configuration(
        env_file=env_file
    )
    result = verify_microsoft(
        source,
        content_uri=content_uri,
        endpoint=endpoint,
        subscription_id=subscription_id,
        resource_group=resource_group,
        account_name=account_name,
        storage_resource_group=storage_resource_group,
        acknowledge_upload=True,
    )
    to_dict = getattr(result, "to_dict", None)
    if not callable(to_dict):
        raise RuntimeError("Microsoft oracle returned an unexpected result object")
    raw_payload: object = to_dict()
    if not isinstance(raw_payload, dict):
        raise RuntimeError("Microsoft oracle returned an unexpected result payload")
    payload = cast("dict[str, object]", raw_payload).copy()
    payload["surface"] = "microsoft-api"
    payload["execution_slot"] = slot
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cli()
