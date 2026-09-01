"""Audit a private SynthID research manifest before training or evaluation.

The research manifest is intentionally separate from ``data/synthid/manifest.csv``.
The latter records a small public regression corpus, while this schema tracks
private experiment lineage, provider-specific oracle evidence, and split groups.

Usage:
    uv run python scripts/synthid_research_manifest.py MANIFEST.csv
    uv run python scripts/synthid_research_manifest.py MANIFEST.csv --verify-files
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import click
from PIL import Image

log = logging.getLogger(__name__)

FIELDNAMES = (
    "artifact_sha256",
    "pixel_sha256",
    "artifact_path",
    "parent_sha256",
    "group_id",
    "target_provider",
    "source_provider",
    "surface",
    "model_epoch",
    "generation_session",
    "content_stratum",
    "width",
    "height",
    "format",
    "transform",
    "split",
    "c2pa_outcome",
    "synthid_outcome",
    "verified_via",
    "evidence_reference",
    "oracle_session",
    "oracle_role",
    "captured_at",
    "oracle_checked_at",
    "notes",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TARGET_PROVIDERS = {"openai", "google"}
_SOURCE_PROVIDERS = {"openai", "google", "camera", "other_ai", "synthetic", "editor"}
_SPLITS = {"discovery", "train", "validation", "test", "temporal"}
_C2PA_OUTCOMES = {"detected", "not_detected", "invalid", "not_present", "not_checked"}
_SYNTHID_OUTCOMES = {"detected", "not_detected", "indeterminate", "refused", "not_checked"}
_VERIFIERS = {"openai-api", "openai-web", "gemini-app", "synthid-portal", "source-evidence", "none"}
_FORMATS = {"png", "jpeg", "webp"}
_ORACLE_ROLES = {"ordinary", "source_control", "candidate", "sham"}
_FINAL_SPLITS = {"train", "validation", "test", "temporal"}
_MATCHING_VERIFIERS = {
    "openai": {"openai-api", "openai-web"},
    "google": {"gemini-app", "synthid-portal"},
}


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read a CSV and return rows plus header errors."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        actual = tuple(reader.fieldnames or ())
        missing = [field for field in FIELDNAMES if field not in actual]
        errors = [f"header: missing required field {field!r}" for field in missing]
        return list(reader), errors


def _is_iso8601(value: str) -> bool:
    """Return whether VALUE is a timezone-aware ISO-8601 timestamp."""
    if not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _file_sha256(path: Path) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_image_fingerprint(path: Path) -> tuple[str, int, int, str]:
    """Return the decoded-RGB digest, geometry, and source format."""
    with Image.open(path) as image:
        image_format = (image.format or path.suffix.lstrip(".")).lower()
        rgb = image.convert("RGB")
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        return digest, rgb.width, rgb.height, "jpeg" if image_format == "jpg" else image_format


def _pixel_sha256(path: Path) -> tuple[str, int, int]:
    """Hash canonical decoded RGB pixels and return hash, width, and height."""
    digest, width, height, _ = decoded_image_fingerprint(path)
    return digest, width, height


def artifact_sha256(path: Path) -> str:
    """Return the artifact digest used by research manifests and inventories."""
    return _file_sha256(path)


def pixel_fingerprint(path: Path) -> tuple[str, int, int]:
    """Return the decoded-RGB digest and geometry used by manifest verification."""
    return _pixel_sha256(path)


def resolve_artifact_path(root: Path, value: str) -> Path | None:
    """Resolve a manifest-relative artifact path without allowing traversal."""
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _row_errors(row: dict[str, str], index: int) -> list[str]:
    """Validate one row without consulting other rows."""
    prefix = f"row {index}"
    errors: list[str] = []

    artifact_sha = row.get("artifact_sha256", "")
    pixel_sha = row.get("pixel_sha256", "")
    parent_sha = row.get("parent_sha256", "")
    if not _SHA256.fullmatch(artifact_sha):
        errors.append(f"{prefix}: invalid artifact_sha256")
    if not _SHA256.fullmatch(pixel_sha):
        errors.append(f"{prefix}: invalid pixel_sha256")
    if parent_sha and not _SHA256.fullmatch(parent_sha):
        errors.append(f"{prefix}: invalid parent_sha256")

    for field in ("group_id", "surface", "model_epoch", "generation_session", "content_stratum", "transform"):
        if not row.get(field, "").strip():
            errors.append(f"{prefix}: {field} must not be empty")

    target = row.get("target_provider", "")
    source = row.get("source_provider", "")
    split = row.get("split", "")
    c2pa = row.get("c2pa_outcome", "")
    synthid = row.get("synthid_outcome", "")
    verifier = row.get("verified_via", "")
    oracle_role = row.get("oracle_role", "")
    image_format = row.get("format", "").lower()

    if target not in _TARGET_PROVIDERS:
        errors.append(f"{prefix}: unsupported target_provider {target!r}")
    if source not in _SOURCE_PROVIDERS:
        errors.append(f"{prefix}: unsupported source_provider {source!r}")
    if split not in _SPLITS:
        errors.append(f"{prefix}: unsupported split {split!r}")
    if c2pa not in _C2PA_OUTCOMES:
        errors.append(f"{prefix}: unsupported c2pa_outcome {c2pa!r}")
    if synthid not in _SYNTHID_OUTCOMES:
        errors.append(f"{prefix}: unsupported synthid_outcome {synthid!r}")
    if verifier not in _VERIFIERS:
        errors.append(f"{prefix}: unsupported verified_via {verifier!r}")
    if image_format not in _FORMATS:
        errors.append(f"{prefix}: unsupported format {image_format!r}")
    if oracle_role not in _ORACLE_ROLES:
        errors.append(f"{prefix}: unsupported oracle_role {oracle_role!r}")

    for dimension in ("width", "height"):
        try:
            if int(row.get(dimension, "")) <= 0:
                raise ValueError
        except ValueError:
            errors.append(f"{prefix}: {dimension} must be a positive integer")

    if not _is_iso8601(row.get("captured_at", "")):
        errors.append(f"{prefix}: captured_at must be timezone-aware ISO-8601")

    final_outcome = synthid in {"detected", "not_detected"}
    if split in _FINAL_SPLITS and not final_outcome:
        errors.append(f"{prefix}: split {split!r} requires a detected or not_detected SynthID outcome")
    if final_outcome and not _is_iso8601(row.get("oracle_checked_at", "")):
        errors.append(f"{prefix}: a final SynthID outcome requires oracle_checked_at")

    matching = _MATCHING_VERIFIERS.get(target, set())
    if synthid == "detected" and verifier not in matching:
        errors.append(f"{prefix}: a detected {target!r} signal requires a matching provider verifier")
    if synthid == "not_detected" and source == target and verifier not in matching:
        errors.append(f"{prefix}: a same-provider negative requires a matching provider verifier")
    if verifier in {"source-evidence", "none"} and synthid == "detected":
        errors.append(f"{prefix}: {verifier!r} cannot establish a positive SynthID label")
    if verifier == "source-evidence" and source == target:
        errors.append(f"{prefix}: source-evidence cannot establish a same-provider negative")
    if verifier == "source-evidence" and not row.get("evidence_reference", "").strip():
        errors.append(f"{prefix}: source-evidence requires evidence_reference")
    if verifier in matching and not row.get("oracle_session", "").strip():
        errors.append(f"{prefix}: provider verification requires oracle_session")
    if oracle_role == "source_control" and synthid != "detected":
        errors.append(f"{prefix}: a source_control must have a detected SynthID outcome")

    transform = row.get("transform", "")
    if transform == "original" and parent_sha:
        errors.append(f"{prefix}: an original must not have parent_sha256")
    if transform != "original" and not parent_sha:
        errors.append(f"{prefix}: a derivative requires parent_sha256")

    return errors


def _lineage_errors(rows: list[dict[str, str]]) -> list[str]:
    """Validate uniqueness, parent links, group splits, and lineage cycles."""
    errors: list[str] = []
    by_sha: dict[str, dict[str, str]] = {}
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    pixel_groups: defaultdict[str, set[str]] = defaultdict(set)

    for index, row in enumerate(rows, start=2):
        artifact_sha = row.get("artifact_sha256", "")
        if artifact_sha in by_sha:
            errors.append(f"row {index}: duplicate artifact_sha256 {artifact_sha}")
        else:
            by_sha[artifact_sha] = row
        group_splits[row.get("group_id", "")].add(row.get("split", ""))
        pixel_groups[row.get("pixel_sha256", "")].add(row.get("group_id", ""))

    for index, row in enumerate(rows, start=2):
        parent_sha = row.get("parent_sha256", "")
        if not parent_sha:
            continue
        parent = by_sha.get(parent_sha)
        if parent is None:
            errors.append(f"row {index}: parent_sha256 is not present in the manifest")
            continue
        if parent.get("group_id") != row.get("group_id"):
            errors.append(f"row {index}: derivative and parent must share group_id")
        if parent.get("target_provider") != row.get("target_provider"):
            errors.append(f"row {index}: derivative and parent must share target_provider")

    for group_id, splits in sorted(group_splits.items()):
        if group_id and len(splits) > 1:
            errors.append(f"group {group_id!r}: leaks across splits {sorted(splits)}")
    for pixel_sha, groups in sorted(pixel_groups.items()):
        if pixel_sha and len(groups) > 1:
            errors.append(f"pixel_sha256 {pixel_sha}: appears in multiple groups {sorted(groups)}")

    for artifact_sha in by_sha:
        seen: set[str] = set()
        current_sha = artifact_sha
        while current_sha:
            if current_sha in seen:
                errors.append(f"artifact_sha256 {artifact_sha}: lineage cycle detected")
                break
            seen.add(current_sha)
            current = by_sha.get(current_sha)
            if current is None:
                break
            current_sha = current.get("parent_sha256", "")

    return errors


def _oracle_session_errors(rows: list[dict[str, str]]) -> list[str]:
    """Require a healthy source control before accepting removal outcomes."""
    errors: list[str] = []
    positive_controls = {
        (row.get("oracle_session", ""), row.get("group_id", ""), row.get("target_provider", ""))
        for row in rows
        if row.get("oracle_role") == "source_control" and row.get("synthid_outcome") == "detected"
    }
    for index, row in enumerate(rows, start=2):
        if row.get("oracle_role") not in {"candidate", "sham"} or row.get("synthid_outcome") != "not_detected":
            continue
        key = (row.get("oracle_session", ""), row.get("group_id", ""), row.get("target_provider", ""))
        if key not in positive_controls:
            errors.append(
                f"row {index}: a not_detected {row.get('oracle_role')} requires a detected "
                "source_control in the same oracle session, group, and provider"
            )
    return errors


def audit_manifest(path: Path, *, verify_files: bool = False) -> list[str]:
    """Return all manifest errors, including optional on-disk hash checks."""
    rows, errors = _read_rows(path)
    if errors:
        return errors
    for index, row in enumerate(rows, start=2):
        errors.extend(_row_errors(row, index))
    errors.extend(_lineage_errors(rows))
    errors.extend(_oracle_session_errors(rows))

    if verify_files:
        root = path.parent
        for index, row in enumerate(rows, start=2):
            artifact = resolve_artifact_path(root, row.get("artifact_path", ""))
            if artifact is None:
                errors.append(f"row {index}: artifact_path must be a safe manifest-relative path")
                continue
            if not artifact.is_file():
                errors.append(f"row {index}: artifact_path does not exist: {row.get('artifact_path', '')}")
                continue
            if _file_sha256(artifact) != row.get("artifact_sha256"):
                errors.append(f"row {index}: artifact_sha256 does not match the file")
            try:
                pixel_sha, width, height = _pixel_sha256(artifact)
            except Exception as exc:  # Pillow intentionally accepts many user-controlled formats.
                errors.append(f"row {index}: could not decode artifact: {exc}")
                continue
            if pixel_sha != row.get("pixel_sha256"):
                errors.append(f"row {index}: pixel_sha256 does not match decoded RGB pixels")
            if str(width) != row.get("width") or str(height) != row.get("height"):
                errors.append(f"row {index}: dimensions do not match decoded pixels")

    return errors


@click.command()
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--verify-files", is_flag=True, help="Verify artifact bytes, decoded pixels, and dimensions.")
def main(manifest: Path, verify_files: bool) -> None:
    """Audit MANIFEST for evidence, lineage, and split integrity."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    errors = audit_manifest(manifest, verify_files=verify_files)
    if errors:
        for error in errors:
            log.error("ERROR: %s", error)
        raise click.ClickException(f"manifest audit failed with {len(errors)} error(s)")
    log.info("Manifest audit passed: %s", manifest)


if __name__ == "__main__":
    main()
