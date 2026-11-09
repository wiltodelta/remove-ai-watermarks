#!/usr/bin/env python3
"""Build a publication-cleared, content-stratified real-image cohort.

The default source is the repository's publication-cleared, prompt-matched
OpenAI/Meta image matrix. Source files are hash-checked, deduplicated by content
digest, and standardized to square RGB PNG carriers before any watermark is
embedded. Derived artifacts stay under a new local output directory.

This is a development-only builder. It calls no provider API or provenance
oracle and does not download a corpus.

    uv run python scripts/watermark_benchmark_real_cohort.py \
      --output-dir .local-eval/watermark-benchmark-real-v1
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageOps

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import engine_selection_manifest as manifest_helpers  # noqa: E402
import watermark_benchmark_cohort as cohort_helpers  # noqa: E402
from engine_selection_manifest import (  # noqa: E402
    REUSE_BASIS as SOURCE_REUSE_BASIS,
)
from engine_selection_manifest import (  # noqa: E402
    ContentFixture as RealCarrier,
)
from engine_selection_manifest import (  # noqa: E402
    load_content_manifest as load_real_carriers,
)
from watermark_benchmark import load_manifest, sha256_file, write_jsonl  # noqa: E402
from watermark_benchmark_cohort import (  # noqa: E402
    DEFAULT_ATTACKS,
    DEFAULT_DWT_SCHEMES,
    AdapterName,
    AttackName,
    CarrierSpec,
    append_dwt_rows,
    append_trustmark_rows,
    append_videoseal_image_rows,
    benchmark_dependencies,
    validate_cohort_options,
)

log = logging.getLogger(__name__)

RECIPE_VERSION = "watermark-benchmark-real-cohort-v1"
DEFAULT_SOURCE_MANIFEST = REPO / "data" / "evaluations" / "engine-selection" / "content-manifest.csv"
REUSE_BASIS = SOURCE_REUSE_BASIS
DEFAULT_ADAPTERS: tuple[AdapterName, ...] = ("dwt-dct", "trustmark")
_ADAPTERS = frozenset(("dwt-dct", "trustmark", "videoseal-image"))


def _standardize(carrier: RealCarrier, *, size: int) -> Image.Image:
    with Image.open(carrier.path) as source:
        return ImageOps.fit(
            source.convert("RGB"),
            (size, size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )


def _source_metadata(
    carrier: RealCarrier,
    *,
    manifest_sha256: str,
    helper_sha256: str,
    manifest_helper_sha256: str,
    size: int,
) -> dict[str, Any]:
    return {
        "manifest_sha256": manifest_sha256,
        "manifest_helper_sha256": manifest_helper_sha256,
        "cohort_helper_sha256": helper_sha256,
        "source_sha256": carrier.sha256,
        "source_payload_sha256": carrier.source_payload_sha256,
        "source_pair_id": carrier.pair_id,
        "provider": carrier.provider,
        "model": carrier.model,
        "content_stratum": carrier.content_stratum,
        "source_size": [carrier.width, carrier.height],
        "source_format": "PNG",
        "reuse_basis": carrier.reuse_basis,
        "prompt_sha256": hashlib.sha256(carrier.prompt.encode()).hexdigest(),
        "standardization": {
            "mode": "center-crop-cover",
            "size": [size, size],
            "resampling": "lanczos",
            "output_mode": "RGB",
            "output_format": "PNG",
        },
    }


def build_real_cohort(
    output_dir: Path,
    *,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    adapters: Sequence[AdapterName] = DEFAULT_ADAPTERS,
    dwt_schemes: Sequence[str] = DEFAULT_DWT_SCHEMES,
    attacks: Sequence[AttackName] = DEFAULT_ATTACKS,
    size: int = 512,
) -> Path:
    """Build and validate a new real-image cohort directory."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    validate_cohort_options(adapters, dwt_schemes, attacks, size)

    carriers = load_real_carriers(source_manifest)
    too_small = [carrier.name for carrier in carriers if min(carrier.width, carrier.height) < size]
    if too_small:
        raise ValueError(f"source images cannot be upscaled to {size}px: {', '.join(too_small)}")

    output_dir.mkdir(parents=True)
    artifacts = output_dir / "artifacts"
    artifacts.mkdir()
    source_manifest = source_manifest.resolve()
    manifest_sha256 = sha256_file(source_manifest)
    recipe_revision = f"{RECIPE_VERSION}@sha256:{sha256_file(Path(__file__))}"
    helper_sha256 = sha256_file(Path(cohort_helpers.__file__))
    manifest_helper_sha256 = sha256_file(Path(manifest_helpers.__file__))
    helper_revision = f"watermark-benchmark-cohort-helpers@sha256:{helper_sha256}"
    manifest_helper_revision = f"engine-selection-manifest-helper@sha256:{manifest_helper_sha256}"
    source_revision = (
        f"engine-selection-content@sha256:{manifest_sha256}+{recipe_revision}+{helper_revision}"
        f"+{manifest_helper_revision}"
    )
    dependencies = benchmark_dependencies()
    prepared: list[tuple[CarrierSpec, Image.Image]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for carrier in carriers:
        image = _standardize(carrier, size=size)
        spec = CarrierSpec(carrier.name, carrier.seed)
        prepared.append((spec, image))
        metadata[carrier.name] = _source_metadata(
            carrier,
            manifest_sha256=manifest_sha256,
            helper_sha256=helper_sha256,
            manifest_helper_sha256=manifest_helper_sha256,
            size=size,
        )

    rows: list[dict[str, Any]] = []
    if "dwt-dct" in adapters:
        append_dwt_rows(
            rows,
            root=output_dir,
            artifacts=artifacts,
            carriers=prepared,
            schemes=dwt_schemes,
            attacks=attacks,
            source_revision=source_revision,
            dependencies=dependencies,
            size=size,
            carrier_metadata=metadata,
            clean_transform_name="real-carrier-standardize",
            clean_transform_revision=recipe_revision,
            attack_transform_revision=helper_revision,
        )
    if "trustmark" in adapters:
        append_trustmark_rows(
            rows,
            root=output_dir,
            artifacts=artifacts,
            carriers=prepared,
            attacks=attacks,
            source_revision=source_revision,
            dependencies=dependencies,
            size=size,
            carrier_metadata=metadata,
            clean_transform_name="real-carrier-standardize",
            clean_transform_revision=recipe_revision,
            attack_transform_revision=helper_revision,
        )
    if "videoseal-image" in adapters:
        append_videoseal_image_rows(
            rows,
            root=output_dir,
            artifacts=artifacts,
            carriers=prepared,
            attacks=attacks,
            source_revision=source_revision,
            dependencies=dependencies,
            size=size,
            carrier_metadata=metadata,
            clean_transform_name="real-carrier-standardize",
            clean_transform_revision=recipe_revision,
            attack_transform_revision=helper_revision,
        )

    manifest = output_dir / "manifest.jsonl"
    write_jsonl(manifest, rows)
    load_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new local cohort directory")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST,
        help="publication-cleared source CSV (default: tracked engine-selection content manifest)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="square carrier side, at least 256 and divisible by 16 (default: 512)",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        choices=tuple(sorted(_ADAPTERS)),
        dest="adapters",
        help="adapter to include; repeat to select both (default: both)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        manifest = build_real_cohort(
            args.output_dir,
            source_manifest=args.source_manifest,
            adapters=tuple(args.adapters or DEFAULT_ADAPTERS),
            size=args.size,
        )
    except (FileNotFoundError, FileExistsError, ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    log.info("Wrote validated real-image benchmark cohort to %s", manifest)


if __name__ == "__main__":
    main()
