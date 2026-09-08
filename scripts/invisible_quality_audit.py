"""Audit invisible-removal output quality by pairing originals with cleaned outputs.

The private corpus pull writes ``<hash>_src.<ext>`` originals and ``<hash>_clean.<ext>``
cleaned outputs. For each pair this computes a structural-similarity score plus
cheap content proxies (detail via Laplacian variance, resolution, aspect), so the
WORST-preserved images can be surfaced and then visually classified.

SSIM alone does NOT equal "bad": a high-texture image legitimately changes under
the SDXL scrub. Use the ranked output to pick candidates, then look at them to
name the failure classes (garbled text, deformed faces, over-smoothed detail).

Operates on gitignored local data only; writes nothing tracked.

    uv run python scripts/invisible_quality_audit.py \
        --originals .local-eval/originals \
        --cleaned .local-eval/results \
        --out .local-eval/quality-audit.csv --worst 25
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import click
import cv2
import numpy as np

from remove_ai_watermarks import image_io

log = logging.getLogger(__name__)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Grayscale SSIM (single-window Gaussian, the Wang et al. formulation)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mu_a = cv2.GaussianBlur(a, k, 1.5)
    mu_b = cv2.GaussianBlur(b, k, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = cv2.GaussianBlur(a * a, k, 1.5) - mu_a2
    sb = cv2.GaussianBlur(b * b, k, 1.5) - mu_b2
    sab = cv2.GaussianBlur(a * b, k, 1.5) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * sab + c2)) / ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))
    return float(ssim_map.mean())


def _stem(name: str) -> str:
    """Strip the _src/_clean suffix and extension to get the pairing key."""
    base = name.rsplit(".", 1)[0]
    for suf in ("_src", "_clean"):
        if base.endswith(suf):
            return base[: -len(suf)]
    return base


@click.command()
@click.option("--originals", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--cleaned", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), default=Path(".local-eval/quality-audit.csv"))
@click.option("--worst", type=int, default=25, help="Print the N lowest-SSIM pairs.")
def main(originals: Path, cleaned: Path, out: Path, worst: int) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    src_by_key = {_stem(p.name): p for p in originals.iterdir() if p.is_file()}
    clean_by_key = {_stem(p.name): p for p in cleaned.iterdir() if p.is_file()}
    keys = sorted(src_by_key.keys() & clean_by_key.keys())
    click.echo(f"Pairs: {len(keys)} ({len(src_by_key)} src, {len(clean_by_key)} clean)")

    rows: list[dict[str, str]] = []
    with click.progressbar(keys, label="ssim") as bar:
        for key in bar:
            a = image_io.imread(src_by_key[key], cv2.IMREAD_COLOR)
            b = image_io.imread(clean_by_key[key], cv2.IMREAD_COLOR)
            if a is None or b is None:
                continue
            if a.shape[:2] != b.shape[:2]:
                b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            h, w = ga.shape
            rows.append(
                {
                    "key": key,
                    "ssim": f"{_ssim(ga, gb):.4f}",
                    "laplacian_var": f"{cv2.Laplacian(ga, cv2.CV_64F).var():.1f}",
                    "width": str(w),
                    "height": str(h),
                    "megapixels": f"{w * h / 1e6:.2f}",
                    "aspect": f"{w / h:.2f}",
                }
            )

    rows.sort(key=lambda r: float(r["ssim"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["key", "ssim", "laplacian_var", "width", "height", "megapixels", "aspect"])
        wtr.writeheader()
        wtr.writerows(rows)

    ssims = [float(r["ssim"]) for r in rows]
    if ssims:
        arr = np.array(ssims)
        click.echo(f"\nSSIM mean={arr.mean():.3f} p10={np.percentile(arr, 10):.3f} min={arr.min():.3f}")
        click.echo(f"Pairs with SSIM < 0.70: {(arr < 0.70).sum()}  | < 0.60: {(arr < 0.60).sum()}")
        click.echo(f"\nWorst {worst} (lowest SSIM):")
        for r in rows[:worst]:
            click.echo(
                f"  ssim={r['ssim']}  lap={r['laplacian_var']:>8}  {r['megapixels']}MP {r['aspect']}  {r['key']}"
            )
    click.echo(f"\nReport: {out}")


if __name__ == "__main__":
    main()
