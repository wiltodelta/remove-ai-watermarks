"""SynthID pixel-carrier probe -- EXPERIMENTAL / DIAGNOSTIC ONLY.

There is no local detector of the SynthID pixel watermark on real content: the
carrier drowns in scene texture (see CLAUDE.md, confirmed repeatedly). This
probe is meaningful ONLY on **solid-color fills**, where the per-pixel deviation
from the image mean is essentially the watermark carrier (almost all the
variance). It answers two controlled questions, neither of which is a
real-content detector:

  consistency IMAGES...
      Mean pairwise normalized cross-correlation (NCC) of the carriers across
      independent solid fills from one model, vs a random baseline. Genuine
      SynthID positives share a fixed carrier, so they correlate well above
      random (the pilot saw ~0.92 on gpt-image black fills); clean fills don't.

  removal --pos P... --cleaned C...
      Build a carrier template from the positive fills, then compare how the
      positives and the pipeline-cleaned fills correlate to it. If removal
      worked, the cleaned correlation collapses toward the random baseline --
      pixel-domain evidence that the pipeline destroys the carrier, not just the
      C2PA metadata.

Do NOT run this on real-content images; the numbers are uninformative there.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import click
import numpy as np
from _plain_console import Console
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)
console = Console()


def load_gray(path: str) -> NDArray[np.float64]:
    """Load an image as a float64 grayscale array (mean of RGB channels)."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float64).mean(axis=2)


def carrier(gray: NDArray[np.float64]) -> NDArray[np.float64]:
    """Zero-mean, unit-norm residual of a solid-fill image -- its carrier.

    Returns a flattened unit-norm vector for NCC comparison. A perfectly flat
    image (std 0, e.g. a synthetic #000000 reference) has no carrier and yields
    an all-zero vector, which correlates to 0 with everything.
    """
    residual = gray - float(gray.mean())
    norm = float(np.linalg.norm(residual))
    if norm == 0.0:
        return residual.ravel()
    return (residual / norm).ravel()


def ncc(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Normalized cross-correlation of two carriers (unit-norm zero-mean vectors)."""
    if a.shape != b.shape:
        raise ValueError("carrier geometry does not match")
    if a.size == 0:
        return 0.0
    return float(np.dot(a, b))


def mean_pairwise_ncc(carriers: list[NDArray[np.float64]]) -> float:
    """Average NCC over all distinct carrier pairs; 0.0 if fewer than two."""
    scores = [ncc(carriers[i], carriers[j]) for i in range(len(carriers)) for j in range(i + 1, len(carriers))]
    return float(np.mean(scores)) if scores else 0.0


def template(carriers: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Average carrier, renormalized to unit norm (the shared-pattern estimate)."""
    avg = np.mean(carriers, axis=0)
    norm = float(np.linalg.norm(avg))
    return avg / norm if norm else avg


def random_baseline(shape: tuple[int, ...], n: int, *, seed: int = 0) -> float:
    """Mean pairwise NCC of ``n`` random-noise carriers of ``shape`` (~0)."""
    rng = np.random.default_rng(seed)
    noise = [carrier(rng.standard_normal(shape)) for _ in range(max(n, 2))]
    return mean_pairwise_ncc(noise)


def _load_carriers(paths: tuple[str, ...]) -> list[NDArray[np.float64]]:
    """Load carriers only when every image has the same comparable geometry."""
    grays = [(p, load_gray(p)) for p in paths]
    shape = grays[0][1].shape
    carriers: list[NDArray[np.float64]] = []
    for p, g in grays:
        if g.shape != shape:
            raise click.ClickException(f"{p}: geometry {g.shape} does not match {shape}")
        carriers.append(carrier(g))
    return carriers


@click.group()
def cli() -> None:
    """SynthID pixel-carrier probe (solid-color fills only)."""


@cli.command()
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True))
def consistency(images: tuple[str, ...]) -> None:
    """Mean pairwise carrier NCC across solid fills, vs the random baseline."""
    carriers = _load_carriers(images)
    if len(carriers) < 2:
        console.print("[red]Need at least two same-shaped images.[/]")
        raise SystemExit(1)
    observed = mean_pairwise_ncc(carriers)
    baseline = random_baseline(carriers[0].shape, len(carriers))
    console.print(f"  carriers:          {len(carriers)}")
    console.print(f"  mean pairwise NCC: [bold]{observed:.3f}[/]")
    console.print(f"  random baseline:   {baseline:.3f}")
    verdict = "shared carrier present" if observed > 0.3 else "no shared carrier (within noise)"
    console.print(f"  verdict: [bold]{verdict}[/]")


@cli.command()
@click.option("--pos", "pos", multiple=True, required=True, type=click.Path(exists=True), help="Positive solid fills.")
@click.option(
    "--cleaned", "cleaned", multiple=True, required=True, type=click.Path(exists=True), help="Pipeline-cleaned fills."
)
def removal(pos: tuple[str, ...], cleaned: tuple[str, ...]) -> None:
    """Does the pipeline drop the carrier correlation toward the random baseline?"""
    carriers = _load_carriers(pos + cleaned)
    pos_carriers = carriers[: len(pos)]
    cleaned_carriers = carriers[len(pos) :]
    if not pos_carriers or not cleaned_carriers:
        console.print("[red]Need at least one positive and one cleaned fill of matching shape.[/]")
        raise SystemExit(1)
    tmpl = template(pos_carriers)
    pos_corr = float(np.mean([ncc(c, tmpl) for c in pos_carriers]))
    cleaned_corr = float(np.mean([ncc(c, tmpl) for c in cleaned_carriers]))
    baseline = random_baseline(tmpl.shape, max(len(cleaned_carriers), 2))
    console.print(f"  positive->template NCC: [bold]{pos_corr:.3f}[/]")
    console.print(f"  cleaned->template NCC:  [bold]{cleaned_corr:.3f}[/]")
    console.print(f"  random baseline:        {baseline:.3f}")
    effective = cleaned_corr < pos_corr / 2
    console.print(f"  verdict: [bold]{'carrier attenuated' if effective else 'carrier survives'}[/]")


if __name__ == "__main__":
    cli()
