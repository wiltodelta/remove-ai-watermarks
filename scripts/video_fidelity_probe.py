"""End-to-end fidelity of a DELIVERED video against its source.

The engine's own ``psnr_db`` is measured against the already-resized frame and
before the encoder, so it reports the VAE round trip plus latent noise and
nothing else. The downscale, the frame decimation and the H.264 encode -- the
three steps that actually cost the user picture -- are invisible to it, and no
metric computed inside the streaming loop can see them, because the candidate
frame is captured before it is written to the encoder pipe.

This script measures what the engine cannot: it decodes the delivered file after
muxing, upscales each frame back to the source geometry, and scores it against
the untouched source frame it came from. That reference is deliberately harsh -
detail the downscale destroyed is unrecoverable, so the absolute number is
dominated by content the pipeline never had a chance to keep. Use it to RANK
configurations against a shared reference, not to attribute loss to one stage.

It streams, for the same reason the engine does: holding a 1080p clip plus its
upscaled candidate and flow maps in memory runs to gigabytes and grows with clip
length. Peak here is a handful of frames regardless of duration.

It is a research tool, not part of the shipped path, and it measures fidelity
only. Nothing here is a watermark verdict: only a provider-oracle row is.

    uv run python scripts/video_fidelity_probe.py source.mp4 out-512.mp4 out-768.mp4
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from remove_ai_watermarks.video_invisible import _iter_sampled_frames as _iter_sampled_frames
from remove_ai_watermarks.video_invisible import _probe_video
from remove_ai_watermarks.video_quality import _iter_frames as _iter_frames

sys.path.insert(0, str(Path(__file__).parent))


log = logging.getLogger(__name__)


def _delivered_geometry(path: Path) -> tuple[int, int, float]:
    from remove_ai_watermarks.video_quality import delivered_geometry

    try:
        return delivered_geometry(path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _measure(
    source: Path,
    delivered: Path,
    *,
    source_geometry: tuple[int, int, float],
    duration: float | None,
    preserve_timestamps: bool = False,
) -> dict[str, Any]:
    from remove_ai_watermarks.video_quality import measure_video_fidelity

    try:
        return measure_video_fidelity(
            source,
            delivered,
            source_geometry=source_geometry,
            duration=duration,
            preserve_timestamps=preserve_timestamps,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


@click.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("delivered", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--duration",
    type=click.FloatRange(min=0.1),
    default=None,
    help="Score only the first N seconds of the source, matching a trimmed sweep candidate.",
)
@click.option("--preserve-timestamps", is_flag=True, help="Require matching full-clip PTS and video duration.")
@click.option("--json-out", type=click.Path(dir_okay=False, path_type=Path), help="Also write the rows as JSON.")
def main(
    source: Path, delivered: tuple[Path, ...], duration: float | None, json_out: Path | None, preserve_timestamps: bool
) -> None:
    """Score each DELIVERED file against SOURCE end to end."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    source_geometry = _probe_video(source)
    log.info("Source %s: %dx%d at %.4f fps", source.name, *source_geometry)

    rows = [
        _measure(
            source, path, source_geometry=source_geometry, duration=duration, preserve_timestamps=preserve_timestamps
        )
        for path in delivered
    ]
    for row in rows:
        log.info(
            "%s: %dx%d at %s fps, %.1f%% of source pixels, PSNR %s dB, SSIM %s, temporal %s, %s kbps",
            row["file"],
            row["width"],
            row["height"],
            row["fps"],
            row["pixel_ratio"] * 100.0,
            row["source_psnr_db"],
            row["source_ssim"],
            row["temporal_residual_ratio"],
            row["file_bitrate_kbps"],
        )
    if json_out is not None:
        json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        log.info("Wrote %s", json_out)


if __name__ == "__main__":
    main()
