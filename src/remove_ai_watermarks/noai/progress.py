"""Terminal progress animation and library output suppression.

This module provides two main capabilities for the CLI:

1. ``run_with_progress`` — a styled two-line terminal animation that
   displays a bouncing highlight bar, a braille spinner, elapsed time,
   and a live operation message while a background task executes.

2. ``silence_library_output`` — a wrapper that suppresses noisy log
   output produced by third-party ML libraries (transformers, diffusers,
   huggingface_hub, tqdm) so the user only sees our own progress messages.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import time
import warnings
from collections.abc import Callable
from typing import Any

# ── ANSI color constants ────────────────────────────────────────────
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

# Bar geometry
_BAR_WIDTH = 32
_HIGHLIGHT_WIDTH = 5


def _no_color() -> bool:
    """Respect the NO_COLOR convention (https://no-color.org/)."""
    return bool(os.environ.get("NO_COLOR"))


def _truncate(text: str, max_len: int = 72) -> str:
    """Shorten a string with an ellipsis if it exceeds *max_len*."""
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _build_bar(step: int) -> str:
    """Build a flowing highlight bar that bounces across the width.

    The highlight segment (5 chars wide) travels left→right→left
    continuously, giving the user a visual "working" signal.
    """
    cycle = _BAR_WIDTH * 2 - 2
    pos = step % cycle
    if pos >= _BAR_WIDTH:
        pos = cycle - pos

    hl_start = max(0, pos - _HIGHLIGHT_WIDTH // 2)
    hl_end = min(_BAR_WIDTH, pos + _HIGHLIGHT_WIDTH // 2 + 1)
    before = "━" * hl_start
    highlight = "━" * (hl_end - hl_start)
    after = "━" * (_BAR_WIDTH - hl_end)

    if _no_color():
        return before + highlight + after
    return f"{_DIM}{before}{_RESET}{_BOLD}{_YELLOW}{highlight}{_RESET}{_DIM}{after}{_RESET}"


def run_with_progress(
    task: Callable[[], Any],
    progress_state: dict[str, str] | None = None,
) -> Any:
    """Execute *task* in a background thread while showing a progress animation.

    The animation renders two lines to ``sys.__stderr__``:

    - **Line 1**: braille spinner + bouncing bar + elapsed seconds
    - **Line 2**: current operation message from *progress_state*

    When the task finishes, a green "Completed" line replaces the animation.

    Args:
        task: A zero-argument callable to run in the background.
        progress_state: Mutable dict whose ``"message"`` key is read
            by the animation loop to display the current operation.

    Returns:
        Whatever *task* returns.

    Raises:
        Any exception raised by *task* is re-raised after the animation
        is cleaned up.
    """
    done = threading.Event()
    output_holder: dict[str, Any] = {"result": None, "error": None}

    def worker() -> None:
        try:
            output_holder["result"] = task()
        except Exception as error:  # pragma: no cover – passthrough
            output_holder["error"] = error
        finally:
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    idx = 0
    start_time = time.time()
    no_color = _no_color()

    def _get_operation() -> str:
        if isinstance(progress_state, dict):
            return progress_state.get("message", "Processing...")
        return "Processing..."

    # ── Animation loop ──────────────────────────────────────────────
    while not done.is_set():
        spinner = spinner_frames[idx % len(spinner_frames)]
        elapsed = int(time.time() - start_time)
        bar_str = _build_bar(idx)
        operation = _truncate(_get_operation())

        if no_color:
            line1 = f"  {spinner}  Processing {bar_str}  {elapsed:>3}s"
            line2 = f"  ╰─ {operation}"
        else:
            line1 = f"  {_CYAN}{spinner}{_RESET}  Processing {bar_str}  {_BOLD}{_YELLOW}{elapsed:>3}s{_RESET}"
            line2 = f"  {_DIM}╰─ {operation}{_RESET}"

        print(
            f"\r\033[2K{line1}\n\033[2K{line2}\033[1A\r",
            end="",
            flush=True,
            file=sys.__stderr__,
        )
        time.sleep(0.08)
        idx += 1

    # ── Final "done" frame ──────────────────────────────────────────
    thread.join()
    total = int(time.time() - start_time)
    final_operation = _truncate(_get_operation())
    done_bar = "━" * _BAR_WIDTH

    if no_color:
        final_line1 = f"  ✓  Completed {done_bar}  {total:>3}s"
        final_line2 = f"  ╰─ {final_operation}"
    else:
        final_line1 = (
            f"  {_GREEN}{_BOLD}✓{_RESET}  {_GREEN}Completed{_RESET} "
            f"{_GREEN}{done_bar}{_RESET}  {_BOLD}{_GREEN}{total:>3}s{_RESET}"
        )
        final_line2 = f"  {_DIM}╰─ {final_operation}{_RESET}"

    print(
        f"\r\033[2K{final_line1}\n\033[2K{final_line2}",
        file=sys.__stderr__,
    )

    if output_holder["error"] is not None:
        raise output_holder["error"]

    return output_holder["result"]


def silence_library_output(
    run_func: Callable[[], Any],
    set_progress: Callable[[str], None] | None = None,
) -> Callable[[], Any]:
    """Return a wrapper that silences noisy ML library output.

    The wrapper:

    1. Disables HuggingFace Hub progress bars via env var.
    2. Sets ``transformers``, ``diffusers``, and ``huggingface_hub``
       loggers to *error* level.
    3. Redirects ``stdout`` and ``stderr`` to ``io.StringIO`` sinks so
       that stray ``tqdm`` bars and model-loading chatter are invisible.
    4. Suppresses all Python warnings during the call.

    Args:
        run_func: The callable to execute silently.
        set_progress: Optional callback to report phase changes.

    Returns:
        A zero-argument callable that, when invoked, runs *run_func*
        inside the silent context.
    """

    def wrapped() -> Any:
        if set_progress:
            set_progress("Configuring runtime and suppressing noisy logs...")

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        for _silence in (
            lambda: __import__("transformers").logging.set_verbosity_error(),
            lambda: _silence_diffusers(),
            lambda: __import__("huggingface_hub").logging.set_verbosity_error(),
        ):
            try:
                _silence()
            except Exception:
                pass

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    if set_progress:
                        set_progress("Executing watermark removal pipeline...")
                    return run_func()

    return wrapped


def _silence_diffusers() -> None:
    """Silence diffusers logging and progress bars."""
    from diffusers.utils import logging as diffusers_logging

    diffusers_logging.set_verbosity_error()
    if hasattr(diffusers_logging, "disable_progress_bar"):
        diffusers_logging.disable_progress_bar()


# ── Shared pipeline progress helpers ─────────────────────────────────

_DEFAULT_PRE_PHASES: list[tuple[int, str]] = [
    (0, "Encoding image with VAE encoder"),
    (3, "Mapping pixel data → latent space"),
    (7, "Injecting noise into latent representation"),
    (12, "Building denoiser schedule"),
    (18, "Starting reverse diffusion sampler"),
    (30, "Running first denoising iteration"),
    (50, "Still processing — this can take a while"),
    (90, "Pipeline running — may take a few minutes"),
]

_DEFAULT_POST_PHASES: list[tuple[int, str]] = [
    (0, "Denoising complete · Running VAE decoder"),
    (2, "Decoding latent channels → RGB color space"),
    (5, "Reconstructing pixel grid from latents"),
    (10, "Applying color space conversion and normalization"),
    (18, "Finalizing pixel output"),
    (30, "Still decoding — large images take longer"),
    (60, "Almost done — large images take longer to decode"),
]


def make_pipeline_progress(
    effective_steps: int,
    device: str,
    set_progress: Callable[[str], None],
    *,
    bar_len: int = 20,
    label: str = "Denoising",
    pre_phases: list[tuple[int, str]] | None = None,
    post_phases: list[tuple[int, str]] | None = None,
) -> tuple[Callable, threading.Event, threading.Event, Callable[[], threading.Thread]]:
    """Create step callback and background updater for a diffusion pipeline.

    Returns:
        (step_callback, first_step_event, pipeline_done_event, start_updater)
        where ``start_updater()`` launches and returns the background thread.
    """
    pre = pre_phases or [(s, f"{m} on {device}") for s, m in _DEFAULT_PRE_PHASES]
    post = post_phases or [(s, f"{m} on {device}") for s, m in _DEFAULT_POST_PHASES]

    t0_holder: list[float] = [time.monotonic()]
    first_step = threading.Event()
    pipeline_done = threading.Event()
    last_cb_time: list[float] = [t0_holder[0]]

    def _background_updater() -> None:
        idx = 0
        while not first_step.is_set():
            elapsed = time.monotonic() - t0_holder[0]
            while idx < len(pre) - 1 and elapsed >= pre[idx + 1][0]:
                idx += 1
            set_progress(pre[idx][1])
            first_step.wait(timeout=0.4)

        idx = 0
        post_start: float | None = None
        while not pipeline_done.is_set():
            since_cb = time.monotonic() - last_cb_time[0]
            if since_cb >= 1.5:
                if post_start is None:
                    post_start = time.monotonic()
                elapsed = time.monotonic() - post_start
                while idx < len(post) - 1 and elapsed >= post[idx + 1][0]:
                    idx += 1
                set_progress(post[idx][1])
            else:
                post_start = None
                idx = 0
            pipeline_done.wait(timeout=0.4)

    def step_callback(step: int, timestep: int, latents: Any) -> None:  # noqa: ARG001
        first_step.set()
        last_cb_time[0] = time.monotonic()
        elapsed = time.monotonic() - t0_holder[0]
        current = step + 1
        per_step = elapsed / max(1, current)
        remaining = per_step * max(0, effective_steps - current)
        filled = int(bar_len * current / max(1, effective_steps))
        bar = "█" * filled + "░" * (bar_len - filled)
        set_progress(
            f"{label} [{bar}] {current}/{effective_steps} | {elapsed:.0f}s elapsed, ~{remaining:.0f}s left | {device}"
        )

    def start_updater() -> threading.Thread:
        t0_holder[0] = time.monotonic()
        last_cb_time[0] = t0_holder[0]
        first_step.clear()
        pipeline_done.clear()
        t = threading.Thread(target=_background_updater, daemon=True)
        t.start()
        return t

    return step_callback, first_step, pipeline_done, start_updater


# ── MPS fallback helper ──────────────────────────────────────────────


def is_mps_error(error: Exception) -> bool:
    """Check whether an exception is an MPS-related runtime error."""
    return "mps" in str(error).lower()
