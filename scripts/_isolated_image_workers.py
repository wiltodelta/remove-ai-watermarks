"""Bound native image work in fresh interpreters, with no parent-process fallback."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _run_one(script: Path, path: str, timeout: float) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), str(script), path]
    try:
        result = subprocess.run(  # noqa: S603 -- fixed interpreter and local worker entry point
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"path": path, "keys": [], "status": "timeout"}
    except OSError as error:
        return {"path": path, "keys": [], "status": f"worker-error:{type(error).__name__}"}
    if result.returncode != 0:
        return {"path": path, "keys": [], "status": "crashed", "returncode": result.returncode}
    try:
        row = json.loads(result.stdout)
        if not isinstance(row, dict) or row.get("status") is None:
            raise ValueError("worker returned an invalid record")
    except ValueError:
        return {"path": path, "keys": [], "status": "invalid-worker-result"}
    return row


def run_batch(script: Path, paths: list[str], *, jobs: int, timeout: float) -> list[dict[str, Any]]:
    """Run at most jobs child processes; each child gets its own timeout.

    Threads only supervise subprocesses. subprocess.run kills and reaps a timed-out
    child before returning, so executor shutdown never waits on unbounded native work.
    No fork or multiprocessing spawn imports the parent's OpenCV/Pillow state.
    """
    if jobs < 1 or timeout <= 0:
        raise ValueError("jobs and timeout must be positive")
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        return list(executor.map(lambda path: _run_one(script, path, timeout), paths))


def main() -> None:
    script = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(script.parent.parent / "src"))
    # stdout is the worker protocol, diagnostics from the audit go to stderr.
    from contextlib import redirect_stdout

    with redirect_stdout(sys.stderr):
        namespace = runpy.run_path(str(script))
        row = namespace["_one"](sys.argv[2])
    sys.stdout.write(json.dumps(row))


if __name__ == "__main__":
    main()
