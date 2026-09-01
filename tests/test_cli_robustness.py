"""A failed write and a directory input must REPORT, not raise a traceback.

Both found 2026-07-20 by the Tier E adversarial sweep (`scripts/robustness_suite.py`),
which drives the real CLI over degenerate inputs. Neither was reachable from the 849-test
suite, because unit tests feed well-formed fixtures into writable directories.

1. READ-ONLY OUTPUT DIRECTORY. `image_io.imwrite` is contractually non-raising -- it
   returns False when the codec rejects the image or the path cannot be written. But
   `write_bgr_with_alpha` discarded that bool and returned None, so no caller could tell a
   failed write from a successful one. `cmd_erase` then ran `output.stat()` on a file that
   was never created and died with `FileNotFoundError`. The signal existed the whole way
   down and was thrown away by the wrapper.

2. A DIRECTORY PASSED WHERE A FILE IS EXPECTED. `click.Path(exists=True)` accepts
   directories unless told otherwise, so `identify <dir>` reached the metadata scanner and
   raised `IsADirectoryError` from `open()`. (`batch` was already correct: it declares
   `file_okay=False`.)
"""

from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

from remove_ai_watermarks import image_io
from remove_ai_watermarks.cli import main


@pytest.fixture
def bgr() -> np.ndarray:
    return np.full((60, 80, 3), 128, np.uint8)


class TestFailedWriteIsReported:
    def test_write_bgr_with_alpha_reports_failure(self, tmp_path, bgr):
        """The wrapper must PROPAGATE imwrite's failure signal, not swallow it.

        Uses a nonexistent directory rather than `chmod`: CI runs windows-latest, where
        `os.chmod` cannot make a DIRECTORY unwritable, so a chmod-based assertion would be
        a platform-dependent flake. `write_bgr_with_alpha` does no mkdir of its own, so a
        missing parent is a genuine write failure on every platform.
        """
        assert image_io.write_bgr_with_alpha(tmp_path / "no-such-dir" / "x.png", bgr, None) is False

    def test_write_bgr_with_alpha_reports_success(self, tmp_path, bgr):
        """The other direction, so the assertion above cannot pass by always being False."""
        assert image_io.write_bgr_with_alpha(tmp_path / "ok.png", bgr, None) is True

    def test_erase_reports_a_failed_write_without_a_traceback(self, tmp_path, bgr, monkeypatch):
        """The CLI must exit non-zero with a readable message, not raise FileNotFoundError.

        The write is forced to fail by patching, not by `chmod`: the CLI mkdirs the parent,
        so a missing directory would not reproduce it, and chmod on a directory is a no-op
        on Windows. Patching states the condition under test directly -- "the write failed".
        """
        src = tmp_path / "in.png"
        image_io.imwrite(src, bgr)
        monkeypatch.setattr(image_io, "write_bgr_with_alpha", lambda *a, **k: False)
        result = CliRunner().invoke(main, ["erase", str(src), "--region", "5,5,20,10", "-o", str(tmp_path / "x.png")])
        assert result.exit_code != 0
        assert not isinstance(result.exception, FileNotFoundError), "write failure escaped as a traceback"
        assert "write" in result.output.lower() or "failed" in result.output.lower()

    def test_batch_counts_a_failed_write_instead_of_exiting_zero(self, tmp_path, bgr, monkeypatch):
        """The worst shape of this bug: no output files AND a success exit code.

        Regression: `batch --mode visible` into a read-only directory wrote no files
        and exited 0, so a wrapping service would treat an
        empty output directory as a completed run. The batch loop counts per-image
        exceptions, so the write must RAISE there, never `SystemExit` (which would abort
        the whole run instead of failing one image).
        """
        indir = tmp_path / "in"
        indir.mkdir()
        for i in range(2):
            image_io.imwrite(indir / f"img{i}.png", bgr)
        monkeypatch.setattr(image_io, "write_bgr_with_alpha", lambda *a, **k: False)
        result = CliRunner().invoke(main, ["batch", str(indir), "-o", str(tmp_path / "out"), "--mode", "visible"])
        assert result.exit_code != 0, "a batch that wrote nothing must not exit 0"


class TestApiReportsFailedWrite:
    """The same bug lived one layer down, in the library API, with a misleading message.

    `api._write_visible_result` also discarded the write flag, then ran the metadata strip
    on a file that was never created. The resulting `FileNotFoundError` surfaced through
    the CLI as `cannot read image <INPUT>: ... <OUTPUT path>` -- it blamed the input while
    quoting the output. A library caller got the same confusing error with no CLI at all.
    """

    def test_remove_visible_raises_a_clear_error_on_unwritable_output(self, tmp_path, bgr, monkeypatch):
        """Patched rather than chmod'd: `_write_visible_result` mkdirs the parent, so a
        missing directory would not reproduce it, and chmod on a directory is a no-op on
        the Windows CI runner."""
        from remove_ai_watermarks.api import remove_visible

        monkeypatch.setattr(image_io, "write_bgr_with_alpha", lambda *a, **k: False)
        with pytest.raises(OSError, match="failed to write output"):
            remove_visible(bgr, tmp_path / "out.png", strip_metadata=False)


class TestDirectoryInputIsRejected:
    @pytest.mark.parametrize("cmd", ["identify", "classify", "visible", "erase", "metadata", "invisible", "all"])
    def test_directory_as_source_is_a_clean_usage_error(self, tmp_path, cmd):
        """A directory must be refused by argument parsing, never reach the scanners."""
        args = [cmd, str(tmp_path)]
        if cmd == "erase":
            args += ["--region", "1,1,5,5"]
        if cmd == "metadata":
            args += ["--check"]
        result = CliRunner().invoke(main, args)
        assert result.exit_code != 0
        assert not isinstance(result.exception, IsADirectoryError), "directory reached the file reader"
