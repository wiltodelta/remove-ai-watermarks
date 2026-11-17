"""Audio capability checks exercise the installed CLI without neural downloads."""

from pathlib import Path

import pytest

from tests.test_agent_skill import _probe_module


@pytest.mark.parametrize("delivered", [False, True])
def test_audio_probe_requires_a_delivered_file(delivered):
    probe = _probe_module()

    def run(argv, timeout=8):
        assert argv[1:3] == ["audio", "invisible"]
        assert "--force" in argv
        assert "--no-verify" in argv
        if delivered:
            Path(argv[argv.index("-o") + 1]).write_bytes(b"RIFF" + b"\0" * 4 + b"WAVE")
        return 0, '{"frames_processed": 1600, "verified_removed": null}', ""

    probe._run = run
    assert probe._audio_stack("/synthetic/cli") == ("ok" if delivered else "unknown")


def test_audio_probe_reports_the_actual_missing_extra():
    probe = _probe_module()
    probe._run = lambda *a, **kw: (1, "", "Audio processing requires 'remove-ai-watermarks[audio]' and ffmpeg")
    assert probe._audio_stack("/synthetic/cli") == "missing"
