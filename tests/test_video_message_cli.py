"""CLI exposes known-message evidence and preserves skip and failure status."""

import json

import pytest
from click.testing import CliRunner

from remove_ai_watermarks import video_messages as video
from remove_ai_watermarks._internal.audio_quality import sha256_file
from remove_ai_watermarks.cli import main


@pytest.mark.parametrize("status", ["not_matched", "unverified"])
def test_unqualified_message_cli_exits_two(tmp_path, monkeypatch, status):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    report = video.VideoMessageReport(source, sha256_file(source), "videoseal", "a5" * 32, status, None)
    monkeypatch.setattr(video, "identify_video_message", lambda *a, **kw: report)
    result = CliRunner().invoke(main, ["video", "message", "remove", str(source), "--reference", "a5" * 32, "--json"])
    assert result.exit_code == 2, result.output
    assert json.loads(result.output)["skipped"] is True
    assert not (tmp_path / "source_message_processed.mp4").exists()


def test_failed_message_cli_keeps_before_after_evidence(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    report = video.VideoMessageReport(source, sha256_file(source), "videoseal", "a5" * 32, "matched", None)

    def fail(*a, **kw):
        raise video.VideoMessageVerificationIncomplete(report, report)

    monkeypatch.setattr(video, "remove_video_message", fail)
    result = CliRunner().invoke(main, ["video", "message", "remove", str(source), "--reference", "a5" * 32, "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["before"]["status"] == "matched"
    assert payload["after"]["status"] == "matched"
