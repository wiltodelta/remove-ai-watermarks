"""Explicit partial regeneration retains rejected text without claiming a full rewrite."""

from __future__ import annotations

import pytest

from remove_ai_watermarks.text import remove_text_all, remove_text_invisible


def _selective_rewriter(passage: str) -> str:
    if "retained" in passage:
        raise RuntimeError("Candidate did not preserve meaning")
    return passage.replace("rewritten", "rephrased")


def test_partial_regeneration_retains_rejected_prose_and_protected_code(tmp_path):
    source = (
        "This paragraph is rewritten.\n\nThis retained paragraph has `original()`.\n\nFinal paragraph is rewritten."
    )
    output = tmp_path / "result.txt"
    result = remove_text_invisible(source, output, rewriter=_selective_rewriter, allow_partial=True)
    assert result.cleaned_text == source.replace("rewritten", "rephrased")
    assert output.read_text() == result.cleaned_text
    assert result.rewritten_spans == 2
    assert result.retained_prose_spans == 2
    assert result.token_watermark_action == "partial_regeneration"


def test_partial_regeneration_counts_an_identity_candidate_as_retained():
    source = "This paragraph is rewritten.\n\nThis paragraph stays original."
    result = remove_text_invisible(source, rewriter=lambda p: p.replace("rewritten", "rephrased"), allow_partial=True)
    assert result.retained_prose_spans == 1
    assert result.cleaned_text.endswith("This paragraph stays original.")
    assert result.token_watermark_action == "partial_regeneration"


def test_all_rejected_partial_regeneration_does_not_replace_existing_output(tmp_path):
    output = tmp_path / "result.txt"
    output.write_text("Previous reviewed output.")
    with pytest.raises(RuntimeError, match="regeneration"):
        remove_text_invisible("Retain every word.", output, rewriter=lambda p: p, allow_partial=True)
    assert output.read_text() == "Previous reviewed output."


def test_partial_regeneration_does_not_swallow_unexpected_errors(tmp_path):
    def broken(_passage):
        raise ValueError("Unexpected invalid configuration")

    output = tmp_path / "result.txt"
    with pytest.raises(ValueError, match="configuration"):
        remove_text_invisible("Some ordinary prose.", output, rewriter=broken, allow_partial=True)
    assert not output.exists()


def test_combined_cleaner_reports_partial_regeneration():
    source = "This paragraph is rewritten.\n\nThis paragraph is retained.\u200b"
    result = remove_text_all(source, include_invisible=True, rewriter=_selective_rewriter, allow_partial=True)
    assert result.cleaned_text == "This paragraph is rephrased.\n\nThis paragraph is retained."
    assert result.retained_prose_spans == 1
    assert result.token_watermark_action == "partial_regeneration"


def test_partial_regeneration_requires_the_invisible_stage():
    with pytest.raises(ValueError, match="include_invisible"):
        remove_text_all("Some prose.", allow_partial=True)


def test_cli_partial_mode_reports_retained_text(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from remove_ai_watermarks import text as text_module
    from remove_ai_watermarks.cli import main

    monkeypatch.setattr(text_module, "_build_local_lm_rewriter", lambda *_a, **_k: _selective_rewriter)
    source = tmp_path / "source.txt"
    source.write_text("This paragraph is rewritten.\n\nThis paragraph is retained.")
    output = tmp_path / "result.txt"
    result = CliRunner().invoke(main, ["text", "invisible", str(source), "-o", str(output), "--allow-partial"])
    assert result.exit_code == 0, result.output
    assert "PARTIAL" in result.output
    assert "retained unchanged" in result.output
    assert output.read_text() == source.read_text().replace("rewritten", "rephrased")


def test_cli_partial_mode_requires_prose_regeneration(tmp_path):
    from click.testing import CliRunner

    from remove_ai_watermarks.cli import main

    source = tmp_path / "source.txt"
    source.write_text("Some ordinary prose.")
    output = tmp_path / "result.txt"
    result = CliRunner().invoke(main, ["text", "all", str(source), "-o", str(output), "--allow-partial"])
    assert result.exit_code != 0
    assert "requires --invisible" in result.output
    assert not output.exists()
