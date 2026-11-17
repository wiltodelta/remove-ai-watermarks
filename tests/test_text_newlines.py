"""Text cleanup must preserve original UTF-8 line endings and locked code bytes."""

from __future__ import annotations

import pytest

from remove_ai_watermarks.text import remove_text_all, remove_text_invisible, remove_text_metadata


@pytest.mark.parametrize("newline", ["\r\n", "\r", "\n"])
@pytest.mark.parametrize("operation", ["metadata", "all", "invisible"])
def test_file_operations_preserve_line_endings_and_locked_code(tmp_path, newline, operation):
    text = newline.join(["Original prose.", "", "```python", 'value = "source"', "```", ""])
    source = tmp_path / "source.md"
    output = tmp_path / "output.md"
    source.write_bytes(text.encode())
    if operation == "metadata":
        result = remove_text_metadata(source, output)
        expected = text
    elif operation == "all":
        result = remove_text_all(source, output)
        expected = text
    else:

        def rewrite(passage):
            assert "value" not in passage, "Fenced code reached the prose rewriter"
            return passage.replace("Original", "Rephrased")

        result = remove_text_invisible(source, output, rewriter=rewrite)
        expected = text.replace("Original", "Rephrased")
    assert result.cleaned_text == expected
    assert output.read_bytes() == expected.encode()
    assert source.read_bytes() == text.encode()
