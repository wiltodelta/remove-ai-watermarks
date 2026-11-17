"""Caller-supplied acceptance checks participate in the library rewrite transaction."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from remove_ai_watermarks import text as lib


def test_custom_validator_rejection_preserves_existing_destination(tmp_path):
    output = tmp_path / "out.txt"
    output.write_text("Previous result.")
    with pytest.raises(RuntimeError, match="validator"):
        lib.remove_text_invisible(
            "Visitors used the room.",
            output,
            rewriter=lambda _: "Tourists used the room.",
            rewrite_validator=lambda _source, candidate: "Tourists" not in candidate,
        )
    assert output.read_text() == "Previous result."


def test_partial_processing_counts_validator_rejections():
    source = "Visitors used the room.\n\nReaders borrowed books."
    result = lib.remove_text_invisible(
        source,
        rewriter=lambda p: p.replace("Visitors", "Tourists").replace("borrowed", "rented"),
        rewrite_validator=lambda _source, candidate: "Tourists" not in candidate,
        allow_partial=True,
    )
    assert result.cleaned_text == "Visitors used the room.\n\nReaders rented books."
    assert result.retained_prose_spans == 1
    assert result.token_watermark_action == "partial_regeneration"


@pytest.mark.parametrize("verdict", ["false", {"equivalent": False}, 1, None])
def test_validator_requires_a_real_boolean(verdict, tmp_path):
    output = tmp_path / "out.txt"
    with pytest.raises(TypeError, match="bool"):
        lib.remove_text_invisible(
            "Visitors used the room.",
            output,
            rewriter=lambda _: "Tourists used the room.",
            rewrite_validator=lambda *_: verdict,
            allow_partial=True,
        )
    assert not output.exists()


def test_combined_pipeline_forwards_the_validator():
    with pytest.raises(RuntimeError, match="validator"):
        lib.remove_text_all(
            "Visitors used the room.",
            include_invisible=True,
            rewriter=lambda _: "Tourists used the room.",
            rewrite_validator=lambda *_: False,
        )


def test_validator_requires_regeneration_in_combined_pipeline(tmp_path):
    output = tmp_path / "out.txt"
    with pytest.raises(ValueError, match="include_invisible"):
        lib.remove_text_all("Some text.", output, rewrite_validator=lambda *_: True)
    assert not output.exists()


def test_local_factory_retries_after_validator_rejection(monkeypatch):
    import sys

    candidates = iter(["Tourists used the reading room.", "The reading room was used by visitors."])
    seen = []

    class Encoding(dict):
        def to(self, _):
            return self

    tokenizer = SimpleNamespace(
        eos_token_id=0,
        apply_chat_template=lambda *_a, **_k: "prompt",
        decode=lambda *_a, **_k: next(candidates),
    )

    class Tokenizer:
        eos_token_id = tokenizer.eos_token_id
        apply_chat_template = staticmethod(tokenizer.apply_chat_template)
        decode = staticmethod(tokenizer.decode)

        def __call__(self, *_a, **_k):
            return Encoding(input_ids=SimpleNamespace(shape=(1, 3)))

    model = SimpleNamespace(generate=lambda **_: [[1, 2, 3, 4]])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda _: None, no_grad=nullcontext))
    monkeypatch.setattr(lib, "_load_local_rewriter_runtime", lambda *_: (Tokenizer(), model, "cpu"))

    def validator(source, candidate):
        seen.append((source, candidate))
        return "Tourists" not in candidate

    rewrite = lib._build_local_lm_rewriter(validator=validator)
    assert rewrite("Visitors used the reading room.") == "The reading room was used by visitors."
    assert len(seen) == 2
    assert all(source == "Visitors used the reading room." for source, _ in seen)


def test_validator_observes_real_protected_text_without_internal_placeholders():
    source = "Explain `value = 12` precisely."
    seen = []

    def validator(before, after):
        seen.append((before, after))
        return True

    result = lib.remove_text_invisible(
        source,
        rewriter=lambda p: p.replace("Explain", "Describe"),
        rewrite_validator=validator,
    )
    assert result.cleaned_text == "Describe `value = 12` precisely."
    assert seen == [(source, result.cleaned_text)]


def test_locked_content_containing_reserved_token_strings_survives():
    source = "Explain `[[RAIW_SPAN_1]]` and `value` precisely."
    result = lib.remove_text_invisible(source, rewriter=lambda p: p.replace("Explain", "Describe"))
    assert result.cleaned_text == source.replace("Explain", "Describe")


def test_literal_placeholder_ids_remain_literal_in_validator_inputs():
    source = "Explain [[RAIW_SPAN_0]] and `value` precisely."
    seen = []

    def validator(before, after):
        seen.append((before, after))
        return True

    result = lib.remove_text_invisible(
        source,
        rewriter=lambda p: p.replace("Explain", "Describe"),
        rewrite_validator=validator,
    )
    assert result.cleaned_text == source.replace("Explain", "Describe")
    assert seen == [(source, result.cleaned_text)]


def test_custom_rewriter_retries_when_the_validator_rejects_first_candidate():
    candidates = iter(["Tourists used the room.", "The room was used by visitors."])
    result = lib.remove_text_invisible(
        "Visitors used the room.",
        rewriter=lambda _: next(candidates),
        rewrite_validator=lambda _source, candidate: "Tourists" not in candidate,
    )
    assert result.cleaned_text == "The room was used by visitors."


def test_fact_mask_does_not_parse_literal_placeholder_ids_as_integers():
    literal = "[[RAIW_SPAN_" + "9" * 5000 + "]]"
    source = "We shipped 12 crates. " + literal
    masked, replacements = lib._mask_local_rewrite_facts(source)
    restored = masked
    for token, value in replacements.items():
        restored = restored.replace(token, value)
    assert restored == source
    assert literal not in replacements


@pytest.mark.parametrize("source", ["Original prose.", "Use `safe()` carefully."])
def test_empty_custom_rewrites_never_delete_prose(source, tmp_path):
    output = tmp_path / "out.txt"
    output.write_text("Previous result.")

    def rewrite(passage):
        return "A changed paragraph." if "[[RAIW_SPAN_" in passage else ""

    with pytest.raises(RuntimeError, match="empty"):
        lib.remove_text_invisible(source, output, rewriter=rewrite)
    assert output.read_text() == "Previous result."


def test_custom_validator_retries_an_echo_without_approving_it():
    candidates = iter(["Visitors used the room.", "The room was used by visitors."])
    seen = []

    def validator(source, candidate):
        seen.append(candidate)
        return True

    result = lib.remove_text_invisible(
        "Visitors used the room.",
        rewriter=lambda _: next(candidates),
        rewrite_validator=validator,
    )
    assert result.cleaned_text == "The room was used by visitors."
    assert seen == [result.cleaned_text]
