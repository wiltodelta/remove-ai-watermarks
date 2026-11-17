"""Runtime guards for local prose regeneration and reference watermark scoring."""

from __future__ import annotations

import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from remove_ai_watermarks import text as text_module


class _Encoding(dict):
    def to(self, _device):
        return self


class _Tokenizer:
    eos_token_id = 0

    def __init__(self, candidate):
        self.candidate = candidate

    def apply_chat_template(self, *_args, **_kwargs):
        return "prompt"

    def __call__(self, *_args, **_kwargs):
        return _Encoding(input_ids=SimpleNamespace(shape=(1, 3)))

    def decode(self, *_args, **_kwargs):
        return self.candidate


@pytest.mark.parametrize(
    "candidate",
    [
        "The shipment contains 99 units.",
        "The shipment contains 12 units.",
        "",
        "This shipment holds 12 units. [[RAIW_SPAN_N]]",
        "The shipment contains at least [[RAIW_SPAN_0]] units.",
    ],
)
def test_local_rewriter_refuses_fact_drift_and_verbatim_echo(monkeypatch, candidate):
    """Exhausted retries must not publish an invalid or unchanged best effort."""
    import sys

    fake_torch = SimpleNamespace(manual_seed=lambda _seed: None, no_grad=nullcontext)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    model = SimpleNamespace(generate=lambda **_kwargs: [[1, 2, 3, 4]])
    tokenizer = _Tokenizer(candidate)
    monkeypatch.setattr(text_module, "_load_local_rewriter_runtime", lambda *_a, **_k: (tokenizer, model, "cpu"))
    rewriter = text_module._build_local_lm_rewriter()
    with pytest.raises(RuntimeError, match="rewrite"):
        rewriter("The shipment contains 12 units.")


def test_custom_identity_rewriter_does_not_claim_regeneration():
    with pytest.raises(RuntimeError, match="unchanged"):
        text_module.remove_text_invisible("Ordinary prose needs actual rewording.", rewriter=lambda text: text)


def test_local_rewriter_refuses_an_unrequested_script_change(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda _seed: None, no_grad=nullcontext))
    model = SimpleNamespace(generate=lambda **_kwargs: [[1, 2, 3, 4]])
    tokenizer = _Tokenizer("A mother reads a book to her child.")
    monkeypatch.setattr(text_module, "_load_local_rewriter_runtime", lambda *_a, **_k: (tokenizer, model, "cpu"))
    with pytest.raises(RuntimeError, match="rewrite"):
        text_module._build_local_lm_rewriter()("мама читает книгу своему ребёнку.")


def test_failed_regeneration_preserves_an_existing_output(tmp_path):
    output = tmp_path / "result.txt"
    output.write_text("Keep this previous result.")
    with pytest.raises(RuntimeError, match="unchanged"):
        text_module.remove_text_invisible("Unchanged source prose.", output, rewriter=lambda text: text)
    assert output.read_text() == "Keep this previous result."


def test_comparative_one_is_not_a_quantity_but_counted_one_is():
    assert text_module.verify_text_rewrite_facts(
        "Cycling is one of the most efficient options.", "Cycling is a highly efficient option."
    )
    assert not text_module.verify_text_rewrite_facts("We shipped one crate.", "We shipped crates.")


def test_list_markers_are_locked_without_locking_the_prose():
    source = "1. Libraries help readers.\n2) Libraries host workshops.\n- Libraries lend books."
    spans = text_module.split_protected_text_spans(source)
    markers = [span.text for span in spans if span.kind == "list_marker"]
    assert markers == ["1. ", "2) ", "- "]
    assert any(not span.locked and "Libraries" in span.text for span in spans)
    result = text_module.remove_text_invisible(
        source, rewriter=lambda passage: passage.replace("Libraries", "Archives")
    )
    assert result.cleaned_text == source.replace("Libraries", "Archives")


def test_synthid_requires_the_optional_reference_runtime(monkeypatch):
    from remove_ai_watermarks import optional_deps

    monkeypatch.setattr(optional_deps, "module_available", lambda *_names: False)
    with pytest.raises(RuntimeError, match="torch"):
        text_module.score_synthid_text(list(range(20)), [777, 778])


@pytest.mark.parametrize("sampling_table_seed", [0, 42])
def test_synthid_score_matches_reference_hashing_and_repetition_mask(sampling_table_seed):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    keys = [101, 102, 103, 104]
    token_ids = [*range(80), *range(80), 90, 91, 92, 93]
    processor = transformers.SynthIDTextWatermarkLogitsProcessor(
        ngram_len=5,
        keys=keys,
        sampling_table_size=65536,
        sampling_table_seed=sampling_table_seed,
        context_history_size=1024,
        device=torch.device("cpu"),
    )
    ids = torch.tensor([token_ids], dtype=torch.long)
    mask = processor.compute_context_repetition_mask(ids)
    values = processor.compute_g_values(ids)[mask].double()
    expected = (float(values.mean()) - 0.5) / math.sqrt(0.25 / values.numel())
    actual = text_module.score_synthid_text(token_ids, keys, sampling_table_seed=sampling_table_seed)
    assert actual.z_score == pytest.approx(expected)
    assert actual.num_tokens == int(mask.sum())


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("We shipped 12 crates.", "We shipped 12 crates in 2025."),
        ("Readers can borrow books.", "Readers can borrow 10 books."),
    ],
)
def test_fact_guard_rejects_invented_quantities(source, candidate):
    assert not text_module.verify_text_rewrite_facts(source, candidate)


def test_local_fact_mask_preserves_literal_values_and_existing_spans():
    passage = "NASA moved 12 crates for Postgres to [[RAIW_SPAN_0]]."
    masked, replacements = text_module._mask_local_rewrite_facts(passage)
    assert "12" not in masked
    assert "Postgres" not in masked
    assert "NASA" not in masked
    assert masked.count("[[RAIW_SPAN_0]]") == 1
    restored = masked
    for token, literal in replacements.items():
        restored = restored.replace(token, literal)
    assert restored == passage
    assert "[[RAIW_SPAN_0]]" not in replacements


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Neighbors help each other.", "Neighbors support one another."),
        ("Neighbors support one another.", "Neighbors help each other."),
    ],
)
def test_reciprocal_one_another_is_not_a_quantity(source, candidate):
    assert text_module.verify_text_rewrite_facts(source, candidate)
    assert not text_module.verify_text_rewrite_facts("We shipped one other crate.", "We shipped another crate.")


@pytest.mark.parametrize(("last_token", "eos_ids", "accept"), [(4, 0, False), (0, 0, True), (2, [0, 2], True)])
def test_local_rewriter_rejects_token_budget_truncation(monkeypatch, last_token, eos_ids, accept):
    import sys

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda _seed: None, no_grad=nullcontext))

    def generate(**kwargs):
        return [[1, 2, 3] + [4] * (kwargs["max_new_tokens"] - 1) + [last_token]]

    model = SimpleNamespace(generate=generate, generation_config=SimpleNamespace(eos_token_id=eos_ids))
    tokenizer = _Tokenizer("The archive lends books to readers.")
    monkeypatch.setattr(text_module, "_load_local_rewriter_runtime", lambda *_a, **_k: (tokenizer, model, "cpu"))
    rewrite = text_module._build_local_lm_rewriter()
    if accept:
        assert rewrite("Readers can borrow books from the archive.") == tokenizer.candidate
    else:
        with pytest.raises(RuntimeError, match="rewrite"):
            rewrite("Readers can borrow books from the archive.")


@pytest.mark.parametrize("collapse", [True, False])
def test_local_rewriter_refuses_obvious_long_passage_collapse(monkeypatch, collapse):
    import sys

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda _seed: None, no_grad=nullcontext))
    model = SimpleNamespace(generate=lambda **_kwargs: [[1, 2, 3, 4]])
    passage = (
        "the archive lends books to readers and holds public workshops. "
        "its rooms provide a place to study and meet neighbors. "
    ) * 4
    candidate = "The archive lends books." if collapse else passage.replace("lends", "loans").replace("holds", "hosts")
    tokenizer = _Tokenizer(candidate)
    monkeypatch.setattr(text_module, "_load_local_rewriter_runtime", lambda *_a, **_k: (tokenizer, model, "cpu"))
    rewrite = text_module._build_local_lm_rewriter()
    if collapse:
        with pytest.raises(RuntimeError, match="rewrite"):
            rewrite(passage)
    else:
        assert rewrite(passage) == candidate.strip()


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("NASA maintains the archive.", "Another organization maintains the archive."),
        ("the note says FINISHED.", "the note contains a message."),
    ],
)
def test_fact_guard_preserves_all_caps_terms_including_sentence_start(source, candidate):
    assert not text_module.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate", "expected"),
    [
        ("Only registered users can borrow equipment.", "Registered users can borrow equipment.", False),
        ("Only registered users can borrow equipment.", "Equipment loans are restricted to registered users.", True),
        ("The quoted request must remain part of the document.", "The document mentions a request.", False),
        ("Participants must register.", "Registration is required for participants.", True),
        ("Регистрация обязательна для доступа.", "Регистрация необязательна для доступа.", False),
        ("Регистрация обязательна для доступа.", "Без регистрации доступ невозможен.", True),
        ("Регистрация необязательна для доступа.", "Доступ возможен без регистрации.", True),
        ("The old building has limited space.", "Space in the old building is restricted.", True),
        ("This not only improves air quality but also saves fuel.", "This improves air quality and saves fuel.", True),
        (
            "Библиотеки не только выдают книги, но и проводят занятия.",
            "Библиотеки выдают книги и проводят занятия.",
            True,
        ),
    ],
)
def test_fact_guard_preserves_restrictions_and_requirements(source, candidate, expected):
    assert text_module.verify_text_rewrite_facts(source, candidate) is expected


@pytest.mark.parametrize(
    ("source", "candidate", "expected"),
    [
        ("Readers can\u2019t borrow books.", "Readers can borrow books.", False),
        ("Users mustn\u2019t share keys.", "Users must not share keys.", True),
        ("Users mustn't share keys.", "Users must share keys.", False),
    ],
)
def test_fact_guard_handles_deontic_and_curly_apostrophe_negation(source, candidate, expected):
    assert text_module.verify_text_rewrite_facts(source, candidate) is expected


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        (
            "Библиотеки помогают развитию местного сообщества.",
            "Библиотеки помогают development местного сообщества.",
        ),
        (
            "Libraries support local community development.",
            "Libraries support local community развитие.",
        ),
    ],
)
def test_script_guard_rejects_new_cross_script_fragments(source, candidate):
    assert not text_module._preserves_dominant_script(source, candidate)
