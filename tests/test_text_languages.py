"""Language-change rejection without treating classifier scores as semantic proof."""

from types import SimpleNamespace

import pytest

from remove_ai_watermarks import text as lib

SOURCE = "The library hosted an open meeting where volunteers explained how to preserve old documents."
CANDIDATE = "La bibliothèque a organisé une réunion ouverte pour expliquer la conservation des documents anciens."


@pytest.mark.parametrize(
    ("source_label", "source_score", "target_label", "target_score", "expected"),
    [
        ("en", 0.99, "fr", 0.99, False),
        ("bg", 0.99, "ru", 0.99, False),
        ("en", 0.95, "es", 0.8, False),
        ("en", 0.949, "fr", 0.99, True),
        ("en", 0.99, "fr", 0.799, True),
        ("en", 0.99, "en", 0.8, True),
        ("und", 0.99, "en", 0.99, True),
        ("zxx", 0.99, "en", 0.99, True),
        ("en", 0.99, "zxx", 0.99, False),
    ],
)
def test_confident_language_changes_are_rejected(
    source_label, source_score, target_label, target_score, expected, monkeypatch
):
    results = iter([(source_label, source_score), (target_label, target_score)])
    monkeypatch.setattr(lib, "_classify_text_language", lambda _: next(results), raising=False)
    assert lib._preserves_rewrite_language(SOURCE, CANDIDATE) is expected


def test_short_and_placeholder_only_samples_abstain_before_loading_a_model(monkeypatch):
    def unexpected(_):
        pytest.fail("A short or placeholder-only sample must not invoke classification")

    monkeypatch.setattr(lib, "_classify_text_language", unexpected, raising=False)
    assert lib._preserves_rewrite_language("Bonjour", "Hello")
    assert lib._preserves_rewrite_language("[[RAIW_SPAN_0]] " * 30, CANDIDATE)
    assert lib._preserves_rewrite_language(SOURCE, "[[RAIW_SPAN_1]] " * 30)


def test_language_sample_removes_service_markers_and_normalizes_only_for_classification(monkeypatch):
    seen = []

    def classify(value):
        seen.append(value)
        return "en", 0.99

    monkeypatch.setattr(lib, "_classify_text_language", classify, raising=False)
    assert lib._preserves_rewrite_language("e\u0301" * 40 + " [[RAIW_SPAN_0]]", "a" * 40 + " [[RAIW_SPAN_1]]")
    assert seen == ["é" * 40, "a" * 40]


@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        ("en",),
        ("", 0.99),
        (None, 0.99),
        ("en", True),
        ("en", "0.99"),
        ("en", float("nan")),
        ("en", -1),
        ("en", 1.1),
    ],
)
def test_invalid_classifier_results_are_failures(result, monkeypatch):
    monkeypatch.setattr(
        lib, "_load_text_language_identifier", lambda: SimpleNamespace(classify=lambda _: result), raising=False
    )
    with pytest.raises(RuntimeError, match="language identifier"):
        lib._classify_text_language(SOURCE)


def test_classifier_returns_native_values_without_caching_user_text(monkeypatch):
    seen = []

    def classify(text):
        seen.append(text)
        return "en", 0.99

    monkeypatch.setattr(
        lib, "_load_text_language_identifier", lambda: SimpleNamespace(classify=classify), raising=False
    )
    assert lib._classify_text_language(SOURCE) == ("en", 0.99)
    assert lib._classify_text_language(SOURCE) == ("en", 0.99)
    assert seen == [SOURCE, SOURCE]


def test_same_script_translation_is_not_a_successful_local_candidate(monkeypatch):
    source = (
        "La bibliothèque a organisé une rencontre publique. Les bénévoles ont présenté des méthodes "
        "pour conserver les vieux documents. Les participants ont reçu une fiche explicative."
    )
    candidate = (
        "The library held an open meeting. Volunteers shared techniques for preserving old documents. "
        "Attendees received an explanation sheet."
    )
    results = iter([("fr", 1.0), ("en", 0.99)])
    monkeypatch.setattr(lib, "_classify_text_language", lambda _: next(results), raising=False)
    assert lib._guard_local_rewrite_candidate(source, source, {}, candidate, None) is None


def test_text_runtime_requires_the_language_classifier(monkeypatch):
    from remove_ai_watermarks import optional_deps

    monkeypatch.setattr(optional_deps, "module_available", lambda *names: "py3langid" not in names)
    assert not lib.is_text_invisible_available()


def test_synthid_reference_runtime_does_not_require_a_language_classifier(monkeypatch):
    from remove_ai_watermarks import optional_deps

    monkeypatch.setattr(optional_deps, "module_available", lambda *names: "py3langid" not in names)
    lib._require_text_reference_runtime()


def test_mlx_runtime_also_requires_the_language_classifier(monkeypatch):
    from remove_ai_watermarks import optional_deps

    monkeypatch.setattr(lib.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(lib.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(optional_deps, "module_available", lambda *names: "py3langid" not in names)
    assert not lib.is_text_invisible_available("mlx")


def test_language_model_is_loaded_from_packaged_assets_once(monkeypatch):
    calls = []
    sentinel = object()

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return sentinel

    module = SimpleNamespace(MODEL_FILE="data/model.npz.xz", LanguageIdentifier=SimpleNamespace(from_model_file=load))
    original_import = lib.importlib.import_module
    monkeypatch.setattr(
        lib.importlib, "import_module", lambda name: module if name == "py3langid.langid" else original_import(name)
    )
    lib._load_text_language_identifier.cache_clear()
    try:
        assert lib._load_text_language_identifier() is sentinel
        assert lib._load_text_language_identifier() is sentinel
        assert calls == [("data/model.npz.xz", {"norm_probs": True})]
    finally:
        lib._load_text_language_identifier.cache_clear()


def test_language_guard_receives_masked_text_before_factual_restoration(monkeypatch):
    source = "The staff recorded the reference name Long Reference Name in the written report."
    masked = source.replace("Long Reference Name", "[[RAIW_SPAN_0]]")
    candidate = masked.replace("recorded", "logged")
    seen = []

    def language(before, after):
        seen.append((before, after))
        return True

    monkeypatch.setattr(lib, "_preserves_rewrite_language", language)
    result = lib._guard_local_rewrite_candidate(
        source, masked, {"[[RAIW_SPAN_0]]": "Long Reference Name"}, candidate, None
    )
    assert result == source.replace("recorded", "logged")
    assert seen == [(masked, candidate)]
