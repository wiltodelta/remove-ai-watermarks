"""Research NLI accounting must expose weak claims and invalid inputs."""

import json

import pytest

from scripts import benchmark_text_fidelity as bench


def test_label_index_is_explicit_and_not_a_substring():
    assert bench.entailment_label_index({"0": "not_entailment", "1": "entailment"}) == 1
    assert bench.entailment_label_index({0: "entailment", 1: "neutral", 2: "contradiction"}) == 0
    with pytest.raises(ValueError, match="NLI"):
        bench.entailment_label_index({0: "not_entailment"})
    with pytest.raises(ValueError, match="NLI"):
        bench.entailment_label_index({1: "entailment"})


def test_both_directions_and_all_sentences_keep_the_lowest_support():
    calls = []

    def predict(source, candidate):
        calls.append((source, candidate))
        return len(calls) / 10

    scores = bench.aggregate_support("First source. Second source.", "First result. Second result.", predict)
    assert scores == {"whole": 0.1, "sentence": 0.3}
    assert calls == [
        ("First source. Second source.", "First result. Second result."),
        ("First result. Second result.", "First source. Second source."),
        ("First source. Second source.", "First result."),
        ("First source. Second source.", "Second result."),
        ("First result. Second result.", "First source."),
        ("First result. Second result.", "Second source."),
    ]


@pytest.mark.parametrize("value", [True, -1, 1.1, float("nan"), float("inf"), "0.9"])
def test_invalid_support_is_not_accepted(value):
    with pytest.raises(ValueError, match="probability"):
        bench.aggregate_support("Source.", "Candidate.", lambda *_: value)


def test_loader_accepts_explicit_unlabeled_diagnostics(tmp_path):
    path = tmp_path / "cases.json"
    rows = [{"id": "case", "source": "Source.", "candidate": "Candidate.", "equivalent": None}]
    path.write_text(json.dumps({"cases": rows}))
    assert bench.load_cases(path) == rows


@pytest.mark.parametrize("change", [{"equivalent": 1}, {"source": ""}, {"candidate": 5}, {"id": ""}])
def test_loader_rejects_invalid_case_fields(tmp_path, change):
    path = tmp_path / "cases.json"
    row = {"id": "case", "source": "Source.", "candidate": "Candidate.", "equivalent": True}
    row.update(change)
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match=r"Cases|equivalent"):
        bench.load_cases(path)


def test_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "cases.json"
    row = {"id": "case", "source": "Source.", "candidate": "Candidate.", "equivalent": True}
    path.write_text(json.dumps([row, row]))
    with pytest.raises(ValueError, match="unique"):
        bench.load_cases(path)


def test_empty_prose_is_rejected_before_inference():
    with pytest.raises(ValueError, match="nonempty"):
        bench.aggregate_support("", "Candidate.", lambda *_: 0.99)


def test_existing_output_is_never_overwritten_or_loaded(tmp_path, monkeypatch):
    from types import SimpleNamespace

    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"id": "case", "source": "Source.", "candidate": "Candidate.", "equivalent": True}]))
    output = tmp_path / "existing"
    output.mkdir()
    report = output / "benchmark.json"
    report.write_text("existing data")
    monkeypatch.setattr(bench.importlib, "import_module", lambda *_: pytest.fail("Must refuse before loading a model"))
    with pytest.raises(FileExistsError):
        bench.evaluate(SimpleNamespace(cases=path, output_dir=output, max_pair_tokens=512))
    assert report.read_text() == "existing data"


def test_oversize_pairs_are_recorded_without_truncation_or_model_inference(tmp_path, monkeypatch):
    from types import SimpleNamespace

    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"id": "case", "source": "Source.", "candidate": "Candidate.", "equivalent": True}]))
    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    token_calls = []

    class Tokenizer:
        model_max_length = 512

        def __call__(self, source, candidate, **kwargs):
            token_calls.append((source, candidate, kwargs))
            return {"input_ids": SimpleNamespace(shape=(1, 2))}

    model = SimpleNamespace(
        config=SimpleNamespace(
            id2label={0: "entailment", 1: "not_entailment"},
            max_position_embeddings=512,
            model_type="bert",
            to_dict=dict,
        ),
        train=lambda _: None,
        to=lambda _: None,
    )
    tokenizer = Tokenizer()
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
        AutoModelForSequenceClassification=SimpleNamespace(
            from_pretrained=lambda *a, **kw: (model, {"missing_keys": [], "unexpected_keys": []})
        ),
    )
    torch = SimpleNamespace(
        float32="float32",
        set_num_threads=lambda _: None,
        inference_mode=lambda: pytest.fail("An oversized input must not reach inference"),
    )
    original_import = bench.importlib.import_module
    monkeypatch.setattr(
        bench.importlib,
        "import_module",
        lambda name: {"torch": torch, "transformers": transformers}.get(name) or original_import(name),
    )
    monkeypatch.setattr(bench.importlib.metadata, "version", lambda _name: "0.0-test")
    report = bench.evaluate(
        SimpleNamespace(
            cases=path, model_dir=model_dir, output_dir=tmp_path / "output", device="cpu", max_pair_tokens=1
        )
    )
    assert report["completed"]
    assert report["calls"] == []
    assert report["cases"][0]["id"] == "case"
    assert report["cases"][0]["status"] == "input_window_exceeded"
    assert token_calls == [("Source.", "Candidate.", {"truncation": False, "return_tensors": "pt"})]
