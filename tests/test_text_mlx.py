"""MLX generation uses the same protected-fact and publication guards as HF."""

from types import SimpleNamespace

import pytest

from remove_ai_watermarks import text as lib


def _runtime(monkeypatch, responses):
    calls = []
    seeds = []

    def no_transformers(*_args):
        raise AssertionError("MLX must not load the Transformers runtime")

    monkeypatch.setattr(lib, "_load_local_rewriter_runtime", no_transformers)
    tokenizer = SimpleNamespace(
        chat_template="template",
        encode=lambda text: list(range(len(text))),
        apply_chat_template=lambda messages, **kwargs: calls.append((messages, kwargs)) or "prompt",
    )
    batches = iter(responses)

    def stream(_model, _tokenizer, **kwargs):
        calls.append(kwargs)
        return iter(next(batches))

    runtime = (tokenizer, object(), SimpleNamespace(random=SimpleNamespace(seed=seeds.append)), stream, lambda **kw: kw)
    monkeypatch.setattr(lib, "_load_mlx_rewriter_runtime", lambda *_args: runtime, raising=False)
    return calls, seeds


def _response(text, reason="stop"):
    return SimpleNamespace(text=text, finish_reason=reason)


def test_mlx_retries_truncation_and_preserves_a_compound_name(monkeypatch):
    calls, seeds = _runtime(
        monkeypatch, [[_response("Truncated", "length")], [_response("Residents in [[RAIW_SPAN_0]] used the room.")]]
    )
    result = lib.remove_text_invisible(
        "The room served residents in Willow Creek.", rewriter_backend="mlx", rewriter_model="local-model", seed=7
    )
    assert result.cleaned_text == "Residents in Willow Creek used the room."
    assert seeds == [7, 8]
    assert calls[0][1]["enable_thinking"] is False


@pytest.mark.parametrize("reason", ["length", None, "unknown"])
def test_mlx_requires_an_explicit_clean_stop_before_publication(monkeypatch, tmp_path, reason):
    _runtime(monkeypatch, [[_response("A rewritten passage.", reason)]] * 3)
    destination = tmp_path / "out.txt"
    destination.write_text("Existing result.")
    with pytest.raises(RuntimeError, match="rewrite"):
        lib.remove_text_invisible("An original passage.", destination, rewriter_backend="mlx")
    assert destination.read_text() == "Existing result."


@pytest.mark.parametrize(
    "candidate",
    [
        "",
        "This source has [[RAIW_SPAN_0]] units.",
        "It contains at least [[RAIW_SPAN_0]] units.",
        "Changed source without a marker.",
    ],
)
def test_mlx_rejects_empty_identity_bound_drift_and_dropped_markers(monkeypatch, candidate):
    _runtime(monkeypatch, [[_response(candidate)]] * 3)
    with pytest.raises(RuntimeError, match="rewrite"):
        lib.remove_text_invisible("This source has 20 units.", rewriter_backend="mlx")


def test_mlx_validator_receives_restored_inline_content_and_can_retry(monkeypatch):
    _runtime(
        monkeypatch, [[_response("Carefully use [[RAIW_SPAN_0]].")], [_response("Use [[RAIW_SPAN_0]] with care.")]]
    )
    seen = []

    def validate(source, candidate):
        seen.append((source, candidate))
        return len(seen) == 2

    result = lib.remove_text_invisible("Use `safe()` carefully.", rewriter_backend="mlx", rewrite_validator=validate)
    assert result.cleaned_text == "Use `safe()` with care."
    assert seen == [
        ("Use `safe()` carefully.", "Carefully use `safe()`."),
        ("Use `safe()` carefully.", "Use `safe()` with care."),
    ]


def test_combined_pipeline_forwards_the_mlx_backend(monkeypatch):
    _runtime(monkeypatch, [[_response("This passage has fresh wording.")]])
    result = lib.remove_text_all("This is the original passage.", include_invisible=True, rewriter_backend="mlx")
    assert result.cleaned_text == "This passage has fresh wording."


@pytest.mark.parametrize("kwargs", [{"rewriter_backend": "bad"}, {"rewriter_backend": "mlx", "rewriter": str}])
def test_invalid_backend_configuration_is_rejected_before_io(tmp_path, kwargs):
    with pytest.raises(ValueError, match="backend"):
        lib.remove_text_invisible(tmp_path / "does-not-exist.txt", **kwargs)


def test_combined_mlx_backend_requires_regeneration():
    with pytest.raises(ValueError, match="include_invisible"):
        lib.remove_text_all("Original prose.", rewriter_backend="mlx")


@pytest.mark.parametrize("command", [["text", "invisible"], ["text", "all", "--invisible"]])
def test_cli_forwards_backend_and_checkpoint(monkeypatch, tmp_path, command):
    from click.testing import CliRunner

    from remove_ai_watermarks.cli import main

    source = tmp_path / "source.txt"
    source.write_text("Original prose.")
    observed = []
    result = SimpleNamespace(
        rewritten_spans=1,
        locked_spans=0,
        locked_characters=0,
        output=tmp_path / "output.txt",
        retained_prose_spans=0,
        removed_signals=(),
        removed_codepoints=0,
        remaining_markers={},
    )

    def process(*args, **kwargs):
        observed.append(kwargs)
        return result

    monkeypatch.setattr(lib, "remove_text_invisible", process)
    monkeypatch.setattr(lib, "remove_text_all", process)
    run = CliRunner().invoke(
        main, [*command, str(source), "--rewriter-backend", "mlx", "--rewriter-model", "local-checkpoint"]
    )
    assert run.exit_code == 0, run.output
    assert observed[0]["rewriter_backend"] == "mlx"
    assert observed[0]["rewriter_model"] == "local-checkpoint"


def test_mlx_explicit_mps_uses_a_gpu_stream(monkeypatch):
    calls, _seeds = _runtime(monkeypatch, [[_response("This passage has fresh wording.")]])
    original = lib._load_mlx_rewriter_runtime

    def runtime(*args):
        tokenizer, model, mx, stream, sampler = original(*args)
        mx.gpu = "gpu"
        mx.new_stream = lambda device: "explicit-gpu-stream"
        return tokenizer, model, mx, stream, sampler

    monkeypatch.setattr(lib, "_load_mlx_rewriter_runtime", runtime)
    lib.remove_text_invisible("This is the original passage.", rewriter_backend="mlx", device="mps")
    assert calls[1]["stream"] == "explicit-gpu-stream"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mlx_rejects_unsupported_device_before_imports(device):
    with pytest.raises(ValueError, match="device"):
        lib._load_mlx_rewriter_runtime("model", device)


def test_locked_only_input_does_not_load_mlx(monkeypatch):
    def fail(*_args):
        raise AssertionError("No runtime is needed for locked-only input")

    monkeypatch.setattr(lib, "_load_mlx_rewriter_runtime", fail)
    result = lib.remove_text_invisible("```python\nvalue = 1\n```", rewriter_backend="mlx")
    assert result.rewritten_spans == 0
    assert result.cleaned_text == "```python\nvalue = 1\n```"


def test_mlx_loader_disallows_remote_code_and_loads_local_chat_template(monkeypatch, tmp_path):
    observed = []
    tokenizer = SimpleNamespace(chat_template=None, name_or_path=str(tmp_path))
    (tmp_path / "chat_template.jinja").write_text("local template")

    def load(model, **kwargs):
        observed.append((model, kwargs))
        return object(), tokenizer

    original_import = lib.importlib.import_module

    def fake_import(name):
        return SimpleNamespace(load=load) if name == "mlx_lm" else original_import(name)

    monkeypatch.setattr(lib.importlib, "import_module", fake_import)
    lib._load_mlx_rewriter_cached.cache_clear()
    try:
        _model, loaded = lib._load_mlx_rewriter_cached("local-checkpoint")
        assert loaded.chat_template == "local template"
        assert observed == [
            ("local-checkpoint", {"trust_remote_code": False, "tokenizer_config": {"trust_remote_code": False}})
        ]
    finally:
        lib._load_mlx_rewriter_cached.cache_clear()


def test_mlx_rejects_invented_service_markers(monkeypatch):
    _runtime(monkeypatch, [[_response("A changed [[RAIW_SPAN_9]] passage.")]] * 3)
    with pytest.raises(RuntimeError, match="rewrite"):
        lib.remove_text_invisible("An original passage.", rewriter_backend="mlx")


def test_mlx_availability_accepts_a_cold_namespace_with_a_concrete_core(tmp_path, monkeypatch):
    import importlib
    import sys

    for name in list(sys.modules):
        if name == "mlx" or name.startswith("mlx.") or name == "mlx_lm":
            monkeypatch.delitem(sys.modules, name)
    package = tmp_path / "mlx"
    package.mkdir()
    (package / "core.py").write_text("# Concrete runtime stand-in.\n")
    (tmp_path / "mlx_lm.py").write_text("# Language model runtime stand-in.\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(lib.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(lib.platform, "machine", lambda: "arm64")
    assert importlib.util.find_spec("mlx").loader is None
    assert lib.is_text_invisible_available("mlx") is True
    importlib.import_module("mlx.core")
    assert lib.is_text_invisible_available("mlx") is True


@pytest.mark.parametrize("parent_exists", [False, True])
def test_mlx_availability_refuses_a_missing_core(tmp_path, monkeypatch, parent_exists):
    import sys

    for name in list(sys.modules):
        if name == "mlx" or name.startswith("mlx.") or name == "mlx_lm":
            monkeypatch.delitem(sys.modules, name)
    if parent_exists:
        (tmp_path / "mlx").mkdir()
    (tmp_path / "mlx_lm.py").write_text("# Language model runtime stand-in.\n")
    # Replace search roots to prevent an optional real MLX install masking the gap.
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    monkeypatch.setattr(lib.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(lib.platform, "machine", lambda: "arm64")
    assert lib.is_text_invisible_available("mlx") is False


def test_mlx_availability_requires_the_language_model_runtime(tmp_path, monkeypatch):
    import sys

    for name in list(sys.modules):
        if name == "mlx" or name.startswith("mlx.") or name == "mlx_lm":
            monkeypatch.delitem(sys.modules, name)
    package = tmp_path / "mlx"
    package.mkdir()
    (package / "core.py").write_text("# Concrete runtime stand-in.\n")
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    monkeypatch.setattr(lib.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(lib.platform, "machine", lambda: "arm64")
    assert lib.is_text_invisible_available("mlx") is False
