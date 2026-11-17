"""Guard benchmark accounting independently of optional ML runtimes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark_text_schemes import assess


@pytest.mark.parametrize(
    ("positive", "negative", "after", "expected"),
    [
        (False, False, False, "invalid_positive_control"),
        (True, True, False, "invalid_negative_control"),
        (True, False, None, "refused"),
        (True, False, True, "retained"),
        (True, False, False, "below_reference_threshold"),
    ],
)
def test_reference_benchmark_never_counts_invalid_controls_or_refusals(positive, negative, after, expected):
    observed = assess(
        {"is_watermarked": positive},
        {"is_watermarked": negative},
        None if after is None else {"is_watermarked": after},
    )
    assert observed == expected


def test_catalog_does_not_present_missing_provider_access_as_a_pass():
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "data/evaluations/text-watermark/catalog.json").read_text())
    assert {row["provider"] for row in catalog["provider_cases"]} == {"anthropic", "openai", "google", "microsoft"}
    assert all(row["test_status"].startswith("blocked_") for row in catalog["provider_cases"])
    assert all(row["source"] in catalog["sources"] for row in catalog["reference_cases"])
    assert len({row["scheme"] for row in catalog["reference_cases"]}) == len(catalog["reference_cases"])


def test_reference_model_bridge_preserves_cache_and_cpu_reference_math():
    from types import SimpleNamespace

    from scripts.benchmark_text_schemes import cpu_reference_model

    class Tensor:
        def __init__(self, device):
            self.device = device

        def to(self, device):
            return Tensor(device)

        def cpu(self):
            return self.to("cpu")

    cache = object()
    calls = []

    def model(tokens, **kwargs):
        calls.append((tokens, kwargs))
        return SimpleNamespace(logits=Tensor("mps"), past_key_values=cache)

    bridge = cpu_reference_model(model, "mps")
    output = bridge(Tensor("cpu"), attention_mask=Tensor("cpu"), past_key_values=cache)
    assert calls[0][0].device == "mps"
    assert calls[0][1]["attention_mask"].device == "mps"
    assert calls[0][1]["past_key_values"] is cache
    assert output.logits.device == "cpu"
    assert output.past_key_values is cache
    assert bridge.device == "cpu"


def test_dense_reference_tables_are_rejected_before_allocation():
    from scripts.benchmark_text_schemes import guard_reference_memory

    with pytest.raises(MemoryError, match="EXPGumbel"):
        guard_reference_memory("EXPGumbel", 151936, {"prefix_length": 2}, 8 * 1024**3)


def test_reference_prompt_fragment_is_not_generated_evidence():
    from scripts.benchmark_text_schemes import strip_reference_prompt

    tagged = "<|im_start|>system\nYou are helpful.<|im_end|>user\nExplain libraries."
    plain = "system\nYou are helpful.user\nExplain libraries."
    with pytest.raises(ValueError, match="prompt"):
        strip_reference_prompt("<|im_start|>system\nYou are helpful.", plain, tagged)
    assert strip_reference_prompt(plain + "New prose.", plain, tagged) == "New prose."


def test_reference_history_is_independent_between_generation_and_detection():
    from types import SimpleNamespace

    from scripts.benchmark_text_schemes import reset_reference_history

    reference = SimpleNamespace(utils=SimpleNamespace(cc_history={"old-context"}, state_indicator=0))
    reset_reference_history(reference, generation=False)
    assert reference.utils.cc_history == set()
    assert reference.utils.state_indicator == 1
    reference.utils.cc_history.add("detected-context")
    reset_reference_history(reference, generation=True)
    assert reference.utils.cc_history == set()
    assert reference.utils.state_indicator == 0


def test_reference_help_ignores_an_unrelated_scripts_package_and_needs_no_torch(tmp_path):
    import subprocess
    import sys

    shadow = tmp_path / "scripts"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("")
    root = Path(__file__).resolve().parents[1]
    code = (
        "import runpy,sys;sys.modules['torch']=None;"
        f"sys.path.insert(0,{str(tmp_path)!r});"
        f"runpy.run_path({str(root / 'scripts/benchmark_textseal_reference.py')!r},run_name='__main__')"
    )
    result = subprocess.run(  # noqa: S603 - fixed Python launcher, repr-escaped paths, no shell
        [sys.executable, "-c", code, "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "--reference-root" in result.stdout


@pytest.mark.parametrize(
    ("after", "retained_spans", "expected"),
    [(False, 0, "below_reference_threshold"), (False, 1, "partial_unverified"), (True, 1, "retained")],
)
def test_reference_assessment_keeps_partial_global_negatives_unverified(after, retained_spans, expected):
    # A frozen ruT5 trial has a negative document but a detected retained paragraph.
    assert (
        assess(
            {"is_watermarked": True},
            {"is_watermarked": False},
            {"is_watermarked": after},
            retained_prose_spans=retained_spans,
        )
        == expected
    )
