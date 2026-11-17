"""Native chat templates must not receive a second automatic BOS token."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from remove_ai_watermarks import text as lib


@pytest.mark.parametrize("chat_template", [True, False])
def test_template_special_tokens_are_not_added_twice_and_plain_prompts_keep_them(monkeypatch, chat_template):
    calls = []

    class Batch(dict):
        def to(self, _device):
            return self

    class InputIds(list):
        @property
        def shape(self):
            return (1, len(self[0]))

    class Tokenizer:
        eos_token_id = 0

        def __init__(self):
            self.chat_template = "native" if chat_template else None

        def apply_chat_template(self, _messages, **_kwargs):
            if self.chat_template is None:
                raise ValueError("No chat template configured")
            return "<bos>Native conversation"

        def __call__(self, prompt, *, return_tensors, add_special_tokens=True):
            assert return_tensors == "pt"
            ids = ([1] if add_special_tokens else []) + ([1] if prompt.startswith("<bos>") else []) + [2, 3]
            calls.append(ids)
            return Batch(input_ids=InputIds([ids]))

        def decode(self, _tokens, **_kwargs):
            return "Rewritten passage."

    def generate(**kwargs):
        return [kwargs["input_ids"][0] + [9, 0]]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda _: None, no_grad=nullcontext))
    tokenizer = Tokenizer()
    model = SimpleNamespace(generate=generate, generation_config=SimpleNamespace(eos_token_id=0))
    monkeypatch.setattr(lib, "_load_local_rewriter_runtime", lambda *_: (tokenizer, model, "cpu"))
    monkeypatch.setattr(
        lib, "_guard_local_rewrite_candidate", lambda _source, _masked, _facts, candidate, _validator: candidate
    )
    assert lib._build_local_lm_rewriter()("Original passage.") == "Rewritten passage."
    assert calls == [[1, 2, 3]]
