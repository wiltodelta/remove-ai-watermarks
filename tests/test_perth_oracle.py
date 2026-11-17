"""Tests for the pinned Perth oracle's pure contract."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import perth_oracle  # noqa: E402


def test_detection_rule_is_upstream_round_at_one_half() -> None:
    assert perth_oracle.PerthReading(0.499).detected is False
    assert perth_oracle.PerthReading(0.5).detected is False
    assert perth_oracle.PerthReading(0.501).detected is True


def test_checkpoint_and_source_are_immutable_pins() -> None:
    assert len(perth_oracle.SOURCE_TREE_SHA256) == 64
    assert len(perth_oracle.CHECKPOINT_SHA256) == 64
    assert all(char in "0123456789abcdef" for char in perth_oracle.CHECKPOINT_SHA256)


def test_perth_has_no_message_payload_contract() -> None:
    reading = perth_oracle.PerthReading(0.75)

    assert not hasattr(reading, "decoded_bits")
    assert not hasattr(perth_oracle, "message_bits")


def test_facade_uses_the_shipped_decoder():
    from remove_ai_watermarks._internal import perth

    assert perth_oracle.read is perth.read
    assert perth_oracle.load_model is perth.load_model
    assert perth_oracle.embed is perth.embed
