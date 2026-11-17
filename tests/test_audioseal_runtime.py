"""The optional runtime reports compatible-model presence, including failures."""

from __future__ import annotations

import numpy as np
import pytest


class _Detector:
    def __call__(self, samples, sample_rate):
        import torch

        assert sample_rate == 16000
        probabilities = torch.tensor([0.51, 0.51, 0.51, 0.0])
        result = torch.stack((1 - probabilities, probabilities)).unsqueeze(0)
        message = torch.tensor([0.0, 1.0] * 8).unsqueeze(0)
        return result, message


def test_presence_uses_fraction_of_positive_samples_not_mean():
    from remove_ai_watermarks._internal.audioseal import read

    result = read(_Detector(), np.zeros(64, dtype=np.float32))
    assert result.detected is True
    assert result.frac_above_threshold == 0.75
    assert result.mean_detect_prob < 0.5
    assert result.decoded_bits == (0, 1) * 8


@pytest.mark.parametrize("invalid", [np.array([], dtype=np.float32), np.array([np.nan]), np.zeros((2, 32))])
def test_invalid_samples_do_not_become_a_negative_reading(invalid):
    from remove_ai_watermarks._internal.audioseal import read

    with pytest.raises(ValueError, match="samples"):
        read(_Detector(), invalid)


def test_invalid_detector_output_does_not_become_a_negative_reading():
    import torch

    from remove_ai_watermarks._internal.audioseal import read

    class InvalidDetector:
        def __call__(self, samples, sample_rate):
            return torch.full((1, 2, 64), float("nan")), torch.zeros((1, 16))

    with pytest.raises(RuntimeError, match="nonfinite"):
        read(InvalidDetector(), np.zeros(64, dtype=np.float32))
