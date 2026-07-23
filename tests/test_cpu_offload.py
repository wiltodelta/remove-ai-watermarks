"""Unit tests for the --cpu-offload device-placement branch (mocked pipeline).

``WatermarkRemover._move_to_device_and_optimize`` chooses between a full
``pipeline.to("cuda")`` and ``enable_model_cpu_offload()`` (low-VRAM streaming).
Constructing the remover is cheap -- the diffusion pipeline is lazy and the
device string is not validated -- so the placement decision is exercised with a
mock pipeline, no model download or GPU required. Gated on torch (the module
imports it at top), so it runs under the ``gpu`` extra and skips the core CI
matrix, matching the model-running test policy.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

pytest.importorskip("torch")

from remove_ai_watermarks.noai.watermark_remover import WatermarkRemover


def _remover(device: str, cpu_offload: bool) -> WatermarkRemover:
    return WatermarkRemover(device=device, pipeline="sdxl", cpu_offload=cpu_offload)


class TestCpuOffloadPlacement:
    def test_offload_enabled_on_cuda_streams_instead_of_moving(self):
        remover = _remover("cuda", cpu_offload=True)
        pipeline = Mock()

        returned = remover._move_to_device_and_optimize(pipeline)

        pipeline.enable_model_cpu_offload.assert_called_once_with()
        pipeline.to.assert_not_called()
        # Offload leaves the pipeline object in place (accelerate hooks handle it).
        assert returned is pipeline

    def test_no_offload_moves_whole_pipeline_to_cuda(self):
        remover = _remover("cuda", cpu_offload=False)
        pipeline = Mock()

        remover._move_to_device_and_optimize(pipeline)

        pipeline.to.assert_called_once_with("cuda")
        pipeline.enable_model_cpu_offload.assert_not_called()

    def test_offload_flag_ignored_off_cuda(self):
        # The flag is CUDA-only: on cpu it must still be a plain .to("cpu").
        remover = _remover("cpu", cpu_offload=True)
        pipeline = Mock()

        remover._move_to_device_and_optimize(pipeline)

        pipeline.to.assert_called_once_with("cpu")
        pipeline.enable_model_cpu_offload.assert_not_called()
