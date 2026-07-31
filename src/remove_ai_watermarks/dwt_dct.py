"""DWT-DCT (dwtDct) open watermark encode/decode.

Vendored from ShieldMnt/invisible-watermark ``imwatermark/maxDct.py`` (MIT),
trimmed to the matrix-DCT path used by Stable Diffusion / SDXL / FLUX. Avoids
pulling the upstream package's torch and non-headless OpenCV dependencies.

Copyright (c) 2021 ShieldMnt
"""

# cv2/pywt/numpy boundary: these libs ship no usable element types; relax the
# unknown-type rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pywt

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Upstream EmbedMaxDct defaults -- must stay bit-identical to imwatermark.
_DEFAULT_SCALES = (0, 36, 36)
_DEFAULT_BLOCK = 4


class _EmbedMaxDct:
    """Frequency-domain embed/extract matching invisible-watermark dwtDct."""

    def __init__(
        self,
        watermarks: list[int] | None = None,
        wm_len: int = 8,
        scales: tuple[int, int, int] = _DEFAULT_SCALES,
        block: int = _DEFAULT_BLOCK,
    ) -> None:
        self._watermarks = watermarks or []
        self._wm_len = wm_len
        self._scales = scales
        self._block = block

    def encode(self, bgr: NDArray[Any]) -> NDArray[Any]:
        row, col, _channels = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        for channel in range(2):
            if self._scales[channel] <= 0:
                continue
            ca1, (h1, v1, d1) = pywt.dwt2(yuv[: row // 4 * 4, : col // 4 * 4, channel], "haar")
            self._encode_frame(ca1, self._scales[channel])
            yuv[: row // 4 * 4, : col // 4 * 4, channel] = pywt.idwt2((ca1, (v1, h1, d1)), "haar")

        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def decode(self, bgr: NDArray[Any]) -> NDArray[Any]:
        row, col, _channels = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        scores: list[list[int]] = [[] for _ in range(self._wm_len)]
        for channel in range(2):
            if self._scales[channel] <= 0:
                continue
            ca1, (_h1, _v1, _d1) = pywt.dwt2(yuv[: row // 4 * 4, : col // 4 * 4, channel], "haar")
            scores = self._decode_frame(ca1, self._scales[channel], scores)

        avg_scores = [float(np.array(s).mean()) if s else 0.0 for s in scores]
        return np.array(avg_scores) * 255 > 127

    def _decode_frame(self, frame: NDArray[Any], scale: int, scores: list[list[int]]) -> list[list[int]]:
        row, col = frame.shape
        num = 0
        for i in range(row // self._block):
            for j in range(col // self._block):
                block = frame[
                    i * self._block : i * self._block + self._block,
                    j * self._block : j * self._block + self._block,
                ]
                scores[num % self._wm_len].append(self._infer_dct_matrix(block, scale))
                num += 1
        return scores

    def _infer_dct_matrix(self, block: NDArray[Any], scale: int) -> int:
        pos = int(np.argmax(np.abs(block.flatten()[1:]))) + 1
        i, j = pos // self._block, pos % self._block
        val = abs(float(block[i][j]))
        return 1 if (val % scale) > 0.5 * scale else 0

    def _diffuse_dct_matrix(self, block: NDArray[Any], wm_bit: int, scale: int) -> NDArray[Any]:
        pos = int(np.argmax(np.abs(block.flatten()[1:]))) + 1
        i, j = pos // self._block, pos % self._block
        val = float(block[i][j])
        if val >= 0.0:
            block[i][j] = (val // scale + 0.25 + 0.5 * wm_bit) * scale
        else:
            val = abs(val)
            block[i][j] = -1.0 * (val // scale + 0.25 + 0.5 * wm_bit) * scale
        return block

    def _encode_frame(self, frame: NDArray[Any], scale: int) -> None:
        row, col = frame.shape
        num = 0
        for i in range(row // self._block):
            for j in range(col // self._block):
                block = frame[
                    i * self._block : i * self._block + self._block,
                    j * self._block : j * self._block + self._block,
                ]
                wm_bit = self._watermarks[num % self._wm_len]
                diffused = self._diffuse_dct_matrix(block, wm_bit, scale)
                frame[
                    i * self._block : i * self._block + self._block,
                    j * self._block : j * self._block + self._block,
                ] = diffused
                num += 1


def decode_dwt_dct(bgr: NDArray[Any], wm_len: int) -> NDArray[Any]:
    """Extract ``wm_len`` watermark bits from a BGR image (dwtDct / max-DCT)."""
    if bgr.size == 0 or min(bgr.shape[:2]) * max(bgr.shape[:2]) < 256 * 256:
        raise RuntimeError("image too small, should be larger than 256x256")
    return _EmbedMaxDct(wm_len=wm_len).decode(bgr)


def encode_dwt_dct(bgr: NDArray[Any], bits: list[int]) -> NDArray[Any]:
    """Embed bit list into a BGR image (dwtDct / max-DCT)."""
    if bgr.size == 0 or min(bgr.shape[:2]) * max(bgr.shape[:2]) < 256 * 256:
        raise RuntimeError("image too small, should be larger than 256x256")
    return _EmbedMaxDct(watermarks=bits, wm_len=len(bits)).encode(bgr)
