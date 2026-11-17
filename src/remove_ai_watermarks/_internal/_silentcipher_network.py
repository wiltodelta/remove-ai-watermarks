"""SilentCipher 1.0.5 decoder architecture and STFT, with lazy Torch imports.

Derived from Sony's model.py and stft.py in the published silentcipher wheel.
The architecture, state keys, training-mode batch normalization, and padding
match that implementation. Encoder and server dependencies are not needed.

MIT License
Copyright (c) 2024 Sony Research Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from torch import Tensor


def build_decoder(*, message_dim: int = 5, message_band_size: int = 1024) -> Any:
    """Build a released decoder shape without importing the old SDK."""
    import torch
    from torch import nn

    class Layer(nn.Module):
        def __init__(self, dim_in: int, dim_out: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, dtype=torch.float32)
            self.gate = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, dtype=torch.float32)
            self.bn = nn.BatchNorm2d(dim_out, dtype=torch.float32)

        def forward(self, x: Tensor) -> Tensor:
            return cast("Tensor", self.bn(self.conv(x) * torch.sigmoid(self.gate(x))))

    class Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.message_band_size = message_band_size
            layers: list[nn.Module] = [nn.Dropout(0), Layer(1, 128)]
            for _ in range(8):
                layers.extend((nn.Dropout(0), Layer(128, 128)))
            layers.extend((nn.Dropout(0), Layer(128, message_dim)))
            self.main = nn.Sequential(*layers)
            self.linear = nn.Linear(message_band_size, 1, dtype=torch.float32)

        def forward(self, x: Tensor) -> Tensor:
            hidden = self.main(x[:, :, : self.message_band_size])
            return cast("Tensor", self.linear(hidden.transpose(2, 3)).squeeze(3).unsqueeze(1))

    return Decoder()


def magnitude(signal: Any, window: Any, *, n_fft: int = 4096, hop_length: int = 2048) -> Any:
    """Retain the upstream padding and zero-magnitude epsilon convention."""
    import torch

    signal = torch.nn.functional.pad(signal, (0, n_fft - signal.shape[1] % n_fft))
    fft = torch.stft(signal, n_fft, hop_length, n_fft, window=window.to(signal.device), return_complex=True)
    squared = fft.real**2 + fft.imag**2
    epsilon = torch.ones_like(squared) * (squared == 0).float() * 1e-24
    return torch.sqrt(squared + epsilon) - torch.sqrt(epsilon)
