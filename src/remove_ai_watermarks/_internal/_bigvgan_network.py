"""Lazy inference-only adaptation of NVIDIA BigVGAN at 7d2b454564a6c7d014227f635b7423881f14bdac.

Copyright (c) 2024 NVIDIA CORPORATION, MIT. SnakeBeta derives from Edward
Dixon's MIT implementation. Alias-free activation derives from alias-free-torch
(Apache-2.0), including Alexandre Defossez's MIT Julius filter design.
Exact license texts and modification/source notices: licenses/bigvgan/.

Modified for this package: lazy imports/builders, only AMPBlock1/SnakeBeta and
pure Torch inference, no training initialization, plotting, CUDA extension or
Hub mixin. Parameter/buffer names and inference mathematics are preserved.
The model's own final clamp is retained; callers must not add output clipping.
"""

from __future__ import annotations

import math
from importlib import import_module
from typing import Any


def build_generator(config: dict[str, Any]) -> Any:
    """Build the checkpoint-compatible inference network without loading files."""
    torch = import_module("torch")
    nn = torch.nn
    functional = torch.nn.functional
    weight_norm = torch.nn.utils.weight_norm
    remove_weight_norm = torch.nn.utils.remove_weight_norm
    if config["resblock"] != "1" or config["activation"] != "snakebeta" or not config["snake_logscale"]:
        raise ValueError("Only the pinned AMPBlock1/SnakeBeta log-scale architecture is supported")

    # Apache-2.0 alias-free-torch, with the MIT Julius low-pass filter design.
    def kaiser_filter(cutoff: float, half_width: float, kernel_size: int) -> Any:
        half_size = kernel_size // 2
        attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
        if attenuation > 50:
            beta = 0.1102 * (attenuation - 8.7)
        elif attenuation >= 21:
            beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21)
        else:
            beta = 0.0
        window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
        time = torch.arange(-half_size, half_size) + 0.5
        kernel = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        kernel /= kernel.sum()
        return kernel.view(1, 1, kernel_size)

    class LowPassFilter1d(nn.Module):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.register_buffer("filter", kaiser_filter(0.25, 0.3, 12))

        def forward(self, x: Any) -> Any:
            channels = x.shape[1]
            x = functional.pad(x, (5, 6), mode="replicate")
            return functional.conv1d(x, self.filter.expand(channels, -1, -1), stride=2, groups=channels)

    class UpSample1d(nn.Module):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.register_buffer("filter", kaiser_filter(0.25, 0.3, 12))

        def forward(self, x: Any) -> Any:
            channels = x.shape[1]
            x = functional.pad(x, (5, 5), mode="replicate")
            x = 2 * functional.conv_transpose1d(x, self.filter.expand(channels, -1, -1), stride=2, groups=channels)
            return x[..., 15:-15]

    class DownSample1d(nn.Module):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.lowpass = LowPassFilter1d()

        def forward(self, x: Any) -> Any:
            return self.lowpass(x)

    # MIT Edward Dixon / NVIDIA SnakeBeta, fixed log-scale parameterization.
    class SnakeBeta(nn.Module):
        def __init__(self, channels: int) -> None:
            nn.Module.__init__(self)
            self.alpha = nn.Parameter(torch.zeros(channels))
            self.beta = nn.Parameter(torch.zeros(channels))

        def forward(self, x: Any) -> Any:
            alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
            beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
            return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)

    # Apache-2.0 alias-free-torch composition.
    class Activation1d(nn.Module):
        def __init__(self, channels: int) -> None:
            nn.Module.__init__(self)
            self.act = SnakeBeta(channels)
            self.upsample = UpSample1d()
            self.downsample = DownSample1d()

        def forward(self, x: Any) -> Any:
            return self.downsample(self.act(self.upsample(x)))

    # NVIDIA MIT AMPBlock1 and generator.
    class AMPBlock1(nn.Module):
        def __init__(self, channels: int, kernel: int, dilation: list[int]) -> None:
            nn.Module.__init__(self)
            self.convs1 = nn.ModuleList(
                [
                    weight_norm(nn.Conv1d(channels, channels, kernel, 1, dilation=d, padding=(kernel * d - d) // 2))
                    for d in dilation
                ]
            )
            self.convs2 = nn.ModuleList(
                [
                    weight_norm(nn.Conv1d(channels, channels, kernel, 1, dilation=1, padding=(kernel - 1) // 2))
                    for _ in dilation
                ]
            )
            self.activations = nn.ModuleList([Activation1d(channels) for _ in range(2 * len(dilation))])

        def forward(self, x: Any) -> Any:
            for c1, c2, a1, a2 in zip(
                self.convs1, self.convs2, self.activations[::2], self.activations[1::2], strict=True
            ):
                x = c2(a2(c1(a1(x)))) + x
            return x

        def remove_weight_norm(self) -> None:
            for layer in [*self.convs1, *self.convs2]:
                remove_weight_norm(layer)

    class BigVGAN(nn.Module):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            initial = config["upsample_initial_channel"]
            self.num_kernels = len(config["resblock_kernel_sizes"])
            self.conv_pre = weight_norm(nn.Conv1d(config["num_mels"], initial, 7, 1, padding=3))
            self.ups = nn.ModuleList()
            self.resblocks = nn.ModuleList()
            for index, (rate, kernel) in enumerate(
                zip(config["upsample_rates"], config["upsample_kernel_sizes"], strict=True)
            ):
                channels = initial // (2 ** (index + 1))
                self.ups.append(
                    nn.ModuleList(
                        [
                            weight_norm(
                                nn.ConvTranspose1d(
                                    initial // (2**index), channels, kernel, rate, padding=(kernel - rate) // 2
                                )
                            )
                        ]
                    )
                )
                for res_kernel, dilation in zip(
                    config["resblock_kernel_sizes"], config["resblock_dilation_sizes"], strict=True
                ):
                    self.resblocks.append(AMPBlock1(channels, res_kernel, dilation))
            channels = initial // (2 ** len(config["upsample_rates"]))
            self.activation_post = Activation1d(channels)
            self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3, bias=config["use_bias_at_final"]))

        def forward(self, x: Any) -> Any:
            x = self.conv_pre(x)
            for index, upsamplers in enumerate(self.ups):
                for upsampler in upsamplers:
                    x = upsampler(x)
                combined = self.resblocks[index * self.num_kernels](x)
                for offset in range(1, self.num_kernels):
                    combined += self.resblocks[index * self.num_kernels + offset](x)
                x = combined / self.num_kernels
            x = self.conv_post(self.activation_post(x))
            return torch.tanh(x) if config["use_tanh_at_final"] else torch.clamp(x, min=-1.0, max=1.0)

        def remove_weight_norm(self) -> None:
            for upsamplers in self.ups:
                for layer in upsamplers:
                    remove_weight_norm(layer)
            for block in self.resblocks:
                block.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)

    return BigVGAN()


def mel_spectrogram(samples: Any) -> Any:
    """Pinned NVIDIA mel frontend, evaluated in FP32 on CPU like qualification."""
    torch = import_module("torch")
    librosa = import_module("librosa")
    y = torch.as_tensor(samples, dtype=torch.float32, device="cpu").unsqueeze(0)
    basis = torch.from_numpy(librosa.filters.mel(sr=44100, n_fft=2048, n_mels=128, fmin=0, fmax=None)).float()
    window = torch.hann_window(2048, dtype=torch.float32, device="cpu")
    y = torch.nn.functional.pad(y.unsqueeze(1), (768, 768), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        2048,
        hop_length=512,
        win_length=2048,
        window=window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    return torch.log(torch.clamp(torch.matmul(basis, magnitude), min=1e-5))
