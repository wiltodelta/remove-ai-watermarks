"""Lazy inference-only adaptation of SNAC 44.1 kHz (hubertsiuzdak/snac).

Copyright (c) 2024 Hubert Siuzdak, MIT. Submodules derive from Descript Audio
Codec (MIT) and Phil Wang's local-attention (MIT). Exact license text and
modification notices ship in licenses/snac/.
"""

from __future__ import annotations

import math
from importlib import import_module
from typing import Any


def build_snac(config: dict[str, Any]) -> Any:
    """Instantiate the pinned 44.1 kHz multi-scale SNAC network without loading files."""
    torch: Any = import_module("torch")
    nn: Any = torch.nn
    functional: Any = torch.nn.functional
    parametrizations: Any = torch.nn.utils.parametrizations

    encoder_dim = int(config["encoder_dim"])
    encoder_rates = tuple(int(r) for r in config["encoder_rates"])
    decoder_dim = int(config["decoder_dim"])
    decoder_rates = tuple(int(r) for r in config["decoder_rates"])
    attn_window_size = int(config["attn_window_size"])
    codebook_size = int(config["codebook_size"])
    codebook_dim = int(config["codebook_dim"])
    vq_strides = tuple(int(s) for s in config["vq_strides"])
    use_noise = bool(config["noise"])
    use_depthwise = bool(config["depthwise"])
    hop_length = math.prod(encoder_rates) * max(vq_strides)

    def wn_conv1d(*args: Any, **kwargs: Any) -> Any:
        return parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))

    def wn_conv_transpose1d(*args: Any, **kwargs: Any) -> Any:
        return parametrizations.weight_norm(nn.ConvTranspose1d(*args, **kwargs))

    def rotate_half(x: Any) -> Any:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    class Snake1d(nn.Module):
        def __init__(self, channels: int) -> None:
            nn.Module.__init__(self)
            self.alpha = nn.Parameter(torch.ones(1, channels, 1))

        def forward(self, x: Any) -> Any:
            return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)

    class ResidualUnit(nn.Module):
        def __init__(self, dim: int, dilation: int = 1, kernel: int = 7, groups: int = 1) -> None:
            nn.Module.__init__(self)
            pad = ((kernel - 1) * dilation) // 2
            self.block = nn.Sequential(
                Snake1d(dim),
                wn_conv1d(dim, dim, kernel_size=kernel, dilation=dilation, padding=pad, groups=groups),
                Snake1d(dim),
                wn_conv1d(dim, dim, kernel_size=1),
            )

        def forward(self, x: Any) -> Any:
            y = self.block(x)
            pad = (x.shape[-1] - y.shape[-1]) // 2
            if pad > 0:
                x = x[..., pad:-pad]
            return x + y

    class SinusoidalEmbeddings(nn.Module):
        def __init__(self, dim: int) -> None:
            nn.Module.__init__(self)
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            self.register_buffer("inv_freq", inv_freq)

        def forward(self, length: int, device: Any, dtype: Any) -> tuple[Any, Any]:
            inv_freq: Any = self._buffers["inv_freq"]
            pos = torch.arange(length, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", pos, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(dtype=dtype)
            return emb.cos(), emb.sin()

    class LocalMHA(nn.Module):
        def __init__(self, dim: int, window_size: int = 32, dim_head: int = 64) -> None:
            nn.Module.__init__(self)
            self.norm = nn.LayerNorm(dim)
            self.heads = dim // dim_head
            self.dim_head = dim_head
            self.window_size = window_size
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
            self.rel_pos = SinusoidalEmbeddings(dim_head)
            self.to_out = nn.Linear(dim, dim, bias=False)
            cos, sin = self.rel_pos(window_size, torch.device("cpu"), torch.float32)
            self.register_buffer("_rope_cos", cos[None, None, None, :, :], persistent=False)
            self.register_buffer("_rope_sin", sin[None, None, None, :, :], persistent=False)

        def forward(self, x: Any) -> Any:
            batch, channels, steps = x.shape
            residual = x
            xt = self.norm(x.transpose(1, 2))
            ws = self.window_size
            pad_t = (ws - (steps % ws)) % ws
            if pad_t:
                xt = functional.pad(xt, (0, 0, 0, pad_t))
            padded_steps = xt.shape[1]
            windows = padded_steps // ws
            q, k, v = self.to_qkv(xt).chunk(3, dim=-1)
            q = q.view(batch, windows, ws, self.heads, self.dim_head).permute(0, 3, 1, 2, 4)
            k = k.view(batch, windows, ws, self.heads, self.dim_head).permute(0, 3, 1, 2, 4)
            v = v.view(batch, windows, ws, self.heads, self.dim_head).permute(0, 3, 1, 2, 4)
            cos = self._buffers["_rope_cos"].to(dtype=x.dtype)
            sin = self._buffers["_rope_sin"].to(dtype=x.dtype)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
            out = functional.scaled_dot_product_attention(q, k, v)
            out = out.permute(0, 2, 3, 1, 4).reshape(batch, padded_steps, channels)
            if pad_t:
                out = out[:, :steps, :]
            return residual + self.to_out(out).transpose(1, 2)

    class EncoderBlock(nn.Module):
        def __init__(self, out_dim: int, stride: int, *, depthwise: bool = True) -> None:
            nn.Module.__init__(self)
            in_dim = out_dim // 2
            groups = in_dim if depthwise else 1
            self.block = nn.Sequential(
                ResidualUnit(in_dim, dilation=1, groups=groups),
                ResidualUnit(in_dim, dilation=3, groups=groups),
                ResidualUnit(in_dim, dilation=9, groups=groups),
                Snake1d(in_dim),
                wn_conv1d(in_dim, out_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
            )

        def forward(self, x: Any) -> Any:
            return self.block(x)

    class NoiseBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            nn.Module.__init__(self)
            self.linear = wn_conv1d(dim, dim, kernel_size=1, bias=False)

        def forward(self, x: Any) -> Any:
            batch, _channels, steps = x.shape
            gen = torch.Generator(device="cpu").manual_seed(0)
            noise = torch.randn((batch, 1, steps), generator=gen, dtype=torch.float32).to(
                device=x.device,
                dtype=x.dtype,
            )
            return x + noise * self.linear(x)

    class DecoderBlock(nn.Module):
        def __init__(
            self, in_dim: int, out_dim: int, stride: int, *, noise: bool = False, depthwise: bool = True
        ) -> None:
            nn.Module.__init__(self)
            groups = out_dim if depthwise else 1
            layers: list[Any] = [
                Snake1d(in_dim),
                wn_conv_transpose1d(
                    in_dim,
                    out_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                    output_padding=stride % 2,
                ),
            ]
            if noise:
                layers.append(NoiseBlock(out_dim))
            layers.extend(
                (
                    ResidualUnit(out_dim, dilation=1, groups=groups),
                    ResidualUnit(out_dim, dilation=3, groups=groups),
                    ResidualUnit(out_dim, dilation=9, groups=groups),
                )
            )
            self.block = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            return self.block(x)

    class VectorQuantize(nn.Module):
        def __init__(self, input_dim: int, cb_size: int, cb_dim: int, stride: int = 1) -> None:
            nn.Module.__init__(self)
            self.stride = stride
            self.in_proj = wn_conv1d(input_dim, cb_dim, kernel_size=1)
            self.out_proj = wn_conv1d(cb_dim, input_dim, kernel_size=1)
            self.codebook = nn.Embedding(cb_size, cb_dim)
            self.register_buffer("_cb_norm", None, persistent=False)

        def encode(self, z: Any) -> Any:
            if self.stride > 1:
                z = functional.avg_pool1d(z, self.stride, stride=self.stride)
            ze_norm = functional.normalize(self.in_proj(z), dim=1)
            cb_norm = self._cb_norm if self._cb_norm is not None else functional.normalize(self.codebook.weight, dim=1)
            return torch.einsum("bdt,nd->bnt", ze_norm, cb_norm).argmax(1)

        def decode_code(self, indices: Any) -> Any:
            zq = functional.embedding(indices, self.codebook.weight).transpose(1, 2)
            out = self.out_proj(zq)
            if self.stride > 1:
                out = out.repeat_interleave(self.stride, dim=-1)
            return out

    class ResidualVectorQuantize(nn.Module):
        def __init__(self, input_dim: int, cb_size: int, cb_dim: int, strides: tuple[int, ...]) -> None:
            nn.Module.__init__(self)
            self.quantizers = nn.ModuleList(
                [VectorQuantize(input_dim, cb_size, cb_dim, stride=stride) for stride in strides]
            )

        def forward(self, z: Any, *, n_quantizers: int | None = None) -> Any:
            active = len(self.quantizers) if n_quantizers is None else n_quantizers
            if active < 1 or active > len(self.quantizers):
                raise ValueError(f"SNAC n_quantizers must be between 1 and {len(self.quantizers)}")
            z_q = torch.zeros_like(z)
            residual = z
            for idx in range(active):
                quantizer = self.quantizers[idx]
                codes = quantizer.encode(residual)
                z_i = quantizer.decode_code(codes)
                z_q = z_q + z_i
                residual = residual - z_i
            return z_q

    class Encoder(nn.Module):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            dim = encoder_dim
            layers: list[Any] = [wn_conv1d(1, dim, kernel_size=7, padding=3)]
            for stride in encoder_rates:
                dim *= 2
                layers.append(EncoderBlock(dim, stride=stride, depthwise=use_depthwise))
            layers.append(LocalMHA(dim=dim, window_size=attn_window_size))
            layers.append(wn_conv1d(dim, dim, kernel_size=7, padding=3, groups=dim))
            self.block = nn.Sequential(*layers)
            self.output_dim = dim

        def forward(self, x: Any) -> Any:
            return self.block(x)

    class Decoder(nn.Module):
        def __init__(self, latent_dim: int) -> None:
            nn.Module.__init__(self)
            layers: list[Any] = [
                wn_conv1d(latent_dim, latent_dim, kernel_size=7, padding=3, groups=latent_dim),
                wn_conv1d(latent_dim, decoder_dim, kernel_size=1),
                LocalMHA(dim=decoder_dim, window_size=attn_window_size),
            ]
            dim = decoder_dim
            for stride in decoder_rates:
                layers.append(DecoderBlock(dim, dim // 2, stride=stride, noise=use_noise, depthwise=use_depthwise))
                dim //= 2
            layers.extend(
                (
                    Snake1d(dim),
                    wn_conv1d(dim, 1, kernel_size=7, padding=3),
                    nn.Tanh(),
                )
            )
            self.model = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            return self.model(x)

    class SNACModel(nn.Module):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.hop_length = hop_length
            self.encoder = Encoder()
            self.quantizer = ResidualVectorQuantize(
                self.encoder.output_dim,
                codebook_size,
                codebook_dim,
                vq_strides,
            )
            self.decoder = Decoder(self.encoder.output_dim)

        def fuse_weight_norm(self) -> None:
            remove = torch.nn.utils.parametrize.remove_parametrizations
            for module in self.modules():
                if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
                    remove(module, "weight", leave_parametrized=True)
                if isinstance(module, VectorQuantize):
                    module.register_buffer(
                        "_cb_norm",
                        functional.normalize(module.codebook.weight.detach(), dim=1),
                        persistent=False,
                    )

        def reconstruct(self, audio_values: Any, *, n_quantizers: int | None = None) -> Any:
            length = int(audio_values.shape[-1])
            pad = (self.hop_length - (length % self.hop_length)) % self.hop_length
            padded = functional.pad(audio_values, (0, pad)) if pad else audio_values
            latents = self.encoder(padded)
            quantized = self.quantizer(latents, n_quantizers=n_quantizers)
            decoded = self.decoder(quantized)
            return decoded[..., :length]

    return SNACModel()
