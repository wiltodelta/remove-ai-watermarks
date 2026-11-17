"""Lazy inference-only adaptation of FocalCodec v0.0.1 (cad8bf81d70875f756770449c34909d8b0951642).

Copyright 2025 Luca Della Libera, Apache-2.0. Submodules derive from Hugging Face
Transformers (Apache-2.0), Microsoft WavLM and FocalNet (MIT), Charactr Vocos
(MIT), and Phil Wang's lookup-free quantization (MIT). Exact license texts and
modification notices ship in licenses/focalcodec/.

Modified for this package: lazy Torch imports/builders, strict 355-key state
verification across all five checkpoint submodules without weight remapping or
remote code execution, and continuous WavLM-to-Vocos feature resynthesis. JIT
exports, Hub download helpers, demo/test harnesses, and the unused kNN matcher
(bshall/knn-vc) are omitted.
"""

from __future__ import annotations

import math
from importlib import import_module
from typing import Any


def build_focalcodec(config: dict[str, Any]) -> Any:
    """Instantiate the pinned v0.0.1 FocalCodec architecture without loading files."""
    torch: Any = import_module("torch")
    nn: Any = torch.nn
    functional: Any = torch.nn.functional

    expected_names = {
        "encoder_name": "WavLM",
        "compressor_name": "FocalEncoder",
        "quantizer_name": "BinarySphericalQuantizer",
        "decompressor_name": "FocalDecoder",
        "decoder_name": "Vocos",
    }
    if any(config.get(key) != value for key, value in expected_names.items()):
        raise ValueError("Only the pinned WavLM/FocalNet/BSQ/Vocos FocalCodec architecture is supported")

    class Fp32LayerNorm(nn.LayerNorm):
        def forward(self, x: Any) -> Any:
            weight = self.weight.float() if self.weight is not None else None
            bias = self.bias.float() if self.bias is not None else None
            out = functional.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps)
            return out.type_as(x)

    class ConvBlock(nn.Module):
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            kernel_size: int,
            stride: int,
            bias: bool = False,
            dropout: float = 0.0,
        ) -> None:
            nn.Module.__init__(self)
            self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, stride, bias=bias)
            self.dropout = nn.Dropout(dropout)
            self.norm = Fp32LayerNorm(output_dim, elementwise_affine=True)
            self.activation = nn.GELU()

        def forward(self, x: Any) -> Any:
            out = self.dropout(self.conv(x))
            out = self.norm(out.movedim(-1, -2)).movedim(-1, -2)
            return self.activation(out)

    class SamePad(nn.Module):
        def __init__(self, kernel_size: int, causal: bool = False) -> None:
            nn.Module.__init__(self)
            self.remove = (kernel_size - 1) if causal else (1 if kernel_size % 2 == 0 else 0)

        def forward(self, x: Any) -> Any:
            return x[..., : -self.remove] if self.remove > 0 else x

    class PositionalConvEmbedding(nn.Module):
        def __init__(self, dim: int, kernel_size: int, groups: int) -> None:
            nn.Module.__init__(self)
            self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=groups)
            self.padding = SamePad(kernel_size)
            self.activation = nn.GELU()

        def forward(self, x: Any) -> Any:
            out = self.padding(self.conv(x.movedim(-1, -2)))
            return self.activation(out).movedim(-1, -2)

    class MultiHeadAttention(nn.Module):
        def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
            nn.Module.__init__(self)
            self.num_heads = num_heads
            self.dropout = dropout
            head_dim = dim // num_heads
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.q_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
            self.gru_rel_pos_linear = nn.Linear(head_dim, 8)
            self.gru_rel_pos_const = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        def forward(self, x: Any, position_bias: Any) -> Any:
            batch, steps, _ = x.shape
            gated = x.reshape((*x.shape[:-1], self.num_heads, -1)).movedim(-3, -2)
            rel_proj = self.gru_rel_pos_linear(gated).reshape((*gated.shape[:-1], 2, 4)).sum(dim=-1)
            gate_a, gate_b = rel_proj.sigmoid().chunk(2, dim=-1)
            gated_bias = (gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0) * position_bias
            qs = self.q_proj(x).reshape(batch, steps, self.num_heads, -1).movedim(-3, -2)
            ks = self.k_proj(x).reshape(batch, steps, self.num_heads, -1).movedim(-3, -2)
            vs = self.v_proj(x).reshape(batch, steps, self.num_heads, -1).movedim(-3, -2)
            out = functional.scaled_dot_product_attention(
                qs,
                ks.type_as(qs),
                vs.type_as(qs),
                attn_mask=gated_bias.type_as(qs),
                dropout_p=0.0,
            )
            return self.out_proj(out.movedim(1, 2).contiguous().reshape(batch, steps, -1))

    class FeedForward(nn.Module):
        def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.0) -> None:
            nn.Module.__init__(self)
            self.in_proj = nn.Linear(dim, ffn_dim)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.out_proj = nn.Linear(ffn_dim, dim)

        def forward(self, x: Any) -> Any:
            out = self.dropout(self.activation(self.in_proj(x)))
            return self.dropout(self.out_proj(out))

    class TransformerLayer(nn.Module):
        def __init__(self, dim: int, ffn_dim: int, num_heads: int, dropout: float = 0.0) -> None:
            nn.Module.__init__(self)
            self.attention_norm = nn.LayerNorm(dim)
            self.attention = MultiHeadAttention(dim, num_heads, dropout)
            self.dropout = nn.Dropout(dropout)
            self.feed_forward_norm = nn.LayerNorm(dim)
            self.feed_forward = FeedForward(dim, ffn_dim, dropout)

        def forward(self, x: Any, position_bias: Any) -> Any:
            out = x + self.dropout(self.attention(self.attention_norm(x), position_bias))
            return out + self.feed_forward(self.feed_forward_norm(out))

    class TransformerEncoder(nn.Module):
        def __init__(
            self,
            num_layers: int,
            dim: int,
            ffn_dim: int,
            num_heads: int,
            num_buckets: int,
            max_distance: int,
            dropout: float,
            conv_pos: int,
            conv_pos_groups: int,
        ) -> None:
            nn.Module.__init__(self)
            self._num_buckets = num_buckets // 2
            self._num_buckets_minus_one = self._num_buckets - 1
            self._max_exact = self._num_buckets // 2
            self._num_buckets_minus_max_exact = self._num_buckets - self._max_exact
            self._log_max_distance_over_max_exact = math.log(max_distance / self._max_exact)
            self.positional_embedding = PositionalConvEmbedding(dim, conv_pos, conv_pos_groups)
            self.relative_embedding = nn.Embedding(num_buckets, num_heads)
            self.layers = nn.ModuleList(
                TransformerLayer(dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            )
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(dim)

        def _relative_positions_bucket(self, relative_positions: Any) -> Any:
            buckets = (relative_positions > 0).to(torch.long) * self._num_buckets
            abs_pos = relative_positions.abs()
            is_small = abs_pos < self._max_exact
            large = (abs_pos.float() / self._max_exact).log() / self._log_max_distance_over_max_exact
            large = (large * self._num_buckets_minus_max_exact + self._max_exact).to(torch.long)
            large = large.clamp(max=self._num_buckets_minus_one)
            return buckets + torch.where(is_small, abs_pos, large)

        def _compute_bias(self, length: int) -> Any:
            ctx = torch.arange(length, dtype=torch.long)[:, None]
            mem = torch.arange(length, dtype=torch.long)[None, :]
            buckets = self._relative_positions_bucket(mem - ctx).to(self.relative_embedding.weight.device)
            return self.relative_embedding(buckets).movedim(-1, 0)

        def forward(self, x: Any, output_norm: bool = False) -> Any:
            steps = int(x.shape[1])
            out = self.dropout(x + self.positional_embedding(x))
            position_bias = self._compute_bias(steps)[:, -steps:].type_as(out)
            for layer in self.layers:
                out = layer(out, position_bias)
            return self.norm(out) if output_norm else out

    class FeatureExtractor(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int] | tuple[int, ...],
            kernel_sizes: list[int] | tuple[int, ...],
            strides: list[int] | tuple[int, ...],
            dropout: float = 0.0,
        ) -> None:
            nn.Module.__init__(self)
            self.layers = nn.ModuleList()
            cur = input_dim
            for hidden_dim, kernel_size, stride in zip(hidden_dims, kernel_sizes, strides, strict=True):
                self.layers.append(ConvBlock(cur, hidden_dim, kernel_size, stride, False, dropout))
                cur = hidden_dim

        def forward(self, x: Any) -> Any:
            out = x.movedim(-1, -2)
            for layer in self.layers:
                out = layer(out)
            return out.movedim(-2, -1)

    class WavLM(nn.Module):
        def __init__(
            self,
            hidden_dims: list[int] | tuple[int, ...],
            kernel_sizes: list[int] | tuple[int, ...],
            strides: list[int] | tuple[int, ...],
            num_layers: int,
            dim: int,
            ffn_dim: int,
            num_heads: int,
            num_buckets: int,
            max_distance: int,
            dropout: float,
            conv_pos: int,
            conv_pos_groups: int,
        ) -> None:
            nn.Module.__init__(self)
            self.feature_extractor = FeatureExtractor(1, hidden_dims, kernel_sizes, strides, dropout)
            self.norm = nn.LayerNorm(hidden_dims[-1])
            self.feature_proj = nn.Linear(hidden_dims[-1], dim)
            self.dropout = nn.Dropout(dropout)
            self.encoder = TransformerEncoder(
                num_layers=num_layers,
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                num_buckets=num_buckets,
                max_distance=max_distance,
                dropout=dropout,
                conv_pos=conv_pos,
                conv_pos_groups=conv_pos_groups,
            )

        def forward(self, x: Any) -> Any:
            out = self.feature_extractor(x[..., None])
            out = self.dropout(self.feature_proj(self.norm(out)))
            return self.encoder(out, output_norm=False)

    class FocalModulation(nn.Module):
        def __init__(
            self,
            dim: int,
            focal_window: int,
            focal_level: int,
            focal_factor: int,
            dropout: float,
            use_post_norm: bool,
            normalize_modulator: bool,
        ) -> None:
            nn.Module.__init__(self)
            self.dim = dim
            self.focal_level = focal_level
            self.use_post_norm = use_post_norm
            self.normalize_modulator = normalize_modulator
            self.in_proj = nn.Linear(dim, 2 * dim + focal_level + 1)
            self.layers = nn.ModuleList()
            self.activation = nn.GELU()
            self.context_proj = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
            self.out_proj = nn.Linear(dim, dim)
            self.dropout = nn.Dropout(dropout)
            for level in range(focal_level):
                kernel_size = focal_factor * level + focal_window
                self.layers.append(
                    nn.Sequential(
                        nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, padding=kernel_size // 2),
                        nn.GELU(),
                    )
                )
            self.norm = nn.LayerNorm(dim) if use_post_norm else nn.Identity()

    class FocalBlock(nn.Module):
        def __init__(
            self,
            dim: int,
            ffn_dim: int,
            focal_window: int,
            focal_level: int,
            focal_factor: int,
            dropout: float,
            use_post_norm: bool,
            use_layerscale: bool,
            layerscale_init: float,
            normalize_modulator: bool,
        ) -> None:
            nn.Module.__init__(self)
            self.modulation_norm = nn.LayerNorm(dim)
            self.modulation = FocalModulation(
                dim=dim,
                focal_window=focal_window,
                focal_level=focal_level,
                focal_factor=focal_factor,
                dropout=dropout,
                use_post_norm=use_post_norm,
                normalize_modulator=normalize_modulator,
            )
            self.feed_forward_norm = nn.LayerNorm(dim)
            self.feed_forward = FeedForward(dim, ffn_dim, dropout)
            if use_layerscale:
                self.modulation_gamma = nn.Parameter(torch.full((dim,), layerscale_init))
                self.feed_forward_gamma = nn.Parameter(torch.full((dim,), layerscale_init))

    class Snake1d(nn.Module):
        def __init__(self, dim: int) -> None:
            nn.Module.__init__(self)
            self.alpha = nn.Parameter(torch.ones(dim, 1))

    class DownScale(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, kernel_size: int, stride: int) -> None:
            nn.Module.__init__(self)
            self.downscale = nn.Conv1d(input_dim, output_dim, kernel_size, stride)
            self.activation = Snake1d(output_dim)

    class UpScale(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, kernel_size: int, stride: int) -> None:
            nn.Module.__init__(self)
            self.activation = Snake1d(input_dim)
            self.upscale = nn.ConvTranspose1d(input_dim, output_dim, kernel_size=kernel_size, stride=stride)

    class FocalEncoder(nn.Module):
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: list[int] | tuple[int, ...],
            downscale_factors: list[int] | tuple[int, ...],
            focal_window: int,
            focal_level: int,
            focal_factor: int,
            dropout: float,
            use_post_norm: bool,
            use_layerscale: bool,
            layerscale_init: float,
            normalize_modulator: bool,
        ) -> None:
            nn.Module.__init__(self)
            self.layers = nn.ModuleList()
            cur = input_dim
            for hidden_dim, factor in zip(hidden_dims, downscale_factors, strict=True):
                self.layers.append(
                    nn.Sequential(
                        DownScale(cur, hidden_dim, factor, factor),
                        FocalBlock(
                            hidden_dim,
                            hidden_dim * 4,
                            focal_window,
                            focal_level,
                            focal_factor,
                            dropout,
                            use_post_norm,
                            use_layerscale,
                            layerscale_init,
                            normalize_modulator,
                        ),
                    )
                )
                cur = hidden_dim
            self.dropout = nn.Dropout(dropout)
            self.out_proj = nn.Linear(cur, output_dim)

    class FocalDecoder(nn.Module):
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: list[int] | tuple[int, ...],
            upscale_factors: list[int] | tuple[int, ...],
            focal_window: int,
            focal_level: int,
            focal_factor: int,
            dropout: float,
            use_post_norm: bool,
            use_layerscale: bool,
            layerscale_init: float,
            normalize_modulator: bool,
        ) -> None:
            nn.Module.__init__(self)
            dims = (*tuple(hidden_dims), output_dim)
            cur = dims[0]
            self.in_proj = nn.Linear(input_dim, cur)
            self.dropout = nn.Dropout(dropout)
            self.layers = nn.ModuleList()
            for hidden_dim, factor in zip(dims[1:], upscale_factors, strict=True):
                self.layers.append(
                    nn.Sequential(
                        FocalBlock(
                            cur,
                            cur * 4,
                            focal_window,
                            focal_level,
                            focal_factor,
                            dropout,
                            use_post_norm,
                            use_layerscale,
                            layerscale_init,
                            normalize_modulator,
                        ),
                        UpScale(cur, hidden_dim, factor, factor),
                    )
                )
                cur = hidden_dim

    class BinarySphericalQuantizer(nn.Module):
        def __init__(self, codebook_size: int = 8192) -> None:
            nn.Module.__init__(self)
            dim = int(math.log2(codebook_size))
            codebook_value = 1 / math.sqrt(dim)
            self.register_buffer("mask", 2 ** torch.arange(dim - 1, -1, -1))
            all_codes = torch.arange(codebook_size)
            bits = ((all_codes[..., None].int() & self.mask) != 0).float()
            self.register_buffer("codebook", ((bits * 2 - 1) * codebook_value).float(), persistent=False)

    class ConvNeXtBlock(nn.Module):
        def __init__(
            self,
            dim: int,
            ffn_dim: int,
            kernel_size: int,
            padding: int,
            layerscale_init: float | None,
        ) -> None:
            nn.Module.__init__(self)
            self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim)
            self.norm = nn.LayerNorm(dim, eps=1e-6)
            self.pwconv1 = nn.Linear(dim, ffn_dim)
            self.activation = nn.GELU()
            self.pwconv2 = nn.Linear(ffn_dim, dim)
            self.gamma = nn.Parameter(torch.full((dim,), layerscale_init)) if layerscale_init is not None else None

        def forward(self, x: Any) -> Any:
            out = self.norm(self.dwconv(x).movedim(-1, -2))
            out = self.pwconv2(self.activation(self.pwconv1(out)))
            if self.gamma is not None:
                out = self.gamma * out
            return x + out.movedim(-1, -2)

    class ISTFT(nn.Module):
        def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
            nn.Module.__init__(self)
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.win_length = win_length
            self.pad = (win_length - hop_length) // 2
            self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        def forward(self, spec: Any) -> Any:
            ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward") * self.window[None, :, None]
            frames = int(spec.shape[-1])
            output_size = (frames - 1) * self.hop_length + self.win_length
            out = functional.fold(
                ifft,
                output_size=(1, output_size),
                kernel_size=(1, self.win_length),
                stride=(1, self.hop_length),
            )[:, 0, 0, self.pad : -self.pad]
            window_sq = (self.window**2).expand(1, frames, -1).movedim(-1, -2)
            envelope = functional.fold(
                window_sq,
                output_size=(1, output_size),
                kernel_size=(1, self.win_length),
                stride=(1, self.hop_length),
            ).squeeze()[self.pad : -self.pad]
            return out / envelope

    class VocosBackbone(nn.Module):
        def __init__(
            self,
            input_channels: int,
            num_layers: int,
            dim: int,
            ffn_dim: int,
            kernel_size: int = 7,
            padding: int = 3,
            layerscale_init: float | None = None,
        ) -> None:
            nn.Module.__init__(self)
            scale = 1 / num_layers if layerscale_init is None else layerscale_init
            self.embedding = nn.Conv1d(input_channels, dim, kernel_size, padding=padding)
            self.input_norm = nn.LayerNorm(dim, eps=1e-6)
            self.layers = nn.ModuleList(
                ConvNeXtBlock(dim, ffn_dim, kernel_size, padding, scale) for _ in range(num_layers)
            )
            self.output_norm = nn.LayerNorm(dim, eps=1e-6)

        def forward(self, x: Any) -> Any:
            out = self.input_norm(self.embedding(x).movedim(-1, -2)).movedim(-1, -2)
            for layer in self.layers:
                out = layer(out)
            return self.output_norm(out.movedim(-1, -2)).movedim(-1, -2)

    class ISTFTHead(nn.Module):
        def __init__(self, dim: int, n_fft: int, hop_length: int) -> None:
            nn.Module.__init__(self)
            self.proj = nn.Linear(dim, n_fft + 2)
            self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft)

        def forward(self, x: Any) -> Any:
            mag, phase = self.proj(x.movedim(-1, -2)).movedim(-1, -2).chunk(2, dim=-2)
            mag = mag.exp().clamp(max=1e2)
            spec = mag * torch.complex(phase.cos(), phase.sin())
            return self.istft(spec)

    class Vocos(nn.Module):
        def __init__(
            self,
            input_channels: int,
            num_layers: int,
            dim: int,
            ffn_dim: int,
            kernel_size: int = 7,
            padding: int = 3,
            layerscale_init: float | None = None,
            n_fft: int = 1024,
            hop_length: int = 320,
        ) -> None:
            nn.Module.__init__(self)
            self.backbone = VocosBackbone(
                input_channels=input_channels,
                num_layers=num_layers,
                dim=dim,
                ffn_dim=ffn_dim,
                kernel_size=kernel_size,
                padding=padding,
                layerscale_init=layerscale_init,
            )
            self.head = ISTFTHead(dim, n_fft, hop_length)

        def forward(self, x: Any) -> Any:
            return self.head(self.backbone(x.movedim(-1, -2)))

    class FocalCodecModel(nn.Module):
        sample_rate = 16000

        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.encoder = WavLM(**config["encoder_config"])
            self.compressor = FocalEncoder(**config["compressor_config"])
            self.quantizer = BinarySphericalQuantizer(**config["quantizer_config"])
            self.decompressor = FocalDecoder(**config["decompressor_config"])
            self.decoder = Vocos(**config["decoder_config"])

        def resynthesize_features(self, padded_wave_16k: Any) -> Any:
            """Encode continuous 50 Hz WavLM features and synthesize via Vocos."""
            return self.decoder(self.encoder(padded_wave_16k))

    return FocalCodecModel()
