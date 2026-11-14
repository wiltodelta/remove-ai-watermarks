"""Tests for the abstaining 10-provider source-pipeline classifier (source_classify)."""

from __future__ import annotations

import json
import re
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image, PngImagePlugin

import remove_ai_watermarks as raiw
from remove_ai_watermarks import source_classify
from remove_ai_watermarks.cli import main

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def _add_stage2_hgb_payload(
    payload: dict[str, Any],
    cls_name: str,
    baseline: Sequence[float],
) -> None:
    """Populate a minimal constant-leaf Stage-2 HGB head for `cls_name`."""
    c_tree = len(baseline)
    payload["stage2_version_classes"] = np.asarray([cls_name], dtype="<U32")
    payload[f"s2_{cls_name}_subfamilies"] = np.asarray([f"{cls_name}:v1", f"{cls_name}:v2"], dtype="<U48")
    payload[f"s2_{cls_name}_hgb_baseline"] = np.asarray(baseline, dtype=np.float32)
    payload[f"s2_{cls_name}_hgb_max_depth"] = np.asarray(1, dtype=np.int16)
    payload[f"s2_{cls_name}_hgb_feat_idx"] = np.zeros((1, c_tree, 1), dtype=np.int16)
    payload[f"s2_{cls_name}_hgb_thresholds"] = np.full((1, c_tree, 1), np.inf, dtype=np.float32)
    payload[f"s2_{cls_name}_hgb_left"] = np.zeros((1, c_tree, 1), dtype=np.uint8)
    payload[f"s2_{cls_name}_hgb_right"] = np.zeros((1, c_tree, 1), dtype=np.uint8)
    payload[f"s2_{cls_name}_hgb_values"] = np.zeros((1, c_tree, 1), dtype=np.float32)


def _write_synthetic_v3_npz(
    path: Path,
    *,
    schema: str = source_classify.MODEL_SCHEMA_V3,
    target_class: str = "openai",
    target_global_bias: float = 4.0,
    target_sub_bias: float = -3.0,
    target_sub_alpha: float = 0.85,
    router_threshold: float = 1.0,
    stage2_win_idx: int = 1,
) -> Path:
    """Create a minimal valid pickle-free multi-provider-source-v3 .npz fixture."""
    d = source_classify.FEATURE_DIM
    payload: dict[str, Any] = {
        "schema": np.asarray(schema),
        "feature_dim": np.asarray(d, dtype=np.int64),
        "provider_classes": np.asarray(source_classify.ALL_PROVIDER_CLASSES, dtype="<U32"),
        "damping_gamma": np.asarray(0.60, dtype=np.float64),
        "mean": np.zeros(d, dtype=np.float64),
        "scale": np.ones(d, dtype=np.float64),
        "router_thresholds": np.full(len(source_classify.ALL_PROVIDER_CLASSES), router_threshold, dtype=np.float64),
    }

    def add_mlp(prefix: str, bias_val: float) -> None:
        payload[f"{prefix}_w0"] = np.zeros((4, d), dtype=np.float32)
        payload[f"{prefix}_b0"] = np.zeros(4, dtype=np.float32)
        payload[f"{prefix}_w1"] = np.zeros((2, 4), dtype=np.float32)
        payload[f"{prefix}_b1"] = np.zeros(2, dtype=np.float32)
        payload[f"{prefix}_w2"] = np.zeros((1, 2), dtype=np.float32)
        payload[f"{prefix}_b2"] = np.asarray([bias_val], dtype=np.float32)

    def add_unit(prefix: str, bias_val: float, *, is_sub: bool, alpha: float = 0.0) -> None:
        payload[f"{prefix}_blend_weights"] = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        payload[f"{prefix}_base_offset"] = np.asarray(0.0, dtype=np.float64)
        if is_sub:
            payload[f"{prefix}_pool_alpha"] = np.asarray(alpha, dtype=np.float64)
        payload[f"{prefix}_soup_count"] = np.asarray(1, dtype=np.int16)
        add_mlp(f"{prefix}_s0", bias_val)
        add_mlp(f"{prefix}_et_mlp", 0.0)
        payload[f"{prefix}_hgb_baseline"] = np.asarray(0.0, dtype=np.float64)
        payload[f"{prefix}_hgb_max_depth"] = np.asarray(1, dtype=np.int16)
        payload[f"{prefix}_hgb_feat_idx"] = np.zeros((1, 1), dtype=np.int16)
        payload[f"{prefix}_hgb_thresholds"] = np.asarray([[np.inf]], dtype=np.float64)
        payload[f"{prefix}_hgb_left"] = np.zeros((1, 1), dtype=np.uint8)
        payload[f"{prefix}_hgb_right"] = np.zeros((1, 1), dtype=np.uint8)
        payload[f"{prefix}_hgb_values"] = np.zeros((1, 1), dtype=np.float64)

    desc_classes: list[str] = []
    desc_names: list[str] = []
    for cls_name in source_classify.ALL_PROVIDER_CLASSES:
        is_target = cls_name == target_class
        g_bias = target_global_bias if is_target else -3.0
        add_unit(f"s1_{cls_name}_global", g_bias, is_sub=False)
        payload[f"s1_{cls_name}_pool_recenter"] = np.asarray(0.0, dtype=np.float64)
        desc_classes.append(cls_name)
        desc_names.append("__global__")

        sub_name = f"{cls_name}:v_sub_a"
        payload[f"s1_{cls_name}_sub_names"] = np.asarray([sub_name], dtype="<U48")
        s_bias = target_sub_bias if is_target else -3.0
        s_alpha = target_sub_alpha if is_target else 0.0
        add_unit(f"s1_{cls_name}_sub0", s_bias, is_sub=True, alpha=s_alpha)
        desc_classes.append(cls_name)
        desc_names.append(sub_name)

    payload["unit_descriptor_classes"] = np.asarray(desc_classes, dtype="<U32")
    payload["unit_descriptor_names"] = np.asarray(desc_names, dtype="<U48")

    _add_stage2_hgb_payload(payload, target_class, [-1.0, 1.0] if stage2_win_idx == 1 else [1.0, -1.0])

    np.savez(path, **payload)
    return path


def test_weights_revision_and_sha256_are_pinned_hex() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", source_classify.WEIGHTS_REVISION)
    assert re.fullmatch(r"[0-9a-f]{64}", source_classify.MODEL_SHA256)
    assert raiw.classify_source is source_classify.classify_source
    assert raiw.SourceClassification is source_classify.SourceClassification


def test_log_spectrum_discards_translation_phase() -> None:
    from remove_ai_watermarks._internal.source_spectral import phase_free_log_spectrum

    rng = np.random.default_rng(17)
    channel = rng.normal(size=(32, 32))

    original = phase_free_log_spectrum(channel)
    translated = phase_free_log_spectrum(np.roll(channel, shift=(5, -7), axis=(0, 1)))

    assert np.allclose(original, translated, rtol=1e-12, atol=1e-12)


def test_spectral_feature_is_finite_and_has_the_frozen_width() -> None:
    from remove_ai_watermarks._internal.source_spectral import feature_from_image, make_geometry

    rng = np.random.default_rng(20260916)
    pixels = rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
    feature = feature_from_image(Image.fromarray(pixels), make_geometry())

    assert feature.shape == (768,)
    assert np.all(np.isfinite(feature))


def test_augment_892d_to_956d_produces_finite_956d_for_1d_and_2d() -> None:
    rng = np.random.default_rng(42)
    vec_892 = rng.normal(size=892).astype(np.float64)
    out_1d = source_classify.augment_892d_to_956d(vec_892)
    assert out_1d.shape == (956,)
    assert np.all(np.isfinite(out_1d))

    mat_892 = np.stack([vec_892, vec_892 * 0.5], axis=0)
    out_2d = source_classify.augment_892d_to_956d(mat_892)
    assert out_2d.shape == (2, 956)
    np.testing.assert_allclose(out_2d[0], out_1d)

    with pytest.raises(ValueError, match="Expected 892-d"):
        source_classify.augment_892d_to_956d(np.zeros(100, dtype=np.float64))


def test_small_image_returns_feature_unavailable_without_loading_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_load(_weights_path: Path | None = None) -> Any:
        raise AssertionError("weights must not load when image is smaller than 256x256")

    monkeypatch.setattr(source_classify, "_load_model", _unexpected_load)
    tiny = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), mode="RGB")
    result = source_classify.classify_source(tiny)
    assert result.label == "unknown"
    assert result.reason == "feature_unavailable"
    assert result.version_subfamily is None
    assert result.scores == {}
    assert result.watermark_truth == "unknown"


def test_positive_part_subunit_pooling_and_threshold_mutation(tmp_path: Path) -> None:
    x_956 = np.zeros(source_classify.FEATURE_DIM, dtype=np.float64)

    # Case 1: global_bias = -1.5, sub_bias = -2.0 (negative sub-unit score must NOT clamp z to 0.0!)
    w_neg = _write_synthetic_v3_npz(
        tmp_path / "neg.npz",
        target_class="openai",
        target_global_bias=-1.5,
        target_sub_bias=-2.0,
        target_sub_alpha=1.0,
        router_threshold=0.5,
    )
    res_neg = source_classify.classify_source_features(x_956, weights_path=w_neg)
    assert res_neg.label == "unknown"
    assert res_neg.reason == "abstained"
    assert res_neg.pooled_scores["openai"] == pytest.approx(-1.5)

    # Case 2: global_bias = -1.5, sub_bias = +3.0, alpha = 0.8 -> pooled = +2.4 >= threshold 1.0
    w_pos = _write_synthetic_v3_npz(
        tmp_path / "pos.npz",
        target_class="openai",
        target_global_bias=-1.5,
        target_sub_bias=3.0,
        target_sub_alpha=0.8,
        router_threshold=1.0,
        stage2_win_idx=1,
    )
    res_pos = source_classify.classify_source_features(x_956, weights_path=w_pos)
    assert res_pos.label == "openai"
    assert res_pos.reason == "classified"
    assert res_pos.pooled_scores["openai"] == pytest.approx(2.4)
    assert res_pos.version_subfamily == "openai:v2"
    assert res_pos.fine_subfamily == "openai:v_sub_a"
    assert res_pos.watermark_truth == "unknown"

    # Case 3: raising router_threshold above 2.4 forces abstention
    w_hi = _write_synthetic_v3_npz(
        tmp_path / "hi.npz",
        target_class="openai",
        target_global_bias=-1.5,
        target_sub_bias=3.0,
        target_sub_alpha=0.8,
        router_threshold=3.0,
    )
    res_hi = source_classify.classify_source_features(x_956, weights_path=w_hi)
    assert res_hi.label == "unknown"
    assert res_hi.reason == "abstained"

    # Confidence must increase monotonically with AI evidence across abstained and classified regimes:
    # res_neg (excess = -2.0) < res_hi (excess = -0.6) < 0.5 <= res_pos (excess = +1.4)
    assert 0.0 < res_neg.confidence < res_hi.confidence < 0.5 <= res_pos.confidence < 1.0


def test_canonical_provider_routing_and_version_head(tmp_path: Path) -> None:
    x_956 = np.zeros(source_classify.FEATURE_DIM, dtype=np.float64)
    w_sd = _write_synthetic_v3_npz(
        tmp_path / "sd.npz",
        target_class="stable_diffusion",
        target_global_bias=3.5,
        router_threshold=1.0,
        stage2_win_idx=0,
    )
    res = source_classify.classify_source_features(x_956, weights_path=w_sd)
    assert res.label == "stable_diffusion"
    assert res.version_subfamily == "stable_diffusion:v1"
    assert set(res.pooled_scores) == set(source_classify.ALL_PROVIDER_CLASSES)


def test_stage15_hinge_meta_router_disambiguation(tmp_path: Path) -> None:
    x_956 = np.zeros(source_classify.FEATURE_DIM, dtype=np.float64)
    w_base = _write_synthetic_v3_npz(
        tmp_path / "base_meta.npz",
        target_class="openai",
        target_global_bias=2.0,
        target_sub_bias=2.5,
        target_sub_alpha=0.0,
        router_threshold=1.0,
    )
    # Without router_meta_w, openai wins with pooled_z = 2.0
    res_base = source_classify.classify_source_features(x_956, weights_path=w_base)
    assert res_base.label == "openai"

    # Add router_meta_tau and router_meta_w where openai:v_sub_a hinge (index 1, value 2.5 - 0.5 = 2.0)
    # suppresses openai (-1.0 * 2.0 = -2.0) and boosts firefly (+2.5 * 2.0 = +5.0 -> +2.0).
    with np.load(w_base, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    n_units = len(payload["unit_descriptor_classes"])
    n_cls = len(source_classify.ALL_PROVIDER_CLASSES)
    tau = np.zeros(n_units, dtype=np.float64)
    tau[1] = 0.5  # openai:v_sub_a has raw score 2.5 -> h_u[1] = 2.0
    w_meta = np.zeros((n_units + n_cls, n_cls), dtype=np.float64)
    openai_idx = source_classify.ALL_PROVIDER_CLASSES.index("openai")
    firefly_idx = source_classify.ALL_PROVIDER_CLASSES.index("firefly")
    w_meta[1, openai_idx] = -1.0  # delta = -2.0 -> pooled_z["openai"] = 0.0
    w_meta[1, firefly_idx] = 2.5  # delta = +5.0 -> pooled_z["firefly"] = -3.0 + 5.0 = +2.0
    payload["router_meta_tau"] = tau
    payload["router_meta_w"] = w_meta
    w_routed = tmp_path / "routed_meta.npz"
    np.savez(w_routed, **payload)

    res_routed = source_classify.classify_source_features(x_956, weights_path=w_routed)
    assert res_routed.label == "firefly"
    assert res_routed.pooled_scores["openai"] == pytest.approx(0.0)
    assert res_routed.pooled_scores["firefly"] == pytest.approx(2.0)


def test_metadata_does_not_change_pixel_decision(tmp_path: Path) -> None:
    weights = _write_synthetic_v3_npz(
        tmp_path / "meta.npz",
        target_class="openai",
        target_global_bias=3.2,
        router_threshold=1.0,
    )
    rng = np.random.default_rng(11)
    pixels = Image.fromarray(rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8), mode="RGB")
    clean = tmp_path / "clean.png"
    tagged = tmp_path / "tagged.png"
    pixels.save(clean)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Comment", "arbitrary metadata")
    pixels.save(tagged, pnginfo=metadata)

    first = source_classify.classify_source(clean, weights_path=weights)
    second = source_classify.classify_source(tagged, weights_path=weights)

    assert first.label == second.label == "openai"
    assert first.scores == second.scores


def test_load_model_rejects_bad_schema_and_hash_mismatch(tmp_path: Path) -> None:
    bad = _write_synthetic_v3_npz(tmp_path / "bad_schema.npz", schema="wrong-schema")
    with pytest.raises(ValueError, match="Unsupported source-classifier model schema"):
        source_classify._load_model(bad)

    with pytest.raises(RuntimeError, match="hash mismatch"):
        source_classify._verify_sha256(bad, "0" * 64)


def test_cli_classify_source_json_and_text_and_missing_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    img_path = tmp_path / "sample.png"
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(img_path)

    weights = _write_synthetic_v3_npz(
        tmp_path / "cli_v3.npz",
        target_class="google",
        target_global_bias=3.2,
        router_threshold=1.0,
        stage2_win_idx=1,
    )
    monkeypatch.setenv(source_classify.WEIGHTS_ENV, str(weights))

    runner = CliRunner()
    res_json = runner.invoke(main, ["classify-source", str(img_path), "--json"])
    assert res_json.exit_code == 0, res_json.output
    payload = json.loads(res_json.output)
    assert payload["label"] == "google"
    assert payload["version_subfamily"] == "google:v2"
    assert payload["watermark_truth"] == "unknown"

    res_text = runner.invoke(main, ["classify-source", str(img_path)])
    assert res_text.exit_code == 0, res_text.output
    assert "Label: google" in res_text.output
    assert "not SynthID detection" in res_text.output

    monkeypatch.setattr(source_classify, "is_available", lambda: False)
    res_missing = runner.invoke(main, ["classify-source", str(img_path)])
    assert res_missing.exit_code == 1
    assert "source-classify" in res_missing.output


def test_identify_does_not_import_source_classify(tmp_path: Path) -> None:
    from remove_ai_watermarks.identify import identify

    img_path = tmp_path / "plain.png"
    Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8), mode="RGB").save(img_path)
    sys.modules.pop("remove_ai_watermarks.source_classify", None)
    identify(img_path, check_visible=False, check_invisible=False)
    assert "remove_ai_watermarks.source_classify" not in sys.modules


def _base_two_tower_npz_payload(
    schema: str,
    *,
    extra_sub_provider: source_classify.ProviderClass,
    n_prov_out: int,
    hgb_baseline: float = 0.0,
    hgb_leaf_val: float = 0.0,
) -> tuple[dict[str, Any], int]:
    """Build shared pickle-free Two-Tower (`v4` / `v5`) trunk, HGB, and gate arrays."""
    d = source_classify.FEATURE_DIM_V4
    r_dim = source_classify.REGIME_DIM_V4
    sub_names = [f"{c}:v1" for c in source_classify.ALL_PROVIDER_CLASSES] + [f"{extra_sub_provider}:v2"]
    sub_provs = [*source_classify.ALL_PROVIDER_CLASSES, extra_sub_provider]
    n_subs = len(sub_names)
    h1, h2 = 8, 8

    payload: dict[str, Any] = {
        "schema": np.asarray(schema),
        "feature_dim": np.asarray(d, dtype=np.int64),
        "provider_classes": np.asarray(source_classify.ALL_PROVIDER_CLASSES, dtype="<U32"),
        "mean": np.zeros(d, dtype=np.float64),
        "scale": np.ones(d, dtype=np.float64),
        "sub_names": np.asarray(sub_names, dtype="<U48"),
        "sub_providers": np.asarray(sub_provs, dtype="<U32"),
        "trunk_count": np.asarray(1, dtype=np.int64),
    }

    p = "t1_s0"
    payload[f"{p}_w_in"] = np.zeros((h1, d), dtype=np.float16)
    payload[f"{p}_b_in"] = np.zeros(h1, dtype=np.float16)
    payload[f"{p}_ln0_g"] = np.ones(h1, dtype=np.float16)
    payload[f"{p}_ln0_b"] = np.zeros(h1, dtype=np.float16)
    payload[f"{p}_w_film"] = np.zeros((2 * h1, r_dim), dtype=np.float16)
    payload[f"{p}_b_film"] = np.zeros(2 * h1, dtype=np.float16)
    payload[f"{p}_w_r1a"] = np.zeros((h1, h1), dtype=np.float16)
    payload[f"{p}_b_r1a"] = np.zeros(h1, dtype=np.float16)
    payload[f"{p}_ln1_g"] = np.ones(h1, dtype=np.float16)
    payload[f"{p}_ln1_b"] = np.zeros(h1, dtype=np.float16)
    payload[f"{p}_w_r1b"] = np.zeros((h1, h1), dtype=np.float16)
    payload[f"{p}_b_r1b"] = np.zeros(h1, dtype=np.float16)
    payload[f"{p}_w_mid"] = np.zeros((h2, h1), dtype=np.float16)
    payload[f"{p}_b_mid"] = np.zeros(h2, dtype=np.float16)
    payload[f"{p}_ln2_g"] = np.ones(h2, dtype=np.float16)
    payload[f"{p}_ln2_b"] = np.zeros(h2, dtype=np.float16)
    payload[f"{p}_w_r2a"] = np.zeros((h2, h2), dtype=np.float16)
    payload[f"{p}_b_r2a"] = np.zeros(h2, dtype=np.float16)
    payload[f"{p}_ln3_g"] = np.ones(h2, dtype=np.float16)
    payload[f"{p}_ln3_b"] = np.zeros(h2, dtype=np.float16)
    payload[f"{p}_w_r2b"] = np.zeros((h2, h2), dtype=np.float16)
    payload[f"{p}_b_r2b"] = np.zeros(h2, dtype=np.float16)
    payload[f"{p}_w_prov"] = np.zeros((n_prov_out, h2), dtype=np.float16)
    payload[f"{p}_b_prov"] = np.zeros(n_prov_out, dtype=np.float16)
    payload[f"{p}_w_sub"] = np.zeros((n_subs, h2), dtype=np.float16)
    b_sub = np.zeros(n_subs, dtype=np.float16)
    b_sub[-1] = np.float16(1.5)
    payload[f"{p}_b_sub"] = b_sub

    payload["hgb_names"] = np.asarray(["spec_0"], dtype="<U32")
    payload["t2_h0_baseline"] = np.asarray(hgb_baseline, dtype=np.float32)
    payload["t2_h0_max_depth"] = np.asarray(1, dtype=np.int16)
    payload["t2_h0_feat_idx"] = np.zeros((1, 1), dtype=np.int16)
    payload["t2_h0_thresholds"] = np.asarray([[0.5]], dtype=np.float32)
    payload["t2_h0_left"] = np.zeros((1, 1), dtype=np.uint8)
    payload["t2_h0_right"] = np.zeros((1, 1), dtype=np.uint8)
    payload["t2_h0_values"] = np.asarray([[hgb_leaf_val]], dtype=np.float32)

    d_u = n_prov_out + n_subs + 1 + r_dim
    payload["router_tau_pos"] = np.zeros(d_u, dtype=np.float64)
    payload["router_tau_neg"] = np.zeros(d_u, dtype=np.float64)
    payload["gate_idx_a"] = np.zeros(0, dtype=np.int64)
    payload["gate_idx_b"] = np.zeros(0, dtype=np.int64)
    payload["gate_sign_a"] = np.zeros(0, dtype=np.float64)
    payload["gate_sign_b"] = np.zeros(0, dtype=np.float64)
    payload["gate_tau_a"] = np.zeros(0, dtype=np.float64)
    payload["gate_tau_b"] = np.zeros(0, dtype=np.float64)
    return payload, d_u


def _write_synthetic_v4_npz(
    path: Path,
    *,
    target_class: source_classify.ProviderClass = "google",
    target_bias: float = 3.0,
    router_threshold: float = 1.5,
) -> Path:
    """Create a minimal valid pickle-free multi-provider-source-v4 .npz fixture."""
    r_dim = source_classify.REGIME_DIM_V4
    n_cls = len(source_classify.ALL_PROVIDER_CLASSES)
    payload, d_u = _base_two_tower_npz_payload(
        source_classify.MODEL_SCHEMA_V4,
        extra_sub_provider=target_class,
        n_prov_out=n_cls + 1,
        hgb_baseline=1.25,
        hgb_leaf_val=0.75,
    )

    t_idx = source_classify.ALL_PROVIDER_CLASSES.index(target_class)
    b_prov = np.full(n_cls + 1, -2.0, dtype=np.float16)
    b_prov[t_idx] = np.float16(2.0)
    b_prov[-1] = np.float16(0.0)
    payload["t1_s0_b_prov"] = b_prov

    rb = np.full(n_cls, -1.0, dtype=np.float64)
    rb[t_idx] = target_bias
    payload["router_bias"] = rb
    payload["router_w"] = np.zeros((3 * d_u, n_cls), dtype=np.float64)

    # null_sig_bias = ln(e^0.75 - 1) so softplus + 0.25 == 1.0 exactly
    sig_b = float(np.log(np.exp(0.75) - 1.0))
    payload["null_mu_bias"] = np.zeros(n_cls, dtype=np.float64)
    payload["null_mu_w"] = np.zeros((r_dim, n_cls), dtype=np.float64)
    payload["null_sig_bias"] = np.full(n_cls, sig_b, dtype=np.float64)
    payload["null_sig_w"] = np.zeros((r_dim, n_cls), dtype=np.float64)
    payload["damping_gammas"] = np.full(n_cls, 0.5, dtype=np.float64)
    payload["router_thresholds"] = np.full(n_cls, router_threshold, dtype=np.float64)
    payload["conf_scale"] = np.asarray(1.0, dtype=np.float64)

    _add_stage2_hgb_payload(payload, target_class, [-0.5, 1.5])

    np.savez(path, **payload)
    return path


def test_v4_two_tower_2d_multiplicative_gate_and_regime_null_standardization(tmp_path: Path) -> None:
    w_base = _write_synthetic_v4_npz(
        tmp_path / "v4_base.npz",
        target_class="google",
        target_bias=2.4,
        router_threshold=1.5,
    )
    x_512 = np.zeros(source_classify.FEATURE_DIM_V4, dtype=np.float64)
    res0 = source_classify.classify_source_features(x_512, weights_path=w_base)
    assert res0.label == "google"
    assert res0.model_revision == source_classify.MODEL_SCHEMA_V4
    assert res0.version_subfamily == "google:v2"
    assert res0.fine_subfamily == "google:v2"
    assert res0.pooled_scores["google"] == pytest.approx(2.4, abs=1e-5)

    # Add a 2D multiplicative gate pairing hgb_0 (raw = 2.0 > 1.0) with regime r[0] (x_512[480] > 1.0)
    # that boosts flux: when r[0] == 0.0, gate == 0 -> google wins; when r[0] == 3.0, gate == 2.0 -> flux wins!
    with np.load(w_base, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    n_cls = len(source_classify.ALL_PROVIDER_CLASSES)
    n_subs = len(payload["sub_names"])
    hgb_u_idx = n_cls + 1 + n_subs  # index of t2_h0 in u_vec (value = 1.25 + 0.75 = 2.0)
    reg0_u_idx = hgb_u_idx + 1  # index of r_std[0] in u_vec
    d_u = reg0_u_idx + source_classify.REGIME_DIM_V4 - 1 + 1

    payload["gate_idx_a"] = np.asarray([hgb_u_idx], dtype=np.int16)
    payload["gate_idx_b"] = np.asarray([reg0_u_idx], dtype=np.int16)
    payload["gate_sign_a"] = np.asarray([1.0], dtype=np.float64)
    payload["gate_sign_b"] = np.asarray([1.0], dtype=np.float64)
    payload["gate_tau_a"] = np.asarray([1.0], dtype=np.float64)  # ga = 2.0 - 1.0 = 1.0
    payload["gate_tau_b"] = np.asarray([1.0], dtype=np.float64)  # gb = ReLU(r[0] - 1.0)
    w_router = np.zeros((3 * d_u + 1, n_cls), dtype=np.float64)
    flux_idx = source_classify.ALL_PROVIDER_CLASSES.index("flux")
    google_idx = source_classify.ALL_PROVIDER_CLASSES.index("google")
    w_router[-1, flux_idx] = 3.5  # when r[0]=3.0 -> gb=2.0 -> g_2d=2.0 -> flux z_raw = -1.0 + 7.0 = +6.0
    # Also test regime null dispersion: make google null_mu_w[1] = +2.0 so r[1]=2.0 shifts google null mean up by +4.0!
    null_mu_w = np.zeros((source_classify.REGIME_DIM_V4, n_cls), dtype=np.float64)
    null_mu_w[1, google_idx] = 2.0
    payload["router_w"] = w_router
    payload["null_mu_w"] = null_mu_w

    w_gated = tmp_path / "v4_gated.npz"
    np.savez(w_gated, **payload)

    # 1) Single-factor active (hgb_0 > 1.0, but r[0] == 0.0 <= 1.0): 2D gate must be 0 -> google still wins
    res_single = source_classify.classify_source_features(x_512, weights_path=w_gated)
    assert res_single.label == "google"
    assert res_single.pooled_scores["flux"] == pytest.approx(-1.0, abs=1e-5)

    # 2) Regime null shift active (r[1] = 2.0 shifts google mu_r to +4.0 -> google z_std = -1.6 < threshold): abstains!
    x_null_shifted = x_512.copy()
    x_null_shifted[source_classify.REGIME_START_V4 + 1] = 2.0
    res_abstain = source_classify.classify_source_features(x_null_shifted, weights_path=w_gated)
    assert res_abstain.label == "unknown"
    assert res_abstain.reason == "abstained"

    # 3) Both 2D gate factors active (r[0] = 3.0 -> g_2d = 2.0 -> flux z_std = +6.0): flux wins cleanly!
    x_both = x_512.copy()
    x_both[source_classify.REGIME_START_V4] = 3.0
    res_both = source_classify.classify_source_features(x_both, weights_path=w_gated)
    assert res_both.label == "flux"
    assert res_both.pooled_scores["flux"] == pytest.approx(6.0, abs=1e-5)


def test_v4_end_to_end_image_classification_ignores_metadata(tmp_path: Path) -> None:
    w_v4 = _write_synthetic_v4_npz(
        tmp_path / "v4_img.npz",
        target_class="bytedance",
        target_bias=3.2,
        router_threshold=1.0,
    )
    rng = np.random.default_rng(20261110)
    pixels = Image.fromarray(rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8), mode="RGB")
    clean = tmp_path / "clean_v4.png"
    tagged = tmp_path / "tagged_v4.png"
    pixels.save(clean)
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Software", "OpenAI GPT-Image")
    pixels.save(tagged, pnginfo=meta)

    r1 = source_classify.classify_source(clean, weights_path=w_v4)
    r2 = source_classify.classify_source(tagged, weights_path=w_v4)
    assert r1.label == r2.label == "bytedance"
    assert r1.model_revision == source_classify.MODEL_SCHEMA_V4
    assert r1.scores == r2.scores


def _write_synthetic_v5_npz(
    path: Path,
    *,
    det_winner: source_classify.ProviderClass = "flux",
    attr_winner: source_classify.ProviderClass = "ideogram",
    det_score: float = 4.0,
    det_threshold: float = 2.0,
    s2_hgb_weight: float = 1.0,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Create a minimal valid pickle-free multi-provider-source-v5 .npz fixture."""
    d = source_classify.FEATURE_DIM_V4
    r_dim = source_classify.REGIME_DIM_V4
    n_cls = len(source_classify.ALL_PROVIDER_CLASSES)
    n_det = n_cls + 1
    payload, d_u = _base_two_tower_npz_payload(
        source_classify.MODEL_SCHEMA_V5,
        extra_sub_provider=attr_winner,
        n_prov_out=n_cls + 2,
    )

    d_phi = 3 * d_u
    payload["u_med"] = np.zeros(d_u, dtype=np.float64)
    payload["u_scale"] = np.ones(d_u, dtype=np.float64)

    det_b = np.full(n_det, -1.0, dtype=np.float64)
    det_idx = source_classify.ALL_PROVIDER_CLASSES.index(det_winner)
    det_b[det_idx] = det_score
    payload["det_bias"] = det_b
    payload["det_w"] = np.zeros((d_phi, n_det), dtype=np.float64)

    h_null_dim = 4
    sig_b = float(np.log(np.exp(0.75) - 1.0))  # softplus + 0.25 == 1.0
    payload["null_h_w"] = np.zeros((r_dim, h_null_dim), dtype=np.float64)
    payload["null_h_b"] = np.zeros(h_null_dim, dtype=np.float64)
    payload["null_lin_mu_w"] = np.zeros((r_dim, n_det), dtype=np.float64)
    payload["null_mu_bias"] = np.zeros(n_det, dtype=np.float64)
    payload["null_mu_w"] = np.zeros((h_null_dim, n_det), dtype=np.float64)
    payload["null_lin_sig_w"] = np.zeros((r_dim, n_det), dtype=np.float64)
    payload["null_sig_bias"] = np.full(n_det, sig_b, dtype=np.float64)
    payload["null_sig_w"] = np.zeros((h_null_dim, n_det), dtype=np.float64)
    payload["det_thresholds"] = np.full(n_det, det_threshold, dtype=np.float64)

    h_attr_dim = 8
    attr_idx = source_classify.ALL_PROVIDER_CLASSES.index(attr_winner)
    attr_b2 = np.full(n_cls, -1.0, dtype=np.float64)
    attr_b2[attr_idx] = 3.5
    payload["attr_w1"] = np.zeros((d_phi, h_attr_dim), dtype=np.float64)
    payload["attr_b1"] = np.zeros(h_attr_dim, dtype=np.float64)
    payload["attr_ln_g"] = np.ones(h_attr_dim, dtype=np.float64)
    payload["attr_ln_b"] = np.zeros(h_attr_dim, dtype=np.float64)
    payload["attr_w2"] = np.zeros((h_attr_dim, n_cls), dtype=np.float64)
    payload["attr_b2"] = attr_b2
    payload["attr_skip_w"] = np.zeros((d_phi, n_cls), dtype=np.float64)
    payload["attr_temp"] = np.asarray(1.0, dtype=np.float64)
    payload["conf_scale"] = np.asarray(1.0, dtype=np.float64)

    stacked_dim = d + d_u
    # HGB prefers index 0 (-2.0 < 0 -> p1 < 0.5 -> v1), Linear prefers index 1 (+3.0 > 0 -> v2)
    _add_stage2_hgb_payload(payload, attr_winner, [-2.0])

    payload[f"s2_{attr_winner}_lin_mean"] = np.zeros(stacked_dim, dtype=np.float64)
    payload[f"s2_{attr_winner}_lin_scale"] = np.ones(stacked_dim, dtype=np.float64)
    payload[f"s2_{attr_winner}_lin_w"] = np.zeros((stacked_dim, 1), dtype=np.float64)
    payload[f"s2_{attr_winner}_lin_b"] = np.asarray([3.0], dtype=np.float64)
    payload[f"s2_{attr_winner}_lin_hgb_weight"] = np.asarray(s2_hgb_weight, dtype=np.float64)

    if overrides:
        for k, v in overrides.items():
            if v is None:
                payload.pop(k, None)
            else:
                payload[k] = v

    np.savez(path, **payload)
    return path


def test_v5_decoupled_acceptance_and_threshold_invariant_attribution(tmp_path: Path) -> None:
    # Even when flux has the highest detection z-score (+4.0) and its threshold is lowered to 0.1,
    # the decoupled attribution head assigns ideogram without threshold-induced misrouting!
    w_v5 = _write_synthetic_v5_npz(
        tmp_path / "v5_decoupled.npz",
        det_winner="flux",
        attr_winner="ideogram",
        det_score=4.0,
        det_threshold=2.0,
        s2_hgb_weight=0.0,
    )
    x_512 = np.zeros(source_classify.FEATURE_DIM_V4, dtype=np.float64)
    res = source_classify.classify_source_features(x_512, weights_path=w_v5)
    assert res.label == "ideogram"
    assert res.model_revision == source_classify.MODEL_SCHEMA_V5
    assert res.version_subfamily == "ideogram:v2"
    assert res.margins["ideogram"] - res.thresholds["ideogram"] == pytest.approx(2.0)

    # Mutate det_thresholds so ideogram threshold is HUGE (99.0) while flux clears (4.0 >= 1.5):
    # acceptance stays True and attribution MUST remain ideogram (threshold invariance)!
    thr = np.full(len(source_classify.ALL_PROVIDER_CLASSES) + 1, 99.0, dtype=np.float64)
    thr[source_classify.ALL_PROVIDER_CLASSES.index("flux")] = 1.5
    w_inv = _write_synthetic_v5_npz(
        tmp_path / "v5_invariant.npz",
        det_winner="flux",
        attr_winner="ideogram",
        det_score=4.0,
        overrides={"det_thresholds": thr},
    )
    res_inv = source_classify.classify_source_features(x_512, weights_path=w_inv)
    assert res_inv.label == "ideogram"
    assert res_inv.reason == "classified"

    # Raising all detection thresholds above 4.0 forces clean abstention with confidence < 0.5
    w_abs = _write_synthetic_v5_npz(
        tmp_path / "v5_abstain.npz",
        det_winner="flux",
        attr_winner="ideogram",
        det_score=4.0,
        det_threshold=5.0,
    )
    res_abs = source_classify.classify_source_features(x_512, weights_path=w_abs)
    assert res_abs.label == "unknown"
    assert res_abs.reason == "abstained"
    assert 0.0 < res_abs.confidence < 0.5 <= res_inv.confidence < 1.0


def test_v5_stage2_linear_blend_and_schema_validation(tmp_path: Path) -> None:
    x_512 = np.zeros(source_classify.FEATURE_DIM_V4, dtype=np.float64)
    w_hgb = _write_synthetic_v5_npz(tmp_path / "v5_s2_hgb.npz", s2_hgb_weight=1.0)
    w_lin = _write_synthetic_v5_npz(tmp_path / "v5_s2_lin.npz", s2_hgb_weight=0.0)
    assert source_classify.classify_source_features(x_512, weights_path=w_hgb).version_subfamily == "ideogram:v1"
    assert source_classify.classify_source_features(x_512, weights_path=w_lin).version_subfamily == "ideogram:v2"

    # End-to-end image path dispatches 512-d extractor for _SourceModelV5
    rng = np.random.default_rng(20261112)
    img = Image.fromarray(rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8), mode="RGB")
    res_img = source_classify.classify_source(img, weights_path=w_lin)
    assert res_img.label == "ideogram"
    assert res_img.model_revision == source_classify.MODEL_SCHEMA_V5

    # Invalid attr_temp <= 0 must raise ValueError
    w_bad = _write_synthetic_v5_npz(
        tmp_path / "v5_bad.npz",
        overrides={"attr_temp": np.asarray(0.0, dtype=np.float64)},
    )
    with pytest.raises(ValueError, match="Invalid v5 attribution head arrays"):
        source_classify._load_model(w_bad)

    # Partial Stage-2 linear blend keys must fail fast rather than silently falling back to HGB-only
    w_partial = _write_synthetic_v5_npz(
        tmp_path / "v5_partial_s2.npz",
        overrides={"s2_ideogram_lin_w": None},
    )
    with pytest.raises(ValueError, match="Incomplete Stage-2 linear blend keys for ideogram"):
        source_classify._load_model(w_partial)


def test_v5_joint_posterior_confidence_caps_ambiguous_attribution(tmp_path: Path) -> None:
    x_512 = np.zeros(source_classify.FEATURE_DIM_V4, dtype=np.float64)
    n_cls = len(source_classify.ALL_PROVIDER_CLASSES)
    ideo_idx = source_classify.ALL_PROVIDER_CLASSES.index("ideogram")
    google_idx = source_classify.ALL_PROVIDER_CLASSES.index("google")

    # 1) High acceptance excess (+4.0 -> sigmoid(4.0) ~= 0.982), near-tie in attribution (gap = 0.05):
    # joint confidence must be capped by sigmoid(0.05) ~= 0.5125 while keeping confidence >= 0.5.
    near_tie_b2 = np.full(n_cls, -1.0, dtype=np.float64)
    near_tie_b2[ideo_idx] = 2.05
    near_tie_b2[google_idx] = 2.00
    w_near_tie = _write_synthetic_v5_npz(
        tmp_path / "v5_near_tie.npz",
        det_winner="flux",
        attr_winner="ideogram",
        det_score=6.0,
        det_threshold=2.0,
        overrides={"attr_b2": near_tie_b2},
    )
    res_near_tie = source_classify.classify_source_features(x_512, weights_path=w_near_tie)
    assert res_near_tie.label == "ideogram"
    assert res_near_tie.reason == "classified"
    assert res_near_tie.confidence == pytest.approx(source_classify._sigmoid_scalar(0.05), abs=1e-6)
    assert 0.5 < res_near_tie.confidence < 0.52

    # 2) Exact attribution tie (gap = 0.0) while accepted (accept_excess = +4.0 > 0):
    # confidence hits the exact 0.5 boundary while preserving confidence >= 0.5 <=> label != "unknown".
    exact_tie_b2 = near_tie_b2.copy()
    exact_tie_b2[ideo_idx] = 2.00
    w_exact_tie = _write_synthetic_v5_npz(
        tmp_path / "v5_exact_tie.npz",
        det_winner="flux",
        attr_winner="ideogram",
        det_score=6.0,
        det_threshold=2.0,
        overrides={"attr_b2": exact_tie_b2},
    )
    res_exact_tie = source_classify.classify_source_features(x_512, weights_path=w_exact_tie)
    assert res_exact_tie.label != "unknown"
    assert res_exact_tie.reason == "classified"
    assert res_exact_tie.confidence == pytest.approx(0.5, abs=1e-12)

    # 3) Unambiguous attribution (gap = 4.5) with moderate acceptance excess (+0.5):
    # acceptance excess binds, so confidence equals sigmoid(conf_scale * accept_excess).
    w_wide = _write_synthetic_v5_npz(
        tmp_path / "v5_wide_gap.npz",
        det_winner="flux",
        attr_winner="ideogram",
        det_score=2.5,
        det_threshold=2.0,
        overrides={"conf_scale": np.asarray(1.2, dtype=np.float64)},
    )
    res_wide = source_classify.classify_source_features(x_512, weights_path=w_wide)
    assert res_wide.label == "ideogram"
    assert res_wide.confidence == pytest.approx(source_classify._sigmoid_scalar(0.6), abs=1e-6)

    # 4) Abstained input (accept_excess = -0.5 < 0) stays strictly below 0.5 regardless of large attr_gap.
    w_abstain = _write_synthetic_v5_npz(
        tmp_path / "v5_abstain_wide.npz",
        det_winner="flux",
        attr_winner="ideogram",
        det_score=1.5,
        det_threshold=2.0,
        overrides={"conf_scale": np.asarray(1.2, dtype=np.float64)},
    )
    res_abstain = source_classify.classify_source_features(x_512, weights_path=w_abstain)
    assert res_abstain.label == "unknown"
    assert res_abstain.reason == "abstained"
    assert res_abstain.confidence == pytest.approx(source_classify._sigmoid_scalar(-0.6), abs=1e-6)
    assert res_abstain.confidence < 0.5
