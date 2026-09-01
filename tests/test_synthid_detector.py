"""Runtime tests for the positive-only SynthID periodic carrier detector."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import synthid_runtime.synthid_detector as detector
from PIL import Image


@pytest.fixture(scope="module")
def supported_images(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Create supported-geometry positive and negative synthetic fixtures."""
    directory = tmp_path_factory.mktemp("synthid-detector")
    template, *_model = detector._load_template()
    scaled_tile = np.rint(template / np.max(np.abs(template)))
    marked = np.full((detector.MODEL_HEIGHT, detector.MODEL_WIDTH, 3), 128, dtype=np.float64)
    marked += np.tile(scaled_tile, (128, 128, 1))

    positive = directory / "positive.png"
    negative = directory / "negative.png"
    Image.fromarray(np.clip(np.rint(marked), 0, 255).astype(np.uint8), "RGB").save(positive)
    Image.new("RGB", (detector.MODEL_WIDTH, detector.MODEL_HEIGHT), (128, 128, 128)).save(negative)
    return positive, negative


@pytest.fixture(scope="module")
def registered_scale_positive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a strong period-12.8 carrier by shrinking a period-16 source."""
    import cv2

    directory = tmp_path_factory.mktemp("synthid-registered")
    template, *_model = detector._load_template()
    scaled_tile = template / np.max(np.abs(template)) * 40.0
    source = np.tile(scaled_tile, (64, 64, 1)) + 128.0
    pixels = cv2.resize(
        np.clip(np.rint(source), 0, 255).astype(np.uint8),
        (819, 819),
        interpolation=cv2.INTER_AREA,
    )
    path = directory / "period-12.8-positive.png"
    Image.fromarray(pixels, "RGB").save(path)
    return path


@pytest.fixture(scope="module")
def opponent_registered_positive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a strong period-10 opponent-color fallback fixture."""
    import cv2

    directory = tmp_path_factory.mktemp("synthid-opponent-registered")
    template, *_model = detector._load_template()
    scaled_tile = template / np.max(np.abs(template)) * 40.0
    source = np.tile(scaled_tile, (128, 128, 1)) + 128.0
    pixels = cv2.resize(
        np.clip(np.rint(source), 0, 255).astype(np.uint8),
        (1280, 1280),
        interpolation=cv2.INTER_AREA,
    )
    path = directory / "period-10-positive.png"
    Image.fromarray(pixels, "RGB").save(path)
    return path


@pytest.fixture(scope="module")
def opponent_period8_positive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a strong period-8 fallback fixture without native JPEG block edges."""
    import cv2

    directory = tmp_path_factory.mktemp("synthid-opponent-period8")
    template, *_model = detector._load_template()
    scaled_tile = template / np.max(np.abs(template)) * 40.0
    source = np.tile(scaled_tile, (128, 128, 1)) + 128.0
    pixels = cv2.resize(
        np.clip(np.rint(source), 0, 255).astype(np.uint8),
        (1024, 1024),
        interpolation=cv2.INTER_AREA,
    )
    path = directory / "period-8-positive.png"
    Image.fromarray(pixels, "RGB").save(path)
    return path


@pytest.fixture(scope="module")
def fine_opponent_registered_positive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a strong period-7.68 carrier missed by the coarse period grid."""
    import cv2

    directory = tmp_path_factory.mktemp("synthid-fine-opponent-registered")
    template, *_model = detector._load_template()
    scaled_tile = template / np.max(np.abs(template)) * 40.0
    source = np.tile(scaled_tile, (144, 144, 1)) + 128.0
    pixels = cv2.resize(
        np.clip(np.rint(source), 0, 255).astype(np.uint8),
        (1106, 1106),
        interpolation=cv2.INTER_AREA,
    )
    path = directory / "period-7.68-positive.png"
    Image.fromarray(pixels, "RGB").save(path)
    return path


def test_bundled_model_is_the_frozen_calibrated_artifact() -> None:
    model = Path(detector.__file__).parent / detector.MODEL_FILENAME

    assert hashlib.sha256(model.read_bytes()).hexdigest() == (
        "ee7838da8542c206c3403284b68e98f0ac99429e82f262c1a438f50a638b488b"
    )


@pytest.mark.parametrize(
    ("width", "height"),
    [(1000, 1000), (1001, 1000), (3000, 6000), (768, 1364)],
)
def test_supported_geometry_uses_the_challenged_pixel_count_range(width: int, height: int) -> None:
    assert detector._geometry_supported(width, height)


@pytest.mark.parametrize(
    ("width", "height"),
    [(999, 1000), (3001, 6000), (64, 32)],
)
def test_geometry_outside_the_challenged_pixel_count_range_is_unsupported(
    width: int,
    height: int,
) -> None:
    assert not detector._geometry_supported(width, height)


@pytest.mark.parametrize(
    ("width", "height", "supported"),
    [
        (500, 500, True),
        (4000, 2500, True),
        (256, 977, True),
        (499, 500, False),
        (4001, 2500, False),
        (255, 981, False),
        (64, 3907, False),
        (32, 7813, False),
    ],
)
def test_registered_geometry_uses_its_measured_pixel_count_range(
    width: int,
    height: int,
    supported: bool,
) -> None:
    assert detector._registered_geometry_supported(width, height) is supported


@pytest.mark.parametrize(
    ("width", "height", "supported"),
    [
        (1000, 1000, True),
        (4000, 2500, True),
        (767, 1304, False),
        (1000, 999, False),
        (4001, 2500, False),
    ],
)
def test_opponent_registered_geometry_uses_its_frozen_domain(
    width: int,
    height: int,
    supported: bool,
) -> None:
    assert detector._opponent_registered_geometry_supported(width, height) is supported


@pytest.mark.parametrize(
    ("width", "height", "supported"),
    [
        (1000, 1000, True),
        (2500, 2000, True),
        (767, 1304, False),
        (1000, 999, False),
        (2501, 2000, False),
    ],
)
def test_fine_opponent_registered_geometry_uses_its_frozen_domain(
    width: int,
    height: int,
    supported: bool,
) -> None:
    assert detector._fine_opponent_registered_geometry_supported(width, height) is supported


@pytest.mark.parametrize(
    ("width", "height", "supported"),
    [
        (4883, 2048, True),
        (3072, 5504, True),
        (2048, 4882, False),
        (2047, 6000, False),
        (3001, 6000, False),
    ],
)
def test_large_geometry_requires_multiple_calibrated_windows(
    width: int,
    height: int,
    supported: bool,
) -> None:
    assert detector._large_geometry_supported(width, height) is supported


def test_large_window_starts_cover_both_edges_on_carrier_phase() -> None:
    starts = detector._large_window_starts(5504)

    assert starts == (0, 2048, 3456)
    assert all(start % detector.LARGE_PHASE == 0 for start in starts)
    assert starts[-1] + detector.LARGE_WINDOW == 5504


def test_large_components_apply_the_portrait_alias_guard_only_to_its_geometry() -> None:
    values = {
        "minimum_fixed_score": 0.28,
        "minimum_red_green_spatial": 0.95,
        "minimum_blue_yellow_spatial": 0.85,
        "minimum_blue_yellow_mid_band": -0.30,
        "maximum_green_mid_band": 0.061,
    }
    portrait = detector.LargeImageComponents(width=3072, height=5504, **values)
    landscape = detector.LargeImageComponents(width=5504, height=3072, **values)

    assert portrait.decision_score < detector.LARGE_THRESHOLD
    assert landscape.decision_score > detector.LARGE_THRESHOLD


def test_large_red_green_gate_mutation_changes_the_real_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    width, height = 4883, 2048
    image = np.broadcast_to(np.zeros((1, 1, 3), dtype=np.uint8), (height, width, 3))
    components = detector.LargeImageComponents(
        width=width,
        height=height,
        minimum_fixed_score=0.28,
        minimum_red_green_spatial=detector.LARGE_RED_GREEN_SPATIAL_MIN,
        minimum_blue_yellow_spatial=0.85,
        minimum_blue_yellow_mid_band=-0.30,
        maximum_green_mid_band=0.0,
    )
    monkeypatch.setattr(detector, "is_available", lambda: True)
    monkeypatch.setattr(detector, "_load_template", lambda: (np.zeros((16, 16, 3)), 1.0, 0, 0, 0, 0))
    monkeypatch.setattr(detector, "large_image_components", lambda *_args: components)

    baseline = detector.detect_synthid("unused.png", image=image)
    monkeypatch.setattr(
        detector,
        "LARGE_RED_GREEN_SPATIAL_MIN",
        float(np.nextafter(components.minimum_red_green_spatial, np.inf)),
    )
    mutated = detector.detect_synthid("unused.png", image=image)

    assert baseline.status == "detected"
    assert baseline.detector == detector.LARGE_DETECTOR_ID
    assert mutated.status == "indeterminate"


def test_uncalibrated_narrow_large_geometry_is_unsupported() -> None:
    image = np.broadcast_to(np.zeros((1, 1, 3), dtype=np.uint8), (11_000, 1000, 3))

    result = detector.detect_synthid("unused.png", image=image)

    assert result.status == "unsupported"
    assert result.detector == detector.LARGE_DETECTOR_ID
    assert result.score is None


def test_registered_mode_rejects_a_side_too_short_for_quadrants(tmp_path: Path) -> None:
    path = tmp_path / "too-narrow.png"
    Image.new("RGB", (32, 7813), "white").save(path)

    result = detector.detect_synthid(path, register_scale=True)

    assert result.status == "unsupported"
    assert result.score is None
    assert result.detector == detector.REGISTERED_DETECTOR_ID


def test_detects_supported_periodic_carrier(supported_images: tuple[Path, Path]) -> None:
    positive, _negative = supported_images

    result = detector.detect_synthid(positive, register_scale=False)

    assert result.status == "detected"
    assert result.detected is True
    assert result.score is not None
    assert result.score > result.threshold
    assert result.to_dict()["detector"] == detector.DETECTOR_ID


def test_detects_unregistered_non_divisible_geometry_in_size_range(tmp_path: Path) -> None:
    width, height = 1001, 1000
    template, *_model = detector._load_template()
    scaled_tile = np.rint(template / np.max(np.abs(template)))
    repeats_y = (height + scaled_tile.shape[0] - 1) // scaled_tile.shape[0]
    repeats_x = (width + scaled_tile.shape[1] - 1) // scaled_tile.shape[1]
    carrier = np.tile(scaled_tile, (repeats_y, repeats_x, 1))[:height, :width]
    pixels = np.clip(np.rint(carrier + 128.0), 0, 255).astype(np.uint8)
    path = tmp_path / "non-divisible-positive.png"
    Image.fromarray(pixels, "RGB").save(path)

    result = detector.detect_synthid(path, register_scale=False)

    assert result.status == "detected"
    assert (result.width, result.height) == (width, height)
    assert result.score is not None
    assert result.score > result.threshold


def test_registered_mode_detects_a_rescaled_carrier(registered_scale_positive: Path) -> None:
    fixed = detector.detect_synthid(registered_scale_positive, register_scale=False)
    default = detector.detect_synthid(registered_scale_positive)
    registered = detector.detect_synthid(registered_scale_positive, register_scale=True)

    assert fixed.status == "unsupported"
    assert default == registered
    assert registered.status == "detected"
    assert registered.score is not None
    assert registered.score > registered.threshold
    assert registered.threshold == detector.REGISTERED_THRESHOLD
    assert registered.detector == detector.REGISTERED_DETECTOR_ID


def test_registered_mode_falls_back_to_the_opponent_color_expert(
    monkeypatch: pytest.MonkeyPatch,
    opponent_registered_positive: Path,
) -> None:
    import synthid_runtime._synthid_registered as registered_detector

    monkeypatch.setattr(registered_detector, "registered_score", lambda *_args: 0.0)

    result = detector.detect_synthid(opponent_registered_positive, register_scale=True)

    assert result.status == "detected"
    assert result.detector == detector.OPPONENT_REGISTERED_DETECTOR_ID
    assert result.score is not None
    assert result.score >= result.threshold


def test_opponent_registered_threshold_mutation_changes_the_real_verdict(
    monkeypatch: pytest.MonkeyPatch,
    opponent_registered_positive: Path,
) -> None:
    import synthid_runtime._synthid_registered as registered_detector

    monkeypatch.setattr(registered_detector, "registered_score", lambda *_args: 0.0)
    baseline = detector.detect_synthid(opponent_registered_positive, register_scale=True)
    assert baseline.score is not None
    assert baseline.detector == detector.OPPONENT_REGISTERED_DETECTOR_ID
    monkeypatch.setattr(
        detector,
        "OPPONENT_REGISTERED_THRESHOLD",
        float(np.nextafter(baseline.score, np.inf)),
    )

    mutated = detector.detect_synthid(opponent_registered_positive, register_scale=True)

    assert mutated.status == "indeterminate"
    assert mutated.detector == detector.REGISTERED_DETECTOR_ID


def test_opponent_fallback_recovers_period8_without_codec_grid(
    monkeypatch: pytest.MonkeyPatch,
    opponent_period8_positive: Path,
) -> None:
    import synthid_runtime._synthid_registered as registered_detector

    monkeypatch.setattr(registered_detector, "registered_score", lambda *_args: 0.0)

    result = detector.detect_synthid(opponent_period8_positive, register_scale=True)

    assert result.status == "detected"
    assert result.detector == detector.OPPONENT_REGISTERED_DETECTOR_ID


def test_fine_opponent_fallback_recovers_off_grid_period(
    monkeypatch: pytest.MonkeyPatch,
    fine_opponent_registered_positive: Path,
) -> None:
    import synthid_runtime._synthid_registered as registered_detector

    monkeypatch.setattr(registered_detector, "registered_score", lambda *_args: 0.0)
    monkeypatch.setattr(registered_detector, "opponent_registered_score", lambda *_args: 0.0)

    result = detector.detect_synthid(fine_opponent_registered_positive, register_scale=True)

    assert result.status == "detected"
    assert result.detector == detector.FINE_OPPONENT_REGISTERED_DETECTOR_ID
    assert result.score is not None
    assert result.score >= detector.FINE_OPPONENT_REGISTERED_THRESHOLD


def test_fine_opponent_threshold_mutation_changes_the_real_verdict(
    monkeypatch: pytest.MonkeyPatch,
    fine_opponent_registered_positive: Path,
) -> None:
    import synthid_runtime._synthid_registered as registered_detector

    monkeypatch.setattr(registered_detector, "registered_score", lambda *_args: 0.0)
    monkeypatch.setattr(registered_detector, "opponent_registered_score", lambda *_args: 0.0)
    baseline = detector.detect_synthid(fine_opponent_registered_positive, register_scale=True)
    assert baseline.score is not None
    assert baseline.detector == detector.FINE_OPPONENT_REGISTERED_DETECTOR_ID
    monkeypatch.setattr(
        detector,
        "FINE_OPPONENT_REGISTERED_THRESHOLD",
        float(np.nextafter(baseline.score, np.inf)),
    )

    mutated = detector.detect_synthid(fine_opponent_registered_positive, register_scale=True)

    assert mutated.status == "indeterminate"
    assert mutated.detector == detector.REGISTERED_DETECTOR_ID


def test_fine_opponent_selector_recovers_the_fractional_period(
    fine_opponent_registered_positive: Path,
) -> None:
    import synthid_runtime._synthid_registered as registered_detector

    template, sigma, *_model = detector._load_template()
    pixels = np.asarray(Image.open(fine_opponent_registered_positive).convert("RGB"), dtype=np.uint8)
    components = registered_detector.fine_opponent_registered_components(pixels, template, sigma)

    assert components.selected_period == pytest.approx(7.68, abs=0.01)
    assert components.fine_decision_score >= detector.FINE_OPPONENT_REGISTERED_THRESHOLD
    assert components.candidate_count >= 100


def test_period8_codec_veto_threshold_mutation_changes_real_components(
    monkeypatch: pytest.MonkeyPatch,
    opponent_period8_positive: Path,
) -> None:
    import synthid_runtime._synthid_registered as registered_detector

    template, sigma, *_model = detector._load_template()
    pixels = np.asarray(Image.open(opponent_period8_positive).convert("RGB"), dtype=np.uint8)
    components = registered_detector.opponent_registered_components(pixels, template, sigma)
    assert components.decision_score >= detector.OPPONENT_REGISTERED_THRESHOLD
    assert components.red_green_p8_edge_ratio is not None
    assert components.blue_yellow_p8_edge_ratio is not None
    monkeypatch.setattr(registered_detector, "OPPONENT_REGISTERED_MAX_P8_EDGE_RATIO", 0.9)

    assert components.decision_score == 0.0


def test_opponent_registered_period_band_and_codec_veto_are_required() -> None:
    from synthid_runtime._synthid_registered import OpponentRegisteredComponents

    values = {
        "spectral_score": 0.8,
        "fixed_score": 0.32,
        "red_green_spatial": 0.9,
        "blue_yellow_spatial": 0.8,
        "candidate_count": 3,
        "red_green_p8_edge_ratio": None,
        "blue_yellow_p8_edge_ratio": None,
    }
    matching = OpponentRegisteredComponents(10.0, 10.0, **values)
    period8 = OpponentRegisteredComponents(
        8.0,
        8.0,
        **{
            **values,
            "red_green_p8_edge_ratio": 1.0,
            "blue_yellow_p8_edge_ratio": 1.0,
        },
    )
    codec_alias = OpponentRegisteredComponents(
        8.0,
        8.0,
        **{
            **values,
            "red_green_p8_edge_ratio": 1.2,
            "blue_yellow_p8_edge_ratio": 1.2,
        },
    )

    assert matching.decision_score > detector.OPPONENT_REGISTERED_THRESHOLD
    assert period8.decision_score > detector.OPPONENT_REGISTERED_THRESHOLD
    assert codec_alias.base_decision_score > detector.OPPONENT_REGISTERED_THRESHOLD
    assert codec_alias.decision_score == 0.0


def test_registered_threshold_mutation_changes_the_real_verdict(
    monkeypatch: pytest.MonkeyPatch,
    registered_scale_positive: Path,
) -> None:
    baseline = detector.detect_synthid(registered_scale_positive, register_scale=True)
    assert baseline.score is not None
    mutated_threshold = float(np.nextafter(baseline.score, np.inf))
    monkeypatch.setattr(detector, "REGISTERED_THRESHOLD", mutated_threshold)

    mutated = detector.detect_synthid(registered_scale_positive, register_scale=True)

    assert mutated.status == "indeterminate"
    assert mutated.threshold == mutated_threshold


def test_registered_period_thresholds_cover_the_bounded_search() -> None:
    from synthid_runtime._synthid_registered import _period_threshold

    assert _period_threshold(7.5) == pytest.approx(0.3770629524888979)
    assert _period_threshold(12.0) == pytest.approx(0.19794247706938645)
    assert _period_threshold(24.5) == pytest.approx(0.3142958338390489)
    with pytest.raises(ValueError, match="outside"):
        _period_threshold(7.49)


def test_registered_amplitude_threshold_mutation_changes_the_real_verdict(
    monkeypatch: pytest.MonkeyPatch,
    registered_scale_positive: Path,
) -> None:
    import synthid_runtime._synthid_registered as registered_detector

    baseline = detector.detect_synthid(registered_scale_positive, register_scale=True)
    assert baseline.status == "detected"
    monkeypatch.setattr(
        registered_detector,
        "_PERIOD_THRESHOLDS",
        ((7.5, 24.5, float("inf")),),
    )

    mutated = detector.detect_synthid(registered_scale_positive, register_scale=True)

    assert mutated.status == "indeterminate"


def test_registered_spectral_candidate_disagreement_blocks_decision() -> None:
    from synthid_runtime._synthid_confirmation import RegisteredConfirmationComponents
    from synthid_runtime._synthid_registered import RegisteredComponents

    confirmation = RegisteredConfirmationComponents(12.8, 0.5, 0.2, 0.5, 8, 8)
    matching = RegisteredComponents(0.5, 0.25, 12.8, 12.8, 0.15, confirmation)
    mismatching = RegisteredComponents(0.5, 0.25, 12.8, 12.9, 0.15, confirmation)
    unconfirmed = RegisteredComponents(0.5, 0.25, 12.8, 12.8, 0.15)

    assert matching.decision_score == pytest.approx(2.0)
    assert mismatching.decision_score == pytest.approx(0.0)
    assert unconfirmed.base_decision_score == pytest.approx(2.0)
    assert unconfirmed.decision_score == pytest.approx(0.0)


def test_registered_high_band_mutation_changes_the_real_verdict(
    monkeypatch: pytest.MonkeyPatch,
    registered_scale_positive: Path,
) -> None:
    import synthid_runtime._synthid_registered as registered_detector

    components = registered_detector.registered_components(
        np.asarray(Image.open(registered_scale_positive).convert("RGB"), dtype=np.uint8),
        detector._load_template()[0],
        detector._load_template()[1],
    )
    assert components.decision_score >= detector.REGISTERED_THRESHOLD
    monkeypatch.setattr(
        registered_detector,
        "REGISTERED_HIGH_BAND_THRESHOLD",
        float(np.nextafter(components.high_band_score, np.inf)),
    )

    mutated = detector.detect_synthid(registered_scale_positive, register_scale=True)

    assert mutated.status == "indeterminate"


def test_registered_confirmation_mutation_changes_the_real_verdict(
    monkeypatch: pytest.MonkeyPatch,
    registered_scale_positive: Path,
) -> None:
    import synthid_runtime._synthid_confirmation as confirmation_detector
    import synthid_runtime._synthid_registered as registered_detector

    components = registered_detector.registered_components(
        np.asarray(Image.open(registered_scale_positive).convert("RGB"), dtype=np.uint8),
        detector._load_template()[0],
        detector._load_template()[1],
    )
    assert components.confirmation is not None
    assert components.decision_score >= detector.REGISTERED_THRESHOLD
    monkeypatch.setattr(
        confirmation_detector,
        "MIN_COHERENCE",
        float(np.nextafter(components.confirmation.joint_coherence, np.inf)),
    )

    mutated = detector.detect_synthid(registered_scale_positive, register_scale=True)

    assert mutated.status == "indeterminate"


def test_supported_negative_does_not_claim_clean(supported_images: tuple[Path, Path]) -> None:
    _positive, negative = supported_images

    result = detector.detect_synthid(negative, register_scale=False)

    assert result.status == "indeterminate"
    assert result.detected is False
    assert result.score == pytest.approx(0.0)


def test_threshold_mutation_changes_the_real_verdict(
    monkeypatch: pytest.MonkeyPatch,
    supported_images: tuple[Path, Path],
) -> None:
    positive, _negative = supported_images
    baseline = detector.detect_synthid(positive, register_scale=False)
    assert baseline.score is not None
    assert baseline.status == "detected"
    mutated_threshold = float(np.nextafter(baseline.score, np.inf))
    assert mutated_threshold > baseline.score

    monkeypatch.setattr(detector, "TILE_THRESHOLD", mutated_threshold)
    mutated = detector.detect_synthid(positive, register_scale=False)

    assert mutated.status == "indeterminate"
    assert mutated.threshold == mutated_threshold


def test_unsupported_geometry_is_distinct_from_negative(tmp_path: Path) -> None:
    path = tmp_path / "small.png"
    Image.new("RGB", (64, 32), "white").save(path)

    result = detector.detect_synthid(path, register_scale=False)

    assert result.status == "unsupported"
    assert result.score is None
    assert (result.width, result.height) == (64, 32)
    assert result.reason is not None
    assert result.to_dict()["metadata_used_for_verdict"] is False
    assert result.to_dict()["provider_scope"] == "provider-neutral"


def test_shared_bgr_decode_matches_file_decode(supported_images: tuple[Path, Path]) -> None:
    import cv2

    positive, _negative = supported_images
    bgr = cv2.imread(str(positive))
    assert bgr is not None

    from_file = detector.detect_synthid(positive, register_scale=False)
    from_array = detector.detect_synthid(positive, image=bgr, register_scale=False)

    assert from_array == from_file


def test_supported_geometry_requires_pixel_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    supported_images: tuple[Path, Path],
) -> None:
    _positive, negative = supported_images
    monkeypatch.setattr(detector, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="needs numpy and OpenCV"):
        detector.detect_synthid(negative, register_scale=False)


def test_fold_accepts_non_divisible_geometry_without_resampling() -> None:
    rng = np.random.default_rng(20260810)
    tile = rng.normal(0.0, 8.0, size=(16, 16, 3))
    repeated = np.tile(tile, (19, 20, 1)) + 128.0

    divisible = detector.fold_residual_template(
        repeated,
        tile_height=16,
        tile_width=16,
        denoise_sigma=1.0,
    )
    non_divisible = detector.fold_residual_template(
        repeated[:299, :317],
        tile_height=16,
        tile_width=16,
        denoise_sigma=1.0,
    )
    divisible_unit, _ = detector.unit_tile(divisible)
    non_divisible_unit, _ = detector.unit_tile(non_divisible)

    assert non_divisible.shape == (16, 16, 3)
    assert float(np.sum(divisible_unit * non_divisible_unit)) > 0.999


def test_non_divisible_fold_matches_modulo_cell_means() -> None:
    import cv2

    rng = np.random.default_rng(44041)
    pixels = rng.integers(0, 256, size=(53, 71, 3), dtype=np.uint8)
    source = pixels.astype(np.float32)
    # Mirror the module's documented per-channel blur: OpenCV's multi-channel
    # GaussianBlur is not bit-identical to per-channel calls (observed 3e-5 on
    # cv2 4.10.0), so a three-channel reference cannot satisfy atol=0.
    residual = np.stack(
        [
            source[:, :, channel]
            - cv2.GaussianBlur(
                source[:, :, channel].copy(),
                (0, 0),
                sigmaX=1.25,
                sigmaY=1.25,
                borderType=cv2.BORDER_REFLECT_101,
            )
            for channel in range(3)
        ],
        axis=2,
    )
    expected = np.empty((16, 16, 3), dtype=np.float64)
    for tile_y in range(16):
        for tile_x in range(16):
            expected[tile_y, tile_x] = residual[tile_y::16, tile_x::16].mean(
                axis=(0, 1),
                dtype=np.float64,
            )
    expected -= np.mean(expected, axis=(0, 1), keepdims=True)

    actual = detector.fold_residual_template(
        pixels,
        tile_height=16,
        tile_width=16,
        denoise_sigma=1.25,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_fold_rejects_tile_larger_than_image() -> None:
    pixels = np.zeros((15, 16, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="at least as large"):
        detector.fold_residual_template(
            pixels,
            tile_height=16,
            tile_width=16,
            denoise_sigma=1.0,
        )


def test_verdict_does_not_claim_the_watermark() -> None:
    """The result must not assert SynthID, because the statistic is not SynthID.

    This was unguarded until 2026-08-16, and the claim had been wrong for months
    without a single test noticing. The fields are pinned by value rather than by
    presence so that a rename back to a watermark claim fails here.
    """
    result = detector.SynthIDDetection(
        status="detected",
        width=4096,
        height=2560,
        score=1.0,
        threshold=1.0,
    )

    payload = result.to_dict()

    assert payload["signal_family"] == "generation-pipeline-lattice"
    assert payload["identifies_watermark"] is False
    assert payload["tile_aligned_crop_required"] is True
    assert "synthid" not in str(payload["signal_family"]).lower()


def test_the_statistic_is_locked_to_the_image_origin() -> None:
    """A crop off the tile grid must destroy the score, and that must stay visible.

    SynthID's published evaluation retains 99.97% TPR under aggressive crop and
    resize. This statistic loses everything to a seven-pixel shift, measured on
    the real runtime at 4096x2560 where aligned crops scored up to 1.069 and
    shifted ones reached -0.438. The property is asserted here so that any future
    expert claiming to read the watermark has to survive the same shift first.
    """
    template, sigma, *_model = detector._load_template()
    tile = template / np.max(np.abs(template))
    pixels = np.full((1024, 1024, 3), 128.0)
    pixels += 6.0 * np.tile(tile, (64, 64, 1))
    aligned = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)

    aligned_score, _folded = detector.folded_template_score(aligned, template, sigma)
    # Seven is deliberately coprime with the 16-pixel tile, so no residual phase survives.
    shifted_score, _shifted_folded = detector.folded_template_score(
        aligned[7:, 7:],
        template,
        sigma,
    )

    assert aligned_score > 0.5
    assert shifted_score < 0.1 * aligned_score
