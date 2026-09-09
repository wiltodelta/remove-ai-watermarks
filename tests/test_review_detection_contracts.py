"""Regressions for provider confirmation and inter-mark comparisons."""

import numpy as np
import pytest

from remove_ai_watermarks.watermark_registry import known_marks


@pytest.mark.parametrize(
    ("producer", "confirmed"),
    [(code, mark.key == "doubao") for mark in known_marks() for code in mark.tc260_producer_codes]
    + [("001191110102MACQD9K64010000", True), ("DouBao", True), ("", False)],
)
def test_doubao_confirmation_reaches_the_video_plan(producer, confirmed):
    from remove_ai_watermarks.video import _visible_removal_plan
    from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan

    box = (800, 470, 120, 30)
    scan = VideoScan(960, 540, 30, tuple(FrameLocalization(i, 0.40, box) for i in range(12)))
    regions, _, _ = _visible_removal_plan("doubao", scan, {"aigc_producer": producer})
    assert regions == ([box] if confirmed else [None]) * 12


@pytest.mark.parametrize(
    "markers",
    [{"aigc_label": "China AIGC label (TC260); producer doubao"}, {"issuer": "ByteDance"}],
)
def test_doubao_confirmation_requires_structural_producer(markers):
    from remove_ai_watermarks.video_visible import has_doubao_video_provenance

    assert not has_doubao_video_provenance(markers)


def test_baidu_margin_scores_the_declared_rival_templates(monkeypatch):
    from remove_ai_watermarks import _text_mark_engine, baidu_engine

    scored = []

    def score(mask, scale, config):
        scored.append(config.asset_name)
        return 0.1

    monkeypatch.setattr(_text_mark_engine, "template_match_score", score)
    engine = _text_mark_engine.TextMarkEngine(baidu_engine._CONFIG)
    assert engine._rival_margin_ok(score=0.9, box_mask=np.zeros((20, 80), dtype=np.uint8), scale_base=960)
    assert scored == list(baidu_engine._CONFIG.rivals)


def test_every_registered_text_engine_resolves_its_declared_rivals():
    from remove_ai_watermarks import _text_mark_engine, watermark_registry

    for key in watermark_registry._ENGINE_CLASS:
        engine = watermark_registry._engine(key)
        if isinstance(engine, _text_mark_engine.TextMarkEngine):
            for asset in engine.config.rivals:
                rival = _text_mark_engine._rival_config(asset)
                assert rival is not None, (key, asset)
                assert rival.asset_name == asset, (key, asset)
