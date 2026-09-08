"""Draft operator-verifiable text lines for verified-text manifests.

Proposal-only OCR: PaddleOCR detection plus three script-chosen recognition
engines (Latin / Cyrillic / CJK), accepted only when three crop paddings
normalize identically and every confidence clears the floor. ``accepted``
means crop-stable, NEVER ground-truth-correct: on the reference posters the
draft's exact-text precision was 90.0% and 94.4% because high-confidence OCR
still lost punctuation (one English comma dropped, one ideographic comma
replaced with ASCII). Every accepted line needs a human yes/no before it may
enter a manifest with ``verified: true``.

Heavy imports (paddle, and the numpy/cv2/PIL pixel stack) stay inside the
call so importing this module costs nothing without the ``text-draft`` extra;
``draft_available()`` reports whether the extra is installed.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from remove_ai_watermarks._internal.text_restoration import source_pixel_sha256

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "DraftLine",
    "RejectedLine",
    "TextDraft",
    "choose_language",
    "draft_available",
    "draft_text_lines",
    "group_word_boxes",
    "source_pixel_sha256",
    "stable_recognition",
]

# Crop paddings probed per line: recognition must be invariant across them.
JITTER_RATIOS: tuple[float, ...] = (0.08, 0.12, 0.2)
DETECT_SCORE_FLOOR = 0.5


@dataclass(frozen=True)
class DraftLine:
    """One crop-stable proposal. ``accepted`` means stable, not correct."""

    box: tuple[int, int, int, int]
    text: str
    script: str
    language: str
    min_score: float


@dataclass(frozen=True)
class RejectedLine:
    """A detected line whose recognition was unstable or low-confidence."""

    box: tuple[int, int, int, int]
    script: str
    language: str
    reads: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class TextDraft:
    """The full proposal set for one image."""

    accepted: tuple[DraftLine, ...] = ()
    rejected: tuple[RejectedLine, ...] = ()


def normalize_text(text: str) -> str:
    """Normalize text for layout-independent comparison (casefold + no spaces)."""
    return "".join(unicodedata.normalize("NFC", text).casefold().split())


def _has_script(text: str, script: str) -> bool:
    return any(script in unicodedata.name(character, "") for character in text)


def choose_language(probes: dict[str, tuple[str, float]]) -> str:
    """Pick ``ch``/``ru``/``en`` from what each probe engine actually read."""
    if _has_script(probes["ch"][0], "CJK"):
        return "ch"
    if _has_script(probes["ru"][0], "CYRILLIC"):
        return "ru"
    return "en"


def stable_recognition(reads: list[tuple[str, float]], min_score: float = 0.85) -> str | None:
    """The proposal text when every read normalizes identically and clears the floor."""
    normalized = {normalize_text(text) for text, _score in reads}
    if len(normalized) != 1 or min(score for _text, score in reads) < min_score:
        return None
    return reads[0][0]


def _vertical_overlap_ratio(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    overlap = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    return overlap / max(1, min(left[3] - left[1], right[3] - right[1]))


def _row_center(box: tuple[int, int, int, int]) -> float:
    return (box[1] + box[3]) / 2


def group_word_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Merge word detections into line boxes by vertical overlap and gap."""
    groups: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: (_row_center(item), item[0])):
        matches: list[int] = []
        for index, group in enumerate(groups):
            if _vertical_overlap_ratio(box, group) < 0.45:
                continue
            horizontal_gap = max(0, max(box[0], group[0]) - min(box[2], group[2]))
            line_height = min(box[3] - box[1], group[3] - group[1])
            if horizontal_gap <= max(24, line_height * 3):
                matches.append(index)
        if not matches:
            groups.append(box)
            continue
        index = max(matches, key=lambda item: _vertical_overlap_ratio(box, groups[item]))
        x1, y1, x2, y2 = groups[index]
        groups[index] = min(x1, box[0]), min(y1, box[1]), max(x2, box[2]), max(y2, box[3])
    return sorted(groups, key=_row_center)


def _recognition_box(
    box: tuple[int, int, int, int],
    script: str,
    width: int,
    height: int,
    vertical_pad_ratio: float | None = None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    line_height = y2 - y1
    if script == "cjk":
        left_pad = max(16, round(line_height * 0.2))
        right_pad = max(16, round(line_height * 0.6))
        return max(0, x1 - left_pad), y1, min(width, x2 + right_pad), y2
    pad_x = max(16, line_height)
    pad_y = max(8, line_height // 3) if vertical_pad_ratio is None else max(8, round(line_height * vertical_pad_ratio))
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)


def _recognize(
    engine: Any,
    image: Any,
    box: tuple[int, int, int, int],
    script: str,
    vertical_pad_ratio: float | None = None,
) -> tuple[str, float]:
    import cv2

    height, width = image.shape[:2]
    x1, y1, x2, y2 = _recognition_box(box, script, width, height, vertical_pad_ratio)
    crop = image[y1:y2, x1:x2]
    if crop.shape[0] < 64:
        scale = 64 / crop.shape[0]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    result = next(iter(engine.predict(crop)))
    return str(result.get("rec_text", "")), float(result.get("rec_score", 0.0))


def _detect_line_boxes(engine: Any, source_rgb: Any) -> list[tuple[int, int, int, int]]:
    import numpy as np

    boxes: list[tuple[int, int, int, int]] = []
    for page in engine.predict(source_rgb):
        detected = page.get("rec_boxes", None)
        if detected is None or len(detected) == 0:
            detected = page.get("rec_polys", [])
        for score, raw_box in zip(page.get("rec_scores", []), detected, strict=False):
            if float(score) < DETECT_SCORE_FLOOR:
                continue
            points = np.asarray(raw_box, dtype=np.float32).reshape(-1)
            if points.size == 4:
                x1, y1, x2, y2 = points
            else:
                points = points.reshape(-1, 2)
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
            boxes.append((round(float(x1)), round(float(y1)), round(float(x2)), round(float(y2))))
    return group_word_boxes(boxes)


def draft_available() -> bool:
    """True when the ``text-draft`` extra (paddleocr + paddle) can run."""
    from remove_ai_watermarks.optional_deps import module_available

    return module_available("paddleocr") and module_available("paddle")


def _build_engines() -> tuple[Any, dict[str, Any]]:
    from paddleocr import PaddleOCR, TextRecognition

    # enable_mkldnn=False is load-bearing on linux: paddle's oneDNN PIR executor
    # raises NotImplementedError on the text-detection graph there (seen on a
    # Linux GPU container, 2026-08-19), while the reference CPU path runs. The
    # eval scripts never hit it because macOS pads take a different path.
    detector = PaddleOCR(
        lang="ch",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    engines = {
        "en": TextRecognition(model_name="en_PP-OCRv5_mobile_rec", enable_mkldnn=False),
        "ru": TextRecognition(model_name="eslav_PP-OCRv5_mobile_rec", enable_mkldnn=False),
        "ch": TextRecognition(model_name="PP-OCRv5_server_rec", enable_mkldnn=False),
    }
    return detector, engines


def draft_text_lines(
    image: Path,
    *,
    min_score: float = 0.85,
    detector: Any | None = None,
    engines: dict[str, Any] | None = None,
    stable: bool = True,
) -> TextDraft:
    """Propose verified-text manifest lines for ``image``; never verified ones.

    Args:
        image: path of the source image (draft boxes are in ITS pixel space).
        min_score: recognition confidence floor. With ``stable=True`` every
            jittered read must clear it; with ``stable=False`` the single
            probe for the chosen language must.
        detector/engines: injectable Paddle objects (tests use fakes); when
            None they are built from the ``text-draft`` extra, which must be
            installed (``draft_available()`` reports it).
        stable: when True (default), accept a line only if three crop paddings
            normalize identically. When False, take one recognition trio and
            accept on score alone. Automatic restoration uses box/script only,
            so the jitter gate is optional there.

    Returns:
        ``TextDraft`` with crop-stable ``accepted`` proposals and unstable
        ``rejected`` lines. Both lists need human review before any manifest
        may claim ``verified: true``; accepted means crop-stable, NOT
        ground-truth-correct.
    """
    import os

    import numpy as np
    from PIL import Image

    if detector is None or engines is None:
        if not draft_available():
            raise RuntimeError(
                "Text drafting requires PaddleOCR. Install: pip install 'remove-ai-watermarks[text-draft]'"
            )
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        detector, engines = _build_engines()

    source_rgb = np.asarray(Image.open(image).convert("RGB"))
    accepted: list[DraftLine] = []
    rejected: list[RejectedLine] = []
    for box in _detect_line_boxes(detector, source_rgb):
        probes: dict[str, tuple[str, float]] = {}
        for language, engine in engines.items():
            script = "cjk" if language == "ch" else "alphabetic"
            probes[language] = _recognize(engine, source_rgb, box, script, 0.1)
        language = choose_language(probes)
        script = "cjk" if language == "ch" else "alphabetic"
        if stable:
            reads = [_recognize(engines[language], source_rgb, box, script, ratio) for ratio in JITTER_RATIOS]
            text = stable_recognition(reads, min_score)
        else:
            text, score = probes[language]
            reads = [(text, score)]
            if not text.strip() or score < min_score:
                text = None
        if text is None:
            rejected.append(
                RejectedLine(
                    box=box,
                    script=script,
                    language=language,
                    reads=tuple((value, score) for value, score in reads),
                )
            )
        else:
            accepted.append(
                DraftLine(
                    box=box,
                    text=text,
                    script=script,
                    language=language,
                    min_score=min(score for _value, score in reads),
                )
            )
    return TextDraft(accepted=tuple(accepted), rejected=tuple(rejected))
