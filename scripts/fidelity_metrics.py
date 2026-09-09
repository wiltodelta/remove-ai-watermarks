# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "numpy",
#   "opencv-python-headless<5",
#   "pillow",
#   "scikit-image",
#   "torch",
#   "lpips",
#   "paddleocr",
#   "paddlepaddle",
#   "insightface",
#   "onnxruntime",
# ]
# ///
"""Objective fidelity metrics for comparing watermark-removal outputs.

Given an ORIGINAL (the reference) and one or more cleaned VARIANTS that have all
ALREADY passed the scrub oracle, this scores how much real detail each variant
preserved -- so "closer to the original" is the right axis here (between two
equally-scrubbed outputs, the one that deviates less from the original wins).

It is a standalone eval tool, NOT part of the package: PEP 723 inline deps let
``uv run`` build a throwaway env so the heavy models (PaddleOCR, insightface,
LPIPS) never touch uv.lock or the shipped library. Metrics self-gate: face
metrics run only where faces are detected, text metrics only where text is.

Two subcommands:

  ocr      -- OCR images (PaddleOCR defaults) into a JSON {basename: text} file.
              Run this on the ORIGINALS, hand-verify/correct the file, and it
              becomes the ground truth for ``compare --ground-truth`` -- the clean
              way to score text, since OCR-vs-OCR is doubly noisy (errors on both
              images + reading-order differences inflate NED even on identical text).

  compare  -- Score each VARIANT against the ORIGINAL across four groups:
              1. Text  -- normalized edit distance (NED) of the variant's OCR vs the
                 verified ground truth (or the original's OCR if no --ground-truth).
              2. Face identity -- insightface (buffalo_l) ArcFace cosine similarity.
              3. Face texture  -- LPIPS + Laplacian-variance ratio on face crops
                 (catches "plastication": ratio < 1 = smoother than the original).
              4. Whole image   -- LPIPS / SSIM / PSNR vs the original.

Usage:
    uv run scripts/fidelity_metrics.py ocr O1.png O2.png --langs en,ru,ch --out gt.json
    # (edit gt.json by hand to fix any OCR slips, then:)
    uv run scripts/fidelity_metrics.py compare --original O1.png \
        --variant controlnet=C.png --variant qwen=Q.png \
        --ocr-langs en,ru,ch --ground-truth gt.json
"""

from __future__ import annotations

import json
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import cv2
import numpy as np
from _plain_console import Console, Table

console = Console()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._text_eval import levenshtein_normalized  # noqa: E402

# ── helpers ──────────────────────────────────────────────────────────


def _load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise click.ClickException(f"cannot read image: {path}")
    return img


def _match_size(variant: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Resize a variant to the reference size (outputs differ by a grid-round)."""
    if variant.shape[:2] != ref.shape[:2]:
        variant = cv2.resize(variant, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    return variant


def _norm(text: str) -> str:
    """Normalize for NED: NFC + drop ALL whitespace (segmentation-order agnostic)."""
    return "".join(unicodedata.normalize("NFC", text).split())


# ── text: PaddleOCR defaults ─────────────────────────────────────────

# Our lang codes -> PaddleOCR lang. The 'ch' model also reads Latin; 'ru' reads
# Cyrillic + Latin. Multiple langs in one image -> run each model, union detections.
_PADDLE_LANG = {"en": "en", "ru": "ru", "ch": "ch", "ch_sim": "ch", "latin": "latin"}
_paddle_cache: dict[str, Any] = {}


def _paddle(lang: str) -> Any:
    if lang not in _paddle_cache:
        from paddleocr import PaddleOCR

        _paddle_cache[lang] = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _paddle_cache[lang]


def _box_xyxy(box: Any) -> tuple[float, float, float, float]:
    """Axis-aligned (x1, y1, x2, y2) of a PaddleOCR rec box ([x1,y1,x2,y2]) or poly (4x2)."""
    arr = np.asarray(box, dtype=np.float32).reshape(-1)
    if arr.size == 4:
        return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
    pts = arr.reshape(-1, 2)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _ocr_lines(bgr: np.ndarray, langs: list[str], min_score: float = 0.5) -> list[str]:
    """Detected text lines in reading order, unioned across lang models with spatial NMS.

    Several language models over one image re-detect the same lines -- and crucially the
    WRONG-script models read e.g. Cyrillic as confident Latin gibberish. So instead of a
    naive union, keep the HIGHEST-score detection per physical location (greedy IoU NMS):
    the model that actually fits a line wins it (the 'ru' model takes the Cyrillic, 'ch'
    the CJK, 'en' the Latin), and the cross-script garbage is dropped.
    """
    raw: list[tuple[float, tuple[float, float, float, float], str]] = []
    for lang in langs:
        plang = _PADDLE_LANG.get(lang, lang)
        for page in _paddle(plang).predict(bgr):
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            boxes = page.get("rec_boxes", None)
            if boxes is None or len(boxes) == 0:
                boxes = page.get("rec_polys", [])
            for text, score, box in zip(texts, scores, boxes, strict=False):
                if score < min_score or not text.strip():
                    continue
                raw.append((float(score), _box_xyxy(box), text.strip()))

    raw.sort(key=lambda d: d[0], reverse=True)
    kept: list[tuple[tuple[float, float, float, float], str]] = []
    for _score, box, text in raw:
        if any(_iou(box, kbox) > 0.3 for kbox, _ in kept):
            continue
        kept.append((box, text))
    kept.sort(key=lambda d: (round(d[0][1] / 20.0), d[0][0]))  # reading order: y then x
    return [t for _, t in kept]


def _text_ned(ref: str, hyp: str) -> float:
    """Case-sensitive edit distance divided by the longer normalized string."""
    return levenshtein_normalized(_norm(ref), _norm(hyp))


# ── face: detection + ArcFace + texture ──────────────────────────────


@dataclass
class FaceStats:
    n_faces: int = 0
    identity: list[float] = field(default_factory=list)
    lpips: list[float] = field(default_factory=list)
    lapvar_ratio: list[float] = field(default_factory=list)


def _lap_var(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _bbox_center(bbox: Any) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _bbox_diag(bbox: Any) -> float:
    return float(((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) ** 0.5)


def assign_faces_one_to_one(
    ref_centers: list[tuple[float, float]],
    var_centers: list[tuple[float, float]],
    ref_diags: list[float],
    max_frac: float = 0.6,
) -> dict[int, int]:
    """One-to-one nearest-center face assignment (pure; unit-tested without insightface).

    Per-face nearest matching collides on multi-face images -- two original faces can both
    pick the SAME variant face (e.g. when regeneration drops a face, so the variant has fewer
    detections), corrupting the identity metric (the lapvar/LPIPS metrics are immune: they are
    anchored to the ORIGINAL bbox on both images). This greedy-by-distance assignment is
    collision-free: it walks candidate pairs nearest-first and never reuses a ref or a variant
    face. Faces are spatially well-separated, so greedy equals the optimal (Hungarian) result
    here without the scipy dependency. A pair is dropped when the center distance exceeds
    ``max_frac`` of the original face diagonal (no plausible match -- the face was lost).

    Returns a dict mapping ref-face index -> variant-face index for matched faces only.
    """
    pairs: list[tuple[float, int, int]] = []
    for i, (rx, ry) in enumerate(ref_centers):
        for j, (vx, vy) in enumerate(var_centers):
            pairs.append((((rx - vx) ** 2 + (ry - vy) ** 2) ** 0.5, i, j))
    pairs.sort()
    used_ref: set[int] = set()
    used_var: set[int] = set()
    matched: dict[int, int] = {}
    for dist, i, j in pairs:
        if i in used_ref or j in used_var:
            continue
        if dist > max_frac * ref_diags[i]:
            continue
        matched[i] = j
        used_ref.add(i)
        used_var.add(j)
    return matched


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _crop(bgr: np.ndarray, bbox: Any) -> np.ndarray:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = (int(max(0, bbox[0])), int(max(0, bbox[1])), int(min(w, bbox[2])), int(min(h, bbox[3])))
    return bgr[y1:y2, x1:x2]


# ── whole image: LPIPS / SSIM / PSNR ─────────────────────────────────


def _lpips_model() -> tuple[Any, Any]:
    import lpips
    import torch

    model = lpips.LPIPS(net="alex", verbose=False)
    model.eval()
    return model, torch


def _lpips_distance(model_torch: tuple[Any, Any], a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    model, torch = model_torch

    def _t(img: np.ndarray) -> Any:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        return float(model(_t(a_bgr), _t(b_bgr)).item())


def _ssim_psnr(a_bgr: np.ndarray, b_bgr: np.ndarray) -> tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    return float(structural_similarity(a, b)), float(peak_signal_noise_ratio(a, b))


# ── reporting ────────────────────────────────────────────────────────


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _fmt(v: float | None, nd: int = 3) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


# ── CLI ──────────────────────────────────────────────────────────────


@click.group()
def cli() -> None:
    """Objective fidelity metrics for watermark-removal outputs."""


@cli.command("ocr")
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--langs", default="en", help="Comma list of OCR langs (en,ru,ch).")
@click.option("--out", type=click.Path(), default=None, help="Write {basename: text} JSON here (for ground truth).")
def ocr_cmd(images: tuple[str, ...], langs: str, out: str | None) -> None:
    """OCR images into a ground-truth seed -- hand-verify the result before using it."""
    lang_list = [x.strip() for x in langs.split(",") if x.strip()]
    result: dict[str, str] = {}
    for path in images:
        lines = _ocr_lines(_load_bgr(path), lang_list)
        text = "\n".join(lines)
        result[Path(path).name] = text
        console.print(f"\n=== {Path(path).name} ===")
        console.print(text or "(no text detected)")
    if out:
        Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"\n  Wrote {out} -- verify/correct it by hand, then pass it to `compare --ground-truth`.")


@cli.command("compare")
@click.option("--original", required=True, type=click.Path(exists=True), help="Reference (unprocessed) image.")
@click.option(
    "--variant", "variants", multiple=True, required=True, help="LABEL=PATH of a cleaned output (repeatable)."
)
@click.option("--ocr-langs", default="en", help="Comma list of OCR langs (en,ru,ch). Empty = skip text.")
@click.option("--ground-truth", type=click.Path(exists=True), default=None, help="Verified {basename: text} JSON.")
@click.option("--no-faces", is_flag=True, help="Skip face metrics.")
def compare(original: str, variants: tuple[str, ...], ocr_langs: str, ground_truth: str | None, no_faces: bool) -> None:
    """Score each VARIANT against ORIGINAL across the four fidelity groups."""
    ref = _load_bgr(original)
    parsed: list[tuple[str, np.ndarray]] = []
    for spec in variants:
        if "=" not in spec:
            raise click.ClickException(f"--variant must be LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        parsed.append((label, _match_size(_load_bgr(path), ref)))

    langs = [x.strip() for x in ocr_langs.split(",") if x.strip()]
    lp = _lpips_model()  # AlexNet LPIPS, loaded once and reused for face crops + whole image

    # ── text ──
    ocr_ned: dict[str, float | None] = {label: None for label, _ in parsed}
    if langs:
        ref_text: str | None = None
        if ground_truth:
            gt = json.loads(Path(ground_truth).read_text(encoding="utf-8"))
            ref_text = gt.get(Path(original).name)
            if ref_text is None:
                console.print(f"  (no ground-truth entry for {Path(original).name}; skipping text)")
        else:
            console.print(f"  OCR original ({','.join(langs)})...")
            ref_text = "\n".join(_ocr_lines(ref, langs))
        if ref_text:
            console.print(f"  OCR variants ({','.join(langs)})...")
            for label, img in parsed:
                ocr_ned[label] = _text_ned(ref_text, "\n".join(_ocr_lines(img, langs)))

    # ── faces ──
    face_stats: dict[str, FaceStats] = {label: FaceStats() for label, _ in parsed}
    if not no_faces:
        console.print("  Faces (insightface buffalo_l)...")
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        ref_faces = app.get(ref)
        if ref_faces:
            ref_centers = [_bbox_center(of.bbox) for of in ref_faces]
            ref_diags = [_bbox_diag(of.bbox) for of in ref_faces]
            for label, img in parsed:
                vfaces = app.get(img)
                st = face_stats[label]
                # One-to-one assignment for identity (collision-free); lapvar/LPIPS stay
                # anchored to the original bbox below, so they need no match.
                matched = assign_faces_one_to_one(ref_centers, [_bbox_center(vf.bbox) for vf in vfaces], ref_diags)
                for oi, of in enumerate(ref_faces):
                    st.n_faces += 1
                    vf = vfaces[matched[oi]] if oi in matched else None
                    if vf is not None:
                        st.identity.append(_cosine(of.normed_embedding, vf.normed_embedding))
                    oc, vc = _crop(ref, of.bbox), _crop(img, of.bbox)
                    if oc.size == 0 or vc.size == 0:
                        continue
                    vc_r = cv2.resize(vc, (oc.shape[1], oc.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    st.lpips.append(_lpips_distance(lp, oc, vc_r))
                    ov = _lap_var(oc)
                    st.lapvar_ratio.append(_lap_var(vc_r) / ov if ov > 1e-6 else 0.0)
        else:
            console.print("  (no faces detected in the original; skipping face metrics)")

    # ── whole image ──
    console.print("  Whole-image LPIPS/SSIM/PSNR...")
    whole: dict[str, tuple[float, float, float]] = {}
    for label, img in parsed:
        ssim, psnr = _ssim_psnr(ref, img)
        whole[label] = (_lpips_distance(lp, ref, img), ssim, psnr)

    # ── report ──
    table = Table(title=f"Fidelity vs {Path(original).name} (reference)")
    for col in ("variant", "text NED↓", "faces", "ID cos↑", "face LPIPS↓", "lapvar↑", "img LPIPS↓", "SSIM↑", "PSNR↑"):
        table.add_column(col)
    for label, _ in parsed:
        st = face_stats[label]
        wl, ws, wp = whole[label]
        table.add_row(
            label,
            _fmt(ocr_ned[label]),
            str(st.n_faces),
            _fmt(_mean(st.identity)),
            _fmt(_mean(st.lpips)),
            _fmt(_mean(st.lapvar_ratio)),
            _fmt(wl),
            _fmt(ws),
            _fmt(wp, 1),
        )
    console.print(table)
    console.print(
        "  Legend: NED lower=better; ID cos higher=better; face LPIPS lower=better; "
        "lapvar ratio ~1=detail kept, <1=smoothed/plastic; img LPIPS lower=better; SSIM/PSNR higher=closer."
    )


if __name__ == "__main__":
    cli()
