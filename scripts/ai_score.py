"""AI-generation scorer for scanned datasets.

Trains a gradient-boosted classifier on the metadata-derived labels of a
scan_dataset.py output and scores every file, WITHOUT needing the
metadata to be present at scoring time: the features are pixel and
container statistics, so a metadata-stripped file still gets a score.

Measured on the production corpus (temporal holdout, honest for drift):
structural-feature GBM reaches AUC ~0.94-0.96 on-distribution. The model
is distribution-specific by design (it is trained on this service's
traffic); do not expect the same numbers on unrelated data. CLIP-ViT-L/14
features were evaluated as an alternative and scored lower (0.82) while
being ~30x slower, so the shipped model uses the structural features the
scanner already collects.

Modes:
    uv run --with scikit-learn python scripts/ai_score.py train <scan_glob> <model.pkl>
    uv run --with scikit-learn python scripts/ai_score.py score <scan_glob> <model.pkl> <out.jsonl>

<scan_glob> is a glob of scan_dataset shards, e.g. 'data/scan/part_*.jsonl'.
score output: one JSON line per file with file, sha256, ai_score, and the
label evidence when a metadata label existed (for monitoring drift).
"""

import glob
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

# AI-generator names in C2PA claim_generator_info (metadata-level truth).
AI_GENERATORS = (
    "openai",
    "adobe firefly",
    "adobe_firefly",
    "microsoft responsible ai",
    "microsoft_designer",
    "black forest labs",
    "fal-ai",
    "bria",
    "chatgpt",
    "stability",
    "dreamina",
    "canva",
)
# Human-origin software (negative evidence, NOT proof by itself).
HUMAN_SOFTWARE = (
    "photoshop",
    "lightroom",
    "picsart",
    "snapseed",
    "paint.net",
    "gimp",
    "capture one",
    "meitu",
    "xingtu",
    "snow",
)

_FEAT_SCALAR = [
    ("noise", "noise_std"),
    ("noise", "noise_kurtosis"),
    ("fft", "cfa_peak"),
    ("ela", "ela_mean"),
    ("ela", "ela_p95"),
    ("gradient", "laplacian_var"),
    ("color", "saturation_mean"),
    ("color", "value_mean"),
    ("dct", "benford_mad"),
]


def label_of(record: dict[str, Any]) -> int | None:
    """1 = strong metadata AI, 0 = human-origin, None = unlabeled."""
    store = record.get("c2pa_store") or {}
    for m in (store.get("manifests") or {}).values():
        for cgi in m.get("claim_generator_info") or []:
            if any(g in str(cgi.get("name", "")).lower() for g in AI_GENERATORS):
                return 1
        for a in m.get("assertions") or []:
            d = a.get("data") if isinstance(a.get("data"), dict) else {}
            if "trainedAlgorithmicMedia" in str(d.get("digitalSourceType", "")):
                return 1
    for c in record.get("png_chunks", []):
        t = c.get("text", "")
        kw = t.split("\x00")[0]
        if kw in ("prompt", "workflow", "parameters") or "AIGC" in t[:80]:
            return 1
    exif = record.get("exif", {})
    z = exif.get("0th") or {}
    e = exif.get("Exif") or {}
    if z.get("Make") and z.get("Model") and (e.get("MakerNote") or e.get("LensModel") or e.get("LensMake")):
        return 0
    for c in record.get("png_chunks", []):
        if c.get("apple_screenshot_marker"):
            return 0
    for v in (record.get("iptc") or {}).values():
        if str(v).strip() == "Screenshot":
            return 0
    if any(t in str(z.get("Software", "")).lower() for t in HUMAN_SOFTWARE):
        return 0
    return None


def features_of(record: dict[str, Any]) -> list[float]:
    """The structural feature vector (pixel + container stats, no metadata)."""
    v = [float((record.get(s) or {}).get(k, np.nan)) for s, k in _FEAT_SCALAR]
    for b in (record.get("fft") or {}).get("fft_band_energy", []):
        v.append(float(b))
    for h in (record.get("gradient") or {}).get("gradient_hist", []):
        v.append(float(h))
    for h in (record.get("color") or {}).get("color_hist_4x4x4", []):
        v.append(float(h))
    jf = record.get("jpeg_forensics") or {}
    v.append(1.0 if jf.get("subsampling") == "4:4:4" else 0.0)
    v.append(1.0 if jf.get("progressive") else 0.0)
    pil = record.get("pil") or {}
    w, h = pil.get("width") or 0, pil.get("height") or 0
    v += [float(w * h) / 1e6, float(w) / max(h, 1)]
    fmt = record.get("content_format")
    v += [1.0 if fmt == "jpeg" else 0.0, 1.0 if fmt == "png" else 0.0]
    return v


def iter_records(pattern: str) -> Any:
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            for line in fh:
                yield json.loads(line)


def cmd_train(pattern: str, model_path: str) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    X, y, dates = [], [], []
    for r in iter_records(pattern):
        if "noise" not in r:
            continue
        lab = label_of(r)
        if lab is None:
            continue
        X.append(features_of(r))
        y.append(lab)
        dates.append(Path(r["file"]).parent.name)
    X, y = np.array(X), np.array(y)
    print(f"labeled: {len(y)} (pos {int(y.sum())}, neg {int((1 - y).sum())}, features {X.shape[1]})")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    clf = HistGradientBoostingClassifier(random_state=0, max_iter=300)
    clf.fit(imp.transform(Xtr), ytr)
    p = clf.predict_proba(imp.transform(Xte))[:, 1]
    print(f"holdout: AUC {roc_auc_score(yte, p):.4f} | AP {average_precision_score(yte, p):.4f}")

    uniq = sorted(set(dates))
    cutoff = uniq[int(len(uniq) * 0.7)]
    tr = np.array([d < cutoff for d in dates])
    if tr.sum() > 100 and (~tr).sum() > 100:
        clf_t = HistGradientBoostingClassifier(random_state=0, max_iter=300)
        imp_t = SimpleImputer(strategy="median").fit(X[tr])
        clf_t.fit(imp_t.transform(X[tr]), y[tr])
        pt = clf_t.predict_proba(imp_t.transform(X[~tr]))[:, 1]
        auc_t = roc_auc_score(y[~tr], pt)
        ap_t = average_precision_score(y[~tr], pt)
        print(f"temporal (<{cutoff}): AUC {auc_t:.4f} | AP {ap_t:.4f}")

    imp = SimpleImputer(strategy="median").fit(X)
    clf = HistGradientBoostingClassifier(random_state=0, max_iter=300)
    clf.fit(imp.transform(X), y)
    with open(model_path, "wb") as f:
        pickle.dump({"imputer": imp, "clf": clf}, f)
    print(f"model written: {model_path}")


def cmd_score(pattern: str, model_path: str, out_path: str) -> None:
    # the model file is produced locally by `train`; do not load third-party pickles
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)  # noqa: S301
    imp, clf = bundle["imputer"], bundle["clf"]
    n = 0
    with open(out_path, "w") as out:
        for r in iter_records(pattern):
            if "noise" not in r:
                continue
            score = float(clf.predict_proba(imp.transform([features_of(r)]))[0, 1])
            lab = label_of(r)
            out.write(
                json.dumps(
                    {
                        "file": r["file"],
                        "sha256": r.get("sha256"),
                        "ai_score": round(score, 4),
                        "metadata_label": lab,
                    }
                )
                + "\n"
            )
            n += 1
            if n % 5000 == 0:
                print(f"  {n}", flush=True)
    print(f"scored {n} -> {out_path}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "score"):
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "train" and len(sys.argv) == 4:
        cmd_train(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "score" and len(sys.argv) == 5:
        cmd_score(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
