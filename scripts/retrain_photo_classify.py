#!/usr/bin/env python3
"""Retrain the 2026-08-31 photo heads from a sha256-keyed cache pack.

No image I/O. Requires catalog.json, embeddings-by-sha256.npz or extra-clip-l.npz
plus embeddings keyed by sha256, features-124d.npz, and probe-weights-clip-l-ft.npz.

    uv run python scripts/retrain_photo_classify.py --pack DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)

DETECTOR_SEED = 20260940
PROVIDER_SEED = 20260956
EPOCHS = 40
MLP_WIDTH = 512
LR = 5e-4
BATCH = 64
TARGET_PHOTO_FPR = 0.0167
FOCAL_GAMMA = 2.0
FOCAL_EPOCHS = 40
FOCAL_LR = 1e-3
FOCAL_BATCH = 64
LABELS = ("openai", "google", "tc260", "meta_muse_image", "no_ai")


def _load_sha_vectors(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if "sha256" in payload.files:
        keys = payload["sha256"]
        vectors = payload["vectors"]
        return {str(key): np.asarray(vector) for key, vector in zip(keys, vectors, strict=True)}
    raise SystemExit(f"{path} must contain sha256 and vectors")


def _mlp(dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, MLP_WIDTH),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(MLP_WIDTH, MLP_WIDTH // 4),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(MLP_WIDTH // 4, 1),
    )


def _provider_head() -> nn.Module:
    return nn.Sequential(
        nn.Linear(124, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )


def _focal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, prob, 1.0 - prob)
    return ((1.0 - pt) ** FOCAL_GAMMA * bce).mean()


def _train_detector(emb: dict[str, np.ndarray], catalog: dict) -> tuple[nn.Module, float]:
    torch.manual_seed(DETECTOR_SEED)
    roles = {row["sha256"]: set(row["roles"]) for row in catalog["rows"]}
    pos, neg = [], []
    for sha, vector in emb.items():
        role = roles.get(sha, set())
        if "detector_ai_train" in role or "detector_ai_extra" in role:
            pos.append(vector)
        if "detector_photo_train" in role or "detector_struct" in role:
            neg.append(vector)
    x_pos = torch.from_numpy(np.stack(pos)).float()
    x_neg = torch.from_numpy(np.stack(neg)).float()
    log.info("detector train pos=%d neg=%d", len(x_pos), len(x_neg))
    model = _mlp(x_pos.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(DETECTOR_SEED)
    model.train()
    for _epoch in range(EPOCHS):
        pp = rng.permutation(len(x_pos))
        pn = rng.permutation(len(x_neg))
        for start in range(0, min(len(pp), len(pn)), BATCH):
            features = torch.cat([x_pos[pp[start : start + BATCH]], x_neg[pn[start : start + BATCH]]])
            labels = torch.cat([torch.ones(min(BATCH, len(pp) - start)), torch.zeros(min(BATCH, len(pn) - start))])
            optimizer.zero_grad()
            loss_fn(model(features).squeeze(-1), labels).backward()
            optimizer.step()
    model.eval()
    dev = np.stack(
        [
            emb[row["sha256"]]
            for row in catalog["rows"]
            if "detector_photo_dev_oi" in row["roles"] and row["sha256"] in emb
        ]
    )
    with torch.inference_mode():
        scores = model(torch.from_numpy(dev).float()).squeeze(-1).numpy()
    threshold = float(np.quantile(scores, 1.0 - TARGET_PHOTO_FPR))
    return model, threshold


def _train_provider(feat: dict[str, np.ndarray], catalog: dict) -> dict[str, nn.Module]:
    def stack(role: str) -> np.ndarray:
        vecs = [feat[row["sha256"]] for row in catalog["rows"] if role in row["roles"] and row["sha256"] in feat]
        return np.stack(vecs) if vecs else np.empty((0, 124), dtype=np.float32)

    extra = stack("provider_openai_extra")
    openai = stack("provider_openai")
    bank = {
        "openai": np.concatenate([openai, extra]) if len(extra) else openai,
        "google": stack("provider_google"),
        "tc260": stack("provider_tc260"),
        "meta_muse_image": stack("provider_meta_train"),
        "no_ai": stack("provider_no_ai"),
    }
    struct = stack("provider_struct")
    all_neg = np.concatenate([bank["no_ai"], struct]) if len(struct) else bank["no_ai"]
    heads = {}
    for i, name in enumerate(LABELS):
        if name == "no_ai":
            positive = all_neg
            negative = np.concatenate([bank[n] for n in LABELS if n != "no_ai"])
        else:
            negative = np.concatenate([bank[n] for n in LABELS if n not in (name, "no_ai")] + [all_neg])
            positive = bank[name]
        torch.manual_seed(PROVIDER_SEED + i)
        rng = np.random.default_rng(PROVIDER_SEED + i)
        max_neg = min(len(negative), len(positive) * 2)
        neg_idx = rng.permutation(len(negative))[:max_neg]
        x_pos = torch.from_numpy(positive.astype(np.float32))
        x_neg = torch.from_numpy(negative[neg_idx].astype(np.float32))
        model = _provider_head()
        optimizer = torch.optim.AdamW(model.parameters(), lr=FOCAL_LR, weight_decay=1e-3)
        model.train()
        for _epoch in range(FOCAL_EPOCHS):
            pos_idx = rng.permutation(len(x_pos))
            neg_order = rng.permutation(len(x_neg))
            for start in range(0, min(len(pos_idx), len(neg_order)), FOCAL_BATCH):
                features = torch.cat(
                    [x_pos[pos_idx[start : start + FOCAL_BATCH]], x_neg[neg_order[start : start + FOCAL_BATCH]]]
                )
                labels = torch.cat(
                    [
                        torch.ones(min(FOCAL_BATCH, len(pos_idx) - start)),
                        torch.zeros(min(FOCAL_BATCH, len(neg_order) - start)),
                    ]
                )
                optimizer.zero_grad()
                _focal_loss(model(features).squeeze(-1), labels).backward()
                optimizer.step()
        heads[name] = model.eval()
        log.info("provider %s pos=%d", name, len(positive))
    return heads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pack = args.pack.expanduser().resolve()
    catalog = json.loads((pack / "catalog.json").read_text())
    emb: dict[str, np.ndarray] = {}
    for name in ("embeddings-by-sha256.npz", "extra-clip-l.npz"):
        path = pack / name
        if path.is_file():
            emb.update(_load_sha_vectors(path))
    feat = _load_sha_vectors(pack / "features-124d.npz")
    probe = np.load(pack / "probe-weights-clip-l-ft.npz")
    log.info("catalog=%d embeddings=%d features=%d", catalog["counts"]["files"], len(emb), len(feat))
    args.out.mkdir(parents=True, exist_ok=True)
    detector, threshold = _train_detector(emb, catalog)
    torch.save(detector.state_dict(), args.out / "detector.pt")
    heads = _train_provider(feat, catalog)
    torch.save({name: head.state_dict() for name, head in heads.items()}, args.out / "provider.pt")
    (args.out / "retrain.json").write_text(
        json.dumps({"detector_threshold": threshold, "probe_thr": float(probe["thr_oi_1pct"])}, indent=2) + "\n"
    )
    log.info("wrote %s threshold=%.6f", args.out, threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
