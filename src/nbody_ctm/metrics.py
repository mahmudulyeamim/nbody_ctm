from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def multiclass_brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1], dtype=np.float64)[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def classification_metrics(logits: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    logits = np.asarray(logits)
    z = logits - logits.max(1, keepdims=True)
    probs = np.exp(z)
    probs /= probs.sum(1, keepdims=True)
    pred = probs.argmax(1)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "ece": expected_calibration_error(probs, y),
        "brier": multiclass_brier(probs, y),
        "mean_confidence": float(probs.max(1).mean()),
    }
    try:
        out["auc_ovr_macro"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except ValueError:
        out["auc_ovr_macro"] = float("nan")
    return out
