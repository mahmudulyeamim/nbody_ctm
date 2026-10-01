#!/usr/bin/env python
"""Dataset audit + shortcut diagnostics.

Use this before expensive training. It measures class balance, event times, counterfactual
coverage, and how well deliberately simple handcrafted features predict the outcome.
If a linear/RandomForest shortcut is already near-perfect, shorten the observation window
or increase counterfactual enrichment before interpreting CTM dynamics.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def summary_features(state: np.ndarray, mass: np.ndarray) -> np.ndarray:
    # state [N,T,B,4]
    pos, vel = state[..., :2], state[..., 2:]
    # center of mass approximate (mass-weighted)
    m = mass[:, None, :, None]
    com = (pos * m).sum(2) / m.sum(2)
    rel = pos - com[:, :, None, :]
    radius = np.linalg.norm(rel, axis=-1)
    speed = np.linalg.norm(vel, axis=-1)
    disp = pos[:, :, :, None, :] - pos[:, :, None, :, :]
    dist = np.linalg.norm(disp, axis=-1)
    eye = np.eye(pos.shape[2], dtype=bool)[None, None]
    dist = np.where(eye, np.inf, dist)
    minsep = dist.min(axis=(2, 3))
    rmax = radius.max(2)
    smax = speed.max(2)
    # small interpretable set: endpoint + extrema + late trends
    k = max(3, state.shape[1] // 5)
    feats = np.stack([
        minsep.min(1), minsep[:, -1],
        rmax.max(1), rmax[:, -1],
        smax.max(1), smax[:, -1],
        rmax[:, -k:].mean(1) - rmax[:, :k].mean(1),
        minsep[:, -k:].mean(1) - minsep[:, :k].mean(1),
        speed.std(axis=2).mean(1),
    ], axis=1)
    return feats.astype(np.float32)


def fit_eval(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    return accuracy_score(yte, p), f1_score(yte, p, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=False)
    report = {}
    for split in ["train", "val", "test"]:
        y = d[f"{split}_label"]
        report[split] = {
            "n": int(len(y)),
            "class_counts": np.bincount(y, minlength=3).tolist(),
            "counterfactual_fraction": float(d[f"{split}_is_counterfactual"].mean()),
            "event_step_quantiles": np.quantile(d[f"{split}_event_step"][d[f"{split}_event_step"] >= 0], [0, .25, .5, .75, 1]).tolist() if np.any(d[f"{split}_event_step"] >= 0) else [],
            "margin_quantiles": np.quantile(d[f"{split}_outcome_margin"], [0, .25, .5, .75, 1]).tolist(),
            "finite": bool(np.isfinite(d[f"{split}_state"]).all() and np.isfinite(d[f"{split}_mass"]).all()),
        }
    print(json.dumps(report, indent=2))

    Xtr_s = summary_features(d["train_state"], d["train_mass"])
    Xte_s = summary_features(d["test_state"], d["test_mass"])
    ytr, yte = d["train_label"], d["test_label"]
    # last physical state flattened + masses
    Xtr_last = np.concatenate([d["train_state"][:, -1].reshape(len(ytr), -1), d["train_mass"]], 1)
    Xte_last = np.concatenate([d["test_state"][:, -1].reshape(len(yte), -1), d["test_mass"]], 1)

    rows = []
    for name, Xtr, Xte in [("handcrafted_summary", Xtr_s, Xte_s), ("last_frame", Xtr_last, Xte_last)]:
        for model_name, model in [
            ("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
            ("random_forest", RandomForestClassifier(n_estimators=250, max_depth=10, n_jobs=-1, random_state=42)),
        ]:
            acc, f1 = fit_eval(model, Xtr, ytr, Xte, yte)
            rows.append({"features": name, "model": model_name, "accuracy": acc, "macro_f1": f1})
    df = pd.DataFrame(rows)
    print("\nShortcut diagnostics (lower is better for a challenging sequence task):")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
