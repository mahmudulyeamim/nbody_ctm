#!/usr/bin/env python
"""Evaluate raw-input temporal masking on a fixed, trained CTM checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from nbody_ctm.config import load_config
from nbody_ctm.dataset import make_datasets
from nbody_ctm.metrics import classification_metrics
from nbody_ctm.models import build_model
from nbody_ctm.utils import choose_device, ensure_dir, save_json


CONDITIONS = ("baseline", "early", "middle", "late")
CLASS_NAMES = ("bound", "collision", "escape")


def thirds(n_steps: int) -> dict[str, np.ndarray]:
    if n_steps < 3:
        raise ValueError("At least three observed timesteps are required.")
    pieces = np.array_split(np.arange(n_steps), 3)
    combined = np.concatenate(pieces)
    if len(np.unique(combined)) != n_steps or not np.array_equal(np.sort(combined), np.arange(n_steps)):
        raise AssertionError("Temporal masks must cover each observed timestep exactly once.")
    return dict(zip(CONDITIONS[1:], pieces))


def probabilities(logits: np.ndarray) -> np.ndarray:
    # Input shape is [condition, example, class, tick].
    z = logits - logits.max(axis=2, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=2, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/longer_obs.yaml")
    parser.add_argument("--checkpoint", default="outputs/longer_obs/ctm_seed42/best_final.pt")
    parser.add_argument("--out_dir", default="outputs/longer_obs/ctm_seed42/temporal_masking")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None, help="Small pipeline check only; omit for the full test set.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("model_name") != "ctm":
        raise ValueError("This script requires a CTM checkpoint.")
    if checkpoint["cfg"]["model"] != cfg["model"]:
        raise ValueError("Checkpoint and config model settings differ.")

    _, _, test, stats = make_datasets(cfg["data"]["output_path"], cfg["data"])
    n_steps = int(cfg["data"]["obs_steps"])
    if test.state.shape[1] != n_steps:
        raise ValueError("Dataset observation length differs from the config.")
    masks = thirds(n_steps)
    n_examples = len(test) if args.max_examples is None else min(args.max_examples, len(test))
    if n_examples < 1:
        raise ValueError("No test examples selected.")
    if args.max_examples is not None:
        test = torch.utils.data.Subset(test, range(n_examples))
    loader = DataLoader(test, batch_size=args.batch_size or int(cfg["training"].get("eval_batch_size", 96)), shuffle=False)

    model = build_model(cfg, "ctm").to(device)
    first = next(iter(loader))
    with torch.no_grad():
        model(first["state"].to(device), first["mass"].to(device))  # initialize lazy layers
    model.load_state_dict(checkpoint["model"])
    model.eval()

    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg["training"].get("bf16", True) else torch.float16
    all_logits = {condition: [] for condition in CONDITIONS}
    labels, indices, pair_ids = [], [], []
    with torch.inference_mode():
        for batch in loader:
            state = batch["state"].to(device)
            mass = batch["mass"].to(device)
            labels.append(batch["label"].numpy())
            indices.append(batch["index"].numpy())
            pair_ids.append(batch["pair_id"].numpy() if "pair_id" in batch else np.full(len(batch["index"]), -1))
            for condition in CONDITIONS:
                masked = state if condition == "baseline" else state.clone()
                if condition != "baseline":
                    # The loader has already normalized raw state with training-set statistics.
                    # Zero here is exactly the per-feature training mean in raw input space.
                    masked[:, masks[condition], :, :] = 0.0
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits, _, _ = model(masked, mass)
                all_logits[condition].append(logits.float().cpu().numpy())

    y = np.concatenate(labels)
    index = np.concatenate(indices)
    pair_id = np.concatenate(pair_ids)
    if not np.array_equal(index, np.arange(n_examples)):
        raise AssertionError("Test examples are not in their original dataset order.")
    logits = np.stack([np.concatenate(all_logits[c]) for c in CONDITIONS])  # [condition, example, class, tick]
    probs = probabilities(logits)
    pred = logits.argmax(axis=2)  # [condition, example, tick]
    correct_prob = np.take_along_axis(probs, y[None, :, None, None].repeat(len(CONDITIONS), axis=0), axis=2).squeeze(2)
    correct_logit = np.take_along_axis(logits, y[None, :, None, None].repeat(len(CONDITIONS), axis=0), axis=2).squeeze(2)
    # The gather index must have one entry per thought tick.
    assert correct_prob.shape == (len(CONDITIONS), n_examples, logits.shape[-1])

    summaries = {}
    rows = []
    for ci, condition in enumerate(CONDITIONS):
        tick_metrics = []
        for tick in range(logits.shape[-1]):
            metrics = classification_metrics(logits[ci, :, :, tick], y)
            metrics["tick"] = tick + 1
            metrics["per_class_recall"] = {
                name: float((pred[ci, y == cls, tick] == cls).mean()) if np.any(y == cls) else None
                for cls, name in enumerate(CLASS_NAMES)
            }
            metrics["mean_correct_class_probability_drop"] = float((correct_prob[0, :, tick] - correct_prob[ci, :, tick]).mean())
            metrics["mean_correct_class_logit_drop"] = float((correct_logit[0, :, tick] - correct_logit[ci, :, tick]).mean())
            tick_metrics.append(metrics)
        summaries[condition] = {"final": tick_metrics[-1], "per_tick": tick_metrics,
                                "previously_correct_now_wrong": int(((pred[0, :, -1] == y) & (pred[ci, :, -1] != y)).sum())}
        for i in range(n_examples):
            rows.append({
                "condition": condition, "index": int(index[i]), "pair_id": int(pair_id[i]),
                "label": int(y[i]), "class_name": CLASS_NAMES[int(y[i])],
                "baseline_prediction": int(pred[0, i, -1]), "masked_prediction": int(pred[ci, i, -1]),
                "baseline_correct": bool(pred[0, i, -1] == y[i]), "masked_correct": bool(pred[ci, i, -1] == y[i]),
                "previously_correct_now_wrong": bool(pred[0, i, -1] == y[i] and pred[ci, i, -1] != y[i]),
                "baseline_correct_class_probability": float(correct_prob[0, i, -1]),
                "masked_correct_class_probability": float(correct_prob[ci, i, -1]),
                "correct_class_probability_drop": float(correct_prob[0, i, -1] - correct_prob[ci, i, -1]),
                "baseline_correct_class_logit": float(correct_logit[0, i, -1]),
                "masked_correct_class_logit": float(correct_logit[ci, i, -1]),
                "correct_class_logit_drop": float(correct_logit[0, i, -1] - correct_logit[ci, i, -1]),
            })

    out = ensure_dir(args.out_dir)
    save_json({"config": str(Path(args.config).resolve()), "checkpoint": str(Path(args.checkpoint).resolve()),
               "dataset": str(Path(cfg["data"]["output_path"]).resolve()), "n_examples": n_examples,
               "condition_order": CONDITIONS, "masked_timesteps_zero_based": {k: v.tolist() for k, v in masks.items()},
               "replacement_raw_state_mean": stats.state_mean.tolist(), "results": summaries}, out / "summary.json")
    pd.DataFrame(rows).to_csv(out / "per_example.csv", index=False)
    np.savez_compressed(out / "paired_predictions.npz", condition=np.array(CONDITIONS), index=index, label=y,
                        pair_id=pair_id, logits=logits, probabilities=probs, prediction=pred,
                        correct_class_probability=correct_prob, correct_class_logit=correct_logit)
    print(f"Saved {n_examples} paired test examples to {out}")


if __name__ == "__main__":
    main()
