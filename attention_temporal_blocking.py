#!/usr/bin/env python
"""Block encoded physical-time tokens only at a trained CTM's attention layer."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from nbody_ctm.config import load_config
from nbody_ctm.dataset import make_datasets
from nbody_ctm.metrics import classification_metrics
from nbody_ctm.models import build_model
from nbody_ctm.utils import choose_device, ensure_dir, save_json
from temporal_masking import CLASS_NAMES, CONDITIONS, probabilities, thirds


def encoded_time_masks(token_time: torch.Tensor) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor]]:
    times = torch.unique(token_time, sorted=True)
    partitions = thirds(len(times))
    masks = {}
    for name, slots in partitions.items():
        selected = times[torch.as_tensor(slots, device=times.device)]
        masks[name] = torch.isin(token_time, selected)
    coverage = sum(mask.to(torch.int32) for mask in masks.values())
    if not torch.all(coverage == 1):
        raise AssertionError("Encoded-time masks must cover each token exactly once.")
    if any(not mask.any() or mask.all() for mask in masks.values()):
        raise AssertionError("Each block must hide some, but not all, attention tokens.")
    return {name: times[slots].cpu().numpy() for name, slots in partitions.items()}, masks


@contextmanager
def block_attention_tokens(attention: torch.nn.MultiheadAttention, blocked: torch.Tensor | None):
    """Add -inf to selected attention logits before softmax, at every CTM tick."""
    if blocked is None:
        yield
        return

    def pre_hook(module, args, kwargs):
        if kwargs.get("attn_mask") is not None:
            raise ValueError("Unexpected existing attention mask.")
        query, key, _ = args[:3]
        if key.size(1) != blocked.numel() or query.size(1) != 1:
            raise ValueError("Attention token dimensions changed.")
        additive = torch.zeros((1, blocked.numel()), device=query.device, dtype=query.dtype)
        additive[:, blocked] = float("-inf")
        return args, {**kwargs, "attn_mask": additive}

    handle = attention.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/longer_obs.yaml")
    parser.add_argument("--checkpoint", default="outputs/longer_obs/ctm_seed42/best_final.pt")
    parser.add_argument("--out_dir", default="outputs/longer_obs/ctm_seed42/attention_temporal_blocking")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None, help="Small pipeline check only; omit for the full test set.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("model_name") != "ctm" or checkpoint["cfg"]["model"] != cfg["model"]:
        raise ValueError("A matching CTM checkpoint and model config are required.")
    _, _, test, _ = make_datasets(cfg["data"]["output_path"], cfg["data"])
    if test.state.shape[1] != int(cfg["data"]["obs_steps"]):
        raise ValueError("Dataset observation length differs from the config.")
    n_examples = len(test) if args.max_examples is None else min(args.max_examples, len(test))
    if n_examples < 1:
        raise ValueError("No test examples selected.")
    if args.max_examples is not None:
        test = Subset(test, range(n_examples))
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
    encoded_times = None
    token_counts = None
    checked_attention_weights = set()
    with torch.inference_mode():
        for batch in loader:
            state = batch["state"].to(device)
            mass = batch["mass"].to(device)
            labels.append(batch["label"].numpy())
            indices.append(batch["index"].numpy())
            pair_ids.append(batch["pair_id"].numpy() if "pair_id" in batch else np.full(len(batch["index"]), -1))
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                # All conditions share these unchanged encoder outputs.
                tokens, token_map = model.encoder(state, mass, return_token_map=True)
                time_groups, masks = encoded_time_masks(token_map["time_index"])
                current_times = {k: v.tolist() for k, v in time_groups.items()}
                if encoded_times is None:
                    encoded_times = current_times
                    token_counts = {k: int(v.sum().item()) for k, v in masks.items()}
                elif encoded_times != current_times:
                    raise AssertionError("Encoded time mapping changed between batches.")
                for condition in CONDITIONS:
                    blocked = None if condition == "baseline" else masks[condition]
                    with block_attention_tokens(model.ctm.attention, blocked):
                        verify = condition != "baseline" and condition not in checked_attention_weights
                        logits, _, tracking = model.ctm(tokens, track=verify)
                    if verify:
                        weights = tracking["attention"]  # [batch, tick, head, token]
                        if not torch.all(weights[..., blocked] == 0):
                            raise AssertionError(f"Blocked {condition} tokens received attention weight.")
                        if not torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=2e-3):
                            raise AssertionError("Attention did not renormalize over unblocked tokens.")
                        checked_attention_weights.add(condition)
                    all_logits[condition].append(logits.float().cpu().numpy())

    y = np.concatenate(labels)
    index = np.concatenate(indices)
    pair_id = np.concatenate(pair_ids)
    if not np.array_equal(index, np.arange(n_examples)):
        raise AssertionError("Test examples are not in dataset order.")
    logits = np.stack([np.concatenate(all_logits[c]) for c in CONDITIONS])  # [condition, example, class, tick]
    probs = probabilities(logits)
    pred = logits.argmax(axis=2)
    label_index = y[None, :, None, None].repeat(len(CONDITIONS), axis=0)
    correct_prob = np.take_along_axis(probs, label_index, axis=2).squeeze(2)
    correct_logit = np.take_along_axis(logits, label_index, axis=2).squeeze(2)
    assert np.allclose(probs.sum(axis=2), 1.0, atol=1e-6)

    # With the full test set and default batching, baseline must reproduce the
    # saved checkpoint evaluation before intervention results are published.
    if args.max_examples is None and args.batch_size is None:
        reference = Path(args.checkpoint).with_name("test_predictions.npz")
        if reference.exists():
            with np.load(reference, allow_pickle=False) as saved:
                if not np.array_equal(saved["index"], index) or not np.array_equal(saved["label"], y):
                    raise AssertionError("Baseline test order differs from saved evaluation.")
                if not np.array_equal(saved["logits"].argmax(axis=1), pred[0]):
                    raise AssertionError("Baseline predictions differ from saved checkpoint evaluation.")

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
        summaries[condition] = {
            "final": tick_metrics[-1], "per_tick": tick_metrics,
            "prediction_flips": int((pred[ci, :, -1] != pred[0, :, -1]).sum()),
            "previously_correct_now_wrong": int(((pred[0, :, -1] == y) & (pred[ci, :, -1] != y)).sum()),
        }
        for i in range(n_examples):
            rows.append({
                "condition": condition, "index": int(index[i]), "pair_id": int(pair_id[i]),
                "label": int(y[i]), "class_name": CLASS_NAMES[int(y[i])],
                "baseline_prediction": int(pred[0, i, -1]), "blocked_prediction": int(pred[ci, i, -1]),
                "baseline_correct": bool(pred[0, i, -1] == y[i]), "blocked_correct": bool(pred[ci, i, -1] == y[i]),
                "prediction_flip": bool(pred[0, i, -1] != pred[ci, i, -1]),
                "previously_correct_now_wrong": bool(pred[0, i, -1] == y[i] and pred[ci, i, -1] != y[i]),
                "baseline_correct_class_probability": float(correct_prob[0, i, -1]),
                "blocked_correct_class_probability": float(correct_prob[ci, i, -1]),
                "correct_class_probability_drop": float(correct_prob[0, i, -1] - correct_prob[ci, i, -1]),
                "baseline_correct_class_logit": float(correct_logit[0, i, -1]),
                "blocked_correct_class_logit": float(correct_logit[ci, i, -1]),
                "correct_class_logit_drop": float(correct_logit[0, i, -1] - correct_logit[ci, i, -1]),
            })

    out = ensure_dir(args.out_dir)
    save_json({"config": str(Path(args.config).resolve()), "checkpoint": str(Path(args.checkpoint).resolve()),
               "dataset": str(Path(cfg["data"]["output_path"]).resolve()), "n_examples": n_examples,
               "condition_order": CONDITIONS, "encoded_time_values_zero_based": encoded_times,
               "blocked_token_counts": token_counts, "results": summaries}, out / "summary.json")
    pd.DataFrame(rows).to_csv(out / "per_example.csv", index=False)
    np.savez_compressed(out / "paired_predictions.npz", condition=np.array(CONDITIONS), index=index, label=y,
                        pair_id=pair_id, logits=logits, probabilities=probs, prediction=pred,
                        correct_class_probability=correct_prob, correct_class_logit=correct_logit)
    print(f"Saved {n_examples} paired test examples to {out}")


if __name__ == "__main__":
    main()
