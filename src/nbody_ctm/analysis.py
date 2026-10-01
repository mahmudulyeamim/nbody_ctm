from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, f1_score, accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from .dataset import make_datasets
from .metrics import classification_metrics
from .models import build_model
from .utils import ensure_dir, save_json


CLASS_NAMES = ["bound", "collision", "escape"]
DIFF_NAMES = ["easy", "medium", "hard"]


def _move(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def assign_class_conditional_difficulty(labels: np.ndarray, margin: np.ndarray) -> np.ndarray:
    """0 easy, 1 medium, 2 hard; small robust margin = hard, independently per class."""
    out = np.zeros_like(labels, dtype=np.int64)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        order = idx[np.argsort(margin[idx])]  # smallest margin hardest
        thirds = np.array_split(order, 3)
        out[thirds[0]] = 2
        out[thirds[1]] = 1
        out[thirds[2]] = 0
    return out


def settle_tick(pred: np.ndarray) -> np.ndarray:
    """Earliest 1-indexed tick after which predicted class never changes again."""
    N, T = pred.shape
    out = np.full(N, T, dtype=np.int64)
    final = pred[:, -1]
    for t in range(T):
        stable = (pred[:, t:] == final[:, None]).all(1)
        unset = out == T
        out[stable & unset] = t + 1
    return out


def _save_line(df, x, ys, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in ys:
        ax.plot(df[x], df[col], marker="o", label=col)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_analysis(cfg: Dict, *, data_path: str, checkpoint: str, out_dir: str, device: torch.device):
    out = ensure_dir(out_dir)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "ctm")
    if model_name not in {"ctm", "linear_ctm"}:
        raise ValueError("Internal analysis requires a CTM checkpoint.")
    model = build_model(cfg, model_name).to(device)

    train_ds, val_ds, test_ds, stats = make_datasets(data_path, cfg["data"])
    # initialize lazy encoder/CTM layers
    first = test_ds[0]
    with torch.no_grad():
        model(first["state"][None].to(device), first["mass"][None].to(device))
    model.load_state_dict(ckpt["model"])
    model.eval()

    ana_cfg = cfg.get("analysis", {})
    batch_size = int(ana_cfg.get("batch_size", 32))
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    max_internal = int(ana_cfg.get("max_internal_examples", 2000))

    labels_all, margins_all, pair_ids_all, event_a_all, event_b_all = [], [], [], [], []
    logits_all, certainty_all, indices_all = [], [], []
    internal_post, internal_sync_out, internal_labels, internal_indices = [], [], [], []
    temporal_attention_all = []
    pair_post = {}

    time_attention_sum = None
    temporal_entropy_sum = None
    temporal_cosine_sum = None
    body_attention_sum = None
    class_time_sum = None
    class_count = np.zeros(3, dtype=np.int64)
    event_body_mass_sum = np.zeros((3, int(cfg["model"]["ctm"]["iterations"])), dtype=np.float64)
    event_body_count = np.zeros(3, dtype=np.int64)
    token_time_np = token_body_np = None
    total_count = 0

    with torch.inference_mode():
        for batch in loader:
            raw_indices = batch["index"].numpy()
            raw_pair = batch.get("pair_id", torch.full_like(batch["index"], -1)).numpy()
            raw_margin = batch.get("outcome_margin", torch.zeros_like(batch["index"], dtype=torch.float32)).numpy()
            raw_ea = batch.get("event_body_a", torch.full_like(batch["index"], -1)).numpy()
            raw_eb = batch.get("event_body_b", torch.full_like(batch["index"], -1)).numpy()
            batch = _move(batch, device)
            pred, cert, tracking = model(batch["state"], batch["mass"], track=True)
            labels = batch["label"].cpu().numpy()
            logits = pred.float().cpu().numpy()
            cert_np = cert.float().cpu().numpy()
            attn = tracking["attention"].float().cpu().numpy().mean(2)  # [B,tick,L], mean heads
            post = tracking["post_activations"].float().cpu().numpy()  # [B,tick,D]
            sync_out = tracking["sync_out"].float().cpu().numpy()
            token_time_np = tracking["token_time_index"].cpu().numpy()
            token_body_np = tracking["token_body_index"].cpu().numpy()
            ticks = logits.shape[-1]
            unique_times = np.unique(token_time_np)
            n_bodies = int(cfg["data"]["n_bodies"])
            if time_attention_sum is None:
                time_attention_sum = np.zeros((ticks, len(unique_times)), dtype=np.float64)
                temporal_entropy_sum = np.zeros(ticks, dtype=np.float64)
                temporal_cosine_sum = np.zeros(ticks, dtype=np.float64)
                body_attention_sum = np.zeros((ticks, n_bodies), dtype=np.float64)
                class_time_sum = np.zeros((3, ticks, len(unique_times)), dtype=np.float64)
            temporal = np.stack([attn[..., token_time_np == pt].sum(-1) for pt in unique_times], axis=-1)
            temporal = temporal / np.maximum(temporal.sum(-1, keepdims=True), 1e-12)
            temporal_attention_all.append(temporal.astype(np.float16))
            temporal_entropy_sum += (-(temporal * np.log(np.maximum(temporal, 1e-12))).sum(-1)).sum(0)
            if ticks > 1:
                a, b = temporal[:, 1:], temporal[:, :-1]
                temporal_cosine_sum[1:] += ((a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)).sum(0)
            for ti, pt in enumerate(unique_times):
                mask = token_time_np == pt
                vals = temporal[..., ti]
                time_attention_sum[:, ti] += vals.sum(0)
                for c in range(3):
                    cm = labels == c
                    if cm.any():
                        class_time_sum[c, :, ti] += vals[cm].sum(0)
            for bi in range(n_bodies):
                mask = token_body_np == bi
                body_attention_sum[:, bi] += attn[..., mask].sum(-1).sum(0)
            for c in range(3):
                cm = labels == c
                class_count[c] += int(cm.sum())
                if c in (1, 2) and cm.any():
                    # attention mass on outcome-involved bodies (collision pair or escaping body)
                    for local in np.where(cm)[0]:
                        involved = [int(raw_ea[local])]
                        if c == 1 and int(raw_eb[local]) >= 0:
                            involved.append(int(raw_eb[local]))
                        involved = [x for x in involved if x >= 0]
                        if involved:
                            mask = np.isin(token_body_np, involved)
                            event_body_mass_sum[c] += attn[local][:, mask].sum(-1)
                            event_body_count[c] += 1

            labels_all.append(labels)
            margins_all.append(raw_margin)
            pair_ids_all.append(raw_pair)
            event_a_all.append(raw_ea)
            event_b_all.append(raw_eb)
            logits_all.append(logits)
            certainty_all.append(cert_np)
            indices_all.append(raw_indices)
            total_count += len(labels)

            remain = max_internal - sum(len(x) for x in internal_labels)
            if remain > 0:
                take = min(remain, len(labels))
                internal_post.append(post[:take])
                internal_sync_out.append(sync_out[:take])
                internal_labels.append(labels[:take])
                internal_indices.append(raw_indices[:take])
            # preserve all counterfactual pairs; expected to be manageable
            for local, pid in enumerate(raw_pair):
                if pid >= 0:
                    pair_post[int(pid)] = pair_post.get(int(pid), []) + [
                        (int(raw_indices[local]), int(labels[local]), post[local], logits[local])
                    ]

    y = np.concatenate(labels_all)
    margin = np.concatenate(margins_all)
    pair_ids = np.concatenate(pair_ids_all)
    logits = np.concatenate(logits_all)  # [N,C,T]
    certainty = np.concatenate(certainty_all)
    indices = np.concatenate(indices_all)
    np.savez_compressed(
        out / "attention_temporal_per_example.npz",
        index=indices,
        physical_step=np.unique(token_time_np),
        attention=np.concatenate(temporal_attention_all),  # [example, CTM tick, encoded time]
    )
    T = logits.shape[-1]
    probs = np.stack([_softmax_np(logits[..., t]) for t in range(T)], axis=-1)  # [N,C,T]
    pred_class = probs.argmax(1)
    difficulty = assign_class_conditional_difficulty(y, margin)
    settle = settle_tick(pred_class)

    # --- Per tick metrics ---
    tick_rows = []
    for t in range(T):
        m = classification_metrics(logits[..., t], y)
        m.update({"tick": t + 1, "certainty": float(certainty[:, t].mean()), "predicted_class_entropy": float(-(np.bincount(pred_class[:, t], minlength=3)/len(y) * np.log(np.maximum(np.bincount(pred_class[:, t], minlength=3)/len(y),1e-12))).sum())})
        for c in range(3):
            m[f"pred_frac_{CLASS_NAMES[c]}"] = float((pred_class[:, t] == c).mean())
        tick_rows.append(m)
    tick_df = pd.DataFrame(tick_rows)
    tick_df.to_csv(out / "per_tick_metrics.csv", index=False)
    _save_line(tick_df, "tick", ["accuracy", "macro_f1", "certainty"], "Prediction quality across CTM thought ticks", "score", out / "01_tick_metrics.png")
    _save_line(tick_df, "tick", ["ece", "brier"], "Calibration across CTM thought ticks", "error", out / "02_tick_calibration.png")

    # --- Difficulty ---
    diff_rows = []
    for d in range(3):
        dm = difficulty == d
        for t in range(T):
            diff_rows.append({
                "difficulty": DIFF_NAMES[d], "tick": t + 1, "n": int(dm.sum()),
                "accuracy": float((pred_class[dm, t] == y[dm]).mean()),
                "certainty": float(certainty[dm, t].mean()),
            })
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(out / "difficulty_by_tick.csv", index=False)
    settle_rows = []
    for d in range(3):
        dm = difficulty == d
        settle_rows.append({
            "difficulty": DIFF_NAMES[d], "n": int(dm.sum()),
            "mean_settle_tick_all": float(settle[dm].mean()),
            "mean_final_certainty": float(certainty[dm, -1].mean()),
            **{f"n_{CLASS_NAMES[c]}": int(((y == c) & dm).sum()) for c in range(3)},
        })
    pd.DataFrame(settle_rows).to_csv(out / "difficulty_summary.csv", index=False)

    # --- Early exit ---
    early_rows = []
    for thr in np.arange(0.50, 0.991, 0.05):
        chosen = np.full(len(y), T - 1, dtype=np.int64)
        hit = certainty >= thr
        for i in range(len(y)):
            where = np.flatnonzero(hit[i])
            if len(where): chosen[i] = where[0]
        ep = pred_class[np.arange(len(y)), chosen]
        early_rows.append({"threshold": float(thr), "accuracy": float((ep == y).mean()), "macro_f1": float(f1_score(y, ep, average="macro")), "mean_ticks": float((chosen + 1).mean())})
    early_df = pd.DataFrame(early_rows)
    early_df.to_csv(out / "early_exit.csv", index=False)

    # --- Population attention ---
    time_attn = time_attention_sum / max(total_count, 1)
    body_attn = body_attention_sum / max(total_count, 1)
    attn_rows = []
    unique_times = np.unique(token_time_np)
    for t in range(T):
        for j, physical_idx in enumerate(unique_times):
            attn_rows.append({"ctm_tick": t + 1, "physical_step": int(physical_idx), "mean_attention_mass": float(time_attn[t, j])})
    pd.DataFrame(attn_rows).to_csv(out / "attention_time_population.csv", index=False)
    thirds = np.array_split(np.arange(len(unique_times)), 3)
    tick_temporal_rows = []
    for t in range(T):
        tick_temporal_rows.append({
            "ctm_tick": t + 1,
            **{f"{name}_attention_mass": float(time_attn[t, slots].sum()) for name, slots in zip(("early", "middle", "late"), thirds)},
            "mean_temporal_entropy": float(temporal_entropy_sum[t] / total_count),
            "mean_consecutive_tick_cosine": float(temporal_cosine_sum[t] / total_count) if t > 0 else np.nan,
        })
    pd.DataFrame(tick_temporal_rows).to_csv(out / "attention_temporal_tick_summary.csv", index=False)
    aggregate_time = time_attn.mean(0)
    aggregate_time /= max(aggregate_time.sum(), 1e-12)
    uniform = 1.0 / len(aggregate_time)
    save_json({
        "n_encoded_time_steps": len(unique_times),
        "physical_steps": unique_times.tolist(),
        "attention_mass_by_physical_step": aggregate_time.tolist(),
        "early_middle_late_mass": {name: float(aggregate_time[slots].sum()) for name, slots in zip(("early", "middle", "late"), thirds)},
        "aggregate_temporal_entropy": float(-(aggregate_time * np.log(np.maximum(aggregate_time, 1e-12))).sum()),
        "aggregate_kl_uniform_to_attention": float((uniform * (np.log(uniform) - np.log(np.maximum(aggregate_time, 1e-12)))).sum()),
    }, out / "attention_temporal_coverage.json")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(time_attn, aspect="auto", origin="lower")
    ax.set_xlabel("encoded physical-time index")
    ax.set_ylabel("CTM tick")
    ax.set_title("Population attention over physical time")
    fig.colorbar(im, ax=ax, label="attention mass")
    fig.tight_layout(); fig.savefig(out / "03_attention_time_heatmap.png", dpi=180); plt.close(fig)

    pd.DataFrame(body_attn, index=np.arange(1, T + 1), columns=[f"body_{i}" for i in range(body_attn.shape[1])]).to_csv(out / "attention_body_population.csv", index_label="ctm_tick")
    class_rows = []
    for c in range(3):
        denom = max(class_count[c], 1)
        arr = class_time_sum[c] / denom
        for t in range(T):
            for j, physical_idx in enumerate(unique_times):
                class_rows.append({"class": CLASS_NAMES[c], "ctm_tick": t + 1, "physical_step": int(physical_idx), "attention_mass": float(arr[t, j])})
    pd.DataFrame(class_rows).to_csv(out / "attention_time_by_class.csv", index=False)
    event_rows = []
    for c in (1, 2):
        if event_body_count[c] > 0:
            vals = event_body_mass_sum[c] / event_body_count[c]
            for t in range(T):
                event_rows.append({"class": CLASS_NAMES[c], "ctm_tick": t + 1, "mean_attention_on_event_bodies": float(vals[t]), "uniform_attention_baseline": float((2 if c == 1 else 1) / int(cfg["data"]["n_bodies"])), "n": int(event_body_count[c])})
    pd.DataFrame(event_rows).to_csv(out / "attention_event_bodies.csv", index=False)

    # --- Neuron dynamics / selectivity / similarity ---
    post = np.concatenate(internal_post) if internal_post else np.empty((0, T, 0))
    int_y = np.concatenate(internal_labels) if internal_labels else np.empty(0, dtype=int)
    sync_out = np.concatenate(internal_sync_out) if internal_sync_out else np.empty((0, T, 0))
    neuron_rows = []
    if len(post):
        D = post.shape[-1]
        overall = post.mean(axis=(0, 1))
        for d in range(D):
            # eta^2 class selectivity at every tick; keep maximum and where it occurs.
            etas = []
            for t in range(T):
                x = post[:, t, d]
                mu = x.mean()
                ss_total = ((x - mu) ** 2).sum() + 1e-12
                ss_between = 0.0
                for c in range(3):
                    xc = x[int_y == c]
                    if len(xc): ss_between += len(xc) * (xc.mean() - mu) ** 2
                etas.append(ss_between / ss_total)
            neuron_rows.append({"neuron": d, "temporal_variance": float(post[:, :, d].var()), "max_class_eta2": float(np.max(etas)), "max_eta2_tick": int(np.argmax(etas) + 1)})
        neuron_df = pd.DataFrame(neuron_rows).sort_values("max_class_eta2", ascending=False)
        neuron_df.to_csv(out / "neuron_selectivity.csv", index=False)
        top = neuron_df.head(min(8, len(neuron_df)))["neuron"].to_numpy()
        fig, ax = plt.subplots(figsize=(9, 5))
        for d in top:
            ax.plot(np.arange(1, T + 1), post[:, :, d].mean(0), marker="o", label=f"n{d}")
        ax.set_xlabel("CTM tick"); ax.set_ylabel("mean activation"); ax.set_title("Top class-selective neuron trajectories"); ax.legend(ncol=2)
        fig.tight_layout(); fig.savefig(out / "04_neuron_trajectories.png", dpi=180); plt.close(fig)
        flat = post.transpose(2, 0, 1).reshape(D, -1)
        corr = np.corrcoef(flat)
        np.save(out / "neuron_correlation.npy", corr)
        fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto"); ax.set_title("Neuron activity correlation"); fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / "05_neuron_similarity.png", dpi=180); plt.close(fig)

        # Representation class separation across internal time.
        rep_rows = []
        max_sil = min(len(post), int(ana_cfg.get("silhouette_examples", 1200)))
        for t in range(T):
            z = post[:max_sil, t]
            yy = int_y[:max_sil]
            sil = silhouette_score(z, yy) if len(np.unique(yy)) > 1 and len(z) > 3 else np.nan
            rep_rows.append({"tick": t + 1, "silhouette_by_outcome": float(sil)})
        rep_df = pd.DataFrame(rep_rows); rep_df.to_csv(out / "representation_separation.csv", index=False)
        _save_line(rep_df, "tick", ["silhouette_by_outcome"], "Outcome separation in CTM state space", "silhouette", out / "06_representation_separation.png")

        # Internal synchronization magnitude.
        sync_rows = []
        for t in range(T):
            sync_rows.append({"tick": t + 1, "mean_abs_output_sync": float(np.abs(sync_out[:, t]).mean())})
        sync_df = pd.DataFrame(sync_rows); sync_df.to_csv(out / "synchronization_by_tick.csv", index=False)
        _save_line(sync_df, "tick", ["mean_abs_output_sync"], "Internal output synchronization", "mean |sync|", out / "07_synchronization.png")

    # --- Counterfactual twin trajectories ---
    pair_rows = []
    pair_sims = []
    for pid, items in pair_post.items():
        if len(items) != 2 or items[0][1] == items[1][1]:
            continue
        _, y0, z0, l0 = items[0]
        _, y1, z1, l1 = items[1]
        num = (z0 * z1).sum(-1)
        den = np.linalg.norm(z0, axis=-1) * np.linalg.norm(z1, axis=-1) + 1e-9
        sim = num / den
        pair_sims.append(sim)
        p0 = l0.argmax(0); p1 = l1.argmax(0)
        for t in range(T):
            pair_rows.append({"pair_id": pid, "tick": t + 1, "label_a": y0, "label_b": y1, "state_cosine_similarity": float(sim[t]), "predictions_differ": bool(p0[t] != p1[t])})
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(out / "counterfactual_pair_dynamics.csv", index=False)
    if pair_sims:
        mean_sim = np.stack(pair_sims).mean(0)
        fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(np.arange(1, T + 1), mean_sim, marker="o"); ax.set_xlabel("CTM tick"); ax.set_ylabel("mean cosine similarity"); ax.set_title("Internal divergence of counterfactual twins"); fig.tight_layout(); fig.savefig(out / "08_counterfactual_divergence.png", dpi=180); plt.close(fig)

    summary = {
        "n_test": int(len(y)),
        "checkpoint": str(checkpoint),
        "final_metrics": classification_metrics(logits[..., -1], y),
        "most_certain_metrics": classification_metrics(logits[np.arange(len(y)), :, certainty.argmax(1)], y),
        "mean_settle_tick": float(settle.mean()),
        "counterfactual_pairs_analyzed": int(len(pair_sims)),
        "attention_event_body_counts": {CLASS_NAMES[c]: int(event_body_count[c]) for c in (1, 2)},
        "notes": [
            "Attention recorded at thought step t is upstream of the synapse/NLM update and therefore feeds the prediction stored for that same loop step; it should not be interpreted as explaining the pre-update state that entered the step.",
            "Outcome margin is a simulator-derived heuristic; difficulty bins are class-conditional to avoid class confounding.",
            "Counterfactual twins differ by a small initial velocity perturbation and have different simulated futures; they are intended for internal-dynamics analysis, not as a causal proof about neurons.",
        ],
    }
    save_json(summary, out / "summary.json")
    np.savez_compressed(out / "analysis_predictions.npz", index=indices, label=y, logits=logits, certainty=certainty, difficulty=difficulty, settle_tick=settle, pair_id=pair_ids)
    return summary
