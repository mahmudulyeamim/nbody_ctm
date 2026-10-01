from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import make_datasets
from .losses import ctm_classification_loss, temporal_attention_regularizers
from .metrics import classification_metrics
from .models import build_model
from .utils import count_parameters, ensure_dir, save_json, seed_everything


def _move(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _make_loaders(data_path: str, cfg: Dict, seed: int):
    train, val, test, stats = make_datasets(data_path, {**cfg["data"], "seed": seed})
    tr = cfg["training"]
    common = dict(num_workers=int(tr.get("num_workers", 0)), pin_memory=torch.cuda.is_available())
    g = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(train, batch_size=int(tr["batch_size"]), shuffle=True, generator=g, drop_last=False, **common),
        "val": DataLoader(val, batch_size=int(tr.get("eval_batch_size", tr["batch_size"])), shuffle=False, **common),
        "test": DataLoader(test, batch_size=int(tr.get("eval_batch_size", tr["batch_size"])), shuffle=False, **common),
    }
    return loaders, stats


def _ctm_eval(model, loader, device, amp_dtype=None):
    model.eval()
    ys, logits_all, cert_all, idx_all = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = _move(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None and device.type == "cuda")):
                pred, cert, _ = model(batch["state"], batch["mass"])
            ys.append(batch["label"].cpu().numpy())
            logits_all.append(pred.float().cpu().numpy())
            cert_all.append(cert.float().cpu().numpy())
            idx_all.append(batch["index"].cpu().numpy())
    y = np.concatenate(ys)
    logits = np.concatenate(logits_all)       # [N,C,T]
    cert = np.concatenate(cert_all)           # [N,T]
    idx = np.concatenate(idx_all)
    final = classification_metrics(logits[..., -1], y)
    mc_tick = cert.argmax(1)
    mc_logits = logits[np.arange(len(y)), :, mc_tick]
    most = classification_metrics(mc_logits, y)
    per_tick = [classification_metrics(logits[..., t], y) for t in range(logits.shape[-1])]
    return {"final": final, "most_certain": most, "per_tick": per_tick}, (idx, y, logits, cert)


def _baseline_eval(model, loader, device, amp_dtype=None):
    model.eval()
    ys, logits_all, idx_all = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = _move(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None and device.type == "cuda")):
                logits = model(batch["state"], batch["mass"])
            ys.append(batch["label"].cpu().numpy())
            logits_all.append(logits.float().cpu().numpy())
            idx_all.append(batch["index"].cpu().numpy())
    y = np.concatenate(ys)
    logits = np.concatenate(logits_all)
    idx = np.concatenate(idx_all)
    return classification_metrics(logits, y), (idx, y, logits)


def train_model(cfg: Dict, *, data_path: str, model_name: str, out_dir: str, seed: int, device: torch.device):
    seed_everything(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    out = ensure_dir(out_dir)
    loaders, norm_stats = _make_loaders(data_path, cfg, seed)
    model = build_model(cfg, model_name).to(device)
    # Initialize lazy modules before parameter count / optimizer by one dry forward.
    first = next(iter(loaders["train"]))
    first = _move(first, device)
    with torch.no_grad():
        if model_name in {"ctm", "linear_ctm"}:
            model(first["state"], first["mass"])
        else:
            model(first["state"], first["mass"])

    tr = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(tr["epochs"])), eta_min=float(tr.get("min_lr", 1e-6)))
    use_amp = bool(tr.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if bool(tr.get("bf16", True)) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    run_info = {"seed": seed, "model_name": model_name, "parameters": count_parameters(model),
                "normalization": norm_stats.as_dict(), "data_path": str(Path(data_path).resolve())}
    generation_report = Path(data_path).with_suffix(".generation.json")
    if generation_report.exists():
        run_info["dataset_generation"] = json.loads(generation_report.read_text(encoding="utf-8"))
    save_json(run_info, out / "run_info.json")
    history = []
    best_final = -1.0
    best_mc = -1.0
    patience = 0
    max_patience = int(tr.get("early_stopping_patience", 6))
    is_ctm = model_name in {"ctm", "linear_ctm"}
    reg_cfg = tr.get("attention_regularization", {})
    lambda_cov = float(reg_cfg.get("lambda_cov", 0.0))
    lambda_div = float(reg_cfg.get("lambda_div", 0.0))
    if lambda_cov < 0 or lambda_div < 0:
        raise ValueError("Attention regularization weights must be nonnegative.")
    use_attention_reg = lambda_cov > 0 or lambda_div > 0
    if use_attention_reg and not is_ctm:
        raise ValueError("Attention regularization requires a CTM model.")

    epoch_bar = tqdm(range(1, int(tr["epochs"]) + 1), desc=f"{model_name} seed={seed} epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        loss_sum, seen = 0.0, 0
        coverage_sum, redundancy_sum = 0.0, 0.0
        tic = time.time()
        pbar = tqdm(loaders["train"], desc=f"{model_name} seed={seed} epoch={epoch}", leave=False)
        for batch in pbar:
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if is_ctm:
                    pred, cert, tracking = model(batch["state"], batch["mass"], track=use_attention_reg)
                    loss, pieces = ctm_classification_loss(
                        pred, cert, batch["label"], final_ce_weight=float(cfg["model"]["ctm"].get("final_ce_weight", 0.5))
                    )
                    if use_attention_reg:
                        coverage, redundancy = temporal_attention_regularizers(
                            tracking["attention"], tracking["token_time_index"]
                        )
                        loss = loss + lambda_cov * coverage + lambda_div * redundancy
                else:
                    logits = model(batch["state"], batch["mass"])
                    loss = F.cross_entropy(logits, batch["label"])
            scaler.scale(loss).backward()
            if float(tr.get("grad_clip", 1.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            bs = batch["label"].size(0)
            loss_sum += float(loss.item()) * bs
            if use_attention_reg:
                coverage_sum += float(coverage.detach().item()) * bs
                redundancy_sum += float(redundancy.detach().item()) * bs
            seen += bs
            pbar.set_postfix(loss=f"{loss_sum/max(seen,1):.4f}")
        scheduler.step()

        if is_ctm:
            val_metrics, _ = _ctm_eval(model, loaders["val"], device, amp_dtype if use_amp else None)
            final_f1 = val_metrics["final"]["macro_f1"]
            mc_f1 = val_metrics["most_certain"]["macro_f1"]
            row = {
                "epoch": epoch, "train_loss": loss_sum / seen,
                "val_final_macro_f1": final_f1, "val_final_acc": val_metrics["final"]["accuracy"],
                "val_most_certain_macro_f1": mc_f1, "val_most_certain_acc": val_metrics["most_certain"]["accuracy"],
                "seconds": time.time() - tic,
            }
            if use_attention_reg:
                row["train_coverage_loss"] = coverage_sum / seen
                row["train_redundancy_loss"] = redundancy_sum / seen
            improved = False
            if final_f1 > best_final:
                best_final = final_f1
                torch.save({"model": model.state_dict(), "seed": seed, "epoch": epoch, "cfg": cfg, "model_name": model_name}, out / "best_final.pt")
                improved = True
            if mc_f1 > best_mc:
                best_mc = mc_f1
                torch.save({"model": model.state_dict(), "seed": seed, "epoch": epoch, "cfg": cfg, "model_name": model_name}, out / "best_most_certain.pt")
        else:
            val_metrics, _ = _baseline_eval(model, loaders["val"], device, amp_dtype if use_amp else None)
            final_f1 = val_metrics["macro_f1"]
            row = {"epoch": epoch, "train_loss": loss_sum / seen, "val_macro_f1": final_f1, "val_acc": val_metrics["accuracy"], "seconds": time.time() - tic}
            improved = final_f1 > best_final
            if improved:
                best_final = final_f1
                torch.save({"model": model.state_dict(), "seed": seed, "epoch": epoch, "cfg": cfg, "model_name": model_name}, out / "best_final.pt")
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        epoch_bar.set_postfix(train_loss=f"{row['train_loss']:.4f}", val_f1=f"{final_f1:.4f}")
        patience = 0 if improved else patience + 1
        if patience >= max_patience:
            break

    # Always evaluate the best-final checkpoint because final tick is the primary readout.
    ckpt = torch.load(out / "best_final.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if is_ctm:
        test_metrics, arrays = _ctm_eval(model, loaders["test"], device, amp_dtype if use_amp else None)
        idx, y, logits, cert = arrays
        pair_ids = loaders["test"].dataset.meta.get("pair_id")
        np.savez_compressed(out / "test_predictions.npz", index=idx, label=y, logits=logits,
                            certainty=cert, pair_id=pair_ids[idx] if pair_ids is not None else np.full_like(idx, -1))
    else:
        test_metrics, arrays = _baseline_eval(model, loaders["test"], device, amp_dtype if use_amp else None)
        idx, y, logits = arrays
        np.savez_compressed(out / "test_predictions.npz", index=idx, label=y, logits=logits)
    save_json(test_metrics, out / "test_metrics.json")
    if device.type == "cuda":
        run_info["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
        save_json(run_info, out / "run_info.json")
    return test_metrics
