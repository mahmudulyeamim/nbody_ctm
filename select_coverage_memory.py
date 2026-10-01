#!/usr/bin/env python
"""Select a coverage-memory sweep run using validation metrics only."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from nbody_ctm.config import load_config
from nbody_ctm.utils import ensure_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("coverage_memory", "coverage_memory_diverse"))
    args = parser.parse_args()
    variant = args.variant
    candidates = []
    for rho in (0.8, 1.0):
        for beta in (0.1, 0.3, 0.5):
            tag = f"rho{str(rho).replace('.', 'p')}_beta{str(beta).replace('.', 'p')}"
            config = Path(f"configs/coverage_memory/{variant}_{tag}.yaml")
            run = Path(f"outputs/coverage_memory_sweep/{variant}/{tag}/ctm_seed42")
            if not all((run / name).is_file() for name in ("history.csv", "best_final.pt", "test_metrics.json", "test_predictions.npz")):
                raise FileNotFoundError(f"Incomplete run: {run}")
            cfg = load_config(config)
            history = pd.read_csv(run / "history.csv")
            # The trainer saves best_final.pt only when this validation F1 improves.
            best = history.loc[history["val_final_macro_f1"].idxmax()]
            candidates.append({
                "rho": rho, "beta": beta, "tag": tag, "config": str(config), "run": str(run),
                "best_epoch": int(best["epoch"]),
                "validation_final_macro_f1": float(best["val_final_macro_f1"]),
                "validation_final_accuracy_at_best_f1": float(best["val_final_acc"]),
                "lambda_cov": float(cfg["training"].get("attention_regularization", {}).get("lambda_cov", 0.0)),
                "lambda_div": float(cfg["training"].get("attention_regularization", {}).get("lambda_div", 0.0)),
            })
    # Accuracy is the stated primary objective. Each candidate's checkpoint
    # remains the existing best-final (validation macro-F1) checkpoint.
    selected = max(candidates, key=lambda c: (c["validation_final_accuracy_at_best_f1"], c["validation_final_macro_f1"]))
    output = ensure_dir(Path("outputs") / variant / "ctm_seed42")
    for filename in ("best_final.pt", "best_most_certain.pt", "history.csv", "run_info.json", "test_metrics.json", "test_predictions.npz"):
        shutil.copy2(Path(selected["run"]) / filename, output / filename)
    shutil.copy2(selected["config"], output / "config.yaml")
    save_json({
        "selection_rule": "highest validation final-tick accuracy at each run's best_final checkpoint; validation macro-F1 breaks ties",
        "test_metrics_used_for_selection": False,
        "selected": selected,
        "candidates": candidates,
    }, output / "selection.json")
    print(f"Selected {variant}: {selected['tag']} at validation accuracy {selected['validation_final_accuracy_at_best_f1']:.6f}")
    print(f"Copied selected checkpoint and predictions to {output}")


if __name__ == "__main__":
    main()
