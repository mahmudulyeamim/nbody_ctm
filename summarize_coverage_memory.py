#!/usr/bin/env python
"""Summarize selected coverage-memory checkpoints against existing models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nbody_ctm.config import load_config
from nbody_ctm.utils import ensure_dir, save_json


MODELS = (
    ("Original", "longer_obs"),
    ("Coverage", "longer_obs_coverage"),
    ("Coverage + diversity", "longer_obs_coverage_diverse"),
    ("Coverage-memory", "coverage_memory"),
    ("Coverage-memory + diversity", "coverage_memory_diverse"),
)


def main() -> None:
    rows = []
    for label, folder in MODELS:
        run = Path("outputs") / folder / "ctm_seed42"
        metrics = json.loads((run / "test_metrics.json").read_text())
        analysis = run / "analysis"
        coverage_path = analysis / "attention_temporal_coverage.json"
        if coverage_path.exists():
            coverage = json.loads(coverage_path.read_text())
            thirds = coverage["early_middle_late_mass"]
            kl = coverage["aggregate_kl_uniform_to_attention"]
        else:
            # Original analysis predates the new temporal-coverage summary.
            time = pd.read_csv(analysis / "attention_time_population.csv")
            matrix = time.pivot(index="ctm_tick", columns="physical_step", values="mean_attention_mass").to_numpy()
            aggregate = matrix.mean(0)
            aggregate /= aggregate.sum()
            parts = np.array_split(np.arange(len(aggregate)), 3)
            thirds = {name: float(aggregate[part].sum()) for name, part in zip(("early", "middle", "late"), parts)}
            uniform = 1 / len(aggregate)
            kl = float((uniform * (np.log(uniform) - np.log(np.maximum(aggregate, 1e-12)))).sum())
        tick_path = analysis / "attention_temporal_tick_summary.csv"
        cosine = float(pd.read_csv(tick_path)["mean_consecutive_tick_cosine"].iloc[1:].mean()) if tick_path.exists() else None
        row = {
            "model": label,
            "final_accuracy": metrics["final"]["accuracy"],
            "final_macro_f1": metrics["final"]["macro_f1"],
            "most_certain_accuracy": metrics["most_certain"]["accuracy"],
            "most_certain_macro_f1": metrics["most_certain"]["macro_f1"],
            "early_attention_mass": thirds["early"],
            "middle_attention_mass": thirds["middle"],
            "late_attention_mass": thirds["late"],
            "coverage_kl_uniform_to_attention": kl,
            "mean_consecutive_tick_cosine": cosine,
        }
        if folder.startswith("coverage_memory"):
            config = load_config(run / "config.yaml")
            selection = json.loads((run / "selection.json").read_text())
            row.update({
                "rho": config["model"]["ctm"]["coverage_memory"]["rho"],
                "beta": config["model"]["ctm"]["coverage_memory"]["beta"],
                "lambda_cov": config["training"].get("attention_regularization", {}).get("lambda_cov", 0.0),
                "lambda_div": config["training"].get("attention_regularization", {}).get("lambda_div", 0.0),
                "selection_validation_macro_f1": selection["selected"]["validation_final_macro_f1"],
            })
        rows.append(row)
    out = ensure_dir("outputs/coverage_memory_comparison")
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    save_json({"models": rows, "shared_config": load_config("configs/longer_obs.yaml")}, out / "comparison.json")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved comparison to {out}")


if __name__ == "__main__":
    main()
