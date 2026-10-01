#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from nbody_ctm.config import load_config
from nbody_ctm.generator import generate_dataset
from nbody_ctm.utils import choose_device, ensure_dir, save_json, seed_everything


def main():
    p = argparse.ArgumentParser(description="Generate the chaotic N-body outcome dataset.")
    p.add_argument("--config", default="configs/quick.yaml")
    p.add_argument("--out", default=None, help="Override config data.output_path")
    p.add_argument("--device", default="auto", help="auto/cuda/cpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    seed_everything(int(data_cfg.get("seed", 42)))
    device = choose_device(args.device)
    out_path = Path(args.out or data_cfg["output_path"])
    ensure_dir(out_path.parent)
    print(f"Generating on {device} -> {out_path}")
    arrays, report = generate_dataset(data_cfg, device=device)
    np.savez_compressed(out_path, **arrays)
    report_path = out_path.with_suffix(".generation.json")
    save_json(report, report_path)
    print(f"Saved dataset: {out_path}")
    print(f"Saved generation report: {report_path}")
    for split, s in report["splits"].items():
        print(split, s["class_counts"], "simulated=", s["simulated"], "cf=", f"{s['counterfactual_fraction_realized']:.3f}")


if __name__ == "__main__":
    main()
