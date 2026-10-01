#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from nbody_ctm.config import load_config
from nbody_ctm.trainer import train_model
from nbody_ctm.utils import choose_device


def main():
    p = argparse.ArgumentParser(description="Train CTM or a compact baseline on N-body outcomes.")
    p.add_argument("--config", default="configs/quick.yaml")
    p.add_argument("--data", default=None)
    p.add_argument("--model", default="ctm", choices=["ctm", "linear_ctm", "graph_pool", "graph_gru"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    cfg = load_config(args.config)
    data_path = args.data or cfg["data"]["output_path"]
    out_dir = args.out_dir or f"outputs/{Path(args.config).stem}/{args.model}_seed{args.seed}"
    device = choose_device(args.device)
    print(f"Training {args.model} on {device}. Data={data_path}")
    metrics = train_model(cfg, data_path=data_path, model_name=args.model, out_dir=out_dir, seed=args.seed, device=device)
    print(metrics)


if __name__ == "__main__":
    main()
