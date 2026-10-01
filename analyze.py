#!/usr/bin/env python
from __future__ import annotations

import argparse

from nbody_ctm.analysis import run_analysis
from nbody_ctm.config import load_config
from nbody_ctm.utils import choose_device


def main():
    p = argparse.ArgumentParser(description="Run CTM internal-dynamics analysis.")
    p.add_argument("--config", default="configs/quick.yaml")
    p.add_argument("--data", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    cfg = load_config(args.config)
    summary = run_analysis(
        cfg,
        data_path=args.data or cfg["data"]["output_path"],
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        device=choose_device(args.device),
    )
    print(summary)


if __name__ == "__main__":
    main()
