#!/usr/bin/env bash
set -euo pipefail
python generate_data.py --config configs/main.yaml
python inspect_data.py --data data/nbody_main.npz
for s in 42 123 2026; do
  python train.py --config configs/main.yaml --model ctm --seed "$s" --out_dir "outputs/main/ctm_seed${s}"
  python analyze.py --config configs/main.yaml --checkpoint "outputs/main/ctm_seed${s}/best_final.pt" --out_dir "outputs/main/ctm_seed${s}/analysis"
done
