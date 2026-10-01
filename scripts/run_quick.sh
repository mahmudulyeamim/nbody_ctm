#!/usr/bin/env bash
set -euo pipefail
python generate_data.py --config configs/quick.yaml
python inspect_data.py --data data/nbody_quick.npz
python train.py --config configs/quick.yaml --model ctm --seed 42 --out_dir outputs/quick/ctm_seed42
python analyze.py --config configs/quick.yaml --checkpoint outputs/quick/ctm_seed42/best_final.pt --out_dir outputs/quick/ctm_seed42/analysis
