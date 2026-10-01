#!/usr/bin/env bash
set -euo pipefail
python generate_data.py --config configs/smoke.yaml --out data/nbody_smoke.npz
python inspect_data.py --data data/nbody_smoke.npz
python train.py --config configs/smoke.yaml --data data/nbody_smoke.npz --model ctm --seed 7 --out_dir outputs/smoke/ctm
python analyze.py --config configs/smoke.yaml --data data/nbody_smoke.npz --checkpoint outputs/smoke/ctm/best_final.pt --out_dir outputs/smoke/ctm/analysis
