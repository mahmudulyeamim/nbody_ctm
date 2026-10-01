# Quickstart

```bash
cd nbody_ctm
python -m venv .venv
source .venv/bin/activate
# Install the correct CUDA PyTorch for your machine first.
pip install -e .
pytest

python generate_data.py --config configs/quick.yaml
python inspect_data.py --data data/nbody_quick.npz
python train.py --config configs/quick.yaml --model ctm --seed 42 --out_dir outputs/quick/ctm_seed42
python analyze.py --config configs/quick.yaml --checkpoint outputs/quick/ctm_seed42/best_final.pt --out_dir outputs/quick/ctm_seed42/analysis
```

Then inspect:

```text
outputs/quick/ctm_seed42/
├── history.csv
├── test_metrics.json
├── test_predictions.npz
├── best_final.pt
├── best_most_certain.pt
└── analysis/
```

If the quick dataset is too easy according to `inspect_data.py`, **do not scale up**. Shorten the observed physical window, increase `counterfactual_fraction`, or adjust proposal distributions using validation/quick data only.
