# Experimental Runbook

## Phase A — local validation

### 1. Environment

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
pytest
```

### 2. Quick dataset

```bash
python generate_data.py --config configs/quick.yaml
python inspect_data.py --data data/nbody_quick.npz
```

Stop if:

- any NaN/inf appears;
- class balance is wrong;
- last-frame/summary shortcut accuracy becomes extremely high;
- counterfactual generation silently fails;
- event times are almost all immediately after the observation window.

### 3. Quick CTM

```bash
python train.py --config configs/quick.yaml --model ctm --seed 42 --out_dir outputs/quick/ctm_seed42
python analyze.py --config configs/quick.yaml --checkpoint outputs/quick/ctm_seed42/best_final.pt --out_dir outputs/quick/ctm_seed42/analysis
```

Do not expect the 600-example quick run to be accurate. Its purpose is to verify learning signal and analysis behavior.

### 4. Optional diagnostic baselines

```bash
python train.py --config configs/quick.yaml --model graph_pool --seed 42 --out_dir outputs/quick/graph_pool_seed42
python train.py --config configs/quick.yaml --model graph_gru  --seed 42 --out_dir outputs/quick/graph_gru_seed42
python train.py --config configs/quick.yaml --model linear_ctm --seed 42 --out_dir outputs/quick/linear_ctm_seed42
```

## Phase B — freeze the data-generating process

Before main generation, write down:

- exact config hash / copy of `main.yaml`;
- outcome definitions;
- observation/hidden horizons;
- target counterfactual fraction;
- primary CTM checkpoint-selection criterion (final-tick validation macro-F1);
- planned seeds.

After this point, avoid changing simulator thresholds based on main-test results.

## Phase C — main dataset

```bash
python generate_data.py --config configs/main.yaml
python inspect_data.py --data data/nbody_main.npz
```

Generation can be done on GPU (`--device cuda`, default auto) and is batched.

## Phase D — main CTM seeds

Recommended:

```bash
for s in 42 123 2026; do
  python train.py --config configs/main.yaml --model ctm --seed $s --out_dir outputs/main/ctm_seed${s}
done
```

Then analyze each best-final checkpoint:

```bash
for s in 42 123 2026; do
  python analyze.py --config configs/main.yaml \
    --checkpoint outputs/main/ctm_seed${s}/best_final.pt \
    --out_dir outputs/main/ctm_seed${s}/analysis
done
```

## Phase E — secondary sanity baselines

Only if needed for the course write-up:

```bash
python train.py --config configs/main.yaml --model graph_pool --seed 42 --out_dir outputs/main/graph_pool_seed42
python train.py --config configs/main.yaml --model graph_gru --seed 42 --out_dir outputs/main/graph_gru_seed42
```

## Phase F — Colab reproducibility

Use `NBody_CTM_Assignment.ipynb` on a fresh Colab GPU session.

Recommended submission-scale check:

1. run tests;
2. generate the quick dataset;
3. train quick CTM;
4. run quick analysis;
5. verify peak VRAM;
6. if course submission requires full training, use `main.yaml` but lower `simulation_batch_size` or training batch size if the assigned Colab GPU is smaller.

The main model was intentionally designed to remain modest. Batch size is the first knob to reduce for VRAM.
