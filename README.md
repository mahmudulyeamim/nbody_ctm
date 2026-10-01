# Chaotic N-Body CTM — Applied ML Assignment Codebase

This project is a **controlled internal-dynamics study of a Continuous Thought Machine (CTM)** on a procedurally generated chaotic gravitational N-body task.

The project is **not designed primarily to beat baselines**. Its main purpose is to ask:

> **How does a CTM's internal computation evolve when the future of a nonlinear chaotic physical system is genuinely ambiguous?**

The model observes only a short early trajectory of a 2-D gravitational system and predicts one of three much-later outcomes:

1. **BOUND** — no collision or escape occurs within the hidden simulation horizon and the system remains comfortably within event boundaries.
2. **COLLISION** — a pair of bodies crosses the collision radius after the observation window.
3. **ESCAPE** — a body becomes sufficiently distant, outward-moving, and energetically unbound after the observation window.

The dataset includes an intentionally difficult subset of **counterfactual twins**: nearly identical initial states with a tiny opposite velocity perturbation but different eventual outcomes. These pairs are especially useful for studying when two initially similar CTM state trajectories diverge.

---

## 1. What is included

```text
nbody_ctm/
├── README.md
├── QUICKSTART.md
├── RUNBOOK.md
├── DATASET.md
├── ARCHITECTURE.md
├── ANALYSIS_GUIDE.md
├── CODEX_HANDOFF.md
├── UPSTREAM_CTM_NOTES.md
├── requirements.txt
├── pyproject.toml
├── generate_data.py
├── inspect_data.py
├── train.py
├── analyze.py
├── NBody_CTM_Assignment.ipynb
├── configs/
│   ├── smoke.yaml
│   ├── quick.yaml
│   └── main.yaml
├── scripts/
│   ├── run_smoke.sh
│   ├── run_quick.sh
│   └── run_main.sh
├── src/nbody_ctm/
│   ├── physics.py
│   ├── generator.py
│   ├── dataset.py
│   ├── graph_encoder.py
│   ├── ctm.py
│   ├── losses.py
│   ├── models.py
│   ├── trainer.py
│   ├── metrics.py
│   ├── analysis.py
│   └── utils.py
└── tests/
```

---

## 2. The task in one picture

```text
EARLY PHYSICAL TIME                         HIDDEN FUTURE

body trajectories                          simulator continues
(x,y,vx,vy)                                but model cannot see it
      │
      ▼
┌─────────────────────────────┐            ┌────────────────────┐
│ Graph + temporal encoder    │            │ possible outcomes  │
│ local motion + interactions │            │                    │
└──────────────┬──────────────┘            │  BOUND             │
               │ tokens                    │  COLLISION         │
               ▼                           │  ESCAPE            │
┌─────────────────────────────┐            └────────────────────┘
│ CTM internal thought ticks  │                      ▲
│                             │                      │ label
│ synchronization -> attention│                      │
│ -> synapse -> private NLMs  │──────────────────────┘
│ -> synchronization -> logits│
└─────────────────────────────┘
```

The **physical time axis** and the **CTM thought-time axis** are distinct. This is central to the analysis.

---

## 3. Why this dataset is deliberately harder than the earlier power-grid pilot

The previous pilot was easy enough that simple models reached >95%. This benchmark is intentionally harder:

- 3 classes instead of binary classification.
- 6–7 mutually interacting bodies.
- nonlinear pairwise gravity and chaotic sensitivity to initial conditions.
- the model sees only a short physical prefix; labels come from a much longer hidden future.
- body order is randomized, and training augmentation can re-permute bodies.
- global rotation augmentation removes fixed coordinate-orientation shortcuts.
- the main dataset deliberately enriches counterfactual near-boundary cases.
- test diagnostics explicitly measure whether last-frame or handcrafted summary shortcuts solve the task.

On the included `quick.yaml` validation dataset, deliberately simple shortcut models were only ~30–54% accurate on a 3-class problem. This is **only a codebase validation observation, not a final experimental result**.

---

## 4. CTM implementation philosophy

The CTM core follows the public Sakana AI implementation's central computation order and mechanisms:

- learned starting post-activation state and pre-activation trace;
- internal recurrent thought ticks independent of physical input time;
- separate synchronization representations for **action/attention** and **output**;
- random neuron pairings with learnable exponential decay;
- synchronization-derived attention query;
- synapse model mixing attended input and current activated state;
- private per-neuron temporal models (NLMs) over pre-activation history;
- synchronization-derived output logits;
- entropy-based certainty at every internal tick.

For this task, the image ResNet is replaced with a **graph-temporal physics encoder**. See `ARCHITECTURE.md` and `UPSTREAM_CTM_NOTES.md`.

The classification loss includes the official-style adaptive objective (minimum-loss tick + most-certain tick) plus a configurable final-tick CE stabilizer. Set `final_ce_weight: 0.0` for the pure adaptive objective.

Official CTM repository: https://github.com/SakanaAI/continuous-thought-machines

---

## 5. Fastest way to start locally

### Environment

```bash
cd nbody_ctm
python -m venv .venv
source .venv/bin/activate           # Linux/macOS
# .venv\\Scripts\\Activate.ps1     # Windows PowerShell
```

Install a CUDA-enabled PyTorch build suitable for your machine **first**. If your current environment already has working CUDA PyTorch, do not replace it unnecessarily.

Then:

```bash
pip install -e .
```

If build isolation tries to download setuptools in an offline environment:

```bash
pip install -e . --no-build-isolation
```

Verify:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

pytest
```

### Run the quick experiment

```bash
python generate_data.py --config configs/quick.yaml
python inspect_data.py --data data/nbody_quick.npz
python train.py --config configs/quick.yaml --model ctm --seed 42 \
  --out_dir outputs/quick/ctm_seed42
python analyze.py --config configs/quick.yaml \
  --checkpoint outputs/quick/ctm_seed42/best_final.pt \
  --out_dir outputs/quick/ctm_seed42/analysis
```

**Do not start with `main.yaml`.** Audit the quick dataset and internal plots first.

---

## 6. Main configuration

`configs/main.yaml` is the intended assignment-scale configuration:

- 7 bodies;
- 48 observed states at `dt=0.009` (~0.43 physical-time units);
- 180 hidden simulation steps (~1.62 additional units);
- 30,000 train / 4,500 validation / 9,000 test;
- exactly balanced 3 classes;
- 25% target counterfactual-twin samples;
- graph-temporal encoder;
- 192 CTM neurons;
- 14 internal thought ticks;
- 320 output sync pairs + 320 action sync pairs;
- private NLM memory length 10;
- ~1.09M trainable parameters after lazy modules initialize.

This remains small in modern terms. The design goal is **richer physical/architectural structure**, not arbitrary VRAM consumption.

---

## 7. Analyses generated automatically

`analyze.py` produces:

- accuracy / macro-F1 / AUC / calibration / certainty by thought tick;
- class prediction proportions by tick;
- class-conditional easy/medium/hard analysis;
- prediction settle tick;
- certainty-based early exit;
- population attention over physical time;
- population attention by outcome class;
- attention mass on the bodies that later collide or escape;
- neuron temporal variance and class selectivity;
- neuron-neuron activity correlation matrix;
- representation separation across CTM ticks (silhouette score);
- internal synchronization magnitude over thought time;
- counterfactual-twin CTM-state cosine similarity over thought ticks.

See `ANALYSIS_GUIDE.md` before interpreting any of these as mechanistic explanations.

---

## 8. Baselines are secondary

The code includes:

- `graph_pool`: the same graph-temporal encoder + invariant pooling + classifier;
- `graph_gru`: the same encoder + physical-time GRU;
- `linear_ctm`: CTM with only a per-token MLP encoder (feature-extractor ablation).

These are sanity checks. The assignment's main question is the CTM's internal dynamics.

---

## 9. Important scientific cautions

1. **Synthetic labels are operational definitions.** Collision/escape/bound are defined by explicit simulator thresholds and a finite horizon.
2. **BOUND means no robust event within the simulated future horizon**, not mathematical eternal stability.
3. **Outcome margin is a heuristic difficulty measure.** Difficulty bins are built separately inside each class to reduce class confounding.
4. **Attention is not causal explanation.** Event-body attention is descriptive unless supported by perturbation/ablation tests.
5. **Neuron selectivity is association, not semantic proof.** A neuron correlated with collision outcomes is not automatically a “collision neuron.”
6. **Counterfactual twins are especially useful but still simulator counterfactuals.** Internal divergence can be studied without claiming biological-style reasoning.
7. Train/val/test splits are generated independently; counterfactual twins are never split across partitions.

---

## 10. Recommended workflow

1. `pytest`
2. `configs/smoke.yaml` only if debugging code.
3. Generate `configs/quick.yaml`.
4. Run `inspect_data.py` and reject the dataset if shortcuts are too strong.
5. Train quick CTM.
6. Inspect internal dynamics.
7. Only then generate `main.yaml`.
8. Freeze the generator/config before looking at main-test results.
9. Train 3 seeds if time permits.
10. Run the notebook once in Colab before submission to verify reproducibility and VRAM.

Read `RUNBOOK.md` for the exact sequence.
