# CTM Internal-Dynamics Analysis Guide

The purpose of this project is **not** primarily leaderboard performance. The main object of study is the trajectory of computation inside CTM.

## RQ1 — How does a decision emerge across thought ticks?

Use `per_tick_metrics.csv` and plots 01/02.

Inspect:

- accuracy / macro-F1;
- multiclass one-vs-rest AUC;
- certainty;
- ECE and Brier score;
- predicted-class proportions.

A useful pattern is when discriminative information appears before well-calibrated decisions. Do not assume later ticks are always better.

## RQ2 — Do harder physical cases change internal computation?

`difficulty_by_tick.csv` and `difficulty_summary.csv` use class-conditional outcome-margin tertiles.

Inspect:

- final certainty;
- prediction settle tick;
- accuracy trajectory;
- whether hard examples continue changing predictions later.

The margin is heuristic. Phrase conclusions as associations with the simulator-derived difficulty proxy.

## RQ3 — Where does CTM look while thinking?

`attention_time_population.csv` maps attention back to physical time.

`attention_time_by_class.csv` stratifies by eventual outcome.

`attention_event_bodies.csv` asks a particularly interpretable question:

- for collision examples, how much attention falls on the two bodies that later collide?
- for escape examples, how much attention falls on the body that later escapes?

Remember the temporal ordering: **attention at CTM tick t feeds the synapse/NLM update and same-step post-update prediction**. It should not be described as causing the already-produced state at t.

Attention is descriptive allocation, not causal importance. A stronger follow-up is to mask high-attention bodies/times and re-evaluate.

## RQ4 — Do neurons acquire different computational roles?

`neuron_selectivity.csv` reports:

- temporal variance;
- maximum class-selectivity eta-squared;
- tick of maximum selectivity.

`04_neuron_trajectories.png` shows high-selectivity neuron trajectories.

`05_neuron_similarity.png` shows pairwise correlation over sample × thought-time activity.

Interesting questions:

- Do neuron groups become active early vs late?
- Do some neurons become outcome-selective only after several ticks?
- Are neurons highly redundant or diverse?
- Does similarity structure change across trained seeds?

Do not call a neuron a “collision neuron” based on one correlation plot.

## RQ5 — Does the CTM representation become more class-structured over thought time?

`representation_separation.csv` tracks silhouette score of the post-activation state by true outcome.

This is a simple population-level measurement of whether internal states become more outcome-separable as computation unfolds.

## RQ6 — How does synchronization itself evolve?

`synchronization_by_tick.csv` tracks mean absolute output-sync magnitude.

Useful extensions for Codex:

- synchronization by class;
- easy vs hard;
- correct vs incorrect;
- before/after prediction-switch events;
- clustering individual synchronization channels.

Do **not** assume CTM neural synchronization should match any physical synchronization measure. The representation is learned and architectural, not a direct physical oscillator model.

## RQ7 — What happens for counterfactual twins?

`counterfactual_pair_dynamics.csv` and `08_counterfactual_divergence.png` are central to the harder dataset.

Each twin pair starts almost identically but a tiny velocity perturbation eventually produces different outcomes.

Measure:

- cosine similarity of CTM post-activation states across thought ticks;
- when predictions first differ;
- when attention begins focusing on different bodies/times;
- whether divergence happens earlier for easier pairs and later for boundary pairs.

This is one of the cleanest ways to study **similar input → initially similar thought → divergent internal computation → different decision**.

## RQ8 — Does the model use all available thought ticks?

`early_exit.csv` applies certainty thresholds.

Report **mean internal ticks**, not “speedup,” unless you also measure wall-clock latency. Feature extraction is a fixed cost outside recurrent CTM ticks.

## Recommended figures for the assignment

If space is limited, prioritize:

1. per-tick accuracy/AUC/certainty/calibration;
2. easy vs hard settle/certainty behavior;
3. attention-over-physical-time heatmap;
4. attention on eventual event bodies;
5. neuron selectivity/similarity;
6. counterfactual-twin internal divergence;
7. early-exit tradeoff.

Baselines can remain one compact sanity-check table.
