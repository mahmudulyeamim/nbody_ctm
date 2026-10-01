# Dataset Design

## Per-sample input

For `N` bodies and `T_obs` observed physical steps:

```text
state: [T_obs, N, 4]
channels = [x, y, vx, vy]
mass:  [N]
label: scalar in {0,1,2}
```

The classifier does **not** see proposal type, future event time, event body identity, virial-ratio target, or hidden-future measurements. Those are metadata used only for diagnostics/analysis.

## Physics

The simulator uses dimensionless Newtonian gravity in 2-D:

`d²r_i/dt² = G Σ_{j != i} m_j (r_j-r_i) / (||r_j-r_i||² + eps²)^(3/2)`

Integration is velocity Verlet with Plummer softening `eps`.

## Observation vs hidden future

The complete simulator runs for:

`T_obs + T_hidden`

but only the first `T_obs` snapshots are saved as model input.

Candidates that already satisfy collision or escape criteria during the observation window are rejected. A guard interval after observation also prevents “the event occurs one integration step later” from dominating the task.

## Labels

### 0 — BOUND

No robust collision/escape event occurs in the hidden horizon and the trajectory remains comfortably inside event boundaries.

This is **finite-horizon operational stability**, not a proof of eternal boundedness.

### 1 — COLLISION

After the observation+guard window, any pair satisfies:

`distance(i,j) <= collision_radius`

The two event-body IDs are stored as metadata for attention analysis.

### 2 — ESCAPE

A body must simultaneously be:

- farther than `escape_radius` from the center of mass;
- outward-moving above `escape_radial_velocity`;
- have approximate positive specific energy above `escape_energy_margin`;
- satisfy the condition for `escape_persistence` consecutive steps.

The escaping body ID is stored as metadata.

## Proposal mixture

Initial-condition proposals are used only to improve coverage:

1. virialized chaotic cluster;
2. cold/close configuration favoring close encounters;
3. hot-kick configuration favoring slingshot/ejection;
4. binary-intruder configuration.

**Proposal type never determines the label.** Every label is obtained from simulation.

Body identities are randomly permuted after proposal construction.

## Counterfactual twins

For a fraction of the dataset, one base initial state is duplicated. A random body receives tiny opposite velocity perturbations:

`v_i+ = v_i + epsilon * direction`

`v_i- = v_i - epsilon * direction`

The pair is retained only when both trajectories are valid and their simulated outcome labels differ.

Both members always remain in the same split. `pair_id` enables later internal-state comparison.

## Difficulty

`outcome_margin` measures how deeply an outcome crosses or avoids simulator event boundaries. It is **heuristic**.

During analysis, easy/medium/hard tertiles are built *separately within each class* and then combined. This avoids the class-confounding issue that occurs when one outcome naturally has smaller margins than another.

## Shortcut audit

Run:

```bash
python inspect_data.py --data data/nbody_quick.npz
```

It tests simple logistic/random-forest classifiers on:

- only the final observed physical state;
- a small set of handcrafted trajectory summaries.

If these become near-perfect, the dataset is not appropriate for studying rich internal computation.
