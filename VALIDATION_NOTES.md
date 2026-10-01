# Validation notes from codebase construction

These checks are **engineering validation only**, not assignment results.

## Unit tests

`pytest` passes 4 tests covering:

- pairwise acceleration / momentum consistency;
- short-horizon velocity-Verlet energy behavior;
- simulator output finiteness and shape;
- CTM forward shapes, tracking tensors, adaptive loss, and backward gradients.

A numerical bug was caught during development: zero-softening self-distance could produce `inf * 0 -> NaN`. The implementation now masks self-distances before inversion.

## End-to-end smoke run

A 90/30/30 dataset with a one-epoch tiny CTM successfully completed:

`generation -> training -> checkpoint -> internal analysis plots/tables`

Chance-level performance is expected at this scale; the run exists to catch pipeline failures.

## Hard quick-dataset audit

A fresh 600/150/300 quick dataset was generated after final guard-window logic.

- exact class balance: bound/collision/escape;
- requested ~10% counterfactual samples realized;
- all saved inputs finite;
- simple shortcut diagnostics were intentionally far from perfect:
  - handcrafted summary + logistic: ~51%;
  - handcrafted summary + RF: ~54%;
  - last frame + logistic: ~30%;
  - last frame + RF: ~46%.

Chance accuracy is 33.3%.

This suggests the quick task is not reducible to a trivial final-frame threshold, but a real full-size GPU experiment is still required before drawing scientific conclusions.

## Model size

After lazy layers initialize:

- quick CTM configuration: ~214k trainable parameters;
- main CTM configuration: ~1.09M trainable parameters.

Thought-tick count changes recurrent compute but not parameter count.
