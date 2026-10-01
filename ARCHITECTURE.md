# Architecture

## Overview

```text
raw trajectory [B,T,N,4] + masses [B,N]
                  │
                  ▼
      Graph-Temporal Feature Encoder
                  │
        tokens [B,L,d_token]
                  │
                  ▼
       Continuous Thought Machine
                  │
      logits [B,3,thought_ticks]
```

## 1. Graph-temporal feature extractor

The upstream CTM image experiments use a visual backbone such as ResNet. N-body dynamics have a different inductive structure: objects interact pairwise and move through physical time.

The replacement backbone therefore has two local operations:

### Temporal residual convolution

Each body is processed through short-kernel 1-D residual temporal blocks. This extracts local velocity/curvature/oscillation motifs without performing global reasoning.

### Dynamic graph message passing

At each physical timestep, every body exchanges messages with every other body. Edge features contain:

- relative position;
- relative velocity;
- pair distance;
- source mass;
- source-body learned features.

The block is permutation-equivariant over body order.

The encoder deliberately contains **no global self-attention**. CTM remains responsible for iterative reading/integration.

The output is flattened into `(physical_time, body)` tokens. A token map is kept so CTM attention can later be mapped back to body and physical time.

## 2. CTM core

The computation follows the public CTM ordering.

At thought tick `t`:

1. Compute **action synchronization** from current post-activations.
2. Project action synchronization into an attention query.
3. Attend over the graph-temporal tokens.
4. Concatenate attended input with the current activated state.
5. Pass through the shared **synapse model**.
6. Append resulting pre-activations to each neuron's private memory trace.
7. Each neuron applies its own **Neuron-Level Model (NLM)** to its history.
8. Compute **output synchronization** from the new post-activations.
9. Project output synchronization to 3-class logits.
10. Compute entropy-derived certainty.

Then the process repeats.

## 3. Private neuron models

If CTM has `D` neurons and memory length `M`, each neuron receives only its own pre-activation history:

`[a_d^(t-M+1), ..., a_d^t]`

Each neuron has private weights implemented efficiently by `SuperLinear` tensors. Deep NLMs use GLU nonlinearities, following the upstream implementation's design.

## 4. Synchronization

The model uses random-pairing synchronization. A fixed set of neuron pairs is selected at model construction. For each pair `(i,j)`:

`pair_product_t = z_i^t * z_j^t`

A learnable exponential decay updates a running numerator/denominator, producing the synchronization representation.

Separate neuron-pair sets are used for:

- action/attention synchronization;
- output/prediction synchronization.

## 5. Loss

Base adaptive loss:

- compute CE at every thought tick;
- choose each sample's minimum-loss tick;
- choose each sample's maximum-certainty tick;
- average those two CE terms.

For stability, this project adds:

`L = L_adaptive + lambda_final * CE(final_tick)`

`lambda_final` is configured by `final_ce_weight`.

Set it to `0.0` for the pure upstream-style adaptive classification objective.

## 6. Main-model size

`configs/main.yaml` initializes to roughly **1.09 million trainable parameters**. Parameter count does not grow with the number of CTM thought ticks because the recurrent modules are reused.

The number of thought ticks increases compute/activation cost, not weights.

## 7. Baselines / ablations

- `linear_ctm`: no graph/temporal feature extractor; simple body-time token projection into CTM.
- `graph_pool`: graph-temporal encoder + mean/max pooling + classifier.
- `graph_gru`: graph-temporal encoder + body pooling + GRU over physical time.

They exist mainly to diagnose whether internal CTM analysis is occurring on a meaningful task.
