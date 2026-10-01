# Relationship to the official Continuous Thought Machine implementation

Official repository:

https://github.com/SakanaAI/continuous-thought-machines

Official project page:

https://pub.sakana.ai/ctm/

The official repository is Apache-2.0 licensed at the time this project was prepared. Consult the upstream repository for the current license and authoritative implementation.

## What was deliberately preserved

This project uses a task-adapted implementation with the same key algorithmic organization:

- internal recurrence / thought ticks;
- learned start activated state and pre-activation trace;
- private neuron-level temporal models;
- GLU-based deep NLM option;
- synchronization as the direct representation for action/attention and output;
- random neuron pairing;
- learnable exponential synchronization decay;
- synchronization-derived single-query multi-head attention;
- shared synapse model before NLM processing;
- output logits at every thought tick;
- normalized-entropy certainty;
- adaptive classification loss using minimum-loss and most-certain ticks.

## What is different

1. The image ResNet/backbone machinery is replaced with a graph-temporal N-body encoder.
2. Tracking returns PyTorch tensors in a structured dictionary rather than NumPy arrays from inside the forward pass.
3. This project supports token metadata mapping attention back to body IDs and physical timesteps.
4. A configurable final-tick CE stabilizer is added (`final_ce_weight`). This is task-specific and not presented as part of the original CTM method.
5. Training/checkpoint/analysis infrastructure is purpose-built for this course experiment.

This repository should therefore be described as a **task-adapted CTM implementation inspired by and aligned with the public reference code**, not as an unmodified copy of Sakana AI's repository.
