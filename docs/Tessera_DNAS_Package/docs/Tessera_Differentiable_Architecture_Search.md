# Tessera Differentiable Architecture Search (DNAS)

This document describes how Tessera can support Differentiable Neural Architecture Search (DNAS).

## Key Ideas
- MixedOp with differentiable gates (e.g., Gumbel-Softmax).
- Bilevel optimization: weights trained on training set, architecture logits on validation set + hardware cost.
- Surrogate cost model: latency, memory, energy estimated differentiably.
- Annealing schedule for gate sharpness.
- Discretization step to finalize architecture.

## Example Flow
1. Define search space as a set of candidate ops.
2. Relax discrete choice into soft gates (softmax).
3. Train weights + gates with alternating updates.
4. Evaluate with hardware-aware loss.
5. Collapse gates into hard choices for final network.
