# Lattice Reasoning Core Benchmark

This benchmark is a proof-gated compiler target for Lattice Deduction
Transformers and related reasoning-model primitives.  It starts as NumPy
reference execution plus artifact visibility; it does not claim native
compiler execution until a Tessera lowering path and oracle fixture prove it.

It covers:

- LDT-style grid powerset lattices with threshold elimination, empty-cell
  conflict detection, singleton solved detection, and deterministic branch-pin.
- MOPD policy distillation loss over on-policy student logits and multiple
  teacher distributions.
- Mamba-2-style chunked selective scan / SSD recurrence.
- GQA decode attention with repeat-free grouped KV indexing.
- Latent MoE shared down/up projection around routed latent experts.

Run:

```bash
PYTHONPATH=python python3 benchmarks/lattice_reasoning_core/benchmark_lattice_reasoning.py --smoke --json /tmp/tessera_lattice_reasoning_smoke.json
```
