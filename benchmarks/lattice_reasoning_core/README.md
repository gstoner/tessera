# Lattice Reasoning Core Benchmark

This benchmark is a proof-gated compiler target for Lattice Deduction
Transformers and related reasoning-model primitives.  It emits NumPy reference
rows, public Tessera primitive rows, and Apple GPU executable rows when
`@jit(target="apple_gpu")` reaches `metal_runtime` and matches the oracle.  Rows
that only reach artifact construction stay labeled artifact-only.

It covers:

- LDT-style grid powerset lattices with threshold elimination, empty-cell
  conflict detection, singleton solved detection, and deterministic branch-pin.
- MOPD policy distillation loss over on-policy student logits and multiple
  teacher distributions.
- Mamba-2-style chunked selective scan / SSD recurrence.
- GQA decode attention with repeat-free grouped KV indexing.
- Latent MoE shared down/up projection around routed latent experts.

Current Apple GPU native rows include scalar-A `selective_ssm` and fused
ragged `grouped_gemm`.  LDT primitive rows are also exercised through the
public Tessera API; the integrated LDT step remains artifact-only until the
whole step lowers and executes as one compiler-owned kernel.

Run:

```bash
PYTHONPATH=python python3 benchmarks/lattice_reasoning_core/benchmark_lattice_reasoning.py --smoke --json /tmp/tessera_lattice_reasoning_smoke.json
```
