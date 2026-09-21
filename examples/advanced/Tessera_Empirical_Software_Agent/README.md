# Kernel-autotuning candidate benchmark

The maintained example is the deterministic candidate benchmark used by an
empirical search loop:

```bash
python3 examples/advanced/Tessera_Empirical_Software_Agent/examples/kernel_autotuning/benchmark_kernel.py
```

It loads an optional tile configuration, runs a tiled matrix multiplication,
checks it against an independent reference, and reports a bounded score. It
fails on invalid tile parameters or numerical mismatch.

The earlier LLM/tree-search agent, sandbox stub, placeholder MLIR pass, and
empty domain-task shells are preserved under
`archive/examples/advanced/Tessera_Empirical_Software_Agent/`. They are not
part of the runnable examples contract.
