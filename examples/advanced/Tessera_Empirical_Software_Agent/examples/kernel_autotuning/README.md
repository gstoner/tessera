# Kernel Autotuning Task

This maintained benchmark provides a concrete optimization target: choose a
tile configuration for a tiny matrix kernel, run correctness checks, and score
throughput.

It is deliberately CPU-only today so the loop is runnable without Tessera runtime
setup. The same scoring contract can wrap a real Tessera kernel benchmark later.

## Run

```bash
python3 examples/advanced/Tessera_Empirical_Software_Agent/examples/kernel_autotuning/benchmark_kernel.py
```

The archived LLM/tree-search orchestrator is preserved under
`archive/examples/advanced/Tessera_Empirical_Software_Agent/`; it is not the
active entry point.

## Scoring

The benchmark prints JSON with:

- `correct`: exactness check against a reference path
- `runtime_s`: measured runtime for the candidate
- `score`: correctness-weighted throughput proxy
