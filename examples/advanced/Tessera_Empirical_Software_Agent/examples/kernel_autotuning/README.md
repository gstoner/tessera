# Kernel Autotuning Task

This task gives the empirical software agent a concrete optimization target:
choose a tile configuration for a tiny matrix kernel, run correctness checks, and
score throughput.

It is deliberately CPU-only today so the loop is runnable without Tessera runtime
setup. The same scoring contract can wrap a real Tessera kernel benchmark later.

## Run

```bash
python -m examples.advanced.Tessera_Empirical_Software_Agent.src.agents.kernel_autotune_loop \
  --task examples/advanced/Tessera_Empirical_Software_Agent/examples/kernel_autotuning
```

## Scoring

The benchmark prints JSON with:

- `correct`: exactness check against a reference path
- `runtime_s`: measured runtime for the candidate
- `score`: correctness-weighted throughput proxy
