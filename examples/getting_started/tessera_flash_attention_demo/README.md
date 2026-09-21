# Tessera Flash Attention

This example runs canonical `@tessera.jit` + `tessera.ops.flash_attn`, checks
the result against a PyTorch SDPA reference, and can save the real Graph,
Schedule, Tile, and Target IR exposed by the compiled `JitFn`.

The Tessera lane intentionally uses `reference_cpu` so the example remains
portable. PyTorch and Tessera wall times are printed separately and are not a
native-device performance comparison.

## Quick start

```bash
PYTHONPATH=python:examples/getting_started/tessera_flash_attention_demo \
  python examples/getting_started/tessera_flash_attention_demo/examples/flash_attention_demo.py \
  --batch 1 --heads 2 --seq 32 --dim 16 --dtype f32 --device cpu --causal --dump-ir
```

Run from the repository root. Missing Tessera, numerical disagreement, or an
empty compiler artifact is a hard failure; the example does not emit
placeholder IR.

Artifacts:
- `artifacts/{graph,schedule,tile,target}_ir.mlir`
- `artifacts/compilation_summary.json`
