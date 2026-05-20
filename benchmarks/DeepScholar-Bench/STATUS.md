# Status: `runnable`

Tracked by `python/tessera/compiler/benchmarks_manifest.py`.

This directory now has a **CPU smoke benchmark** that uses only the
current Tessera compiler surface:

* `@tessera.jit`
* `tessera.ops.matmul`
* `tessera.ops.softmax`
* `tessera.ops.layer_norm`
* NumPy-backed deterministic text/source embeddings

The smoke writes `tessera.deepscholar_smoke.v1` JSON with compiler
metadata, artifact levels, execution kind, runtime status, and a
correctness check against the same public operator chain.

## What is not claimed

The full LOTUS/DeepScholar research workflow is not ported.  The
optional `tessera_lotus_deepscholar.py` adapter imports cleanly without
LOTUS or pandas, but it raises a clear optional-dependency error until
that stack is installed and configured.

## Smoke command

```bash
PYTHONPATH=python python benchmarks/DeepScholar-Bench/tessera_deepscholar_model.py \
  --output /tmp/tessera_deepscholar_smoke.json
```
