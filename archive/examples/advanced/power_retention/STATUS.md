# Status: `scaffold`

Tracked by `python/tessera/compiler/examples_manifest.py`.

This directory is a **placeholder**, not a runnable example today.

## Why

The declared entry point `examples/minimal_power_attn.py` currently
contains only:

```python
print('example')
```

The real implementation lives in the `python/tessera_power/`
subpackage (a separate installable that ships CUDA scaffolds for
retention inference + a Vidrial-style staged-SMEM kernel structure).
That subpackage is not wired into the audit harness yet and assumes
a CUDA build that the default CPU-only CI does not provide.

## Path forward

Either:

1. Flesh out `examples/minimal_power_attn.py` against the canonical
   `@tessera.jit` surface so it runs on CPU as a numpy reference, then
   promote the row to `runnable`.
2. Or keep the `tessera_power/` CUDA scaffolds as the primary surface
   and declare a `runnable_optional` row gated on a `tessera_power`
   import.

Until that decision lands, this scaffold ships unchanged.
