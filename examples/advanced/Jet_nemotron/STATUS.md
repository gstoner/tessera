# Status: `runnable` compiler slice; optional full-model scaffold

Tracked by `python/tessera/compiler/examples_manifest.py`.

The manifest entry point is `tests/smoke_linear_attention.py`. It executes a
canonical `tessera.ops.linear_attn` graph at D=128, checks a NumPy oracle, and
requires all four compiler artifacts.

## Full-model boundary

`transformer_block.py` imports `from tessera.stdlib import rmsnorm_safe,
dropout`. `tessera.stdlib` is part of the broader Jet-Nemotron research
stack and is **not** part of the standalone compiler surface that ships
under `python/tessera/`.

`tests/test_sanity.py` is the honest CI guard for this directory:

* It locks the post-2026-05-19 fix that replaced the bogus
  `from tessera_jetnemotron.transformer_block import ...` ghost-package
  import with a `sys.path` bootstrap pointing at the sibling
  `transformer_block.py`.
* When the upstream `tessera.stdlib` stack is not on `PATH`, the test
  emits an explicit `pytest.skip` naming the missing module rather
  than silently passing.

Install the upstream Jet-Nemotron research stack (which ships
`tessera.stdlib`) to exercise `examples/e2e_infer.py` end-to-end. That optional
surface remains outside the runnable manifest entry.

The D=128 smoke is host-portable compiler and numerical evidence only. It does
not close or provide exact-device evidence for the ROCm load/wait scheduling
gap.
