# Status: `scaffold`

Tracked by `python/tessera/compiler/examples_manifest.py`.

This directory is a **research sketch**, not a runnable example today.

## Why

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

## Path forward

Install the upstream Jet-Nemotron research stack (which ships
`tessera.stdlib`) to exercise the example end-to-end. The skeleton
under this directory is correct; only the optional dependency is
missing.

Until then, this scaffold ships unchanged. The manifest's drift gate
will flag any README that claims it is runnable in the default venv.
