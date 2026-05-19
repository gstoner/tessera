# Status: `scaffold`

Tracked by `python/tessera/compiler/examples_manifest.py`.

This directory is a **placeholder**, not a runnable example today.

## Why

The directory currently contains only:

* `README.md` — top-level scoping note.
* `src/` — empty / stub layout.

There is no entry-point script for the manifest to point at, so the
audit row uses `README.md` as the nominal entry point with status
`scaffold`.

## Path forward

This slot is reserved for **autotune + roofline tooling** examples
(`tools/roofline_tools/`, `tessera.autotune`, `tprof`). Land at least
one CPU-runnable script here, then add a `runnable` manifest row that
points at it.

Until that work lands, this scaffold ships unchanged.
