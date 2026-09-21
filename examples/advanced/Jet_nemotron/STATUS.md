# Status: `runnable`

Tracked by `python/tessera/compiler/examples_manifest.py`.

The manifest entry point is `tests/smoke_linear_attention.py`. It executes a
canonical `tessera.ops.linear_attn` graph at D=128, checks a NumPy oracle, and
requires all four compiler artifacts.

The incomplete full-model/PostNAS source drop has moved to
`archive/examples/advanced/Jet_nemotron/`; the active directory contains only
the maintained compiler slice and its documentation.

The D=128 smoke is host-portable compiler and numerical evidence only. It does
not close or provide exact-device evidence for the ROCm load/wait scheduling
gap.
