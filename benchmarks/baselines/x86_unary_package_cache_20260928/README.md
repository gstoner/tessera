# x86 unary exact-package cache, 2026-09-28

Princess-Luna (Zen 5 AVX-512, WSL2), Tessera `2a70e08a` with the exact
package-cache patch. [`cost.json`](cost.json) is the unedited output of
`benchmarks/x86/measure_x86_unary_route_cost.py` with the rebuilt compiler.
Each family includes eight distinct shapes and an immediate repeated direct
package call for each shape.

| Family | Retired first package | Compiled first package | Compiled exact repeat |
| --- | ---: | ---: | ---: |
| Softmax | 32.45 ms | 100.33 ms | 0.20 ms |
| Reduction | 32.24 ms | 100.44 ms | 0.20 ms |

The first compiled package for a new shape still pays Graph → Schedule →
Tile, ancestry replay and Target lowering. The cache removes that work for
an exact repeated package request; it is bounded to 64 entries and keys the
compiler and shared-image file identities. The driver's scheduled-artifact
cache is covered by focused tests, while these timing rows exercise the direct
packagers. Runtime launch timing is a separate measurement.
