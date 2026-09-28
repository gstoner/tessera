# Exact Graph package cache, Princess-Luna (2026-09-28)

`benchmarks/x86/measure_x86_scheduled_absolute_cache.py` measured native `trunc`
packaging in the Princess-Luna WSL project environment. Each row uses a fresh
Graph shape for the first call and seven exact repeats; values are milliseconds.
The package still comes from Graph → Schedule → Tile → x86 Target → native image.
The repeat returns a defensive copy of that verified package.

| Shape | First call | Repeat median |
|---|---:|---:|
| 51 | 104.183 | 0.314 |
| 3×17 | 96.850 | 0.336 |
| 2×3×17 | 98.217 | 0.313 |
| 5×23 | 100.746 | 0.317 |

This is package wall time, not kernel execution time. The cache key includes
exact canonical Graph text, pipeline, compiler/image identity and the native
lowering functions. New shapes still pay about 100 ms. Cache reuse does not
change the currently prepackaged AVX-512 kernel body for `trunc`.
