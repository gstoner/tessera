# gfx1151 paged-KV native Schedule slice — 2026-09-28

Owner: E2E-REAL-6. Host: Princess-Luna WSL2, `gfx1151` Radeon 8060S, with
`TESSERA_ROCM_CHIP=gfx1151` and the rebuilt production `tessera-opt`.

`bench.py` first checks each exact native result against a permuted-page NumPy
oracle, then reports medians of seven warm package calls and 21 `rt.launch`
calls. These are `perf_counter_ns` **host wall times**, including Python,
compiler replay, HIP allocation/copies and synchronization. They are not
device kernel times and cannot justify a throughput claim.

| Logical interval | Native image digest | Warm package median | Launch median |
| --- | --- | ---: | ---: |
| `[0, 1)` | `51a14ca50ecea604deb4ac349a54423a299ff27756d7fe929d1c0deec92227db` | 102.669 ms | 2.259 ms |
| `[3, 10)` | same | 96.497 ms | 2.164 ms |
| `[0, 16)` | same | 94.040 ms | 2.163 ms |

The same image digest across intervals validates this shape-free image reuse
for the tested envelope. The ~100 ms warm package cost includes repeated
native Graph/Schedule/Tile replay; a package cache is the next overhead
experiment. The ~2 ms launch wall time includes the host-to-device and
device-to-host path, so it does not isolate the paged-read kernel.

The focused device suite also passed four oracle intervals and a replay-drift
refusal. Run the probe from the Tessera root with `PYTHONPATH=$PWD/python:$PWD`
and `TESSERA_OPT` pointing to the rebuilt compiler.
