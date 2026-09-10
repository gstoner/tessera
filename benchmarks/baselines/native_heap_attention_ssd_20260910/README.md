# Serial SSD owning-device correctness

Recorded 2026-09-10 from the uncommitted continuation of PR #740 using `benchmarks/record_ssd_gpu.py`. `cuda.json` is RTX 5070 SM120 on Super-Bear; `rocm.json` is gfx1151 on Princess-Luna. Compiler and image digests bind each record to its materialization.

Each run uses T=5, H=2, N=3, P=2 and chunks 1, 2, 5. Independent NumPy recurrence checks cover Y, final carry and chunk-end checkpoints, including a partial chunk. All five device input buffers must remain bitwise unchanged. Packages use one thread and replay their incoming Schedule IR; no timing, cooperative execution or performance promotion is established.

Native host heap validation separately compiles the production C++ allocator and tests cycles, exhaustion, stale handles and partial payload reuse. Attention tests cover two symbolic recipe buckets and reject altered recurrences and stale head-width witnesses. These tests do not establish full CPython frames or GPU attention candidate admission.
