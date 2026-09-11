# Broader native adapters and profiler attribution

2026-09-10, Super-Bear RTX 5070 SM120 and Princess-Luna Radeon 8060S gfx1151,
both WSL. No performance promotion.

`nvidia-suite.json` and `rocm-suite.json` preserve the normal SuperBench runner
results: baseline ANN, 64×8 square ANN, serial SSD and cooperative SSD all pass.
SSD shape is 32×2×8×4, chunk 8. Host latency and device-event windows remain
separate; these single-process diagnostics do not establish speedup eligibility.

`cuda-profiled.json` is a fresh dedicated cooperative SSD measurement under
Nsight Systems 2026.1.3. `cuda-attribution.json` records all 701 kernels matched
to successful CUDA launch API calls by process and correlation ID, with capture
hash, artifact/image identities and separate API/kernel timestamps. Raw capture
and SQLite remain on Bear at `/tmp/ssd-adapter-profile.{nsys-rep,sqlite}`.
The normalized attribution preserves every matched timestamp/correlation.

`rocm-profiled.json` records the same checked workload under rocprofv3 1.3.5.
The retained HIP API and agent CSVs are the only emitted trace files despite
`--hip-trace --kernel-trace --memory-copy-trace`. Kernel/copy attribution is
unavailable here; HIP event timing alone does not fill that gap.

Reproduction: run the native SuperBench configs with `TESSERA_OPT` set. For
CUDA profiling, use `nsys profile --trace=cuda --sample=none --cpuctxsw=none`
around `record_ssd_gpu.py --backend nvidia --compiler <tool> --shape 32 2 8 4
--chunk 8 --cooperative --profile --output <packet>`, export SQLite, then run
`attribute_ssd_cuda.py --packet <packet> --sqlite <capture> --output <result>`.
The attribution envelope is a dedicated single-artifact process, not general
mixed-workload attribution. No counters or clean bare-metal promotion claimed.
