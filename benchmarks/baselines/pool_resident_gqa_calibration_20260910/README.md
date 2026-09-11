# GPU pool, resident SSD and CUDA calibration — 2026-09-10

This is correctness and calibration evidence from dirty continuations after
087e2dd438914fe0fcf107a65030e246bd2ac73b. It does not promote a production route.

- CUDA: RTX 5070 SM120 on Super-Bear WSL, assertions-enabled LLVM 23 compiler.
- ROCm: AMD 8060S gfx1151 on Princess-Luna WSL, assertions-enabled LLVM 23 compiler.
- Each packet records its actual compiler and four package binding hashes. The
  CUDA compiler includes the new symbolic GQA instantiator; ROCm's compiler is the
  preceding build, whose unchanged pool/SSD pipeline executes this increment.
- Source hashes describe the accompanying source snapshot, not a clean commit.

`record_pool_resident_ssd.py` validates three reusable fixed-width slots:
rooted cycles survive partial collection, an unrooted cycle is collected, stale
edge generations refuse without a sweep, invalid live generations refuse, and
repeated allocation/collection reuses slots with incrementing generations. The
same resident GPU allocations are retained throughout. Readers must be quiescent;
this is bounded stop-the-world numeric storage, not arbitrary/concurrent GC.

The SSD owner automatically constructs forward/VJP packages, snapshots inputs
on-device, keeps checkpoints private and exposes five resident gradients. Each
gradient is checked against float64 finite differences, including after the caller
overwrites its original X allocation. Closed outputs refuse access, and both
capture/backward and the value_and_grad entry point execute. These are synchronous
first-order tests with a T=3, H=N=P=2, chunk=2 program.

CUDA causal GQA is separately validated by test_attention_loop_idiom.py with
Hq=4, Hkv=2 and Q/K lengths 3/5 and 5/3. Wrong mask alignment and nondivisible heads
refuse. Apple, x86 and ROCm do not inherit this execution certificate.

`cuda-calibration.json` contains seven 100-launch event windows and the raw
701 Nsight product-kernel intervals (one checked warm-up). The comparison uses
first-start to last-end spans, including submission gaps; kernel active time is
reported separately. Maximum event/span disagreement is 2.2576%. Profiler overhead
is 1.9573x relative to a separate clean-image run, so the fixed 1.05 limit refuses.
Dirty source and WSL are additional refusal reasons. This is one calibration,
not the 18 distinct eligible process calibrations required by SSD admission.

Raw source capture on Super-Bear:
`/tmp/ssd-calibration-wave.nsys-rep` and `/tmp/ssd-calibration-wave.sqlite`;
the packet binds the SQLite SHA256 and captured CUDA architecture/device UUID.
Use benchmarks/calibrate_ssd_cuda.py to regenerate from the SQLite, profiled JSON
and clean-image JSON. Older nine-pair packets retain their previous compiler
identity and must not be silently reused with the rebuilt CUDA compiler.
