# GFX1201 streams, external readers and sparse Target IR

Owner: E2E-REAL-6 / ROCM-2. Sync: GFX1201-STREAM-SPARSE-IR-2026-09-13.
Measured 2026-09-13 on Tajasarus, Radeon RX 9070 XT gfx1201, ROCm 10.0
(HIP 7.15), Ubuntu 26.04 WSL2, LLVM 23.1.1 with assertions enabled.
Sources are uncommitted; `source-identity.json` binds their bytes and compiler binaries.

## Scope and evidence

- `tests.txt`: focused resident attention, same-device external read-only output
  leases, queued retirement, retryable release failure, multi-reader retirement,
  wrong-device refusal, sparse Target verification and registry/contract gates.
  Hardware tests explicitly forbid `hipDeviceSynchronize`; reader coverage copies
  device gradients on an external stream before releasing and retiring the owner.
  Imported NVIDIA readiness warnings are not NVIDIA validation.
- `sparse-ir.json`: six exact f16/bf16 comparisons. Packed A/B/index inputs enter
  registered `tessera_rocm.swmmac`, the production Target-to-ROCDL pass, and the
  MLIR GPU-to-HSACO pipeline. Native disassembly contains the expected SWMMAC.
  The op validates gfx1201 wave32 fragment types; gfx1151 and malformed results
  refuse. It is an internal physical op, not a public Graph operation or AD rule.
- `overlap.json`: two independent resident recompute-attention owners, private
  nonblocking HIP streams, shape B=1, Hq=4, Hkv=2, M=129, N=131, D=64, fp16.
  All Q/K/V gradients match the reference. Four of five paired ordered-program
  windows intersect by 0.579–0.788 ms; the first does not intersect. HIP events
  bracket each whole program, including resets. This can include interleaved
  kernels and does not prove simultaneous execution of individual kernels.

## Acceptance boundary

Saved-LSE admission remains closed. These leases protect existing device gradient
outputs; they do not establish reusable persisted-LSE ownership. Calls within one
owner remain ordered and host results are copied snapshots. Capture/allocation,
blocking close, and driver module unload are not fully asynchronous or bounded.
Uncertain teardown retains resources rather than claiming recovery.

The Target sparse producer/consumer is executable, but public sparse
Graph/Schedule/Tile packaging, additional sparse forms and general matrix
admission remain open. No new public dtype, operation, batching or AD contract is
claimed. No Apple, CUDA, gfx1151 or CPU execution proof transfers from this packet.

These are uncalibrated one-process WSL event windows. Kernel-dispatch/counter
attribution and clean multi-process performance evidence remain separate gates.
`performance_eligible` remains false; there is no performance promotion.

## Reproduction

On the owning host, source `~/.config/tessera/env.sh`, then run:

```sh
python -m benchmarks.rocm.record_gfx1201_sparse_wmma --output /tmp/sparse-ir.json
python -m benchmarks.rocm.record_gfx1201_attention_overlap --output /tmp/overlap.json
TESSERA_GFX1201_DEVICE_PROOF=1 python -m pytest -q tests/unit/test_resident_rocm_attention.py tests/unit/test_rocm_sparse_packing.py tests/unit/test_rocm_gfx1201_scheduled.py
```
