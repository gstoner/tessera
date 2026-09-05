# Native asynchronous producer/consumer GEMM

Owner: W2.4a / CAKE / SO-2. Synchronization key: `IR-NATIVE-FOUNDATION-1`.

`benchmark_async_producer_consumer.py` exercises the production SM120 macro-CTA
F16 GEMM lowering. Threads produce the next A/B panel with `cp.async` into an
alternating shared-memory slot while tensor-core MMA consumes the current panel.
The existing kernel's wait and CTA barriers handle prime, publish and reuse.
This is useful matrix multiplication, not an empty copy or synchronization loop.
It does not yet connect the generic allocation-lifetime analysis to this kernel.

The input is the native macro-CTA lit fixture, compiled through Tessera's NVIDIA
lowering and the production downstream MLIR/LLVM/PTX passes. The fixture's
schedule hash is a placeholder, not a production schedule attestation. The
benchmark is experimental and never packages its ablated image as that schedule.

The control inserts one immediate `nvvm.cp.async.wait.group 0` at the native
loop-prefetch commit. It retains all existing instructions, two slots, tile
sizes, grid, types and arithmetic. It fails if the expected prime/prefetch
commit structure changes. There is no product compiler or selector modification.

## Reproduce on Super-Bear WSL

From the Tessera checkout, using its Python environment:

```sh
export CUDA_HOME=/usr/local/cuda-13.3
source scripts/_nvidia_env.sh
export PATH=/usr/lib/llvm-23/bin:$PATH
export TESSERA_NVIDIA_OPT=/home/angstorms/programming/tessera/build-nvidia-cuda/src/compiler/codegen/tessera_gpu_backend_NVIDIA/tools/tessera-nvidia-opt
.venv/bin/python benchmarks/nvidia/benchmark_async_producer_consumer.py \
  --artifacts /tmp/async-gemm --output /tmp/async-gemm-timing.json
```

Compilation, host copies and the NumPy oracle are outside timing. Each variant
gets a correctness launch and three warmups. Seven samples alternate route order;
each sample averages twenty resident launches using CUDA events. Small shapes
are correctness coverage: host launch gaps can dominate their event intervals.
The 1024 and 2048 grids have 1024 and 4096 CTAs. This is one process on one device,
not a cross-run confidence bound or evidence for selector promotion.

For each variant, run Nsight Compute separately with `--profile async` or
`--profile serialized`, using a distinct output path. Select `--launch-skip 4
--launch-count 1` (oracle plus three warmups precede capture), and sections
`LaunchStats`, `Occupancy`, `SchedulerStats`, `WarpStateStats`,
`MemoryWorkloadAnalysis`. Profile Nsight Systems separately with
`--trace=cuda,nvtx --sample=none --cpuctxsw=none`. Profiler output packets carry a
non-null `profile`; their event durations include instrumentation and must never
be used as benchmark samples. Run all GPU captures sequentially.

## RTX 5070 evidence, 2026-09-05

Driver 610.88; Nsight Compute 2026.2.1; Nsight Systems 2026.1.3.
Source, compiler tools and cubin SHA256s are recorded in
[`async_gemm_nvidia.json`](../baselines/async_gemm_nvidia.json). The profiler
captures used matching cubin hashes. All six shapes passed the F32 NumPy oracle
with F16 inputs, including a single partial panel, a two-panel tail, and 33-panel
reuse. Maximum absolute error across the tested shapes was 4.92e-5.

| Square M=N=K | Async median | Immediate-wait median | Latency reduction |
|---|---:|---:|---:|
| 512 | 0.021330 ms | 0.022704 ms | 6.1% |
| 1024 | 0.125928 ms | 0.129741 ms | 2.9% |
| 2048 | 0.941216 ms | 0.963016 ms | 2.3% |

Both cubins use 56 registers/thread, 4096 bytes static shared memory and no
spills. SASS contains the same `LDGSTS.E.128.ZFILL` producers. In the async loop,
`HMMA` at 0x1050 precedes `DEPBAR` at 0x1060; the control adds `DEPBAR` at 0x0e60
before the first loop `HMMA` at 0x1060. This proves the intended issue/wait order
survived native compilation, not a measured overlap percentage.

The exported [async](../baselines/async_gemm_nvidia_ncu_async.csv) and
[serialized](../baselines/async_gemm_nvidia_ncu_serialized.csv) Compute captures
show long-scoreboard stalled warps per issue-active falling from 6.061 to 2.081.
Barrier stalls rise from 3.987 to 4.842; achieved occupancy is 64.9% versus 63.3%.
These observations support reduced exposed memory waits without a resource
capacity improvement. Replay durations are excluded from the timing table.
Systems [kernel](../baselines/async_gemm_nvidia_nsys_kernels.csv) and
[API](../baselines/async_gemm_nvidia_nsys_api.csv) exports separate kernel work
from process/context setup; a timeline alone cannot prove intra-kernel overlap.

**Sanitizer blocker:** Compute Sanitizer reported “Failed to initialize WDDM
debugger interface” and “Device not supported.” Its final two-error summary
reflects initialization failures, not identified kernel races. No racecheck or
synccheck pass is claimed. Enabling the host debugger interface and rerunning
racecheck/synccheck remains required before calling synchronization sanitizer-proven.

Next: connect explicit allocation/release tokens to this native schedule, rerun
with sanitizer access and multiple independent timing processes, and evaluate a
ROCm-specific asynchronous producer on gfx1151. CUDA schedule evidence does not
establish ROCm, Metal, or x86 parity.
