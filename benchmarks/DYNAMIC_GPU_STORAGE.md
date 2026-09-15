# Native dynamic GPU storage experiment

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`. Recorded 2026-09-05.

Both RTX 5070 (Super-Bear) and gfx1151 (Princess-Luna) pass exact numerical
checks with compiler-generated dynamic shared storage and native host sizing.
This closes the bounded compiler-to-device experiment. The subsequent
[package proof](NATIVE_GPU_STORAGE_PACKAGE.md) adds a raw native package binding,
uniform nested lifetimes and a bounded NVGPU producer integration.

The input is the [native MLIR fixture](../tests/tessera-ir/phase3/tile_dynamic_gpu_arena.mlir).
`TileBufferArena` emits workgroup views and a host sizing companion; LLVM
compiles the companion into machine code. The recorder calls that native
function for each launch byte count. The numerical oracle independently checks
the count and every output. In this historical packet, negative and oversized sizes aborted in fresh native
processes. The current companion instead returns a checked failure value; the
package proof validates rejection before dispatch without aborting the caller. The emitted GPU module independently lowers through NVVM/NVPTX or
ROCDL/AMDGPU; no CUDA/HIP source generator or Python execution fallback is used.

Each case uses 256 blocks and 17 iterations. Every thread writes scratch, waits
at a publish barrier, reads its neighbor, and waits at a release barrier before
the next write. The static comparison specializes the same source per extent;
one dynamic image serves all four extents. Five samples each contain 20
launches, timed by CUDA/HIP events after three warmups. Outputs are checked
before and after timing.

| Device | Threads | Static ms | Dynamic ms | Dynamic / static |
|---|---:|---:|---:|---:|
| RTX 5070 | 32 | 0.009078 | 0.009395 | 1.035 |
| RTX 5070 | 64 | 0.009938 | 0.011858 | 1.193 |
| RTX 5070 | 128 | 0.012021 | 0.010342 | 0.860 |
| RTX 5070 | 256 | 0.010898 | 0.009670 | 0.887 |
| gfx1151 | 32 | 0.002759 | 0.005790 | 2.099 |
| gfx1151 | 64 | 0.003699 | 0.003350 | 0.906 |
| gfx1151 | 128 | 0.003053 | 0.002778 | 0.910 |
| gfx1151 | 256 | 0.004610 | 0.004396 | 0.954 |

These are event-span averages, which can include gaps between launches. They
are not isolated kernel latency or cross-run confidence bounds. The short
workload and order effects limit performance conclusions; no general speedup,
selector eligibility or cross-architecture timing equivalence is claimed.
Static images specialize allocation size as well as storage placement.

Evidence:

- [NVIDIA packet](baselines/dynamic_gpu_storage_nvidia.json)
- [ROCm packet](baselines/dynamic_gpu_storage_rocm.json)
- [Recorder](record_dynamic_gpu_storage.py)

Packets record the actual core compiler, LLVM driver, host library, recorder,
fixture and device-image hashes, GPU identity, launch counts, resource reports,
raw samples and native rejection cases. Both hosts used the same core compiler
binary; device compilation and execution were independent. The driver-reported
static shared bytes are retained separately from launch-time dynamic bytes.

Reproduce on the owning WSL host with LLVM 23 tools and its GPU visible:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/record_dynamic_gpu_storage.py \
  --backend rocm \
  --tessera-opt build/tools/tessera-opt/tessera-opt \
  --artifacts /tmp/tessera-dynamic-gpu-rocm \
  --output /tmp/dynamic_gpu_storage_rocm.json
```

For NVIDIA, use `--backend nvidia`, the exact intended core compiler path,
CUDA 13.4 (the pinned toolkit since 2026-09-15; the 13.3 packets are historical) and `scripts/_nvidia_env.sh`. GPU images, emitted host LLVM IR and the
native sizing library remain in the chosen artifacts directory for inspection.

Apple requires its own MSL threadgroup-argument materializer and launch ABI.
The x86 execution backend retains ordinary host allocation; the host sizing
companion itself was executed natively on both WSL x86 machines. This experiment
does not add an Apple or x86 kernel performance claim.
