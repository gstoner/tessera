# Tessera ROCm Backend

This backend defines ROCm hardware-free Target IR contracts, the AMDGCN emit
path, and the runtime lane that **executes on real AMD GPUs**. The active Python
artifact path lowers through the verified object compiler spine:

```text
textual DSL / @jit -> Graph IR -> Schedule IR -> Tile IR -> ROCm Target IR
```

`python/tessera/compiler/target_ir.py` maps Tile IR to:

- `tessera_rocm.wmma` for matmul/MMA contracts on **RDNA** (gfx11xx) targets
- `tessera_rocm.mfma` for matmul/MMA contracts on **CDNA** (gfx9xx) targets
- `tessera_rocm.async_copy` and `tessera_rocm.wait` for LDS movement contracts
- `tessera_rocm.elementwise` for generic elementwise artifacts
- `tessera.target.diagnostic` for contracts a target cannot lower (e.g. KV-cache
  on an arch without it) — a stable, op-and-target-named diagnostic, never a
  silent no-op (Decision #21)

`lower-tile-to-rocm` selects WMMA vs MFMA by arch and preserves the
no-FP8-WMMA-on-RDNA gate (gfx1151 is RDNA 3.5: WMMA F16/BF16/IU8/IU4 only; FP8/BF8
WMMA + sparse SWMMAC are RDNA4-only). These artifacts are covered by
`tests/unit/test_target_ir.py` and `tests/unit/test_target_ir_contract.py`.

## Hardware execution — gfx1151 (Strix Halo, RDNA 3.5) is live

A real Strix Halo box (Ryzen AI Max+ 395, ROCm 7.2.4, LLVM/MLIR 22) executes a
**compiler-generated matmul + flash-attention family** on the `gfx1151` iGPU via
`runtime.launch(target="rocm")` — the first non-Apple `backend_kernel` proof:

- **`matmul`** — tiled/K-looped RDNA **WMMA** GEMM (`fp16`/`bf16`, fp32 accumulate,
  16×16 output tiles, ragged edges zero-padded), with a perf ladder and a fused
  bias/relu/gelu/silu epilogue. Shipped C-ABI symbols `tessera_rocm_wmma_gemm_f16`
  / `_bf16` in `libtessera_rocm_gemm.so` HIPRTC-compile the kernel for whatever
  arch the device enumerates at load (no `hipcc`-as-compiler needed). The
  `backend_manifest` matmul row for the rocm target is **`hardware_verified`**
  (`fp16,bf16`), proven by execute-and-compare vs numpy.
- **`flash_attn`** and **`GQA`/`MQA`** — forward **and** backward.
- **sliding-window** attention and **Gemma-2 logit soft-capping** — forward
  (compiler-generated, executes on gfx1151).

The emit + launch mechanics: `rocdl_emit.py` (the AMD analog of `ptx_emit.py`)
emits `llvm.amdgcn.wmma.*` LLVM IR and `llc`-assembles it to real `v_wmma_*`
AMDGCN; `runtime.launch()` dispatches through the backend-agnostic C-ABI launch
bridge (`tsrLaunchKernel` → a registered `tsrGpuLauncherFn` → HIP launch),
mirroring the Apple G7 bridge.

> **`@jit(target="rocm")` auto-stamp is intentionally not wired** —
> `JitFn.is_executable` reads compile-time `execution_kind`, which a host runtime
> probe can't honestly drive. `runtime.launch()` is the wired execution lane
> (same path Apple used to earn its matrix row before full JIT support).

**Honest scope:** **CDNA (MI300X/MI325X)** is unproven — it uses a distinct MFMA
shape table + FP4/FP6 and stays hardware-gated. The RDNA op surface beyond matmul
+ attention stays `artifact_only`, and the per-primitive `backend_kernel` axis is
still open across targets. Counts/rows are owned by the generated dashboards
([`runtime_execution_matrix.md`](../../../../docs/audit/generated/runtime_execution_matrix.md),
[`rocm_target_map.md`](../../../../docs/audit/generated/rocm_target_map.md)) — see
[`docs/audit/backend/rocm/ROCM_AUDIT.md`](../../../../docs/audit/backend/rocm/ROCM_AUDIT.md)
and [`STRIX_HALO_EXECUTION_PLAN.md`](../../../../docs/audit/backend/rocm/STRIX_HALO_EXECUTION_PLAN.md)
for the full bring-up story.

## Build
```bash
mkdir build && cd build
cmake -G Ninja -DMLIR_DIR=<path>/lib/cmake/mlir -DLLVM_DIR=<path>/lib/cmake/llvm \
      -DTESSERA_ROCM_BUILD_TOOLS=ON -DTESSERA_ROCM_BUILD_RUNTIME=ON ..
ninja
```

On the Strix Halo box, configure with `-DTESSERA_ENABLE_HIP=ON
-DTESSERA_BUILD_ROCM_BACKEND=ON -DCMAKE_PREFIX_PATH=/opt/rocm` (ROCm 7.2.4). The
build and lit fixtures need no GPU; kernel execution needs a GPU + `/dev/kfd`.

## Lower + Emit
```bash
# Lowering tests
ninja check-tessera-rocm

# Emit HSACO (auto-detects toolchain; gfx1151 = Strix Halo RDNA 3.5,
# gfx90a/gfx94x/gfx950 = CDNA, gfx1100 = RDNA 3 discrete)
./tessera-rocm-emit ../test/rocm/async_and_mfma_realish.mlir out.hsaco --mcpu=gfx1151
# Writes out.hsaco and out.hsaco.metadata.json
```

## Launch (HIP)
```bash
./launch_demo out.hsaco <kernel_symbol>
```

## CK Bridge
Enable with `-DTESSERA_ROCM_ENABLE_CK=ON`. If `composable_kernels` is discoverable, the bridge uses it; else it logs a stub.
