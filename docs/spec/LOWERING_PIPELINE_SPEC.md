---
status: Normative
classification: Normative
last_updated: 2026-07-13
---

# Tessera Lowering Pipeline Specification
**Status:** Normative — grounded in `src/transforms/lib/` and `src/compiler/tile_opt_fa4/lib/` Phase 1–8 implementations
**Last updated:** July 13, 2026
**Cross-references:** `docs/spec/COMPILER_REFERENCE.md` §Pass Pipeline Registry, `docs/spec/GRAPH_IR_SPEC.md`, `docs/spec/TARGET_IR_SPEC.md`

---

## Current status and proof taxonomy

This specification separates a registered/lit-tested compiler pipeline from
native execution. A family name never inherits hardware proof from a different
architecture: the generated execution matrix is the authority for the exact
`(target, compiler_path)` proof row.

- `rocm` is the family selector; the current native HIP evidence is for
  `rocm_gfx1151` (RDNA 3.5 / Wave32) and is carried by the generic runtime
  route's exact-target evidence.
- `nvidia_sm120` (consumer Blackwell) has native CUDA execution for its proven
  rows. SM80, SM90, and SM100 pipeline contracts and specialized WGMMA/TCGEN05
  paths remain separately scoped until they have matching-device evidence.
- An artifact-only row is an unsupported or unproven *row*, not a statement
  that the complete ROCm or NVIDIA backend cannot execute.

See [`runtime_execution_matrix.md`](../audit/generated/runtime_execution_matrix.md)
and [`rocm_target_map.md`](../audit/generated/rocm_target_map.md) for current
per-op proof, fallback, and exact-device scope.

### Shipped named lowering pipelines (canonical truth)

CLAUDE.md Architecture Decision #19 + #20 are normative for these. The
current set registered in `tools/tessera-opt/tessera-opt.cpp` is:

| Pipeline | Target | Phase | Status |
|----------|--------|-------|--------|
| `tessera-lower-to-x86` | x86 AMX/AVX-512 | 2 | ✅ executable runtime |
| `tessera-lower-to-gpu` | NVIDIA CUDA target family: SM80/90/100 contracts plus SM120 consumer Blackwell | 3 | ✅ implemented / lit-testable; native CUDA runtime is proven for supported `nvidia_sm120` rows; other architecture-specific contracts remain separately scoped |
| `tessera-lower-to-rocm` | AMD ROCm target family: CDNA MFMA and RDNA WMMA, including gfx1151 RDNA 3.5 | 8 | ✅ implemented / lit-testable; native HIP runtime is proven for supported `rocm_gfx1151` rows; CDNA and other RDNA targets require exact-device promotion |
| `tessera-lower-to-apple_cpu` | Apple Silicon CPU (artifact) | 8.1 | ✅ lit-testable |
| `tessera-lower-to-apple_cpu-runtime` | Apple Silicon CPU (Accelerate cblas_sgemm + BNNS f16/bf16) | 8.2 | ✅ executable runtime |
| `tessera-lower-to-apple_gpu` | Apple Silicon GPU (artifact) | 8.1 | ✅ lit-testable |
| `tessera-lower-to-apple_gpu-runtime` | Apple Silicon GPU (MPS, MPSGraph, custom MSL, additive Metal 4 lanes, and packaged-kernel ABI validation) | 8.3 → Metal 4 + PK1–PK7 | ✅ executable runtime on capable Darwin hosts |
| `tessera-nvidia-pipeline-{sm90,sm100,sm120}` | NVIDIA target-specific composition | 3+ | ✅ registered / lit-testable; runtime proof is target- and op-specific (currently sm120 rows) |
| `tessera-spectral-pipeline` | Spectral solver end-to-end | 5 | ✅ via `ts-spectral-opt` |
| `tpp-space-time` | Tensor Parallel Primitives | 5 | ✅ via `tessera-opt` (4/4 lit fixtures) |

The §1 table below lists only the original x86 / GPU canonical pair to
keep that section focused on Phase 2–3 contract; the full inventory
above is the current truth.

**Fusion-intent stamping in the compile path (2026-06-11).** For Apple
targets, `driver.compile_graph_module` now calls
`canonical_compile.stamp_fusion_intents(module)` before rendering Graph
IR, tagging each recognized linear chain's terminal op with
`tessera.fusion.intent` from the canonical `_KNOWN_FUSION_CHAINS`. The
Apple Target IR fusion passes consume that intent (descriptor-driven
fusion) and fall back to structural re-discovery when it is absent — see
[`TARGET_IR_SPEC.md`](TARGET_IR_SPEC.md) §"Fusion descriptors" for the
emit/consume contract and the `tessera.fusion.{kernel,source}` attributes.

### Apple packaged-kernel path

The packaged-kernel sprint (PK1–PK7) adds a runtime/compiler-adjacent path
beside source-generated Target IR:

```text
BackendKernelEntry(status="packaged", packaged_pipeline_path=...)
  + AppleKernelBindingSpec(entries=AppleTensorBindingSpec(...))
  -> tessera.apple_mlpkg.compile_mlpackage(...)
  -> reflection extraction / validate_bindings(...)
  -> ArgumentLayout / packaged dispatch
```

This is not a new MLIR pass pipeline. It is a packaged Apple ML artifact
execution path with compiler-emitted tensor-binding contracts and runtime
reflection validation. Production packaged kernels live in
`python/tessera/compiler/apple_packaged_manifest.py`; the checked-in Apple
matrix-multiplication package is a fixture proving the lifecycle, not a
Tessera-authored production kernel.

### Python driver lowering paths

`python/tessera/compiler/jit.py::JitFn` accepts `target=` as either a
`GPUTargetProfile` or a string alias. String dispatch goes through
`compiler/matmul_pipeline.py`; valid string targets per CLAUDE.md
Architecture Decision #20 are: `"rocm"`, `"apple_cpu"`,
`"apple_gpu"`. The Python object-model lowering passes (preserve debug
markers across Schedule and Tile lowering, drop them at Target IR) are
in `python/tessera/compiler/{schedule_ir,tile_ir,target_ir}.py`. Source
of truth for marker elision is `target_ir.py`; lit fixtures verify the
contract under `tests/tessera-ir/`.

### Halo + spectral pass-order matrices

The halo (stencil → bc-lower → halo-mesh-integration → halo-transport)
and spectral (LegalizeSpectral → SpectralMXP → TransposePlan → Autotune
→ LowerToTargetIR → DistributedFFT) pass-order contracts shipped under
`tests/tessera-ir/` lit fixtures and Layer-6 execute-and-compare lanes
(`tests/unit/test_halo_execution_lane.py`,
`tests/unit/test_spectral_solver_passes.py`). They are documented
end-to-end in `docs/audit/compiler/COMPILER_AUDIT.md`.

---

## 1. Overview

Tessera has multiple named lowering pipelines. The two foundational chains
below explain the x86 and NVIDIA Tile-IR lineage; Apple and ROCm have their own
registered target pipelines listed above. Pipeline registration does not itself
claim universal native execution.

| Pipeline | CLI flag | Target | Phase |
|----------|----------|--------|-------|
| `tessera-lower-to-x86` | `--tessera-lower-to-x86` | CPU (x86 AMX / AVX-512) | 2 |
| `tessera-lower-to-gpu` | `--tessera-lower-to-gpu` | NVIDIA GPU family (SM80/90/100/120) | 3 |

Both pipelines start from the same Graph IR input (emitted by `@tessera.jit`) and produce different backend-specific IR. The IR stack at each stage is:

```
Graph IR (tessera dialect)
  → [EffectAnnotationPass]         tessera.effect attrs on func.func
  → [CanonicalizeTesseraIRPass]    fusion patterns applied
  → [DistributionLoweringPass]     tessera.shard → schedule.mesh.*
  → [TilingPass]                   tessera.matmul → scf.for + tensor slices   ← x86 only
  → [TileToX86Pass]                tiled matmul → func.call @tessera_x86_*    ← x86 only
  → [TileIRLoweringPass]           schedule.mesh.region → tile.* + attn.*     ← GPU only
  → [WarpSpecializationPass]       warp role assignment + queue barriers       ← GPU only
  → [AsyncCopyLoweringPass]        tile.async_copy → TMA / cp.async           ← GPU only
  → [NVWGMMALoweringPass]          tile.mma → wgmma.mma_async PTX             ← GPU only
  → [NVTMADescriptorPass]          TMA descriptor hoisting + mbarrier init     ← GPU only
  → [NVFlashAttnKernelEmitter]     FA-4 kernel finalisation                    ← GPU only
```

---

## 1.1 Python Object-Model Lowering Path

In addition to the C++ pass pipelines, the active Python compiler exposes a
hardware-free object-model lowering path through `compile_graph_module`:

```text
GraphIRModule
  -> ScheduleIRModule
  -> TileIRModule
  -> TargetIRModule / RuntimeArtifact
```

This path constructs IR objects directly, runs verifier checks before textual
artifact emission, and records graph/schedule/tile/target hashes in compile
bundle metadata. It supports CPU/x86, NVIDIA, Apple CPU/GPU, and ROCm targets.
Native execution is available for supported x86/CPU and Apple CPU/GPU rows,
for proven CUDA `nvidia_sm120` rows, and for proven HIP `rocm_gfx1151` rows.
Unsupported dtype/layout/shape combinations and unproven sibling architectures
remain explicit artifact or reference-fallback cases.

Debug switches:

| Variable | Behavior |
|----------|----------|
| `TESSERA_DEBUG_IR=1` | Write Graph/Schedule/Tile/Target artifact text. |
| `TESSERA_DUMP_STATE=1` | Write compile metadata and Chrome trace JSON. |
| `TESSERA_DUMP_DIR=...` | Select artifact output root. |

These switches are developer-tool contracts, not semantic lowering passes.

---

## 2. Named Pipelines

Pipeline status is split by behavior type:

| Pipeline | Semantic compiler behavior | Target artifact generation | Mock/runtime fallback | Native hardware runtime |
|----------|----------------------------|----------------------------|-----------------------|-------------------------|
| `tessera-lower-to-x86` | implemented / lit-testable | x86/CPU call artifacts | NumPy-backed CPU fallback for supported ops | AMX/AVX-512 hardware-runtime tests cover supported paths |
| `tessera-lower-to-gpu` | implemented / lit-testable | NVIDIA target artifacts | explicit artifact/reference fallback outside the native envelope | native CUDA execution for supported `nvidia_sm120` rows; other SM-specific paths remain separately proven |
| `tessera-lower-to-rocm` | implemented / lit-testable | ROCm Target IR / HSACO contracts | explicit artifact/reference fallback outside the native envelope | native HIP execution for supported `rocm_gfx1151` rows; CDNA and other RDNA targets remain separately proven |

### 2.1 `tessera-lower-to-x86`

Registered pass sequence (executed in this order):

1. `tessera-effect-annotation`
2. `tessera-canonicalize`
3. `tessera-distribution-lowering`
4. `tessera-tiling`
5. `tessera-tile-to-x86`

### 2.2 `tessera-lower-to-gpu`

Registered pass sequence (executed in this order):

1. `tessera-effect-annotation`
2. `tessera-canonicalize`
3. `tessera-distribution-lowering`
4. `tessera-tile-ir-lowering`
5. `tessera-warp-specialization`
6. `tessera-async-copy-lowering`
7. `tessera-nvwgmma-lowering`
8. `tessera-nvtma-descriptor`
9. `tessera-nvflash-attn-emitter`

---

## 3. Pass Specifications

Each pass entry covers: purpose, CLI flag, input IR contract, output IR contract, invariants, pass options, and an IR before/after example.

---

### 3.1 `EffectAnnotationPass`

**File:** `src/transforms/lib/EffectAnnotationPass.cpp`  
**CLI flag:** `--tessera-effect-annotation`  
**Pipeline position:** 1 (both pipelines)

#### Purpose

Infers the side-effect class of each `func.func` in the module and attaches `tessera.effect` as a string function attribute. This annotation is consumed downstream by `DistributionLoweringPass` (collective insertion) and by Python-side `@jit(deterministic=True)` validation.

#### Input IR contract

- Valid `ModuleOp` containing `func.func` operations.
- `tessera.*` ops may appear in function bodies.
- Some functions may already have a `tessera.effect` attribute (set by `GraphIRBuilder` for `deterministic=True` functions).

#### Output IR contract

- Every `func.func` in the module has a `tessera.effect` string attribute.
- Attribute value is one of: `"pure"`, `"random"`, `"movement"`, `"state"`, `"collective"`, `"memory"`, `"io"`.
- No ops are modified or reordered.

#### Effect inference rules

Applied in order; the highest-level effect found wins (lattice join):

| Condition in function body | Effect level |
|---------------------------|--------------|
| `tessera.flash_attn` with `dropout_p` attr present and `!= 0.0` | `random` |
| `tessera.copy` op | `memory` |
| `schedule.prefetch`, `schedule.async_copy`, `schedule.await_movement`, `tile.async_copy`, or `tile.wait_async` | `movement` |
| `tessera.kv_cache.*`, `tessera.ring.*`, `cache.*`, or `ring.*` | `state` |
| `tessera_collective.*` or Graph-level `tessera.{all_reduce,reduce_scatter,all_gather,all_to_all}` | `collective` |
| `rng.uniform` or `tessera.rng.*` | `random` |
| Any argument with `tessera.effect = "write"` or `"reduce_*"` attribute | `memory` |
| `func.call` to external non-tessera function | `io` |
| None of the above | `pure` |

The `tessera_collective` Target dialect also owns the rank-local one-sided
sequence `window.register` → `put_signal`/`signal`/`wait_signal` →
`window.deregister`. These operations carry an SSA window resource and exact
buffer extent, dtype, peer, offset, signal-index, and RMA-context attributes.
They are legal only inside an RCCL `gin_rma` artifact with symmetric windows,
strict ordering, a matching communicator digest, host-RMA support, and a
nonzero GIN type. Package verification rejects duplicate registration,
use-before-registration, and leaked windows before runtime dispatch.

#### Invariants

- **Pre-condition:** `tessera.effect` attribute, if already set, must equal `"pure"`. If it is set to any other value, the pass treats it as an override and skips inference for that function.
- **Post-condition:** If a function's body infers an effect level higher than `"pure"` but the function already carries `tessera.effect = "pure"`, the pass emits an error and signals pipeline failure. This enforces the `@jit(deterministic=True)` contract.
- The pass does not modify any ops — it is annotation-only.

#### IR example

**Before:**
```mlir
func.func @stable_fwd(%x: tensor<128x256xbf16>) -> tensor<128x256xbf16> {
  %r = tessera.matmul %x, %x : (tensor<128x256xbf16>, tensor<128x256xbf16>) -> tensor<128x256xf32>
  return %r : tensor<128x256xf32>
}
```

**After:**
```mlir
func.func @stable_fwd(%x: tensor<128x256xbf16>) -> tensor<128x256xbf16>
    attributes {tessera.effect = "pure"} {
  %r = tessera.matmul %x, %x : (tensor<128x256xbf16>, tensor<128x256xbf16>) -> tensor<128x256xf32>
  return %r : tensor<128x256xf32>
}
```

---

### 3.2 `CanonicalizeTesseraIRPass`

**File:** `src/transforms/lib/CanonicalizeTesseraIR.cpp`  
**CLI flag:** `--tessera-canonicalize`  
**Pipeline position:** 2 (both pipelines)

#### Purpose

Applies four greedy rewrite patterns to simplify and fuse Graph IR ops. Runs `applyPatternsAndFoldGreedily` — patterns may fire repeatedly until fixed point.

#### Input IR contract

- Valid tessera dialect ops in `func.func` bodies.
- `tessera.effect` attributes already set (by pass 3.1).

#### Output IR contract

- All `tessera.transpose → tessera.matmul` chains replaced by `tessera.matmul` with `transposeA`/`transposeB` flags.
- All `tessera.matmul → tessera.add → tessera.gelu` chains replaced by `tessera.fused_epilogue {Gelu}`.
- All `tessera.conv2d_nhwc → tessera.relu` chains replaced by `tessera.conv2d_nhwc {epilogue=Relu}`.
- All `tessera.flash_attn` ops with `dropout_p = 0.0` have the `dropout_p` attribute removed.
- No `tessera.transpose` ops remain whose consumers are `tessera.matmul`.

#### Patterns (see `GRAPH_IR_SPEC.md §5` for full details)

| Pattern | Benefit | Match | Result |
|---------|---------|-------|--------|
| `FuseMatmulBiasGELU` | 2 | `gelu(add(matmul(A,B), bias))` | `fused_epilogue(A,B,bias, Gelu)` |
| `FuseConvRelu` | 2 | `relu(conv2d_nhwc(...))` | `conv2d_nhwc(..., epilogue=Relu)` |
| `DropoutZeroSimplify` | 1 | `flash_attn {dropout_p=0.0}` | `flash_attn` without `dropout_p` |
| `TransposeIntoMatmul` | 1 | `matmul(transpose(A), B)` or `matmul(A, transpose(B))` | `matmul(A, B, transposeA/B=true)` |

#### Invariants

- Idempotent: running the pass twice produces the same result.
- Does not change the mathematical semantics of any op.

---

### 3.3 `DistributionLoweringPass`

**File:** `src/transforms/lib/DistributionLoweringPass.cpp`  
**CLI flag:** `--tessera-distribution-lowering`  
**Pass options:** `--mesh-axes=<str>`, `--mesh-sizes=<str>`  
**Pipeline position:** 3 (both pipelines)

#### Purpose

Converts `tessera.shard` argument attributes on `func.func` arguments into `schedule.mesh.define` + `schedule.mesh.region` ops that wrap the function body. Bridges from Graph IR (tessera dialect) to Schedule IR (schedule dialect).

#### Pass options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mesh-axes` | `string` | `""` | Comma-separated mesh axis names, e.g. `"dp,tp"`. Overrides per-arg shard attrs. |
| `--mesh-sizes` | `string` | `""` | Comma-separated axis sizes matching `--mesh-axes`, e.g. `"4,2"`. |

If a function has no `tessera.shard` attributes and no pass options are provided, the function is left unchanged.

#### Input IR contract

- `func.func` arguments may carry `tessera.shard = {axes = [...], dims = [...]}` attributes (set by `GraphIRBuilder`).

#### Output IR contract

- `tessera.shard` attributes removed from all function arguments.
- `schedule.mesh.define` emitted at the top of the function body.
- Original function body wrapped in `schedule.mesh.region` with a `schedule.yield` terminator.
- Function body ops unchanged.

#### Invariants

- Only processes functions with at least one `tessera.shard` argument attribute (or explicit pass options).
- Does not modify the ops inside the mesh region.
- The `schedule.mesh.define` dims and axis_names must reflect all unique axes found in the function's shard attributes.

#### IR example

**Before:**
```mlir
func.func @step(
    %a: tensor<128x256xbf16> {tessera.shard = {axes = ["dp"], dims = [0]}}
) attributes {tessera.effect = "memory"} {
  %0 = tessera.matmul %a, %a : (tensor<128x256xbf16>, tensor<128x256xbf16>) -> tensor<128x256xf32>
  return
}
```

**After:**
```mlir
func.func @step(%a: tensor<128x256xbf16>) attributes {tessera.effect = "memory"} {
  schedule.mesh.define {dims = [4], axis_names = ["dp"]}
  schedule.mesh.region {mesh = @dp, axis = "dp"} {
    %0 = tessera.matmul %a, %a : (tensor<128x256xbf16>, tensor<128x256xbf16>) -> tensor<128x256xf32>
    schedule.yield
  }
  return
}
```

---

### 3.4 `TilingPass`

**File:** `src/transforms/lib/TilingPass.cpp`  
**CLI flag:** `--tessera-tiling`  
**Pass options:** `--tile-m=<int>`, `--tile-n=<int>`, `--tile-k=<int>`
**Pipeline position:** 4 (x86 pipeline only)

#### Purpose

Tiles `tessera.matmul` ops into `scf.for` loop nests over M, N, and the K
reduction. The K loop carries an FP32 or INT32 accumulator and an SSA pipeline
state. Static logical shapes are zero-padded to physical tile multiples, making
ragged M/N/K tails explicit and every physical slice in-bounds. Dynamic,
transposed, or unsupported accumulator forms are left unchanged for a later
target-specific gate.

#### Pass options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--tile-m` | `int` | `16` | M-dimension tile size (rows per outer loop step). |
| `--tile-n` | `int` | `16` | N-dimension tile size (cols per outer loop step). |
| `--tile-k` | `int` | `16` | K-reduction tile size (lanes per reduction step). |

#### Input IR contract

- `tessera.matmul` ops in function bodies (inside or outside `schedule.mesh.region`).
- Operands must be non-transposed, statically-shaped rank-2 tensors.
- The result element type must be FP32 or INT32.

#### Output IR contract

- Each `tessera.matmul %A, %B : tensor<MxKxeT>, tensor<KxNxeT> -> tensor<MxNxeT>` replaced by:
  ```mlir
  %init = arith.constant dense<0> : tensor<MpadxNpadxaccT>
  %Cpad = scf.for %i = 0 to Mpad step tile_m iter_args(%acc0 = %init) {
    %C1 = scf.for %j = 0 to Npad step tile_n iter_args(%acc1 = %acc0) {
      %state = tile.pipeline_init {...} : !tile.pipeline_state
      %C2, %next_state = scf.for %k = 0 to Kpad step tile_k
          iter_args(%acc2 = %acc1, %s = %state)
          -> (tensor<MpadxNpadxaccT>, !tile.pipeline_state) {
        %a_sl = tensor.extract_slice %Apad[%i, %k][tile_m, tile_k][1, 1]
        %b_sl = tensor.extract_slice %Bpad[%k, %j][tile_k, tile_n][1, 1]
        %partial = tessera.matmul %a_sl, %b_sl
        %next = tessera.add %old, %partial
        %acc3 = tensor.insert_slice %next into %acc2[...]
        %s2 = tile.pipeline_advance %s, %next
        scf.yield %acc3, %s2
      }
      scf.yield %C2
    }
    scf.yield %C1
  }
  %C = tensor.extract_slice %Cpad[0, 0][M, N][1, 1]
  ```
- `tessera.matmul` ops with dynamic shapes are left unchanged.
- Ops other than `tessera.matmul` (for example `tessera.flash_attn` and
  `tessera.fused_epilogue`) are untouched by the canonical GEMM expansion.
  `TileIRLoweringPass` owns the separate FlashAttention KV-loop expansion.

#### Invariants

- All statically-shaped `tessera.matmul` ops have been expanded into tiled loops.
- Physical tile sizes divide padded dimensions; zero padding is the explicit
  boundary policy for ragged M/N/K.
- Inner matmul steps carry `tessera.tile_m/n/k`,
  `tessera.canonical_k_step`, and `tessera.ragged_zero_pad`.

---

### 3.5 `TileToX86Pass`

**File:** `src/transforms/lib/TileToX86Pass.cpp`  
**CLI flag:** `--tessera-tile-to-x86`  
**Pass options:** `--prefer-amx=<bool>`  
**Pipeline position:** 5 (x86 pipeline only)

#### Purpose

Replaces `tessera.matmul` and `tessera.fused_epilogue` ops (with static BF16/F16 input tensors) with `func.call` to tessera x86 backend C functions. This is the final x86 lowering step that produces callable code against the pre-built x86 AMX/AVX-512 GEMM kernels.

#### Pass option

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--prefer-amx` | `bool` | `true` | If `true`, emit `tessera_x86_amx_gemm_bf16`. If `false`, always emit `tessera_x86_avx512_gemm_bf16`. |

#### x86 backend C functions called

| Function | Signature | Description |
|----------|-----------|-------------|
| `tessera_x86_amx_gemm_bf16` | `(i64 aPtr, i64 bPtr, i64 cPtr, i64 M, i64 N, i64 K, f32 beta)` | AMX BF16 GEMM |
| `tessera_x86_avx512_gemm_bf16` | same | AVX-512 BF16 GEMM fallback |
| `tessera_x86_epilogue_bias_fp32` | `(i64 cPtr, i64 biasPtr, i64 M, i64 N)` | Bias add |
| `tessera_x86_epilogue_bias_gelu_fp32` | same | Bias add + GELU |

#### Input IR contract

- `tessera.matmul` ops with static ranked BF16/F16 input tensors (typically tiled by `TilingPass`).
- `tessera.fused_epilogue` ops with static shapes.

#### Output IR contract

For each `tessera.matmul %A, %B : tensor<MxKxbf16>, tensor<KxNxbf16> -> tensor<MxNxf32>`:

1. `bufferization.to_memref %A` → `memref<MxKxbf16>`
2. `bufferization.to_memref %B` → `memref<KxNxbf16>`
3. `memref.alloc()` → `memref<MxNxf32>`
4. External C function declaration added to the module (once per unique function name).
5. `memref.extract_aligned_pointer_as_index` + `arith.index_cast` to extract raw `i64` pointers.
6. `func.call @tessera_x86_amx_gemm_bf16(aPtr, bPtr, cPtr, M, N, K, beta)`
7. `bufferization.to_tensor %C_buf` → `tensor<MxNxf32>`

For `tessera.fused_epilogue`: same GEMM lowering, followed by the appropriate epilogue C function call.

#### Invariants

- After this pass, no `tessera.matmul` or `tessera.fused_epilogue` ops remain for static BF16/F16 types.
- All required external C function declarations are present exactly once in the module.

---

### 3.6 `TileIRLoweringPass`

**File:** `src/transforms/lib/TileIRLoweringPass.cpp`  
**CLI flag:** `--tessera-tile-ir-lowering`  
**Pass options:** `--tile-q=<int>`, `--tile-kv=<int>`, `--sm=<int>`  
**Pipeline position:** 4 (GPU pipeline only)

#### Purpose

Lowers `schedule.mesh.region` bodies containing `tessera.flash_attn` into FA-4 Tile IR ops. Also handles `tessera.matmul` inside `mesh.region` bodies by emitting the `tile.async_copy` + `tile.mma` + `tile.wait_async` GPU tiling sequence.

#### Pass options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--tile-q` | `int` | `64` | Q tile rows. Must match the GPU WGMMA tile granularity. |
| `--tile-kv` | `int` | `64` | KV tile cols. |
| `--sm` | `int` | `90` | Target SM version as integer (e.g. `90` for SM_90). Controls whether `CausalMaskOp` and `DropoutMaskOp` are emitted. |

#### Input IR contract

- `schedule.mesh.region` bodies containing `tessera.flash_attn` ops.
- `tessera.flash_attn` ops may carry `tessera.tile_q`, `tessera.tile_kv`, and `causal` attributes.
- `tessera.matmul` ops inside mesh regions.

#### Output IR contract

For a statically shaped rank-2 `tessera.flash_attn(Q, K, V)`:

```mlir
%q_tile, %q_token = tile.async_copy %Q ... {layout = #tile.layout<...>}
%q_ready = tile.wait_async %q_token
%producer = tile.pipeline_init {role = "producer", phase = 1}
%consumer = tile.pipeline_init {role = "consumer", phase = 0}
%acc0 = arith.constant dense<0.0> : tensor<QxDvxf32>
%m0 = arith.constant dense<0xFF800000> : tensor<Qxf32>
%l0 = arith.constant dense<0.0> : tensor<Qxf32>

%acc, %m, %l, %producer_final, %consumer_final, %boundary_final =
  scf.for %kv = %c0 to %padded_sk step %tile_kv
      iter_args(%acc_i = %acc0, %m_i = %m0, %l_i = %l0,
                %p_i = %producer, %c_i = %consumer,
                %boundary_i = %c0) {
    %k_slice = tensor.extract_slice %K[%kv, 0] ...
    %v_slice = tensor.extract_slice %V[%kv, 0] ...
    %k_tile, %kt = tile.async_copy %k_slice, %kv, %c0 ...
        {coordinate_count = 2, layout = #tile.layout<...>}
    %v_tile, %vt = tile.async_copy %v_slice, %kv, %c0 ...
        {coordinate_count = 2, layout = #tile.layout<...>}
    %deps = tile.wait_async %kt, %vt
    %p_next = tile.pipeline_advance %p_i, %deps
    %scores = tessera_attn.scaled_dot_product %q_tile, %k_tile ...
    %bounded = tessera_attn.boundary_mask %scores, %c0, %boundary_i
        {causal = true, window_left = -1, window_right = -1}
    %acc_next, %m_next, %l_next =
        tessera_attn.streaming_update %bounded, %v_tile,
            %m_i, %l_i, %acc_i
    %c_next = tile.pipeline_advance %c_i, %acc_next
    %next_boundary = arith.addi %boundary_i, %tile_kv
    scf.yield %acc_next, %m_next, %l_next,
              %p_next, %c_next, %next_boundary
  }

%output, %lse = tessera_attn.lse_accumulate %acc, %m, %l
```

The copy descriptor retains the unpadded logical K/V source extent. Dynamic
slice coordinates are explicit index operands, so the target copy consumer can
zero-fill the final ragged KV block without inventing a computed base pointer.
`boundary_mask` carries causal and sliding-window policy with absolute Q/KV
offsets. Non-zero dropout uses `block_dropout`, keyed by the absolute KV offset
**and a per-instance `stream_offset`**, and requires an explicit seed. The
stream offset is the second counter axis: a rank-4 attention is distributed
into `B*H` rank-2 instances whose KV offsets each restart at 0, so without it
every instance would replay one identical mask (Decision #18). The producer
passes `(b*H + h) * Sq * Sk_padded` — a disjoint counter block per instance —
and an attention that was never distributed passes 0.

For `tessera.matmul` inside a mesh region:
```mlir
%a_tile = tile.async_copy %A {tile_rows = 64, tile_cols = 64}
%b_tile = tile.async_copy %B {tile_rows = 64, tile_cols = 64}
tile.wait_async
%c_tile = tile.mma %a_tile, %b_tile : ...
```

#### Invariants

- Supported static rank-2 `tessera.flash_attn` ops inside
  `schedule.mesh.region` are fully replaced by a KV-block `scf.for` carrying
  output accumulation, running maximum, running normalization sum,
  producer/consumer pipeline state, and boundary offset.
- Rank-4 distributed batch/head attention and dynamic shapes are retained
  fail-closed for a later distribution-owned lowering; they are not silently
  represented by the rank-2 contract.
- Deterministic NVIDIA backward launch descriptors must declare
  `workspace_owner = "output_element"`; split-workspace ownership remains an
  architecture-owned runtime contract.
- `tessera.matmul` ops inside `schedule.mesh.region` are fully replaced by `tile.async_copy` + `tile.mma`.
- `tessera.flash_attn` ops outside `schedule.mesh.region` are left unchanged (handled differently — should not exist after `DistributionLoweringPass`).

---

### 3.7 `WarpSpecializationPass`

**File:** `src/compiler/tile_opt_fa4/lib/WarpSpecializationPass.cpp`  
**CLI flag:** `--tessera-warp-specialization`  
**Pipeline position:** 5 (GPU pipeline only)

#### Purpose

Assigns producer/consumer warp roles to the FA-4 Tile IR ops and synchronizes
the roles through `!tile.pipeline_state` + `!tile.async_token` SSA chains
(`tile.pipeline_init` / `tile.pipeline_advance`). This is required for the
WGMMA warp specialization model on SM_90+.

> **Spec correction (2026-08-10).** Earlier revisions of this section claimed
> the pass inserts `tessera.queue.create/push/pop` ops. It never did; the
> `tessera.queue` MLIR dialect was deleted under Decisions #29/#31. The
> pipeline-state SSA contract below is what the pass actually emits.

#### Input IR contract

- FA-4 Tile IR ops (`tile.async_copy`, `tile.wait_async`, `tessera_attn.*`, `tile.mma`) inside function bodies.

#### Output IR contract

- Function body split into `tessera.schedule.warp` regions stamped with
  `tile.warp_role = "producer"` / `"consumer"` and a shared `tile.pipeline` id.
- Each warp region owns a `tile.pipeline_init` SSA value
  (`!tile.pipeline_state`, with `depth`/`stage`/`phase`/`role` attributes);
  cross-role ordering flows through `tile.pipeline_advance` operands.
- `tile.async_copy` and `tile.wait_async` ops enclosed in the `producer` region.
- `tessera_attn.*` compute ops and `tile.mma` ops enclosed in the `consumer` region.

#### Key design contract

Warp role separation is **structural, not advisory**. The backend allocates separate register files and barrier (mbarrier) slots per role. Producer warps are dedicated to TMA prefetch; consumer warps run WGMMA MMA ops. They never execute the other role's code.

#### IR example

**Before:**
```mlir
%q_tile = tile.async_copy %Q {tile_rows = 64, tile_cols = 64}
tile.wait_async
%scores = tessera_attn.scaled_dot_product %q_tile, %k_tile scale = 0.125 : ...
```

**After:**
```mlir
tessera.schedule.warp {tile.warp_role = "producer", tile.pipeline = "pipe0"} {
  %ps = tile.pipeline_init {depth = 2, stage = 0, phase = 0, role = "producer"} : !tile.pipeline_state
  %q_tile, %tok = tile.async_copy %Q {tile_rows = 64, tile_cols = 64}
  %ps1 = tile.pipeline_advance %ps, %tok : !tile.pipeline_state
}
tessera.schedule.warp {tile.warp_role = "consumer", tile.pipeline = "pipe0"} {
  %ps = tile.pipeline_init {depth = 2, stage = 0, phase = 1, role = "consumer"} : !tile.pipeline_state
  %scores = tessera_attn.scaled_dot_product %q_tile, %k_tile scale = 0.125 : ...
  %ps1 = tile.pipeline_advance %ps, %scores : !tile.pipeline_state
}
```

---

### 3.8 `AsyncCopyLoweringPass`

**File:** `src/compiler/tile_opt_fa4/lib/AsyncCopyLoweringPass.cpp`  
**CLI flag:** `--tessera-async-copy-lowering`  
**Pipeline position:** 6 (GPU pipeline only)

#### Purpose

Lowers `tile.async_copy` ops to either TMA descriptor-based async copies (SM_90+) or `tessera.cp_async.*` ops (SM_80/86/89). The target SM version is read from the `tessera.target_sm` module attribute set by `@jit(target=GPUTargetProfile(...))`.

#### Input IR contract

- `tile.async_copy` ops inside warp-specialized producer regions.

#### Output IR contract

**For SM_90+:**
```mlir
tessera.tma.async_copy %descriptor, %smem_buf, %mbarrier : ...
```

**For SM_80/86/89:**
```mlir
tessera.cp_async.cg %smem_buf, %gmem_ptr {size = 16} : ...
```

#### Invariants

- All `tile.async_copy` ops are replaced.
- `tile.wait_async` ops are replaced by appropriate barrier ops (`tessera.tma.wait_async` or `tessera.cp_async.wait_group`).

---

### 3.9 `NVWGMMALoweringPass`

**File:** `src/compiler/codegen/tessera_gpu_backend_NVIDIA/NVWGMMALoweringPass.cpp`  
**CLI flag:** `--tessera-nvwgmma-lowering`  
**Pipeline position:** 7 (GPU pipeline only)

#### Purpose

Lowers `tile.mma` ops to WGMMA (Warpgroup Matrix Multiply Accumulate) inline PTX (`tessera.nvgpu.wgmma.mma_async`) for SM_90+, or falls back to legacy WMMA for SM_80/86/89.

#### Input IR contract

- `tile.mma` ops inside warp-specialized consumer regions.
- `tessera.target_sm` module attribute present.

#### Output IR contract

**SM_90+ (WGMMA):**
```mlir
tessera.nvgpu.wgmma.mma_async %a_desc, %b_desc, %c_acc
    {m = 64, n = 64, k = 16, dtype = "bf16"} : ...
```

**SM_80/86/89 (WMMA fallback):**
```mlir
tessera.nvgpu.wmma.mma %a_frag, %b_frag, %c_frag
    {m = 16, n = 16, k = 16, dtype = "bf16"} : ...
```

#### Invariants

- All `tile.mma` ops replaced by hardware-specific MMA intrinsics.
- SM version gating is strict: no WGMMA ops emitted when `tessera.target_sm < 90`.

---

### 3.10 `NVTMADescriptorPass`

**File:** `src/compiler/codegen/tessera_gpu_backend_NVIDIA/NVTMADescriptorPass.cpp`  
**CLI flag:** `--tessera-nvtma-descriptor`  
**Pipeline position:** 8 (GPU pipeline only)

#### Purpose

Hoists TMA descriptor setup to the kernel preamble and assigns mbarrier slots. TMA descriptors describe how global memory tiles are staged into shared memory. They must be constructed once per kernel launch (not once per tile loop iteration).

#### Key design contract

TMA descriptors are generated **once per kernel**, not once per tile. `cp.async.bulk.tensor` calls in the tile loop reference the descriptor; they do not rebuild it.

#### Input IR contract

- `tessera.tma.async_copy` ops referencing tensor operands.

#### Output IR contract

- TMA descriptor setup ops hoisted to kernel preamble.
- `cp.async.bulk.tensor.1d` (or 2d/3d) emitted in tile loop.
- `mbarrier.init`, `mbarrier.arrive`, `mbarrier.wait` sequences inserted at correct points.

**Kernel preamble example:**
```mlir
// Preamble (hoisted by NVTMADescriptorPass)
%q_desc = tessera.tma.make_descriptor %Q_global {tile_shape = [64, 64]} : ...
%mbar_0 = tessera.mbarrier.init {count = 1} : ...

// In tile loop (after hoisting)
tessera.tma.bulk_copy %q_desc, %smem_q, %mbar_0 : ...
tessera.mbarrier.arrive %mbar_0 : ...
tessera.mbarrier.wait %mbar_0 {phase = 0} : ...
```

---

### 3.11 `NVFlashAttnKernelEmitter`

**File:** `src/compiler/codegen/tessera_gpu_backend_NVIDIA/NVFlashAttnKernelEmitter.cpp`  
**CLI flag:** `--tessera-nvflash-attn-emitter`  
**Pipeline position:** 9 (GPU pipeline only)

#### Purpose

Finalises the FA-4 kernel. Resolves the attention scale sentinel (replaces the `1/sqrt(D)` placeholder with the concrete float value), emits the full mbarrier arrive/wait sequence for double-buffering, and attaches CUDA launch bounds as `nvvm.maxntidx` attributes.

#### Input IR contract

- Full warp-specialized, descriptor-hoisted FA-4 kernel with `tessera_attn.*` ops.
- `tessera.flash_attn` `scale = -1.0` sentinel value indicating "auto-compute from head_dim".

#### Output IR contract

- `scale` sentinel resolved to concrete `1 / sqrt(head_dim)` float constant.
- Complete mbarrier arrive/wait synchronisation sequence present for all double-buffer stages.
- `nvvm.maxntidx` annotation attached to the kernel function: `warps_per_cta * 32` threads.
- `nvvm.kernel` attribute set on the function to mark it as a CUDA kernel entry point.

#### Invariants

- No `tessera_attn.scaled_dot_product` ops remain with a sentinel scale value.
- All FA-4 attn ops are enclosed in a complete mbarrier synchronisation region.
- The emitted kernel is directly translatable to PTX by LLVM's NVPTX backend.

---

## 4. Pass Ordering Constraints

The following ordering constraints are hard requirements (violating them produces incorrect IR):

| Constraint | Reason |
|-----------|--------|
| `EffectAnnotation` before `DistributionLowering` | Distribution pass reads `tessera.effect` to identify gradient tensors for collective insertion. |
| `CanonicalizeTesseraIR` before `TilingPass` | Transpose flags must be folded before tiling to avoid tiling transposed ops incorrectly. |
| `DistributionLowering` before `TileIRLowering` | Tile IR lowering operates on `schedule.mesh.region` bodies — these must exist before Tile IR lowering. |
| `TileIRLowering` before `WarpSpecialization` | Warp specialization assigns roles to tile ops — these ops don't exist until after Tile IR lowering. |
| `WarpSpecialization` before `AsyncCopyLowering` | Async copy lowering converts `tile.async_copy` inside producer regions — the regions must exist first. |
| `AsyncCopyLowering` before `NVTMADescriptor` | TMA descriptor hoisting operates on `tessera.tma.async_copy` ops — these are emitted by async copy lowering. |
| `NVWGMMALowering` before `NVFlashAttnKernelEmitter` | Kernel emitter finalises mbarrier slots, which depend on the WGMMA op structure. |

---

## 5. IR Layer Transitions Summary

| Transition | From | To | Pass |
|-----------|------|----|------|
| Graph IR → Graph IR + effects | `tessera.*` | `func.func {tessera.effect}` | `EffectAnnotationPass` |
| Graph IR → canonicalised Graph IR | `tessera.*` chains | fused `tessera.*` | `CanonicalizeTesseraIRPass` |
| Graph IR → Schedule IR | `tessera.shard` attrs | `schedule.mesh.*` | `DistributionLoweringPass` |
| Schedule IR → Tiled Graph IR (x86) | `tessera.matmul` | `scf.for + tensor.*` | `TilingPass` |
| Tiled Graph IR → x86 calls | `tessera.matmul/fused_epilogue` | `func.call @tessera_x86_*` | `TileToX86Pass` |
| Schedule IR → Tile IR (GPU) | `schedule.mesh.region + tessera.flash_attn` | `tile.* + tessera_attn.*` | `TileIRLoweringPass` |
| Tile IR → Warp-specialised IR | `tile.*` | `tessera.schedule.warp + tessera.queue.*` | `WarpSpecializationPass` |
| Warp IR → TMA/cp.async IR | `tile.async_copy` | `tessera.tma.*` or `tessera.cp_async.*` | `AsyncCopyLoweringPass` |
| TMA IR → WGMMA PTX IR | `tile.mma` | `tessera.nvgpu.wgmma.*` | `NVWGMMALoweringPass` |
| WGMMA IR → kernel IR | `tessera.tma.*` scattered | hoisted descriptors + mbarriers | `NVTMADescriptorPass` |
| Kernel IR → final PTX-ready IR | sentinel scale, partial barriers | concrete values + full mbarrier seq | `NVFlashAttnKernelEmitter` |
