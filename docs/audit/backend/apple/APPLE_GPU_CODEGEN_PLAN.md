# Apple GPU Codegen Plan — Dispatcher → Compiler

Status surface for growing the Apple GPU lane from a **runtime dispatcher**
(pattern-match op names → MPS / MPSGraph / hand-written MSL) into a real
**codegen path** (emit Metal from IR), the GPU analogue of the `tessera_jit` CPU
lane. Grounded in the Apple "Metal Feature Set Tables" PDF (rev 2026-05-21) — see
the `apple7-m1max-gpu-feature-set` memory and Decision #27.

Last updated: 2026-06-15.

## Where we are (the foundation is strong)

Three things are already real on the Apple GPU lane, so this is *growth*, not a
green field:

1. **`compiler/fusion.py` is a mature MSL synthesizer** — it emits Metal Shading
   Language from a `FusedRegion` (matmul root + pointwise epilogue + optional
   reduction) and runs it through the runtime's `compile_msl_kernel`
   (`newLibraryWithSource:`). It already has: a stack variant (N≤1024), a
   threadgroup-tiled variant (N≤8192), a **`simdgroup_matrix` (coopmat) MMA**
   variant — the Apple7 matrix-unit path — and f16/f32 I/O. Oracle-gated
   (`verify_synthesized_region`) and autotuned (`best_variant_for`).
2. **The runtime executes** via MPS / MPSGraph + ~30 hand-written fused MSL
   kernels (incl. the streaming online-softmax `kFlashAttnF32Source`).
3. **The seam is closed** for fusion dispatch (Phase 0a–0c): the executor
   consumes the compiler's fusion decision (carried `dispatch` roles) instead of
   re-matching op names.

## Apple7 (M1 Max) hardware grounding — what the silicon actually supports

From the feature-set PDF (`apple7-m1max-gpu-feature-set` memory):
- **`simdgroup_matrix` MMA** (SIMD-scoped matrix multiply) — the GPU GEMM
  primitive. `fusion.py`'s coopmat variant already targets it.
- **Native `bfloat`** (`MTLDataType.bfloat`, Apple6+) — bf16 is native on the GPU,
  unlike the M1 *CPU* (NEON, ARMv8.5) where we emulate it via f32.
- **Metal 4 Tensors + ML encoding** (Apple7) — MTLTensor formats expose the whole
  low-precision zoo: **FP8** (`MetalFloat8E5M2`/`E4M3`), **FP4** (`MetalFloat4E2M1`),
  **MX block-scale** (`MetalFloat8UE8M0`), plus int2/4/8/16/32, f16, bf16, f32.
  Tessera's planned/gated `fp8_e4m3`/`fp8_e5m2`/`fp4_e2m1`/`nvfp4` dtypes map
  directly onto these — so they have a **real execution path on this Mac**, not
  the NVIDIA-only frontier they were assumed to be.

## Milestones

### M1 — Native bf16 ✅ landed (2026-06-15)
The synthesizer emulated bf16 by host-upcasting to f32 even though Apple7
supports native `bfloat`. Now: `_io_type("bf16") → "bfloat"`, a
`_run_fused_region_bf16` that emits native `bfloat` MSL and reuses the f16 synth
symbol's raw-uint16 ABI (bf16 is 2-byte; the MSL `bfloat` type gives the bits
meaning), fp32 accumulators inside, with a clean fallback to f32-emulation if the
runtime's MSL predates `bfloat`. **Verified on M1 Max**: native bfloat compiles +
runs (`metal_runtime`), bf16-precision-correct. Guard:
`tests/unit/test_fusion_bf16_native.py`. *Follow-on:* extend the coopmat
(simdgroup_matrix) variant to native bf16 (MSL `simdgroup_matrix<bfloat>`).

### M2 — Generalize the synthesizer beyond matmul-epilogue (HF)
- **`norm_chain` region** ✅ **landed (2026-06-15).** `norm(x [+ residual])` —
  the pre-norm transformer pattern (`rmsnorm(x + attn_out)`) — fused into ONE
  synthesized MSL kernel (non-matmul-rooted). `NormChainRegion` + a new
  `tessera_apple_gpu_synth_norm_chain_f32` runtime symbol (one row/thread, X(0)
  O(1) M(2) N(3) residual(4); fp32 accumulators); reuses the `REDUCTION_OPS`
  reduction blocks verbatim (and **added `layer_norm`** to them). Discoverer
  `discover_norm_chain_regions` is a **norm-centric scan** — from each norm it
  walks back to a single-use residual add and forward to the post-norm affine
  (`mul(γ) → add(β)`), wired into `_apple_gpu_try_synthesized_fusion` (skips ops a
  matmul-epilogue region claimed; a bare norm stays on the MPSGraph rowop lane).
  **Three sub-pieces landed (2026-06-15), all verified on M1 Max:**
  - **Residual fusion** — `norm(x + residual)` fused; `metal_runtime`, fp32-exact.
  - **f16/bf16 (dtype breadth)** — a uint16 symbol variant (one symbol serves
    `half` + native `bfloat`). Both run **native** (`metal_runtime`). This drove
    two runtime fixes (2026-06-15): (a) **MSL compile errors are now surfaced** —
    `compile_msl_kernel` records the `newLibraryWithSource` `NSError` via the
    last-error channel instead of silently returning nil; (b) the surfaced error
    revealed `bfloat` rejects *implicit* float→bfloat assignment (unlike `half`),
    so every synthesized O-write now goes through an explicit `ST(...)` cast
    (a `using ST = <io>;` alias) across `REDUCTION_OPS`, the matmul-epilogue
    stack+tiled kernels, and norm_chain; and `compile_msl_kernel` now targets
    **MSL 3.1** (the `bfloat` floor, was 3.0). Net: bf16 matmul-epilogue was
    *silently f32-emulated* before — it (and norm_chain) is now genuinely native.
  - **Post-norm affine** — `norm(...) * γ [+ β]` (the real transformer
    RMSNorm/LayerNorm with affine). γ/β are appended per-feature buffers (5/6);
    applied in a second pass reusing the reduction block verbatim. Llama
    `rmsnorm(x+r)*γ` fires end-to-end from `@jit`.

  Guards: `tests/unit/test_fusion_norm_chain.py` (30). *Follow-on:* native bf16
  norm_chain; multi-residual (`x + r1 + r2`) pre-op chains.
- **Attention region** (`matmul→softmax→matmul`) ✅ **landed (2026-06-15).** The
  K-orientation blocker is closed: the score matmul's `transpose_b`/`transpose_a`
  is read from the IR and carried two ways — (1) as a Phase-0a **dispatch role**
  (`_chain_dispatch_roles` records `transpose_a`/`transpose_b`; the authoritative
  `_auth_dispatch_matmul_*` handlers flip the operand via `_auth_score_ab`), and
  (2) on the **`AttentionRegion`** (`q_transposed`/`k_transposed`, set by
  `discover_attention_regions`, applied by `_natural` in the reference +
  `run_fused_attention`). `discover_attention_regions` is now **wired into
  `_apple_gpu_try_synthesized_fusion`** (oracle- + cost-gated). This fixed a
  *pre-existing correctness gap*: canonical `softmax(Q·Kᵀ)·V` (`transpose_b=True`)
  was computing `Q·K` — now correct across shapes incl. the ambiguous D==Nk case.
  Guards: `tests/unit/test_fusion_attention_orientation.py` (8). *Follow-on:*
  route large-N/scale/causal attention to the synthesizer; f16/bf16 attention.
- Boundary rule: **synthesize MSL** for memory-bound fusable glue; **keep MPSGraph**
  for compute-bound primitives Apple ships a tuned kernel for (large-N GEMM, bmm,
  reductions). Never displace a working MPSGraph call — only the numpy/reference
  tail.

### M3 — Low-precision (FP8 / FP4 / MX) contract — compiler side ✅ (2026-06-15); execution toolchain-gated
**Toolchain grounding (Decision #27, exhaustive).** The macOS **26.5 CLT SDK** on
this machine exposes `MTLTensorDataType` only up to f32/f16/bf16/int8-32 **+
Int4/UInt4 (`@26.4`)** — **no FP8 (E4M3/E5M2), FP4 (E2M1), or MX (UE8M0)**, and
**no multi-plane/auxiliary-plane machinery**, in Metal, MPS, MPSGraph, or any
installed SDK. So **real-silicon FP8/FP4 execution is toolchain-gated** (not
hardware — the M1 Max GPU likely supports it; there's just no public API to drive
it here on 26.5).

**The unlock is concrete and dated: macOS 27.0 (Decision #27, doc dump 2026-06-16).**
The 27.0 SDK ships exactly the types the contract anticipated —
`MTLTensorDataType.{float8e4m3, float8e5m2, float4e2m1, float8ue8m0, int2, uint2}`
— **plus the multi-plane tensor machinery that is the runtime image of a
`ScaleLayout`**: `MTLTensorAuxiliaryPlaneDescriptor.blockFactors` ("data-plane
elements per scale element"), `MTLTensorDescriptor.auxiliaryPlanes`,
`MTLTensorBufferAttachments` (per-plane backing storage), and per-plane
`getBytes/replace(...plane:)`. A microscaled tensor = one data plane (element
dtype) + one auxiliary scale plane (`float8ue8m0` for MX / `float8e4m3` for NVFP4)
whose `blockFactors` encode the block size. This validates the hardware-free
design 1:1 and dates the execution gate (a 27.0-beta SDK on this M1 Max, no new
silicon).

**The compiler-side contract landed (hardware-free).** `compiler/microscaling.py`
— the **scale layout as a first-class operand** (the DeepGEMM extraction): a
`ScaleLayout(block_size, axis, scale_dtype)` + `MicroscaledArray(codes, scales,
format)` triple (scale-layout enforced at construction), with bit-accurate
`quantize`/`dequantize`/`fake_quantize`/`mx_matmul` over `FORMATS` = {fp8_e4m3,
fp8_e5m2, mxfp8_e4m3 (E8M0 scale, block 32), mxfp4_e2m1, nvfp4 (E4M3 scale, block
16)}, faithful to OCP-MX / NVFP4 (via `ml_dtypes`). `numeric_policy_for` exposes
the storage/accum/scale-layout contract (Decision #15a: storage ≠ accumulator).
**Metamorphic proof** (backend-independent — survives the future Metal lowering):
quantize idempotence, exact power-of-2 scale-invariance, and microscaled GEMM ≈
fp32 within the format grid.

**Registry-integrated + unified + silicon-validated (2026-06-15):**
- **Single source of truth** — `microscaling.format_for_dtype(dtype)` derives the
  executable format *from* `grouped_layout.scale_layout_for(dtype)` (the registry's
  declared scale-layout contract), so the audit contract and the bit-accurate
  reference can't drift; a cross-check test pins it per dtype.
- **Fake-quant unified** — `tessera.quantize_nvfp4` (which feeds the grouped-GEMM
  / MoE quant path) was a loose fake-quant with an fp32 block scale; it now
  delegates to the faithful contract (spec E4M3 block scale, bit-accurate E2M1
  codes). 170 grouped-GEMM/MoE/quant tests green — no numeric breakage.
- **int8 validated on real silicon** — FP8/FP4 can't execute here (SDK gate), but
  the int8-quantized GEMM (per-tensor) is validated DESIL-style: the same
  dequantized operands matmul'd via numpy vs the real `@jit(target="apple_gpu")`
  Metal lane agree — the contract executes correctly on hardware for the one
  low-precision dtype this SDK exposes.

**Metal bridge landed (hardware-free, 2026-06-16):** `microscaling.py` now maps
the contract onto the concrete 27.0 API — `mtl_tensor_data_type(dtype)` →
`MetalTensorType(swift_case, mtl_symbol, min_macos)` (fp8/fp4/e8m0 → `27.0`;
int4/uint4 → `26.4`; int8/f32 → `26.0`), and `metal_plane_plan(fmt, shape)` →
`MetalPlanePlan(element, aux_planes, min_macos)` emitting the multi-plane recipe
(per-tensor int8 → no aux plane @ `26.0`; MX/NVFP4 → one scale plane with
`block_factors`/`scale_shape` @ `27.0`). The `scale_shape` round-trips the
contract's own `ScaleLayout.scale_shape` (one source of truth). Guards:
`tests/unit/test_microscaling.py` (29) + `test_microscaling_metal_bridge.py` (7).
**Runtime stub landed (version-gated no-op, 2026-06-16):** `apple_gpu_runtime.mm`
adds `tessera_apple_gpu_supports_microscaling()` (runtime probe — 1 only when
built against a >=27.0 SDK AND running on 27.0+) and
`tessera_apple_gpu_microscaled_descriptor_probe(element_code, scale_code, rank,
dims, block_factors)` — the multi-plane `MTLTensorDescriptor` construction
sketch (data plane + one `MTLTensorAuxiliaryPlaneDescriptor` scale plane with
`blockFactors`, keyed `MTLTensorPlaneTypeScales`). The real construction is
**compile-time gated** behind `TESSERA_HAVE_MICROSCALING_SDK` (`__MAC_27_0`), so
it's excluded on the 26.5 SDK and the entry points are honest no-ops that set
last-error **kind 4 (toolchain_gated)**. Verified: the runtime builds clean
against 26.5, `supports_microscaling()==0`, `descriptor_probe(...)==0` + kind 4.
Python surface `microscaling.metal_microscaling_available()` reads the probe.
Guards: `test_microscaling_metal_bridge.py` (8, incl. runtime-gate↔contract).
*Follow-on (needs a 27.0 SDK):* finish the construction (id<MTLTensor> +
`MTLTensorBufferAttachments` for data+scale planes), then FP8/FP4 GEMM via the ML
encoder/MSL — the Python bridge + this descriptor sketch give it the exact
per-plane dtype + blockFactors target.

### M4 — Whole-graph MSL emitter (the GPU `tessera_jit`) — **first cut landed (2026-06-15)**
The GPU analogue of the CPU `run_graph_ops`/`tessera_jit` lane: emit ONE Metal
kernel for a whole region of the graph and run it in a single dispatch.
**Landed: the pointwise whole-graph emitter** — an arbitrary connected DAG of
same-shape elementwise ops (`add`/`sub`/`mul`/`div` + `relu`/`sigmoid`/`tanh`/
`silu`/`gelu`/`neg`/`abs`/`exp`) compiles to one MSL kernel (one thread/element,
fp32 temps, `ST(...)` store) instead of N MPSGraph dispatches with host
round-trips. Pieces: `PointwiseGraphRegion` + `synthesize_pointwise_graph_msl`
(variable input-buffer count) + `discover_pointwise_graph` (union-find over
candidate ops → connected components, single-exit, ≥2 ops so a lone op stays on
the MPSGraph lane) + `run_pointwise_graph`, a **variable-arity** runtime symbol
`tessera_apple_gpu_synth_pointwise_f32/f16` (pointer-array input ABI; a
`std::vector<MetalBufferGuard>` keeps the variable buffer count RAII-correct),
wired into `_apple_gpu_try_synthesized_fusion` (claims a region only when it
genuinely fuses on Metal; broadcast/unsupported fall to per-op). **Verified on M1
Max**: `gelu(add(mul(x,a),b))` runs as one kernel, fp32-exact; f16 native; chains
+ diamonds + matmul-bounded tails covered. Guard:
`tests/unit/test_pointwise_graph_fusion.py` (10).
*Follow-ons:* broadcast operands (per-feature bias/scale); compose pointwise with
the matmul/reduction synthesizers into a single whole-graph kernel; the bigger
`graph_ir → MSL` whole-program emitter (or retarget the CPU JIT's
`tessera→linalg` spine to `linalg→gpu→{SPIR-V/Metal}`).

### M5 — Displace the dispatcher lane-by-lane (HF, Evaluator-gated) — **gate landed (2026-06-16)**
Migrate op families from name→MPS/MSL dispatch to synthesizer codegen one at a
time, each gated by the Evaluator's horizontal/DESIL oracle (codegen ≡ library on
hidden inputs). Keep the MPSGraph fast paths; target the numpy/reference tail.

**Landed: the reusable displacement gate** — `compiler/fusion_equivalence.py`.
`displacement_verdict(kind, shape, *, seed)` runs one synthesizer codegen lane on
**hidden** inputs (fresh RNG the codegen never saw) and returns `equivalent`
(ran on Metal AND matched the reference within tol), `divergent` (ran on Metal
but mismatched — a real codegen bug, blocks shipping), or `not_displaced` (fell
back — no Metal execution to credit). The provenance gate is the Evaluator's
invariant: a silent numpy fallback can *never* earn `equivalent`. `gate_all`
sweeps every shipped lane. **Verified on M1 Max** — all four lanes
(`matmul_epilogue`, `norm_chain`, `attention`, `pointwise`) return `equivalent`
on `metal_runtime` at rel_err ~1e-7. This makes the M2/M4 displacements provably
safe and is the gate every future lane migration calls before shipping. Guard:
`tests/unit/test_fusion_displacement_gate.py` (11).
*Follow-ons:* extend coverage to the remaining per-op MPSGraph tail (claim
multi-op reduction chains beyond rmsnorm/softmax/layer_norm); wire `gate_all`
into the generated conformance dashboard so a divergent lane fails CI.

## Dependency notes
- M1 done. M2 (`norm_chain`) is the natural next step — pure Python, reuses the
  synth symbol, execution-reaching. M2 (attention orientation) connects to the
  Phase 0a `dispatch`-role mechanism. M3 is the highest-leverage frontier but
  needs runtime MTLTensor plumbing. M4 is the unifying long game.
- All of M1–M3 are hardware-free on this M1 Max (Apple7). No NVIDIA/ROCm silicon
  required — that's the point of grounding in Apple7 + Metal 4.
