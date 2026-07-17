---
status: Proposal
classification: Architecture / Tile IR
last_updated: 2026-07-14
---

# Portable Tile Fragment ABI

## Decision

Make the Tile IR responsible for describing a **logical tile**, its placement,
and the selected matrix instruction.  Make each target backend responsible for
the exact mapping from logical elements to lanes and registers.

The shared IR must never use NVIDIA's `vector<2xf16>` fragment shape, nor an
AMD VGPR ordering, as its portable representation.  Those are ABI details of a
particular instruction and ISA revision.  The portable contract is an opaque
`!tile.fragment` value with a verifier-checked descriptor and layout.

This closes the current gap where `tile.mma` can reach NVIDIA `nvvm.mma.sync`
only after a test supplies pre-packed LLVM vectors, while ROCm forwards generic
operands to WMMA/MFMA without a materialization contract.

## Scope and non-goals

This proposal covers register fragments for cooperative matrix operations on
NVIDIA and AMD ROCm.  It builds on existing `#tile.layout`,
`#tile.buffer_ref`, `!tile.async_token`, and pipeline/barrier attributes.

It does not introduce a general CuTe clone, mandate a C++ CUDA Tile dependency,
or prescribe a universal physical layout.  Tensor-map descriptors, NVIDIA
TMEM, and AMD LDS remain target-specific memory facilities behind the same
logical source interface.

## New IR vocabulary

### Types and attributes

```mlir
!tile.tile<element = f16, shape = [16, 16], layout = #tile.layout<...>>
!tile.fragment<role = a, element = f16, mma = #tile.mma_desc<...>>
#tile.mma_desc<family = auto, m = 16, n = 8, k = 16,
               a = f16, b = f16, acc = f16,
               a_layout = row_major, b_layout = col_major>
```

`!tile.tile` is a logical, distributed value.  It says what region of a source
is owned by the participating threads; it is not a promise about a pointer,
vector width, or register order.  `!tile.fragment` is deliberately opaque to
generic Tile passes.  In the first implementation its role and descriptor are
carried by its defining operation, rather than type parameters, so generic
passes cannot accidentally reconstruct a vendor register ABI. Its role is
`a`, `b`, or `acc`; block-scaled descriptors may additionally use `scale_a`
and `scale_b`. Scale fragments are logical scale tiles, never packed vendor
selector registers. Its descriptor determines the compatible MMA instruction
family.

`#tile.mma_desc` is target-neutral at construction time (`family = auto`).
Target selection resolves it to a concrete instruction capability during
lowering.  A target may reject it if no exact instruction exists.  A resolved
descriptor records instruction shape, input/accumulator dtypes, logical A/B
orientation, and any semantic packing granularity (for example K-blocks), but
not lane-to-register numbering.

### Operations

```mlir
%tile = tile.view %source[%m, %k]
    {shape = [16, 16], layout = #tile.layout<...>,
     tile.memory = #tile.memory_layout<space = "smem", order = "row_major", leading_dim = 16>, bounds = mask}
    : (...) -> !tile.tile<...>

%a = tile.fragment_pack %tile
    {role = "a", mma = #tile.mma_desc<...>}
    : !tile.tile<...> -> !tile.fragment<role = a, ...>

%d = tile.mma %a, %b, %c {mma = #tile.mma_desc<...>}
    : (!tile.fragment<role = a, ...>, !tile.fragment<role = b, ...>,
       !tile.fragment<role = acc, ...>) -> !tile.fragment<role = acc, ...>

%result = tile.fragment_unpack %d {layout = #tile.layout<...>}
    : !tile.fragment<role = acc, ...> -> !tile.tile<...>

tile.store %result, %destination[%m, %n] {bounds = mask}
    : !tile.tile<...>, ... -> ()
```

`tile.view` has no memory effect and expresses logical slicing, coordinate
origin, layout, and edge mask.  It replaces the current stringly staged-buffer
convention as the operand that a pack can consume.  `tile.load` is a convenience
form that combines `view` and a synchronous materialization; it is optional for
the first implementation because `tile.async_copy` + `tile.wait_async` already
provide the global-to-shared/LDS pipeline.

A `tile.view` becomes materializable when it carries `tile.memory` and has
`(base_pointer, row_origin, column_origin)` operands. `#tile.memory_layout`
names gmem/smem/lds, row/column order, and leading dimension in elements. This
is the physical boundary the NVIDIA and ROCm packers consume; a tensor-only
logical view remains valid Tile IR but cannot be lowered to machine loads.

`tile.fragment_pack` is the important missing operation.  It consumes either a
tile value or a waited staged tile and produces a role-specific fragment.  It
is the only boundary at which a backend may choose its lane/register ordering.
`tile.fragment_unpack` is its inverse at a logical-result boundary.  Neither
op is a no-op conversion: both have a defined logical element mapping and must
be lowered or rejected.

The existing permissive `tile.mma` form stays accepted during migration.  The
typed three-fragment form is eligible for ordinary hardware MMA lowering.
NVFP4 uses the verifier-checked five-fragment form `(A, B, accumulator,
scale_a, scale_b)`; the legacy form remains an abstract/value-lane carrier.

## Implemented first slice

The dialect now registers `!tile.tile`, `!tile.fragment`,
`#tile.mma_desc`, `tile.view`, `tile.fragment_pack`, `tile.fragment_unpack`,
`tile.store`, launch-level `tile.matmul_kernel`, and portable `#tile.epilogue`.
Their structural verifiers enforce role/descriptor
agreement through `tile.mma`, require a logical layout at view/unpack
boundaries, and retain legacy untyped `tile.mma` compatibility.

The NVIDIA backend now materializes exact `sm_120` f16, bf16, TF32, FP8,
int8, and block-scaled NVFP4 slices. NVFP4 consumes nibble-packed logical
`16x64`/`64x8` matrices plus logical UE4M3 `16x4`/`4x8` scale views; only the
NVIDIA lowering maps them to four A registers, two B registers, per-lane scale
selectors, and `mma.sync...mxf4nvf4.block_scale`. The ordinary f16/bf16 path
uses pointer-backed A/B views and a
portable `tile.fragment_zero` f32 accumulator. It emits four A and two B
registers using the tested lane map (`vector<2xf16>` for f16; packed `i32` for
bf16, with the bitcast owned by the backend), reaches the real NVVM MMA op,
then extracts the four f32 results per lane and stores the logical 16x8 tile.
The emitted LLVM kernel translates to PTX, assembles for sm_120, and executes
against a NumPy GEMM oracle on an RTX 5070 Ti. Unsupported shapes, storage
orders, swizzles, sources, or live fragment results are named errors.

The ROCm backend materializes the corresponding gfx1151 Wave32
`m16n16k16.row.col` slice for f16, bf16, signed int8, and signed int4. It lowers
the portable view/pack/zero/MMA/unpack/store chain to the real RDNA WMMA
intrinsic and executes the backend-owned lane/register map through the HIP
module runtime. FP8/BF8 and contradictory physical layouts fail with named diagnostics;
RDNA 4 WMMA v2 and CDNA MFMA maps remain architecture-specific follow-on work.

Fixtures:

- `tile_fragment_abi.mlir` — portable parser/verifier contract;
- `sm120_pointer_fragment_pack.mlir` — six physical loads through real MMA;
- `sm120_pointer_fragment_pack_invalid.mlir` — wrong B order rejection.
- `sm120_pointer_fragment_store.mlir` — f32 accumulator unpack and output store;
- `sm120_pointer_fragment_store_bf16.mlir` — explicit packed-i32 bf16 ABI proof;
- `test_nvidia_tile_fragment_compiler_path.py` — LLVM/PTX/cubin generation and
  on-device numerical comparison.
- `gfx1151_tile_fragment_store.mlir` — portable gfx1151 pack/MMA/unpack/store
  and real ROCDL intrinsic proof;
- `test_rocm_wmma_gemm_generated.py` — gfx1151 hsaco generation, production HIP
  launch, all four supported dtype comparisons, and aligned/ragged launch proof.

## Required verifier invariants

- All fragments used by `tile.mma` have the same resolved MMA descriptor.
- Ordinary roles are exactly A, B, accumulator. Block-scaled NVFP4 additionally
  requires scale-A and scale-B; output is always an accumulator fragment.
- Fragment element types and logical extents match the descriptor.  An
  accumulator conversion is explicit; it is never inferred from a vector type.
- The source tile layout covers each required logical element once, except where
  a descriptor explicitly permits replication or a boundary mask supplies a
  zero/identity value.
- A pack from `smem`/`lds` that depends on `tile.async_copy` must carry the
  relevant `!tile.async_token` through `tile.wait_async`.  The existing backend
  pipeline legality passes remain the authority for multi-stage ordering.
- `fragment_unpack`/`store` cannot silently discard masked or replicated
  elements; their boundary policy is explicit.

## Backend lowering contract

| Portable operation | NVIDIA | AMD ROCm |
|---|---|---|
| Resolve descriptor | Select `mma.sync`, WGMMA, or tcgen05 only when that target supports the exact dtype/shape. | Call the existing `MmaDescriptor` selector; it chooses RDNA WMMA or CDNA MFMA by gfx architecture. |
| `fragment_pack` | Map Tile layout to instruction fragments. The `sm_120` `mma.sync.m16n8k16` path gives A four and B two physical registers: `vector<2xf16>` for f16 or packed `i32` for bf16; the f32 accumulator has four scalar registers. | Map the same logical tile to the selected WMMA/MFMA VGPR operands. Register order remains owned by the ROCm lowering, not exposed in Tile IR. |
| `tile.mma` | Lower typed fragments to the matching NVVM MMA op; preserve the accumulator. | Lower typed fragments to `tessera_rocm.wmma` or `tessera_rocm.mfma`; preserve the selected `MmaDescriptor` metadata. |
| `fragment_unpack` | Materialize the logical accumulator tile for an epilogue/store. | Materialize the logical accumulator tile for an epilogue/store. |

This design deliberately accommodates different shapes: NVIDIA's initial
`m16n8k16` f16 path, RDNA WMMA's 16x16 variants, and CDNA MFMA variants do not
need a shared physical fragment width to share a Tile program.

ROCm now resolves that backend column through an exact architecture-owned
descriptor rather than a gfx-prefix convention. gfx11 uses duplicated 16-value
inputs and its padded accumulator map; RDNA 4 uses dense eight-value SOA inputs;
gfx125x WMMA-v2 uses K32 16-value f16/bf16 inputs plus explicit sign, C-modifier,
and operand-reuse properties; CDNA2/3/4 use Wave64 four-value f16/bf16 MFMA
inputs and accumulators. The C++ materializer and Python selector mirror the
same family and intrinsic-ABI names, with a unit ratchet preventing drift.

The shared architecture fixture cross-assembles without spills for gfx1100,
gfx1151, gfx1200, gfx1201, gfx1250, gfx1251, gfx90a, gfx942, and gfx950. gfx940
real-MFMA lowering is covered but object serialization is toolchain-gated in the
installed Debian LLVM 22 build. These object results prove instruction legality
and resource use, not remote-device numerical behavior or performance; those
claims remain exact-device gated.

## Delivery plan

1. Add the opaque types, `#tile.mma_desc`, and structural verifier.  Keep the
   existing generic `tile.mma` path working.
2. Add `tile.view`, `tile.fragment_pack`, and `tile.fragment_unpack` with
   canonical generic assembly and negative verifier tests.
3. Complete: extend the NVIDIA `sm_120` f16 A/B pointer materializer with f32
   accumulator unpack/store, emit a launchable cubin, and compare the generated
   kernel against NumPy on the RTX 5070 Ti.
4. Complete at compiler/object level: ROCm descriptor resolution and
   pack/unpack cover RDNA3, RDNA4, gfx125x WMMA-v2, and CDNA2/3/4 MFMA. The
   identical logical Tile fixture executes against a reference on gfx1151;
   RDNA4, gfx125x, and CDNA exact-device numerical/performance closure remains
   gated on access to matching silicon.
5. bf16 is complete for the canonical sm_120 Tile path. The shipped CUDA GEMM
   ABI and composed transformer lanes now provide TF32/E4M3/E5M2 execution
   oracles; their portable Tile fragment pack/layout variants remain open. Add
   each Tile variant only with matching layout tests. Int8 and FP8/FP4 Tile
   variants remain gated on their fragment contracts (NVFP4 also remains gated
   on correct block-scale numerics).

## Acceptance criteria for the first slice

The first slice is complete when one source-level logical `16x16` A tile,
`16x8` B tile, and `16x8` accumulator tile can be packed, executed, unpacked,
and stored on `sm_120` f16 without test-authored LLVM fragment vectors; the
same Tile fixture is either executed on a supported ROCm WMMA/MFMA target or
fails at descriptor resolution with a named capability error.

## sm_120 A/B mapping reference

The first NVIDIA materializer is constrained to `m16n8k16.row.col` f16/bf16. Its
tested A/B mapping is in `python/tessera/compiler/nvidia_fragment_layout.py`,
derived directly from the on-silicon PTX kernel: `gid = lane >> 2`,
`tig = lane & 3`; A loads rows `{gid, gid+8}` and K pairs at `{2*tig, 2*tig+8}`;
B loads N column `gid` and K pairs at `{2*tig, 2*tig+8}`.  The map returns
logical element pairs, not byte addresses, so it is valid for global or staged
shared-memory sources.

For the f32 accumulator, each lane owns `(gid, 2*tig)`, `(gid, 2*tig+1)`,
`(gid+8, 2*tig)`, and `(gid+8, 2*tig+1)`. The generated-kernel execution oracle
validates this mapping independently of the input-fragment map.

## Launch-level materialization

`tile.matmul_kernel` is the portable bridge from one logical contraction to a
launchable tiled kernel. Its pointer ABI is A, B, optional bias, D, M, N, K;
`#tile.mma_desc` selects the contraction and `#tile.epilogue` owns bias,
activation, and output-conversion semantics without exposing physical fragment
registers. The NVIDIA materializer retains a one-warp/direct-global 16x8
baseline and adds a four-warp shared-memory path. The latter cooperatively
stages A[32,16] and B[16,32], then each warp computes two adjacent m16n8
fragments, producing a 32x32 CTA macro-tile (eight MMA instructions per barrier
interval). Block Y/X derive macro-tile origins, masked loads zero-fill ragged
M/N/K, an `scf.for` carries eight f32 accumulator values per warp across
16-wide K panels, and masked stores suppress edge writes. Bias,
ReLU/GELU/SiLU, and f32-to-f16 conversion reuse the accumulator mapping
immediately before stores. NVIDIA and ROCm call the same inline Tile activation
emitter; address-space-specific bias loads and masked stores remain backend
owned.

### Shared epilogue order and rejection contract

The backend-neutral `FusedRegion` oracle defines the semantic order as f32
accumulation, the authored pointwise epilogue chain (including at most one
per-column bias), optional full-shape residual addition, then any terminal row
reduction. Storage rounding occurs at the operand ABI; it does not change the
accumulator or reorder bias, activation, and residual. Missing bias/residual
operands and unsupported dtype/op/order combinations fail with registered
`E_FUSED_EPILOGUE_*` diagnostics. A backend may decline a physical fusion, but
must report a non-fused execution route rather than label it native fused work.

Kernel-only measurements on the RTX 5070 Ti show why both paths remain useful:
the staged path is slower at 256/512 square GEMMs, but is 1.20x faster at 1024³
(0.2225 ms versus 0.2679 ms) and 1.37x at 2048³ (1.5060 ms versus 2.0628 ms).
Dispatch should therefore select by measured shape rather than globally
replacing direct fragment loads.

That dispatch is now wired through the normal D2 candidate arbiter. The runtime
compiles both canonical Tile schedules through `tessera-nvidia-opt` → NVVM →
LLVM NVPTX, registers their PTX with the shipped launch bridge, and exposes
`nvidia_tile_matmul_direct` and `nvidia_tile_matmul_shared` as separate emitted
matmul candidates. The bridge owns their distinct launch contracts: direct uses
ceil(N/8)×ceil(M/16) blocks of 32 threads; shared uses
ceil(N/32)×ceil(M/32) blocks of 128 threads. Both take i64 M/N/K arguments and
accept ragged shapes.

Normal dispatch consults the device/shape-bucket/dtype corpus before tier
priority. Online tuning compares the two Tile schedules with the shipped and
legacy-emitted GEMMs under the same correctness gate and end-to-end latency
metric, then persists the fastest candidate. Explicit candidate forcing remains
available for compiler-path isolation. This deliberately distinguishes the
kernel-only crossover from application-visible selection: shared staging can
beat direct fragment loads while the shipped kernel still wins after considering
the complete candidate set.

Server-oriented tuning also has a distinct `timing="device"` mode. The PTX
bridge allocates and uploads A/B once, warms both schedules, and uses CUDA driver
events around repeated kernel launches; allocation, H2D, and D2H are outside the
timed interval. D2 includes the timing mode in its persisted key, so device-only
evidence cannot overwrite or be mistaken for end-to-end evidence. Candidates
without a device-resident measurement hook are excluded from this mode rather
than assigned a fabricated latency. On the RTX 5070 Ti the stable f16 crossover
is direct at 512³ (0.0403 versus 0.0532 ms) and shared at 1024³ (0.2208 versus
0.2692 ms); shared reaches 1.36x at 2048³.

This launch form executes aligned and ragged shapes on sm_120 and gfx1151. The
ROCm path converts `tile.matmul_kernel` directly into an in-memory GEMM request
consumed by its production WMMA K-loop/epilogue generator. The legacy
`tessera_rocm.wmma_gemm` directive is only a compatibility adapter into that
same request; the portable path creates no temporary marker operation. ROCm
also hoists each per-column bias load outside the accumulator-element loop.
Other ROCm architecture families now resolve their own WMMA-v2 or MFMA fragment
maps. Forms whose physical materialization is not yet proven—gfx125x FP8 and
additional CDNA low-precision variants—retain named readiness guards.
