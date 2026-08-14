---
last_updated: 2026-08-14
audit_role: reference
---

# CDNA5 gfx1250 MI455X / gfx1251 MI430X — Source-Grounded Compiler Reference

> **Purpose.** A primary-source reference for the CDNA5 gfx1250 (AMD Instinct
> MI455X) and its distinct gfx1251 (MI430X) sibling, assembled for compiler and backend work: hardware constants,
> the WMMA pipeline and its hazards, the three data-movement mechanisms, the
> split completion model, workgroup clusters, device-initiated SDMA, and the
> scale-up fabric.
>
> **This is not a status surface.** Nothing here claims Tessera support for
> anything. For ROCm status see [`ROCM_AUDIT.md`](ROCM_AUDIT.md); for counts see
> `docs/audit/generated/`. For general AMD-ecosystem design patterns see
> [`ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md`](ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md),
> which this document extends with gfx1250-specific depth.
>
> **Provenance per claim:** **[V]** verified from primary source (LLVM
> tablegen/source, ROCm repo source, or the on-machine assembler); **[A]**
> verified by assembling the instruction locally; **[S]** stated by an AMD
> engineering document in-repo (first-party but prose); **[I]** inference or
> recommendation — explicitly not established.
>
> Researched 2026-07-28 against `ROCm/llvm-project@amd-staging`,
> `ROCm/rocm-systems@develop`, `ROCm/rocm-libraries@develop`, and both local
> LLVMs (Homebrew `llvm/22.1.8` and LLVM 23.1.0-rc1) — see the toolchain note in §10.

---

## 0. Evidence ladder

When these sources disagree — and they do — resolve in this order:

1. **The assembler** (`llvm-mc -arch=amdgcn -mcpu=gfx1250`). It either encodes or
   it does not. Cheapest and most decisive.
2. **LLVM tablegen** (`AMDGPU.td`, `IntrinsicsAMDGPU.td`). Machine-consumed, so
   errors surface as miscompiles rather than stale prose.
3. **ROCm library source** (`rocshmem`, `rccl`, `stinkytofu`, `rocke`).
   Production code, but the comments can lag the code.
4. **AMD engineering prose** (in-repo design docs, blogs). Useful for intent and
   measurements; demonstrably self-contradictory in places — see §8.

A worked example of why the order matters: `rocke`'s MHA case study §6 states
gfx1250 has "no async global→LDS DMA" and "no `ds_read_tr`", while §5 of the
*same file* describes building a `ds_load_tr16_b128` path, and the capability
table in the sibling plan asserts both exist. The assembler settles it — both
exist. **[V]**

---

## 1. Target identity and hardware constants

gfx1250 is AMD Instinct **MI455X**. gfx1251 is AMD Instinct **MI430X**, a
sibling target in the same CDNA5 family — *not* an alias (see §2.4). Both derive from
`FeatureISAVersion12_50_Common` in `AMDGPU.td`. **[V]**

Despite the `gfx12xx` numbering it is **not RDNA 4**. It is wave32 with WMMA and
no MFMA — a matrix pipeline lineage distinct from both RDNA 4 (gfx1200/1201,
16x16x16 WMMA) and CDNA (gfx942/gfx950, MFMA). **[V]**

| Constant | Value | Source |
|---|---|---|
| Addressable LDS | **327680 B (320 KiB)** | `FeatureAddressableLocalMemorySize327680`, `AMDGPU.td:2292` **[V]** |
| Addressable VGPRs | **1024** | `Feature1024AddressableVGPRs`, `AMDGPU.td:2208` **[V]** |
| Wavefront | 32 | `FeatureWavefrontSize32` **[V]** |
| Virtual address bits | 57 | ROCKE capability table **[S]** |
| Data cache line | 128 B | `FeatureDataCacheLineSize128` **[V]** |
| Matrix path | WMMA; `has_mfma = false` | ROCKE **[S]**, corroborated by `AMDGPU.td:418` **[V]** |

For scale: gfx950 is 163840 B LDS, RDNA 3/3.5/4 are 65536 B. gfx1250 doubles
CDNA 4 and is 5× RDNA. The VGPR count doubles CDNA's 512 combined
(256 VGPR + 256 AGPR); gfx1250 has no AGPR file, so 1024 is all architectural
VGPR. **[V]**

AMD's own ROCKE capability table for the gfx1250 `ArchTarget` **[S]**:

```text
wave_size = 32                 matrix_path = wmma         has_mfma = false
has_wmma = true                has_swmma = true           has_tdm = true
has_async_global_lds = true    has_ds_load_tr = true      max_lds_bytes = 320*1024
wgp_cache_lds_shared = true    virtual_address_bits = 57
waitcnt_model = split_gfx1250  barrier_model = split_named_cluster
requires_shader_end_padding = true
```

`max_lds_bytes = 320*1024` = 327680 matches the LLVM feature exactly — an
independent confirmation from AMD's kernel team against AMD's compiler team.

### 1.1 Notable subtarget features

Present on gfx1250 **[V]**:

`FeatureClusters`, `FeatureMcastLoadInsts`, `FeatureAsyncLoadToLDSInsts`,
`FeatureAsyncStoreFromLDSInsts`, `FeatureAsynccnt`, `FeatureWaitXcnt`,
`FeatureTransposeLoadF4F6Insts`, `FeatureSWMMACGfx1250Insts`,
`FeatureWMMACoexecutionHazards`, `FeatureTransCoexecutionHazard`,
`FeatureLdsBarrierArriveAtomic`, `FeatureSWakeupBarrier`,
`FeatureSetPrioIncWgInst`, `FeatureGloballyAddressableScratch`,
`FeatureKernargPreload`, `FeatureVmemPrefInsts`, `FeatureSmemPrefetchInsts`,
`FeatureXNACK`, `FeatureSupportsSRAMECC`, `FeatureRealTrue16Insts`,
`Feature45BitNumRecordsBufferResource`.

Absent on gfx1250, present on gfx1251: `FeatureGFX125xLowestRateWMMA`,
`FeatureFullRate64Ops`, `FeaturePackedFP64Ops`, `FeatureGFX1251GEMMInsts`. **[V]**

### 1.2 Cache hierarchy and the SCOPE ladder

From `AMDGPUUsage.rst`, "Memory Model GFX125x". **[V]**

```
agent ── shader engines (SE) ── shader arrays (SA) ── work-group processors (WGP)
                                                       └── 4× SIMD32 (2 SIMD32-pairs)
```

- Each WGP has a **single write-through WGP$ shared between LDS and vector L0**.
  Vector L0 holds clean data only. This is the hardware fact behind ROCKE's
  `wgp_cache_lds_shared = true`, and behind the TDM guidance about bypassing vs
  routing through the per-WGP cache.
- **Each WGP$ has two request queues, one per SIMD32-pair.** Each handles both
  LDS and vector L0 requests; requests within a queue are serial and in-order,
  but are *not* ordered against the other queue. (This is the "2 memory units per
  WGP" figure that appears in AMD's marketing material.)
- Scalar memory uses a separate scalar L0, **not** kept coherent with vector L0
  by hardware — safe only because scalar ops are restricted to memory proven
  not to change during the dispatch.
- All WGPs on an SE share an **L1 buffer**, with a separate request queue per
  WGP$ (again in-order within a queue, unordered across queues).
- An agent may have **multiple L2 caches**. Virtual address ranges can be
  configured non-hardware-coherent, read-write coherent with other L2s on the
  same or other agents, or L2-bypassing for system coherence.

The `SCOPE` field on vector memory operations names a cache level **[V]**:

| `SCOPE` | Level |
|---|---|
| `SCOPE_CU` | WGP — the compiler's default, omitted in textual asm |
| `SCOPE_SE` | Shader Engine |
| `SCOPE_DEV` | Device / agent |
| `SCOPE_SYS` | System |

An operation reaching a cache with a *smaller* scope is forwarded onward; at a
cache whose scope is ≥ its own it can complete locally (read hits, write
completes and reports, RMW done locally). Hardware assigns each cache a scope
per agent configuration, which is what lets `SCOPE_DEV` implement agent
coherence even with multiple non-coherent L2s.

Also present: an `nv` ("non-volatile") bit marking memory not expected to change
during the kernel, propagated to cache lines as `$nv`; and `global_inv` /
`global_wb` / `global_wbinv` cache-control instructions whose affected levels are
selected by `SCOPE`, completing via `s_wait_storecnt`. **[V]**

---

## 2. Matrix pipeline — WMMA

### 2.1 Shapes

The gfx1250 WMMA family is far wider than RDNA 4's single 16x16x16 **[V]**:

| Class | Intrinsics |
|---|---|
| f16 / bf16, K=32 | `wmma_f32_16x16x32_{f16,bf16}`, `wmma_f16_16x16x32_f16`, `wmma_bf16_16x16x32_bf16`, `wmma_bf16f32_16x16x32_bf16` |
| fp8 / bf8, K=64 | `wmma_{f32,f16}_16x16x64_{fp8,bf8}_{fp8,bf8}` (4 combos each) |
| fp8 / bf8, K=128 | `wmma_{f32,f16}_16x16x128_{fp8,bf8}_{fp8,bf8}` |
| int, K=64 | `wmma_i32_16x16x64_iu8` |
| block-scaled MX | `wmma_f32_16x16x128_f8f6f4`, `wmma_scale_f32_16x16x128_f8f6f4`, `wmma_scale16_...` (i32 vs i64 scale) |
| f4 | `wmma_scale_f32_32x16x128_f4`, `wmma_scale16_f32_32x16x128_f4` |
| sparse | `SWMMAC` at 16x16x64 / 16x16x128 (`FeatureSWMMACGfx1250Insts`) |

`wmma_f32_16x16x128_f8f6f4` uses `AMDGPUWmmaIntrinsicModsC_MatrixFMT` — a
**matrix-format operand**, so one instruction covers fp8/fp6/fp4 selected by a
format field rather than distinct opcodes. Any dtype contract modelling this
must treat element format as an *operand*, not part of the opcode identity. **[V]**

### 2.2 Fragment ABI

Per AMD's own probe **[S]**, corroborated by the intrinsic signatures **[V]**:

- A/B operands: `<16 x half>` per lane — the 32 K-elements split across the two
  lane-halves, 16 each.
- Accumulator: gfx12 column-distributed `<8 x float>`.
- bf16 uses native `<16 x bfloat>`, **not** `<_ x i16>`.
- The "v2" ModsC ABI carries an `i16` C-modifier immediate plus trailing
  `i1, i1` reuse flags — RDNA 4 uses the plain 3-arg form.

This matches what `python/tessera/compiler/rocdl_emit.py` already asserts.

### 2.3 The WMMA co-execution hazard — mandatory

`FeatureWMMACoexecutionHazards` and `FeatureTransCoexecutionHazard` are on
gfx1250. LLVM implements the fix in `GCNHazardRecognizer.cpp`
(`checkWMMACoexecutionHazards` / `fixWMMACoexecutionHazards`), including V_NOP
hoisting out of loops, citing hardware spec **SPG 4.6.12.1 "Requirements for
WMMA data hazards"**. **[V]**

```c
const int WMMAWaitStates[] = {5, 9, 3, 5, 9, 17, 2};   // next op is a WMMA
const int VALUWaitStates[] = {4, 8, 2, 4, 8, 16, 1};   // next op is a VALU
```

Categories are derived from instruction latency **[V]**:

| Cat | Latency | Instructions |
|---|---|---|
| 0 | 8 | WMMA f16/bf16; 16x16x128 fp8/bf8; f8f6f4 (not both-F4) |
| 1 | 16 | WMMA IU8 |
| 2 | 8 | SWMMAC f16/bf16/fp8/bf8 |
| 3 | 16 | SWMMAC IU8 |
| 4 | 16 | **gfx1251** 16-pass forms |
| 5 | 32 | **gfx1251** 32-pass forms |
| 6 | 4 | **gfx1250** 16x16x64 fp8/bf8; f8f6f4 (both-F4) |

The hazard fires only on register overlap — WMMA0's `vdst` against WMMA1's
`src0`/`src1`/`Idx`, or against a co-executable VALU's operands. Independent
work already scheduled between them counts toward the requirement.

**Consequences.**

- Lowering through LLVM (Target IR → LLVM IR → AMDGPU) gets this handling for
  free. **[V]**
- Hand-emitting assembly does not. AMD's own ROCKE hits the bug precisely
  because it emits asm directly: at high occupancy it reports intermittent
  garbage (>1e20) in the P·V accumulator in roughly 1/3 of runs, seed- and
  timing-dependent, and names "proper backend WMMA scheduler" as the correct
  fix. **[S]**
- This is a strong argument for Decision #19's discipline on this target, and a
  reason not to add an asm fast path for gfx1250. **[I]**

A second-order trap from the same source **[S]**: switching their softmax from
`ds_swizzle` to DPP *removed* an LDS serialization that had been
**incidentally** supplying the hazard gap, and the kernel began producing NaNs.
The fix was to auto-bump WMMA spacing whenever DPP softmax is enabled. A pure
performance change silently broke correctness through an unmodelled coupling.

### 2.4 gfx1250 ≠ gfx1251

`FeatureGFX125xLowestRateWMMA` is on **gfx1251** and `gfx12-5-generic`, not on
gfx1250. It selects hazard categories 4/5 (latency 16/32) instead of 0/6
(latency 8/4) — the **same instruction at 4× different throughput**. **[V]**

Treating the two as one class is correct only for their **common low-precision
ABI** (identical low-precision intrinsics and operand forms). gfx1251 adds an
FP64 WMMA form absent from gfx1250. Sharing either extension legality or any
**cost model, scheduler, or autotuner** decision is wrong.

### 2.5 The machine model behind the hazard categories

`SISchedule.td` carries three distinct models — `GFX1250SpeedModel`,
`GFX1251SpeedModel`, `GFX125xGenericSpeedModel` — and they are what
`computeInstrLatency` reads to pick a hazard category (§2.3). **[V]**

**WMMA runs on a dedicated XDL pipe, not the VALU.** That is the structural
reason co-execution hazards exist at all:

```tablegen
def HWXDL : ProcResource<1>;                       // matrix pipe
let ReleaseAtCycles = [4]  in def : HWWriteRes<WriteXDL1PassWMMA, [HWXDL], 4>;
let ReleaseAtCycles = [8]  in def : HWWriteRes<WriteXDL2PassWMMA, [HWXDL], 8>;
let ReleaseAtCycles = [16] in def : HWWriteRes<WriteXDL4PassWMMA, [HWXDL], 16>;
let ReleaseAtCycles = [32] in def : HWWriteRes<WriteXDL8PassWMMA, [HWXDL], 32>;

def : HWWriteRes<Write4PassWMMA,  [HWVALU], 16>;   // RDNA-style WMMA: VALU pipe
def : HWWriteRes<Write8PassWMMA,  [HWVALU], 32>;
def : HWWriteRes<Write16PassWMMA, [HWVALU], 64>;
```

`ReleaseAtCycles == latency` on the XDL entries means the matrix pipe is held for
the full duration — back-to-back WMMA of the same class cannot overlap, but VALU
work *can* run alongside. That overlap is the co-execution the hazard table
governs.

Note the contrast with RDNA: `Write*PassWMMA` targets `HWVALU`, so **gfx1151 WMMA
occupies the vector ALU** and competes with all vector work, whereas **gfx1250
WMMA occupies a separate pipe**. A cost model ported from gfx1151 to gfx1250
without this distinction will misprice every mixed WMMA/VALU loop. **[I]**

Per-instruction assignment **[V]**:

| Instruction class | gfx1250 | gfx1251 |
|---|---|---|
| 16x16x64 FP8/BF8 | XDL 1-pass — **4 cyc** | XDL 4-pass — 16 cyc |
| F16 / BF16 | XDL 2-pass — **8 cyc** | XDL 4-pass — 16 cyc |
| 16x16x128 FP8/BF8 | XDL 2-pass — **8 cyc** | XDL 8-pass — **32 cyc** |
| SWMMAC 16x16x128 FP8/BF8 | (2-pass class) | XDL 4-pass — 16 cyc |
| IU8 / IU4 | XDL 4-pass — 16 cyc | XDL 8-pass — 32 cyc |
| 32x16x128 F4 | XDL 2-pass — 8 cyc | XDL 8-pass — 32 cyc |
| F8F6F4 scaled | `WriteWMMAScale_16X16X128_F8F6F4` | `WriteWMMAScaleFP4_16X16X128_F8F6F4` |
| `V_WMMA_F64_16X16X4_F64` | *absent* | `Write4PassWMMA` (VALU, 16 cyc) |

And the scalar side inverts:

| | gfx1250 | gfx1251 |
|---|---|---|
| `WriteDouble` | 37 cyc | **6 cyc** |
| `WriteDoubleAdd` | 37 | **5** |
| `WriteTrans64` | 38 | **7** |

So the two are **complementary SKUs, not revisions**: gfx1250 is the
low-precision matrix part (4–16 cycle WMMA, 37-cycle FP64); gfx1251 is the FP64
part (16–32 cycle WMMA, 6-cycle FP64, plus an FP64 WMMA gfx1250 lacks). **[V]**

Shared latencies from `GFX125xCommonWriteRes`, useful as a first-order cost
model **[V]**:

| Class | Cycles |
|---|---|
| `WriteVMEM` | **320** |
| `WriteLDS` / `WriteSMEM` | 20 |
| `WriteBranch` | 32 |
| `WriteExport` | 16 |
| `WriteSFPU` | 4 |
| `WriteVALUDummy` | 5 |
| `WriteSALU` | 2 |
| `WriteBarrier` | **2000** |

The 320-cycle VMEM and 2000-cycle barrier are the numbers that make the
pipelining arithmetic in AMD's decode work (§8) come out the way it does.

---

## 3. Data movement — three distinct mechanisms

gfx1250 has three ways to move data that a compiler must model separately. They
use different instructions, different completion counters, and have different
arch availability.

### 3.1 Async global→LDS

```
global_load_async_to_lds_b{8,32,64,128}      → ASYNCcnt
global_store_async_from_lds_b128             → ASYNCcnt
```

Intrinsics: `int_amdgcn_global_load_async_to_lds_b*`, signature
`(global_ptr, lds_ptr, offset_imm, cachepolicy_imm)`. **[V]** All assemble on
gfx1250. **[A]**

**gfx1250 does not have `global_load_lds`** — that is the gfx950 mechanism. The
instruction *and* the counter both differ. Code keyed on "async global→LDS" as a
single concept across gfx950 and gfx1250 is wrong on both axes. **[A]**

### 3.2 TDM — `tensor_load_to_lds`

Descriptor-driven bulk tensor transfer; the AMD analog of NVIDIA's TMA.
**gfx1250-exclusive** — rejected by the assembler on gfx950, gfx1151, and
gfx1201. **[A]**

```c
int_amdgcn_tensor_load_to_lds(v4i32 D#g0, v8i32 D#g1, v4i32 D#g2,
                              v4i32 D#g3, v8i32 D#g4, i32 cachepolicy)
```

- **28 i32 (112 B) of descriptor state** across five register groups. Groups 2
  and 3 are zero for ≤2D tensors; group 4 is reserved and silently ignored.
- **No pointer operands.** Both the global address and the LDS destination live
  inside the descriptor. This is a genuinely different op shape from a
  `(dst, src, bytes)` copy — not something a byte-count copy op grows into.
- `cachepolicy`: bits[0-2] `th`, bits[3-4] `scope`.
- `IntrConvergent` — a wave-collective operation.
- Completion is **TENSORcnt**, not ASYNCcnt. **[V]**

Assembly forms **[A]**:

```asm
tensor_load_to_lds    s[0:3], s[4:11]                          ; ≤2D short form
tensor_load_to_lds    s[0:3], s[4:11], s[12:15], s[16:19]      ; full 4-group
tensor_load_to_lds    s[0:3], s[4:11] th:TH_LOAD_NT scope:SCOPE_SYS
tensor_store_from_lds s[0:3], s[4:11]
```

12-byte encoding; unused descriptor groups encode as the null-SGPR `0x7c`.

Production usage: AMD's Tensile gfx1250 pipeline anchors its entire
cluster-barrier insertion pass on `tensor_load_to_lds` sites, and
`stinkytofu`'s hardware model registers it as a first-class instruction. **[V]**

There are also `TENSOR_SAVE` / `TENSOR_STOP` VFLAT instructions (opcodes
0x06e/0x06f) for context save/preemption of the tensor unit — not part of the
data path. **[V]**

**Open:** the D# *field* layout — what the 28 dwords actually contain — is not
in LLVM (the intrinsic passes register groups through opaquely). It needs the
gfx1250 ISA guide or a Tensile descriptor builder. See §11.

### 3.3 Transposed LDS reads

`ds_load_tr8_b64` and family (`FeatureTransposeLoadF4F6Insts`) — read LDS
directly into the WMMA B-operand layout. Assembles on gfx1250. **[A]**

Measured caveat **[S]**: AMD built this path (`ds_load_tr16_b128`) for decode
and measured it **neutral** — the `ds_bpermute` lane-pair stitch needed to
assemble the K=32 B operand ate the LDS-read savings. See §8.

---

## 4. Completion model — split wait counters

gfx1250 replaces the single combined counter with a split model
(`waitcnt_model = split_gfx1250`). All of these assemble **[A]**:

```
s_wait_loadcnt  s_wait_storecnt  s_wait_dscnt  s_wait_kmcnt
s_wait_asynccnt  s_wait_tensorcnt  s_wait_xcnt
```

Mapping, per `AMDGPUUsage.rst` "Memory Model GFX125x" **[V]**:

| Operation | Counter |
|---|---|
| Load (global, scratch, flat, buffer) | `s_wait_loadcnt` |
| Store (global, scratch, flat, buffer) | `s_wait_storecnt` |
| non-ASYNC LDS | `s_wait_dscnt` |
| **ASYNC LDS** (`global_load_async_to_lds_*`, `cluster_load_async_to_lds_*`) | `s_wait_asynccnt` |
| **Tensor** (`tensor_load_to_lds` / `tensor_store_from_lds`) | `s_wait_tensorcnt` |
| scalar memory (`s_load_*`) | `s_wait_kmcnt` |

**`s_wait_xcnt` is different in kind** — it increments when a memory operation is
*issued* and decrements when that instruction's **address translation** completes.
Waiting on any memory counter `s_wait_*cnt N` also waits on `s_wait_xcnt N`. It
carries one hard correctness requirement **[V]**:

> `s_wait_xcnt 0x0` is required before flat and global atomic stores /
> read-modify-write operations to guarantee atomicity during an xnack replay.

Two ordering caveats that constrain any scheduler **[V]**:

- Completion (counter decrement) is reported **in issue order within a type**, but
  in **no particular order between types**.
- The order in which *data reaches registers* can differ from issue order even
  though completion is reported in order — so a `s_wait_*cnt` is required to stop
  two in-flight loads targeting the same register from racing.

### 4.1 Waits are positional, not identity-based

This is the single most important semantic for anyone lowering async ops:

> `s_wait_*cnt N` blocks until **at most `N`** ops of that kind remain
> outstanding (it keeps the `N` most-recently-issued in flight and drains
> everything older). **[S]**

```
w(D) = n - i - 1     # producer D at index i (0 = oldest) in a FIFO of size n
                     # emitted wait = countFrom(D) - 1
                     # min across all deps constraining the same counter
```

**You cannot say "wait for this specific copy."** You can only bound how many
are outstanding, and computing that bound requires the producer's FIFO position
along every CFG path reaching the consumer. `N = 0` is always correct and always
a full drain — it forfeits exactly the overlap async copies exist to provide.

### 4.2 How AMD lowers tokens into positions

AMD's production solution is instructive because it starts from the same
abstraction we do. `StinkyBuildImplicitDependencyPass` attaches `MemTokenData`
token IDs to tensor loads, DS ops, and barriers, materializing LDS ordering as
pseudo-register defs/uses — because "a `tensor_load_to_lds` writes an LDS region
and a later `ds_read` of that region depends on it, but there is no vreg linking
them." Then SSA def-use with PHIs at dominance frontiers feeds a forward
dataflow solver. **[S]**

The parts that make it work:

- Per-counter FIFO models **tagged per CFG predecessor edge**, so a join
  consumer sees each path's depth rather than a collapsed union.
- PHI summaries at merges, taking `min` of `countFrom(src) - 1` over constrained
  incoming paths.
- A documented escape hatch: on iteration-cap hit, force every counter to 0 —
  "a fully-drained, always-safe plan."
- Anti-dependencies (WAR-on-LDS, barrier ordering) still come from token overlap,
  not the SSA RAW chain.

**Wave-count-dependent policy** **[S]**: tensor-counter RAW deps drain only at
barriers in multi-wave kernels (cross-wave LDS visibility is the barrier's job),
but at *every consumer* when `NumWaves == 1`. Wave count changes wait placement.

### 4.3 The LLVM lowering seam — `asyncmark`

**This is the most important section in this document for backend work.**

`SIInsertWaitcnts.cpp` models `ASYNC_CNT` and `TENSOR_CNT`, but **not the same
way it models the other counters**, and the difference defines the seam between
what LLVM does and what a frontend must do. **[V]**

The official documentation states it outright:

> ASYNC LDS and tensor vector memory operations are **not covered by the memory
> model** implemented by the AMDGPU backend. Neither `s_wait_asynccnt` nor
> `s_wait_tensorcnt` are inserted automatically. **They must be emitted using
> compiler built-in calls.**
> — `AMDGPUUsage.rst`, "Memory Model GFX125x"

and the implementation agrees:

> `AsyncCnt` and `TensorCnt` always default to `~0u` (don't wait for it). They
> are only updated when a call to `@llvm.amdgcn.wait.asyncmark()` is processed.
> — `SIInsertWaitcnts.cpp:231`

LLVM's normal waitcnt machinery is **register-dependency driven** (score
brackets over vreg defs/uses). An async LDS-DMA or TDM load writes an *LDS
region*; there is no register linking it to a later `ds_read`. LLVM therefore
cannot infer the dependency — the same reason AMD's asm-level solver needed
`MemTokenData` pseudo-registers (§4.2). So for these two counters LLVM does not
attempt inference, and instead exposes an explicit marker protocol.

**Two meta intrinsics** — both emit *no hardware instruction*; they are consumed
by the pass, which emits the real `s_wait_asynccnt` / `s_wait_tensorcnt`
immediates **[V]**:

```llvm
; "Sets a marker in the stream of async requests"
declare void @llvm.amdgcn.asyncmark()                ; __builtin_amdgcn_asyncmark
; "Waits until the Nth previous marker is completed, if it exists"
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) ; __builtin_amdgcn_wait_asyncmark
```

Gated by `hasAsyncMark()` = `HasVMemToLDSLoad && GFX1250Plus` (`AMDGPU.td:1458`).

**Mechanism** **[V]**:

1. `AsyncScore[T]` accumulates a per-counter snapshot as async/tensor ops are
   seen (`shouldUpdateAsyncMark` routes TDM → `TENSOR_CNT`, async LDS-DMA →
   `ASYNC_CNT`, non-async LDS-DMA → `LOAD_CNT`).
2. `ASYNCMARK` → `recordAsyncMark()`: pushes `AsyncScore` onto the `AsyncMarks`
   vector and **resets it**. Each mark therefore captures the *batch* of async
   ops issued since the previous mark.
3. `WAIT_ASYNCMARK N` → `determineAsyncWait(N)`: indexes
   `AsyncMarks[size - N - 1]`, derives the real per-counter immediate via
   `determineWaitForScore`, then **erases that mark and all older ones**.

**Division of labour:**

| LLVM owns | The frontend owns |
|---|---|
| Counter arithmetic and score brackets | **Where the markers go** |
| CFG join merging (`mergeAsyncMarks`) | The value of `N` |
| Loop handling and mark truncation | |
| Overflow guards (`min(UB - Score, getLimit(T) - 1)`) | |
| Emitting the final `s_wait_*cnt N` | |

So the answer to "must we build a FIFO dataflow solver?" is **no** — LLVM does
the dataflow, including joins. What a frontend must supply is *batch marking*,
which is a substantially smaller and more local problem, and one that maps
naturally onto a token-typed async op.

**Semantics that bite** **[V]**:

- **A too-large `N` produces no wait at all**, silently: `if (AsyncMarks.size()
  <= N) return {};`. This matches the intrinsic's "if it exists" wording. The
  failure mode of a wrong `N` is therefore a **silent race, not a conservative
  stall** — correctness sits entirely with the producer of the marks, so any
  lowering needs its own verification rather than trusting LLVM to catch it.
- **`MaxAsyncMarks = 16`.** At the cap, `N = min(N, 15)`. The comment is explicit
  that this exists to *ensure a non-trivial wait is still generated* after a
  merge truncation — so the clamp errs toward waiting more, not less.
- **Marks are consumed.** After servicing, the waited mark and all older ones are
  erased, so indices are relative to the live set, not to absolute program order.
- **At joins**, `mergeAsyncMarks` pads the shorter list with zero-marks at the
  *front* and merges pairwise from the end — marks align **by recency**, not by
  absolute index.
- **Calls do not drain these counters.** `Inst.isCall()` applies a blanket wait
  "but `AsyncCnt` and `TensorCnt` are never included in such blanket waits"
  (`:2733`). Async state survives a call boundary.
- `ASYNCMARK` blocks waitcnt merging across it, so a mark is also a scheduling
  barrier for wait placement.

---

## 5. Workgroup clusters

`FeatureClusters` appears in exactly two feature sets in all of `AMDGPU.td`:
`FeatureISAVersion12_50_Common` (gfx1250/gfx1251) and `FeatureISAVersion13`.
**gfx950 does not have it.** **[V]**

### 5.1 Barrier encoding

There is no distinct cluster-barrier opcode in emitted assembly. Scope is
selected by **barrier ID** **[S]**, all four forms assembling on gfx1250 **[A]**:

```asm
s_barrier_signal -1  /  s_barrier_wait -1      ; workgroup scope
s_barrier_signal -3  /  s_barrier_wait -3      ; cluster scope
```

LLVM also exposes `int_amdgcn_s_cluster_barrier` /
`__builtin_amdgcn_s_cluster_barrier` as a convenience wrapper **[V]**, but
Tensile emits the raw `-3` form.

### 5.2 Clusters are Shader-Engine scoped — the key architectural fact

`cluster` is a **first-class LLVM IR syncscope**, and on gfx125x it lowers to
`scope:SCOPE_SE` **[V]**:

| LLVM syncscope | ISA |
|---|---|
| *none*, `one-as` | `scope:SCOPE_SYS` |
| `system`, `system-one-as` | `scope:SCOPE_SYS` |
| `agent`, `agent-one-as` | `scope:SCOPE_DEV` |
| **`cluster`, `cluster-one-as`** | **`scope:SCOPE_SE`** |
| `workgroup` / `wavefront` / `singlethread` (+ `-one-as`) | `scope:SCOPE_CU` (default, omitted in asm) |

That single row explains the whole cluster design. A cluster lives **within one
Shader Engine**, and the SE-shared **L1 buffer** (§1.2) is its coherence point.
Multicast-into-peer-LDS works because the participating WGPs sit behind a common
L1 — which is also why there is no distributed shared address space (§5.7): the
sharing happens in the cache hierarchy, not the address space.

Semantics of the scope **[V]**: `cluster` synchronizes with `system`, `agent`, or
`cluster` operations executed by a thread **on the same cluster**, plus
`workgroup`/`wavefront` operations in the same work-group/wavefront, for all
address spaces except private. Critically:

> On targets that do not support workgroup cluster launch mode, this behaves like
> `agent` scope instead.

So `cluster` syncscope is **portable by construction** — it degrades to `agent`
rather than failing to compile. That makes it safe to emit unconditionally from a
target-independent layer.

### 5.3 Compile-time cluster declaration

| Mechanism | Meaning |
|---|---|
| `"amdgpu-cluster-dims"="x,y,z"` fn attr | `"0,0,0"` = cluster disabled; `"1024,1024,1024"` = enabled but dimensions unknown at compile time; anything else = explicit dims. Only meaningful on targets with cluster support. **[V]** |
| `.cluster_dims` | Cluster dimensions recorded in the code-object metadata. **[V]** |
| `"amdgpu-no-cluster-id-{x,y,z}"` | Asserts the kernel never reads the corresponding `llvm.amdgcn.cluster.id.*`, enabling preload/ABI trimming. **[V]** |

The `"1024,1024,1024"` sentinel is worth noting: it distinguishes "clusters are
on but the shape is dynamic" from "clusters are off", which a lowering must not
conflate.

### 5.4 Documented gaps — read before planning cluster work

`AMDGPUUsage.rst`'s GFX125x memory model carries an explicit incompleteness note
**[V]**:

> This section is currently incomplete as work on the compiler is still ongoing.
> The following is a non-exhaustive list of unimplemented/undocumented features:
> non-volatile bit code sequences, globally accessing scratch atomics,
> **multicast loads**, **barriers (including split barriers) and cooperative
> atomics**. Scalar operations memory model needs more elaboration as well.

So the two mechanisms clusters are *built on* — multicast loads and cluster
barriers — are not yet covered by the backend's documented memory model, even
though the instructions encode and the intrinsics exist. Anything built on them
today is ahead of the formal model. **[I]**

### 5.5 Cooperative atomics

A separate gfx125x mechanism worth recording: wide cooperative load/store across
naturally-aligned, contiguous lane groups within one wave32 **[V]**:

| Intrinsic | Lane groups |
|---|---|
| `llvm.amdgcn.cooperative.atomic.{load,store}.32x4B` | `0-31` |
| `llvm.amdgcn.cooperative.atomic.{load,store}.16x8B` | `0-15`, `16-31` |
| `llvm.amdgcn.cooperative.atomic.{load,store}.8x16B` | `0-7`, `8-15`, `16-23`, `24-31` |

Undefined behaviour if used outside the global address space, across a bus that
cannot carry 128B/256B requests (e.g. host memory over PCIe), with an unsupported
lane group, or with more lane groups per wave than the maximum.

### 5.6 Identity

```
__builtin_amdgcn_cluster_id{_x,_y,_z}
__builtin_amdgcn_cluster_workgroup_id{_x,_y,_z}
__builtin_amdgcn_cluster_workgroup_flat_id
__builtin_amdgcn_cluster_workgroup_max_id{_x,_y,_z}
__builtin_amdgcn_cluster_workgroup_max_flat_id
```
`IntrinsicsAMDGPU.td:168-178`. **[V]**

### 5.7 Multicast — and what clusters are *not*

```c
AMDGPUAsyncClusterLoadLDS: (global_ptr, lds_ptr, offset, cachepolicy,
                            workgroup_broadcast_mask → M0)
int_amdgcn_cluster_load_async_to_lds_b{8,32,64,128}
int_amdgcn_cluster_load_b{32,64,128}
```

**AMD clusters are a replication/broadcast primitive plus a barrier — not a
shared address space.** A grep of the full intrinsics file and the HIP API
surfaces no peer-LDS addressing, no `dsmem` analog, no cluster address space.
NVIDIA's `cluster.map_shared_rank` has no counterpart. Data is *multicast into
each workgroup's own LDS*, selected by a broadcast bitmask. **[V]**

Porting an NVIDIA CGA kernel that reads a peer CTA's shared memory will not
lower. The correct mental model is TMA multicast, not distributed shared memory.

### 5.8 Host launch

HIP ≥ 7.0. **[V]**

```c
hipLaunchAttribute attr[1];
attr[0].id = hipLaunchAttributeClusterDimension;
attr[0].val.clusterDim = {2, 1, 1};   // must evenly divide gridDim
config.attrs = attr; config.numAttrs = 1;
hipLaunchKernelExC(&config, kernel, params);
```

Supporting surface: `hipClusterSchedulingPolicy{Default,Spread,LoadBalancing}`,
`hipOccupancyMaxActiveClusters`, `hipOccupancyMaxPotentialClusterSize`,
`hipFuncAttributeRequiredCluster{Width,Height,Depth}`,
`hipFuncAttributeClusterDimMustBeSet`,
`hipFuncAttributeNonPortableClusterSizeAllowed`, `hipErrorInvalidClusterSize`.

Device-side directive: `__attribute__((cluster_dims(x,y,z)))` (`Attr.td:1626`),
mutually exclusive with `no_cluster`; HIP wraps it as `CLUSTER_DIMS(X,Y,Z)`. **[V]**

**Capability detection is runtime, not arch-keyed** (`hip_device.cpp:739`) **[V]**:

```c
// A cluster of size 1 is a regular single-block launch (legal on all GPUs);
// clusterLaunch advertises multi-block cluster support...
deviceProps.clusterLaunch = info.clusterMaxSize_ > 1;
```

The canonical guard in AMD's own tests is `devProp.clusterLaunch != 0`.

### 5.9 The correctness discipline

From AMD's Tensile cluster-barrier insertion pass **[S]** — five rules whose
entire purpose is keeping `signal -3` / `wait -3` **paired on every
control-flow path**:

- Every cluster signal is preceded by a workgroup-scope `signal -1` / `wait -1`
  pair, so all waves reach the join before any wave issues the cluster signal.
- Only `WaveIdx == 0` executes the cluster signal.
- The hard cases are loop-entry guards and drain iterations where the paired
  `tensor_load_to_lds` is disabled — the handshake must be suppressed on exactly
  the same paths, or the pairing breaks.
- Each rule carries its own idempotency check so re-running is a no-op.

Unbalanced signal/wait is the failure mode, and it is a control-flow problem,
not a local one.

---

## 6. Device-initiated SDMA (inter-GPU)

Distinct from TDM. TDM is shader-issued, intra-GPU, global→LDS. SDMA is a
separate copy engine, and this code drives it **from inside a kernel** for
inter-GPU transfer. Implemented in `rocshmem/src/sdma/`, consumed by RCCL's
`anvil_sdma` GIN backend. **[V]**

**Arch support is a closed set**: gfx90a, gfx942, gfx950, gfx1250. Every other
target hits `LOGD_ERROR_ABORT("SDMA is not supported on this architecture")`.
**gfx1151 is not supported.** **[V]**

### 6.1 Packet ISA — OSS7.0

`sdma_pkt_struct_mi4.h`: "OSS7.0 SDMA packet structures (CDNA4 / MI350X and
later)", auto-generated from `OSS_70-sDMA_MAS.md`. **[V]**

Base opcodes (all generations): `NOP=0`, `COPY=1`, `WRITE=2`, `FENCE=5`,
`TRAP=6`, `POLL_REGMEM=8`, `ATOMIC=10`, `CONST_FILL=11`, `TIMESTAMP=13`.

MI4-specific sub-opcodes **[V]**:

| Sub-op | Value | Capability |
|---|---|---|
| `COPY_LINEAR_WAIT_SIGNAL` | 0x0 | **fused wait → copy → signal** |
| `COPY_LINEAR_PHY` | 0x8 | physical-address copy |
| `COPY_SWAP_WAIT_SIGNAL` | 0x9 | swap + fused wait/signal |
| `COPY_LINEAR_MULTICAST` | 0xa | **one copy, many destinations** |
| `COPY_MULTICAST_WAIT_SIGNAL` | 0xa | multicast + fused wait/signal |
| `COPY_PAGE_TRANSFER` | 0xc | page-granular transfer |
| `FENCE_64B` / `POLL_MEM_64B` | 0x2 / 0x5 | 64-bit fence / poll |
| `CONSTANT_FILL_PAGE` | 0x4 | page-granular fill |

Plus `SDMA_SIGNAL_OP_ADD64_MI4 = 111`, `SDMA_WAIT_FUNC_GEQ_MI4 = 5`.

### 6.2 The fused packet

`SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4` — **19 DWORDs**, `static_assert`-checked **[V]**:

```
header : op, subop, tmz, npd, wait, signal          ← wait/signal are enable bits
wait   : wait_function, wait_scope, wait_temporal_hint,
         wait_addr_31_3 / _63_32,                   ← bit 3: 8-byte aligned
         wait_reference_31_0 / _63_32,              ← 64-bit reference
         wait_mask_31_0 / _63_32                    ← 64-bit mask
copy   : copy_count,
         src_scope, src_temporal_hint,
         dst_scope, dst_temporal_hint,
         src_addr_31_0 / _63_32, dst_addr_31_0 / _63_32
```

One descriptor expresses: **block until `(mem[wait_addr] & mask) GEQ reference`,
then copy `src → dst`, then signal** — a complete producer-consumer handoff with
independent cache scope and temporal hint on each of the wait, source, and
destination sides.

Sizes for comparison: `COPY_LINEAR_PHY` 8 DW, `FENCE` 4 DW, `FENCE_64B` 5 DW.
The fused packet costs ~2.4× a plain copy in ring space, which is why callers
gate it behind `useSdmaFusedSignal(...)`.

### 6.3 Device-side ring protocol

1 MB ring (`SDMA_QUEUE_SIZE`, "matches rocm-xio sdma-ep"), queue created through
`hsakmt`. **[V]**

- **Reservation**: lock-free multi-producer CAS on `cachedWptr`. `CanWriteUpto`
  caches `rptr` and only re-reads the hardware register when the cached view says
  full.
- **Wraparound**: pads with NOPs, count encoded in the first DWORD as
  `((numOffsetDwords - 1) & 0xFFFF) << 16`.
- **Packet write**: `static_assert(sizeof(PacketType)/sizeof(uint32_t) <= 64)` —
  "Ensure that one warp can write the whole packet."
- **Commit is strictly in reservation order**: spin until `committedWptr == base`.
- **Publish**: three stores, each separated by a full drain,
  `__builtin_amdgcn_wave_barrier()`, and a signal fence —
  `wptr` (AGENT) → `doorbell` (SYSTEM) → `committedWptr` (AGENT).
- **Completion**: `quietAll()` spin-polls `rptr` until it reaches the target.
- A `SdmaQueueSingleProducerDeviceHandle` subclass drops the CAS for the
  one-thread-per-queue case, with identical binary layout.

### 6.4 gfx1250-specific divergences

Two places where the chip differs from CDNA, both confirming §4 **[V]**:

```c
// memory drains
#if defined(__gfx1250__)
  asm volatile("s_wait_loadcnt 0x0\n s_wait_storecnt 0x0" ::: "memory");
#elif defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__)
  __builtin_amdgcn_s_waitcnt(0);
#endif

// atomics — different mnemonic AND different cache-modifier syntax
// gfx1250: flat_atomic_cmpswap_b64 %0, %1, %2 scope:SCOPE_SYS nt
// CDNA:    flat_atomic_cmpswap_x2  %0, %1, %2 sc0 nt
```

gfx1250 replaced CDNA's `sc0`/`sc1` bit flags with **named scopes**
(`scope:SCOPE_SYS`) across both the DMA path and the TDM instructions.

### 6.5 SIMD reconvergence deadlock

From `submitPacket` **[S]**:

> All stores inside the loop to avoid SIMD reconvergence deadlock: the
> `committedWptr` update must complete before this lane becomes inactive, so
> that other lanes in the same wavefront can proceed.

A lock-free in-order commit protocol executed by lanes of the same wavefront can
deadlock on reconvergence: if a lane wins its turn, exits the loop, and goes
inactive before publishing, the remaining lanes spin forever on a turn that is
never released. The fix is structural — keep every store inside the loop body.
**Any device-side lock-free protocol has this failure mode.**

---

## 7. Scale-up fabric — UALink / IFoE

Not compiler-facing today, recorded for the distributed track.

**Stack** **[V]**:

| Layer | Artifact |
|---|---|
| Kernel | `ifoe.ko`, `ifoe-cfg.ko`, `ifoe-cmd.ko`; `/dev/cbl-cfg-ifoe.cfg.0` |
| sysfs | `/sys/class/drm/renderD<N>/device/ualink/` — presence gates everything |
| amdsmi | `projects/amdsmi/{src,include}/ualoe_lib/` (~350 KB of headers) |
| ROCr | `props->FabricHandleSupported` in HSA node properties |
| RCCL | `ARSMI_get_fabric_info()` → `struct ARSMI_fabricInfo` |
| HIP | `hipMemFabricHandle_t`, `hipMemHandleTypeFabric = 0x8`, `hipDeviceAttributeHandleTypeFabricSupported` |
| rocSHMEM | `src/memory/hip_allocator_vmm_fabric.cpp` |

**Topology model** **[V]**:

```c
typedef enum { ARSMI_FABRIC_TYPE_UALOE = 0,      // over Ethernet
               ARSMI_FABRIC_TYPE_UALLINK = 1,    // native
               ARSMI_FABRIC_TYPE_UNKNOWN = 2 } ARSMI_fabric_type_t;

struct ARSMI_fabricInfo {
  int supported; ARSMI_fabric_type_t fabric_type;
  ARSMI_fabric_accelerator_vpod_state_t accel_state;   // UNCONFIGURED→CONFIGURED→READY→ACTIVE→ERROR
  ARSMI_fabric_npa_address_mode_t addr_mode;
  uint32_t accel_id;
  uint8_t  ppod_id[16]; uint32_t ppod_size;            // physical pod
  uint32_t bandwidth;   // Mb/s
  uint32_t latency;     // ns
  uint32_t vpod_id;     uint32_t vpod_size;            // virtual pod (the partition you run in)
};
```

Three points worth carrying forward **[I]**:

1. Ethernet-transported and native UALink are two values of one enum behind one
   query — software does not branch on transport, it reads bandwidth/latency.
2. **The fabric reports its own bandwidth and latency.** A planner can query the
   interconnect cost model instead of hardcoding it.
3. Physical vs virtual pod are separate; collective planning must key on the
   **virtual** pod, since a rack may be partitioned.

**Device-initiated comms** (RCCL GIN — upstream NCCL device API with pluggable
backends: `gdaki`, `proxy`, `rocshmem_gda`, `anvil_sdma`) **[V]**:

- Peer addressing is **arithmetic**: `base + (peer - rank) * vmmStride + off`.
  VMM fabric handles map every peer's buffer at uniform stride in one VA space.
  Fallbacks: `remote_vas[peer]`, then an IPC table scan.
- "LSA memory is VMM-mapped fine-grain (cache-coherent via Infinity Fabric).
  Plain stores are visible to all peers through the HW coherence domain."
- **Threshold-driven transport arbiter**: `bytes <= rsCtx->sdmaThreshold` →
  `ipcPut` (coherent stores); else the SDMA queue; degrades to `ipcPut` if no
  queue handle. This is a measured arbiter over two implementations of one
  logical op — the same shape as Decision #28's kernel arbiter, applied to
  transport.
- **Fencing is path-dependent**: `sdma_anvil::quiet(handle)` vs
  `__builtin_amdgcn_fence(RELEASE, "agent")` vs `__threadfence_system()`,
  selected by which data path the put took, plus a `cuda::thread_scope`
  negotiation that skips the fence when the caller already gave a stronger scope.

An **[I]** worth flagging: MCDI, and block names like the **EX** switch,
**XRSEC** crypto, and **XRPFC** flow control, are distinctly Solarflare/Xilinx
vocabulary. This reads like Xilinx/Solarflare NIC IP repurposed as the fabric
NIC, but the source never says so — treat as hypothesis.

---

## 8. Measured lessons from AMD's own gfx1250 decode kernels

From `rocke`'s MHA/decode optimization case study **[S]**. The decode kernel is
memory-latency/bandwidth bound (GEMV-like, arithmetic intensity ≈ 1), so
compute- and LDS-oriented levers do not touch the bottleneck.

**Worked:**

| Lever | Result |
|---|---|
| **DPP `row_xmask` softmax butterfly** | replaced a 4-stage `ds_swizzle_b32` chain (LDS-port, `lgkmcnt`-serialized) with a VALU DPP butterfly: **128 `ds_swizzle_b32` → 0**; **1.4–1.56×** at low wave count; numerically exact (`max_abs ≈ 4e-5`) |
| Cooperative multi-wave CTA | up to ~1.5× on **small batch only** |
| `num_segments` (split-KV) | AITER-style occupancy knob; helps small batch, clamp for large |

**Did not work** — the recurring cause is the **wave32 cross-lane tax**:
optimizations that win on gfx950 (wave64, MFMA) lose on gfx1250 because
assembling operands costs more shuffles.

| Lever | Result |
|---|---|
| Register-resident P (skip P→LDS) | **~2× slower** (259→671 µs) |
| HW transpose LDS reads (`ds_load_tr16_b128`) | **neutral** — `ds_bpermute` stitch eats the savings |
| SW pipeline + `iglp_opt` / `sched_group_barrier` | **neutral** — waitcnt already tuned; softmax chain was the real critical path |
| Double-buffered V | **slower** (259→335 µs) — halves occupancy to hide already-hidden latency |
| Multi-wave at large batch | **4–10× slower** (259→1061 µs) — device already saturated |
| Native fp8 PV GEMM | not built — ceiling probe showed PV+V-staging is **0.6%** of the kernel |

**Methodology worth adopting** **[S]**: AMD's WMMA layout probe uses **random
asymmetric A/B** deliberately — "a row/col swap in the lane map transposes the
result and fails verify, so a PASS at multiple tile counts uniquely confirms the
mapping." A symmetric or structured test matrix passes with a transposed lane
map. They also treat their own lane maps as "a hypothesis until proven on
silicon."

**Also worth adopting** **[S]**: the `ablate_pv` ceiling probe — before building
a from-scratch optimization, measure its *ceiling* by ablating the region it
would improve. That is what killed the native-fp8 PV work at 0.6% before any
implementation cost was paid.

---

## 9. Observations against the current Tessera tree

**These are observations from this survey, not an accepted backlog.** The
authoritative ROCm queue is the open-actions table in
[`ROCM_AUDIT.md`](ROCM_AUDIT.md); anything below needs its own verification and
triage before it earns an ID.

| # | Observation | Where |
|---|---|---|
| 1 | `cluster_mode` is inverted: asserted `"ready"` for gfx950 (which lacks `FeatureClusters`) and `"tba"` for gfx1250/1251 (which have it). `supports_cluster_mode()` is a live predicate over that dict, but has no codegen consumer today — latent, not miscompiling. | `python/tessera/compiler/rocm_target.py` |
| 2 | gfx1250 LDS is `65536` marked PROVISIONAL; grounded value is `327680`. | `rocm_target.py` |
| 3 | gfx1250 VGPR budget is `256` (RDNA-derived); grounded value is `1024`. Matters because `rocm_tiling.py`'s thesis is that the register budget is the dominant tiling lever. | `rocm_target.py` |
| 4 | `ROCM_WaitTokenOp` has **no wait immediate** — only a counter name, and the enum admits just `vmcnt`/`lgkmcnt`. It can express `N = 0` (full drain) and nothing else, which is correct but forfeits async overlap. The token-typed design matches AMD's own `MemTokenData`. **Scoped by §4.3:** LLVM owns the dataflow, so the work is *batch marking* — lower `async_copy` groups to `llvm.amdgcn.asyncmark()` and `wait(token)` to `llvm.amdgcn.wait.asyncmark(N)`, where `N` is the number of marks between the token's batch and the wait. No FIFO solver needed. Note the failure mode: a too-large `N` yields **no wait**, so this needs its own verification. | `src/compiler/codegen/Tessera_ROCM_Backend/include/TesseraROCM/IR/TesseraROCMOps.td` |
| 5 | `_GFX1250_CLASS_ARCHES = {gfx1250, gfx1251}` is correct for ABI, wrong for cost models (§2.4). Worth a boundary comment. | `python/tessera/compiler/rocdl_emit.py` |
| 6 | `capabilities.py` lists gfx1250 dtypes as bf16/fp16/fp32/int8 — no fp8 — while gfx950 above it carries fp8 and a `wmma_f8` flag. gfx1250 is the arch with the scaled-fp8 matrix path. | `python/tessera/compiler/capabilities.py` |
| 7 | Module docstring calls gfx950 "MI325X"; MI325X is CDNA 3 / gfx942. `gpu_target_map.py` has it right (MI350X/MI355X/MI350P). | `rocm_target.py` |
| 8 | Generated softmax/reduce kernels use `gpu.shuffle xor` butterflies, which lower to `ds_bpermute`/`ds_swizzle` — the exact LDS-port pattern AMD replaced for 1.4–1.56×. The DPP `row_xmask` lever is available on **gfx1151** with an arch-keyed mnemonic (`v_max_f32_dpp` on gfx1151; `v_max_num_f32_dpp` on gfx12+; `v_add_f32_dpp` is common). Needs the ROCM-6 A/B ratchet — DPP is a latency-hiding lever and can lose when VALU-throughput-bound. | `GenerateROCMSoftmaxKernel.cpp:89`, `GenerateROCMReduceKernel.cpp:125`, `GenerateROCMArgReduceKernel.cpp:113` |
| 9 | `attn_split_kv.py` (`plan_split_kv`) has no consumer outside its own unit test. AMD's `num_segments` is the same knob and is occupancy-gated in their production dispatcher. | `python/tessera/compiler/attn_split_kv.py` |
| 10 | WMMA fragment fixtures should use **random asymmetric** operands (§8) or they can pass with a transposed lane map. | `tests/unit/test_rocm_wmma_gemm_generated.py` |
| 11 | `distributed_planner.py` has no link bandwidth/latency terms. If it grows them, `(bandwidth, latency, ppod, vpod)` is the shape the hardware offers (§7). | `python/tessera/compiler/distributed_planner.py` |

### 9.1 The cross-cutting theme

Three independent mechanisms, one pattern:

| Mechanism | How the dependency is carried |
|---|---|
| TDM (`tensor_load_to_lds`) | descriptor register groups, no pointer operands |
| waitcnt (`s_wait_*cnt N`) | FIFO **position**, not identity |
| SDMA (`COPY_LINEAR_WAIT_SIGNAL`) | wait predicate fused **into** the copy descriptor |

AMD hardware consistently wants the dependency expressed **inside the
descriptor**, not as a separate ordering instruction the compiler infers.
Token-typed async ops are the right *source* abstraction — AMD's own design
confirms it — but every lowering target on this vendor converts tokens into a
positional count or a descriptor field. That argues for a single "async
dependency → target completion model" lowering interface rather than three
ad-hoc ones. **[I]**

---

## 10. Reproducing every claim

The assembler probes need no AMD hardware and no ROCm install — any LLVM with
the AMDGPU target works (verified on Homebrew LLVM 22.1.8, arm64 macOS).

```bash
LLC=$(brew --prefix llvm)/bin/llc
MC=$(brew --prefix llvm)/bin/llvm-mc

# Subtarget features visible to this LLVM
$LLC -march=amdgcn -mcpu=gfx1250 -mattr=help 2>&1 | grep -iE "cluster|async|1024|tensor"

# Does an instruction exist on this arch?
echo "tensor_load_to_lds s[0:3], s[4:11]" | $MC -arch=amdgcn -mcpu=gfx1250 -show-encoding
echo "s_barrier_signal -3"                | $MC -arch=amdgcn -mcpu=gfx1250 -show-encoding
echo "s_wait_tensorcnt 0"                 | $MC -arch=amdgcn -mcpu=gfx1250 -show-encoding
```

Fetching primary sources (needs an authenticated `gh`):

```bash
gh api -H "Accept: application/vnd.github.raw" \
  "repos/ROCm/llvm-project/contents/llvm/lib/Target/AMDGPU/AMDGPU.td?ref=amd-staging"
```

Useful paths: `llvm/lib/Target/AMDGPU/{AMDGPU.td,GCNHazardRecognizer.cpp}`,
`llvm/include/llvm/IR/IntrinsicsAMDGPU.td`, `clang/include/clang/Basic/Attr.td`
(all `ROCm/llvm-project@amd-staging`);
`projects/{hip,rccl,rocshmem,amdsmi,clr,hip-tests}` (`ROCm/rocm-systems@develop`);
`shared/stinkytofu`, `dnn-providers/hip-kernel-provider/rocke`
(`ROCm/rocm-libraries@develop`).

**Toolchain note.** Two LLVMs are present on the current Mac: the Homebrew keg
`llvm/22.1.8` (on `PATH`), and **LLVM 23.1.0-rc1 at
`/opt/homebrew/llvm-23.1.0-rc1/`** — the latter is what `build/CMakeCache.txt`
pins as `LLVM_DIR`. Note it is *not* at the `/opt/homebrew/opt/llvm@23` path
`CLAUDE.md` cites, so that reference is stale even though the toolchain itself is
present.

Both versions were probed and agree on everything in this document: each encodes
the gfx1250 WMMA family, TDM (`tensor_load_to_lds`), the cluster-scope barrier
IDs, and the split wait counters (`s_wait_tensorcnt`, `s_wait_asynccnt`), and
both reject the `s_cluster_barrier` convenience mnemonic. Either is sufficient
for the probes above.

---

## 11. Unexplored sources, ranked

Everything below is a real gap in this survey, not a formality.

> **Two former top items have been read.**
> **`SIInsertWaitcnts.cpp`** → §4.3: LLVM owns the counter dataflow including CFG
> joins; the frontend owns marker placement via
> `llvm.amdgcn.{asyncmark,wait.asyncmark}`. Resolves the question behind §9 #4.
> **`AMDGPUUsage.rst`** → §1.2 (cache hierarchy + SCOPE ladder), §4 (official
> counter table, `s_wait_xcnt`), §4.3 (official confirmation), §5.2–5.5 (cluster
> syncscope, `amdgpu-cluster-dims`, documented gaps, cooperative atomics).
>
> **A dead end worth recording:** `AMDGPUUsage.rst` refers three times to a
> section `amdgpu-dma-operations` for "full documentation" of the async-LDS and
> tensor intrinsics — and that anchor **is never defined anywhere in the file**.
> The TDM descriptor layout is not merely hard to find in LLVM's docs; the
> section that would hold it is an unwritten placeholder. **[V]**

**High value for compiler work:**

1. **`SIMemoryLegalizer.cpp`** — still worth reading for *how* the fence code
   sequences in the GFX125x tables are actually emitted, though
   `AMDGPUUsage.rst` now gives us the normative sequences themselves.
2. **`SISchedule.td` / the gfx1250 scheduler model** — the WMMA latencies
   (4/8/16/32) that drive §2.3 come from `computeInstrLatency`; the model behind
   them is unread.
3. **The gfx1250 ISA guide** — now confirmed as the *only* remaining path to the
   TDM D# field layout (§3.2), LDS/VGPR allocation granularity, and the
   SPG 4.6.12.1 hazard section LLVM cites. Neither LLVM's docs nor its tablegen
   carry the descriptor field definitions. `rocm_target.py` grounds gfx1151 in
   the published RDNA3.5 guide, so an equivalent document may exist.

**High value for kernel/codegen design:**

4. **`shared/stinkytofu` in full** — an entire asm-level optimizing compiler from
   AMD: pass pipeline, DAG scheduler (`ReadyQueue.hpp`, and a `CDNA5.hpp` we did
   not open), `AsmVerifierPass`, `stinkytofu-opt` tool, `waitcnt-check`. We read
   two docs out of many.
5. **`shared/rocroller`** — AMD's GEMM kernel generator with its own IR. It uses
   `tensor_load_to_lds` in `MemoryInstructions.cpp`, so it likely contains the
   descriptor construction we could not find.
6. **Triton's AMD backend** — `third_party/amd` in `triton-lang/triton`,
   including `mxfp_fa_gfx1250.py` (the blog's kernel) and the Gluon layout
   system. The concrete fragment layouts and the `cga_layout` field live here.
7. **`rocke`'s attention kernels themselves** — `wmma_attention_fwd.py`,
   `_wmma_attention_common.py`, `attention_tiled_2d.py`. We read the design docs,
   not the code.
8. **`projects/composablekernel`** — AMD's tile-programming library. Relevant to
    a tile-centric compiler on general principle; unexamined here.
9. **rocWMMA's gfx1250 support** — fragment layouts in a portable, documented form.

**Lower priority / situational:**

10. **MES (Micro Engine Scheduler)** for cluster scheduling — we already keep an
    RDNA MES write-up at `docs/reference/isa/rdna/mes/`; the gfx1250 cluster
    dispatch path likely involves it.
11. **`clr`/`hipamd` cluster implementation** — how `hipLaunchKernelExC` actually
    programs cluster dims into the AQL packet.
12. **`rocprofiler-sdk`** counters for TDM/cluster/SDMA — required before any
    measured gfx1250 work, irrelevant before it.
13. **KFD / kernel driver ABI** for TDM and cluster enablement.

---

## 12. Sources

**LLVM** (`ROCm/llvm-project@amd-staging`) — `llvm/lib/Target/AMDGPU/AMDGPU.td`,
`GCNHazardRecognizer.cpp`, `FLATInstructions.td`;
`llvm/include/llvm/IR/IntrinsicsAMDGPU.td`; `clang/include/clang/Basic/Attr.td`.

**ROCm systems** (`ROCm/rocm-systems@develop`) —
`projects/hip/include/hip/hip_runtime_api.h`;
`projects/clr/hipamd/src/hip_device.cpp`;
`projects/hip-tests/catch/unit/cluster/{hipClusterLaunch.cc,hipClusterCompilerDirective.cc,ClusterHelper.hpp}`;
`projects/rocshmem/src/sdma/{anvil_device.hpp,sdma_opcodes.h,sdma_pkt_struct_mi4.h}`;
`projects/rccl/src/include/nccl_device/gin/{gin_device_api.h,anvil_sdma/*}`;
`projects/rccl/src/include/alt_rsmi.h`;
`projects/rocr-runtime/libhsakmt/src/topology.c`;
`projects/amdsmi/{src,include}/ualoe_lib/*`.

**ROCm libraries** (`ROCm/rocm-libraries@develop`) —
`shared/stinkytofu/docs/developer/cluster-barrier.md`,
`shared/stinkytofu/docs/user/stinky-waitcnt-insertion-pass.md`,
`shared/stinkytofu/hardware/src/gfx/Gfx1250/Gfx1250.cpp`;
`dnn-providers/hip-kernel-provider/rocke/library/builders/gfx1250/attention/{gfx1250_mha_optimization_case_study.md,gfx1250_universal_attention_plan.md}`;
`dnn-providers/hip-kernel-provider/rocke/platform/python/rocke/examples/gfx1250/wmma_probe.py`.

**AMD blog** — "Attention Decode on MI450 with Gluon",
https://rocm.blogs.amd.com/software-tools-optimization/gluon-attention-decode-mi450/README.html
(secondary; every claim taken from it was re-derived above or is marked **[S]**).

---

*See also: [`ROCM_AUDIT.md`](ROCM_AUDIT.md) (status) ·
[`ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md`](ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md)
(ecosystem patterns) · [`STRIX_HALO_EXECUTION_PLAN.md`](STRIX_HALO_EXECUTION_PLAN.md)
(gfx1151 bring-up) · [`../BACKEND_AUDIT.md`](../BACKEND_AUDIT.md) (cross-backend).*
