---
last_updated: 2026-08-02
audit_role: reference
scope: TesseraROCMOps.td, TesseraAppleOps.td, TesseraNVIDIADialect.td, tessera_x86_backend, target_ir.py, GenerateROCM*Kernel passes, test_target_ir_contract.py
companions: IR_STACK_INTEGRATION_REVIEW.md · INTEGRATED_COMPILER_PLAN.md
---

# Target IR and the Schedule → Tile → Target Integration

Completes the IR-stack sweep. The
[previous review](IR_STACK_INTEGRATION_REVIEW.md) covered Graph → Schedule →
Tile and closed with an open question: *does the `Variadic<AnyType>` finding
repeat in the Target IR dialects?*

**It does — in the matrix-core operations, which is the worst possible place.**
And the sweep surfaced something the earlier reviews did not: one backend has no
Target IR dialect at all, and the two designated lead performance targets have
opposite codegen architectures.

Status truth stays with the generated dashboards (Decision #26).

---

## 0. Four backends, four different designs, one missing

| Backend | Dialect | Ops | Abstraction level | `AnyType` |
|---|---|---:|---|---:|
| ROCm | `tessera_rocm` | 70 | **hardware instruction** (`mfma`, `wmma`) + whole-kernel (`flash_attn`, `linear_attn`) | 10 |
| NVIDIA | `tessera_nvidia` | 24 | **hardware instruction** (`mma_sync`, `wgmma`) + quantization (`nvfp`, `mx_block_scale_mma`) | 3 |
| Apple | `tessera_apple` | 19 | **library call** (`accelerate_gemm`, `mps_matmul`, `metal_kernel`) — split CPU/GPU namespaces | 12 |
| x86 | **none** | — | — | — |

Across all three existing dialects: `RankedTensorType` / `MemRefOf` / `TensorOf`
appear **zero** times, and `EnumAttr` appears **zero** times.

---

## 1. Findings

### X1 — x86 has no Target IR dialect, which is a Decision #19 violation in the oldest backend

Decision #19 is unambiguous:

> New backends MUST expose a hardware-free Target IR dialect before
> hardware-specific lowering… **never lower Tile IR directly to PTX/HIP/Metal
> source.**

`find src/compiler/codegen/tessera_x86_backend -name '*.td'` returns nothing.
The only file in the tree mentioning `tessera_x86` in ODS is the *ROCm* dialect.
x86 lowers Tile IR straight to AVX-512/AMX kernels through `TileToX86Pass`.

The irony is that x86 is the backend `CLAUDE.md` describes as **"works
end-to-end"** and Decision #1 makes the CPU-first spine. The one backend with a
complete execution story is the one that skipped the architectural contract
written to make backends testable.

Two readings, and they need different responses:

- *If Decision #19 is right*, this is real debt: x86 needs a `tessera_x86`
  dialect (AMX tile ops, AVX-512 vector ops, packing/layout ops) before its
  lowering can be lit-tested the way ROCm's and Apple's are.
- *If x86's shortcut is fine* — because the CPU backend genuinely has no
  hardware-free intermediate worth naming — then Decision #19 needs an explicit
  carve-out saying so.

Silence is the one option that should be closed. Today the decision reads as
universal and one backend quietly doesn't follow it.

### X2 — The `AnyType` pattern repeats at Target IR, in the matrix-core ops

```tablegen
def ROCM_MFMAOp : TesseraROCM_Op<"mfma"> {
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$acc);
  let results   = (outs AnyType:$res);
}
```

Identical for `ROCM_WMMAOp`. These are the operations that *are* the backend —
the whole reason a ROCm Target IR exists is to name MFMA and WMMA before they
become text.

The nearby `ROCM_WMMAGemmOp` description documents one concrete f16 WMMA
materialization:

> …loads the A/B tiles into RDNA WMMA fragment vectors (`vector<16xf16>`), calls
> `tessera_rocm.wmma` (Stage J → real `rocdl.wmma`), and stores the
> `vector<8xf32>` accumulator with the wave32 lane/element layout.

Those vectors are evidence for that f16 gfx11 variant, not a universal signature
for `ROCM_WMMAOp`: the directive also admits bf16 and integer storage variants,
and MFMA has architecture/shape-dependent fragment forms.  NVIDIA's shared op
base likewise has no equivalent fixed-vector prose for all `mma_sync`/`wgmma`
variants.  The defect is still real—matrix contracts are open—but the fix must
be variant-aware types or verifier constraints keyed by architecture, shape,
operand dtype, accumulator dtype, and role.  Hard-coding the two f16 vector
types would reject valid families and encode one AMD schedule as fleet policy.

**This resizes Wave 1 of the [integrated plan](INTEGRATED_COMPILER_PLAN.md)** —
typing the Tile dialect was scoped at 3 weeks; the Target dialects add roughly
two more, and they are the cheaper half because the types are `vector<…>` rather
than new parameterized types.

### X3 — Target semantics are stringly typed; zero `EnumAttr` anywhere

Across `TesseraROCMOps.td`: **62 × `StrAttr:$name`**, 4 × `StrAttr:$kind`,
1 × `StrAttr:$mode`. Across all three target dialects: **no `EnumAttr`,
`I32EnumAttr`, or `StrEnumAttr` at all.**

This is the same defect class as the `manifold` `StrAttr` in the EBM dialect
([GA/EBM §1.1](../domain/GA_EBM_ARCHITECTURE_REVIEW.md)): a misspelled or
unrecognized `kind` parses cleanly and is discovered — if at all — by whatever
downstream code fails to match it. Decision #21a (semantic keys never default)
applies directly, and an ODS enum is the enforcement mechanism.

### X4 — The "Target IR contract" test is a substring assertion on Python-generated text

Decision #19 closes with: *"The hardware-free layer is what makes backends
lit-testable; validated by `test_target_ir_contract.py`."*

That file, in full for the ROCm case:

```python
assert 'target = "rocm"' in mm.target_ir
assert "tessera_rocm.mfma" in mm.target_ir
assert "tessera_rocm.async_copy" in mm.target_ir
assert "tessera_rocm.wait" in mm.target_ir
```

`mm.target_ir` is a **string produced by the Python `target_ir.py`**. The test
does not parse MLIR, does not construct a `MLIRContext`, does not load the
dialect, does not run the verifier, and does not check a single type. It asserts
that four substrings appear in generated text.

That is a useful smoke test and it is not a contract. The named validation for
Tessera's Target IR architectural decision is `str.__contains__`.

### X5 — The compatibility Target-IR path is Python-canonical; native packaging is mixed

`target_ir.py` (2004 lines) defines `lower_tile_to_target_ir` plus
`_lower_rocm_op`, `_lower_cpu_op`, and `_lower_apple_gpu_fusion` — hand-written
per-backend lowering functions, mirroring `_lower_schedule_ops` in `tile_ir.py`
and `_lower_graph_ops` in `schedule_ir.py`.

That is not, however, the whole production picture.  For the native package
families admitted by `rocm_native.native_package_kind`, canonical compilation
replaces the compatibility artifact with MLIR produced by `tessera-opt` and
lowers it through the C++ ROCm pipeline to ROCDL/HSACO.  The Python Target-IR
path remains canonical for compatibility/value artifacts and unsupported
families, but it is not the canonical Tile→Target implementation for the five
auto-promoted gfx1151 package families.

The full picture below Graph IR is therefore mixed:

| Boundary | Canonical | Parallel MLIR |
|---|---|---|
| Graph → Schedule | Python `lower_graph_to_schedule_ir` | `GraphToSchedulePass` (co-located with driver pipeline code, but compiled into the linkable `TesseraPM` library) |
| Schedule → Tile | Python `lower_schedule_to_tile_ir` | `TileIRLoweringPass` |
| Tile → Target | Python `lower_tile_to_target_ir` for compatibility/value artifacts; C++ MLIR for admitted native packages | 67 registered `GenerateROCM*` family passes; only a bounded subset participates in the default ROCm backend pipeline |

There is still no differential test that makes the ownership boundary explicit.
The cleanup target is not "replace three Python-canonical boundaries"; it is to
name one production owner for each `(target, package family)` and prove that any
retained compatibility/oracle path agrees at that boundary.  The integrated
plan's W3.2 still spans three boundaries, but must preserve the already-canonical
ROCm MLIR/LLVM package lane.

### X6 — ROCm has three differently scoped generation surfaces with unclear ownership

Decision #28 names ROCm and CUDA as *the* lead performance targets whose ceiling
shared infrastructure must never cap. Their codegen could hardly be less shared:

- **ROCm: 67 `GenerateROCM*Kernel.cpp` C++ passes** — one per op family.
  `GenerateROCMActivationKernel`, `GenerateROCMAlibiKernel`,
  `GenerateROCMControlForWmmaTileKernel`, `GenerateROCMControlScanRnnKernel`,
  `GenerateROCMDeltaNetKernel`, `GenerateROCMEbmAffineLangevinKernel`, …
- **NVIDIA: zero such passes.** Its codegen is Python — `emit/nvidia_cuda.py`,
  4722 lines.

The three ROCm surfaces are not interchangeable implementations of one job:

- the C++ passes own the canonical MLIR→ROCDL/HSACO native package lane;
- `target_ir.py::_lower_rocm_op` owns compatibility/value Target-IR text and
  host-free inspection for families not promoted to a native package; and
- `emit/rocm_hip.py` is an arbiter candidate for generic fused regions plus a
  runner for selected shipped kernels.  Its own contract explicitly keeps
  crown-jewel WMMA/MFMA codegen first-class.

The defect is therefore **unclear and weakly tested ownership**, not evidence
that all 67 passes should move to HIP source synthesis.  Moving the canonical
ROCm generators onto `emit/rocm_hip.py` would demote the backend from the
compiler-owned MLIR/LLVM spine to vendor-source synthesis and contradict the
execution-spine direction.  The right first move is an inventory and
differential gate keyed by package family; only genuinely duplicate producers
should later be retired.

### X7 — "Hardware-free" is doing less work than Decision #19 implies

`tessera_rocm.mfma` and `tessera_nvidia.wgmma` **are hardware instructions** —
one-to-one with `rocdl.mfma` and PTX `wgmma`. `tessera_apple.cpu.accelerate_gemm`
is a *library call*. These are not the same abstraction level, and that is why no
shared Tile → Target lowering exists: there is nothing common to lower *to*.

Decision #19's intent — a layer that is testable without hardware — is satisfied.
Its wording, "hardware-free," is not accurate for two of the three dialects and
invites the expectation of a portable middle layer that was never built. Worth
either rewording the decision or stating explicitly that Target IR is per-backend
by design and portability lives at Tile IR.

---

## 2. What is right here

- **The ROCm FP8 diagnostic.** From `ROCM_WMMAOp`'s description: *"RDNA 3 / 3.5
  (gfx1100, gfx1151) expose no FP8 WMMA — an FP8 matmul on those arches is a
  hard, named error in the Tile->ROCm lowering rather than a silent fallback."*
  That is Decision #21 done exactly right, grounded in real ISA data
  (`docs/reference/isa/rdna/`), and it is the counter-example to every fail-open
  finding in these reviews. It should be the template for X3's enum work.
- **The ROCm descriptions carry real hardware knowledge** — wave32 lane/element
  layout, instruction tile shapes, the RDNA-vs-CDNA distinction. The knowledge
  is present and precise. X2 is only about moving it from prose into ODS.
- **NVIDIA's dialect covers the frontier ops** — `nvfp`, `mx_block_scale_mma`,
  `fp_quant` alongside `mma_sync`/`wgmma`. The low-precision surface is modelled
  at Target IR rather than smuggled through attributes.

---

## 3. Updates

| # | Item | Effort | Where it lands |
|---|---|---|---|
| X-U1 | Type or verify Target-IR matrix ops with variant-aware `(arch, instruction, shape, operand dtype, accumulator dtype, role)` contracts; use the documented gfx11 f16 vectors as one fixture, not the universal signature | 2w design/migration estimate | **grows W1.1** |
| X-U2 | `EnumAttr` for every semantic `StrAttr:$kind`/`$mode` across the three target dialects (per #21a) | 1w | **grows W1.1** |
| X-U3 | Replace `test_target_ir_contract.py`'s substring assertions with a real parse + dialect-load + verifier run; keep the substring test as a smoke check | 1w | **new, Tier 0-adjacent** |
| X-U4 | Resolve x86's Decision #19 status: either build `tessera_x86` (AMX tile / AVX-512 vector / pack ops) or add an explicit carve-out to the decision | 3w *or* 1h | **new** — decide before building |
| X-U5 | Tile → Target joins the boundary convergence (three boundaries, not two) | — | **already in W3.2** |
| X-U6 | Define ROCm producer ownership per package family and add a differential gate between each production owner and any retained compatibility/oracle surface; preserve C++ MLIR→ROCDL/HSACO as the canonical native spine | 2w initial inventory/gate | **new, W3-adjacent** — see §4 |

X-U1 + X-U2 + X-U3 ≈ 4 weeks and are Mac-doable (ODS, lit, unit tests, no
hardware). X-U4's *decision* costs an hour; only the "build it" branch costs 3
weeks.

---

## 4. Fleet routing — what must run on which box

Per the standing fleet split (Mac M1 Max / Strix Halo gfx1151 / NR2 Pro sm_120),
and the user's note that ROCm work belongs on the ROCm system:

| Work | Box | Why |
|---|---|---|
| X-U1, X-U2 (ODS typing + enums) | **Mac** | ODS + `tessera-opt` + lit; no device. Tightening types is a compile-time contract change. |
| X-U3 (real contract test) | **Mac** | Parse + verify only. |
| X-U4 (x86 dialect, if built) | **Mac** for ODS/lit; **Zen5 box** for AMX/AVX-512 execution proof | The Core Ultra 7 in the NR2 Pro has no AVX-512/AMX. |
| **X-U6 (ROCm ownership + differential gate)** | **Host-free for inventory/IR equivalence; Strix Halo gfx1151 for any kernel-producing migration** | Merely classifying owners does not change codegen. Any later retirement or producer change must execute-and-compare on the owning device. |
| ROCm hardware-gated arch breadth (gfx950/gfx1201/gfx1250) | deferred — no silicon | MASTER_AUDIT P2. |

**Concrete recommendation for X-U6:** first classify every registered generator
as default-pipeline, native-package-only, compatibility-only, candidate, or
unconsumed.  Then add a differential harness per package family — the same
harness shape W3.1 needs for trace-vs-AST and W3.2 needs for
Python-spine-vs-MLIR.  Do not schedule a blanket 67-pass migration.  When a
specific producer is changed or retired, run its numerical comparison on
gfx1151.

X-U1's ROCm typing may produce compile failures in registered generators; those
are host-free contract failures and should be fixed before device work.  A
gfx1151 run becomes mandatory only when the selected producer or emitted kernel
changes, not merely because an ODS signature tightened.

---

## 5. Net effect on the integrated plan

| Wave | Change |
|---|---|
| **Tier 0 / W0** | + X-U3 (real contract test, 1w); + X-U4's *decision* (1h) |
| **W1** | W1.1 grows from 3w to **5w** — Tile IR **and** the three Target IR dialects; + X-U2 enums (1w). Wave 1 total 8w → **10w** |
| **W3.2** | unchanged in cost; confirmed as **three** boundaries |
| **W3** | + X-U6 ROCm producer-ownership inventory and differential gate (2w initial slice; device proof only for later producer changes) |
| **new** | X-U4 build branch (3w) only if Decision #19 is affirmed for x86 |

The previous numeric plan total is not retained: W1.1 includes variant-aware
type-system design, while the blanket six-week ROCm migration is not justified.
Re-estimate after the W1.1 design spike and X-U6 ownership inventory.

---

## 6. Scope

Examined: the three Target IR ODS files, `target_ir.py`, the
`GenerateROCM*Kernel` pass inventory, `test_target_ir_contract.py`, and the x86
backend tree.

Not examined: the bodies of the 67 ROCm generators, `emit/nvidia_cuda.py`
internals, the Apple `apple_gpu_runtime.mm` MSL set, the collectives and
neighbors dialects, and the RubinCPX backend. `emit/rocm_hip.py` was inspected
far enough to establish its candidate/runner role. The ROCm generator bodies
need host-side ownership classification before X-U6; gfx1151 is required for a
later migration that changes a selected producer or emitted kernel.
