---
last_updated: 2026-07-28
audit_role: reference
---

# AMD Kernel-Compiler Survey — StinkyTofu, rocRoller, Composable Kernel & hipBLASLt

> **Purpose.** Four production AMD kernel *compilers, tile frameworks and
> selection engines* read at source level for transferable architecture,
> algorithms, and codegen technique.
> This is design input for Tessera's compiler, not a target-facts reference and
> not a status surface.
>
> For gfx1250/MI450 **target facts** (ISA, hazards, counters, clusters, SDMA) see
> [`../backend/rocm/GFX1250_MI450_COMPILER_REFERENCE.md`](../backend/rocm/GFX1250_MI450_COMPILER_REFERENCE.md).
> For the AMD **ecosystem survey** (AITER, hipBLASLt, rocWMMA, Mori, Iris, XIO,
> Gluon) see
> [`../backend/rocm/ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md`](../backend/rocm/ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md).
> For our own direction see
> [`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md).
>
> **Provenance:** **[V]** verified from source; **[S]** stated by an in-repo AMD
> design document; **[I]** inference or recommendation.
>
> Surveyed 2026-07-28 against `ROCm/rocm-libraries@develop`.

---

## 0. Why these four

Most of the AMD ecosystem is *kernel libraries*. These four are compilers,
compile-time frameworks, or selection engines, and together they cover the whole
pipeline from problem to chosen kernel:

| | StinkyTofu | rocRoller | Composable Kernel (`ck_tile`) | hipBLASLt / TensileLite |
|---|---|---|---|---|
| Input | existing AMD **assembly** | a high-level `Command` | C++ template kernel source | a runtime GEMM call |
| Output | optimized assembly | assembly → Comgr → HIP execution | compile-time-specialized kernels | a *selected* kernel + launch |
| Core artifact | Logical IR → Asm IR | `KernelGraph` (dual graph), lowered in place | coordinate algebra in templates | `SolutionLibrary` selection tree |
| Role | post-pass optimizer for hipBLASLt/TensileLite | standalone kernel generator | tile programming model + kernel library | the shipping GEMM library + arbiter |
| Arch scope | **gfx1250 and beyond** (older use `rocisa`) | gfx9 → gfx12 | gfx9 → gfx12 | all supported |

StinkyTofu is what AMD built *specifically because gfx1250 needed a new asm-level
optimizer*. rocRoller is the closest existing thing to what Tessera is trying to
be. **Composable Kernel is the most mature tile-programming model in the
ecosystem, and its coordinate framework (§3) is the single most transferable idea
in this survey.** **hipBLASLt/TensileLite (§4)** is the production arbiter — how a
shipping library actually chooses among generated, hand-written and predicted
kernels. Reading all four shows which of our design choices are convergent and
which are idiosyncratic.

---

## 1. StinkyTofu — an LLVM-inspired asm-level pass optimizer

> "StinkyTofu is an LLVM-inspired pass-based IR optimizer for AMD GPU assembly
> kernels, used by hipBLASLt/TensileLite via Python bindings." **[S]**

### 1.1 Two IR levels — independent convergence with Decision #19

**Logical IR** (architecture-agnostic, high-level) → **Asm IR** (concrete,
architecture-specific). Passes operate on Asm IR; core types are
`StinkyInstruction`, `Function`, `BasicBlock`. **[S]**

That is exactly Decision #19 ("backends expose a hardware-free Target IR before
hardware-specific lowering"), arrived at independently by a team whose only goal
was making hipBLASLt faster. Worth citing the next time the two-level cost is
questioned.

### 1.2 Cost is *in* the IR

The textual form carries scheduling cost as instruction attributes **[S]**:

```
st.func @name() {
^entry:
  v0 = "st.v_mul_f32"(v1, v2) { issueCycles = 1, latencyCycles = 5 }
}
```

`issueCycles` / `latencyCycles` are materialized on the instruction rather than
looked up in a side table at scheduling time. Consequences worth stealing **[I]**:
the IR is self-describing for cost, a dumped kernel is a complete scheduling
problem, and cost-model changes are visible as IR diffs rather than invisible
behavioural drift. Our `#tile.layout`-style attributes already prove we can carry
this kind of data; cost is a natural next tenant.

### 1.3 The hardware description is data, not code

```
hardware/src/gfx/GfxXXX/
  GfxXXXInstructions.def   (DEF_T / DEF_BATCH)
  GfxXXXFormats.def
  arch.cmake               (ARCH_MAJOR, ARCH_WAVEFRONT, costs, register limits)
        │ tablegen
        ▼
hardware/generated/  GfxXXX_{init,costs,operands,block}.inc
        ▼
  gfxisa lib → stinkytofu lib → tools / Python bindings
```

> "New architectures require only adding a `hardware/src/gfx/GfxXXX/` directory
> with `.def` files — **no C++ edits** for instruction definitions." **[S]**

This is Decision #28's `TargetPlugin` seam realized as **data plus codegen**
rather than as a C++ interface. The test of the seam is falsifiable and cheap:
*can a new arch land without touching the optimizer's source?* Worth adopting as
an explicit acceptance criterion for our own backend-plugin work. **[I]**

### 1.4 Region-scoped scheduling via `ScopeAdaptor`

> "`ScopeAdaptor` extracts instruction regions (identified by named groups like
> `loopWithPrefetch`) into temporary `Function` objects for isolated scheduling,
> then splices results back." **[S]**

A cheap way to get region-local scheduling without a real outlining pass or a
region dialect: lift, schedule in isolation, splice. The named-group tagging
(`loopWithPrefetch`) is applied by the *kernel generator* upstream — the
optimizer never has to rediscover loop structure heuristically. That
producer-tags/consumer-trusts split recurs (see §1.6) and is the single most
repeated idea in this codebase. **[I]**

### 1.5 Declarative peephole patterns

Optimizations are compiled from `.pattern` files with a three-block grammar
**[V]**:

```
pattern PatternName {
  match       { $mul = v_mul_f32 $tmp, $a, $b
                $add = v_add_f32 $dst, $tmp, $c }   // temporal order, dest first
  constraints { ... }
  rewrite     { ... }
}
```

Variables are untyped (`$x`) and the system infers instruction-vs-register-vs-
constant from context. Instructions in `match` are listed in execution order, and
data flow is expressed by reusing a variable (`$tmp`).

The transferable part is not the syntax but the **separation of the pattern
corpus from the pass**: peepholes become a reviewable, testable data file rather
than C++ that only its author can audit. **[I]**

### 1.6 `StinkyWmmaVgprReorderPass` — the best-structured pass in either project

A **read-only analysis** that finds VGPR savings in N-buffered GEMM loops. It
never mutates anything; downstream passes act on its result. **[S]**

**The optimization.** In an N-buffered GEMM the WMMA instructions split into N
pools, each with its own registers for the pool-varying operand. When that
operand is the *outer* loop dimension its registers stay live across every inner
iteration, blocking cross-pool aliasing. Making it the *inner* dimension —
grouping its instructions contiguously within each pool — tightens the liveness
interval so pool N's registers die before pool 0 next needs them, and the two
pools can alias the same physical registers.

**The structure — three swappable layers**, each replaceable without touching the
others:

| Layer | Interface | Built-in |
|---|---|---|
| 1 | `IRegLivenessAnalysis::computeLiveness(bb, wmmaSeq)` | `WmmaIntervalLiveness` — interval = `[first WMMA reading the group, last]`. A full-instruction liveness backend drops in unchanged. |
| 2 | `IWmmaReorderAlgorithm::solve(pools, liveness)` → `{desiredOrder, aliases}` | `PoolVaryingReorderAlgorithm` — fires when `interval width > number of distinct B groups` |
| 3 | `WmmaReorderAnalysisResult` | `{applicable, desiredWmmaOrder, replacements, totalVgprSaved}` — the stable contract |

Both layers are injected at construction; `nullptr` selects the default.

Two details worth copying wholesale **[I]**:

- **Pool tagging over heuristics.** TensileLite stamps each WMMA with a
  `WmmaPoolData{poolIndex}` modifier at generation time. *"If any wmma
  instruction is missing this modifier the pass bails out for the entire basic
  block. Partial tagging indicates a misconfigured pipeline and the pass must not
  proceed on incomplete information."* Fail loudly on partial metadata rather
  than silently degrading — that is Decision #21's spirit applied to an analysis.
- **Symmetry via detection, not special-casing.** `detectABIndices` decides which
  operand is pool-varying by checking register-group intersection across pools,
  so the pass handles hardware-A-varying and hardware-B-varying kernels with one
  code path.

### 1.7 Optimization remarks as a first-class output

`LoopRegionRemarkPass` emits remarks on **loop health**: region count, boundary
causes, **`s_nop` waste**, branch count. **[S]**

The compiler reporting *why* a schedule is bad — and quantifying wasted issue
slots — is a diagnostics idea we do not currently have. It pairs naturally with
the Evaluator: a remark stream is a cheap, structured signal that does not need
hardware to be useful. **[I]**

### 1.8 Other notes

- **Pseudo-PHI nodes.** `buildUseDefChain()` inserts PHIs at CFG joins for
  cross-block def-use; they are never emitted (`AsmEmitter` skips them), and any
  code that counts instructions must skip `GFX::PHI`. Same device as the
  `MemTokenData` pseudo-registers used for wait insertion. **[S]**
- **Intrinsic system.** High-level ops (ReLU, Clamp) live in
  `src/ir/logical/Intrinsics.intrinsic`, compile to a binary `intrinsics.st.bc`
  at build time, and load at runtime through `IntrinsicRegistry` — a two-stage
  build that avoids a circular dependency with TableGen. **[S]**
- **`stinkytofu-opt` + `stinkytofu-check`.** An opt-style driver with a pass
  registry plus FileCheck-style tests. Structurally identical to
  `tessera-opt` + lit. **[S]**

---

## 2. rocRoller — a dual-graph kernel generator

> "RocRoller transforms high-level kernel specifications (`Command`) through a
> dual-graph IR (`KernelGraph` with `ControlGraph` + `CoordinateGraph`) into
> optimized GPU assembly, then assembles to binary via AMD Comgr and executes via
> HIP." **[S]**

### 2.1 The KernelGraph — the headline idea

A single `KernelGraph` encodes **three** things at once, and is *iteratively
rewritten in place* until it is low-level enough to emit from **[S]**:

1. **Coordinate transforms** — how vector/matrix/tensor indices are computed from
   each other.
2. **Control flow and operations** — operations and their dependencies.
3. **Data flow and distribution** — how data moves through the GPU and how it is
   distributed.

**The coordinate graph is a hypergraph.** Nodes are `Dimension` variants (a size
and stride; or a `for`-loop index). Edges are *index transforms* — e.g. `Flatten`
takes several source indices and produces a row-major contiguous index — and an
edge connects a **tail set** of sources to a **head set** of destinations. **[S]**

**The control graph** has `Operation` nodes (`LoadVGPR`, `Multiply`, `ForLoop`,
`If`), where control constructs **contain nested control graphs as their bodies**.

Why this matters to us **[I]**: this is a first-class *layout/index algebra* held
as a graph, separate from control flow, in the same IR. It is the concrete
realization of the "layout algebra with named hardware axes" idea we noted from
the MLSys GPU book. Our stack expresses layout through attributes
(`#tile.layout`) hung on ops in a dialect tower; rocRoller makes the index
transform itself the graph and lowers by rewriting it. That is a genuinely
different point in the design space, and the hypergraph edge (many→many) is what
makes fusion/splitting of dimensions natural rather than encoded.

It is worth being clear about the trade: one mutable graph lowered in place gives
up MLIR's verifiable stage boundaries and round-trippable textual dialects, which
are load-bearing for our lit-test discipline. **The idea to take is the
coordinate hypergraph as a representation, not the single-IR architecture.** **[I]**

### 2.2 `Expression` and `EvaluationTime` — staging as a type

Expressions are `std::variant` trees visited with `std::visit`, with
transformations in `ExpressionTransformations.hpp`. The key annotation **[S]**:

> "they can only contain certain types of values at certain points. The
> `EvaluationTime` enum can be used to describe when a certain expression can be
> used."

An explicit *when-is-this-knowable* tag on every expression. That is precisely
what Decision #28's symbolic-dim policy (`static | bucket | dynamic`) needs to be
enforceable rather than conventional — the same concept, and evidence it belongs
on the expression rather than on the op. **[I]**

### 2.3 Observer-based scheduling — `peek` / `modify` / `observe`

The single best structural idea in either codebase. `Context::schedule()` runs
every instruction through a `MetaObserver` composing many `IObserver`s **[S]**:

| Method | Role |
|---|---|
| `peek` | *what would happen* if this were scheduled — stall cycles, errors — **without committing** |
| `modify` | mutate the instruction (e.g. attach a `WaitCount`) |
| `observe` | update machine state after scheduling (queue occupancy, hazard windows) |

`WaitcntObserver` is just one observer, reading `GPUArchitecture` to decide waits.
Hazard handling, wait insertion, and cost accounting become **composable observers
over one instruction stream** instead of separate passes that must agree with each
other.

Why this is the right shape for us **[I]**: `peek` is a **cost query before
commitment**, which is exactly the primitive a measured arbiter or a scheduling
search needs and which a pass pipeline cannot express. Our three AMD completion
mechanisms (waitcnt positions, TDM descriptors, SDMA fused packets — see the
gfx1250 reference §9.1) are three observers over one stream, not three passes.
This is the most directly actionable idea in this document.

### 2.4 `Component` — a runtime plugin registry

A `ComponentBase` declares an `Argument` type and a `Basename`. Concrete
components supply `Match(args)` (a predicate), `Build(args)` (a factory), and a
`Name`. `Component::Get<Base>(args)` searches registrations, caches instances for
non-single-use components, and is thread-safe via reader-writer locks. Used
throughout for architecture-specific codegen. **[S]**

The distinguishing feature versus a plain virtual interface is that **selection is
a predicate over arguments**, not a type switch — so a new backend registers a
`Match` and never edits a dispatch site. Compare Decision #28's
`KernelEmitter`/`TargetPlugin` seam. **[I]**

### 2.5 `Generator<T>` — instruction streams as C++20 coroutines

Lazy sequences via `co_yield`, modelling `std::ranges::input_range`, with
`map`/`filter`/`take`/`only`/`empty` and `.to<Container>()`. Movable, not
copyable; nothing executes until the first value is pulled. **[S]**

Instruction generation is a *stream* rather than a materialized vector, which is
what lets `peek` interleave with generation. Mostly a C++ implementation choice,
but it is the mechanism that makes §2.3 ergonomic. **[I]**

### 2.6 The IR serializes to YAML

`Command`, `KernelGraph`, `Expression`, and `Operation` all implement
serialization traits (`toYAML` / `fromYAML` / `writeYAML`), *"particularly useful
for debugging, caching compiled kernels, and inspecting internal structures."*
**[S]**

A serializable IR is a **cache key, a bug report, and a regression fixture in one
artifact**. Our AOT/compilation-cache and Evaluator work both want this; worth
checking what our Graph/Schedule/Tile IR can currently round-trip. **[I]**

---

## 3. Composable Kernel — the P/Y/X/D coordinate framework

CK ships roughly **400 KB of conceptual documentation** across ~28 files under
`docs/conceptual/ck_tile/`. It is the most thoroughly explained tile-programming
model in the AMD ecosystem, and the framework below is why.

### 3.1 Four coordinate spaces, two transformations

CK's thesis is that thread→data assignment should be a **composition of
well-defined mappings between named coordinate spaces**, resolved at compile
time. **[S]**

| Space | Meaning | Coordinates |
|---|---|---|
| **P** | Thread Position — the hardware execution hierarchy | `thread_x`, `thread_y`, `warp_id`, `block_id` |
| **Y** | Local Data — the *algorithm's* view of its own work | `y0, y1, y2, y3` (algorithm-specific) |
| **X** | Global Position — coordinates in the problem domain | matrix row/col, image spatial coords |
| **D** | Memory Address — linearized, after padding/interleaving | a single offset |

```
P ─┐
   ├─► (P + Y → X) ─► X ─► (X → D) ─► D
Y ─┘   distribution      layout
       strategy          optimization
```

The separation of responsibility is the payload **[S]**:

- **`P + Y → X`** encodes the **distribution strategy** — how work is partitioned
  across threads. Structuring this transform correctly is what *guarantees memory
  coalescing*.
- **`X → D`** encodes **layout optimization** — padding, interleaving, address
  space. Designing this is what *minimizes bank conflicts*.
- Architecture portability comes from **changing transform parameters while the
  algorithm text stays fixed**.

Why this matters to us **[I]**: Y-space is the one that does not appear in our
IR. We have global/tile coordinates and we have layouts, but the *algorithm's
local view of its own work*, as a coordinate space distinct from both thread
position and global position, is not named. It is what lets CK express an
algorithm "in its natural form" while the distribution stays a separate,
swappable object. Naming it would give the tile dialect a place to put the
per-thread iteration space that today is implicit in lowering.

This framework is strictly more refined than rocRoller's coordinate graph (§2.1):
same underlying idea, but with the spaces *named* and each transformation
assigned a distinct optimization duty.

### 3.2 The transform algebra — a closed operator set

Transforms map between a **lower** (source) and **upper** (target) dimension
space, and are **bidirectional** **[V]**:

- `calculate_lower_index()` — upper → lower (where to actually find the element)
- `calculate_upper_index()` — lower → upper (recover the original coordinate)
- `update_lower_index()` — **incremental** movement by a delta, without
  recomputing from scratch

The operator set is small and closed **[V]**:

| Transform | Role |
|---|---|
| `MergeTransform` | multi-D → linear |
| `UnmergeTransform` | linear → multi-D |
| `EmbedTransform` | linear → multi-D strided |
| `ReplicateTransform` | 0-D → multi-D broadcast |
| `OffsetTransform` | translation |
| `PassThroughTransform` | identity |
| `PadTransform` | boundaries |
| `XorTransform` | **swizzle** |
| `SliceTransform`, `ModuloTransform` | sub-ranges, wrapping |

Three observations **[I]**:

1. **`XorTransform` is a first-class member of the algebra.** The bank-conflict
   swizzle is not a special case bolted onto the LDS path — it is an index
   transform like any other, so it composes and inverts. Compare §4.1, where the
   same optimization is a hand-written parameter block.
2. **`ReplicateTransform` makes broadcast a coordinate operation**, which is how
   a value shared across threads stays inside one framework rather than becoming
   a separate mechanism.
3. **Everything is zero-copy and logical.** *"The actual tensor data remains
   stored in memory in linear fashion, exactly as specified by the original tensor
   shape and strides at creation time."* Transforms create views; they never move
   data. That invariant is what makes composition safe.

`update_lower_index()` deserves separate mention: incremental coordinate movement
is what makes sliding a tile window cheap, and CK gives it a dedicated 18 KB
document (`coordinate_movement.rst`). A tile IR that can only recompute absolute
coordinates will pay for it in every loop.

### 3.3 `tile_distribution_encoding` — the distribution as compile-time data

The whole thread↔data mapping is one declarative template parameter pack **[V]**:

```cpp
using Encoding = tile_distribution_encoding<
    sequence<>,                             // R  — replication dims (none here)
    tuple<sequence<4, 2, 8, 4>,             // H  — hierarchical lengths, X dim 0 (M)
          sequence<4, 2, 8, 4>>,            //      hierarchical lengths, X dim 1 (N)
    tuple<sequence<1, 2>, sequence<1, 2>>,  // P → RH  major
    tuple<sequence<1, 1>, sequence<2, 2>>,  // P → RH  minor
    sequence<1, 1, 2, 2>,                   // Y → RH  major
    sequence<0, 3, 0, 3>>;                  // Y → RH  minor
```

`sequence<4, 2, 8, 4>` is a four-level decomposition reading directly onto
hardware: *four repetitions per thread, two warps per block, eight threads per
warp, four elements per vector op.* The `P → RH` and `Y → RH` maps are a **wiring
diagram** stating which hierarchy level of which X dimension each thread-position
coordinate and each local-data coordinate indexes. **[S]**

The transferable property **[I]**: the distribution is a **value**, separable from
the kernel body, comparable, and enumerable. That is exactly the shape an
autotuner or a measured arbiter wants — a distribution becomes a search
coordinate rather than a code variant.

**How the wiring actually works.** P and Y both index a single unified
**RH-space** — the concatenation of the R (replication) dims and the H
(hierarchy) groups — addressed by a `(major, minor)` pair **[V]**:

- **major** — which RH group: `0` = R, `1..N` = the H group for X dimension *n*
- **minor** — which component within that group

So `Ps2RHssMajor`/`Minor` and `Ys2RHsMajor`/`Minor` are literally a permutation
table: *"P coordinate i indexes component `minor[i]` of group `major[i]`."* The
whole distribution strategy is a wiring diagram over a hierarchical index space.

The H hierarchy has a canonical four-level reading for GEMM **[S]**:

```cpp
using HsLengthss = Tuple<
    Sequence<MRepeat, MWarp, MThread, MVec>,   // M
    Sequence<NRepeat, NWarp, NThread, NVec>>;  // N
//   ^iterations   ^warps  ^threads   ^vector width
//    per thread   in M    per warp   per access
```

R-dimensions exist for three stated purposes **[S]**: data reuse (the same input
feeding multiple output computations), reduction (several threads collaborating
on one result), and bandwidth reduction.

**The encoding → transform chain is mechanical.** `make_ps_ys_to_xs_adaptor()`
builds a fixed three-stage chain straight from the encoding, with no per-kernel
hand-authoring **[V]**:

```
combine(P, Y) → ReplicateTransform (if R dims) → UnmergeTransform (into H dims)
              → MergeTransform (into X dims) → X
```

That is the implementability result: **given the encoding, the coordinate
machinery is generated, not written.** A distribution search therefore explores
encodings, and the transform chain follows for free. **[I]**

Per-thread storage is handled separately by a `ys_to_d_descriptor` — a plain
lengths/strides pair giving `offset = Σ y[i] * stride[i]`, chosen so that vector
loads land contiguously (e.g. layout `[M/VectorSize][N][VectorSize]`). Y→D is
where register-level layout is decided, distinct from the X→D global layout. **[V]**

### 3.4 Pipeline = Problem + Policy

A CK kernel is composed from **a pipeline, a tile partitioner, and an epilogue**,
and the pipeline itself splits in two **[S]**:

| Component | Question it answers |
|---|---|
| **Problem** | *What* to compute — shapes, dtypes, the math (GEMM, conv) |
| **Policy** | *How* to move data — access patterns, hardware-specific choices |
| **Tile Partitioner** | problem dims (M, N, K) → workgroup tiles (kM, kN, kK) → grid |
| **Epilogue** | activation, bias, post-processing |

Holding *what* and *how* apart at the type level is the same instinct as our
Graph IR / Schedule IR split, arrived at inside a template library. The useful
detail is that **Policy is a substitutable object**, so the same Problem can be
retargeted or retuned without touching the algorithm.

Supporting vocabulary worth adopting for its precision **[S]**: **Tile Window**
(a viewport into a larger tensor defining the current tile's position and
bounds), **Block Tile** / **Wave Tile** (workgroup- and wave-granularity
sub-tiles), **Load Tile** / **Store Tile** (global↔LDS↔register transfers as
named operations).

### 3.5 Intrawave vs interwave scheduling

A clean statement of a real scheduling dichotomy for K-loop accumulation **[S]**:

| | Mechanism | Best for |
|---|---|---|
| **Interwave** | K split into chunks; the *same* chunk loaded into every wave; all waves sync between chunks | **memory-bound** — coordinated accesses, optimized cache hit rate |
| **Intrawave** | full K loaded per wave; each wave runs independently, no sync; the CU interleaves | **compute-bound** — CU has scheduling freedom |

Both ship, selected per workload. This is a *policy* choice in the §3.4 sense,
and it is the kind of discrete, nameable schedule axis our Schedule IR should be
able to carry rather than rediscover. **[I]**

### 3.6 Coordinate movement — why tile iteration is cheap

A coordinate in CK is not a position. It is the **materialized state of an entire
transform chain** **[V]**:

```cpp
class TensorAdaptorCoordinate {
    MultiIndex top_index_;      // input position
    MultiIndex bottom_index_;   // output after all transforms
    MultiIndex hidden_index_;   // cached intermediate results
};
```

`hidden_index_` — the cached intermediates of each stage — is what makes partial
recomputation possible. Movement then has a fast path **[V]**:

```cpp
coord.top_index_ += step;
if (transformation_affects_movement(desc, step)) {
    coord.hidden_index_ = desc.calculate_bottom_index(coord.top_index_);
    coord.offset_      = desc.calculate_offset(coord.top_index_);
} else {
    coord.offset_ += calculate_step_offset(desc, step);   // single add
}
```

**If a step does not cross a transform boundary — no carry into a merged
dimension — the address update is one addition.** That is the entire reason
sliding a tile window is cheap, and it is a property of holding the chain state
rather than the coordinate alone.

The consequence for us **[I]**: an IR whose only operation is "compute the
address at coordinate C" cannot express this. It needs a *move-by-delta*
operation over a coordinate that carries chain state, plus the ability to decide
statically whether a given delta is boundary-crossing. This is the concrete
mechanism behind the `update_lower_index()` note in §3.2, and it is the part most
likely to be missed when porting the coordinate algebra alone.

### 3.7 Space-filling curves and a compile-time locality metric

`SpaceFillingCurve` maps a 1-D access index to multi-dimensional coordinates, so
"traverse this tile well" becomes a linear loop **[S]**. Parameters: dimension
`Order` (row- vs column-major traversal), `scalars_per_access` (vector width per
dimension), and a **snake** flag that reverses direction on alternate rows/planes
to keep consecutive accesses spatially close.

Stated best practices **[S]**: match traversal order to storage order; size the
vector as `min(fast_dim_length, cache_line_size / sizeof(T))`; enable snake for
large tensors; and `static_assert` that vector access aligns to cache lines.

**The part worth taking is the analyzer.** CK ships a compile-time traversal-
quality metric that walks the curve, takes the Manhattan distance between
consecutive accesses, and bins them **[V]**:

```cpp
const auto step = sfc.get_step_between(i, i + 1);
index_t distance = Σ |step[d]|;
if      (distance <= 1)  sequential_steps++;
else if (distance <= 16) cache_line_jumps++;   // within a cache line
else                     large_jumps++;
```

This is a **hardware-free, static locality score for an access pattern**. That is
precisely the gap [`TILESIGHT_ASSESSMENT.md`](TILESIGHT_ASSESSMENT.md) recorded —
that the analytical cost model our arbiter falls back on was a mock. A
step-distance histogram over a materialized access order is cheap, needs no
device, and is a real signal. **[I]**

### 3.8 Two worked swizzles — and what they prove about the algebra

CK documents two independent bank-conflict swizzles. Both are built **entirely
from the §3.2 operator set** — no new primitives — which is the strongest
available evidence that the algebra is actually closed. **[V]**

#### XOR preshuffle (`lds_index_swapping.rst`)

Operates on a 3-D LDS coordinate `[K0, M, K1]`, where `K1 = KPack` (the thread's
vector width along K) and `K0 = KPerBlock/KPack`. Three stages:

```
1. XOR      K0' = K0 XOR (M % (KPerBlock/KPack * MLdsLayer))
2. Unmerge  L    = K0' / (KPerBlock/KPack)        // MLdsLayer == 1 ⟹ L = 0
            K0'' = K0' % (KPerBlock/KPack)
3. Merge    (L, M) → M'        (K0'', K1) → K'
```

Stage 1 mixes M-dimension bits into the K0 index, redistributing accesses across
banks. `MLdsLayer` is the knob that lets **several rows share one bank set**, which
is what keeps small tiles from wasting bank capacity — stage 2 exists only to
carve that layer index back out.

In code it is ordinary descriptor composition — `make_xor_transform` alongside a
`make_pass_through_transform`, with explicit lower/upper dimension index lists:

```cpp
transform_tensor_descriptor(
    BaseDescriptor{},
    make_tuple(make_xor_transform(Sequence<M_over_layer, K0_times_layer>{}),
               make_pass_through_transform(Number<KPack>{})),
    Sequence<1, 0>{},   // XOR consumes dims [1,0]
    Sequence<2>{});     // pass through dim 2
```

Stated configuration heuristics **[S]**: `MLdsLayer` = 1 / 2 / 4 for tile sizes
≤32 / ≤64 / larger; `KPack` = 8 for fp16/bf16, 4 for fp32, 2 otherwise; with
`static_assert(TileSize % (MLdsLayer * KPack) == 0)`.

#### Morton / Z-order (`swizzling_example.rst`)

The result worth internalizing:

> "MergeTransform enables Morton ordering by reordering and merging coordinate
> bits."

```cpp
using SplitTransform       = UnmergeTransform<Sequence<2, 2>>;        // coord → bits
using MortonMergeTransform = MergeTransform<Sequence<2, 2, 2, 2>>;    // bits → index
// merge computes: morton_idx = y₁·8 + x₁·4 + y₀·2 + x₀
```

**Morton ordering is not a special function — it is a `Merge` over bit-split
dimensions taken in a permuted order.** Unmerge each coordinate down to
individual bits, then Merge them back in interleaved order, and bit-interleaving
falls out of the ordinary linearization arithmetic. The 4×4 tile layout it
produces:

```
 0  1  4  5
 2  3  6  7
 8  9 12 13
10 11 14 15
```

Why this matters for us **[I]**: it upgrades take-list item 2 from "add an XOR
transform" to something stronger — **a `Merge`/`Unmerge` pair that can address
individual bits covers an entire family of swizzles** (Z-order, and by
construction other bit-interleavings) with no new operators at all. If our layout
algebra gets bit-granular split/join, the swizzle family comes free; if it only
handles whole dimensions, every swizzle stays a special case.

#### A second static metric — and this one is assertable

Both documents ship the same analyzer shape **[V]**:

```cpp
for (tid = 0; tid < WarpSize; ++tid) {
    offset = desc.calculate_offset(coords_for(tid));
    bank   = (offset * sizeof(T) / BankWidth) % NumBanks;
    bank_access[bank]++;
}
max_conflict = max over banks;          // "N-way bank conflict"
```

The Morton document runs it **comparatively** — row-major versus Morton — turning
"is this layout better" into a number computed from the descriptor alone.

This is a companion to §3.7's locality histogram, and it is the more immediately
useful of the two for us because it is **a property a unit test can assert**:
*this descriptor is conflict-free for a warp-wide access on N banks.* No device,
no fixture, no measurement — just the descriptor and the bank count. That is
exactly the kind of gate our LDS-layout work currently lacks. **[I]**

### 3.9 The user-facing surface — `TileWindow` and `sweep_tile`

These two APIs are where the framework pays off, and reading them settles what
the encoding is actually *for*.

**`TileWindow` separates "which data is mine" from "how to fetch it."** **[S]**

> "While TileDistribution solves the problem of work assignment by mapping
> threads to tensor coordinates, it does not address *how* threads access the
> data at those coordinates. TileWindow serves as the critical bridge."

```cpp
template <typename TensorView_, typename WindowLengths_, typename TileDistribution_>
struct tile_window_with_static_distribution {
    TensorView   tensor_view_;
    Distribution distribution_;
    array<index_t, N> origin_;                       // runtime
    static constexpr auto window_lengths = ...;      // compile-time
};
```

Note the static/dynamic split: **window lengths are compile-time, the origin is
runtime.** That is exactly the `static | bucket | dynamic` distinction Decision
#28 requires, landed on the natural seam — shape is static, position is not. **[I]**

**`LoadStoreTraits` derives the access pattern from the distribution.** It is a
compile-time engine performing three analyses **[S]**:

1. **Vector dimension identification** — which Y dimension has stride 1
2. **Access pattern calculation** — how many memory operations, in what order
3. **Space-filling curve construction** — the traversal order itself

So the SFC of §3.7 is **not hand-picked — it is computed from the encoding.**
`scalar_per_vector` likewise. This is the payoff of distribution-as-a-value: once
the encoding exists, vector width, access count, and traversal order all follow.

The load loop makes the entire chain concrete **[V]**:

```cpp
static_for<0, Traits::num_access, 1>{}([&](auto i_access) {
    const auto y_indices      = Traits::get_y_indices(i_access);          // ← SFC
    const auto x_indices      = distribution_.calculate_x_from_y(y_indices); // ← P+Y→X
    const auto global_indices = add_arrays(origin_, x_indices);
    if constexpr (Traits::scalar_per_vector > 1) { /* vector load */ }
    else                                        { /* scalar load */ }
});
```

`SFC → Y → (P+Y→X) → +origin → global`, with vectorization as a `constexpr`
branch rather than a runtime one. Window movement is `set_window_origin(...)`,
O(1) on the precomputed coordinates from §3.6.

**`sweep_tile` is the iteration surface** — "load once, use many times": load the
X data into registers once, then sweep Y positions while X stays resident.
Implemented as compile-time recursive `static_for` over Y-space lengths, so it
unrolls with zero runtime overhead. Named use cases: matmul (reuse A columns),
convolution (reuse filter weights), reduction (accumulate over Y), broadcast
(apply X across all Y). **[S]**

CK states the layering as a four-line contract **[S]**:

> 1. **TileDistribution**: "Here's how to divide work"
> 2. **TileWindow**: "Here's the data, loaded efficiently"
> 3. **Sweep operations**: "Here's how to process every element"
> 4. **User code**: "Thanks! *does computation*"

**Why this is the important section for us** **[I]**: it is a clean separation of
four concerns our stack partly conflates —

| Concern | CK owns it in | Derived or written? |
|---|---|---|
| which data is mine | `TileDistribution` encoding | **written** (the tunable) |
| how to fetch it | `TileWindow` + `LoadStoreTraits` | **derived** |
| what order to traverse | space-filling curve | **derived** |
| what to compute | the `sweep_tile` lambda | **written** (the algorithm) |

Only two of the four are authored. Vectorization, coalescing, access count, and
traversal order are *consequences* of the encoding, not independent knobs a
kernel author sets. That is the strongest argument in this survey for
distribution-as-a-value (take-list item 11): it is not merely an autotuning
convenience, it is what makes everything else derivable.

---

## 4. hipBLASLt / TensileLite — the arbiter, in production

This is the closest existing thing to Decision #28's measured arbiter, and it is
richer than the "Tensile generates kernels" summary suggests.

### 4.1 Five pathways, not one

The runtime flow is `hipblaslt.cpp` → `rocblaslt_mat.cpp` → `tensile_host.cpp` →
TensileLite host → a lazily-loaded `.hsaco`/`.co` from a *device library*
directory. But several kernel sources coexist **[S]**:

| Pathway | Where | Status |
|---|---|---|
| **TensileLite** — generated assembly kernels | `tensilelite/` | primary; "what new work uses" |
| **rocRoller** custom kernels | `library/src/amd_detail/rocblaslt/src/rocroller/` | ships alongside, gated by `HIPBLASLT_ENABLE_ROCROLLER`, **ON by default** |
| **ExtOps** (softmax, layernorm, AMax) | `device-library/extops/`, generated by `SoftmaxGenerator.py` etc. | separate device library |
| **Matrix transform** | `device-library/matrix-transform/`, `rocblaslt_transform.cpp` | separate op family |
| **User-driven tuning** | `UserDrivenTuningParser.cpp` | runtime override of selection |

So a production GEMM library runs **a generated-kernel tier, an alternate-generator
tier, and a user-override tier simultaneously**, with a build flag to drop one. That
is Decision #28's three-tier model observed in the wild, plus a fourth tier we do
not currently model: explicit user override at runtime. **[I]**

### 4.2 Selection is a composable tree of single-concern nodes

The design statement, from `SolutionLibrary.hpp` **[V]**:

> A complete SolutionLibrary is a **tree of objects which each handles a single
> aspect of selecting a solution** for a given problem. Each node in the tree will
> handle an aspect such as:
> - Compatibility with a particular model of GPU
> - Selecting kernels that solve a particular type of problem (transpose, data
>   type, etc.)
> - Selecting the fastest kernel based on benchmark results or other logic
> - Ensuring that a problem is compatible with any assumptions made by a
>   particular kernel (e.g. size or stride requirements)

with the documented example composition:

```
MasterSolutionLibrary            (serialization)
 └─ GPU selection
     └─ Problem type selection
         └─ Predicated logic for specific sizes
             └─ Matching library based on benchmarks
                 └─ Individual kernels
```

The node types available are effectively a **taxonomy of selection strategies**,
each a `SolutionLibrary` subclass **[V]**:

| Node | Strategy |
|---|---|
| `MasterSolutionLibrary` | root; owns serialization |
| `MapLibrary` | dispatch by key |
| `ExactLogicLibrary` | exact match on a predicate set |
| `ProblemMatchingLibrary` | *"Uses a distance function to select solutions based on benchmarks… At runtime, we find the benchmarked size that is closest to the size asked for."* |
| `GranularitySelectionLibrary` | *"Compares the tile sizes of each kernel, the dimensions of the problem, and the number of compute units… to select a kernel that fits the best on the GPU with the lowest amount of waste"* |
| `FreeSizeLibrary` | free-size problems |
| `PredictionLibrary` / `MLPClassificationLibrary` | learned models (§4.5) |
| `PlaceholderLibrary` | lazy code-object loading |
| `CachingLibrary` | memoized selection |
| `EmbeddedLibrary` | compiled-in library data |

This is the structural answer to "how do you combine hand-tuned, generated, and
predicted kernels without a tangle of special cases": **you don't build one
arbiter, you build a tree whose nodes each decide one thing.** Analytical
selection, benchmark lookup, and a learned model are sibling node types, not
competing designs. **[I]**

### 4.3 The interface has the shape an arbiter needs

```cpp
virtual std::shared_ptr<MySolution>
    findBestSolution(MyProblem const&, Hardware const&, double* fitness = nullptr) const = 0;

virtual SolutionSet<MySolution>
    findAllSolutions(MyProblem const&, Hardware const&, SolutionLibrarySearchType) const = 0;

virtual SolutionVector<MySolution>
    findTopSolutions(MyProblem const&, Hardware const&, int numSolutions) const;
```

Three details worth copying **[V]**:

- **`fitness` is an out-parameter of "find best."** Selection returns not only a
  choice but a *score for that choice* — which is what lets a caller decide
  whether to trust it, and what an accuracy-budgeted arbiter needs.
- **`findTopSolutions(N)` is first-class**, alongside `findAllSolutions` and
  `getSolutionByIndex`. An arbiter that can only ask "what's best" cannot
  benchmark candidates; top-N is the primitive that makes measurement possible.
- **Search strictness is an enum**, not a boolean: `DEFAULT` (full predicates),
  `GEMM_TYPE_ONLY` (dtype/transpose/grouped/mx-block match only), `HARDWARE_ONLY`
  (accept everything). Progressive relaxation is built into the query.

Grouped-GEMM gets parallel overloads throughout (`findAllSolutionsGroupedGemm`,
`findTopSolutionsGroupedGemm`), i.e. "a batch of problems selecting one solution"
is a modelled case, not an afterthought.

### 4.4 A pluggable distance-metric set

`Distance.hpp` provides, as interchangeable policies **[V]**: `Equality`,
`Range`, `RatioDistance`, `ManhattanDistance`, `EuclideanDistance`,
**`JSDivergence`**, `RandomDistance`, `GridBasedDistance`.

`RatioDistance` is the interesting default-case metric for GEMM — problem sizes
matter multiplicatively, not additively, so nearest-neighbour in log space beats
Euclidean. `RandomDistance` exists for exploration/testing. That a
Jensen–Shannon divergence is on the list at all indicates how much of this is
treated as a genuine statistical matching problem. **[I]**

### 4.5 The learned path — a residual MLP over *derived* features

`MLPClassification.hpp` **[V]**:

> Neural net used to **estimate efficiency values for solutions in the library**.

Structure: `StandardScaler` (mean/scale normalisation) → `DenseLayer`s →
`ResBlock`s, in `float`. Note it predicts a **per-solution efficiency**, not a
kernel-id class — so its output composes with the rest of the tree as a score
rather than replacing selection.

The feature set (`MLFeatures.hpp`) is the transferable part **[V]**:

| Feature | What it is |
|---|---|
| `FreeSizeA`, `FreeSizeB` | M, N |
| `BatchSize`, `BoundSize` | batch, K |
| `Tile0Granularity`, `Tile1Granularity` | `1/mt0`, `1/mt1` |
| `CUGranularity` | how well the tile grid fills the CUs |
| `WavesPerSIMD` | occupancy |

Only the first four are problem dimensions. The rest are **derived
occupancy/granularity ratios** — quantities that plausibly *cause* performance
rather than merely correlate with it. Any model we build, learned or analytical,
should be fed granularity and occupancy terms rather than raw shapes. **[I]**

### 4.6 Reject-and-continue, with a one-time diagnostic

A worked example of a hardware quirk handled inside the selection tree **[V]**:

> `streamKDynamicQueueSupported()` excludes StreamK dynamic-queue / work-stealing
> solutions (SK4 and the dynamic sub-path of SK5) on devices whose **XCD count is
> not a power of two**, warning the user once. This is **reject-and-continue**:
> selection falls through to another (SK3-static / non-StreamK) solution for the
> GEMM.

Three things at once: a hardware-shape predicate (XCD count parity), graceful
degradation to a different algorithm rather than failure, and a **warn-once**
diagnostic so the fallback is visible without being noisy. That combination is a
good template for our capability gates, which today tend to be binary
supported/unsupported. **[I]**

### 4.7 `rocisa` — the Python/C++ boundary, and what it costs

TensileLite's assembly generator is not Python. `rocisa/` is a C++ module bound via
**nanobind**, and `KernelWriter.py` calls into it to emit instructions. Reading its
developer docs is worthwhile because it is the same hybrid we are: a Python compiler
driving a C++ core. **[V]**

- **IR nodes carry a mandatory deep-copy contract.** Anything inheriting `Item` or
  `Instruction` must supply a copy constructor *and* override
  `clone() -> std::shared_ptr<Item>`, with the instruction to "make sure you deepcopy any
  pointer". Python-side `__deepcopy__` is wired explicitly per class. Deep copy is treated
  as a first-class requirement of the node type, not an afterthought.
- **The boundary is sharper than it looks.** *"Vector memory management between Python and
  C++ is different, so exporting vectors to Python is copy instead of reference."* So
  `module.items()` returns a **copy** — elements are `shared_ptr`, so you can mutate what
  they point at, but **you cannot assign or replace an element** through that handle. That
  is exactly the class of bug that is silent in Python and impossible to reason about from
  the C++ side.
- **Convenience costs throughput at the boundary.** `countType(module, Instruction)` is kept
  because it is handy, with the explicit note that it *"runs slower than directly using
  templates"* and that a templated `countInstruction(module)` should be added and exported
  when it matters. Convenience wrappers are marked as prototyping tools rather than
  quietly becoming the default.
- **Staleness is a hard error, not a warning.** `rocisa/__init__.py` compares source
  timestamps against a generated `_build_info.py`; if any `.cpp/.hpp/.h/.def/.inc` is newer
  than the loaded `.so`, import raises with a rebuild message. Pre-built wheels omit
  `_build_info.py` and skip the check.

**[I]** Three of these are directly worth stealing for our own Python↔C++ seam. The
import-time staleness check is the cheapest and highest-value — we have already lost time
this session to a `tessera-opt` binary that silently did not match its sources, and an
equivalent guard turns that from a confusing test failure into one clear message. The
copy-not-reference boundary rule and the "convenience wrapper is for prototyping" note are
both things better written down than rediscovered.

---

## 5. Concrete algorithms worth taking

### 5.1 LDS swizzle for bank-conflict elimination

The most immediately usable algorithm in the survey, and it lands directly on the
ROCm audit's open "bounds-aware swizzle" item. **[V]**

**Hardware model (GFX950).** LDS has 64 banks × 4 B = 256 B per bank row = exactly
16 `dwordx4` columns. Other architectures have 32 banks → 128 B rows → 8 columns.
When a tile row's K spans fewer than the columns-per-bank-row, multiple tile rows
pack into one bank row, and reads from different rows at the same column offset
collide.

**Parameters** (`LDSSwizzleParams` in `LowerTile.cpp`):

```
numColumns        = tileK / (128 / elementBits)      // dwordx4 chunks per tile row
columnsPerBankRow = numBanks / 4                     // 16 on GFX950, 8 on 32-bank parts
rowsPerBankRow    = columnsPerBankRow / numColumns
bankRowIdx        = row / rowsPerBankRow
```

| dtype / macK | cols/row × rows/bank-row |
|---|---|
| FP4 macK=128 | 4 × 4 |
| FP4 macK=256 | 8 × 2 |
| FP8 macK=128 | 8 × 2 |
| FP16 macK=128 | 16 × 1 — **no conflict, swizzle skipped** |
| FP16 macK=64 | 8 × 2 |
| FP16 macK=32 | 4 × 4 |

**The transform:** column-level pair-swap plus circular rotation. *Only column
indices are permuted; row assignments are unchanged.* Skipped entirely when
`numColumns >= columnsPerBankRow` (`LDSSwizzleParams::noConflicts()`).

**The non-obvious hardware detail** — and the reason a naive swizzle is wrong:

> "The LDS unit processes a `ds_read_b128` from a 64-lane wave in **4 phases,
> each executing 16 threads** simultaneously. The 16 threads in a phase access LDS
> in parallel, so **bank conflicts only occur between threads within the same
> phase**."

And the phases are **not contiguous lane ranges**:

| Phase | Threads |
|---|---|
| 0 | T0-3, T12-15, T20-23, T24-27 |
| 1 | T32-35, T44-47, T52-55, T56-59 |
| 2 | T4-7, T8-11, T16-19, T28-31 |

Any conflict analysis that assumes lanes 0–15 form a phase will compute the wrong
answer. **[V]**

Note this table is GFX950/wave64; gfx1151 is wave32 with a different bank count,
so the *method* transfers but every constant must be re-derived per target. **[I]**

### 5.2 Workgroup mapping for cache locality

> "Workgroup mapping parameters specify how rocRoller maps the GPU workgroup
> number (hardware) to tile numbers (software). Workgroup mapping is done to
> increase cache efficiency." **[S]**

Mechanically: the hardware workgroup number is a pre-populated SGPR, exposed in
the coordinate graph as a `Workgroup` node, attached to dangling
`MacroTileNumber` leaves during the `ConnectWorkgroups` transformation — and the
mapping policy is applied **in that same pass**.

The idea worth taking is the *placement* **[I]**: grid→tile remapping is a
**coordinate-graph rewrite**, not a special case buried in the launcher. Our
existing `head_first_xcd` swizzle is the same class of optimization implemented
ad hoc; making it a layout/index transform would let the cost model see it.

### 5.3 Register placeholders — a codegen anti-pattern

`Register::Value::Placeholder` forces a specific physical register. It is
*required* when a value must persist in the same register across loop iterations.
Used unnecessarily, it defeats an optimization inside `Expression::generate()`
that would otherwise assign the destination directly — costing an extra register
and an extra `v_mov` per use. The optimization only fires when `nullptr` is passed
to `generate()`. **[S]**

The general lesson **[I]**: *pinning is a constraint, and constraints must be
justified per use.* A codegen API that makes pinning the ergonomic default will
silently inflate register pressure — the exact quantity `rocm_tiling.py` treats as
the dominant tiling lever.

---

## 6. Ranked take / skip for Tessera

**Take — high value, low coupling:**

1. **Observer scheduling (`peek`/`modify`/`observe`)** — §2.3. Gives a
   before-commitment cost query and unifies our three AMD completion mechanisms
   as observers rather than passes. Biggest single *mechanism* here.
2. **Bit-granular `Merge`/`Unmerge`, plus `XorTransform`** — §3.2, §3.8. Swizzle
   stops being a special case: XOR preshuffle is one transform in a chain, and
   Morton/Z-order is *just* a `Merge` over bit-split dimensions in permuted
   order. Bit-granular split/join buys the whole swizzle family with no new
   operators.
3. **The static bank-conflict analyzer** — §3.8. Computes N-way conflict from a
   descriptor alone, so "this layout is conflict-free for a warp-wide access" is
   a **unit-testable assertion** with no device. The cheapest correctness gate we
   could add to LDS-layout work.
4. **The LDS swizzle algorithm** — §5.1. Directly serves the open bounds-aware
   swizzle item; method transfers, constants must be re-derived per target.
5. **Fail-loud on partial metadata** — §1.6. Cheap, and it converts a class of
   silent mis-optimization into a diagnostic.
6. **`EvaluationTime` on expressions** — §2.2. Makes the symbolic-dim policy
   enforceable instead of conventional.
7. **Optimization remarks / loop health** — §1.7. Structured "why is this
   schedule bad" output that needs no hardware.
8. **Intrawave/interwave as a named schedule axis** — §3.5. A discrete,
   documented policy choice our Schedule IR should carry explicitly.
9. **The compile-time locality metric** — §3.7. A step-distance histogram over a
   materialized access order is a hardware-free static cost signal, and it lands
   squarely on the mock-analytical-cost-model finding in
   [`TILESIGHT_ASSESSMENT.md`](TILESIGHT_ASSESSMENT.md). Cheapest real answer to
   that gap.

10. **Selection as a composable tree of single-concern nodes** — §4.2. The
    structural answer to combining hand-tuned, generated and predicted kernels:
    not one arbiter, but a tree where analytical selection, benchmark lookup and
    a learned model are sibling node types.
11. **`fitness` out-param and `findTopSolutions(N)`** — §4.3. An arbiter that
    only answers "what's best" cannot measure; top-N plus a confidence score are
    the primitives that make an accuracy budget enforceable.
12. **Derived granularity features, not raw shapes** — §4.5. Tile/CU granularity
    and waves-per-SIMD over M/N/K, for any cost model we build, learned or
    analytical.
13. **Reject-and-continue with warn-once** — §4.6. Capability gates that degrade
    to another algorithm and say so once, instead of binary supported/unsupported.

14. **Import-time staleness check on the C++ extension** — §4.7. rocisa raises on
    import if any source is newer than the built `.so`. We lost time this session to a
    stale `tessera-opt`; this converts that into one clear message.

**Take — larger, worth designing toward:**

15. **Name Y-space** — §3.1. The algorithm's local view of its own work is the one
   coordinate space our IR does not have, and it is what lets the distribution
   become a separate swappable object. Highest-leverage *conceptual* item in this
   document.
16. **Distribution-as-a-value** — §3.3, §3.9. `tile_distribution_encoding` makes
    the thread↔data mapping comparable and enumerable — a search coordinate for
    the autotuner rather than a code variant. §3.9 raises the stakes: vector
    width, access count, and traversal order are all **derived** from the
    encoding, so this is what makes the rest of the machinery generated rather
    than authored.
17. **Derive the access pattern, don't author it** — §3.9. Only *which data is
    mine* and *what to compute* are written; how to fetch it and in what order
    are consequences. Worth testing our tile lowering against: how many of those
    four are currently hand-specified?
18. **Static shape / dynamic origin as the symbolic-dim seam** — §3.9. CK puts
    window *lengths* in the type and the *origin* in a runtime field. That is a
    clean, load-bearing instance of Decision #28's `static | bucket | dynamic`
    policy.
19. **A closed transform operator set with bidirectional + incremental ops** —
    §3.2. Especially `update_lower_index()`: a tile IR that can only recompute
    absolute coordinates pays for it in every loop.
20. **The coordinate hypergraph as a representation** — §2.1. Take the index
    algebra; do *not* take the single-mutable-IR architecture, which trades away
    the verifiable stage boundaries our lit discipline depends on.
21. **Serializable IR as cache key + fixture** — §2.6.
22. **"New arch = data only, no source edits"** as an explicit acceptance test for
    the backend-plugin seam — §1.3.

**Skip / already have:**

- Two IR levels (§1.1) and an opt-style driver with FileCheck tests (§1.8) — we
  have both; note them as convergence evidence, not work.
- Problem/Policy split (§3.4) — our Graph IR / Schedule IR split is the same
  instinct; the only delta is making Policy substitutable as a value.
- `Component` registry (§2.4) — our capability/pipeline registries already cover
  this; the only delta is predicate-based selection.
- `Generator<T>` coroutines (§2.5) — a C++ ergonomics choice, not an architecture.
- Declarative peephole DSL (§1.5) — attractive, but MLIR PDL/rewrite patterns
  already occupy this slot for us.
- CK's template-metaprogramming implementation strategy — the *framework* is the
  idea; C++ compile-time specialization is not a model we should copy into an
  MLIR-based compiler.

---

## 7. Not read

**Within these three projects.** StinkyTofu: DAG scheduler internals
(`src/transforms/asm/dag/` — `ReadyQueue.hpp`, and a `CDNA5.hpp` implying a
generation beyond gfx1250), `AsmVerifierPass`, `adding-architecture.md`, the
`python_module` bindings. rocRoller: `CoordinateGraph/` docs, `lib/` source,
`GPUArchitectureGenerator`.

Composable Kernel is the largest partial read — roughly **17 of its ~28
conceptual documents remain unread**. Still on-topic:

| Doc | Why it matters |
|---|---|
| `static_distributed_tensor.rst` (16 KB), `load_store_traits.rst` (16 KB) | the zero-overhead implementation strategy |
| `buffer_views.rst` (23 KB), `tensor_views.rst` (16 KB), `descriptors.rst`, `adaptors.rst` | the layered view stack under the algebra |
| `thread_mapping.rst` (18 KB), `hardware/` | logical→physical thread mapping |
| `coordinate_systems.rst` (20 KB), `tensor_coordinates.rst` (14 KB) | the P/Y/X/D treatment in full |
| `convolution_example.rst` (20 KB) | the framework applied to a non-GEMM op |

Read in this survey: `introduction_motivation`, `transforms`, `tile_distribution`,
`encoding_internals`, `coordinate_movement`, `space_filling_curve`,
`swizzling_example`, `lds_index_swapping`, `tile_window`, `sweep_tile`,
`CK-Tile-intra-inter-wave`, `TERMINOLOGY` — 12 of ~28.

Also unread in CK: `tile_engine/`, `codegen/`, `experimental/`, and the actual
`include/ck_tile/` headers — this survey read the *documentation*, not the
implementation.

**Beyond these three**, still unread:

1. **hipBLASLt kernel *generation*** — §4 read the selection side only.
   `KernelWriter.py` / `KernelWriterAssembly.py`, the `rocisa` C++ *sources* (its
   developer docs are read in §4.7; `include/` and `src/` are not), `Components/` (modular MAC / global-read / scheduling blocks), and the
   three-phase `BenchmarkProblems → LibraryLogic → ClientWriter` tuning pipeline
   are all unread. `ContractionProblemPredicates.hpp` alone is 119 KB.
2. **`ExactLogicLibrary` / `MapLibrary` / `CachingLibrary` internals** — §4.2
   names them from their headers; only `MatchingLibrary` and
   `GranularitySelectionLibrary` were read in any detail.
3. **rocWMMA** — cooperative-matrix fragments in portable form.
4. **AITER** — surveyed at brief depth in the ecosystem doc, never at source
   level.

---

## 8. Sources

All `ROCm/rocm-libraries@develop`.

**StinkyTofu** (`shared/stinkytofu/`) — `docs/developer/architecture.md`,
`wmma-vgpr-reorder-pass.md`, `pattern-grammar.md`, `cluster-barrier.md`;
`docs/user/stinky-waitcnt-insertion-pass.md`; `hardware/src/gfx/Gfx1250/`;
`include/stinkytofu/core/Types.hpp`.

**rocRoller** (`shared/rocroller/`) — `CLAUDE.md`; `docs/src/DesignOverview.md`,
`LDSSwizzling.md`, `RegisterPlaceholders.md`, `WorkgroupMapping.rst`.

**Composable Kernel** (`projects/composablekernel/`) — `TERMINOLOGY.md`;
`docs/conceptual/CK-Tile-intra-inter-wave.rst`;
`docs/conceptual/ck_tile/{introduction_motivation,transforms,tile_distribution,
encoding_internals,coordinate_movement,space_filling_curve,swizzling_example,
lds_index_swapping,tile_window,sweep_tile}.rst`.

**hipBLASLt / TensileLite** (`projects/hipblaslt/`) — `CLAUDE.md`,
`tensilelite/CLAUDE.md`; `tensilelite/include/Tensile/{SolutionLibrary,Distance,
MatchingLibrary,GranularitySelectionLibrary,MLPClassification,MLFeatures}.hpp`;
`tensilelite/rocisa/{README.md,docs/}`.

**LLVM** (`ROCm/llvm-project@amd-staging`) — `llvm/lib/Target/AMDGPU/SISchedule.td`
(the gfx1250/gfx1251 machine model, recorded in the gfx1250 target reference §2.5
rather than here).

---

*See also:
[`../backend/rocm/GFX1250_MI450_COMPILER_REFERENCE.md`](../backend/rocm/GFX1250_MI450_COMPILER_REFERENCE.md)
(target facts) ·
[`../backend/rocm/ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md`](../backend/rocm/ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md)
(ecosystem survey) ·
[`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md) (our
direction) · [`TILESIGHT_ASSESSMENT.md`](TILESIGHT_ASSESSMENT.md) (external
assessment, same reference role).*
