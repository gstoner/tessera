---
last_updated: 2026-07-06
audit_role: reference
---

# Backend Plugin Handoff — pick up Workstream C on your box

> **Who this is for:** a Claude (or human) working on the **Strix Halo**
> (ROCm gfx1151 + x86 Zen 5) or **NR2 Pro** (NVIDIA sm_120) box, picking up
> Workstream **C** — the per-arch codegen plugins — against the target-agnostic
> synthesizer that Workstream **B** already built and merged on the Mac.
>
> **Read first:** [`COMPILER_REFACTOR_PLAN.md`](COMPILER_REFACTOR_PLAN.md) §3
> (workstreams), §5 (definition of done), §7 (three-system coordination + setup
> pins). This doc is the *build recipe* those sections point at.

You do **not** need to understand the whole synthesizer. You implement **three
small seams** for your target; the arch-agnostic middle-end (region discovery,
fusion cost, the F4 correctness oracle, the shape-bucket cache) is done and calls
into your seams. Apple MSL is the fully-worked reference implementation — copy its
shape.

---

## 1. What Workstream B already gives you (all merged on `main`)

```
Graph IR ─▶ fusion_core (arch-agnostic)                         ── DONE (B1)
              • discover_* : find fusable regions
              • *_cost / should_fuse_* : profitability
              • FusedRegion / AttentionRegion / … : the region model  ← YOUR INPUT
              • verify_synthesized_* : the F4 numpy-reference oracle   ← GATES YOU
                 │
                 ▼
           emit.kernel_emitter  (the plugin protocol)             ── DONE (B2)
              • KernelEmitter.emit(region, spec, dtype, dims) → KernelSource   ← SEAM 1
              • KernelRunner.run_* (region, *inputs) → (out, execution)        ← SEAM 3
              • SpecPolicy(static|bucket|dynamic) + bucket_key(dims, spec)
                 │
                 ▼
           emit.kernel_cache  (synth→compile→cache loop)          ── DONE (B4)
              • register_compiler(target, compile_fn)  compile_fn(src)→artifact ← SEAM 2
              • cache_key(source, dtype, target) : content-addressed
              • build(region, target, spec, dims, dtype) → CompiledKernel
```

Key source files (Python, on the Mac and mirrored to your box):

| File | What it is |
|---|---|
| `python/tessera/compiler/fusion_core.py` | Region model + discovery + cost + the F4 oracles. **Read the `FusedRegion` etc. dataclasses — they are your emitter's input.** |
| `python/tessera/compiler/emit/kernel_emitter.py` | `KernelEmitter`, `KernelRunner`, `SpecPolicy`, `KernelSource`, `bucket_key`, `REFERENCE_EXECUTIONS`, the registries. |
| `python/tessera/compiler/emit/kernel_cache.py` | `register_compiler`/`get_compiler`, `cache_key`, `KernelCache`, `build`. |
| `python/tessera/compiler/emit/apple_msl.py` | **The reference impl** — `AppleMSLEmitter`, `AppleMSLRunner`, `_apple_compile_fn`. Your `emit/<target>.py` mirrors this structure. |
| `tests/unit/test_kernel_emitter.py`, `tests/unit/test_kernel_cache.py` | The contract tests. Your backend adds analogous ones. |

---

## 2. The three seams you implement

Everything lives in one new module, `python/tessera/compiler/emit/<target>.py`
(e.g. `x86_llvm.py`, `rocm_hip.py`, `nvidia_ptx.py`). It registers itself on
import, exactly like `apple_msl.py`.

### Seam 1 — `KernelEmitter` (region → source)

```python
class MyEmitter(KernelEmitter):
    target = "x86"          # your backend id
    lang   = "c-llvm"       # source dialect tag ("msl" | "ptx" | "amdgcn" | "c-llvm")

    def can_emit(self, region) -> bool:
        return isinstance(region, (FusedRegion, NormChainRegion, ...))

    def emit(self, region, *, spec=SpecPolicy.BUCKET, dtype="f32", dims=None) -> KernelSource:
        source = my_synthesize(region, dtype)      # ← your codegen: emit the kernel text
        entry  = "tessera_x86_matmul_epi"          # entry-point symbol the runtime launches
        key    = bucket_key(dims, spec, dim_names=getattr(region, "dim_names", None))
        return KernelSource(source=source, entry=entry, lang=self.lang, spec=spec, shape_key=key)
```

Rules: raise `EmitError` for a region kind or `SpecPolicy` you don't support —
**never** emit a differently-specialized kernel and mislabel it (Decision #21).
`DYNAMIC` isn't required yet; `bucket` is the target. Keep `emit` **pure** (no
device calls) so it stays host-free/testable on any box.

### Seam 2 — `compile_fn` (source → artifact)

```python
def my_compile_fn(source: KernelSource):
    # x86:     clang -O3 -mavx512f -shared  → .so path or bytes
    # NVIDIA:  ptxas source.source          → CUBIN bytes
    # ROCm:    hipcc / clang-amdgcn         → HSACO bytes
    return artifact          # opaque handle; return None ONLY for compile-on-launch
```

Register with `register_compiler("x86", my_compile_fn)`. The `build()` loop keys
the artifact by `cache_key(source, dtype, target)` and reuses it on a hit — you
get the shape-bucket cache for free. Raise (or let the toolchain raise) on a
compile failure; `build` wraps it in `CompileError` (never a silent no-op).

### Seam 3 — `KernelRunner` (execute → `(out, execution)`)

```python
class MyRunner(KernelRunner):
    target = "x86"
    def run_fused_region(self, region, A, B, bias=None, *a, **k):
        out = my_launch(region, A, B, bias)        # dlopen the .so, call entry, run
        return out, "x86_native"                   # ← your REAL-execution tag
    # ... run_fused_attention / run_gated_matmul_region / run_pointwise_graph
```

**The execution-tag contract (this is what makes the F4 oracle gate _you_):**
return your own real tag (`"x86_native"`, `"rocm_hip"`, `"cuda"`) when a real
device kernel ran, or a tag in `REFERENCE_EXECUTIONS` (`"reference"`/`"fallback"`)
when you fell back to numpy. The oracle compares your output to the numpy
reference **iff** a real kernel ran — you do **not** pretend to be Metal.

Register: `register_runner(MyRunner(), default=False)` — **`default=False`** so
you don't hijack the active runner from Apple; the D1 arbiter selects per
`(op, shape-bucket, dtype, target)` later.

---

## 3. Copy-paste skeleton

```python
"""Workstream C — <target> codegen plugin. Mirrors emit/apple_msl.py."""
from __future__ import annotations

from tessera.compiler.emit.kernel_cache import register_compiler
from tessera.compiler.emit.kernel_emitter import (
    EmitError, KernelEmitter, KernelRunner, KernelSource, SpecPolicy,
    bucket_key, register_emitter, register_runner,
)
from tessera.compiler.fusion_core import (
    AttentionRegion, FusedRegion, GatedMatmulRegion, NormChainRegion,
    PointwiseGraphRegion, PointwiseReduceRegion,
)

_TARGET = "x86"          # TODO: your backend id


class _Emitter(KernelEmitter):
    target = _TARGET
    lang = "c-llvm"      # TODO
    def can_emit(self, region):
        return isinstance(region, FusedRegion)          # TODO: widen as you add kinds
    def emit(self, region, *, spec=SpecPolicy.BUCKET, dtype="f32", dims=None):
        if not self.can_emit(region):
            raise EmitError(f"{_TARGET}: cannot emit {type(region).__name__}")
        source = _synthesize(region, dtype)             # TODO: your codegen
        key = bucket_key(dims, spec, dim_names=getattr(region, "dim_names", None))
        return KernelSource(source=source, entry="tessera_x86_kernel",
                            lang=self.lang, spec=spec, shape_key=key)


def _compile_fn(source):
    raise NotImplementedError("TODO: clang/ptxas/hipcc → artifact")


class _Runner(KernelRunner):
    target = _TARGET
    def run_fused_region(self, region, *a, **k):
        raise NotImplementedError("TODO: launch → (out, 'x86_native')")
    def run_fused_attention(self, region, *a, **k): raise NotImplementedError
    def run_gated_matmul_region(self, region, *a, **k): raise NotImplementedError
    def run_pointwise_graph(self, region, *a, **k): raise NotImplementedError


register_emitter(_Emitter())
register_compiler(_TARGET, _compile_fn)
register_runner(_Runner(), default=False)
```

---

## 4. How you verify (per plan §5 definition of done)

1. **F4 correctness — the anti-cheat gate.** For every region your backend runs:
   `verify_synthesized_region(region, runner=MyRunner())` must return `True`
   (matches numpy) — and demonstrably `False` if you inject a wrong kernel. This
   is the same oracle Apple uses; it now compares *your* real-execution tag. This
   is the one gate that makes a compiled kernel *safe to prefer* over a hand-tuned
   one.
2. **Host-free authoring stays green on the Mac** — `emit` is pure, so
   `mypy python/tessera/`, `ruff`, and the emitter/cache unit tests run without
   your hardware. Add `tests/unit/test_<target>_plugin.py` mirroring
   `test_kernel_emitter.py` (registration, `emit` shape, `EmitError` paths).
3. **Silicon proof is a committed artifact** (plan §7.3): an
   `execute_compare_fixture` (your kernel == numpy on real hardware) + a
   `<target>_hot_paths.json` perf ratchet. The Mac's host-free gate then asserts
   their *shape* between silicon runs.
4. **Golden-IR (E1)** — snapshot your emitted Target IR for the fixture set; any
   later change that perturbs it fails on the Mac. Keep fixtures on ordered
   `CHECK`, not `CHECK-DAG`.

---

## 5. Per-backend task cards

### C1 · x86 clang plugin — `[MAC]` author, `[AMD]` execute on Zen 5
- **Box:** Strix Halo (Ryzen AI Max+ 395, Zen 5). **AVX-512 only — no AMX** on
  this fleet; never emit AMX. The NR2 Pro's 265F CPU has **no AVX-512** — do not
  build the x86 backend there.
- **compile_fn:** `clang -O3 -mavx512f -mavx512bf16 -shared` → `.so`; launch via
  `ctypes`/`dlopen`. Cheapest 2nd impl — start here to validate the framework.
- **Reuse:** the existing `TileToX86Pass` AMX/AVX512 GEMM
  (`src/compiler/codegen/tessera_x86_backend/`) as the codegen reference; the
  `spec_policy` replaces its hard `"requires static shapes"` gate.
- **Optional Tier-3 candidate:** register **AOCL-DLP** (`amd/aocl-dlp`, AVX-512
  BLIS-family DL primitives) as a hand-tuned candidate the arbiter measures — see
  plan §C1 + the `reference_aocl_dlp_x86` note. Check its license before linking.

### C2 · NVIDIA emit pipeline — `[MAC]` author → `[NV]` prove
- **Box:** NR2 Pro (RTX 5070 Ti, **sm_120a**, CUDA 13.3, PTX ISA 9.3, 100 KB
  smem/SM, FP4 `mma.sync.block_scale`; **no tcgen05/wgmma** — consumer Blackwell).
- **compile_fn:** serialize emitted PTX → `ptxas` → CUBIN →
  `tsrRegisterGpuLauncher` launch bridge. **The bridge is the long pole** (the
  NVIDIA counterpart to `apple_gpu_runtime.mm`), ahead of broadening shapes.
- **Reuse:** `python/tessera/compiler/ptx_emit.py` (keep sm_120 `mma.sync`; it's
  bf16-only/few-shape today — extend). Today's executing sm_120 matmul runs via
  the shipped `libtessera_nvidia_gemm.so`, **not** the emit path — so C2 is a
  *new lead lane*, not a refactor. **Never route crown-jewel GEMM through the
  generic synthesizer unless it measures competitive** (lead-safety).

### C3 · ROCm emit pipeline — `[MAC]` author → `[AMD]` prove
- **Box:** Strix Halo, ROCm 7.2.4, **gfx1151 = RDNA 3.5**: WMMA 16×16×16
  F16/BF16/IU8/IU4, **no FP8/BF8 WMMA** (RDNA4-only). Verify any op exists on the
  target in `docs/reference/isa/rdna/` before emitting.
- **compile_fn:** drive the existing gfx1151 WMMA + CDNA MFMA `Generate*Kernel`
  passes through the shared loop → `hipcc`/clang-amdgcn → HSACO → launch bridge;
  reuse the async-token SSA model in `ROCMWaveLdsPipeline`.

---

## 6. Lead-safety — the rules you must not break

ROCm and CUDA are the **lead performance targets**; the generic framework raises
the floor and must **never cap their ceiling** (Theory §1, Decision #28):

- Hand-emitted `wgmma`/`mma.sync`/MFMA/WMMA stay **first-class arbiter
  candidates** — displaced only when a compiled kernel is both *faster* and
  *in accuracy budget*.
- Crown-jewel GEMM/attention stay Tier 2/3; the generic synthesizer targets the
  **fusable-DAG middle ground** (epilogues, pointwise chains, small attention).
- Every merge is **neutral-or-better** on the lead perf ratchets (E2). No
  silent regression.

---

## 7. Get your box ready

Setup pins are in [`COMPILER_REFACTOR_PLAN.md`](COMPILER_REFACTOR_PLAN.md) §7.4
and [`../../GETTING_STARTED.md`](../../GETTING_STARTED.md):

- **Strix Halo:** Ubuntu 24.04 → `bash scripts/setup_ubuntu.sh` (LLVM/MLIR 22 from
  apt.llvm.org — ROCm's bundled LLVM has no MLIR); ROCm 7.2.4 at `/opt/rocm`;
  `-DTESSERA_ENABLE_HIP=ON -DTESSERA_BUILD_ROCM_BACKEND=ON`; `.venv` caps
  numpy<2.2. gfx1151 execution needs a GPU + `/dev/kfd`.
- **NR2 Pro:** CUDA 13.3; `-DTESSERA_ENABLE_CUDA=ON`; target `sm_120a`.

`git pull`, `export PYTHONPATH=python`, then `python3 -m pytest tests/unit/ -m
"not slow"` to confirm the framework is present before you add your plugin.

---

## 8. The one-paragraph summary

The synthesizer is now a target-agnostic framework. To bring your backend online:
write `emit/<target>.py` with an **emitter** (region → source), a **compile_fn**
(source → artifact), and a **runner** (execute → `(out, "<your>_native")`),
register all three, and prove it with the shared **F4 oracle**
(`verify_synthesized_*(runner=…)`) + a committed silicon fixture + perf ratchet.
Apple (`emit/apple_msl.py`) is the worked example. Keep authoring host-free, keep
the leads' hand-tuned kernels first-class, and never regress a lead ratchet.
