"""Per-target kernel emit / arbiter subsystem (COMPILER_REFACTOR_PLAN A–E).

Public façade. This re-exports the stable, target-agnostic surface — the emit
protocol (B2), the synth→compile→cache loop (B4), the accuracy-budgeted
candidate arbiter (D1), and the measured autotune loop (D2) — so callers use
``from tessera.compiler import emit`` instead of reaching into submodules.

Deliberately NOT eagerly imported: the per-arch backend plugins
(``apple_msl``, ``nvidia_cuda``, ``rocm_hip``, ``x86_llvm``, ``x86_aocl_dlp``).
Each self-registers its emitter / compiler / runner / candidates as an import
side effect, and ``get_emitter`` / ``get_compiler`` bootstrap the Apple
reference on demand — so importing this package stays cheap and does not force a
particular backend's registration. Import a plugin module explicitly (or call a
``get_*`` that bootstraps it) to register its lanes.

The two ``default_cache`` accessors are re-exported under distinct names —
:func:`kernel_cache_default` (compiled-kernel cache, B4) and
:func:`measure_cache_default` (autotune measure cache, D2) — since the subsystem
has one of each.
"""

# B2 — target-agnostic emit / execute protocol + registries.
from tessera.compiler.emit.kernel_emitter import (
    METAL_TARGETS,
    REFERENCE_EXECUTIONS,
    EmitError,
    KernelEmitter,
    KernelRunner,
    KernelSource,
    RunnerError,
    SpecPolicy,
    active_runner,
    bucket_key,
    emit_kernel,
    get_emitter,
    get_runner,
    register_emitter,
    register_runner,
)

# B4 — content-addressed synth→compile→cache loop.
from tessera.compiler.emit.kernel_cache import (
    CompiledKernel,
    CompileError,
    CompileFn,
    KernelCache,
    build,
    cache_key,
    get_compiler,
    register_compiler,
)
from tessera.compiler.emit.kernel_cache import default_cache as kernel_cache_default

# D1 — accuracy-budgeted candidate arbiter (+ D3 dispatch log).
from tessera.compiler.emit.candidate import (
    OP_ATTENTION,
    OP_FUSED_REGION,
    OP_GATED_MATMUL,
    OP_MATMUL,
    OP_POINTWISE,
    ArbiterError,
    Candidate,
    Tier,
    arbiter_dispatch_histogram,
    arbiter_dispatch_log,
    arbitrate,
    candidates_for,
    register_candidate,
    register_op_kind,
    reset_arbiter_dispatch_log,
    run_arbitrated,
    verify_by_reference,
    verify_candidate,
)

# D2 — measured autotune loop + fleet-shared corpus.
from tessera.compiler.emit.autotune import (
    MeasureCache,
    MeasureRecord,
    TIMING_DEVICE,
    TIMING_END_TO_END,
    corpus_winner,
    corpus_path,
    load_corpus,
    measure_latency,
    measured_arbitrate,
    run_measured_arbitrated,
    save_corpus,
)
from tessera.compiler.emit.autotune import default_cache as measure_cache_default

__all__ = [
    # B2 protocol
    "SpecPolicy", "KernelSource", "KernelEmitter", "KernelRunner", "EmitError",
    "RunnerError", "register_emitter", "get_emitter", "emit_kernel",
    "register_runner", "get_runner", "active_runner", "bucket_key",
    "METAL_TARGETS", "REFERENCE_EXECUTIONS",
    # B4 cache
    "CompiledKernel", "CompileError", "CompileFn", "KernelCache", "build",
    "cache_key", "register_compiler", "get_compiler", "kernel_cache_default",
    # D1 arbiter
    "Tier", "Candidate", "ArbiterError", "register_candidate", "candidates_for",
    "arbitrate", "run_arbitrated", "verify_candidate", "register_op_kind",
    "verify_by_reference", "arbiter_dispatch_log", "arbiter_dispatch_histogram",
    "reset_arbiter_dispatch_log", "OP_FUSED_REGION", "OP_ATTENTION",
    "OP_GATED_MATMUL", "OP_POINTWISE", "OP_MATMUL",
    # D2 autotune
    "MeasureCache", "MeasureRecord", "measured_arbitrate",
    "run_measured_arbitrated", "measure_latency", "load_corpus", "save_corpus",
    "corpus_path", "corpus_winner", "measure_cache_default",
    "TIMING_END_TO_END", "TIMING_DEVICE",
]
