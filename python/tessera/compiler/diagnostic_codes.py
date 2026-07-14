"""Arch-1 (2026-05-22) — Central registry of Tessera diagnostic codes.

Before this sprint, diagnostic codes (e.g., ``SYMDIM_BINDING_VIOLATION``,
``QUEUE_PUSH_QUEUE_PROVENANCE``) were defined only at the C++ ``emitOpError``
site.  Discovering them required ``grep`` across ``src/``; their meaning lived
in the surrounding code comments and in sprint-specific lit fixtures.

TSOL-2 (2026-05-22) extends the registry to cover three Python-side
families too, so MLIR and Python codes share one drift gate:

  * ``E_*``        — :class:`tessera.diagnostics.TesseraErrorCode` enum
                     (raised by Python frontend / shape inference paths).
  * ``JIT_*``      — :class:`tessera.compiler.JitDiagnosticCode` enum
                     (JIT-level outcomes from P0-2 sprint).
  * ``TS_ERR_*``   — TSOL spec contracts.  Listed for spec traceability;
                     status reflects whether they're implemented in
                     Python today (most are advisory contracts the
                     implementation should honor as it grows).

This module is the single source of truth that:

  * Names every code Tessera emits or contractually promises, with
    severity / pass-origin / human summary / fix-hint / spec back-link
    / language (mlir vs python) / status (implemented vs spec_contract).
  * Lets a drift gate cross-check across BOTH src/ (C++) and
    python/tessera/ (Python) emission sites.

The registry is consulted by:

  * ``tests/unit/test_diagnostic_code_registry.py`` (drift gate).
  * ``docs/audit/diagnostic_codes.md`` (generated dashboard).
  * Future ``JitFn.explain()`` extensions that translate raw MLIR
    diagnostic strings to actionable Python guidance.

Code emission patterns scanned by the drift gate:

  * C++ (MLIR-side): ``op->emitOpError("CODE_NAME: human detail...")``
    — the regex matches the all-caps prefix before the first ``:``.
  * Python (E_*/JIT_*): ``"CODE_NAME"`` as a string literal in
    enum values or assertion messages.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticCode:
    """One Tessera diagnostic code.

    Fields
    ------
    code
        The token Tessera emits.  For MLIR codes this is the all-caps
        prefix before the ``:`` in ``emitOpError`` calls; for Python
        ``E_*`` / ``JIT_*`` codes it's the enum value string; for
        TSOL ``TS_ERR_*`` it's the contract identifier from the spec.
    pass_origin
        Symbolic name of the pass / verifier / Python module that
        emits the code.  Use the C++ class name (``SymbolicDimEquality``)
        for MLIR codes or the Python module path
        (``tessera.diagnostics``) for Python codes.  TSOL contracts
        use ``"TSOL spec"``.
    severity
        ``"error"`` (default — failure of ``verify()`` / pass) or
        ``"warning"`` (advisory; rarely used today).
    summary
        One-sentence human-readable explanation of what the code means.
    fix_hint
        Concrete action the user can take to silence the diagnostic.
    spec
        Optional path + section into the spec corpus that documents the
        invariant the code enforces (e.g.,
        ``"docs/spec/SHAPE_SYSTEM.md §11.2"``).
    sprint
        Which sprint introduced the code, for archaeological context.
    language
        TSOL-2 (2026-05-22): ``"mlir"`` for C++ ``emitOpError`` codes,
        ``"python"`` for Python-side enum values / exception messages.
        Drives which source tree the drift gate scans for the code's
        emission site.
    status
        TSOL-2 (2026-05-22): ``"implemented"`` (default — the code is
        emitted by real code today) or ``"spec_contract"`` (named in
        the TSOL spec but no Python emission site exists yet — the
        registry tracks it for spec traceability without requiring
        an implementation today).
    """

    code: str
    pass_origin: str
    severity: str
    summary: str
    fix_hint: str
    spec: str | None
    sprint: str
    language: str = "mlir"
    status: str = "implemented"


# ─────────────────────────────────────────────────────────────────────────
# Registry — keep alphabetised by code for easy review.
# ─────────────────────────────────────────────────────────────────────────

REGISTERED_CODES: tuple[DiagnosticCode, ...] = (
    # ── Python-side: TesseraErrorCode enum (TSOL-2, 2026-05-22) ──────────
    # These E_* codes live in `python/tessera/diagnostics.py`.  The
    # enum class is `TesseraErrorCode`; values are emitted via raised
    # exceptions like `TesseraShapeError`.
    DiagnosticCode(
        code="E_SHAPE_MISMATCH",
        pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary=(
            "A shape contract failed at Python frontend / shape "
            "inference / runtime witness time."
        ),
        fix_hint=(
            "Inspect the JIT signature's symbolic dims against the "
            "actual call-site shapes; the `TesseraShapeError` message "
            "carries the offending op + source location."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md",
        sprint="Phase 1",
        language="python",
        status="implemented",
    ),
    DiagnosticCode(
        code="E_TARGET_CODEGEN",
        pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary=(
            "Backend code generation failed (NVIDIA / ROCm / Apple / "
            "x86 lowering chain rejected the IR)."
        ),
        fix_hint=(
            "Check whether the op + dtype is in the target's "
            "capability matrix; see `docs/audit/standalone_primitive_coverage.md`."
        ),
        spec="docs/spec/TARGET_IR_SPEC.md",
        sprint="Phase 6",
        language="python",
        status="implemented",
    ),
    DiagnosticCode(
        code="E_TILE_LOWERING",
        pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary=(
            "Tile IR lowering rejected the schedule (warp / tile / "
            "mma fragment shape illegal for target)."
        ),
        fix_hint=(
            "Verify the schedule's tile knobs against the target "
            "profile's accept-set (e.g., WGMMA tile shapes from "
            "`docs/backends/nvidia/kernel-inventory.md`)."
        ),
        spec="docs/spec/TILE_IR.md",
        sprint="Phase 3",
        language="python",
        status="implemented",
    ),
    # TesseraErrorCode long-tail (TSOL-2, 2026-05-22): register the
    # remaining 22 enum values for completeness.  Each one is a real
    # E_* string emitted via the enum in
    # `python/tessera/diagnostics.py`; the drift gate verifies
    # presence in the Python source tree.
    DiagnosticCode(
        code="E_CACHE_IO", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="Schedule cache read/write failed.",
        fix_hint="Inspect filesystem permissions on the cache path or clear stale entries.",
        spec=None, sprint="Phase 6", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_COMM_INIT", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="NCCL / RCCL / NVSHMEM collective initialization failed.",
        fix_hint="Check that the matched library version meets the NCCL/RCCL ≥ 2.22 pin.",
        spec=None, sprint="Phase 4", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_DESYNC", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="Ranks diverged in a collective protocol (different reduce trees / shapes).",
        fix_hint="Ensure every rank invokes collectives in the same order with matching shapes.",
        spec=None, sprint="Phase 4", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_DRIVER", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="Underlying device driver returned an error.",
        fix_hint="Check the driver version against the pinned CUDA 13.3 / ROCm 7.2.4 minima.",
        spec=None, sprint="Phase 6", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_GRAPH_INVALID", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="Graph IR failed validation (cycles, dangling references, mismatched effects).",
        fix_hint="Inspect via `tessera.compiler.dry_run(fn)` to see which op tripped validation.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_ILLEGAL_ADDRESS", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="A kernel accessed memory outside the allocated region.",
        fix_hint="Run under cuda-memcheck / hip-sanitizer; check tile boundary conditions.",
        spec=None, sprint="Phase 6", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_LAUNCH_BAD_LAYOUT", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="Kernel launch parameters declare a layout incompatible with the kernel's accept-set.",
        fix_hint="Insert a `tessera.cast` to convert to a layout in the kernel's accept-set.",
        spec="docs/spec/SHAPE_SYSTEM.md", sprint="Phase 3",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_LAUNCH_DEVICE_MISMATCH", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="Kernel launch targeted a different device than the input tensor's residence.",
        fix_hint="Move tensors via `tensor.to(device)` before launch.",
        spec=None, sprint="Phase 6", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_LAUNCH_INVALID_SHAPE", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="Kernel launch shape (blocks/threads/cluster) is invalid for the target.",
        fix_hint="Consult the target profile's launch constraints in `gpu_target.py`.",
        spec=None, sprint="Phase 3", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_LAUNCH_STREAM_BUSY", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="Submitted launch but the target stream is already in an error state.",
        fix_hint="Synchronize and check the prior async error via the runtime's last-error API.",
        spec=None, sprint="Phase 6", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_LOSS_SCALING", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="Mixed-precision loss scaling lost too many gradient bits.",
        fix_hint="Lower the initial scale or enable dynamic scaling via `GradScaler`.",
        spec=None, sprint="Phase 5", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_MISALIGNED_ACCESS", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="A kernel performed a misaligned load/store (vectorized access on bad address).",
        fix_hint="Inspect tile alignment; ensure shared-memory bank sizes match the tile contract.",
        spec=None, sprint="Phase 3", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_NAN_INF", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="NaN / Inf detected in tensor output (caught by NaN/Inf guard).",
        fix_hint="Enable mixed-precision loss scaling or check op numerical stability.",
        spec=None, sprint="Phase 5", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_NONDETERMINISTIC", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="Deterministic mode was requested but the chosen path can't honor it.",
        fix_hint="Disable the offending fast-path or accept nondeterminism via numeric policy.",
        spec="docs/operations/Tessera_Standard_Operations.md §Determinism Contract",
        sprint="Phase 5", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_OOM", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="Host or device allocation failed.",
        fix_hint="Reduce batch/sequence dimensions or inspect buffer-pool stats.",
        spec=None, sprint="Phase 6", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_SCHEDULE_FUSE_FAIL", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="Schedule-IR fusion pass rejected the requested fusion.",
        fix_hint="Inspect via `tessera.compiler.dry_run(fn)`; some fusions need explicit attrs.",
        spec=None, sprint="Phase 3", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_TIMEOUT", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="A compute kernel exceeded its watchdog deadline.",
        fix_hint="Increase the watchdog budget or split the work into smaller launches.",
        spec=None, sprint="Phase 6", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_TIMEOUT_COMM", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="A collective operation exceeded its watchdog deadline.",
        fix_hint="Check rank health and topology; verify NCCL/RCCL fabric is healthy.",
        spec=None, sprint="Phase 4", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_TOPOLOGY", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="The mesh topology declared by the user is inconsistent with the device fabric.",
        fix_hint="Verify the mesh axes match the physical topology (NVL/PCIe/RDMA).",
        spec=None, sprint="Phase 4", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_TUNE_MEASURE_FAIL", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="Autotuner measurement run failed (kernel crashed during timing).",
        fix_hint="Inspect the autotuner SQLite cache for the failing config; mark as bad.",
        spec=None, sprint="Phase 5", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_TUNE_SPACE_EMPTY", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="Autotuner search space evaluated to zero valid configs.",
        fix_hint="Relax constraints in the autotuner search spec or fall back to a default tile.",
        spec=None, sprint="Phase 5", language="python", status="implemented",
    ),
    DiagnosticCode(
        code="E_UNKNOWN", pass_origin="tessera.diagnostics.TesseraErrorCode",
        severity="error",
        summary="An unclassified Tessera failure occurred.",
        fix_hint="Inspect the wrapped exception chain for the underlying cause.",
        spec=None, sprint="Phase 1", language="python", status="implemented",
    ),

    # ── Python-side: JitDiagnosticCode enum (P0-2 sprint) ───────────────
    # These JIT_* codes live in `python/tessera/compiler/diagnostics.py`.
    # The enum class is `JitDiagnosticCode`; values are tagged onto
    # `Diagnostic` instances surfaced by `JitFn.explain()`.
    DiagnosticCode(
        code="JIT_COMPILED_CPU",
        pass_origin="tessera.compiler.JitDiagnosticCode",
        severity="warning",
        summary=(
            "JIT compiled the function down the CPU lane — useful "
            "context, not a failure."
        ),
        fix_hint=(
            "No action required; this is an informational telemetry "
            "code emitted on successful CPU compilation."
        ),
        spec=None,
        sprint="P0-2",
        language="python",
        status="implemented",
    ),
    DiagnosticCode(
        code="JIT_COMPILED_TARGET_RUNTIME",
        pass_origin="tessera.compiler.JitDiagnosticCode",
        severity="warning",
        summary=(
            "JIT emitted Target IR for a supported program and selected the "
            "target runtime dispatch lane."
        ),
        fix_hint=(
            "No action required; inspect the runtime artifact for the exact "
            "launch contract and any explicit reference fallback."
        ),
        spec=None,
        sprint="Apple optimizer vertical slice",
        language="python",
        status="implemented",
    ),
    DiagnosticCode(
        code="JIT_EAGER_FALLBACK_ARITY",
        pass_origin="tessera.compiler.JitDiagnosticCode",
        severity="warning",
        summary=(
            "JIT fell back to eager-Python execution because the "
            "function's arity didn't match the expected JIT shape."
        ),
        fix_hint=(
            "Inspect the function signature; the JIT requires a "
            "fixed positional arity for compiled paths."
        ),
        spec=None,
        sprint="P0-2",
        language="python",
        status="implemented",
    ),
    DiagnosticCode(
        code="JIT_EAGER_FALLBACK_EMPTY",
        pass_origin="tessera.compiler.JitDiagnosticCode",
        severity="warning",
        summary=(
            "JIT fell back to eager-Python because the function "
            "produced an empty Graph IR (no ops emitted)."
        ),
        fix_hint=(
            "Confirm the function calls at least one `tessera.ops.*` "
            "or `tessera.nn.*` API; pure Python bodies don't lower."
        ),
        spec=None,
        sprint="P0-2",
        language="python",
        status="implemented",
    ),
    DiagnosticCode(
        code="JIT_EAGER_FALLBACK_UNSUPPORTED_BODY",
        pass_origin="tessera.compiler.JitDiagnosticCode",
        severity="warning",
        summary=(
            "JIT fell back to eager-Python because the function body "
            "used a Python construct the IR builder can't translate."
        ),
        fix_hint=(
            "Rewrite control flow using `tessera.control.cond` / "
            "`scan` / `while_loop` rather than native Python."
        ),
        spec=None,
        sprint="P0-2",
        language="python",
        status="implemented",
    ),
    DiagnosticCode(
        code="JIT_EAGER_FALLBACK_UNSUPPORTED_OP",
        pass_origin="tessera.compiler.JitDiagnosticCode",
        severity="warning",
        summary=(
            "JIT fell back to eager-Python because the function "
            "called an op the IR builder doesn't yet recognize."
        ),
        fix_hint=(
            "Check `op_catalog.py` for the canonical op name + "
            "namespace; some `tessera.nn.*` paths still route to "
            "Python today."
        ),
        spec=None,
        sprint="P0-2",
        language="python",
        status="implemented",
    ),
    DiagnosticCode(
        # A.2 (2026-05-31) — distinct code for scf.* eager fallback so
        # the dashboard can show structured control flow as an expected
        # eager path rather than a generic unknown-op miss.
        code="JIT_EAGER_FALLBACK_CONTROL_FLOW",
        pass_origin="tessera.compiler.JitDiagnosticCode",
        severity="warning",
        summary=(
            "JIT fell back to eager-Python because the function "
            "contains structured control flow (`tessera.scf.*` "
            "markers) that no backend currently lowers to executable "
            "code. The function runs correctly through Python; only "
            "the compiled fast path is missing."
        ),
        fix_hint=(
            "Eager Python is numerically correct and safe. To get the "
            "fast path, implement a backend pass that lowers "
            "`tessera.scf.if`/`scf.for`/`scf.while` (see "
            "`docs/audit/compiler/COMPILER_AUDIT.md` §10)."
        ),
        spec=None,
        sprint="audit-followup-A.2",
        language="python",
        status="implemented",
    ),
    DiagnosticCode(
        code="JIT_SOURCE_PROVIDED",
        pass_origin="tessera.compiler.JitDiagnosticCode",
        severity="warning",
        summary=(
            "JIT compiled using source provided via "
            "`tessera.from_text(source=...)` rather than inspected "
            "via `inspect.getsource(fn)`."
        ),
        fix_hint=(
            "Informational; no action required unless the source "
            "is unexpectedly empty."
        ),
        spec=None,
        sprint="P0-2",
        language="python",
        status="implemented",
    ),
    DiagnosticCode(
        code="JIT_SOURCE_UNAVAILABLE",
        pass_origin="tessera.compiler.JitDiagnosticCode",
        severity="error",
        summary=(
            "JIT couldn't inspect the function source (heredoc / "
            "REPL / lambda) so no constraint enforcement is possible."
        ),
        fix_hint=(
            "Pass the source explicitly via "
            "`tessera.from_text(source=...)` or move the function "
            "into an importable module."
        ),
        spec=None,
        sprint="P0-2",
        language="python",
        status="implemented",
    ),
    DiagnosticCode(
        code="JIT_TARGET_IR_ARTIFACT_ONLY",
        pass_origin="tessera.compiler.JitDiagnosticCode",
        severity="warning",
        summary=(
            "JIT compiled the function down to Target IR but the "
            "current backend ships only an artifact (no runtime "
            "dispatch path)."
        ),
        fix_hint=(
            "Inspect via `tessera.compiler.dry_run(fn)` or "
            "`JitFn.runtime_artifact()` to confirm artifact-only "
            "status is expected for the target."
        ),
        spec=None,
        sprint="P0-2",
        language="python",
        status="implemented",
    ),
    DiagnosticCode(
        code="JIT_APPLE_GPU_TRACE_DEFERRED",
        pass_origin="tessera.compiler.JitDiagnosticCode",
        severity="warning",
        summary=(
            "AST Graph IR emission failed for an apple_gpu function, but "
            "decoration did not hard-fail — the Phase-F tracer executes "
            "the function by running it (it never reads the AST graph_ir), "
            "so the body still decorates and runs via the tracer at call time."
        ),
        fix_hint=(
            "Informational; no action required. Use tessera.control.cond / "
            "while_loop for data-dependent control flow if a raw `if`/`while` "
            "on a traced value raises at call time."
        ),
        spec=None,
        sprint="phase-f-followon",
        language="python",
        status="implemented",
    ),
    DiagnosticCode(
        code="JIT_APPLE_GPU_AUTO_BATCH",
        pass_origin="tessera.compiler.JitDiagnosticCode",
        # Registry severity is warning per the registry's accepted set; the
        # actual JitDiagnostic is emitted at "info" (matching JIT_SOURCE_PROVIDED).
        severity="warning",
        summary=(
            "The apple_gpu one-command-buffer route (auto_batch) is active for "
            "this function — either requested explicitly or auto-detected as a "
            "recognized decode chain. The tracer runs the body directly, so the "
            "AST Graph IR emission it would otherwise do is skipped as unused."
        ),
        fix_hint=(
            "Informational; no action required. Pass @jit(auto_batch=False) to "
            "force the per-op eager path, or auto_batch=True to force the route "
            "on for a body auto-detection did not recognize."
        ),
        spec=None,
        sprint="p3-auto-batch-polish",
        language="python",
        status="implemented",
    ),

    # ── LayoutLegalityPass (V2 + V4a) ────────────────────────────────────
    DiagnosticCode(
        code="LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH",
        pass_origin="LayoutLegalityPass",
        severity="error",
        summary=(
            "A `tessera.matmul` operand's producer carries a `tessera.layout` "
            "attribute outside matmul's accept-set {row_major, col_major}, "
            "and no intervening cast converts it."
        ),
        fix_hint=(
            "Insert a `tessera.cast` that converts the producer's layout to "
            "either row_major or col_major before the matmul."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V4a",
    ),
    DiagnosticCode(
        code="LAYOUT_LEGALITY_UNKNOWN_LAYOUT",
        pass_origin="LayoutLegalityPass",
        severity="error",
        summary=(
            "A `tessera.cast` op carries a `tessera.layout` string attribute "
            "that is not in the canonical 8-name accept-set "
            "{row_major, col_major, nhwc, nchw, bhsd, tile, bsr, packed}."
        ),
        fix_hint=(
            "Use one of the canonical layout names listed in "
            "SHAPE_SYSTEM.md §2.1, or update the accept-set if a new "
            "canonical layout is needed."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §2.1",
        sprint="V2",
    ),
    DiagnosticCode(
        code="LAYOUT_LEGALITY_SCALE_WITHOUT_LAYOUT",
        pass_origin="LayoutLegalityPass",
        severity="error",
        summary=(
            "A `tessera.grouped_gemm` / `tessera.moe_swiglu_block` carries a "
            "low-precision scale *operand* but no `scale_layout` attribute — an "
            "untyped scale tensor has no compiler-visible layout contract."
        ),
        fix_hint=(
            "Declare a `scale_layout` attribute (granularity / block / packing / "
            "transposed) describing the scale operand's packed layout, or drop "
            "the scale operand to use the unscaled form."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="DeepGEMM-keystone",
    ),

    # ── IRContractLegalityPass (dtype / aliasing / buffer-binding) ───────
    DiagnosticCode(
        code="DTYPE_LEGALITY_TF32_AS_STORAGE",
        pass_origin="IRContractLegalityPass",
        severity="error",
        summary=(
            "`numeric_policy.storage = \"tf32\"` is illegal — TF32 is a "
            "`math_mode` on fp32 storage, not a storage dtype."
        ),
        fix_hint=(
            "Set `numeric_policy.storage = \"fp32\"` and express TF32 via "
            "`numeric_policy.math_mode = \"tf32\"`."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="IRContractLegality",
    ),
    DiagnosticCode(
        code="DTYPE_LEGALITY_UNKNOWN_STORAGE",
        pass_origin="IRContractLegalityPass",
        severity="error",
        summary=(
            "`numeric_policy.storage` names a dtype outside the canonical + "
            "known-gated storage set."
        ),
        fix_hint=(
            "Use a canonical dtype name from "
            "docs/reference/tessera_tensor_attributes.md, or declare the "
            "planned-gated dtype in the known-gated storage set."
        ),
        spec="docs/reference/tessera_tensor_attributes.md",
        sprint="IRContractLegality",
    ),
    DiagnosticCode(
        code="DTYPE_LEGALITY_LOWP_WITHOUT_WIDE_ACCUM",
        pass_origin="IRContractLegalityPass",
        severity="error",
        summary=(
            "A low-precision storage (fp8*/fp6*/fp4*/nvfp4/int4/int8) must "
            "declare a wider accumulator (fp32/fp16/bf16/int32); storage and "
            "accumulator are distinct contracts (Decision #15a)."
        ),
        fix_hint=(
            "Declare `numeric_policy.accum` as a wider dtype than the "
            "low-precision storage instead of relying on a single fused dtype."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="IRContractLegality",
    ),
    DiagnosticCode(
        code="ALIAS_LEGALITY_MISSING_ALIASES",
        pass_origin="IRContractLegalityPass",
        severity="error",
        summary=(
            "An op marked `tessera.inplace = true` must declare "
            "`tessera.aliases` (the operand index its result aliases) — an "
            "undeclared in-place mutation has no aliasing contract the "
            "scheduler can honor."
        ),
        fix_hint=(
            "Add a `tessera.aliases` integer attribute naming the operand "
            "index the in-place result aliases, or drop `tessera.inplace`."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="IRContractLegality",
    ),
    DiagnosticCode(
        code="ALIAS_LEGALITY_OPERAND_OOB",
        pass_origin="IRContractLegalityPass",
        severity="error",
        summary="`tessera.aliases` indexes past the operand list.",
        fix_hint=(
            "Set `tessera.aliases` to a valid operand index in "
            "[0, num_operands)."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="IRContractLegality",
    ),
    DiagnosticCode(
        code="BUFFER_BINDING_UNKNOWN_ROLE",
        pass_origin="IRContractLegalityPass",
        severity="error",
        summary=(
            "`tessera.buffer_role` is outside the accept-set "
            "{input, output, scratch, accumulator, weight}."
        ),
        fix_hint=(
            "Use one of the canonical buffer roles "
            "{input, output, scratch, accumulator, weight}."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="IRContractLegality",
    ),
    DiagnosticCode(
        code="BUFFER_BINDING_CONFLICT",
        pass_origin="IRContractLegalityPass",
        severity="error",
        summary=(
            "Two ops bind the same `tessera.binding` id to different roles — a "
            "buffer can't be both (e.g.) an input and a scratch in one program."
        ),
        fix_hint=(
            "Give the conflicting buffers distinct `tessera.binding` ids, or "
            "reconcile their `tessera.buffer_role` to a single role."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="IRContractLegality",
    ),

    # ── CF0 — control-flow target guard ──────────────────────────────────
    DiagnosticCode(
        code="CONTROL_FLOW_UNSUPPORTED_ON_TARGET",
        pass_origin="ControlFlowTargetGuard",
        severity="error",
        summary=(
            "A tessera.control_{for,if,while,scan} op reached a backend with "
            "no lowering for this control-flow form/envelope. Some targets "
            "support only narrow executable subsets; unsupported forms must "
            "fail before backend codegen."
        ),
        fix_hint=(
            "Use a target-supported control-flow envelope (for example the "
            "CF4 ROCm elementwise rank-1 control_for/if/while kernels), or "
            "hoist this loop/branch to the host. See "
            "docs/spec/CONTROL_FLOW_CONTRACT.md §5."
        ),
        spec="docs/spec/CONTROL_FLOW_CONTRACT.md §5",
        sprint="CF0",
    ),

    # ── Queue dialect verifiers (V8) ─────────────────────────────────────
    DiagnosticCode(
        code="QUEUE_CREATE_OPERAND_COUNT",
        pass_origin="Queue.CreateOp::verify",
        severity="error",
        summary=(
            "`tessera.queue.create` must have zero operands; future TD "
            "revisions that accidentally add one are caught at the IR layer."
        ),
        fix_hint=(
            "Remove the extra operand or align Queue.td with the canonical "
            "zero-operand contract."
        ),
        spec=None,
        sprint="V8",
    ),
    DiagnosticCode(
        code="QUEUE_POP_QUEUE_PROVENANCE",
        pass_origin="Queue.PopOp::verify",
        severity="error",
        summary=(
            "The queue handle operand of a `tessera.queue.pop` is not "
            "defined by a `tessera.queue.create` — data-flow malformed."
        ),
        fix_hint=(
            "Ensure the queue handle traces back to a `tessera.queue.create` "
            "op (not a function argument, not a block argument)."
        ),
        spec=None,
        sprint="V8",
    ),
    DiagnosticCode(
        code="QUEUE_POP_TILE_TYPE",
        pass_origin="Queue.PopOp::verify",
        severity="error",
        summary=(
            "The result of a `tessera.queue.pop` is neither a ranked tensor "
            "nor a memref — the TD's `AnyType` was too permissive."
        ),
        fix_hint=(
            "Constrain the result type to a ranked tensor or memref shape."
        ),
        spec=None,
        sprint="V8",
    ),
    DiagnosticCode(
        code="QUEUE_POP_TOKEN_PROVENANCE",
        pass_origin="Queue.PopOp::verify",
        severity="error",
        summary=(
            "The dependency token operand of a `tessera.queue.pop` is not "
            "defined by a `tessera.queue.push` — the token must come from a "
            "matching push."
        ),
        fix_hint=(
            "Wire the dep token from a preceding `tessera.queue.push`; "
            "function-argument tokens are not legal in FA-4 IR."
        ),
        spec=None,
        sprint="V8",
    ),
    DiagnosticCode(
        code="QUEUE_PUSH_QUEUE_PROVENANCE",
        pass_origin="Queue.PushOp::verify",
        severity="error",
        summary=(
            "The queue handle operand of a `tessera.queue.push` is not "
            "defined by a `tessera.queue.create`."
        ),
        fix_hint=(
            "Trace the queue handle back to a `tessera.queue.create` op."
        ),
        spec=None,
        sprint="V8",
    ),
    DiagnosticCode(
        code="QUEUE_PUSH_TILE_TYPE",
        pass_origin="Queue.PushOp::verify",
        severity="error",
        summary=(
            "The tile operand of `tessera.queue.push` is neither a ranked "
            "tensor nor a memref."
        ),
        fix_hint=(
            "Pass a tile-shaped value (ranked tensor / memref) — scalars "
            "and opaque tokens are not legal queue payloads."
        ),
        spec=None,
        sprint="V8",
    ),

    # ── SymbolicDimEqualityPass family (V5 + V2-flow + V3a + V3b + V3c) ──
    DiagnosticCode(
        code="SYMDIM_BINDING_VIOLATION",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "A function-level `tessera.dim_bindings` equation (e.g., "
            "`D = H * Dh + K`) is contradicted by the function's "
            "`tessera.dim_sizes` (the concrete sizes evaluate to a "
            "different value than the LHS claims)."
        ),
        fix_hint=(
            "Either correct the concrete sizes in `tessera.dim_sizes` to "
            "match the binding, or update the binding equation to reflect "
            "the actual shape relationship."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V5",
    ),
    DiagnosticCode(
        code="SYMDIM_CALL_ARG_MISMATCH",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "A `func.call` site passes operands whose propagated dim-names "
            "disagree with the callee's declared `tessera.arg_dim_names`."
        ),
        fix_hint=(
            "Update the caller to pass values with matching dim-names, or "
            "update the callee's `tessera.arg_dim_names` declaration."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V3b",
    ),
    DiagnosticCode(
        code="SYMDIM_FLOW_INCONSISTENCY",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "SSA-value flow-propagated dim-names disagree with an explicit "
            "per-op `tessera.dim_names_in` / `tessera.dim_names_out` / "
            "`tessera.dim_names_lhs` / `tessera.dim_names_rhs` annotation."
        ),
        fix_hint=(
            "Either remove the explicit annotation (let propagation infer) "
            "or correct it to match the propagated names."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V2-flow",
    ),
    DiagnosticCode(
        code="SYMDIM_IF_BRANCH_MISMATCH",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "An `scf.if` op's then-branch and else-branch yield values "
            "with different propagated dim-names for the same result "
            "position."
        ),
        fix_hint=(
            "Make both branches yield values that share the same dim-name "
            "structure (transpose / reshape in the branch as needed)."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V3c",
    ),
    DiagnosticCode(
        code="SYMDIM_LOOP_YIELD_MISMATCH",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "An `scf.for` op's `scf.yield` operand carries dim-names that "
            "differ from the corresponding iter_arg's dim-names — the loop "
            "is not name-invariant."
        ),
        fix_hint=(
            "Restructure the body so the yielded value preserves the "
            "iter_arg's dim-name ordering (no transpose, or undo the "
            "transpose before yielding)."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V3c",
    ),
    DiagnosticCode(
        code="SYMDIM_MATMUL_CONTRACT_VIOLATION",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "A `tessera.matmul` op declares `tessera.dim_names_lhs` and "
            "`tessera.dim_names_rhs` whose contracting symbols disagree "
            "(lhs.back() != rhs.front())."
        ),
        fix_hint=(
            "Rename one side's contracting dim so both ends agree on the "
            "K symbol, or fix the per-op annotation."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V5",
    ),
    DiagnosticCode(
        code="SYMDIM_RESHAPE_VIOLATION",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "A `tessera.reshape` op's `tessera.dim_names_in` and "
            "`tessera.dim_names_out` resolve to different element counts "
            "given the function's `tessera.dim_sizes` + bindings — the "
            "reshape cannot hold."
        ),
        fix_hint=(
            "Fix the dim_names list so the product of resolved sizes "
            "matches on both sides, or correct dim_sizes if the symbolic "
            "model is wrong."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V5",
    ),
    DiagnosticCode(
        code="SYMDIM_TRANSPOSE_VIOLATION",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "A `tessera.transpose` op's `tessera.dim_names_in` and "
            "`tessera.dim_names_out` are not a permutation of each other."
        ),
        fix_hint=(
            "Adjust the output names so they're a reordering of the input "
            "names (same multiset)."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="V5",
    ),

    # ── TSOL spec contracts (TSOL-2, 2026-05-22) ─────────────────────────
    # The TS_ERR_* family is named in the TSOL spec at
    # `docs/operations/Tessera_Standard_Operations.md` §"Error Handling".
    # status="spec_contract" — these codes are listed for spec
    # traceability today; the Python implementation currently raises
    # `TesseraShapeError` / `TesseraTargetError` / etc. with the E_*
    # enum values from above.  When the implementation grows TS_ERR_*
    # tagging (a future sprint), flip status to "implemented" and the
    # drift gate will require a Python emission site.
    DiagnosticCode(
        code="TS_ERR_BACKEND_FAILURE",
        pass_origin="TSOL spec",
        severity="error",
        summary=(
            "Wrapped backend failure (CUDA / ROCm / NCCL / RCCL / "
            "NVSHMEM / Metal / x86 runtime returned an error)."
        ),
        fix_hint=(
            "Inspect the wrapped backend error message in the exception "
            "chain; check toolchain pin (CUDA 13.3 / ROCm 7.2.4) "
            "compatibility with the installed driver."
        ),
        spec="docs/operations/Tessera_Standard_Operations.md §Error Handling",
        sprint="TSOL spec",
        language="python",
        status="spec_contract",
    ),
    DiagnosticCode(
        code="TS_ERR_INVALID_ARG",
        pass_origin="TSOL spec",
        severity="error",
        summary=(
            "An operator received an invalid value, option, or "
            "malformed metadata (e.g., negative axis on a "
            "single-axis op, bad reduction op string)."
        ),
        fix_hint=(
            "Check the op's signature in "
            "`docs/operations/Tessera_Standard_Operations.md` "
            "against the call-site arguments."
        ),
        spec="docs/operations/Tessera_Standard_Operations.md §Error Handling",
        sprint="TSOL spec",
        language="python",
        status="spec_contract",
    ),
    DiagnosticCode(
        code="TS_ERR_NONDETERMINISM",
        pass_origin="TSOL spec",
        severity="error",
        summary=(
            "Deterministic mode was requested but the chosen backend "
            "cannot honor it (e.g., NCCL ring schedule isn't "
            "deterministic on this build, or a fused kernel uses "
            "atomic accumulation)."
        ),
        fix_hint=(
            "Disable the offending fast-path via numeric policy or "
            "switch to a backend with deterministic guarantees; see "
            "`docs/operations/Tessera_Standard_Operations.md` "
            "§Determinism Contract."
        ),
        spec="docs/operations/Tessera_Standard_Operations.md §Determinism Contract",
        sprint="TSOL spec",
        language="python",
        status="spec_contract",
    ),
    DiagnosticCode(
        code="TS_ERR_OOM",
        pass_origin="TSOL spec",
        severity="error",
        summary=(
            "Allocation failed (host or device).  Includes "
            "command-buffer / scratch-buffer / KV-cache exhaustion "
            "as well as raw `cudaMalloc` / `hipMalloc` failures."
        ),
        fix_hint=(
            "Shrink batch / sequence dimensions, increase memory "
            "budget, or check the buffer-pool capacity via "
            "`tessera.runtime.memory_stats()`."
        ),
        spec="docs/operations/Tessera_Standard_Operations.md §Error Handling",
        sprint="TSOL spec",
        language="python",
        status="spec_contract",
    ),
    DiagnosticCode(
        code="TS_ERR_SHAPE_MISMATCH",
        pass_origin="TSOL spec",
        severity="error",
        summary=(
            "TSOL spec-level shape contract failed.  Maps to today's "
            "Python `TesseraShapeError` / `E_SHAPE_MISMATCH` until the "
            "spec contract codes are wired into raises directly."
        ),
        fix_hint=(
            "Same as `E_SHAPE_MISMATCH`: inspect the JIT signature "
            "against the call-site shapes."
        ),
        spec="docs/operations/Tessera_Standard_Operations.md §Error Handling",
        sprint="TSOL spec",
        language="python",
        status="spec_contract",
    ),
    DiagnosticCode(
        code="TS_ERR_UNSUPPORTED_DTYPE",
        pass_origin="TSOL spec",
        severity="error",
        summary=(
            "The backend or operator can't support the requested "
            "storage dtype / numeric policy (e.g., FP4 on a pre-"
            "Blackwell NVIDIA GPU)."
        ),
        fix_hint=(
            "Consult the per-target dtype matrix in "
            "`docs/audit/standalone_primitive_coverage.md`; downcast "
            "via `tessera.dtype.canonicalize` if a fallback is "
            "acceptable."
        ),
        spec="docs/operations/Tessera_Standard_Operations.md §Error Handling",
        sprint="TSOL spec",
        language="python",
        status="spec_contract",
    ),

    # ───────────────────────────────────────────────────────────────────────
    # TIRx review (C1–C6, 2026-06-23) — Tile-IR layout/barrier/pipeline
    # verifiers + the C2/C3/C6 legality-pass gates. See COMPILER_AUDIT items
    # C1–C6.
    # ───────────────────────────────────────────────────────────────────────
    # C1 — #tile.layout / #tile.swizzle attribute verifier (TileLayoutAttr).
    DiagnosticCode(
        code="TILE_LAYOUT_RANK_MISMATCH", pass_origin="TileLayoutAttr",
        severity="error",
        summary="A #tile.layout shard/replica's extents, strides, and axes arrays differ in length.",
        fix_hint="Give the shard (and replica) equal-length [extents]:[strides] on [axes].",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C1", sprint="C1 (TIRx)",
    ),
    DiagnosticCode(
        code="TILE_LAYOUT_NONPOSITIVE_EXTENT", pass_origin="TileLayoutAttr",
        severity="error",
        summary="A #tile.layout shard/replica extent is <= 0.",
        fix_hint="Use positive extents; a dynamic tile carries no layout (buffer identity only).",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C1", sprint="C1 (TIRx)",
    ),
    DiagnosticCode(
        code="TILE_LAYOUT_UNKNOWN_AXIS", pass_origin="TileLayoutAttr",
        severity="error",
        summary="A #tile.layout axis is not a known hardware axis.",
        fix_hint="Use a known hardware axis — NVIDIA: m/tlane/tcol/laneid/warpid/reg/...; AMD: lds (shared) / waveid; plus bx/by/bz, cbx/cby/cbz, gpuid_x/y.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C1", sprint="C1 (TIRx)",
    ),
    DiagnosticCode(
        code="TILE_LAYOUT_BAD_SWIZZLE", pass_origin="TileLayoutAttr",
        severity="error",
        summary="The swizzle clause of a #tile.layout is not a #tile.swizzle attribute.",
        fix_hint="Use `swizzle = #tile.swizzle<per_element=…, len=…, atom=…>`.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C1", sprint="C1 (TIRx)",
    ),
    # C3 — #tile.barrier / #tile.pipeline_state attribute verifiers.
    DiagnosticCode(
        code="TILE_BARRIER_UNKNOWN_KIND", pass_origin="TileBarrierAttr",
        severity="error",
        summary="A #tile.barrier kind is not one of {tma, tcgen05, mbarrier}.",
        fix_hint="Pick the completion semantics — NVIDIA: tma (byte-count) / tcgen05 (MMA) / mbarrier (thread-arrival); AMD: s_barrier (workgroup arrival) / waitcnt (async counter).",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C3", sprint="C3 (TIRx)",
    ),
    DiagnosticCode(
        code="TILE_BARRIER_NEGATIVE_EXPECT", pass_origin="TileBarrierAttr",
        severity="error",
        summary="A #tile.barrier expect (arrival / byte count) is negative.",
        fix_hint="Use expect >= 0.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C3", sprint="C3 (TIRx)",
    ),
    DiagnosticCode(
        code="TILE_PIPELINE_BAD_DEPTH", pass_origin="TilePipelineStateAttr",
        severity="error",
        summary="A #tile.pipeline_state depth is < 1.",
        fix_hint="Ring depth must be >= 1.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C3", sprint="C3 (TIRx)",
    ),
    DiagnosticCode(
        code="TILE_PIPELINE_STAGE_OOB", pass_origin="TilePipelineStateAttr",
        severity="error",
        summary="A #tile.pipeline_state stage is not in [0, depth).",
        fix_hint="Keep stage within the ring: 0 <= stage < depth.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C3", sprint="C3 (TIRx)",
    ),
    DiagnosticCode(
        code="TILE_PIPELINE_BAD_PHASE", pass_origin="TilePipelineStateAttr",
        severity="error",
        summary="A #tile.pipeline_state phase parity bit is not 0 or 1.",
        fix_hint="phase is the parity bit — 0 or 1.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C3", sprint="C3 (TIRx)",
    ),
    DiagnosticCode(
        code="TILE_PIPELINE_BAD_ROLE", pass_origin="TilePipelineStateAttr",
        severity="error",
        summary="A #tile.pipeline_state role is not producer or consumer.",
        fix_hint="role is producer | consumer.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C3", sprint="C3 (TIRx)",
    ),
    # Typed #tile.buffer_ref contract verifier (TileBufferRefAttr) — replaces
    # the tile.buffer/tile.access string markers with a typed space + access.
    DiagnosticCode(
        code="TILE_BUFFER_REF_EMPTY_NAME", pass_origin="TileBufferRefAttr",
        severity="error",
        summary="A #tile.buffer_ref has an empty name.",
        fix_hint="Name the buffer the reference points at.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C2", sprint="C2 (TIRx)",
    ),
    DiagnosticCode(
        code="TILE_BUFFER_REF_BAD_SPACE", pass_origin="TileBufferRefAttr",
        severity="error",
        summary="A #tile.buffer_ref space is not one of {smem, lds, tmem, gmem, reg}.",
        fix_hint="Use a known memory space: smem (NVIDIA shared) / lds (AMD) / tmem / gmem / reg.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C2", sprint="C2 (TIRx)",
    ),
    DiagnosticCode(
        code="TILE_BUFFER_REF_BAD_ACCESS", pass_origin="TileBufferRefAttr",
        severity="error",
        summary="A #tile.buffer_ref access is not one of {read, write, free}.",
        fix_hint="Use a known access mode: read / write / free.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C2", sprint="C2 (TIRx)",
    ),
    # C5 — #tile.pipeline_depths attribute verifier.
    DiagnosticCode(
        code="TILE_PIPELINE_DEPTHS_NONPOSITIVE", pass_origin="TilePipelineDepthsAttr",
        severity="error",
        summary="A #tile.pipeline_depths ring depth (q/kv/tmem) is < 1.",
        fix_hint="Each independent ring depth must be >= 1 (book defaults q=2, kv=3, tmem=2).",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C5", sprint="C5 (TIRx)",
    ),
    # C2 — TileBarrierReuseLegalityPass.
    DiagnosticCode(
        code="TILE_BARRIER_REUSE_MISSING_BARRIER", pass_origin="TileBarrierReuseLegality",
        severity="error",
        summary="A buffer is written over an overlapping storage footprint with no intervening barrier — a reuse race.",
        fix_hint="Insert an mbarrier / wait_async between the two writes to the reused region.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C2", sprint="C2 (TIRx)",
    ),
    # C3 — TilePipelineLegalityPass.
    DiagnosticCode(
        code="TILE_PIPELINE_PHASE_ASYMMETRY", pass_origin="TilePipelineLegality",
        severity="error",
        summary="A pipeline's initial producer is not phase=1 / consumer not phase=0 — the off-by-one ring deadlock.",
        fix_hint="Initialize the producer ring at phase=1 and the consumer at phase=0.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C3", sprint="C3 (TIRx)",
    ),
    DiagnosticCode(
        code="TILE_PIPELINE_BARRIER_KIND_MISMATCH", pass_origin="TilePipelineLegality",
        severity="error",
        summary="One tile.barrier_id is used with two different #tile.barrier kinds.",
        fix_hint="Keep one completion semantics (kind) per barrier id.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C3", sprint="C3 (TIRx)",
    ),
    # C6 — WarpSpecLegalityPass (the 7 appendix invariants).
    DiagnosticCode(
        code="WARPSPEC_INIT_UNDER_GUARD", pass_origin="WarpSpecLegality",
        severity="error",
        summary="A barrier init runs inside a warp-role-guarded region instead of CTA top level.",
        fix_hint="Hoist mbarrier init to CTA scope (thread 0), outside any warp-role region.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C6", sprint="C6 (TIRx)",
    ),
    DiagnosticCode(
        code="WARPSPEC_COLLECTIVE_IN_DIVERGENT_BRANCH", pass_origin="WarpSpecLegality",
        severity="error",
        summary="A collective (cta_sync / cluster_sync / next_tile) sits inside a warp-role-guarded region.",
        fix_hint="Move the collective to a point all warps reach (outside the warp-role region).",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C6", sprint="C6 (TIRx)",
    ),
    DiagnosticCode(
        code="WARPSPEC_LOOP_COUNT_DISAGREE", pass_origin="WarpSpecLegality",
        severity="error",
        summary="Producer/consumer loops on one tile.pipeline declare different tile.trip_count.",
        fix_hint="Give the producer (TMA) and consumer (MMA) loops the same trip count.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C6", sprint="C6 (TIRx)",
    ),
    DiagnosticCode(
        code="WARPSPEC_MISSING_VISIBILITY_FENCE", pass_origin="WarpSpecLegality",
        severity="error",
        summary="A TMA store has no prior visibility fence (fence.proxy_async / commit_group) in its block.",
        fix_hint="Emit a fence.proxy_async before the TMA store so the async engine sees fresh shared memory.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C6", sprint="C6 (TIRx)",
    ),
    DiagnosticCode(
        code="WARPSPEC_MMA_NOT_TOKEN_SYNCED", pass_origin="WarpSpecLegality",
        severity="error",
        summary="A consumer tile.mma reads a producer's async-staged tile but has no !tile.async_token completion edge to it — the matrix op is not gated on copy completion.",
        fix_hint="Thread the producer copy's !tile.async_token into the mma (WarpSpecialization auto-mints it from the mma's data operands); this is the SSA ordering half of the arrival==init check.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §op-layer convergence", sprint="Phase C-NV",
    ),
    DiagnosticCode(
        code="WARPSPEC_MMA_TOKEN_NOT_RETIRED", pass_origin="WarpSpecLegality",
        severity="error",
        summary="A tile.mma holds a tile.async_copy/tma.copy_async completion token that no prior tile.wait_async retired — the copy is still in flight when the matrix op runs (held-but-unwaited race).",
        fix_hint="Add a tile.wait_async on the token before the mma. Converges with the ROCm legality, which also requires retirement, not just token presence.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §op-layer convergence", sprint="Phase C-NV",
    ),
    DiagnosticCode(
        code="ASYNC_COPY_TOKEN_NO_CP_ASYNC_PATH", pass_origin="AsyncCopyLowering",
        severity="error",
        summary="A tile.async_copy carries a !tile.async_token but the SM<90 cp.async fallback has no SSA completion-token path.",
        fix_hint="Thread async tokens only on the SM>=90 TMA path; drop the !tile.async_token result before the cp.async fallback.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §op-layer convergence", sprint="Phase C-NV",
    ),
    DiagnosticCode(
        code="WARPSPEC_ARRIVAL_COUNT_MISMATCH", pass_origin="WarpSpecLegality",
        severity="error",
        summary="#tile.barrier sites on one tile.barrier_id disagree on expect (arrival count != init count).",
        fix_hint="Match the arrive count (copy_async expect_tx) to the init count (setup_descriptor).",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C6", sprint="C6 (TIRx)",
    ),
    DiagnosticCode(
        code="WARPSPEC_USE_AFTER_FREE", pass_origin="WarpSpecLegality",
        severity="error",
        summary="A buffer free has no prior cta_sync in its block — a warp may still be reading it during writeback.",
        fix_hint="Emit a cta_sync before deallocating the buffer (the writeback-dealloc epilogue).",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C6", sprint="C6 (TIRx)",
    ),

    # C4 reconciliation (2026-06-23) — the ROCm WMMA kernel generator consumes
    # the storage-pack descriptor; its factor must match the WMMA int pack mode.
    DiagnosticCode(
        code="DTYPE_PACK_FACTOR_MISMATCH", pass_origin="GenerateWMMAGemmKernel",
        severity="error",
        summary="A tessera.storage_pack factor disagrees with the ROCm WMMA integer pack mode (int8->1, int4->2) for the dtype.",
        fix_hint="Make the storage-pack factor (container_bits/storage_bits) match the WMMA ABI pack mode; they describe the same packing.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C4", sprint="C4 (TIRx)",
    ),
    # C4 part 1 (2026-06-23) — the storage-pack consumer (StoragePackConsume).
    DiagnosticCode(
        code="DTYPE_PACK_BAD_WIDTHS", pass_origin="StoragePackConsume",
        severity="error",
        summary="A storage_packed op's logical storage cannot pack into its container (unknown dtype, or storage wider than the container).",
        fix_hint="Mark sub-byte storage (fp4/nvfp4/fp6/int4) with a wider byte container (int8); storage bits must be <= container bits.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C4", sprint="C4 (TIRx)",
    ),

    # ROCm shared Tile-IR convergence (2026-06-23) — AMD consumes the shared
    # Tile contract but keeps LDS / waitcnt legality target-specific.
    DiagnosticCode(
        code="ROCM_LOWERING_LAYOUT_NOT_LDS", pass_origin="LowerTileToROCMPass",
        severity="error",
        summary="ROCm lowering saw a #tile.layout on tile.async_copy that does not place storage on the lds axis.",
        fix_hint="Use #tile.layout with an lds shard axis for ROCm global-to-LDS movement, or omit layout when unknown.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §ROCm Tile-IR convergence", sprint="ROCm Tile-IR convergence",
    ),
    DiagnosticCode(
        code="ROCM_LOWERING_NON_LDS_BUFFER", pass_origin="LowerTileToROCMPass",
        severity="error",
        summary="ROCm lowering saw tile.async_copy with a #tile.buffer_ref that is not space=lds.",
        fix_hint="Use #tile.buffer_ref<space = \"lds\", access = \"write\"> for ROCm async global-to-LDS staging.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §ROCm Tile-IR convergence", sprint="ROCm Tile-IR convergence",
    ),
    DiagnosticCode(
        code="ROCM_LOWERING_NON_WRITE_BUFFER", pass_origin="LowerTileToROCMPass",
        severity="error",
        summary="ROCm lowering saw tile.async_copy with an LDS buffer_ref whose access is not write.",
        fix_hint="Mark the destination staging reference access = \"write\".",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §ROCm Tile-IR convergence", sprint="ROCm Tile-IR convergence",
    ),
    DiagnosticCode(
        code="ROCM_LOWERING_UNCONSUMED_STORAGE_PACK", pass_origin="LowerTileToROCMPass",
        severity="error",
        summary="Packed low-precision storage reached ROCm lowering without a backend storage-pack consumer descriptor.",
        fix_hint="Run tessera-storage-pack-consume, or add an explicit ROCm packed-load/store consumer before lower-tile-to-rocm.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C4", sprint="ROCm Tile-IR convergence",
    ),
    DiagnosticCode(
        code="ROCM_WAVE_LDS_MISSING_WAITCNT", pass_origin="ROCMWaveLdsLegalityPass",
        severity="error",
        summary="A tile.mma reads from an outstanding global-to-LDS async copy without an intervening tile.wait_async / waitcnt.",
        fix_hint="Insert tile.wait_async so ROCm lowering emits tessera_rocm.wait counter=vmcnt before the LDS-dependent matrix op.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §ROCm Tile-IR convergence", sprint="ROCm Tile-IR convergence",
    ),
    DiagnosticCode(
        code="ROCM_WAVE_LDS_UNSUPPORTED_NV_CONSTRUCT", pass_origin="ROCMWaveLdsLegalityPass",
        severity="error",
        summary="An NVIDIA-only Tile op (tile.mbarrier.* / tile.tma.* / tile.tmem.*) appears on the ROCm path.",
        fix_hint="Use LDS / waitcnt / s_barrier contracts on ROCm; NVIDIA TMA/TMEM/mbarrier constructs have no AMD lowering.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §ROCm Tile-IR convergence", sprint="ROCm Tile-IR convergence",
    ),
    DiagnosticCode(
        code="ROCM_WAVE_LDS_OVERLAPPING_WRITE", pass_origin="ROCMWaveLdsLegalityPass",
        severity="error",
        summary="An LDS buffer is written over an overlapping layout region with no intervening waitcnt or barrier.",
        fix_hint="Use a different LDS stage/buffer or insert the necessary wait/barrier before reusing the region.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §ROCm Tile-IR convergence", sprint="ROCm Tile-IR convergence",
    ),
    DiagnosticCode(
        code="ROCM_WAVE_LDS_UNSUPPORTED_BARRIER_KIND", pass_origin="ROCMWaveLdsLegalityPass",
        severity="error",
        summary="ROCm Tile-IR legality saw NVIDIA-only TMA/TCGen05/mbarrier completion semantics.",
        fix_hint="Use AMD waitcnt for counter waits or s_barrier for true workgroup synchronization.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §ROCm Tile-IR convergence", sprint="ROCm Tile-IR convergence",
    ),
    DiagnosticCode(
        code="ROCM_WAVE_LDS_UNSUPPORTED_TMEM", pass_origin="ROCMWaveLdsLegalityPass",
        severity="error",
        summary="ROCm Tile-IR legality saw TMEM-only operations or buffer spaces.",
        fix_hint="Use ROCm LDS/register contracts; TMEM is not available on the ROCm path.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §ROCm Tile-IR convergence", sprint="ROCm Tile-IR convergence",
    ),

    # ───────────────────────────────────────────────────────────────────────
    # Pipeline-parallel layer (2026-06-23) — the 1F1B schedule proof
    # (PipelineScheduleLegality), paired with the real PipelineStagePartition.
    # ───────────────────────────────────────────────────────────────────────
    DiagnosticCode(
        code="PP_MICRO_BATCHES_TOO_FEW", pass_origin="PipelineScheduleLegality",
        severity="error",
        summary="Fewer micro-batches than the 1F1B pipeline needs to fill (num_stages, or 2*num_stages interleaved; Decision #17).",
        fix_hint="Raise num_micro_batches to >= num_stages (>= 2*num_stages for interleaved).",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §pipeline", sprint="Pipeline-PP",
    ),
    DiagnosticCode(
        code="PP_EMPTY_STAGE", pass_origin="PipelineScheduleLegality",
        severity="error",
        summary="A declared pipeline stage owns no op — the partition produced fewer real stages than declared.",
        fix_hint="Reduce num_stages or give every stage work; an empty stage holes the send/recv chain.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §pipeline", sprint="Pipeline-PP",
    ),
    DiagnosticCode(
        code="PP_SEND_WITHOUT_RECV", pass_origin="PipelineScheduleLegality",
        severity="error",
        summary="A pipeline send from stage k has no matching recv at stage k+1 — a dropped activation / deadlock.",
        fix_hint="Ensure the forward-adjacent send/recv chain is complete (one recv at k+1 per send at k).",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §pipeline", sprint="Pipeline-PP",
    ),
    DiagnosticCode(
        code="PP_RECV_WITHOUT_SEND", pass_origin="PipelineScheduleLegality",
        severity="error",
        summary="A pipeline recv at stage j has no matching send from stage j-1 — an unpaired / stage-skipping comm.",
        fix_hint="Ensure every recv at j is fed by a send from j-1 (forward-adjacent chain only).",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §pipeline", sprint="Pipeline-PP",
    ),
    DiagnosticCode(
        code="PP_UNROUTED_CROSS_STAGE_VALUE", pass_origin="PipelineScheduleLegality",
        severity="error",
        summary="A value flows directly from one stage to another without a send/recv — the boundary rewrite missed it (e.g. a stage-skipping edge).",
        fix_hint="Route every cross-stage activation through send/recv; avoid stage-skipping SSA edges (or partition them adjacently).",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §pipeline", sprint="Pipeline-PP",
    ),
)


# ─────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────


def all_codes() -> tuple[str, ...]:
    """Return all registered code names, sorted."""
    return tuple(sorted(c.code for c in REGISTERED_CODES))


def code_lookup(code: str) -> DiagnosticCode | None:
    """Look up a single code by name. Returns None if not registered."""
    for entry in REGISTERED_CODES:
        if entry.code == code:
            return entry
    return None


def codes_by_pass(pass_origin: str) -> tuple[DiagnosticCode, ...]:
    """Return all codes emitted by a given pass / verifier."""
    return tuple(c for c in REGISTERED_CODES if c.pass_origin == pass_origin)


def codes_by_sprint(sprint: str) -> tuple[DiagnosticCode, ...]:
    """Return all codes introduced by a given sprint label."""
    return tuple(c for c in REGISTERED_CODES if c.sprint == sprint)


def codes_by_language(language: str) -> tuple[DiagnosticCode, ...]:
    """TSOL-2: return all codes for a given language ("mlir" or "python")."""
    return tuple(c for c in REGISTERED_CODES if c.language == language)


def codes_by_status(status: str) -> tuple[DiagnosticCode, ...]:
    """TSOL-2: return all codes by implementation status
    ("implemented" or "spec_contract")."""
    return tuple(c for c in REGISTERED_CODES if c.status == status)


__all__ = [
    "DiagnosticCode",
    "REGISTERED_CODES",
    "all_codes",
    "code_lookup",
    "codes_by_pass",
    "codes_by_sprint",
    "codes_by_language",
    "codes_by_status",
]
