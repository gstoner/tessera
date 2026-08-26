"""Arch-1 (2026-05-22) — Central registry of Tessera diagnostic codes.

Before this sprint, diagnostic codes (e.g., ``SYMDIM_BINDING_VIOLATION``)
were defined only at the C++ ``emitOpError``
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
    DiagnosticCode(
        code="AUTODIFF_CONTROL_SCAN_UNSUPPORTED",
        pass_origin="AutodiffPairedPass",
        severity="error",
        summary=(
            "tessera.control_scan is on the gradient path and has no reverse "
            "rule yet — the fourth control primitive is the only one without."
        ),
        fix_hint=(
            "The mathematics is settled (adjoint of a scan is a scan over "
            "reversed t, verified against central differences). What is "
            "missing is the body's paired backward and a residual tape of the "
            "intermediate carries, so the rule belongs in the paired pass "
            "beside the scf region handling, not in "
            "AdjointInterface::buildAdjoint. Until then, restructure the "
            "recurrence as a counted control_for whose carry is explicit."
        ),
        spec="docs/spec/CONTROL_FLOW_CONTRACT.md",
        sprint="W4-PRODUCT-1",
    ),
    DiagnosticCode(
        code="AUTODIFF_STOP_GRADIENT_RESIDUAL_REQUIRED",
        pass_origin="AutodiffPairedPass",
        severity="error",
        summary=(
            "The recompute-all paired pass cannot preserve a stopped "
            "intermediate without replaying its inactive producer cone."
        ),
        fix_hint=(
            "Stop a function input directly or select a paired residual "
            "policy that saves the stopped primal."
        ),
        spec="docs/audit/compiler/AUTODIFF_UNIFICATION_PLAN.md",
        sprint="AD-CORE-EFFECT-CONTROL-1",
    ),
    DiagnosticCode(
        code="AUTODIFF_STOCHASTIC_NO_PRODUCT",
        pass_origin="AutodiffPairedPass",
        severity="error",
        summary=(
            "A stochastic operation entered a differentiated region without a "
            "recorded product, so its replay is not a function of recorded "
            "data."
        ),
        fix_hint=(
            "Attach a keyed_rng recorded product (W4-EFFECTS-1 E1) carrying "
            "the draw's key, shape and dtype, with a 64-character content "
            "digest. A seed in the op's attributes is not a recorded product. "
            "An ambient draw, or one from a caller-owned generator whose "
            "position advances per call, has no product and stays fail-closed."
        ),
        spec="docs/audit/compiler/W4_ADMISSIBLE_EFFECTS_PLAN.md",
        sprint="W4-EFFECTS-1",
    ),
    DiagnosticCode(
        code="AUTODIFF_STOCHASTIC_UNKEYED",
        pass_origin="AutodiffPairedPass",
        severity="error",
        summary=(
            "A stochastic operation carries a recorded product whose class "
            "does not establish reproducibility for a draw."
        ),
        fix_hint=(
            "Only a keyed_rng product makes a draw replayable; the counter-"
            "based generator is a pure function of its key. Re-record the "
            "effect with that class, or leave the operation outside the "
            "differentiated region."
        ),
        spec="docs/audit/compiler/W4_ADMISSIBLE_EFFECTS_PLAN.md",
        sprint="W4-EFFECTS-1",
    ),
    DiagnosticCode(
        code="AUTODIFF_STOCHASTIC_EFFECT",
        pass_origin="AutodiffPass/AutodiffPairedPass",
        severity="error",
        summary="An active stochastic Graph operation has no explicit gradient estimator.",
        fix_hint=(
            "Insert tessera.stop_gradient or register an explicit pathwise or "
            "score-function adjoint for the stochastic operation."
        ),
        spec="docs/audit/compiler/AUTODIFF_UNIFICATION_PLAN.md",
        sprint="AD-CORE-EFFECT-CONTROL-1",
    ),
    DiagnosticCode(
        code="E_LINALG_CONTRACT",
        pass_origin="tessera.linalg_ops",
        severity="error",
        summary=(
            "A matrix-function primitive was called outside its domain: a "
            "non-square operand, a numerically singular matrix, a negative "
            "determinant under logdet, or an unsupported `ord` for norm."
        ),
        fix_hint=(
            "Supply an operand in the op's domain. `ord` is a semantic key and "
            "is never defaulted to another norm; a singular matrix has neither "
            "a value nor a derivative here."
        ),
        spec="docs/audit/compiler/MATRIX_CALCULUS_REVIEW.md MC1",
        sprint="MATRIX-CALCULUS-MC1", language="python",
    ),
    DiagnosticCode(
        code="E_METRIC_CONTRACT",
        pass_origin="tessera.metric",
        severity="error",
        summary=(
            "An object used as a metric is not one: weights that are not "
            "positive, a metric matrix that is not symmetric positive "
            "definite, or a value not implementing the Metric protocol."
        ),
        fix_hint=(
            "Use tessera.metric.Euclidean/Weighted/Sphere/Orthogonal, or a type "
            "providing inner/sharp/project_tangent/retract. An indefinite W is "
            "not an inner product and its 'gradient' can point uphill."
        ),
        spec="docs/audit/compiler/MATRIX_CALCULUS_REVIEW.md MC3",
        sprint="MATRIX-CALCULUS-MC3", language="python",
    ),
    DiagnosticCode(
        code="E_DEGENERATE_FACTORIZATION",
        pass_origin="tessera.autodiff.degeneracy",
        severity="error",
        summary=(
            "A matrix-factorization derivative (svd/qr/cholesky/tri_solve) was "
            "requested at an input where it does not exist: coincident singular "
            "values, or a numerically rank-deficient triangular factor."
        ),
        fix_hint=(
            "Supply a regular input, or select the degeneracy policy explicitly "
            "with tessera.autodiff.degeneracy.degeneracy_policy('generalized' | "
            "'damped:<tau>' | 'unchecked')."
        ),
        spec="docs/audit/compiler/MATRIX_CALCULUS_REVIEW.md MC2",
        sprint="MATRIX-CALCULUS-MC2", language="python",
    ),
    DiagnosticCode(
        code="E_TENSOR_UNWRAP",
        pass_origin="tessera.ops",
        severity="error",
        summary=(
            "An operand could not be unwrapped to numeric data — a wrapper "
            "chain that does not reach an ndarray, or a non-tensor element "
            "inside a sequence operand such as ops.cat/ops.stack."
        ),
        fix_hint=(
            "Pass a Tensor, nn.Parameter, DistributedArray, or numpy array. "
            "The message names the op and, for sequence operands, the index of "
            "the offending element."
        ),
        spec="docs/audit/compiler/MATRIX_CALCULUS_REVIEW.md MC8",
        sprint="MATRIX-CALCULUS-MC8", language="python",
    ),
    DiagnosticCode(
        code="E_FUSED_EPILOGUE_BAD_DTYPE",
        pass_origin="tessera.compiler.fusion_core.FusedRegion",
        severity="error",
        summary="A fused epilogue requested an unsupported storage dtype.",
        fix_hint="Use f16, bf16, f32, fp8_e4m3, or fp8_e5m2 storage.",
        spec="docs/architecture/proposals/tile_fragment_abi.md",
        sprint="NVIDIA-PARITY-EPILOGUE", language="python",
    ),
    DiagnosticCode(
        code="E_FUSED_EPILOGUE_BAD_OP",
        pass_origin="tessera.compiler.fusion_core.FusedRegion",
        severity="error",
        summary="A fused prologue or epilogue requested an unknown operation.",
        fix_hint="Use a registered EPILOGUE_OPS operation.",
        spec="docs/architecture/proposals/tile_fragment_abi.md",
        sprint="NVIDIA-PARITY-EPILOGUE", language="python",
    ),
    DiagnosticCode(
        code="E_FUSED_EPILOGUE_BAD_ORDER",
        pass_origin="tessera.compiler.fusion_core.FusedRegion",
        severity="error",
        summary="A fused epilogue chain violates the shared operation-order contract.",
        fix_hint="Use one bias, pointwise activations, optional residual, then reduction.",
        spec="docs/architecture/proposals/tile_fragment_abi.md",
        sprint="NVIDIA-PARITY-EPILOGUE", language="python",
    ),
    DiagnosticCode(
        code="E_FUSED_EPILOGUE_MISSING_OPERAND",
        pass_origin="tessera.compiler.fusion_core.FusedRegion",
        severity="error",
        summary="A declared bias or residual operand was not supplied.",
        fix_hint="Supply the bias/residual buffer required by the fused region.",
        spec="docs/architecture/proposals/tile_fragment_abi.md",
        sprint="NVIDIA-PARITY-EPILOGUE", language="python",
    ),
    DiagnosticCode(
        code="E_LAUNCH_BINDING_MISMATCH",
        pass_origin="tessera.compiler.native_artifact.LaunchDescriptor",
        severity="error",
        summary="Runtime buffers, scalars, or dynamic shapes violate a compiler-emitted launch contract.",
        fix_hint="Bind exactly the declared names, dtypes, ranks, layouts, alignments, and guarded shapes.",
        spec="docs/spec/NATIVE_ARTIFACT_SPEC.md §3",
        sprint="E2E-SPINE-1", language="python",
    ),
    DiagnosticCode(
        code="E_LAUNCH_DESCRIPTOR_SCHEMA",
        pass_origin="tessera.compiler.native_artifact.LaunchDescriptor",
        severity="error",
        summary="A launch descriptor is malformed, unsupported, or fails its serialized-content fingerprint.",
        fix_hint="Re-emit the descriptor with the current schema and valid ordered ABI, geometry, workspace, and ordering fields.",
        spec="docs/spec/NATIVE_ARTIFACT_SPEC.md §3",
        sprint="E2E-SPINE-1", language="python",
    ),
    DiagnosticCode(
        code="E_LAUNCH_STALE_IMAGE",
        pass_origin="tessera.compiler.native_artifact.LaunchDescriptor",
        severity="error",
        summary="A launch descriptor references a different image, missing symbol, or incompatible ABI identifier.",
        fix_hint="Discard the stale descriptor and load the descriptor emitted with this exact native image.",
        spec="docs/spec/NATIVE_ARTIFACT_SPEC.md §3",
        sprint="E2E-SPINE-1", language="python",
    ),
    DiagnosticCode(
        code="E_NATIVE_IMAGE_DIGEST_MISMATCH",
        pass_origin="tessera.compiler.native_artifact.NativeImageArtifact",
        severity="error",
        summary="Serialized native-image payload, identity, or cache data does not match its SHA-256 fingerprints.",
        fix_hint="Discard the corrupt or stale cache entry and rebuild the native image from its recorded Target IR.",
        spec="docs/spec/NATIVE_ARTIFACT_SPEC.md §2",
        sprint="E2E-SPINE-1", language="python",
    ),
    DiagnosticCode(
        code="E_NATIVE_IMAGE_SCHEMA",
        pass_origin="tessera.compiler.native_artifact.NativeImageArtifact",
        severity="error",
        summary="A native-image artifact has an unsupported version, format, target, pipeline, ABI, or field value.",
        fix_hint="Emit a current versioned artifact using registered targets, pipelines, formats, and non-empty compiler fingerprints.",
        spec="docs/spec/NATIVE_ARTIFACT_SPEC.md §2",
        sprint="E2E-SPINE-1", language="python",
    ),
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
        code="JIT_APPLE_GPU_TRACE_FAILED",
        pass_origin="tessera.compiler.JitDiagnosticCode",
        severity="error",
        summary=(
            "The apple_gpu tracer could not execute the function body. The "
            "tracer is the only execution path for a body whose AST Graph IR "
            "emission was deferred (JIT_APPLE_GPU_TRACE_DEFERRED), so a "
            "construct the tracer does not model -- calling an unmodelled "
            "method on a Tracer, coercing one to a Python scalar -- would "
            "otherwise escape as a raw interpreter exception. Decision #21: "
            "an unsupported lowering names the construct and the target."
        ),
        fix_hint=(
            "Read the quoted decoration-time reason: it names what the AST "
            "front end could not lower. Rewrite that construct in terms of "
            "`tessera.ops.*` / `tessera.control.*`, or drop it from the "
            "jitted region."
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
            "A GEMM, convolution, attention, or last-axis reduction operand's "
            "producer carries a `tessera.layout` attribute outside that "
            "consumer's architecture-neutral accept-set, and no intervening "
            "cast converts it."
        ),
        fix_hint=(
            "Insert a `tessera.cast` that converts the producer's layout to "
            "one of the consumer's accepted layouts before the operation."
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

    # ── async_copy/wait_async single-contract reconciliation (2026-08-10) ──
    DiagnosticCode(
        code="MATMUL_SCHEDULE_ACCUM_UNSUPPORTED",
        pass_origin="GraphToSchedulePass",
        severity="error",
        summary=(
            "numeric_policy.accum names an accumulator the selected matmul "
            "schedule does not provide."
        ),
        fix_hint=(
            "The schedule infers accum from operand/result element types per "
            "target. `accum` selects what the program computes, so a mismatch "
            "is refused rather than replaced by the inference. Declare the "
            "accumulator the target provides, or omit the key."
        ),
        spec="docs/audit/compiler/INTEGRATED_COMPILER_PLAN.md",
        sprint="NUMPOL-CARRIER-1",
    ),
    DiagnosticCode(
        code="NUMERIC_POLICY_ACCUM_UNREALIZABLE",
        pass_origin="TesseraToLinalgPass",
        severity="error",
        summary=(
            "numeric_policy.accum is more precise than storage but the same "
            "width, so no cast on this lowering path expresses it."
        ),
        fix_hint=(
            "bf16 storage with an fp16 accumulator is the case: 8 significand "
            "bits to 11, both 16 bits wide, so arith.extf cannot express it. "
            "Computing in fp32 instead would deliver precision the program "
            "did not ask for. Declare an accumulator wider in bits, or drop "
            "the key."
        ),
        spec="docs/reference/tessera_tensor_attributes.md",
        sprint="NUMPOL-CARRIER-1",
    ),
    DiagnosticCode(
        code="NUMERIC_POLICY_MATH_MODE_NOT_REDUCING",
        pass_origin="IRContractLegalityPass",
        severity="error",
        summary=(
            "numeric_policy.math_mode names an arithmetic at least as precise "
            "as the storage it claims to reduce, so it rounds nothing."
        ),
        fix_hint=(
            "A math mode is a NARROWER arithmetic performed on wider storage. "
            'TF32 (11 significand bits) is an fp32 mode; on bf16 storage (8) '
            "it is a no-op. Drop it, or widen storage. See Decision #15a."
        ),
        spec="docs/reference/tessera_tensor_attributes.md",
        sprint="NUMPOL-CARRIER-1",
    ),
    DiagnosticCode(
        code="NUMERIC_POLICY_NARROWING_ACCUM",
        pass_origin="IRContractLegalityPass",
        severity="error",
        summary=(
            "numeric_policy declares an accumulator with fewer significand "
            "bits than its storage, which is strictly dominated at its own "
            "bit budget and makes the wider storage unobservable."
        ),
        fix_hint=(
            "Widen accum to at least the storage dtype, or narrow storage to "
            "the accum dtype and keep the bandwidth. Measured: at 48 dtype "
            "bits, fp16/fp32 is 25.8x more accurate than fp32/fp16, and the "
            "fp32/bf16 result is bit-identical to bf16/bf16."
        ),
        spec="docs/audit/compiler/INTEGRATED_COMPILER_PLAN.md",
        sprint="NUMPOL-CARRIER-1",
    ),
    DiagnosticCode(
        code="NUMERIC_POLICY_NON_STRING_VALUE",
        pass_origin="IRContractLegalityPass",
        severity="error",
        summary=(
            "A numeric_policy entry holds a non-string value, which reads "
            "back as ABSENT through StringAttr lookup."
        ),
        fix_hint="Write the dtype or mode as a quoted name.",
        spec="docs/reference/tessera_tensor_attributes.md",
        sprint="NUMPOL-CARRIER-1",
    ),
    DiagnosticCode(
        code="NUMERIC_POLICY_NOT_A_DICTIONARY",
        pass_origin="IRContractLegalityPass",
        severity="error",
        summary=(
            "numeric_policy is present but is not a dictionary, so every "
            "consumer's DictionaryAttr lookup reads it back as absent."
        ),
        fix_hint=(
            "A wrongly typed attribute is invisible, not merely unchecked. "
            "If this is a different contract, give it a different name — the "
            "spectral reduction-order contract became "
            "tessera.spectral_accumulation / tessera.spectral_normalization "
            "for exactly this reason. Ops that declare numeric_policy in ODS "
            "are already covered by the attribute constraint; this catches "
            "the discardable case on ops that do not."
        ),
        spec="docs/reference/tessera_tensor_attributes.md",
        sprint="NUMPOL-CARRIER-1",
    ),
    DiagnosticCode(
        code="NUMERIC_POLICY_UNKNOWN_ACCUM",
        pass_origin="IRContractLegalityPass",
        severity="error",
        summary="numeric_policy.accum is not a known accumulator dtype.",
        fix_hint=(
            "Use fp64/fp32/fp16/bf16 or int64/int32/int16/int8. Sub-8-bit "
            "formats are storage dtypes, not accumulators."
        ),
        spec="docs/reference/tessera_tensor_attributes.md",
        sprint="NUMPOL-CARRIER-1",
    ),
    DiagnosticCode(
        code="NUMERIC_POLICY_UNKNOWN_KEY",
        pass_origin="IRContractLegalityPass",
        severity="error",
        summary=(
            "numeric_policy contains a key no Tessera contract defines — the "
            "way a misspelling becomes a silently absent semantic contract."
        ),
        fix_hint=(
            "Legal keys: storage, accum, math_mode, rounding_mode, softmax. "
            "A typo'd `accum` leaves the op with no accumulator contract "
            "while appearing to state one (Decisions #15a/#21a)."
        ),
        spec="docs/reference/tessera_tensor_attributes.md",
        sprint="NUMPOL-CARRIER-1",
    ),
    DiagnosticCode(
        code="NUMERIC_POLICY_UNKNOWN_MATH_MODE",
        pass_origin="IRContractLegalityPass",
        severity="error",
        summary="numeric_policy.math_mode is outside the declared legal set.",
        fix_hint="Use ieee, default, tf32, bf16x3 or fp16x2.",
        spec="docs/reference/tessera_tensor_attributes.md",
        sprint="NUMPOL-CARRIER-1",
    ),
    DiagnosticCode(
        code="NUMERIC_POLICY_UNKNOWN_ROUNDING_MODE",
        pass_origin="IRContractLegalityPass",
        severity="error",
        summary="numeric_policy.rounding_mode is outside the declared legal set.",
        fix_hint=(
            "Use round_to_nearest_even/_away, round_toward_zero/_positive/"
            "_negative, or stochastic."
        ),
        spec="python/tessera/compiler/rounding.py",
        sprint="NUMPOL-CARRIER-1",
    ),
    DiagnosticCode(
        code="NVIDIA_MATH_MODE_UNAVAILABLE",
        # Emitted from runtime.py, not a C++ pass: the selection was
        # deliberately extracted into a pure Python function so its contract
        # is testable on a host with no NVIDIA device. The registry's
        # mlir/python split keeps the C++-emission cross-check honest, and it
        # caught this entry the first time the full sweep ran.
        language="python",
        pass_origin="runtime._nvidia_gemm_selection",
        severity="error",
        summary=(
            "numeric_policy.math_mode names an arithmetic the shipped NVIDIA "
            "mma.sync lane does not provide."
        ),
        fix_hint=(
            "The only fp32-storage kernel is mma.sync m16n8k8 in TF32 (11 "
            "significand bits vs fp32's 24); tensor cores have no IEEE-fp32 "
            'instruction. Declare math_mode="tf32" to accept that, or route '
            "the matmul to a non-tensor-core path."
        ),
        spec="docs/reference/tessera_tensor_attributes.md",
        sprint="NUMPOL-CARRIER-1",
    ),
    DiagnosticCode(
        code="ROCM_WMMA_ACCUM_UNSUPPORTED",
        pass_origin="GenerateWMMAGemmKernel",
        severity="error",
        summary=(
            "numeric_policy.accum names an accumulator the emitted gfx1151 "
            "WMMA path does not provide."
        ),
        fix_hint=(
            "`accum` selects what the program computes, so it is refused "
            "rather than substituted. The f16-accumulate WMMA "
            "(V_WMMA_F16_16X16X16_F16) exists on RDNA 3.5 but has a different "
            "accumulator ABI (v16f16 + opsel) and is not wired yet; declare "
            'accum="fp32" (or "int32" for integer storage), or omit it.'
        ),
        spec="docs/audit/compiler/INTEGRATED_COMPILER_PLAN.md",
        sprint="NUMPOL-CARRIER-1",
    ),
    DiagnosticCode(
        code="TILE_ASYNC_STAGE_NEGATIVE",
        pass_origin="AsyncCopyOp::verify / WaitAsyncOp::verify",
        severity="error",
        summary=(
            "`stage` on tile.async_copy / tile.wait_async is an optional "
            "legacy-form grouping key, but when present it must be >= 0."
        ),
        fix_hint=(
            "Drop the `stage` attribute (typed !tile.async_token SSA form) "
            "or set it to a non-negative stage index."
        ),
        spec="docs/spec/TILE_IR.md",
        sprint="TILE-SYNC-RECONCILE-2026-08-10",
    ),

    # ── TILE-SYNC-TYPED-2026-08-15 — CAKE Phase 1 typed sync verifiers ─────
    DiagnosticCode(
        code="TILE_WAIT_UNTYPED_DEPENDENCY",
        pass_origin="MBarrierWaitOp::verify",
        severity="error",
        summary=(
            "A tile.mbarrier.wait dependency operand is not a typed sync "
            "value (!tile.async_token / !tile.mbarrier_token) — an arbitrary "
            "SSA value is not a completion edge."
        ),
        fix_hint=(
            "Thread the copy's !tile.async_token or the arrive's "
            "!tile.mbarrier_token into the wait instead of a data/index value."
        ),
        spec="docs/audit/compiler/compiler_enhancement.md §5.2.1",
        sprint="TILE-SYNC-TYPED-2026-08-15",
    ),
    DiagnosticCode(
        code="TILE_WAIT_GATES_ON_NOTHING",
        pass_origin="MBarrierWaitOp::verify",
        severity="error",
        summary=(
            "A tile.mbarrier.wait has no mbarrier token, no async-token "
            "dependency, and no explicit tile.retire_all marker — it releases "
            "against no completion edge."
        ),
        fix_hint=(
            "Consume the arrive's token or a copy's async token, or stamp "
            "tile.retire_all for the declared retire-everything semantics."
        ),
        spec="docs/audit/compiler/compiler_enhancement.md §5.2.1",
        sprint="TILE-SYNC-TYPED-2026-08-15",
    ),
    DiagnosticCode(
        code="TILE_TMA_COPY_GATES_ON_NOTHING",
        pass_origin="TMACopyAsyncOp::verify",
        severity="error",
        summary=(
            "A tile.tma.copy_async has no SSA mbarrier operand, no "
            "!tile.async_token result, and no legacy grouping key — no wait "
            "can ever retire it."
        ),
        fix_hint=(
            "Bind an SSA mbarrier (with expect_tx), return an async token, "
            "or carry a legacy tile.barrier_id / stage grouping key."
        ),
        spec="docs/audit/compiler/compiler_enhancement.md §5.2.1",
        sprint="TILE-SYNC-TYPED-2026-08-15",
    ),
    DiagnosticCode(
        code="TILE_WAIT_TOKEN_UNPAIRED",
        pass_origin="TileDataflowLegality",
        severity="error",
        summary=(
            "A wait's !tile.mbarrier_token cannot be resolved (across "
            "scf.for block-argument edges) to tile.mbarrier.arrive_expect_tx "
            "results."
        ),
        fix_hint=(
            "Produce the token from arrive_expect_tx and carry it through "
            "loop iter_args; an underivable pairing fails closed."
        ),
        spec="docs/audit/compiler/compiler_enhancement.md §5.3.1",
        sprint="TILE-SYNC-TYPED-2026-08-15",
    ),
    DiagnosticCode(
        code="TILE_WAIT_SLOT_MISMATCH",
        pass_origin="TileDataflowLegality",
        severity="error",
        summary=(
            "The wait and its resolved arrive name different mbarrier slots "
            "— the wait never releases on hardware."
        ),
        fix_hint="Use one slot index on the arrive/wait pair per barrier phase.",
        spec="docs/audit/compiler/compiler_enhancement.md §5.3.1",
        sprint="TILE-SYNC-TYPED-2026-08-15",
    ),
    DiagnosticCode(
        code="TILE_WAIT_BARRIER_DISAGREES",
        pass_origin="TileDataflowLegality",
        severity="error",
        summary=(
            "The wait's barrier and its token's arrive barrier resolve to "
            "different tile.mbarrier.init roots."
        ),
        fix_hint="Arrive and wait on the same SSA barrier value.",
        spec="docs/audit/compiler/compiler_enhancement.md §5.3.1",
        sprint="TILE-SYNC-TYPED-2026-08-15",
    ),
    DiagnosticCode(
        code="TILE_BARRIER_ORIGIN_UNRESOLVED",
        pass_origin="TileDataflowLegality",
        severity="error",
        summary=(
            "A !tile.mbarrier operand cannot be resolved to a "
            "tile.mbarrier.init across its def-use/loop-carry chain — an "
            "unprovable origin is unproven, not permitted."
        ),
        fix_hint=(
            "Initialize the barrier with tile.mbarrier.init in the kernel "
            "and thread it through scf.for iter_args, not opaque values."
        ),
        spec="docs/audit/compiler/compiler_enhancement.md §5.3.1",
        sprint="TILE-SYNC-TYPED-2026-08-15",
    ),
    DiagnosticCode(
        code="TILE_ROLE_RELATION_INVALID",
        pass_origin="TileDataflowLegality",
        severity="error",
        summary=(
            "A logical producer/consumer role cannot be resolved to registered "
            "tile.role declarations, disagrees with its pipeline role, or "
            "leaves a role-bearing barrier without both role sets."
        ),
        fix_hint=(
            "Create producer and consumer tile.role SSA values, thread roles "
            "through supported scf.for iter_args, and bind them to the owning "
            "pipeline or mbarrier rather than using string role markers."
        ),
        spec="docs/audit/compiler/compiler_enhancement.md §6.2–§6.4",
        sprint="LAYOUT-SCHEDULE-OBJECT-2026-08-16",
    ),
    DiagnosticCode(
        code="TILE_PIPELINE_RING_STALE",
        pass_origin="TileDataflowLegality",
        severity="error",
        summary=(
            "A loop-carried !tile.pipeline_state is yielded un-advanced (or "
            "advances a different state) — the ring never moves."
        ),
        fix_hint=(
            "Yield the tile.pipeline_advance result of THIS loop's iter_arg "
            "on the back edge."
        ),
        spec="docs/audit/compiler/compiler_enhancement.md §5.3.1",
        sprint="TILE-SYNC-TYPED-2026-08-15",
    ),
    DiagnosticCode(
        code="TILE_PIPELINE_RING_UNDERIVED",
        pass_origin="TileDataflowLegality",
        severity="error",
        summary=(
            "The yielded !tile.pipeline_state is not derived from "
            "tile.pipeline_advance on the loop's iter_arg; an unprovable "
            "ring fails closed."
        ),
        fix_hint=(
            "Derive the yielded state from pipeline_advance of the carried "
            "state instead of re-initializing or forwarding foreign state."
        ),
        spec="docs/audit/compiler/compiler_enhancement.md §5.3.1",
        sprint="TILE-SYNC-TYPED-2026-08-15",
    ),
    DiagnosticCode(
        code="TILE_TMA_EXPECT_MISMATCH",
        pass_origin="TileDataflowLegality",
        severity="error",
        summary=(
            "A tile.tma.copy_async and the descriptor it reaches through SSA "
            "declare different expected transaction byte counts — the wait "
            "on this barrier never releases."
        ),
        fix_hint=(
            "Declare one expect_tx on the descriptor/copy pair (SSA identity "
            "is authoritative; string barrier ids do not override it)."
        ),
        spec="docs/audit/compiler/compiler_enhancement.md §5.3.1",
        sprint="TILE-SYNC-TYPED-2026-08-15",
    ),
    DiagnosticCode(
        code="TILE_TMA_DESC_ORIGIN_UNRESOLVED",
        pass_origin="TileDataflowLegality",
        severity="error",
        summary=(
            "A copy's descriptor operand cannot be resolved to a "
            "tile.tma.descriptor across the def-use/loop-carry chain."
        ),
        fix_hint=(
            "Produce the descriptor with tile.tma.descriptor in-kernel and "
            "carry it through loop iter_args."
        ),
        spec="docs/audit/compiler/compiler_enhancement.md §5.3.1",
        sprint="TILE-SYNC-TYPED-2026-08-15",
    ),

    # ── W1.1 step 3 — pointer-backed Tile address contracts ────────────────
    DiagnosticCode(
        code="TILE_VIEW_POINTER_ARITY",
        pass_origin="ViewOp::verify",
        severity="error",
        summary=(
            "A pointer-backed tile.view takes (base, rowOrigin, colOrigin), "
            "optionally followed by (rowBound, colBound); when memory layout "
            "leading_dim is zero, a final SSA leading dimension is required."
        ),
        fix_hint=(
            "Use 3/5 operands for a static leading dimension or 4/6 for a "
            "dynamic one, with the leading dimension last."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_STORE_POINTER_ARITY",
        pass_origin="StoreOp::verify",
        severity="error",
        summary=(
            "A pointer-backed tile.store takes tile, base, row/column origin, "
            "optional row/column bounds, and a final SSA leading dimension "
            "exactly when memory layout leading_dim is zero."
        ),
        fix_hint=(
            "Use 4/6 operands for a static leading dimension or 5/7 for a "
            "dynamic one, with the leading dimension last."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),

    # ── W1.1 step 2b (guard) — NVIDIA WGMMA accumulator ────────────────────
    DiagnosticCode(
        code="NVWGMMA_ACCUMULATOR_DROPPED",
        pass_origin="NVWGMMALoweringPass",
        severity="error",
        summary=(
            "NVWGMMALoweringPass accepts A/B and an optional accumulator but "
            "cannot represent additional architecture-specific data operands. "
            "It refuses those extended forms instead of silently dropping an "
            "operand at the opaque backend-call boundary."
        ),
        fix_hint=(
            "Use the architecture-owned NVIDIA lowering for extended MMA forms "
            "such as block-scaled operands; the legacy pass supports only "
            "A/B/C."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),

    # ── W1.1 step 2 — the type-based tile.mma contract ─────────────────────
    DiagnosticCode(
        code="TILE_MMA_BARE_FRAGMENT_REMOVED",
        pass_origin="MMAOp::verify",
        severity="error",
        summary=(
            "A tile.mma operand uses the historical all-unknown !tile.fragment spelling after the permissive producer-chasing contract was removed."
        ),
        fix_hint=(
            "Parameterize the fragment with m/n/k, elem, acc, role, layout, and family."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_MMA_VALUE_ARITY",
        pass_origin="MMAOp::verify",
        severity="error",
        summary="The temporary tensor value lane has an invalid operand/result arity.",
        fix_hint="Use A, B, optional accumulator, and exactly one result; migrate physical producers to typed fragments.",
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_MMA_VALUE_ACCUMULATOR",
        pass_origin="MMAOp::verify",
        severity="error",
        summary="A tensor-valued tile.mma accumulator disagrees with its result type.",
        fix_hint="Make the accumulator and result types identical.",
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_MMA_VALUE_TYPE",
        pass_origin="MMAOp::verify",
        severity="error",
        summary="A non-fragment tile.mma value operand is not a ranked tensor.",
        fix_hint="Use the tile.matmul value lane or parameterized !tile.fragment operands.",
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_TCGEN05_FRAGMENT_CONTRACT",
        pass_origin="TCGen05MMAOp::verify",
        severity="error",
        summary="TCGen05 operands do not carry the required typed fragment ABI.",
        fix_hint=(
            "Pack parameterized role-a/role-b tcgen05 fragments matching the "
            "MMA descriptor."
        ),
        spec="docs/audit/compiler/compiler_enhancement.md#52-ods-tightening",
        sprint="W1.1/CAKE-P1",
    ),
    DiagnosticCode(
        code="TILE_MMA_MIXED_FRAGMENT_FORMS",
        pass_origin="MMAOp::verify",
        severity="error",
        summary=(
            "A tile.mma mixes parameterized and bare !tile.fragment operands. The typed form is all-or-nothing: otherwise the op gets the weaker contract on whichever operand still carries the bare type."
        ),
        fix_hint=(
            "Migrate every operand of this op together, or leave them all bare."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_MMA_ARITY",
        pass_origin="MMAOp::verify",
        severity="error",
        summary=(
            "Wrong number of DATA operands (3, or 5 for NVFP4). Control operands such as !tile.async_token are excluded from the count."
        ),
        fix_hint=(
            "Pass A, B, accumulator (plus scale_a/scale_b for NVFP4)."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_MMA_OPERAND_ROLE",
        pass_origin="MMAOp::verify",
        severity="error",
        summary=(
            "Operand roles are positional -- A, B, accumulator. A fragment in the wrong slot is a swapped-operand bug that type-checks under any contract that does not state the role."
        ),
        fix_hint=(
            "Reorder the operands, or fix the role in the fragment type."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_MMA_SHAPE_MISMATCH",
        pass_origin="MMAOp::verify",
        severity="error",
        summary=(
            "Operands state different m/n/k. One MMA is one instruction shape."
        ),
        fix_hint=(
            "Give every operand of this MMA the same m/n/k."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_MMA_FAMILY_MISMATCH",
        pass_origin="MMAOp::verify",
        severity="error",
        summary=(
            "Operands state different instruction families. family selects a physical register ABI (wave 32 for RDNA/WMMA vs 64 for CDNA/MFMA), so fragments from different families are not interchangeable."
        ),
        fix_hint=(
            "Use one family per MMA; auto before target resolution."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_MMA_ACCUM_MISMATCH",
        pass_origin="MMAOp::verify",
        severity="error",
        summary=(
            "Operands state different accumulator dtypes. Decision #15a: one accumulator contract per MMA -- and this is the fact W1.3's boundary verifier carries down from Graph IR."
        ),
        fix_hint=(
            "Give every operand the same acc."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_MMA_ACCUM_ELEMENT",
        pass_origin="MMAOp::verify",
        severity="error",
        summary=(
            "The accumulator fragment's elem is not its acc. Without this a producer could hand over a bf16-element fragment while every operand agreed on an fp32 accumulator -- accumulator-width confusion wearing a correct label."
        ),
        fix_hint=(
            "Set the accumulator fragment's elem to its acc."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_MMA_RESULT_TYPE",
        pass_origin="MMAOp::verify",
        severity="error",
        summary=(
            "tile.mma returns exactly one value, the updated accumulator, whose type must equal the accumulator operand's. This equality is what lets scf.for's own iter-arg/yield rule close a K-loop with no Tile-specific loop reasoning."
        ),
        fix_hint=(
            "Give the result the accumulator operand's type."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_MMA_DESC_DISAGREES",
        pass_origin="tile::descriptorAgreesWithFragment",
        severity="error",
        summary=(
            "A #tile.mma_desc contradicts the fragment types it accompanies. The descriptor is optional on the typed path (it still carries k_blocks), but a stale one must not contradict the types that superseded it."
        ),
        fix_hint=(
            "Update or remove the descriptor."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_FRAGMENT_ROLE_DISAGREES",
        pass_origin="FragmentPackOp::verify / FragmentZeroOp::verify",
        severity="error",
        summary=(
            "A role attribute contradicts the result fragment type's role."
        ),
        fix_hint=(
            "Drop the attribute -- the type states it -- or make them agree."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_FRAGMENT_UNPACK_ROLE",
        pass_origin="FragmentUnpackOp::verify",
        severity="error",
        summary=(
            "Only an accumulator fragment may be unpacked. Read from the input "
            "TYPE: the previous check chased the producing op, so it rejected a "
            "K-loop accumulator unpacked after the loop (an scf.for result, "
            "whose defining op is the loop and carries no descriptor)."
        ),
        fix_hint="Unpack the accumulator, whose fragment type has role acc.",
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_FRAGMENT_ZERO_ROLE",
        pass_origin="FragmentZeroOp::verify",
        severity="error",
        summary=(
            "tile.fragment_zero produces the accumulator, so its result type must have role acc."
        ),
        fix_hint=(
            "Set the result fragment type's role to acc."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),

    # ── W1.1 step 1 — !tile.fragment domain validation ────────────────────
    DiagnosticCode(
        code="TILE_FRAGMENT_PARTIAL_CONTRACT",
        pass_origin="FragmentType::verify",
        severity="error",
        summary=(
            "A `!tile.fragment` states part of its instruction contract and not the rest. This is the dangerous middle state: `isUnknown()` returns false for it, so type-based verification reads it as a stated contract when the producer only filled in half."
        ),
        fix_hint=(
            "State the whole contract, or write the bare `!tile.fragment` for the unknown form."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_FRAGMENT_BAD_FAMILY",
        pass_origin="FragmentType::verify",
        severity="error",
        summary=(
            "`family` is outside {auto, mma_sync, wgmma, tcgen05, wmma, mfma}. It selects a physical register ABI (wave 32 for RDNA/WMMA vs 64 for CDNA/MFMA), not just a mnemonic."
        ),
        fix_hint=(
            "Use one of the six families, or `auto` before target resolution."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_FRAGMENT_NONPOSITIVE_SHAPE",
        pass_origin="FragmentType::verify",
        severity="error",
        summary=(
            "A fragment's m/n/k must be > 0. Mirrors TILE_MMA_DESC_NONPOSITIVE_SHAPE."
        ),
        fix_hint=(
            "State the instruction tile shape."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_FRAGMENT_BAD_ROLE",
        pass_origin="FragmentType::verify",
        severity="error",
        summary=(
            "`role` is outside {a, b, acc, scale_a, scale_b}. The name is overloaded in this tree — producer/consumer/manager are WARP roles and input/scratch are BUFFER roles — so a plausible value from a neighbouring vocabulary is the likely mistake."
        ),
        fix_hint=(
            "Use a fragment role: a, b, acc, scale_a, or scale_b."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),
    DiagnosticCode(
        code="TILE_FRAGMENT_BAD_LAYOUT",
        pass_origin="FragmentType::verify",
        severity="error",
        summary=(
            "A fragment's `layout` is the operand layout the INSTRUCTION requires (row_major/col_major), not the `#tile.layout` shard map. Similar names, different facts."
        ),
        fix_hint=(
            "Use row_major or col_major."
        ),
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md",
        sprint="W1.1",
    ),

    # ── W1.3 — Decision #32 metadata lowering obligation ──────────────────
    #
    # A boundary lowering carries each Decision #15a attribute forward, or
    # records a named reason it dropped it. These five are the refusals.
    DiagnosticCode(
        code="METADATA_OBLIGATION_SILENT_DROP",
        pass_origin="VerifyMetadataObligationPass",
        severity="error",
        summary=(
            "A Decision #15a attribute was present before a level boundary and "
            "is gone after it, with no reason recorded — the defect Decision "
            "#32 names: `numeric_policy` stated at Graph IR and absent by the "
            "time codegen picks an MMA instruction."
        ),
        fix_hint=(
            "Carry the attribute forward in the target level's vocabulary (a "
            "re-spelling such as `tessera.layout` -> `tile.layout` counts), or "
            "declare `tessera.lowering.dropped = { <attr> = \"<reason>\" }` on "
            "the function or module."
        ),
        spec="docs/audit/compiler/IR_STACK_INTEGRATION_REVIEW.md §U5",
        sprint="W1.3",
    ),
    DiagnosticCode(
        code="METADATA_OBLIGATION_VALUE_DROP",
        pass_origin="VerifyMetadataObligationPass",
        severity="error",
        summary=(
            "A Decision #15a attribute still exists after the boundary, but a "
            "VALUE it carried before does not — an occurrence was lost, or a "
            "policy was replaced. The first version of this pass snapshotted "
            "only attribute NAMES, so `accum = \"fp32\"` becoming "
            "`accum = \"fp16\"` fired nothing: exactly the instruction-selection "
            "corruption the verifier exists to prevent."
        ),
        fix_hint=(
            "Carry the value forward, or declare a reason — `re_expressed` when "
            "the value was re-encoded in the target level's vocabulary (e.g. "
            "`layout = \"row_major\"` becoming `#tile.layout<...>`)."
        ),
        spec="docs/audit/compiler/IR_STACK_INTEGRATION_REVIEW.md §U5",
        sprint="W1.3",
    ),
    DiagnosticCode(
        code="METADATA_OBLIGATION_UNKNOWN_REASON",
        pass_origin="VerifyMetadataObligationPass",
        severity="error",
        summary=(
            "The recorded drop reason is outside the closed set. Per Decision "
            "#21a the reason selects semantics — it distinguishes \"moved into "
            "the type\" from \"nobody has done this yet\" — so it fails closed."
        ),
        fix_hint=(
            "Use one of: represented_in_type, target_invariant, "
            "consumed_by_pass, not_yet_carried:<plan item>."
        ),
        spec="docs/audit/compiler/IR_STACK_INTEGRATION_REVIEW.md §U5",
        sprint="W1.3",
    ),
    DiagnosticCode(
        code="METADATA_OBLIGATION_DEBT_UNATTRIBUTED",
        pass_origin="VerifyMetadataObligationPass",
        severity="error",
        summary=(
            "A drop declared `not_yet_carried` with no plan item — a silent "
            "drop with extra syntax: it satisfies the verifier's letter, "
            "records nothing actionable, and no item is on the hook for it."
        ),
        fix_hint="Write `not_yet_carried:<item>`, e.g. not_yet_carried:W1.1.",
        spec="docs/audit/compiler/IR_STACK_INTEGRATION_REVIEW.md §U5",
        sprint="W1.3",
    ),
    DiagnosticCode(
        code="METADATA_OBLIGATION_STALE_DECLARATION",
        pass_origin="VerifyMetadataObligationPass",
        severity="error",
        summary=(
            "An attribute is declared dropped but is still present. Decision "
            "#29 applied to this mechanism: the exception carries nothing, yet "
            "reads in review as a considered decision and would license a real "
            "future drop nobody looked at."
        ),
        fix_hint="Remove the `tessera.lowering.dropped` entry.",
        spec="docs/audit/compiler/IR_STACK_INTEGRATION_REVIEW.md §U5",
        sprint="W1.3",
    ),
    DiagnosticCode(
        code="METADATA_OBLIGATION_NO_SNAPSHOT",
        pass_origin="VerifyMetadataObligationPass",
        severity="error",
        summary=(
            "The verify pass ran with no recorded snapshot. Failing closed is "
            "the point: succeeding would make the gate green on every pipeline "
            "that forgot to record, which is indistinguishable from a pipeline "
            "with no losses — an unrun check must not look like a passed one."
        ),
        fix_hint=(
            "Run --tessera-record-metadata before the boundary lowering in the "
            "same tessera-opt invocation."
        ),
        spec="docs/audit/compiler/IR_STACK_INTEGRATION_REVIEW.md §U5",
        sprint="W1.3",
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

    # ── Queue dialect verifiers (V8) — REMOVED 2026-08-10 ────────────────
    # The six QUEUE_* codes were deleted with the `tessera.queue` MLIR
    # dialect (Decisions #29/#31): the verifiers that emitted them had zero
    # producers and the dotted type syntax was unparseable in lit IR.  The
    # live Python tile IR queue vocabulary reports through TILE_IR_* /
    # MEM_* codes instead (tile_ir.py, memory_verifier.py).

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
        code="SYMDIM_PRESBURGER_MALFORMED",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "The typed `tessera.presburger_constraints` carrier has an "
            "unsupported version or malformed symbol/coefficient rows."
        ),
        fix_hint=(
            "Regenerate the carrier through `PresburgerSystem`; every row "
            "must have one coefficient per unique symbol."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="W4.2-2026-08-12",
    ),
    DiagnosticCode(
        code="SYMDIM_PRESBURGER_UNSATISFIABLE",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary=(
            "The typed integer-affine shape constraints have no integer "
            "solution under the available dimension witnesses."
        ),
        fix_hint=(
            "Correct the affine constraint rows or the concrete values in "
            "`tessera.dim_sizes`; nonlinear relations require a separate guard."
        ),
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="W4.2-2026-08-12",
    ),
    DiagnosticCode(
        code="SYMDIM_NONLINEAR_GUARD_MALFORMED",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary="A typed polynomial shape-guard carrier is malformed or overflows exact i64 evaluation.",
        fix_hint="Regenerate the carrier through NonlinearWitnessSystem with total power rows and bounded powers.",
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="W4-PRODUCT-1-2026-08-18",
    ),
    DiagnosticCode(
        code="SYMDIM_NONLINEAR_GUARD_INCOMPLETE",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary="A polynomial shape guard lacks a concrete witness for one or more symbols.",
        fix_hint="Specialize every guard symbol through tessera.dim_sizes or omit the unproved nonlinear guard.",
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="W4-PRODUCT-1-2026-08-18",
    ),
    DiagnosticCode(
        code="SYMDIM_NONLINEAR_GUARD_VIOLATION",
        pass_origin="SymbolicDimEqualityPass",
        severity="error",
        summary="A fully witnessed polynomial shape guard evaluates false.",
        fix_hint="Correct the specialization dimensions or the polynomial relation before lowering.",
        spec="docs/spec/SHAPE_SYSTEM.md §11.2",
        sprint="W4-PRODUCT-1-2026-08-18",
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
    DiagnosticCode(
        code="TILE_COMPOSED_LAYOUT_BAD_LEAF", pass_origin="TileComposedLayoutAttr",
        severity="error",
        summary="A composed-layout shape or stride tree contains an invalid leaf value.",
        fix_hint="Use positive shape leaves or -1 dynamic residues, and nonnegative stride leaves or -1 dynamic residues.",
        spec="docs/audit/compiler/CUTE_IR_ASSESSMENT.md §L1", sprint="LAYOUT-ALG-1/L1",
    ),
    DiagnosticCode(
        code="TILE_COMPOSED_LAYOUT_BAD_TREE", pass_origin="TileComposedLayoutAttr",
        severity="error",
        summary="A composed-layout tree is empty or contains a non-integer, non-group node.",
        fix_hint="Encode nested modes solely as non-empty ArrayAttr groups with IntegerAttr leaves.",
        spec="docs/audit/compiler/CUTE_IR_ASSESSMENT.md §L1", sprint="LAYOUT-ALG-1/L1",
    ),
    DiagnosticCode(
        code="TILE_COMPOSED_LAYOUT_PROFILE_MISMATCH", pass_origin="TileComposedLayoutAttr",
        severity="error",
        summary="A composed-layout shape and stride tree have different nesting profiles.",
        fix_hint="Give each shape tree an identically grouped stride tree.",
        spec="docs/audit/compiler/CUTE_IR_ASSESSMENT.md §L1", sprint="LAYOUT-ALG-1/L1",
    ),
    DiagnosticCode(
        code="TILE_COMPOSED_LAYOUT_BASIS_RANK", pass_origin="TileComposedLayoutAttr",
        severity="error",
        summary="The composed-layout basis or offsets do not match the outer coordinate rank.",
        fix_hint="Provide one paired basis tree and one offset for every outer coordinate leaf.",
        spec="docs/audit/compiler/CUTE_IR_ASSESSMENT.md §L1", sprint="LAYOUT-ALG-1/L1",
    ),
    DiagnosticCode(
        code="TILE_COMPOSED_LAYOUT_BAD_BASIS", pass_origin="TileComposedLayoutAttr",
        severity="error",
        summary="A composed-layout basis entry is not a [shape_tree, stride_tree] pair.",
        fix_hint="Encode every tuple basis component as exactly two nested integer trees.",
        spec="docs/audit/compiler/CUTE_IR_ASSESSMENT.md §L1", sprint="LAYOUT-ALG-1/L1",
    ),
    DiagnosticCode(
        code="TILE_COMPOSED_LAYOUT_NOT_MATERIALIZABLE", pass_origin="MaterializeComposedLayoutOp",
        severity="error",
        summary="A composed layout failed canonical validation by the native layout authority.",
        fix_hint="Use a scalar affine basis and pass one i64 per dynamic leaf after the logical coordinates in canonical preorder.",
        spec="docs/audit/compiler/CUTE_IR_ASSESSMENT.md §L1", sprint="LAYOUT-ALG-1/L1",
    ),
    DiagnosticCode(
        code="TILE_COMPOSED_LAYOUT_TUPLE_ARITY", pass_origin="MaterializeComposedLayoutTupleOp",
        severity="error",
        summary="A tuple composed-layout materializer has an invalid coordinate or result arity.",
        fix_hint="Pass one i64 coordinate per shared domain leaf and request one i64 result per tuple component.",
        spec="docs/audit/compiler/CUTE_IR_ASSESSMENT.md §L5", sprint="LAYOUT-ALG-1/L5",
    ),
    DiagnosticCode(
        code="TILE_COMPOSED_LAYOUT_TUPLE_NOT_MATERIALIZABLE", pass_origin="MaterializeComposedLayoutTupleOp",
        severity="error",
        summary="A tuple component is outside the target-proven composed-layout subset.",
        fix_hint="Use independently materializable static scalar components; retain dynamic or non-separable tuples as carrier-only IR.",
        spec="docs/audit/compiler/CUTE_IR_ASSESSMENT.md §L5", sprint="LAYOUT-ALG-1/L5",
    ),
    DiagnosticCode(
        code="TILE_COMPOSED_LAYOUT_TUPLE_DOMAIN", pass_origin="MaterializeComposedLayoutTupleOp",
        severity="error",
        summary="Tuple composed-layout components do not share one coordinate domain.",
        fix_hint="Give every scalar component the same outer shape and coordinate rank.",
        spec="docs/audit/compiler/CUTE_IR_ASSESSMENT.md §L5", sprint="LAYOUT-ALG-1/L5",
    ),
    DiagnosticCode(
        code="TILE_COMPOSED_LAYOUT_TUPLE_COORDINATES", pass_origin="MaterializeComposedLayoutTupleOp",
        severity="error",
        summary="The tuple composed-layout coordinate count does not match its shared domain.",
        fix_hint="Pass exactly one i64 coordinate for each flattened outer-domain leaf.",
        spec="docs/audit/compiler/CUTE_IR_ASSESSMENT.md §L5", sprint="LAYOUT-ALG-1/L5",
    ),
    DiagnosticCode(
        code="TILE_PACKED_FORMAT_INVALID", pass_origin="TilePackedFormatAttr",
        severity="error",
        summary="A packed logical dtype, integer container, encoding, or lane-order contract is inconsistent.",
        fix_hint="Use the canonical logical bit width, container factor, signedness, encoding, and lane order for the packed dtype.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C4",
        sprint="NVIDIA-PACKED-SSA-FOUNDATION",
    ),
    DiagnosticCode(
        code="TILE_PACKED_VIEW_INVALID", pass_origin="TilePackedPhysicalViewAttr",
        severity="error",
        summary="A concrete packed buffer view has invalid axes, strides, alignment, offset, or scale binding metadata.",
        fix_hint="Bind a valid packed format to an explicit packing axis and positive container strides, with compatible scale metadata.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C4",
        sprint="NVIDIA-PACKED-SSA-FOUNDATION",
    ),
    DiagnosticCode(
        code="TILE_SCALE_LAYOUT_INVALID", pass_origin="TileScaleLayoutAttr",
        severity="error",
        summary="A block-scale dtype, block size, axis, layout, stride, alignment, or offset is inconsistent.",
        fix_hint="Use the canonical scale dtype and positive block/stride contract, or the exact dtype=none sentinel for unscaled storage.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C4",
        sprint="NVIDIA-PACKED-SSA-FOUNDATION",
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
    # Portable fragment ABI attributes used by Tile-to-target materializers.
    DiagnosticCode(
        code="TILE_EPILOGUE_BAD_ACTIVATION", pass_origin="TileEpilogueAttr",
        severity="error",
        summary="A #tile.epilogue activation is not a supported portable activation.",
        fix_hint="Use none, relu, gelu, or silu, or add an explicitly verified lowering.",
        spec="docs/architecture/proposals/tile_fragment_abi.md",
        sprint="Portable Tile fragment ABI",
    ),
    DiagnosticCode(
        code="TILE_EPILOGUE_BAD_OUTPUT", pass_origin="TileEpilogueAttr",
        severity="error",
        summary="A #tile.epilogue output dtype is not f32, f16, or i32.",
        fix_hint="Use output=f32, output=f16, or output=i32.",
        spec="docs/architecture/proposals/tile_fragment_abi.md",
        sprint="Portable Tile fragment ABI",
    ),
    DiagnosticCode(
        code="TILE_MEMORY_LAYOUT_BAD_LEADING_DIM",
        pass_origin="TileMemoryLayoutAttr", severity="error",
        summary="A #tile.memory_layout leading dimension is less than one.",
        fix_hint="Set leading_dim to the positive physical stride in elements.",
        spec="docs/architecture/proposals/tile_fragment_abi.md",
        sprint="Portable Tile fragment ABI",
    ),
    DiagnosticCode(
        code="TILE_MEMORY_LAYOUT_BAD_ORDER", pass_origin="TileMemoryLayoutAttr",
        severity="error",
        summary="A #tile.memory_layout order is not row_major or col_major.",
        fix_hint="Use row_major or col_major and let the backend own register packing.",
        spec="docs/architecture/proposals/tile_fragment_abi.md",
        sprint="Portable Tile fragment ABI",
    ),
    DiagnosticCode(
        code="TILE_MEMORY_LAYOUT_BAD_SPACE", pass_origin="TileMemoryLayoutAttr",
        severity="error",
        summary="A #tile.memory_layout space is not gmem, smem, or lds.",
        fix_hint="Use gmem, NVIDIA smem, or AMD lds for a materializable tile view.",
        spec="docs/architecture/proposals/tile_fragment_abi.md",
        sprint="Portable Tile fragment ABI",
    ),
    DiagnosticCode(
        code="TILE_MMA_DESC_BAD_FAMILY", pass_origin="TileMmaDescAttr",
        severity="error",
        summary="A #tile.mma_desc names an unknown matrix-instruction family.",
        fix_hint="Use auto, mma_sync, wgmma, tcgen05, wmma, or mfma.",
        spec="docs/architecture/proposals/tile_fragment_abi.md",
        sprint="Portable Tile fragment ABI",
    ),
    DiagnosticCode(
        code="TILE_MMA_DESC_BAD_K_BLOCKS", pass_origin="TileMmaDescAttr",
        severity="error",
        summary="A #tile.mma_desc has fewer than one semantic K block.",
        fix_hint="Set k_blocks >= 1.",
        spec="docs/architecture/proposals/tile_fragment_abi.md",
        sprint="Portable Tile fragment ABI",
    ),
    DiagnosticCode(
        code="TILE_MMA_DESC_BAD_LAYOUT", pass_origin="TileMmaDescAttr",
        severity="error",
        summary="A #tile.mma_desc A or B layout is not row_major or col_major.",
        fix_hint="Use row_major or col_major for each logical multiplicand.",
        spec="docs/architecture/proposals/tile_fragment_abi.md",
        sprint="Portable Tile fragment ABI",
    ),
    DiagnosticCode(
        code="TILE_MMA_DESC_EMPTY_DTYPE", pass_origin="TileMmaDescAttr",
        severity="error",
        summary="A #tile.mma_desc omits an A, B, or accumulator dtype.",
        fix_hint="Name all three dtypes explicitly in the MMA descriptor.",
        spec="docs/architecture/proposals/tile_fragment_abi.md",
        sprint="Portable Tile fragment ABI",
    ),
    DiagnosticCode(
        code="TILE_MMA_DESC_NONPOSITIVE_SHAPE", pass_origin="TileMmaDescAttr",
        severity="error",
        summary="A #tile.mma_desc M, N, or K instruction extent is nonpositive.",
        fix_hint="Use positive instruction extents matching a target capability.",
        spec="docs/architecture/proposals/tile_fragment_abi.md",
        sprint="Portable Tile fragment ABI",
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
        code="TILE_PIPELINE_LEGACY_METADATA", pass_origin="TilePipelineLegality",
        severity="error",
        summary="Annotation-only #tile.pipeline_state metadata remains after the SSA pipeline migration.",
        fix_hint="Thread !tile.pipeline_state values from tile.pipeline_init through tile.pipeline_advance.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C3", sprint="C3 (TIRx)",
    ),
    DiagnosticCode(
        code="TILE_PIPELINE_BARRIER_KIND_MISMATCH", pass_origin="TilePipelineLegality",
        severity="error",
        summary="One tile.barrier_id is used with two different #tile.barrier kinds.",
        fix_hint="Keep one completion semantics (kind) per barrier id.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C3", sprint="C3 (TIRx)",
    ),
    # APPLE-PIPE-1 — AppleThreadgroupPipelinePass. Apple's declared position on
    # the shared Tile physical-allocation / staged-pipeline SSA vocabulary.
    DiagnosticCode(
        code="APPLE_THREADGROUP_SPACE_UNSUPPORTED", pass_origin="AppleThreadgroupPipeline",
        severity="error",
        summary="apple_gpu places only 'smem' Tile allocations in Metal threadgroup memory; 'tmem'/'gmem' have no threadgroup realization.",
        fix_hint="Allocate staged operands with space=\"smem\"; tensor memory is NVIDIA-only and device memory is not this pass's contract.",
        spec="docs/audit/backend/apple/todo.md APPLE-PIPE-1", sprint="APPLE-PIPE-1",
    ),
    DiagnosticCode(
        code="APPLE_THREADGROUP_MEMORY_EXCEEDED", pass_origin="AppleThreadgroupPipeline",
        severity="error",
        summary="The function's placed threadgroup arena exceeds the Apple target's threadgroup-memory capacity.",
        fix_hint="Shrink the staged tile, drop double buffering, or raise threadgroup-capacity-bytes to the probed device limit.",
        spec="docs/audit/backend/apple/todo.md APPLE-PIPE-1", sprint="APPLE-PIPE-1",
    ),
    DiagnosticCode(
        code="E_PIPE_LAYOUT_MISMATCH", pass_origin="AppleMslMaterializer",
        severity="error",
        summary="The compiler-provided Apple staging-byte contract disagrees with the requested simdgroup GEMM tile.",
        fix_hint="Regenerate the descriptor from the canonical Tile loop; do not substitute a runtime-default staging layout.",
        spec="docs/audit/backend/apple/todo.md APPLE-PIPE-1", sprint="APPLE-PIPE-1", language="python",
    ),
    DiagnosticCode(
        code="APPLE_THREADGROUP_MALFORMED_ALLOC", pass_origin="AppleThreadgroupPipeline",
        severity="error",
        summary="A tile.alloc reached the Apple placer without the 'space' and 'bytes' attributes placement requires.",
        fix_hint="Emit tile.alloc with both space and bytes; placement is not inferred from the layout attribute.",
        spec="docs/audit/backend/apple/todo.md APPLE-PIPE-1", sprint="APPLE-PIPE-1",
    ),
    DiagnosticCode(
        code="APPLE_THREADGROUP_LEGACY_METADATA", pass_origin="AppleThreadgroupPipeline",
        severity="error",
        summary="Name-based #tile.buffer_ref identity is not an Apple physical allocation.",
        fix_hint="Thread !tile.buffer from tile.alloc so Metal placement follows SSA def-use identity.",
        spec="docs/audit/backend/apple/todo.md APPLE-PIPE-1", sprint="APPLE-PIPE-1",
    ),
    DiagnosticCode(
        code="APPLE_THREADGROUP_INVALID_OPTION", pass_origin="AppleThreadgroupPipeline",
        severity="error",
        summary="threadgroup-capacity-bytes or max-stage-depth was given a nonpositive value.",
        fix_hint="Pass positive values; a zero capacity would reject every allocation for the wrong reason.",
        spec="docs/audit/backend/apple/todo.md APPLE-PIPE-1", sprint="APPLE-PIPE-1",
    ),
    DiagnosticCode(
        code="APPLE_STAGE_DEPTH_UNSUPPORTED", pass_origin="AppleThreadgroupPipeline",
        severity="error",
        summary="A staged producer/consumer ring is deeper than the Metal ping-pong pair can realize.",
        fix_hint="Use depth 1 (single) or 2 (ping-pong); silently narrowing a deeper ring would change the program's synchronization structure.",
        spec="docs/audit/backend/apple/todo.md APPLE-PIPE-1", sprint="APPLE-PIPE-1",
    ),
    DiagnosticCode(
        code="APPLE_STAGE_MALFORMED_INIT", pass_origin="AppleThreadgroupPipeline",
        severity="error",
        summary="A tile.pipeline_init reached the Apple staging claim without its 'depth' and 'role' attributes.",
        fix_hint="Emit tile.pipeline_init with depth and role; the buffering mode is derived from depth, never assumed.",
        spec="docs/audit/backend/apple/todo.md APPLE-PIPE-1", sprint="APPLE-PIPE-1",
    ),
    DiagnosticCode(
        code="APPLE_STAGE_UNROOTED_ADVANCE", pass_origin="AppleThreadgroupPipeline",
        severity="error",
        summary="A tile.pipeline_advance carries no !tile.pipeline_state operand traced to an Apple-claimed tile.pipeline_init.",
        fix_hint="Thread the pipeline state value from its init through every advance so ordering is an SSA edge.",
        spec="docs/audit/backend/apple/todo.md APPLE-PIPE-1", sprint="APPLE-PIPE-1",
    ),
    DiagnosticCode(
        code="APPLE_MMA_STORAGE_UNSUPPORTED", pass_origin="AppleThreadgroupPipeline",
        severity="error",
        summary="A cooperative-matrix descriptor claims a storage dtype apple_gpu has no simdgroup_matrix route for.",
        fix_hint="Use fp16/bf16 storage with fp32 accumulation; FP8/FP4/MX MTLTensor dtypes are macOS-27 SDK-gated (APPLE-DTYPE-1).",
        spec="docs/audit/backend/apple/todo.md APPLE-DTYPE-1-REJECT", sprint="APPLE-DTYPE-1-REJECT",
    ),
    DiagnosticCode(
        code="APPLE_TILE_UNSUPPORTED_VOCABULARY", pass_origin="AppleThreadgroupPipeline",
        severity="error",
        summary="NVIDIA physical Tile vocabulary (TMA descriptor, mbarrier, tensor memory, TCGen05 MMA) reached the Apple pipeline; Metal has no equivalent.",
        fix_hint="Select an Apple-owned schedule; threadgroup_barrier and ping-pong staging are not substitutes for a copy engine or transaction barrier.",
        spec="docs/audit/backend/apple/todo.md APPLE-PIPE-1", sprint="APPLE-PIPE-1",
    ),
    # APPLE-TILE-2 — CanonicalGemmToAppleGPUPass. Apple's consumer of the shared
    # canonical M/N/K GEMM reduction contract.
    DiagnosticCode(
        code="APPLE_CANONICAL_GEMM_UNRECOGNIZED", pass_origin="CanonicalGemmToAppleGPU",
        severity="error",
        summary="A canonical K step does not slice two loop-invariant operands, so it is not a single dense contraction.",
        fix_hint="Hoist the contracted operands out of the nest, or leave the reduction on a non-GEMM lowering path.",
        spec="docs/audit/backend/apple/todo.md APPLE-TILE-2", sprint="APPLE-TILE-2",
    ),
    DiagnosticCode(
        code="APPLE_CANONICAL_GEMM_SHAPE_UNSUPPORTED", pass_origin="CanonicalGemmToAppleGPU",
        severity="error",
        summary="The Apple simdgroup route requires static rank-2 operands and result.",
        fix_hint="Bucket-specialize the dynamic dimension, or keep the contraction on the value-mode Accelerate/MPS route.",
        spec="docs/audit/backend/apple/todo.md APPLE-TILE-2", sprint="APPLE-TILE-2",
    ),
    DiagnosticCode(
        code="APPLE_CANONICAL_GEMM_DTYPE_UNSUPPORTED", pass_origin="CanonicalGemmToAppleGPU",
        severity="error",
        summary="apple_gpu re-forms the canonical reduction only for fp16/bf16 storage; simdgroup_matrix has no f32 operand form.",
        fix_hint="Leave f32 contractions on the incumbent Accelerate/MPS value route rather than rerouting them.",
        spec="docs/audit/backend/apple/todo.md APPLE-TILE-2", sprint="APPLE-TILE-2",
    ),
    DiagnosticCode(
        code="APPLE_CANONICAL_GEMM_ACCUM_UNSUPPORTED", pass_origin="CanonicalGemmToAppleGPU",
        severity="error",
        summary="The canonical reduction must accumulate in fp32 for the Apple simdgroup route.",
        fix_hint="Keep numeric_policy.accum = fp32; reduced-precision accumulation is not a simdgroup_matrix contract.",
        spec="docs/audit/backend/apple/todo.md APPLE-TILE-2", sprint="APPLE-TILE-2",
    ),
    # APPLE-ATTN-STREAM-1 — StreamingAttentionToAppleGPUPass.
    DiagnosticCode(
        code="APPLE_STREAMING_ATTN_UNRECOGNIZED", pass_origin="StreamingAttentionToAppleGPU",
        severity="error",
        summary="A marked streaming-attention loop lacks the shared score/update ops or its Q/K/V are not loop-invariant.",
        fix_hint="Keep the shared tessera_attn.scaled_dot_product / streaming_update shape, or drop the tessera.streaming_attention marker.",
        spec="docs/audit/backend/apple/todo.md APPLE-ATTN-STREAM-1", sprint="APPLE-ATTN-STREAM-1",
    ),
    DiagnosticCode(
        code="APPLE_STREAMING_ATTN_LSE_UNSUPPORTED", pass_origin="StreamingAttentionToAppleGPU",
        severity="error",
        summary="The program retains the per-row log-sum-exp, but the Apple fused flash-attention ABI returns the output only.",
        fix_hint="Extend the Apple fused ABI to return LSE, or elide lse.save when the LSE is provably dead; re-forming anyway would recompute the recurrence or drop a backward checkpoint.",
        spec="docs/audit/backend/apple/todo.md APPLE-ATTN-STREAM-1", sprint="APPLE-ATTN-STREAM-1",
    ),
    DiagnosticCode(
        code="APPLE_STREAMING_ATTN_DROPOUT_UNSUPPORTED", pass_origin="StreamingAttentionToAppleGPU",
        severity="error",
        summary="apple_gpu has no fused offset-keyed block-dropout attention route.",
        fix_hint="Run the dropout variant on the decomposed path; claiming the loop would silently drop the mask.",
        spec="docs/audit/backend/apple/todo.md APPLE-ATTN-STREAM-1", sprint="APPLE-ATTN-STREAM-1",
    ),
    DiagnosticCode(
        code="APPLE_STREAMING_ATTN_SHAPE_UNSUPPORTED", pass_origin="StreamingAttentionToAppleGPU",
        severity="error",
        summary="The Apple fused attention route requires a static rank-2 Q.",
        fix_hint="Bucket-specialize the dynamic dimension before the Apple consumer runs.",
        spec="docs/audit/backend/apple/todo.md APPLE-ATTN-STREAM-1", sprint="APPLE-ATTN-STREAM-1",
    ),
    DiagnosticCode(
        code="APPLE_STREAMING_ATTN_HEAD_DIM_UNSUPPORTED", pass_origin="StreamingAttentionToAppleGPU",
        severity="error",
        summary="head_dim exceeds the Apple fused-chain score cap derived from the 1 KiB per-thread fp32 stack budget.",
        fix_hint="Split the head dimension, or use the decomposed attention path.",
        spec="docs/audit/backend/apple/todo.md APPLE-ATTN-STREAM-1", sprint="APPLE-ATTN-STREAM-1",
    ),
    DiagnosticCode(
        code="APPLE_STREAMING_ATTN_DTYPE_UNSUPPORTED", pass_origin="StreamingAttentionToAppleGPU",
        severity="error",
        summary="The Apple fused flash-attention route accepts f32/f16/bf16 storage.",
        fix_hint="Convert storage at the boundary, or keep the recurrence on the decomposed path.",
        spec="docs/audit/backend/apple/todo.md APPLE-ATTN-STREAM-1", sprint="APPLE-ATTN-STREAM-1",
    ),
    DiagnosticCode(
        code="APPLE_STREAMING_ATTN_RANK4_MODIFIER_UNSUPPORTED", pass_origin="StreamingAttentionToAppleGPU",
        severity="error",
        summary="The rank-4 Apple GQA ABI carries causal and scale but not sliding windows, score bias, or softcap.",
        fix_hint="Keep modifier-bearing rank-4 attention on the shared decomposed path until the Apple GQA ABI carries every requested modifier.",
        spec="docs/audit/backend/apple/todo.md APPLE-ATTN-STREAM-2", sprint="APPLE-ATTN-STREAM-2",
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
    DiagnosticCode(
        code="DTYPE_PACK_SIGNEDNESS_MISMATCH", pass_origin="GenerateWMMAGemmKernel",
        severity="error",
        summary="A gfx1151 int4 WMMA request does not declare signed two's-complement packed storage.",
        fix_hint="Set tessera.storage_pack.signedness to signed_twos_complement; unsigned IU4 requires a distinct dtype and ABI contract.",
        spec="docs/reference/tessera_tensor_attributes.md#canonical-dtypes",
        sprint="ROCM-DTYPE-1",
    ),
    # C4 part 1 (2026-06-23) — the storage-pack consumer (StoragePackConsume).
    DiagnosticCode(
        code="DTYPE_PACK_BAD_WIDTHS", pass_origin="StoragePackConsume",
        severity="error",
        summary="A storage_packed op's logical storage cannot pack into its container (unknown dtype, or storage wider than the container).",
        fix_hint="Mark sub-byte storage (fp4/nvfp4/fp6/int4) with a wider byte container (int8); storage bits must be <= container bits.",
        spec="docs/audit/compiler/COMPILER_AUDIT.md §C4", sprint="C4 (TIRx)",
    ),
    DiagnosticCode(
        code="REMAT_EFFECTFUL",
        pass_origin="ActivationRematerializationPass",
        severity="error",
        summary="An operation explicitly selected for rematerialization is not provably side-effect-free.",
        fix_hint="Remove the recompute marker or provide a pure, region-free activation producer; RNG, collectives, stores, and unknown-effect operations cannot be replayed.",
        spec="docs/spec/AUTODIFF_SPEC.md §Phase F2",
        sprint="CORE-COMPILER-CFG-MEMORY-BUDGETS-2026-07-24",
    ),
    DiagnosticCode(
        code="REMAT_MODEL_BUDGET_INVALID",
        pass_origin="ActivationRematerializationPass",
        severity="error",
        summary="The device/model memory envelope cannot be converted into a safe signed-i64 activation budget.",
        fix_hint="Use non-negative capacity/state inputs, reserve basis points in [0, 10000], and explicit byte bounds for dynamic model parameters.",
        spec="docs/spec/AUTODIFF_SPEC.md §Phase F2",
        sprint="CORE-COMPILER-CFG-MEMORY-BUDGETS-2026-07-24",
    ),
    DiagnosticCode(
        code="REMAT_NON_CLONABLE",
        pass_origin="ActivationRematerializationPass",
        severity="error",
        summary="An operation explicitly selected for rematerialization owns nested regions and cannot be cloned safely at its consumers.",
        fix_hint="Select pure region-free activation producers, or first outline/canonicalize the nested control flow into a replay-safe operation.",
        spec="docs/spec/AUTODIFF_SPEC.md §Phase F2",
        sprint="CORE-COMPILER-CFG-MEMORY-BUDGETS-2026-07-24",
    ),
    DiagnosticCode(
        code="ROCM_DYNAMIC_LDS_SIZE_NOT_KERNEL_ARGUMENT",
        pass_origin="ROCMDynamicLDS",
        severity="error",
        summary="A runtime LDS arena size is neither a same-typed LLVM kernel argument nor a CFG block argument forwarding kernel-argument leaves.",
        fix_hint="Hoist the guarded byte-size expression into the kernel launch ABI; arbitrary local SSA arithmetic requires a future serializable expression carrier.",
        spec="docs/audit/compiler/COMPILER_THEORY_OF_OPERATION.md §W3",
        sprint="CORE-COMPILER-CFG-MEMORY-BUDGETS-2026-07-24",
    ),

    # ROCm shared Tile-IR convergence (2026-06-23) — AMD consumes the shared
    # Tile contract but keeps LDS / waitcnt legality target-specific.
    DiagnosticCode(
        code="ROCM_FRAGMENT_ILLEGAL_ARCH_DESCRIPTOR", pass_origin="LowerTileToROCMPass",
        severity="error",
        summary="No exact RDNA3, RDNA4, gfx125x, or CDNA physical fragment descriptor accepts the requested Tile MMA family, dtype, shape, and layouts.",
        fix_hint="Select a matrix family, dtype, shape, and logical A/B layout supported by the exact gfx architecture; do not reuse another architecture's fragment ABI.",
        spec="docs/audit/backend/rocm/todo.md §ROCM-5", sprint="ROCM-5",
    ),
    DiagnosticCode(
        code="ROCM_FRAGMENT_TYPE_DISAGREES", pass_origin="LowerTileToROCMPass",
        severity="error",
        summary="The materialized fragment value's type differs from the type the Tile fragment type converter promised.",
        fix_hint="Derive both the pack materialization and the converted fragment type from packedFragmentType(); sub-16-bit inputs pack into i32 registers rather than staying element-typed.",
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md §4.6", sprint="W1.1",
    ),
    DiagnosticCode(
        code="ROCM_FRAGMENT_UNPACK_UNCONSUMED", pass_origin="LowerTileToROCMPass",
        severity="error",
        summary="A typed tile.fragment_unpack has no single tile.store consumer, so its accumulator has no physical form on this target.",
        fix_hint="Consume the unpacked accumulator with exactly one tile.store; the !tile.tile it yields only becomes a physical value at the store.",
        spec="docs/audit/compiler/W1_1_TYPING_DESIGN.md §4.6", sprint="W1.1",
    ),
    DiagnosticCode(
        code="ROCM_FRAGMENT_MATERIALIZATION_GATED", pass_origin="LowerTileToROCMPass",
        severity="error",
        summary="The exact architecture recognizes the matrix instruction ABI, but Tessera has not enabled its physical fragment pack/unpack map.",
        fix_hint="Keep the dtype gated until its architecture-specific packing map, real intrinsic lowering, exact-target assembly, and numerical oracle are implemented.",
        spec="docs/audit/backend/rocm/todo.md §ROCM-5", sprint="ROCM-5",
    ),
    DiagnosticCode(
        code="ROCM_FRAGMENT_MISSING_CONTRACT", pass_origin="LowerTileToROCMPass",
        severity="error",
        summary="ROCm fragment materialization is missing a Tile view, role, or MMA descriptor required to resolve the physical ABI.",
        fix_hint="Feed fragment_pack from tile.view and attach matching role and #tile.mma_desc attributes to every fragment operation.",
        spec="docs/architecture/proposals/tile_fragment_abi.md", sprint="ROCM-5",
    ),
    DiagnosticCode(
        code="ROCM_FRAGMENT_SOURCE_RANK", pass_origin="LowerTileToROCMPass",
        severity="error",
        summary="The current ROCm fragment materializer received a source that is not a rank-one memref view.",
        fix_hint="Flatten the backing allocation to the supported rank-one memref ABI and express matrix coordinates through tile.view origins and memory layout.",
        spec="docs/architecture/proposals/tile_fragment_abi.md", sprint="ROCM-5",
    ),
    DiagnosticCode(
        code="ROCM_FRAGMENT_SOURCE_TYPE", pass_origin="LowerTileToROCMPass",
        severity="error",
        summary="The fragment source element type disagrees with the exact architecture MMA descriptor.",
        fix_hint="Make the source memref element type match the descriptor's A/B dtype and use an explicit storage-pack conversion for sub-byte formats.",
        spec="docs/architecture/proposals/tile_fragment_abi.md", sprint="ROCM-5",
    ),
    DiagnosticCode(
        code="ROCM_FRAGMENT_STORE_LAYOUT", pass_origin="LowerTileToROCMPass",
        severity="error",
        summary="ROCm fragment unpack/store received an unsupported output layout or memory order.",
        fix_hint="Use an unswizzled 16x16 row-major global-memory output layout, or implement and test a new architecture-specific accumulator map.",
        spec="docs/architecture/proposals/tile_fragment_abi.md", sprint="ROCM-5",
    ),
    DiagnosticCode(
        code="ROCM_FRAGMENT_STORE_TYPE", pass_origin="LowerTileToROCMPass",
        severity="error",
        summary="The fragment store destination type does not match the selected f32 or i32 accumulator contract.",
        fix_hint="Store floating matrix results to rank-one f32 memrefs and integer WMMA results to rank-one i32 memrefs, or add an explicit epilogue conversion.",
        spec="docs/architecture/proposals/tile_fragment_abi.md", sprint="ROCM-5",
    ),
    DiagnosticCode(
        code="ROCM_FRAGMENT_UNSUPPORTED_SOURCE_LAYOUT", pass_origin="LowerTileToROCMPass",
        severity="error",
        summary="The Tile source memory order, shard shape, space, or swizzle cannot be packed by the selected architecture fragment map.",
        fix_hint="Use the supported row-major A and column-major B global-memory views with the exact descriptor shape, or implement a tested layout transform.",
        spec="docs/architecture/proposals/tile_fragment_abi.md", sprint="ROCM-5",
    ),
    DiagnosticCode(
        code="ROCM_LOWERING_LAYOUT_NOT_LDS", pass_origin="LowerTileToROCMPass",
        severity="error",
        summary="ROCm lowering saw a #tile.layout on tile.async_copy that does not place storage on the lds axis.",
        fix_hint="Use #tile.layout with an lds shard axis for ROCm global-to-LDS movement, or omit layout when unknown.",
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
        code="ROCM_WAVE_LDS_ROLE_UNRESOLVED",
        pass_origin="ROCMWaveLdsLegalityPass",
        severity="error",
        summary=(
            "A structured ROCm LDS operation has no pipeline state rooted in "
            "a matching producer/consumer tile.role SSA declaration."
        ),
        fix_hint=(
            "Run rocm-wave-lds-pipeline and preserve the role-bearing "
            "tile.pipeline_init/advance chain into the physical consumer."
        ),
        spec="docs/audit/compiler/compiler_enhancement.md §6.2–§6.4",
        sprint="LAYOUT-SCHEDULE-OBJECT-2026-08-16",
    ),
    DiagnosticCode(
        code="ROCM_TILE_UNSUPPORTED_DTYPE", pass_origin="LowerTileToROCMPass",
        severity="error",
        summary="A tile.mma on gfx1151 (RDNA 3.5) requested an FP8/BF8 matrix form, which RDNA 3.5 WMMA does not provide.",
        fix_hint="Use f16, bf16, int8, or int4 for the gfx1151 WMMA path, or select an exact RDNA 4 / CDNA target that has FP8/BF8 matrix instructions.",
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

    # ── GRAPH_IR_* — the Python Graph IR verifier ──────────────────────────
    #
    # W1.3 (2026-08-03). Review asked for the new
    # `GRAPH_IR_SSA_VALUE_IN_ATTRIBUTE` to be registered here, and the gap
    # turned out to be family-wide: NONE of the codes emitted by
    # `GraphIRModule.verify()` were registered. Adding one and leaving eight
    # siblings undiscoverable would reproduce the asymmetry Decision #29 is
    # about, so the whole family lands together.
    DiagnosticCode(
        code="GRAPH_IR_SSA_VALUE_IN_ATTRIBUTE", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="An attribute holds an SSA name that is not also an operand — a dataflow edge hidden in an attribute.",
        fix_hint="Declare the parameter in graph_ir._KEYWORD_OPERANDS so it emits as an operand; an attribute is not a value edge.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="W1.3",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="GRAPH_IR_UNRESOLVED_OPERAND", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="An op carries the placeholder operand `%?` — the frontend could not lower that argument to a value.",
        fix_hint="If the argument is an attribute rather than a tensor, declare it in graph_ir._POSITIONAL_ATTR_PARAMS.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="GRAPH_IR_UNDEFINED_OPERAND", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="An op reads an SSA value that no argument or earlier result defines.",
        fix_hint="Check that the producing op is emitted before this use and that its result name matches.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="GRAPH_IR_OPERAND_TYPE_MISMATCH", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="An op's operand count disagrees with its operand-type count.",
        fix_hint="Emit one type per operand; appending an operand without its type is the usual cause.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="GRAPH_IR_RETURN_MISSING", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="A function declares result types but returns no values.",
        fix_hint="Emit a return with one value per declared result type.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="GRAPH_IR_RETURN_ARITY", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="A function returns a different number of values than it declares results.",
        fix_hint="For a multi-result op use graph_ir._infer_result_types, which states the full contract.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="GRAPH_IR_RETURN_UNDEFINED", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="A function returns an SSA value that is never defined in its body.",
        fix_hint="Check the returned name against the emitted result names.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="GRAPH_IR_CONTROL_UNBALANCED", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="A control region (if / for / while) was opened and never closed.",
        fix_hint="Emit the matching region terminator; a lowering path that returns early is the usual cause.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="GRAPH_IR_CONTROL_ELSE", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="An `else` region appears without an enclosing `if`.",
        fix_hint="Emit the `if` region before its `else`.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="GRAPH_IR_DUP_FUNC", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="Two functions in one module share a name.",
        fix_hint="Rename one; module-level function names must be unique.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="GRAPH_IR_DUP_ARG", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="Two arguments of one function share a name.",
        fix_hint="Rename one; argument names form the initial SSA scope.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="GRAPH_IR_DUP_VALUE", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="An SSA result name is assigned twice — SSA values are single-assignment.",
        fix_hint="Allocate a fresh result name; reusing one silently rebinds later uses.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="GRAPH_IR_DUP_MESH", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="Two mesh declarations in one module share a name.",
        fix_hint="Rename one; mesh names are module-scoped.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="GRAPH_IR_DUP_TYPE_ALIAS", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="Two type aliases in one module share a name.",
        fix_hint="Rename one; alias names are module-scoped.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="GRAPH_IR_DUP_CONSTANT", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="Two module-level constants share a name.",
        fix_hint="Rename one; constant names are module-scoped.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="GRAPH_IR_MATMUL_SHAPE", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="A matmul's K dimensions disagree between its two operands.",
        fix_hint="Check the contracted dimension; lhs[1] must equal rhs[0].",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="GRAPH_IR_MESH_RANK", pass_origin="GraphIRModule.verify",
        severity="error",
        summary="A mesh declaration's axis count disagrees with its shape rank.",
        fix_hint="Give one extent per named axis in the mesh declaration.",
        spec="docs/spec/GRAPH_IR_SPEC.md", sprint="Phase 2",
        language="python", status="implemented",
    ),
    DiagnosticCode(
        code="NEIGHBORS_TOPOLOGY_UNKNOWN_KIND",
        pass_origin="CreateTopologyOp::verify",
        severity="error",
        summary="A topology.create operation names an unregistered topology kind.",
        fix_hint=(
            "Use 2d_mesh, 3d_mesh, hex_2d, custom_graph, dynamic, adaptive, or fault."
        ),
        spec="docs/audit/compiler/INTEGRATED_COMPILER_PLAN.md",
        sprint="W1.1b",
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
