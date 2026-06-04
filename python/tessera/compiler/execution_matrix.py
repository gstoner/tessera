"""G4 — single-source runtime execution matrix.

Before this module, three places each had their own answer to "given an artifact,
what does the runtime actually do with it?":

- `capabilities.py` knew the per-target / per-op compile-time status
  (`ready` / `artifact_only` / `unimplemented`).
- `runtime.launch()` had a chain of hard-coded `target == "apple_cpu" and ...`,
  `target == "apple_gpu" and ...`, and `target != "cpu" -> unimplemented` branches.
- The docs / dashboards described it in prose.

They could drift. This module is the **one place** that maps a
``(target, compiler_path)`` pair to a structured `ExecutionRow`. The row tells
``launch()`` *which* executor to call (when any), what telemetry strings to use,
and what to return when no executor exists. ``capabilities.py`` consults the same
table to know which (target, compiler_path) pairs have a real runtime executor
backing the compile-time status. A generated dashboard renders the table for
humans; a drift test fails if anything diverges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from .capabilities import TARGET_CAPABILITIES, normalize_target


# An executor takes (artifact, args) and returns the op output. Resolved lazily
# from runtime.py via name to avoid an import cycle (runtime imports this module).
EXECUTOR_ID = str
EXECUTOR_FN = Callable[..., object]


@dataclass(frozen=True)
class ExecutionRow:
    """One row of the execution matrix.

    A `(target, compiler_path)` pair resolves to **exactly one** row. The row is
    the runtime's contract: it names the executor (if any), the labels to use in
    telemetry + the result dict, and a precise reason when no executor exists.
    """

    target: str               # canonical target name (matches TARGET_CAPABILITIES)
    compiler_path: str        # e.g. "apple_cpu_accelerate", "apple_gpu_mps",
                              # "jit_cpu_numpy", "native_cpu", "artifact_only"
    execution_kind: str       # telemetry label: "native_cpu" / "native_gpu" /
                              # "reference_cpu" / "cpu_accelerate" / "artifact_only"
    executable: bool          # True iff there's a real executor function below
    executor_id: Optional[EXECUTOR_ID]   # symbolic name resolved at launch time
    runtime_status: str       # what to report when there's no executor:
                              # "unimplemented" / "missing_backend" / etc.
    reason: str = ""          # human-readable explanation for telemetry / errors
    execution_mode: str = ""  # telemetry-only: "metal_runtime" / "cpu_accelerate" / ""


# Catalog of every executor name → docstring describing what it runs. The actual
# functions live in `runtime.py`; this module deliberately does NOT import
# runtime.py (avoid the cycle — runtime.py imports `execution_matrix`).
KNOWN_EXECUTORS: dict[EXECUTOR_ID, str] = {
    "apple_cpu_accelerate": "Apple Silicon CPU via the Accelerate cblas_sgemm shim",
    "apple_gpu_mps":        "Apple Silicon GPU via MPS / MSL / MPSGraph (per envelope)",
    "apple_value_target_ir": "Apple CPU value-call dispatch — invokes the C ABI "
                             "symbol named in a tessera_apple.cpu.call value op "
                             "(Value Target IR sprint; CPU cholesky executable)",
    "apple_gpu_value_target_ir": "Apple GPU value-call dispatch — invokes the C "
                             "ABI symbol named in a tessera_apple.gpu.kernel_call "
                             "value op (rank-3 batched matmul f32/f16/bf16; "
                             "native sparse attention and PPO policy-loss variants "
                             "plus EBM quadratic energy/Langevin value kernels "
                             "when their Metal/MPSGraph executor probes are active)",
    "native_cpu":           "x86 AMX / native CPU runtime via the C runtime ABI",
    "jit_cpu_numpy":        "JIT CPU fallback via the numpy reference path",
    # Note: pure-numpy `reference_cpu` is reached only as an internal *fallback*
    # inside `launch()`'s native_cpu branch (when `_execute_native_cpu_artifact`
    # raises and `_execute_jit_cpu_artifact` succeeds). It's not a directly
    # dispatched executor — no matrix row points at it — so it's intentionally
    # not in this catalog (the drift test would flag dead entries otherwise).
}


# The execution matrix itself: (target, compiler_path) -> ExecutionRow. Adding a
# new backend executor means (1) adding the function in runtime.py, (2) adding it
# to KNOWN_EXECUTORS, (3) adding an ExecutionRow here. `launch()` picks it up
# automatically; the dashboard regenerates; the drift test enforces it.
_MATRIX: dict[tuple[str, str], ExecutionRow] = {
    # --- Apple Silicon CPU (Accelerate) ---
    ("apple_cpu", "apple_cpu_accelerate"): ExecutionRow(
        target="apple_cpu", compiler_path="apple_cpu_accelerate",
        execution_kind="native_cpu", executable=True,
        executor_id="apple_cpu_accelerate", runtime_status="success",
        reason="Apple CPU artifact runs through Accelerate cblas_sgemm + multi-op chain.",
        execution_mode="cpu_accelerate"),
    # --- Apple Silicon GPU (MPS / MSL / MPSGraph) ---
    ("apple_gpu", "apple_gpu_mps"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_gpu_mps",
        execution_kind="native_gpu", executable=True,
        executor_id="apple_gpu_mps", runtime_status="success",
        reason="Apple GPU artifact runs through MPS / MSL / MPSGraph per the runtime envelope.",
        execution_mode="metal_runtime"),
    # --- Apple Value Target IR (sprint 2) — CPU value-call execution ---
    # The value-preserving `-full` lane lowers to tessera_apple.cpu.call value
    # ops; this row executes them by invoking the C ABI `symbol` named in the IR
    # (read from metadata["apple_value_calls"]). CPU cholesky is executable now.
    ("apple_cpu", "apple_value_target_ir"): ExecutionRow(
        target="apple_cpu", compiler_path="apple_value_target_ir",
        execution_kind="native_cpu", executable=True,
        executor_id="apple_value_target_ir", runtime_status="success",
        reason="Apple CPU value-call (tessera_apple.cpu.call) dispatches to the "
               "named Accelerate/LAPACK C ABI symbol.",
        execution_mode="cpu_accelerate"),
    # Apple GPU value-call execution for narrow, explicitly allowlisted lanes:
    # rank-3 batched matmul (Sprint 8), native sparse attention (Sprint 11),
    # PPO policy loss (Stages 13/14), and the first EBM value kernels. The
    # executor rejects cpu.call, package_call, multi-op programs, inactive
    # stubs, and off-allowlist symbols.
    ("apple_gpu", "apple_value_target_ir"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_value_target_ir",
        execution_kind="native_gpu", executable=True,
        executor_id="apple_gpu_value_target_ir", runtime_status="success",
        reason="Apple GPU value-call (tessera_apple.gpu.kernel_call) dispatches "
               "named C ABI symbols for strict rank-3 batched matmul, native "
               "sparse attention, PPO policy-loss, and EBM value envelopes.",
        execution_mode="metal_runtime"),
    # --- x86 / native CPU (AMX path) ---
    ("cpu", "native_cpu"): ExecutionRow(
        target="cpu", compiler_path="native_cpu",
        execution_kind="native_cpu", executable=True,
        executor_id="native_cpu", runtime_status="success",
        reason="CPU artifact runs through the x86 AMX / native CPU runtime."),
    # --- CPU JIT (numpy reference for non-AMX ops) ---
    ("cpu", "jit_cpu_numpy"): ExecutionRow(
        target="cpu", compiler_path="jit_cpu_numpy",
        execution_kind="reference_cpu", executable=True,
        executor_id="jit_cpu_numpy", runtime_status="success",
        reason="CPU JIT artifact runs through the numpy reference path."),
}


# Targets recognized by the capability registry but with NO executable runtime
# row (yet). `launch()` reports `unimplemented` (target capability present) or
# `missing_backend` (target capability absent). Listed explicitly so the drift
# test catches accidental status drift.
_UNIMPLEMENTED_TARGETS: tuple[str, ...] = (
    "nvidia_sm80", "nvidia_sm90", "nvidia_sm100", "nvidia_sm120",
    "rocm", "rocm_gfx90a", "rocm_gfx940", "rocm_gfx942", "rocm_gfx950",
    "rocm_gfx1100", "rocm_gfx1200",
    "metalium",
)


def lookup(target: str, compiler_path: str) -> Optional[ExecutionRow]:
    """The exact matrix lookup. Returns None when (target, compiler_path) isn't a
    runtime-executable pair — `launch()` then falls back to the
    target-default-status path (unimplemented / missing_backend)."""
    return _MATRIX.get((target, compiler_path))


def executor_for_metadata(metadata: Mapping[str, object]) -> Optional[ExecutionRow]:
    """The interpretation `launch()` uses: read `target` + `compiler_path` from
    an artifact's metadata and resolve the row. None if there is no executor."""
    target = str(metadata.get("target", "cpu") or "cpu")
    compiler_path = str(metadata.get("compiler_path", "") or "")
    if not compiler_path:
        # Legacy artifacts without compiler_path: fall through to the historical
        # `executable + execution_kind == native_cpu` logic in launch().
        return None
    return lookup(target, compiler_path)


def all_rows() -> list[ExecutionRow]:
    """Stable order: by (target, compiler_path) — what the dashboard renders."""
    return [_MATRIX[k] for k in sorted(_MATRIX)]


def unimplemented_targets() -> tuple[str, ...]:
    """The targets the capability registry knows about but for which no
    executable row exists; `launch()` reports unimplemented / missing_backend."""
    return _UNIMPLEMENTED_TARGETS


def render_dashboard() -> str:
    """Render the matrix as a Markdown table for `docs/audit/generated/runtime_execution_matrix.md`.
    Pure function so the drift test can compare bytes."""
    lines = [
        "# Runtime execution matrix",
        "",
        "**Generated from `tessera.compiler.execution_matrix._MATRIX` — do not hand-edit.**",
        "Regenerate with:",
        "",
        "```",
        "python3 -c 'from tessera.compiler.execution_matrix import write_dashboard; write_dashboard()'",
        "```",
        "",
        "Single source of truth for what `runtime.launch()` does with each "
        "`(target, compiler_path)` pair. `capabilities.py`, `runtime.launch()`, "
        "and this dashboard all derive from the same `_MATRIX`. The drift test "
        "`test_runtime_execution_matrix` fails if they diverge.",
        "",
        "## Executable rows",
        "",
        "| Target | Compiler path | Executor | Execution kind | Telemetry mode | Reason |",
        "|--------|---------------|----------|----------------|----------------|--------|",
    ]
    for row in all_rows():
        lines.append(
            f"| `{row.target}` | `{row.compiler_path}` | "
            f"`{row.executor_id or '-'}` | `{row.execution_kind}` | "
            f"{'`' + row.execution_mode + '`' if row.execution_mode else '-'} | "
            f"{row.reason} |"
        )
    lines += [
        "",
        "## Targets with no executable row",
        "",
        "These targets are recognized by the capability registry (so an artifact "
        "can carry them and lower correctly) but have no executable runtime row. "
        "`launch()` returns `runtime_status = \"unimplemented\"` when the target "
        "capability is present, or `\"missing_backend\"` otherwise — never silent "
        "success, never a fabricated output.",
        "",
        "```",
        ", ".join(unimplemented_targets()),
        "```",
        "",
        "## Known executor IDs",
        "",
        "| Executor ID | What it runs |",
        "|-------------|--------------|",
    ]
    for eid in sorted(KNOWN_EXECUTORS):
        lines.append(f"| `{eid}` | {KNOWN_EXECUTORS[eid]} |")
    lines.append("")
    return "\n".join(lines)


def write_dashboard() -> str:
    """Render and write the dashboard; returns the path."""
    from pathlib import Path
    p = (Path(__file__).resolve().parents[2].parent / "docs" / "audit"
         / "generated" / "runtime_execution_matrix.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_dashboard())
    return str(p)


def validate_against_capabilities() -> list[str]:
    """Cross-check: every executable row's target must exist in the capability
    registry, and every `_UNIMPLEMENTED_TARGETS` entry too. Returns a list of
    error strings (empty = OK). Used by the drift test."""
    errors: list[str] = []
    for row in all_rows():
        try:
            normalize_target(row.target)
        except ValueError:
            errors.append(f"matrix row target {row.target!r} is not in TARGET_CAPABILITIES")
        if row.executor_id is not None and row.executor_id not in KNOWN_EXECUTORS:
            errors.append(f"matrix row uses executor_id {row.executor_id!r} not in KNOWN_EXECUTORS")
    for t in unimplemented_targets():
        if t not in TARGET_CAPABILITIES:
            errors.append(f"_UNIMPLEMENTED_TARGETS entry {t!r} is not in TARGET_CAPABILITIES")
    return errors
