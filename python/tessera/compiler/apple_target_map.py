"""Apple target map — unified per-op view across apple_cpu + apple_gpu.

This module is the **single source of truth** for "is this op Apple-native,
and how do we prove it?".  It pulls from:

* ``capabilities.py`` — per-target ``OpCapability`` entries (runtime status,
  supported dtypes).
* ``backend_manifest.py`` — fused / reference / planned manifest entries with
  per-op symbol / framework / ABI / dtypes.
* ``driver.py`` — the Apple GPU dispatch routes (the ``tessera_apple_gpu_*``
  symbol table used by ``_backend_artifact_for``).
* The GA / EBM / M7 fused-kernel tables (``_CLIFFORD_APPLE_GPU_FUSED`` /
  ``_EBM_APPLE_GPU_FUSED`` / ``_COMPLEX_APPLE_GPU_FUSED``).

The rendered dashboard at ``docs/audit/generated/apple_target_map.md``
gives reviewers one table to scan for an answer to:

  * What does Apple CPU vs Apple GPU support today?
  * Which framework does each kernel use (Accelerate / BNNS / MPS / MSL)?
  * Which dispatch path (manifest vs driver) carries each op?
  * Where is the proof — which test or benchmark exercises it?

The drift gate at ``tests/unit/test_apple_target_map.py`` re-renders the
dashboard and fails CI if it diverges from the committed file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import backend_manifest as _bm


_REPO_ROOT = Path(__file__).resolve().parents[3]
_GENERATED_DOC = (
    _REPO_ROOT / "docs" / "audit" / "generated" / "apple_target_map.md"
)


# ─────────────────────────────────────────────────────────────────────
# Per-op family inventory.  Drawn from the same tables the audit walker
# uses so the map stays in lockstep with backend_manifest.
# ─────────────────────────────────────────────────────────────────────


def _generic_tensor_ops() -> tuple[str, ...]:
    """Generic-tensor ops that have any Apple coverage.

    Today: matmul (+ its fp16/bf16 lanes), softmax, softmax_safe, gelu,
    rope, flash_attn, rmsnorm — i.e., everything in
    ``backend_manifest._APPLE_GPU_KERNELS``.  Plus matmul/gemm on the
    CPU side via ``_APPLE_CPU_KERNELS``.
    """
    keys = set(_bm._APPLE_GPU_KERNELS.keys()) | set(_bm._APPLE_CPU_KERNELS.keys())
    return tuple(sorted(keys))


def _ga_ops() -> tuple[str, ...]:
    return tuple(sorted(_bm._CLIFFORD_APPLE_GPU_FUSED.keys()))


def _ebm_ops() -> tuple[str, ...]:
    return tuple(sorted(_bm._EBM_APPLE_GPU_FUSED.keys()))


def _m7_fused_ops() -> tuple[str, ...]:
    return tuple(sorted(_bm._COMPLEX_APPLE_GPU_FUSED.keys()))


# ─────────────────────────────────────────────────────────────────────
# Row schema.  Symmetric apple_cpu vs apple_gpu so the rendered table
# can be a wide grid (one column per target * field) or split into two
# halves.  We render the wide grid below.
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AppleTargetRow:
    op_name: str
    family: str
    # apple_cpu fields
    cpu_status: str
    cpu_framework: str
    cpu_dtypes: tuple[str, ...]
    cpu_symbol: str
    cpu_execution_kind: str
    # apple_gpu fields
    gpu_status: str
    gpu_framework: str
    gpu_dtypes: tuple[str, ...]
    gpu_symbol: str
    gpu_dispatch: str       # "manifest" / "driver" / "absent"
    # proof
    proof_test: str
    notes: str = ""

    def as_row(self) -> list[str]:
        """Return the markdown row cells in declaration order."""
        return [
            self.op_name, self.family,
            self.cpu_status, self.cpu_framework,
            ",".join(self.cpu_dtypes) or "-",
            self.cpu_symbol or "-",
            self.cpu_execution_kind,
            self.gpu_status, self.gpu_framework,
            ",".join(self.gpu_dtypes) or "-",
            self.gpu_symbol or "-",
            self.gpu_dispatch,
            self.proof_test or "-",
        ]


# ─────────────────────────────────────────────────────────────────────
# Per-op execution-kind table for apple_cpu — explicit so we don't
# silently re-claim every numpy-backed op as "Accelerate-native".  An
# op not listed here defaults to ``numpy_reference``.
# ─────────────────────────────────────────────────────────────────────


_APPLE_CPU_EXECUTION_KIND: dict[str, str] = {
    "matmul": "accelerate_native",      # cblas_sgemm
    "gemm":   "accelerate_native",      # alias for matmul
    # BNNS handles fp16 + bf16 matmul under the same Tessera API entry,
    # but the execution_kind axis is per-op, not per-dtype.  Per-dtype
    # detail lives in the dtype-lanes column.
}

# Map ``apple_cpu`` per-op kernel notes to the framework label we'll
# render.  Falls back to "numpy_reference" for anything without a
# fused entry.
_APPLE_CPU_FRAMEWORK_HINTS: dict[str, str] = {
    "matmul": "Accelerate (cblas_sgemm) + BNNS (fp16/bf16)",
    "gemm":   "Accelerate (cblas_sgemm) + BNNS (fp16/bf16)",
}

# Apple GPU framework labels per op.  Driver routes some ops through
# MPS and others through MSL (custom kernels) — the dashboard surfaces
# the distinction so readers can see which path each op takes.
_APPLE_GPU_FRAMEWORK_HINTS: dict[str, str] = {
    "matmul":       "MetalPerformanceShaders",
    "softmax":      "Metal (MSL)",
    "softmax_safe": "Metal (MSL)",
    "gelu":         "Metal (MSL)",
    "rope":         "Metal (MSL)",
    "flash_attn":   "Metal (MSL)",
    "rmsnorm":      "Metal (MSL)",
}

# Apple GPU dispatch route per op family.  "manifest" = goes through
# ``jit_bridge.dispatch_via_manifest`` (GA/EBM/M7 path).  "driver" =
# goes through ``compiler/driver.py::_backend_artifact_for`` (generic
# tensor path).  D-phase work will eventually unify these but we
# surface the distinction here today so the gap is visible.
_DRIVER_DISPATCH_OPS: frozenset[str] = frozenset({
    "matmul", "softmax", "softmax_safe", "gelu",
    "rope", "flash_attn", "rmsnorm",
    # Followup A.1 (2026-05-31) — manifest entries landed for these
    # three; they dispatch via the driver (MPSGraph unary opcode /
    # native multitile MPP / MPS matmul) so the drift gate routes
    # them as ``driver`` not ``manifest``.
    "relu", "conv2d", "kv_cache_read",
    # Project 2.1c / 3 (2026-06-01) — encode-session ops dispatch via
    # the driver's encode lane (MPSGraph bmm / rowop / unary opcode).
    "bmm", "layer_norm", "silu",
    # Followup (2026-06-10) — MoE expert-FFN fused kernels dispatch via the
    # runtime's lane→handler table (the driver path), like matmul.
    "grouped_gemm", "moe_swiglu_block",
})


# ─────────────────────────────────────────────────────────────────────
# Proof-test labels.  Hand-maintained but tested for filesystem
# existence by the drift gate so a stale entry breaks CI.
# ─────────────────────────────────────────────────────────────────────


_PROOF_TESTS: dict[str, str] = {
    # generic-tensor proof anchors — every path must exist on disk
    # (the drift gate checks).  Glob patterns are accepted but must
    # match at least one file.
    "matmul":       "tests/unit/test_apple_backend_roadmap.py",
    "softmax":      "tests/unit/test_apple_backend_roadmap.py",
    "softmax_safe": "tests/unit/test_apple_backend_roadmap.py",
    "gelu":         "tests/unit/test_apple_backend_roadmap.py",
    "rope":         "tests/unit/test_apple_backend_roadmap.py",
    "flash_attn":   "tests/unit/test_apple_backend_roadmap.py",
    "rmsnorm":      "tests/unit/test_apple_backend_roadmap.py",
    "gemm":         "tests/unit/test_apple_backend_roadmap.py",
}


def _proof_for(op_name: str, family: str) -> str:
    if family == "ga":
        return "benchmarks/apple_gpu/benchmark_ga_ebm.py"
    if family == "ebm":
        return "benchmarks/apple_gpu/benchmark_ga_ebm.py"
    if family == "m7":
        # Per-op M7 tests live under tests/unit/test_complex_*.py
        # (e.g., test_complex_mobius.py covers mobius;
        # test_complex_stereographic.py covers stereographic).
        return "tests/unit/test_complex_*.py"
    return _PROOF_TESTS.get(op_name, "tests/unit/test_apple_*.py")


# ─────────────────────────────────────────────────────────────────────
# Row construction
# ─────────────────────────────────────────────────────────────────────


def _row_for_generic(op_name: str) -> AppleTargetRow:
    """Build an apple_cpu + apple_gpu row for a generic-tensor op."""

    cpu_entry = _bm._APPLE_CPU_KERNELS.get(op_name)
    gpu_entry = _bm._APPLE_GPU_KERNELS.get(op_name)

    # apple_cpu fields
    cpu_status = cpu_entry["status"] if cpu_entry else (
        "reference" if op_name in _APPLE_CPU_FRAMEWORK_HINTS else "reference"
    )
    cpu_framework = _APPLE_CPU_FRAMEWORK_HINTS.get(op_name, "numpy_reference")
    cpu_dtypes: tuple[str, ...] = tuple(cpu_entry["dtypes"]) if cpu_entry else ("fp32",)
    cpu_symbol = ""
    if op_name in {"matmul", "gemm"}:
        cpu_symbol = "tessera_apple_cpu_gemm_{f32,f16,bf16}"
    cpu_kind = _APPLE_CPU_EXECUTION_KIND.get(op_name, "numpy_reference")

    # apple_gpu fields
    gpu_status = gpu_entry["status"] if gpu_entry else "absent"
    gpu_framework = _APPLE_GPU_FRAMEWORK_HINTS.get(op_name, "")
    gpu_dtypes: tuple[str, ...] = tuple(gpu_entry["dtypes"]) if gpu_entry else ()
    gpu_symbol = ""
    if op_name in _APPLE_GPU_KERNELS_SYMBOL_MAP:
        gpu_symbol = _APPLE_GPU_KERNELS_SYMBOL_MAP[op_name]
    gpu_dispatch = "driver" if op_name in _DRIVER_DISPATCH_OPS else ("absent" if gpu_status == "absent" else "manifest")

    return AppleTargetRow(
        op_name=op_name,
        family="tensor",
        cpu_status=cpu_status,
        cpu_framework=cpu_framework,
        cpu_dtypes=cpu_dtypes,
        cpu_symbol=cpu_symbol,
        cpu_execution_kind=cpu_kind,
        gpu_status=gpu_status,
        gpu_framework=gpu_framework,
        gpu_dtypes=gpu_dtypes,
        gpu_symbol=gpu_symbol,
        gpu_dispatch=gpu_dispatch,
        proof_test=_proof_for(op_name, "tensor"),
    )


# Symbol map mirroring ``driver._backend_artifact_for`` so we render
# the same symbol the driver actually dispatches.  When that table
# grows new entries this map needs to grow too — the drift gate flags
# missing entries (any op in ``_APPLE_GPU_KERNELS`` without an entry).
_APPLE_GPU_KERNELS_SYMBOL_MAP: dict[str, str] = {
    "matmul":       "tessera_apple_gpu_mps_matmul_{f32,f16,bf16}",
    "softmax":      "tessera_apple_gpu_softmax_f32",
    "softmax_safe": "tessera_apple_gpu_softmax_f32",  # shares the kernel
    "gelu":         "tessera_apple_gpu_gelu_f32",
    "rope":         "tessera_apple_gpu_rope_f32",
    "flash_attn":   "tessera_apple_gpu_flash_attn_f32",
    "rmsnorm":      "tessera_apple_gpu_rmsnorm_f32",
    # Followup A.1 (2026-05-31) — these three ops gained manifest
    # entries to close the runtime-envelope-vs-manifest gap; the
    # drift gate ``test_every_apple_gpu_msl_kernel_has_dispatch_symbol``
    # requires a parallel entry here naming the dispatch symbol.
    "relu":         "tessera_apple_gpu_mpsgraph_unary_f32",  # opcode-dispatched
    "conv2d":       "tessera_apple_gpu_conv2d_dev_{f32,f16,bf16}_enc",
    "kv_cache_read": "tessera_apple_gpu_mps_matmul_{f32,f16,bf16}",  # cache pages dispatch via MPS
    # Project 2.1c / 3 (2026-06-01) — encode-session ops that gained
    # manifest entries (and hardware_verified promotion). They dispatch
    # through the driver's encode-session lane; name the per-op encode
    # C ABI symbol the driver routes to.
    "bmm":          "tessera_apple_gpu_bmm_dev_{f32,f16,bf16}_enc",
    "layer_norm":   "tessera_apple_gpu_layer_norm_dev_{f32,f16,bf16}_enc",
    "silu":         "tessera_apple_gpu_unary_dev_{f32,f16,bf16}_enc",  # opcode 4
    # Followup (2026-06-10) — MoE expert-FFN fused MSL kernels (f32). Both are
    # bespoke fused kernels (not encode-session), dispatched via the runtime's
    # grouped_gemm / moe_swiglu_block lanes.
    "grouped_gemm":     "tessera_apple_gpu_grouped_gemm_f32",
    "moe_swiglu_block": "tessera_apple_gpu_moe_swiglu_f32",
}


def _row_for_ga(op_name: str) -> AppleTargetRow:
    spec = _bm._CLIFFORD_APPLE_GPU_FUSED[op_name]
    dtypes = tuple(spec["dtypes"])
    symbol_prefix = spec["symbol_prefix"]
    return AppleTargetRow(
        op_name=op_name,
        family="ga",
        cpu_status="reference",
        cpu_framework="numpy_reference",
        cpu_dtypes=("fp32",),
        cpu_symbol="",
        cpu_execution_kind="numpy_reference",
        gpu_status="fused",
        gpu_framework="Metal (MSL)",
        gpu_dtypes=dtypes,
        gpu_symbol=f"{symbol_prefix}{{{','.join(dtypes)}}}",
        gpu_dispatch="manifest",
        proof_test="benchmarks/apple_gpu/benchmark_ga_ebm.py",
    )


def _row_for_ebm(op_name: str) -> AppleTargetRow:
    spec = _bm._EBM_APPLE_GPU_FUSED[op_name]
    dtypes = tuple(spec["dtypes"])
    return AppleTargetRow(
        op_name=op_name,
        family="ebm",
        cpu_status="reference",
        cpu_framework="numpy_reference",
        cpu_dtypes=("fp32", "fp64"),
        cpu_symbol="",
        cpu_execution_kind="numpy_reference",
        gpu_status="fused",
        gpu_framework="Metal (MSL)",
        gpu_dtypes=dtypes,
        gpu_symbol=str(spec["symbol"]),
        gpu_dispatch="manifest",
        proof_test="benchmarks/apple_gpu/benchmark_ga_ebm.py",
    )


def _row_for_m7(op_name: str) -> AppleTargetRow:
    spec = _bm._COMPLEX_APPLE_GPU_FUSED[op_name]
    dtypes = tuple(spec["dtypes"])
    return AppleTargetRow(
        op_name=op_name,
        family="m7",
        cpu_status="reference",
        cpu_framework="numpy_reference",
        cpu_dtypes=("fp32",),
        cpu_symbol="",
        cpu_execution_kind="numpy_reference",
        gpu_status="fused",
        gpu_framework="Metal (MSL)",
        gpu_dtypes=dtypes,
        gpu_symbol=str(spec["symbol"]),
        gpu_dispatch="manifest",
        proof_test="tests/unit/test_complex_*.py",
    )


def all_rows() -> list[AppleTargetRow]:
    """Every Apple-touching op the manifest knows about, sorted by family."""

    rows: list[AppleTargetRow] = []
    for op in _generic_tensor_ops():
        rows.append(_row_for_generic(op))
    for op in _ga_ops():
        rows.append(_row_for_ga(op))
    for op in _ebm_ops():
        rows.append(_row_for_ebm(op))
    for op in _m7_fused_ops():
        rows.append(_row_for_m7(op))
    return rows


# ─────────────────────────────────────────────────────────────────────
# CSV rendering — the canonical machine-readable artifact (the Markdown is
# the human companion). One row per op, sorted by (family, op_name);
# multi-value dtype lists are comma-joined inside a single quoted cell.
# ─────────────────────────────────────────────────────────────────────

APPLE_TARGET_CSV_COLUMNS = (
    "op_name", "family",
    "cpu_status", "cpu_framework", "cpu_dtypes", "cpu_symbol", "cpu_execution_kind",
    "gpu_status", "gpu_framework", "gpu_dtypes", "gpu_symbol", "gpu_dispatch",
    "proof_test", "notes",
)


def render_csv(rows: Iterable[AppleTargetRow] | None = None) -> str:
    import csv as _csv
    import io as _io

    rows_list = sorted(
        list(rows) if rows is not None else all_rows(),
        key=lambda r: (r.family, r.op_name),
    )
    buf = _io.StringIO()
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(APPLE_TARGET_CSV_COLUMNS)
    for r in rows_list:
        writer.writerow([
            r.op_name, r.family,
            r.cpu_status, r.cpu_framework, ",".join(r.cpu_dtypes),
            r.cpu_symbol, r.cpu_execution_kind,
            r.gpu_status, r.gpu_framework, ",".join(r.gpu_dtypes),
            r.gpu_symbol, r.gpu_dispatch,
            r.proof_test, r.notes,
        ])
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────
# Markdown rendering
# ─────────────────────────────────────────────────────────────────────


def render_markdown(rows: Iterable[AppleTargetRow] | None = None) -> str:
    rows_list = list(rows) if rows is not None else all_rows()
    rows_list.sort(key=lambda r: (
        {"tensor": 0, "ga": 1, "ebm": 2, "m7": 3}.get(r.family, 9),
        r.op_name,
    ))
    # Per-family counts for the summary block.
    by_family: dict[str, list[AppleTargetRow]] = {}
    for r in rows_list:
        by_family.setdefault(r.family, []).append(r)

    lines: list[str] = [
        "<!-- AUTO-GENERATED by python/tessera/compiler/apple_target_map.py — DO NOT EDIT BY HAND. -->",
        "<!-- Regenerate via: python -m tessera.cli.apple_target_map --render -->",
        "",
        "# Apple target map — apple_cpu vs apple_gpu",
        "",
        "Unified view of every op that has any Apple coverage today,",
        "pulled from ``capabilities.py``, ``backend_manifest.py``,",
        "and ``compiler/driver.py``.  Use this table as the single",
        "source of truth for \"is this op Apple-native, and how do we",
        "prove it?\".  The fields:",
        "",
        "* **status** — ``fused`` (native kernel ships), ``reference`` ",
        "  (numpy/runtime reference path runs), ``absent`` (no",
        "  Apple-side coverage), ``planned`` (slot reserved for a",
        "  future kernel — only used by docs).",
        "* **framework** — Apple framework that runs the kernel:",
        "  Accelerate (cblas_sgemm), BNNS (fp16/bf16),",
        "  MetalPerformanceShaders, Metal (MSL).",
        "* **dtypes** — per-op dtype set the kernel actually supports",
        "  (per ``backend_manifest`` — *not* the broader target-level",
        "  tuple).",
        "* **execution_kind** — for apple_cpu: ``accelerate_native`` /",
        "  ``bnns_native`` / ``numpy_reference``.  Distinguishes the",
        "  fast-path lane from the numpy fallback without downgrading",
        "  the ``runtime_status`` (the op still runs).",
        "* **gpu_dispatch** — ``manifest`` for the GA/EBM/M7 manifest",
        "  + jit_bridge dispatch path; ``driver`` for the generic",
        "  tensor lane (``compiler/driver.py::_backend_artifact_for``).",
        "  Phase D unifies these.",
        "* **proof_test** — the test or benchmark that exercises this",
        "  op's Apple-native path.",
        "",
        "## Metal 4 lane (not captured by the framework column above)",
        "",
        "The ``framework`` column reflects each op's **default** apple_gpu lane",
        "(MPSGraph / MetalPerformanceShaders / MSL). A **parallel Metal 4 lane**",
        "(macOS 26+, MSL 4.0) runs the MetalPerformancePrimitives cooperative",
        "``matmul2d`` on the GPU matrix units and is *not* shown per-row here",
        "(it lives in ``runtime.py``'s router, not ``capabilities``/``manifest``/",
        "``driver``). Summary — see ``docs/apple_gpu_metal4_adoption.md`` (ladder)",
        "and ``docs/apple_backend.md`` (Metal 4 implementation-state review):",
        "",
        "* **matmul** — bf16 routes to ``tessera_apple_gpu_mtl4_matmul2d_bf16`` by",
        "  **default** (M6/P5; beats the fp32-conversion fallback ~10–15×); fp16",
        "  ``matmul2d`` beats MPS but stays opt-in; f32 stays on MPS.",
        "* **linear+bias+activation** — ``matmul→add(bias)→{gelu,relu,silu}`` in",
        "  f16/bf16 auto-fuses to one ``matmul2d`` epilogue dispatch (M7/P6).",
        "* **conv2d** — f16/bf16 via im2col + the matmul2d epilogue, opt-in",
        "  (``TESSERA_APPLE_GPU_MTL4_CONV=1``) pending a GPU im2col (P8).",
        "* Resident-weight MLP-block **session** (M8) + MTL4 pipeline **archives**",
        "  (P4) amortize decode / process-start overhead.",
        "",
        "## Counts by family",
        "",
        "| Family | Rows | apple_gpu fused | apple_cpu accelerate_native |",
        "|---|---:|---:|---:|",
    ]
    for family in ("tensor", "ga", "ebm", "m7"):
        family_rows = by_family.get(family, [])
        fused_gpu = sum(1 for r in family_rows if r.gpu_status == "fused")
        acc_cpu = sum(1 for r in family_rows
                      if r.cpu_execution_kind == "accelerate_native")
        lines.append(
            f"| {family} | {len(family_rows)} | {fused_gpu} | {acc_cpu} |"
        )
    lines.append("")

    for family in ("tensor", "ga", "ebm", "m7"):
        family_rows = by_family.get(family, [])
        if not family_rows:
            continue
        lines.append(f"## {family} ({len(family_rows)})")
        lines.append("")
        lines.append(
            "| Op | apple_cpu status | cpu framework | cpu dtypes | "
            "cpu symbol | execution_kind | apple_gpu status | "
            "gpu framework | gpu dtypes | gpu symbol | gpu_dispatch | "
            "proof |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|---|---|---|---|---|"
        )
        for r in family_rows:
            cells = r.as_row()
            # Skip the family column in the per-family table — it's
            # already the section header.
            cells = [cells[0]] + cells[2:]
            lines.append("| " + " | ".join(_md_escape(c) for c in cells) + " |")
        lines.append("")

    lines.append("## How to interpret the dispatch column")
    lines.append("")
    lines.append(
        "* ``manifest`` ops go through ``jit_bridge.dispatch_via_manifest``"
        " — the GA / EBM / M7 path with explicit ``(symbol, framework,"
        " abi, notes)`` metadata."
    )
    lines.append(
        "* ``driver`` ops go through ``compiler/driver.py::_backend_artifact_for``"
        " — the generic tensor lane.  These ops *are* fused MSL kernels,"
        " but they're routed through a different proof envelope today."
        "  Phase D of the Apple plan unifies these so both paths emit"
        " the same ``(symbol, framework, abi, dispatch_path, fallback_reason)``"
        " tuple in their CompileReports."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _md_escape(text: str) -> str:
    # Markdown pipe + literal-backtick safety inside table cells.
    if not text:
        return "-"
    return text.replace("|", "\\|")


def write_doc(out_path: Path | None = None) -> Path:
    path = out_path if out_path is not None else _GENERATED_DOC
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(), encoding="utf-8")
    return path


__all__ = [
    "AppleTargetRow",
    "all_rows",
    "render_markdown",
    "write_doc",
]
