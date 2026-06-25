"""GPU target maps — NVIDIA + ROCm dashboards
(Apple follow-up #3, 2026-05-20).

The Apple target map (``apple_target_map.py``) was the prototype:
a per-op view that pulls capabilities + backend_manifest + driver
dispatch into one drift-gated dashboard, with explicit columns for
``execution_kind`` / framework / symbol / dispatch path.

Today the NVIDIA and ROCm lanes don't yet have hardware proof —
they're at ``artifact_only`` status across the board.  But the
*structure* of the dashboard can already land, populated from the
planned-slot tables in ``backend_manifest``.  When Phase G (NVIDIA
H100) or Phase H (ROCm MI300X) bring-up moves a row from
``artifact_only`` → ``compileable`` → ``executable`` → ``fused``,
the dashboard auto-promotes without any new infrastructure.

This module ships:

* ``GPUTargetRow`` — same schema shape as ``AppleTargetRow`` but
  with target-family-specific columns (cuda_arch_min /
  wgmma_shape / mfu / roofline for NVIDIA; mfma_shape /
  hipcc_version_min for ROCm).
* ``all_nvidia_rows()`` / ``all_rocm_rows()`` — pulled from
  ``backend_manifest._NVIDIA_ARTIFACT`` / ``_ROCM_ARTIFACT``.
* ``render_markdown(target)`` — produces the per-target
  ``<target>_target_map.md`` (e.g., ``nvidia_target_map.md``).
* Drift gates in ``tests/unit/test_gpu_target_maps.py``.

When NVIDIA / ROCm move to hardware proof, the existing
``release_gate.py`` matrix grows per-target gates that wrap this
dashboard exactly like ``release_gate_apple_gpu`` does.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import backend_manifest as _bm
from . import capabilities as _cap


_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUDIT_DIR = _REPO_ROOT / "docs" / "audit" / "generated"


# ─────────────────────────────────────────────────────────────────────
# Row schema — symmetric to AppleTargetRow but adds GPU-toolchain
# columns (arch, tile shapes, MFU, roofline).
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GPUTargetRow:
    op_name: str
    family: str               # tensor / attention / norm / activation / ...
    status: str               # planned / artifact_only / compileable / fused
    dtypes: tuple[str, ...]
    arch_min: str             # sm_80 / sm_90a / gfx942 / gfx950
    tile_shape: str           # "(64, 256, 16)" or "(32, 32, 8, 1)" — empty if unknown
    expected_mfu: float | None
    roofline: str             # roofline label or ""
    notes: str = ""

    def as_row(self) -> list[str]:
        return [
            self.op_name, self.family, self.status,
            ",".join(self.dtypes) or "-",
            self.arch_min, self.tile_shape or "-",
            f"{self.expected_mfu:.0%}" if self.expected_mfu is not None else "-",
            self.roofline or "-",
        ]


# ─────────────────────────────────────────────────────────────────────
# Family classifier — bucket ops for readability in the dashboard.
# Drawn from the same op-name patterns the audit walker uses.
# ─────────────────────────────────────────────────────────────────────


_TENSOR_OPS = {
    "matmul", "batched_gemm", "einsum", "linear_general",
    "qkv_projection", "fused_epilogue", "factorized_matmul",
}
_ATTENTION_OPS = {
    "flash_attn", "multi_head_attention", "gqa_attention", "mqa_attention",
    "mla_decode", "mla_decode_fused", "deepseek_sparse_attention",
    "attn_top_k_blocks", "attn_compressed_blocks", "attn_sliding_window",
    "lightning_attention", "linear_attn", "gated_deltanet",
    "kimi_delta_attention", "modified_delta_attention", "gated_attention",
    "hybrid_attention",
}
_NORM_OPS = {"layer_norm", "rmsnorm", "rmsnorm_safe"}
_ACTIVATION_OPS = {
    "gelu", "silu", "silu_mul", "softmax", "softmax_safe", "online_softmax",
}
_POSITION_OPS = {"rope", "alibi"}


def _family_for(op_name: str) -> str:
    if op_name.startswith("tessera."):
        op_name = op_name.removeprefix("tessera.")
    if op_name in _TENSOR_OPS:
        return "tensor"
    if op_name in _ATTENTION_OPS:
        return "attention"
    if op_name in _NORM_OPS:
        return "norm"
    if op_name in _ACTIVATION_OPS:
        return "activation"
    if op_name in _POSITION_OPS:
        return "position_encoding"
    return "other"


# ─────────────────────────────────────────────────────────────────────
# Row construction — pulls the per-target capability entry, looks up
# tile / MFU / roofline from ``backend_manifest``.
# ─────────────────────────────────────────────────────────────────────


def _row_for(op_full_name: str, target: str) -> GPUTargetRow | None:
    """Build a row for an op + target.  Returns None if the target
    has no capability entry for this op."""
    cap = _cap.TARGET_CAPABILITIES.get(target)
    if cap is None:
        return None
    op_cap = cap.supported_ops.get(op_full_name)
    if op_cap is None:
        return None

    # Plain op name without ``tessera.`` prefix for the dashboard.
    short = op_full_name.removeprefix("tessera.")

    # Tile shapes + MFU + roofline live in backend_manifest's per-op
    # tables.  Use ``manifest_for(short)`` and pick the row matching
    # the target.
    manifest_rows = _bm.manifest_for(short)
    target_row = next(
        (r for r in manifest_rows if r.target == target), None,
    )
    if target_row is None:
        # Fall back to the capability registry's runtime_status (which
        # may say "artifact_only" while no manifest row exists yet).
        return GPUTargetRow(
            op_name=short,
            family=_family_for(short),
            status=op_cap.runtime_status,
            dtypes=tuple(op_cap.dtypes),
            arch_min="-",
            tile_shape="",
            expected_mfu=None,
            roofline="",
        )

    tile = ""
    if target_row.wgmma_shape is not None:
        tile = str(target_row.wgmma_shape)
    elif target_row.mfma_shape is not None:
        tile = str(target_row.mfma_shape)

    arch_min = ""
    if target_row.cuda_arch_min:
        arch_min = target_row.cuda_arch_min
    elif target_row.hipcc_version_min:
        # ROCm rows don't carry a per-arch tag the same way — surface
        # the hipcc version pin instead.
        arch_min = f"hipcc≥{target_row.hipcc_version_min}"

    return GPUTargetRow(
        op_name=short,
        family=_family_for(short),
        status=target_row.status,
        dtypes=tuple(target_row.dtypes),
        arch_min=arch_min or "-",
        tile_shape=tile,
        expected_mfu=target_row.expected_mfu,
        roofline=target_row.roofline_target or "",
        notes=target_row.notes or "",
    )


def all_nvidia_rows(arch: str = "nvidia_sm90") -> list[GPUTargetRow]:
    """One row per NVIDIA-supported op for ``arch`` (default sm_90).
    Pulled from ``capabilities[arch].supported_ops`` ∪
    ``backend_manifest._NVIDIA_ARTIFACT``."""
    cap = _cap.TARGET_CAPABILITIES.get(arch)
    if cap is None:
        return []
    rows: list[GPUTargetRow] = []
    for op_full_name in sorted(cap.supported_ops):
        row = _row_for(op_full_name, arch)
        if row is not None:
            rows.append(row)
    return rows


def all_rocm_rows(arch: str = "rocm") -> list[GPUTargetRow]:
    """One row per ROCm-supported op for ``arch`` (default the
    ``rocm`` alias = MI300X gfx942)."""
    return all_nvidia_rows(arch)  # same data path, different target


# ─────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────


GPU_TARGET_CSV_COLUMNS = (
    "op_name", "family", "status", "dtypes",
    "arch_min", "tile_shape", "expected_mfu", "roofline", "notes",
)


def render_csv(target: str) -> str:
    """Canonical machine-readable per-target capability CSV (the Markdown is the
    human companion). One row per op, sorted by (family, op_name)."""
    import csv as _csv
    import io as _io

    rows = sorted(all_nvidia_rows(target), key=lambda r: (r.family, r.op_name))
    buf = _io.StringIO()
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(GPU_TARGET_CSV_COLUMNS)
    for r in rows:
        writer.writerow([
            r.op_name, r.family, r.status, ",".join(r.dtypes),
            r.arch_min, r.tile_shape,
            f"{r.expected_mfu:.4f}" if r.expected_mfu is not None else "",
            r.roofline, r.notes,
        ])
    return buf.getvalue()


def _rocm_fp8_section() -> list[str]:
    """Per-arch FP8 numeric-semantics table (A6) — FNUZ vs OCP.

    The same canonical ``fp8_e4m3`` / ``fp8_e5m2`` dtype encodes different bits
    across AMD generations; this surfaces that arch-keyed distinction in the
    drift-gated dashboard.  Source of truth:
    ``rocm_target._FP8_SEMANTICS`` / ``fp8_dtype_flavor``.
    """
    from . import rocm_target as _rt

    arches = (
        _rt.AMDArch.GFX_90A, _rt.AMDArch.GFX_940, _rt.AMDArch.GFX_942,
        _rt.AMDArch.GFX_950, _rt.AMDArch.GFX_1100, _rt.AMDArch.GFX_1151,
        _rt.AMDArch.GFX_1200, _rt.AMDArch.GFX_1250, _rt.AMDArch.GFX_1251,
    )
    lines = [
        "## FP8 numeric semantics (per arch)",
        "",
        "The same canonical `fp8_e4m3` / `fp8_e5m2` dtype encodes **different "
        "bits** across AMD generations (FNUZ vs OCP) — a correctness-critical, "
        "arch-keyed distinction (A6). Source: "
        "`tessera.compiler.rocm_target._FP8_SEMANTICS`.",
        "",
        "| arch | FP8 semantics | e4m3 flavor | e5m2 flavor |",
        "|---|---|---|---|",
    ]
    for a in arches:
        sem = _rt.fp8_semantics(a)
        if sem == "none":
            e4 = e5 = "-"
        else:
            e4 = _rt.fp8_dtype_flavor(a, "fp8_e4m3")
            e5 = _rt.fp8_dtype_flavor(a, "fp8_e5m2")
        lines.append(f"| `{_rt.rocm_arch_string(a)}` | {sem} | {e4} | {e5} |")
    lines.append("")
    return lines


def render_markdown(target: str) -> str:
    """Render the per-target dashboard.

    ``target`` is the canonical capability name
    (e.g., ``"nvidia_sm90"``, ``"rocm"`` / ``"rocm_gfx942"``).
    """
    rows = all_nvidia_rows(target)  # data path is target-agnostic
    if target.startswith("nvidia"):
        title = f"NVIDIA target map — `{target}`"
        family = "NVIDIA"
    elif target.startswith("rocm"):
        title = f"ROCm target map — `{target}`"
        family = "ROCm"
    else:
        title = f"GPU target map — `{target}`"
        family = target

    rows.sort(key=lambda r: (r.family, r.op_name))
    by_family: dict[str, list[GPUTargetRow]] = {}
    for r in rows:
        by_family.setdefault(r.family, []).append(r)

    lines: list[str] = [
        "<!-- AUTO-GENERATED by python/tessera/compiler/gpu_target_map.py — DO NOT EDIT BY HAND. -->",
        f"<!-- Regenerate via: python -m tessera.cli.gpu_target_map --target={target} --render -->",
        "",
        f"# {title}",
        "",
        f"Per-op view of {family} coverage today (2026-05-20).  Same "
        f"row schema as the Apple target map "
        f"(``docs/audit/generated/apple_target_map.md``); pulled from "
        f"``capabilities[{target!r}]`` + ``backend_manifest._{family.upper()}_ARTIFACT``.",
        "",
        f"**Status story today:** most {family} rows are at "
        f"``artifact_only`` or ``planned`` — IR/PTX artifact emission "
        f"is in tree, but hardware execution is gated on the bring-up "
        f"sprint (Phase G for NVIDIA, Phase H for ROCm).  The rows that "
        f"have been proven on real hardware carry an execution rung: "
        f"``hardware_verified`` (a shipped C-ABI ``runtime_symbol`` + a "
        f"numerical fixture) or ``compiled`` (a compiler-generated hsaco "
        f"that executes via ``runtime.launch()`` + a numerical fixture, "
        f"but no shipped C symbol).  ``release_gate.py --target={target}`` "
        f"gains the target-specific gates (canonical native dispatch, "
        f"per-target benchmarks, hardware-marked tests) for those rows the "
        f"same way ``--target=apple_gpu`` does today.",
        "",
        "## Status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    for status in ("hardware_verified", "compiled", "fused", "compileable",
                   "executable", "artifact_only", "reference", "planned"):
        n = counts.get(status, 0)
        if n:
            lines.append(f"| ``{status}`` | {n} |")
    lines.append(f"| **total** | **{len(rows)}** |")
    lines.append("")

    if target.startswith("rocm"):
        lines.extend(_rocm_fp8_section())

    for fam in sorted(by_family):
        fam_rows = by_family[fam]
        lines.append(f"## {fam} ({len(fam_rows)})")
        lines.append("")
        lines.append(
            "| Op | status | dtypes | arch_min | tile shape | expected MFU | roofline |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|"
        )
        for r in fam_rows:
            cells = r.as_row()
            # Drop the family column since it's the section header.
            cells = [cells[0]] + cells[2:]
            lines.append("| " + " | ".join(_escape(c) for c in cells) + " |")
        lines.append("")

    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "* **status** uses the same vocabulary as ``apple_target_map.md`` "
        "(``fused`` / ``compileable`` / ``executable`` / ``artifact_only`` "
        "/ ``planned``), plus two execution rungs: ``hardware_verified`` (executes "
        "on real hardware via a shipped C-ABI ``runtime_symbol`` + numerical "
        "fixture) and ``compiled`` (executes via ``runtime.launch()`` as a "
        "compiler-generated hsaco + numerical fixture, but NO shipped C symbol — "
        "one rung below ``hardware_verified``)."
    )
    lines.append(
        "* **dtypes** is the per-op kernel dtype matrix — same "
        "interpretation rule as ``BackendKernelEntry.dtypes``: on a "
        "``planned`` row, the dtype tuple is the target kernel dtype "
        "matrix, not what runs today."
    )
    lines.append(
        "* **tile shape** is the WGMMA `(M, N, K)` for NVIDIA Hopper+ "
        "or the MFMA `(M, N, K, K_blocks)` for ROCm CDNA."
    )
    lines.append(
        "* **arch_min** is the minimum target arch the kernel "
        "compiles for (``sm_90a`` for Hopper WGMMA, ``hipcc≥7.2.4`` "
        "for ROCm)."
    )
    return "\n".join(lines) + "\n"


def _escape(text: str) -> str:
    if not text:
        return "-"
    return text.replace("|", "\\|")


def write_doc(target: str, out_path: Path | None = None) -> Path:
    if out_path is None:
        # Default location: docs/audit/generated/<target>_target_map.md.
        # For ``rocm`` alias use ``rocm_target_map.md``; for full names
        # use the full name.
        name = target if target != "rocm" else "rocm"
        out_path = _AUDIT_DIR / f"{name}_target_map.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(target), encoding="utf-8")
    return out_path


__all__ = [
    "GPUTargetRow",
    "all_nvidia_rows",
    "all_rocm_rows",
    "render_markdown",
    "write_doc",
]
