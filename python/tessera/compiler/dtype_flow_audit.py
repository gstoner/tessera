"""Generated end-to-end operator datatype-flow audit.

This module is a projection, not a new declaration registry.  It joins the
existing authorities that answer different parts of the datatype question:

* ``op_catalog`` and ``primitive_coverage`` own frontend/Graph identity and
  the target-independent numeric policy;
* ``audit`` owns the Graph/Schedule/Tile compiler-axis state;
* ``backend_manifest`` owns per-operation physical dtype and evidence claims;
* ``capabilities`` owns explicit target/runtime dtype declarations;
* ``tsol_coverage`` owns membership in the portable TSOL surface; and
* ``rocm_isa_contract`` contributes architecture-only matrix legality for
  gfx1151, gfx1200, gfx1250, and gfx1251 without promoting it to execution.

The canonical CSV has one row per ``(operator, logical tensor dtype)`` and
records the corresponding physical ABI storage separately. Accumulator dtypes
are also separate because treating an fp32 accumulator as fp32 input support is
a recurring source of false backend claims. The Markdown is the human-readable
companion and groups dtype states per operator.
"""

from __future__ import annotations

import csv
import functools
import io
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, Mapping

from tessera.dtype import canonicalize_dtype

from . import audit, backend_manifest
from .capabilities import TARGET_CAPABILITIES, canonical_op
from .op_catalog import OP_SPECS, shape_rule_for
from .primitive_coverage import all_primitive_coverages
from .tsol_coverage import all_tsol_op_names


REPORT_TARGETS: tuple[str, ...] = (
    "cpu",
    "x86",
    "apple_cpu",
    "apple_gpu",
    "nvidia_sm120",
    "rocm_gfx1151",
    "rocm_gfx1200",
    "rocm_gfx1250",
    "rocm_gfx1251",
)

_MANIFEST_TARGET_ALIASES: Mapping[str, tuple[str, ...]] = {
    "cpu": ("cpu",),
    "x86": ("x86",),
    "apple_cpu": ("apple_cpu",),
    "apple_gpu": ("apple_gpu",),
    "nvidia_sm120": ("nvidia_sm120",),
    # The generic ROCm manifest rows are the established gfx1151 lane.  They
    # must not be transferred to later AMD architectures.
    "rocm_gfx1151": ("rocm", "rocm_gfx1151"),
    "rocm_gfx1200": ("rocm_gfx1200",),
    "rocm_gfx1250": ("rocm_gfx1250",),
    "rocm_gfx1251": ("rocm_gfx1251",),
}

_STATUS_RANK = {
    "device_verified_jit": 0,
    "device_verified_abi": 1,
    "packaged": 2,
    "fused": 3,
    "ready": 4,
    "compileable": 5,
    "artifact_only": 6,
    "reference": 7,
    "planned_gated": 8,
    "planned": 9,
    "legal_only": 10,
    "partial": 11,
    "unsupported": 12,
    "absent": 13,
    "not_applicable": 14,
}

_PHYSICAL_STATES = frozenset({
    "device_verified_jit",
    "device_verified_abi",
    "packaged",
    "fused",
    "ready",
    "compileable",
    "artifact_only",
})


@dataclass(frozen=True)
class TargetDtypeState:
    status: str
    source: str
    evidence: bool = False


@dataclass(frozen=True)
class DtypeFlowRow:
    op_name: str
    graph_name: str
    family: str
    tsol: bool
    dtype: str
    physical_storage: str
    accumulators: tuple[str, ...]
    frontend: str
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    targets: Mapping[str, TargetDtypeState]
    gaps: tuple[str, ...]


def _canonical_dtype(value: object) -> str:
    return canonicalize_dtype(str(value), allow_planned_gated=True)


_COMPLEX_COMPONENT_STORAGE = {
    "complex64": "fp32x2_interleaved",
    "complex128": "fp64x2_interleaved",
}


def _physical_storage(dtype: str) -> str:
    """Return the target ABI encoding, distinct from the logical dtype."""
    return _COMPLEX_COMPONENT_STORAGE.get(dtype, dtype)


def _manifest_dtype_candidates(dtype: str) -> frozenset[str]:
    """Manifest spellings that can physically carry one logical dtype."""
    component = {
        "complex64": "fp32",
        "complex128": "fp64",
    }.get(dtype)
    return frozenset((dtype, component)) if component else frozenset((dtype,))


def _manifest_rows(coverage) -> tuple[Mapping[str, object], ...]:
    raw = coverage.metadata.get("backend_kernel_manifest", ())
    return tuple(row for row in raw if isinstance(row, Mapping))


def _declared_dtypes(coverage, graph_name: str) -> tuple[str, ...]:
    declared: set[str] = set()
    policy = coverage.metadata.get("numeric_policy")
    if isinstance(policy, Mapping) and policy.get("storage"):
        declared.add(_canonical_dtype(policy["storage"]))
    for row in _manifest_rows(coverage):
        for dtype in row.get("dtypes", ()):
            declared.add(_canonical_dtype(dtype))
    canonical = canonical_op(graph_name)
    for target in TARGET_CAPABILITIES.values():
        op_cap = target.supported_ops.get(canonical)
        if op_cap is not None and not op_cap.dtypes_derived:
            declared.update(_canonical_dtype(dtype) for dtype in op_cap.dtypes)
    # Complex is a Graph-level logical dtype implemented as two real ABI
    # components. Include it for operations whose result or input carries a
    # complex value even when the physical manifest correctly names fp32.
    rule = shape_rule_for(graph_name)
    if rule in {
        "complex_same", "complex_from_coords", "complex_to_real",
        "rfft", "irfft", "stft", "istft",
    } or graph_name == "tessera.spectral_filter":
        if "fp32" in declared:
            declared.add("complex64")
        if "fp64" in declared:
            declared.add("complex128")
    return tuple(sorted(declared))


def _matrix_accumulators(dtype: str) -> set[str]:
    out: set[str] = set()
    from .nvidia_dtype_contract import SM120_DTYPE_CONTRACTS
    from .rocm_isa_contract import AMD_DTYPE_CONTRACTS
    from .x86_dtype_contract import X86_DTYPE_CONTRACTS

    for rows in AMD_DTYPE_CONTRACTS.values():
        contract = rows.get(dtype)
        if contract is not None and contract.isa_matrix_supported:
            out.update(contract.accumulators)
    for contract in SM120_DTYPE_CONTRACTS:
        if contract.storage == dtype and contract.tensor_core != "unsupported":
            out.update(contract.accumulators)
    for contract in X86_DTYPE_CONTRACTS:
        if contract.storage == dtype and contract.matrix_state != "unsupported":
            if contract.accumulator:
                out.add(contract.accumulator)
    return {_canonical_dtype(value) for value in out}


def _accumulators(coverage, op_name: str, dtype: str) -> tuple[str, ...]:
    if op_name in {"gemm", "matmul"}:
        matrix = _matrix_accumulators(dtype)
        if matrix:
            return tuple(sorted(matrix))
    out: set[str] = set()
    policy = coverage.metadata.get("numeric_policy")
    if isinstance(policy, Mapping) and policy.get("accum"):
        out.add(_canonical_dtype(policy["accum"]))
    for row in _manifest_rows(coverage):
        if dtype not in {_canonical_dtype(value) for value in row.get("dtypes", ())}:
            continue
        for key in ("mma_selection", "mma_descriptor"):
            selection = row.get(key)
            if isinstance(selection, Mapping) and selection.get("acc_dtype"):
                out.add(_canonical_dtype(selection["acc_dtype"]))
    return tuple(sorted(out))


def _best(states: Iterable[TargetDtypeState]) -> TargetDtypeState:
    rows = tuple(states)
    if not rows:
        return TargetDtypeState("absent", "no explicit target row")
    return min(rows, key=lambda row: _STATUS_RANK.get(row.status, 99))


def _manifest_target_state(coverage, target: str, dtype: str) -> TargetDtypeState | None:
    aliases = _MANIFEST_TARGET_ALIASES[target]
    states: list[TargetDtypeState] = []
    target_declared = False
    for row in _manifest_rows(coverage):
        if str(row.get("target")) not in aliases:
            continue
        target_declared = True
        row_dtypes = {_canonical_dtype(value) for value in row.get("dtypes", ())}
        if not (_manifest_dtype_candidates(dtype) & row_dtypes):
            continue
        # A lit/FileCheck fixture is structural evidence and a benchmark file
        # is performance evidence. Neither proves numerical execution. Keep
        # direct comparison evidence separate so the report cannot silently
        # promote an artifact-only row to a tested kernel.
        evidence = bool(row.get("execute_compare_fixture"))
        states.append(TargetDtypeState(
            str(row.get("status", "planned")),
            f"backend_manifest[{row.get('target')}]:{_physical_storage(dtype)}",
            evidence,
        ))
    if states:
        return _best(states)
    if target_declared:
        # A per-op manifest row is the kernel dtype authority. Its absence for
        # this dtype is an explicit negative and cannot be widened by a target-
        # global, policy-derived capability row.
        return TargetDtypeState(
            "unsupported", f"backend_manifest[{target}]:dtype_absent"
        )
    return None


def _capability_target_state(graph_name: str, target: str, dtype: str) -> TargetDtypeState | None:
    capability = TARGET_CAPABILITIES.get(target)
    if capability is None:
        return None
    op_cap = capability.supported_ops.get(canonical_op(graph_name))
    if op_cap is None:
        return None
    dtypes = {_canonical_dtype(value) for value in op_cap.dtypes}
    if dtype not in dtypes:
        return TargetDtypeState("unsupported", f"capabilities[{target}]")
    status = op_cap.runtime_status
    source = f"capabilities[{target}]"
    if op_cap.dtypes_derived:
        # Derived capability dtypes mean the target type system and the op's
        # accumulator can represent this storage. They do not establish that
        # an operator-specific physical kernel exists.
        status = "legal_only"
        source += ":derived_legality"
    return TargetDtypeState(status, source)


def _amd_matmul_isa_state(op_name: str, target: str, dtype: str) -> TargetDtypeState | None:
    if op_name not in {"gemm", "matmul"}:
        return None
    arch_names = {
        "rocm_gfx1151": "GFX_1151",
        "rocm_gfx1200": "GFX_1200",
        "rocm_gfx1250": "GFX_1250",
        "rocm_gfx1251": "GFX_1251",
    }
    arch_name = arch_names.get(target)
    if arch_name is None:
        return None
    from .rocm_isa_contract import AMD_DTYPE_CONTRACTS
    from .rocm_target import AMDArch

    contract = AMD_DTYPE_CONTRACTS[AMDArch[arch_name]].get(dtype)
    if contract is None:
        return None
    return TargetDtypeState(
        contract.dense_matrix,
        f"rocm_isa_contract[{arch_name}]",
    )


def _target_state(coverage, op_name: str, graph_name: str,
                  target: str, dtype: str) -> TargetDtypeState:
    manifest = _manifest_target_state(coverage, target, dtype)
    if manifest is not None:
        # The per-op kernel manifest is more specific than a target-wide
        # capability row. In particular, CPU `ready` capability must not
        # overwrite a manifest's honest `reference` execution state.
        return manifest
    candidates = [
        state for state in (
            _capability_target_state(graph_name, target, dtype),
            _amd_matmul_isa_state(op_name, target, dtype),
        ) if state is not None
    ]
    return _best(candidates)


def _tile_dtype_state(audit_row, targets: Mapping[str, TargetDtypeState]) -> str:
    positive = [row.status for row in targets.values() if row.status in _PHYSICAL_STATES]
    if positive:
        return audit_row.cells["tile_ir"].status
    if any(row.status == "reference" for row in targets.values()):
        return "reference"
    return "planned"


def _gaps(graph: str, schedule: str, tile: str,
          targets: Mapping[str, TargetDtypeState]) -> tuple[str, ...]:
    gaps: list[str] = []
    if graph not in {"registered", "not_applicable", "host_materialized", "runtime_only"}:
        gaps.append("graph")
    if schedule in {"planned", "partial", "missing"}:
        gaps.append("schedule")
    if tile in {"planned", "partial", "missing"}:
        gaps.append("tile")
    native = {
        row.status for target, row in targets.items()
        if target != "cpu"
    } & _PHYSICAL_STATES
    if not native:
        gaps.append("physical")
    if not any(row.evidence for target, row in targets.items() if target != "cpu"):
        gaps.append("direct_test")
    return tuple(gaps)


@functools.cache
def collect_dtype_flow_rows() -> tuple[DtypeFlowRow, ...]:
    coverages = all_primitive_coverages()
    tsol = set(all_tsol_op_names())
    rows: list[DtypeFlowRow] = []
    for audit_row in audit.all_support_rows():
        op_name = audit_row.op_name
        coverage = coverages.get(op_name)
        if coverage is None:
            # The support audit intentionally includes benchmark-visible
            # domain ops in addition to the primitive registry. Preserve them
            # and their backend dtype rows while flagging the missing registry
            # through the Graph/Schedule axes instead of crashing generation.
            coverage = SimpleNamespace(
                graph_name=None,
                metadata={
                    "backend_kernel_manifest": tuple(
                        entry.as_dict()
                        for entry in backend_manifest.manifest_for(op_name)
                    )
                },
            )
        spec = OP_SPECS.get(op_name)
        graph_name = (
            spec.graph_name if spec is not None
            else coverage.graph_name or f"tessera.{op_name}"
        )
        dtypes = _declared_dtypes(coverage, graph_name)
        if not dtypes:
            # Non-tensor compiler/runtime surfaces remain visible without
            # inventing a dtype. Tensor operators are guarded by tests below.
            dtypes = ("",)
        for dtype in dtypes:
            target_states = {
                target: (
                    _target_state(coverage, op_name, graph_name, target, dtype)
                    if dtype else TargetDtypeState("not_applicable", "no dtype contract")
                )
                for target in REPORT_TARGETS
            }
            graph = audit_row.cells["graph_ir"].status
            schedule = audit_row.cells["schedule_ir"].status
            tile = _tile_dtype_state(audit_row, target_states) if dtype else "not_applicable"
            rows.append(DtypeFlowRow(
                op_name=op_name,
                graph_name=graph_name,
                family=audit_row.family,
                tsol=op_name in tsol,
                dtype=dtype,
                physical_storage=_physical_storage(dtype) if dtype else "",
                accumulators=_accumulators(coverage, op_name, dtype) if dtype else (),
                frontend=audit_row.cells["frontend"].status,
                graph_ir=graph,
                schedule_ir=schedule,
                tile_ir=tile,
                targets=target_states,
                gaps=_gaps(graph, schedule, tile, target_states) if dtype else (),
            ))
    return tuple(sorted(rows, key=lambda row: (row.family, row.op_name, row.dtype)))


CSV_COLUMNS = (
    "op", "graph_name", "family", "tsol", "dtype", "physical_storage",
    "accumulators",
    "frontend", "graph_ir", "schedule_ir", "tile_ir",
    *REPORT_TARGETS, "gaps",
)


def render_csv(rows: Iterable[DtypeFlowRow] | None = None) -> str:
    rows = tuple(rows) if rows is not None else collect_dtype_flow_rows()
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        writer.writerow([
            row.op_name,
            row.graph_name,
            row.family,
            "yes" if row.tsol else "no",
            row.dtype,
            row.physical_storage,
            "/".join(row.accumulators),
            row.frontend,
            row.graph_ir,
            row.schedule_ir,
            row.tile_ir,
            *(row.targets[target].status for target in REPORT_TARGETS),
            "/".join(row.gaps),
        ])
    return stream.getvalue()


def _group_states(rows: Iterable[DtypeFlowRow], target: str) -> str:
    by_status: dict[str, list[str]] = {}
    for row in rows:
        state = row.targets[target].status
        if state in {"absent", "unsupported", "not_applicable"} or not row.dtype:
            continue
        by_status.setdefault(state, []).append(row.dtype)
    if not by_status:
        return "-"
    return "; ".join(
        f"{status}: {','.join(sorted(set(dtypes)))}"
        for status, dtypes in sorted(
            by_status.items(), key=lambda item: _STATUS_RANK.get(item[0], 99)
        )
    )


def _dtype_display(rows: Iterable[DtypeFlowRow]) -> str:
    values = []
    for row in rows:
        if not row.dtype:
            continue
        values.append(
            row.dtype
            if row.physical_storage == row.dtype
            else f"{row.dtype}→{row.physical_storage}"
        )
    return ",".join(values) or "-"


def render_markdown(rows: Iterable[DtypeFlowRow] | None = None) -> str:
    rows = tuple(rows) if rows is not None else collect_dtype_flow_rows()
    by_op: dict[str, list[DtypeFlowRow]] = {}
    for row in rows:
        by_op.setdefault(row.op_name, []).append(row)
    tsol_count = len({row.op_name for row in rows if row.tsol})
    op_count = len(by_op)
    dtype_rows = sum(1 for row in rows if row.dtype)
    gap_rows = sum(1 for row in rows if row.gaps)
    lines = [
        "<!-- AUTO-GENERATED by tessera.compiler.dtype_flow_audit; do not edit. -->",
        "<!-- Regenerate: python -m tessera.compiler.generated_docs --write dtype_flow -->",
        "",
        "# Operator datatype flow",
        "",
        "This dashboard joins the existing frontend, Graph, Schedule, Tile, target",
        "capability, backend manifest, TSOL, and architecture dtype registries. An",
        "ISA-supported or planned dtype is not reported as executable unless a",
        "physical backend row says so. Accumulators are listed separately from",
        "logical tensor dtype. Complex logical values display their physical",
        "interleaved-component ABI explicitly.",
        "",
        f"- Operators: **{op_count}**",
        f"- TSOL operators: **{tsol_count}**",
        f"- Operator/logical-dtype rows: **{dtype_rows}**",
        f"- Rows retaining at least one compiler/evidence gap: **{gap_rows}**",
        "",
        "The CSV companion is the canonical normalized matrix: one row per",
        "`(operator, logical dtype)`, with ABI storage and independent states",
        "for every IR level",
        "and selected exact targets.",
        "",
        "## TSOL dtype flow",
        "",
        "| Operator | Logical dtype → ABI storage | Accumulators | Graph / Schedule / Tile | x86 | Apple GPU | NVIDIA SM120 | gfx1151 | gfx1200 | gfx1250 | Gaps |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for op_name, op_rows in sorted(by_op.items(), key=lambda item: (item[1][0].family, item[0])):
        if not op_rows[0].tsol:
            continue
        dtypes = _dtype_display(op_rows)
        accums = sorted({value for row in op_rows for value in row.accumulators})
        axes = f"{op_rows[0].graph_ir} / {op_rows[0].schedule_ir} / {op_rows[0].tile_ir}"
        gaps = sorted({gap for row in op_rows for gap in row.gaps})
        lines.append(
            f"| `{op_name}` | {dtypes} | {','.join(accums) or '-'} | {axes} | "
            f"{_group_states(op_rows, 'x86')} | {_group_states(op_rows, 'apple_gpu')} | "
            f"{_group_states(op_rows, 'nvidia_sm120')} | "
            f"{_group_states(op_rows, 'rocm_gfx1151')} | "
            f"{_group_states(op_rows, 'rocm_gfx1200')} | "
            f"{_group_states(op_rows, 'rocm_gfx1250')} | "
            f"{','.join(gaps) or '-'} |"
        )
    lines.extend([
        "",
        "## All-operator gap queue",
        "",
        "Only operators with a retained dtype-flow gap appear here. Use the CSV",
        "for the complete per-dtype and per-target matrix.",
        "",
        "| Operator | Family | TSOL | Dtypes | Gaps |",
        "|---|---|---:|---|---|",
    ])
    for op_name, op_rows in sorted(by_op.items(), key=lambda item: (item[1][0].family, item[0])):
        gaps = sorted({gap for row in op_rows for gap in row.gaps})
        if not gaps:
            continue
        dtypes = _dtype_display(op_rows)
        lines.append(
            f"| `{op_name}` | {op_rows[0].family} | "
            f"{'yes' if op_rows[0].tsol else 'no'} | {dtypes} | {','.join(gaps)} |"
        )
    lines.extend([
        "",
        "## Status interpretation",
        "",
        "- `device_verified_jit` / `device_verified_abi`: exact execution evidence exists.",
        "- `fused`, `packaged`, `ready`: a physical consumer exists; consult the CSV source registries for its evidence boundary.",
        "- `compileable` / `artifact_only`: compiler artifact only; no execution promotion.",
        "- `legal_only`: dtype/accumulator legality only; no operator-specific physical kernel claim.",
        "- `planned_gated` / `planned`: declared future support, not executable support.",
        "- `reference`: reference implementation only.",
        "- `absent` / `unsupported`: no declared target path for that operator/dtype.",
        "",
    ])
    return "\n".join(lines)


__all__ = [
    "CSV_COLUMNS",
    "DtypeFlowRow",
    "REPORT_TARGETS",
    "TargetDtypeState",
    "collect_dtype_flow_rows",
    "render_csv",
    "render_markdown",
]
