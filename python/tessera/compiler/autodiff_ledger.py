"""Compiler-autodiff connection ledger — a *projection*, not a registry.

Phase 0 deliverable of
``docs/audit/compiler/AUTODIFF_UNIFICATION_PLAN.md``.

This module introduces **no new source of truth**.  It joins three existing
registries into one op-family × rung view whose single new contribution is the
dimension none of them makes explicit today: **forward vs. backward, per
target.**  Decision #24 is preserved — ``primitive_coverage`` stays the audit
truth for the ``vjp``/``jvp`` axes; this ledger only *reads* them.

The six rungs (see plan §3) and where each is sourced from:

======================  ==================================================
rung                    source (existing, read-only here)
======================  ==================================================
``python_reference``    ``primitive_coverage`` ``vjp``/``jvp`` contract axis
``ir_adjoint``          ``AdjointInterface.cpp`` — a *native* ``buildAdjoint``
                        (emits real Graph IR) vs. a ``placeholder`` one
                        (``custom_adjoint_call`` → round-trips to the Python
                        VJP, i.e. **not** native) vs. ``none``
``target_lowered``      backward lowering probe (Phase 3) — none wired yet
``runtime_bound``       ``runtime_execution_matrix`` backward column (Phase 4)
``oracle_proven``       ``op_target_conformance`` backward fixture (Phase 3)
``hardware_proven``     ``runtime_execution_matrix`` ``hardware_verified``
                        backward rows (Phase 4/6)
======================  ==================================================

The four backward-execution rungs are **empty today by construction**: no
backward launch ABI, backward matrix column, or backward oracle fixture exists
yet (Phases 3–4 create them).  The ledger surfaces that emptiness honestly
rather than letting "forward native" read as "training supported."
"""

from __future__ import annotations

import csv as _csv
import io as _io
import re
from pathlib import Path

from . import primitive_coverage

# Repo root: python/tessera/compiler/autodiff_ledger.py → parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ADJOINT_CPP = _REPO_ROOT / "src" / "compiler" / "ir" / "AdjointInterface.cpp"

# The backend targets the backward-execution rungs are tracked against.  Kept in
# sync with the runtime execution matrix's target set; the ledger reports which
# of these reach each backward rung (none, today).
_TARGETS: tuple[str, ...] = ("cpu", "x86", "apple_gpu", "rocm", "nvidia")

# Ops whose `buildAdjoint` emits a native `tessera.custom_adjoint_call`
# placeholder that round-trips to the Python VJP registry at runtime.  These are
# NOT native Graph-IR adjoints — they are the compiler saying "ask Python".
_PLACEHOLDER_MACRO_RE = re.compile(r'POINTWISE_BUILD_ADJOINT\(\s*\w+\s*,\s*"([^"]+)"\s*\)')

# Ops with a hand-written, explicit `<OpName>Op::buildAdjoint` definition emit
# real backward Graph IR (matmul's transposed matmuls; tanh/sigmoid's W5
# closed forms). These are the NATIVE adjoints. The macro-generated ops above
# are placeholder round-trips, not native. The macro *body* also textually
# contains `OPNAME::buildAdjoint`, so the literal token "OPNAME" is filtered.
_EXPLICIT_BUILDADJOINT_RE = re.compile(r'(\w+)Op::buildAdjoint\b')

# Phase 3 (2026-07-11) — families whose compiler-emitted backward IR is
# **oracle-verified on CPU by interpretation**: the actual
# `--tessera-autodiff-paired` output is numerically interpreted and its gradients
# match an independent NumPy VJP.  Proven by
# tests/unit/test_autodiff_paired_cpu_oracle.py.  This is the CPU IR-execution
# rung — strictly weaker than native `oracle_proven` (native LLVM/runtime
# execution, Phase 4) and `hardware_proven`; it does NOT set those columns.
_BWD_IR_ORACLE_CPU: frozenset[str] = frozenset({"matmul", "tanh", "sigmoid"})


class LedgerError(RuntimeError):
    """Raised when the ledger cannot read a source it must join."""


def _read_adjoint_source() -> str:
    if not _ADJOINT_CPP.is_file():
        raise LedgerError(
            f"cannot build the autodiff ledger: {_ADJOINT_CPP} is missing — the "
            "`ir_adjoint` rung is sourced from it (Decision #26: no silent "
            "no-op)."
        )
    return _ADJOINT_CPP.read_text(encoding="utf-8")


def _ir_adjoint_classes() -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(native_keys, placeholder_keys)`` parsed from the C++ source.

    ``native`` = emits real backward Graph IR (static-shape path).
    ``placeholder`` = ``custom_adjoint_call`` → Python VJP round-trip.
    """
    text = _read_adjoint_source()
    placeholder = frozenset(_PLACEHOLDER_MACRO_RE.findall(text))
    # Explicit `<OpName>Op::buildAdjoint` defs → native. Map OpName → family key
    # by lowercasing (the regex already dropped the trailing "Op"); filter the
    # macro's literal "OPNAME" template token.
    native = frozenset(
        name.lower()
        for name in _EXPLICIT_BUILDADJOINT_RE.findall(text)
        if name != "OPNAME"
    )
    # A native-capable op must never also be counted as placeholder-only.
    placeholder = placeholder - native
    if not native and not placeholder:
        raise LedgerError(
            f"parsed no adjoint keys from {_ADJOINT_CPP}; the macro/fallback "
            "conventions changed — update the ledger regexes rather than "
            "silently reporting zero IR adjoints."
        )
    return native, placeholder


def _is_differentiable(cov: "primitive_coverage.PrimitiveCoverage") -> bool:
    cs = cov.contract_status
    return cs.get("vjp") == "complete" or cs.get("jvp") == "complete"


def _match_keys(cov: "primitive_coverage.PrimitiveCoverage") -> set[str]:
    """The names the C++ adjoint keys could match this primitive by."""
    keys = {cov.name}
    if cov.graph_name:
        keys.add(cov.graph_name)
    return keys


class LedgerRow:
    __slots__ = ("family", "category", "python_reference", "ir_adjoint",
                 "bwd_cpu_ir_oracle", "bwd_target_lowered", "bwd_runtime_bound",
                 "bwd_oracle_proven", "bwd_hardware_proven", "notes")

    def __init__(self, family, category, python_reference, ir_adjoint, notes):
        self.family = family
        self.category = category
        self.python_reference = python_reference
        self.ir_adjoint = ir_adjoint
        # Phase 3 — compiler-emitted backward IR oracle-verified on CPU by
        # interpretation (weaker than native execution; see _BWD_IR_ORACLE_CPU).
        self.bwd_cpu_ir_oracle: bool = family in _BWD_IR_ORACLE_CPU
        # Native backward-execution rungs are per-target target lists; empty
        # until the backward matrix column / fixtures land (Phase 4).
        self.bwd_target_lowered: tuple[str, ...] = ()
        self.bwd_runtime_bound: tuple[str, ...] = ()
        self.bwd_oracle_proven: tuple[str, ...] = ()
        self.bwd_hardware_proven: tuple[str, ...] = ()
        self.notes = notes


def collect_rows() -> list[LedgerRow]:
    """Join primitive_coverage + the C++ adjoint classes into ledger rows.

    A row is emitted for every primitive that is either differentiable (has a
    Python VJP/JVP reference) or carries any IR adjoint — the two facts the
    ledger cross-references.
    """
    native, placeholder = _ir_adjoint_classes()
    covs = primitive_coverage.all_primitive_coverages()
    # Phase 4 (A2): the native backward-execution rungs are sourced from the
    # runtime execution matrix's backward rows, never asserted here.
    bwd = _native_backward_by_family()

    rows: list[LedgerRow] = []
    for name in sorted(covs):
        cov = covs[name]
        keys = _match_keys(cov)
        if keys & native:
            ir_adjoint = "native"
            notes = "native static-shape adjoint (W5); dynamic → placeholder"
        elif keys & placeholder:
            ir_adjoint = "placeholder"
            notes = "custom_adjoint_call → Python VJP (not native IR)"
        else:
            ir_adjoint = "none"
            notes = ""
        differentiable = _is_differentiable(cov)
        if not differentiable and ir_adjoint == "none":
            continue
        row = LedgerRow(
            family=name,
            category=cov.category,
            python_reference="yes" if differentiable else "no",
            ir_adjoint=ir_adjoint,
            notes=notes,
        )
        # Fill the native backward rungs from the matrix, matching on the
        # primitive's name or graph_name against the row's op_family.
        info = next((bwd[k] for k in keys if k in bwd), None)
        if info is not None:
            row.bwd_runtime_bound = info["runtime_bound"]
            row.bwd_hardware_proven = info["hardware_proven"]
            if info["hardware_proven"] and "native backward" not in row.notes:
                targets = ", ".join(info["hardware_proven"])
                row.notes = (row.notes + "; " if row.notes else "") + \
                    f"native backward executes on {targets} (Phase 4)"
        rows.append(row)
    return rows


def _native_backward_by_family() -> dict[str, dict[str, tuple[str, ...]]]:
    """The execution matrix's native backward launches, per op-family (A2). Lazy
    import to keep the module import-cycle-free (execution_matrix does not import
    this ledger)."""
    from . import execution_matrix
    return execution_matrix.native_backward_targets()


def python_reference_families() -> frozenset[str]:
    """The families the ledger reports at ``python_reference`` — used by the
    reconciliation test to assert no divergence from primitive_coverage."""
    return frozenset(r.family for r in collect_rows() if r.python_reference == "yes")


def _summary(rows: list[LedgerRow]) -> dict[str, int]:
    return {
        "families": len(rows),
        "python_reference": sum(1 for r in rows if r.python_reference == "yes"),
        "ir_adjoint_native": sum(1 for r in rows if r.ir_adjoint == "native"),
        "ir_adjoint_placeholder": sum(1 for r in rows if r.ir_adjoint == "placeholder"),
        "backward_cpu_ir_oracle": sum(1 for r in rows if r.bwd_cpu_ir_oracle),
        "backward_runtime_bound": sum(1 for r in rows if r.bwd_runtime_bound),
        "backward_oracle_proven": sum(1 for r in rows if r.bwd_oracle_proven),
        "backward_hardware_proven": sum(1 for r in rows if r.bwd_hardware_proven),
    }


def _fmt_targets(ts: tuple[str, ...]) -> str:
    return ",".join(ts) if ts else ""


def render_csv() -> str:
    rows = collect_rows()
    cols = (
        "family", "category", "python_reference", "ir_adjoint",
        "bwd_cpu_ir_oracle", "bwd_target_lowered", "bwd_runtime_bound",
        "bwd_oracle_proven", "bwd_hardware_proven", "notes",
    )
    buf = _io.StringIO()
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(cols)
    for r in rows:
        writer.writerow([
            r.family, r.category, r.python_reference, r.ir_adjoint,
            "cpu" if r.bwd_cpu_ir_oracle else "",
            _fmt_targets(r.bwd_target_lowered), _fmt_targets(r.bwd_runtime_bound),
            _fmt_targets(r.bwd_oracle_proven), _fmt_targets(r.bwd_hardware_proven),
            r.notes,
        ])
    return buf.getvalue()


def render_markdown() -> str:
    rows = collect_rows()
    s = _summary(rows)
    native, placeholder = _ir_adjoint_classes()
    lines: list[str] = [
        "<!-- AUTO-GENERATED by tessera.compiler.autodiff_ledger — do not edit. "
        "Regenerate via scripts/check_generated_docs.sh --write -->",
        "",
        "# Compiler-Autodiff Connection Ledger (generated)",
        "",
        "One row per differentiable **op family**, over the six rungs of "
        "[`AUTODIFF_UNIFICATION_PLAN.md`](../compiler/AUTODIFF_UNIFICATION_PLAN.md) "
        "§3. This is a **projection** that joins `primitive_coverage` "
        "(`vjp`/`jvp` axes — Decision #24 truth), the native/placeholder "
        "adjoint classes parsed from `src/compiler/ir/AdjointInterface.cpp`, and "
        "(when they exist) the backward lanes of the runtime execution matrix "
        "and conformance fixtures. It is **not** a new source of truth. "
        "Regenerate via `scripts/check_generated_docs.sh --write`.",
        "",
        "## What the columns mean",
        "",
        "- **python_reference** — a numerically-checked Python VJP/JVP exists "
        "(the semantic oracle; *never* evidence of native compiler support).",
        "- **ir_adjoint** — `native`: `AutodiffPass` emits real backward Graph "
        "IR (static-shape W5 path); `placeholder`: `buildAdjoint` emits a "
        "`custom_adjoint_call` that round-trips to the Python VJP at runtime "
        "(**not** native); `none`: no IR adjoint.",
        "- **bwd_cpu_ir_oracle** — the compiler-emitted paired backward IR "
        "(`--tessera-autodiff-paired`) is numerically **interpreted on CPU and "
        "matches an independent NumPy VJP oracle** (Phase 3). Strictly weaker "
        "than native `oracle_proven`: it proves the *IR is correct*, not that a "
        "compiled/native backward executes. Proven by "
        "`tests/unit/test_autodiff_paired_cpu_oracle.py`.",
        "- **bwd_target_lowered / bwd_runtime_bound / bwd_oracle_proven / "
        "bwd_hardware_proven** — targets at which *backward* lowers / has a "
        "launch ABI / matches the oracle via **native** execution / is "
        "device-proven. **Empty today by construction** — the backward launch "
        "ABI, matrix column, and native oracle fixtures land in Phase 4; the "
        "ledger will fill from those sources, never by assertion.",
        "",
        "## Summary",
        "",
        f"- Differentiable families tracked: **{s['families']}**",
        f"- `python_reference` (Python VJP/JVP): **{s['python_reference']}**",
        f"- `ir_adjoint = native`: **{s['ir_adjoint_native']}** "
        f"({', '.join(sorted(native)) or '—'})",
        f"- `ir_adjoint = placeholder` (Python round-trip, not native): "
        f"**{s['ir_adjoint_placeholder']}** ({', '.join(sorted(placeholder)) or '—'})",
        f"- backward IR **oracle-verified on CPU** (interpreted): "
        f"**{s['backward_cpu_ir_oracle']}** ({', '.join(sorted(_BWD_IR_ORACLE_CPU)) or '—'})",
        f"- backward `runtime_bound` (native) on any target: **{s['backward_runtime_bound']}**",
        f"- backward `oracle_proven` (native) on any target: **{s['backward_oracle_proven']}**",
        f"- backward `hardware_proven` on any target: **{s['backward_hardware_proven']}**",
        "",
        "> **Headline:** the Python reference/oracle is broad, a handful of ops "
        "have a native IR adjoint, several more only *look* differentiable in "
        "IR but actually call back into Python. The `matmul`/`tanh`/`sigmoid` "
        "backward **IR is oracle-verified on CPU** (Phase 3). **Phase 4 (A2) has "
        "landed the first native backward**: the families below whose "
        "`bwd hardware_proven` column is non-empty execute their backward on "
        "real hardware — sourced from the runtime execution matrix's backward "
        "rows, not asserted. ROCm gfx1151 `flash_attn` (covering MHA + GQA/MQA) "
        "and `selective_ssm` (Mamba2) are the first two native backward launch "
        "lanes. Remaining families are still Phase 4/5 work.",
        "",
        "## Ledger",
        "",
        "| Family | Category | python_reference | ir_adjoint | bwd cpu_ir_oracle | bwd runtime_bound | bwd oracle_proven | bwd hardware_proven | Notes |",
        "|---|---|:--:|:--:|:--:|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.family}` | {r.category} | {r.python_reference} | "
            f"{r.ir_adjoint} | {'cpu' if r.bwd_cpu_ir_oracle else '—'} | "
            f"{_fmt_targets(r.bwd_runtime_bound) or '—'} | "
            f"{_fmt_targets(r.bwd_oracle_proven) or '—'} | "
            f"{_fmt_targets(r.bwd_hardware_proven) or '—'} | {r.notes} |"
        )
    lines.append("")
    lines.append(
        f"Backward-execution rungs are tracked against targets: "
        f"{', '.join(_TARGETS)}."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - manual inspection
    print(render_markdown())
