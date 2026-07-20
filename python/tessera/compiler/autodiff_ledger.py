"""Compiler-autodiff connection ledger — a *projection*, not a registry.

Phase 0 deliverable of
``docs/audit/compiler/AUTODIFF_UNIFICATION_PLAN.md``.

This module introduces **no new source of truth**.  It joins three existing
registries into one op-family × rung view whose single new contribution is the
dimension none of them makes explicit today: **forward vs. backward, per
target.**  Decision #24 is preserved — ``primitive_coverage`` stays the audit
truth for the ``vjp``/``jvp`` axes; this ledger only *reads* them.

The independent proof axes (see plan §3) and where each is sourced from:

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
``device_verified_jit`` generated target binary launched on the exact target
                        and numerically verified
``device_verified_abi`` shipped stable C ABI launched on the exact target and
                        numerically verified
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
# of these may reach each backward proof axis.
_TARGETS: tuple[str, ...] = (
    "cpu", "cpu_x86_64", "x86_avx512", "apple_cpu", "apple_gpu", "rocm_gfx1151",
    "nvidia_sm80", "nvidia_sm90", "nvidia_sm100", "nvidia_sm120",
)

# Ops whose `buildAdjoint` emits a native `tessera.custom_adjoint_call`
# placeholder that round-trips to the Python VJP registry at runtime.  These are
# NOT native Graph-IR adjoints — they are the compiler saying "ask Python".
_PLACEHOLDER_MACRO_RE = re.compile(r'POINTWISE_BUILD_ADJOINT\(\s*\w+\s*,\s*"([^"]+)"\s*\)')

# Every hand-written `<OpName>Op::buildAdjoint` definition. Whether it is a
# NATIVE adjoint (emits real backward Graph IR — matmul's transposed matmuls,
# tanh/sigmoid's W5 closed forms) or a PLACEHOLDER round-trip is decided by the
# body, NOT by the mere existence of the def: a hand-written body that itself
# constructs a `CustomAdjointCallOp` (LayerNormOp, SoftmaxOp) is a Python
# round-trip, exactly like the macro-generated ops — it only *looks* native
# because it has an explicit definition. (The macro body also textually contains
# `OPNAME::buildAdjoint`, but "OPNAME" carries no "Op::" so it never matches; the
# filter below is belt-and-suspenders.)
_EXPLICIT_BUILDADJOINT_RE = re.compile(r'(\w+)Op::buildAdjoint\b')

# The runtime VJP key a placeholder body carries (e.g. getStringAttr("layer_norm")).
# This — not the lowercased OpName — is the family key that matches the primitive
# (LayerNormOp emits key "layer_norm", not "layernorm"), so the ledger row lands
# on the right primitive instead of falling through to `none`.
_GETSTRINGATTR_RE = re.compile(r'getStringAttr\(\s*"([^"]+)"\s*\)')

# Phase 3 (2026-07-11) — families whose compiler-emitted backward IR is
# **oracle-verified on CPU by interpretation**: the actual
# `--tessera-autodiff-paired` output is numerically interpreted and its gradients
# match an independent NumPy VJP.  Proven by
# tests/unit/test_autodiff_paired_cpu_oracle.py.  This is the CPU IR-execution
# rung — strictly weaker than native `oracle_proven` (native LLVM/runtime
# execution, Phase 4) and device verification; it does NOT set those columns.
_BWD_IR_ORACLE_CPU: frozenset[str] = frozenset({"matmul", "tanh", "sigmoid"})

# Stable build identifiers used in the ledger. They describe the configuration
# that validates a claim, not the host directory in which somebody happened to
# build it.
_PYTHON_REFERENCE_BUILD = "python-unit-registry"
_CORE_ADJOINT_BUILD = "llvm23-core"


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


def _buildadjoint_body(text: str, sig_start: int) -> str:
    """The brace-balanced body of the `buildAdjoint` definition starting at
    ``sig_start``. Bounding to the real function body (not a coarse span to the
    next def) keeps a native op that happens to precede the placeholder macro /
    the `placeholderAdjoint` helper from inheriting their `CustomAdjointCallOp`
    text and being misread as a round-trip."""
    open_brace = text.find("{", sig_start)
    if open_brace < 0:
        return ""
    depth = 0
    for i in range(open_brace, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace:i + 1]
    return text[open_brace:]  # unbalanced — return the tail, caller still safe


def _cpp_op_family(name: str) -> str:
    """Convert an ODS C++ op stem (``AllReduce``) to ``all_reduce``."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _ir_adjoint_classes() -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(native_keys, placeholder_keys)`` parsed from the C++ source.

    ``native`` = the ``buildAdjoint`` body emits real backward Graph IR
    (static-shape path). ``placeholder`` = the body constructs a
    ``CustomAdjointCallOp`` → Python VJP round-trip. A hand-written body is
    classified by what it *emits*, not by the mere existence of the definition —
    so ``LayerNormOp`` / ``SoftmaxOp`` (explicit defs that emit placeholders)
    count as placeholder, matching their runtime behavior.
    """
    text = _read_adjoint_source()
    # Macro-generated placeholder ops: the 2nd macro arg is the family key.
    placeholder: set[str] = set(_PLACEHOLDER_MACRO_RE.findall(text))
    native: set[str] = set()
    for m in _EXPLICIT_BUILDADJOINT_RE.finditer(text):
        name = m.group(1)
        if name == "OPNAME":  # macro template token, not a real op
            continue
        body = _buildadjoint_body(text, m.start())
        if "CustomAdjointCallOp" in body:
            # Hand-written placeholder round-trip — key it by the runtime VJP
            # string it constructs (getStringAttr("...")) so it matches the
            # primitive; fall back to the OpName only if none is present.
            keym = _GETSTRINGATTR_RE.search(body)
            placeholder.add(keym.group(1) if keym else name.lower())
        else:
            native.add(_cpp_op_family(name))
    # A native op must never also be counted as placeholder.
    placeholder -= native
    if not native and not placeholder:
        raise LedgerError(
            f"parsed no adjoint keys from {_ADJOINT_CPP}; the macro/fallback "
            "conventions changed — update the ledger regexes rather than "
            "silently reporting zero IR adjoints."
        )
    return frozenset(native), frozenset(placeholder)


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
                 "bwd_oracle_proven", "bwd_device_verified_jit",
                 "bwd_device_verified_abi", "bwd_residual_policy",
                 "bwd_implementation", "build_evidence", "notes")

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
        self.bwd_device_verified_jit: tuple[str, ...] = ()
        self.bwd_device_verified_abi: tuple[str, ...] = ()
        self.bwd_residual_policy: tuple[str, ...] = ()
        self.bwd_implementation: tuple[str, ...] = ()
        self.build_evidence: tuple[str, ...] = ()
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
            notes = (
                "native static-shape adjoint (W5); dynamic → placeholder"
                if name in {"matmul", "tanh", "sigmoid"}
                else "native compiler adjoint"
            )
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
        builds: list[str] = []
        if differentiable:
            builds.append(f"python_reference={_PYTHON_REFERENCE_BUILD}")
        if ir_adjoint != "none":
            builds.append(f"ir_adjoint={_CORE_ADJOINT_BUILD}")
        if row.bwd_cpu_ir_oracle:
            builds.append(f"bwd_cpu_ir_oracle={_CORE_ADJOINT_BUILD}")
        # Fill the native backward rungs from the matrix, matching on the
        # primitive's name or graph_name against the row's op_family.
        info = next((bwd[k] for k in keys if k in bwd), None)
        if info is not None:
            row.bwd_runtime_bound = info["runtime_bound"]
            row.bwd_oracle_proven = info["oracle_proven"]
            row.bwd_device_verified_jit = info["device_verified_jit"]
            row.bwd_device_verified_abi = info["device_verified_abi"]
            row.bwd_residual_policy = info["residual_policies"]
            row.bwd_implementation = info["implementations"]
            # A generated target binary necessarily crossed target lowering.
            # A shipped ABI kernel does not, by itself, prove compiler lowering.
            row.bwd_target_lowered = row.bwd_device_verified_jit
            verified = tuple(sorted(
                set(row.bwd_device_verified_jit) | set(row.bwd_device_verified_abi)))
            if verified and "native backward" not in row.notes:
                targets = ", ".join(verified)
                row.notes = (row.notes + "; " if row.notes else "") + \
                    f"native backward executes on {targets} (Phase 4)"
            builds.extend(
                f"device[{item}]" for item in info["proof_builds"])
        row.build_evidence = tuple(builds)
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
        "backward_target_lowered": sum(1 for r in rows if r.bwd_target_lowered),
        "backward_runtime_bound": sum(1 for r in rows if r.bwd_runtime_bound),
        "backward_oracle_proven": sum(1 for r in rows if r.bwd_oracle_proven),
        "backward_device_verified_jit": sum(
            1 for r in rows if r.bwd_device_verified_jit),
        "backward_device_verified_abi": sum(
            1 for r in rows if r.bwd_device_verified_abi),
    }


def _fmt_targets(ts: tuple[str, ...]) -> str:
    return ",".join(ts) if ts else ""


def render_csv() -> str:
    rows = collect_rows()
    cols = (
        "family", "category", "python_reference", "ir_adjoint",
        "bwd_cpu_ir_oracle", "bwd_target_lowered", "bwd_runtime_bound",
        "bwd_oracle_proven", "bwd_device_verified_jit",
        "bwd_device_verified_abi", "build_evidence", "notes",
        "bwd_residual_policy", "bwd_implementation",
    )
    buf = _io.StringIO()
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(cols)
    for r in rows:
        writer.writerow([
            r.family, r.category, r.python_reference, r.ir_adjoint,
            "cpu" if r.bwd_cpu_ir_oracle else "",
            _fmt_targets(r.bwd_target_lowered), _fmt_targets(r.bwd_runtime_bound),
            _fmt_targets(r.bwd_oracle_proven),
            _fmt_targets(r.bwd_device_verified_jit),
            _fmt_targets(r.bwd_device_verified_abi),
            ";".join(r.build_evidence), r.notes,
            ";".join(r.bwd_residual_policy), ";".join(r.bwd_implementation),
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
        "One row per differentiable **op family**, over the independent proof axes of "
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
        "- **python_reference** — a Python VJP/JVP semantic reference is "
        "registered; this axis alone does not claim a numerical derivative "
        "test or native compiler support.",
        "- **ir_adjoint** — `native`: `AutodiffPass` emits real backward Graph "
        "IR (static-shape W5 path); `placeholder`: `buildAdjoint` emits a "
        "`custom_adjoint_call` that round-trips to the Python VJP at runtime "
        "(**not** native); `none`: no IR adjoint.",
        "- **bwd_cpu_ir_oracle** — the compiler-emitted paired backward IR "
        "(`--tessera-autodiff-paired`) is numerically **interpreted on CPU and "
        "matches an independent NumPy VJP oracle** (Phase 3). Strictly weaker "
        "than native `oracle_proven`: it proves the *IR is correct*, not that a "
        "device_verified_jit/native backward executes. Proven by "
        "`tests/unit/test_autodiff_paired_cpu_oracle.py`.",
        "- **bwd_target_lowered / bwd_runtime_bound / bwd_oracle_proven / "
        "bwd_device_verified_jit / bwd_device_verified_abi** — exact targets at "
        "which backward lowers / has a launch ABI / matches an independent "
        "oracle / is verified through a generated binary or shipped stable C "
        "ABI. `execution_kind` alone proves none of the device axes. Every "
        "device-verified row must name an exact evidence target and checked-in "
        "execute-and-compare fixture in the runtime execution matrix.",
        "- **bwd_residual_policy / bwd_implementation** — the selected "
        "per-target residual contract and whether it is a dedicated backward "
        "kernel or a composition of existing native lanes.",
        "- **build_evidence** — the stable build configuration that validates "
        "each populated claim. `llvm23-core` owns compiler adjoint/paired-IR "
        "claims; exact device rows carry their build from the execution matrix.",
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
        f"- backward `target_lowered` on any exact target: "
        f"**{s['backward_target_lowered']}**",
        f"- backward `runtime_bound` (native) on any target: **{s['backward_runtime_bound']}**",
        f"- backward `oracle_proven` (native) on any target: **{s['backward_oracle_proven']}**",
        f"- backward `device_verified_jit` on any exact target: "
        f"**{s['backward_device_verified_jit']}**",
        f"- backward `device_verified_abi` on any exact target: "
        f"**{s['backward_device_verified_abi']}**",
        "",
        "> **Headline:** the Python reference/oracle is broad, a handful of ops "
        "have a native IR adjoint, several more only *look* differentiable in "
        "IR but actually call back into Python. The `matmul`/`tanh`/`sigmoid` "
        "backward **IR is oracle-verified on CPU** (Phase 3). **Phase 4 A1–A4 "
        "have landed native backward proof, alias/composition identity, and "
        "per-target residual policy**. The leaders listed below are derived "
        "from the exact-target proof columns; no family or architecture is "
        "hard-coded into this headline. Remaining families are Phase 4/5 work.",
        "",
        "### Device-verified leaders",
        "",
        *[
            f"- `{r.family}` — "
            + "; ".join(filter(None, (
                ("device_verified_jit: " + _fmt_targets(r.bwd_device_verified_jit))
                if r.bwd_device_verified_jit else "",
                ("device_verified_abi: " + _fmt_targets(r.bwd_device_verified_abi))
                if r.bwd_device_verified_abi else "",
            )))
            for r in rows
            if r.bwd_device_verified_jit or r.bwd_device_verified_abi
        ],
        "",
        "## Ledger",
        "",
        "| Family | Category | python_reference | ir_adjoint | bwd cpu_ir_oracle | bwd target_lowered | bwd runtime_bound | bwd oracle_proven | bwd device_verified_jit | bwd device_verified_abi | Residual policy | Implementation | Build evidence | Notes |",
        "|---|---|:--:|:--:|:--:|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.family}` | {r.category} | {r.python_reference} | "
            f"{r.ir_adjoint} | {'cpu' if r.bwd_cpu_ir_oracle else '—'} | "
            f"{_fmt_targets(r.bwd_target_lowered) or '—'} | "
            f"{_fmt_targets(r.bwd_runtime_bound) or '—'} | "
            f"{_fmt_targets(r.bwd_oracle_proven) or '—'} | "
            f"{_fmt_targets(r.bwd_device_verified_jit) or '—'} | "
            f"{_fmt_targets(r.bwd_device_verified_abi) or '—'} | "
            f"{'; '.join(r.bwd_residual_policy) or '—'} | "
            f"{'; '.join(r.bwd_implementation) or '—'} | "
            f"{'; '.join(r.build_evidence) or '—'} | {r.notes} |"
        )
    lines.append("")
    lines.append(
        f"Backward-execution rungs are tracked against targets: "
        f"{', '.join(_TARGETS)}."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - manual inspection
    print(render_markdown())
