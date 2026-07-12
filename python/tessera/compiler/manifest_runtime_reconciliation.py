"""Manifest-vs-runtime reconciliation audit.

The backend manifest (``backend_manifest``) is the *audit* truth for the
per-``(op, target)`` ``backend_kernel`` status. The **runtime** (``runtime.py``)
is where ops actually dispatch to native device lanes, gated by op-name sets
(``_ROCM_NORM_OPS``, ``_X86_FA_OPS``, ``_SM_OPS``, …). The two drift: an op can
gain a runtime lane while its manifest row still reads ``reference`` / missing —
so the enablement dashboards *under*-count native coverage (the manifest lags the
runtime, found repeatedly: ``relu``, ``online_softmax``, ``lightning_attention``,
the DeltaNet family, …).

This module makes that drift **computable**. It parses the runtime's op-name
dispatch sets (AST — no execution, no hardware) into a ``(op, target)`` map of
"the runtime can dispatch this natively", cross-references the manifest, and
reports the reconciliation gaps: ``(op, target)`` the runtime dispatches but the
manifest does **not** call native. Each gap is a candidate cheap close (declare
the manifest row + a fixture) or a real "runtime overclaims" bug.

It is **read-only** — no new source of truth. The manifest stays authoritative;
this cross-checks it against the runtime and surfaces the delta honestly.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from . import backend_manifest

_RUNTIME = Path(__file__).resolve().parent.parent / "runtime.py"

#: manifest statuses that count as a genuine native kernel for a target.
_NATIVE_STATUSES = frozenset({"fused", "compiled", "hardware_verified", "packaged"})

#: canonical targets this audit reconciles (the ones with runtime op-name lanes).
_TARGETS = ("rocm", "x86", "apple_gpu", "apple_cpu")

# var-name prefix → target, for module-level op-name sets whose name states it.
_PREFIX_TARGET = (
    ("_ROCM", "rocm"),
    ("_X86", "x86"),
    ("_APPLE_GPU", "apple_gpu"),
    ("_APPLE_CPU", "apple_cpu"),
    ("_APPLE", "apple_gpu"),
    ("_MPS", "apple_gpu"),
    ("_MSL", "apple_gpu"),
)


class ReconciliationError(RuntimeError):
    """The audit cannot read a source it must join (Decision #26: never a silent
    no-op)."""


def _target_from_name(name: str) -> str | None:
    up = name.upper()
    for pref, tgt in _PREFIX_TARGET:
        if up.startswith(pref) or up.startswith(pref.lstrip("_")):
            return tgt
    return None


def _target_from_func(func_name: str | None) -> str | None:
    if not func_name:
        return None
    m = re.search(r"_(rocm|x86|apple_gpu|apple_cpu|apple)", func_name)
    if not m:
        return None
    t = m.group(1)
    return "apple_gpu" if t == "apple" else t


def _ops_in_value(node: ast.AST) -> list[str]:
    """The ``tessera.<op>`` op names inside an op-gate value.

    Handles the literal container forms (tuple/list/set — a dict uses its *keys*)
    **and** the constructor-call form ``frozenset({...})`` / ``set([...])`` /
    ``tuple(...)`` / ``list(...)``: the call is unwrapped to its container
    argument first. Without this, a gate written as
    ``_APPLE_CPU_ACCELERATE_OPS = frozenset({"tessera.matmul", …})`` parses as an
    ``ast.Call`` that matches none of the literal branches and is silently
    dropped — a false-negative that hides real manifest lag. Set unions
    (``A | {"tessera.x"}``) are walked on both sides so composed gates count too.
    """
    # Unwrap ``frozenset(...)`` / ``set(...)`` / ``tuple(...)`` / ``list(...)``.
    if isinstance(node, ast.Call):
        if (isinstance(node.func, ast.Name)
                and node.func.id in {"frozenset", "set", "tuple", "list"}
                and node.args):
            return _ops_in_value(node.args[0])
        return []
    # Composed gates: ``_BASE_OPS | {"tessera.x"}`` (Name sides contribute none).
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _ops_in_value(node.left) + _ops_in_value(node.right)

    ops: list[str] = []
    consts: list[ast.expr] = []
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        consts = list(node.elts)
    elif isinstance(node, ast.Dict):
        # dict keys can be None for a ``**expansion`` entry — skip those.
        consts = [k for k in node.keys if k is not None]
    for c in consts:
        if isinstance(c, ast.Constant) and isinstance(c.value, str) \
                and c.value.startswith("tessera."):
            ops.append(c.value[len("tessera."):])
    return ops


def runtime_dispatch_map() -> dict[str, set[str]]:
    """Parse the runtime's op-name dispatch sets → ``{target: {op, …}}``.

    An op-name set is any assignment to a name containing ``OPS`` whose value is a
    tuple/list/set/dict of ``tessera.*`` strings (the executor's ``op_name not in
    <set>`` acceptance gate). The target comes from the var-name prefix
    (``_ROCM_*`` / ``_X86_*`` / ``_APPLE_*``) or, for ambiguous names, the
    enclosing ``_execute_<target>_*`` function."""
    if not _RUNTIME.is_file():
        raise ReconciliationError(
            f"cannot run the reconciliation audit: {_RUNTIME} is missing — the "
            "runtime dispatch map is parsed from it.")
    tree = ast.parse(_RUNTIME.read_text(encoding="utf-8"), filename=str(_RUNTIME))

    out: dict[str, set[str]] = {t: set() for t in _TARGETS}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.func: str | None = None

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            prev = self.func
            self.func = node.name
            self.generic_visit(node)
            self.func = prev

        def visit_Assign(self, node: ast.Assign) -> None:
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for nm in names:
                if "OPS" not in nm.upper():
                    continue
                ops = _ops_in_value(node.value)
                if not ops:
                    continue
                tgt = _target_from_name(nm) or _target_from_func(self.func)
                if tgt in out:
                    out[tgt].update(ops)
            self.generic_visit(node)

        # AnnAssign (e.g. ``_ROCM_NORM_OPS: dict[...] = {...}``)
        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.target, ast.Name) and node.value is not None:
                nm = node.target.id
                if "OPS" in nm.upper():
                    ops = _ops_in_value(node.value)
                    tgt = _target_from_name(nm) or _target_from_func(self.func)
                    if ops and tgt in out:
                        out[tgt].update(ops)
            self.generic_visit(node)

    Visitor().visit(tree)
    if not any(out.values()):
        raise ReconciliationError(
            f"parsed no op-name dispatch sets from {_RUNTIME}; the executor "
            "op-gate convention changed — update this audit rather than silently "
            "reporting zero runtime lanes.")
    return out


def _manifest_status(op: str, target: str) -> str | None:
    try:
        rows = backend_manifest.manifest_for(op)
    except Exception:
        return None
    for e in rows:
        if e.target == target:
            return e.status
    return None


def _canonical_primitives() -> frozenset[str]:
    """The op names the manifest is *meant* to track — the primitive_coverage
    registry (Decision #24 audit truth), by both ``name`` and ``graph_name``.

    Runtime dispatch sets also accept numpy-style **aliases** (``divide`` for
    ``div``, ``multiply`` for ``mul``, ``swish`` for ``silu``, ``swiglu`` …) that
    are *not* separate primitives — their canonical form is what the manifest
    declares. Filtering to canonical primitives keeps the real gaps (a genuine
    primitive that executes but is undeclared, e.g. ``relu``) and drops the
    alias noise."""
    from . import primitive_coverage
    names: set[str] = set()
    for cov in primitive_coverage.all_primitive_coverages().values():
        names.add(cov.name)
        if cov.graph_name:
            names.add(cov.graph_name)
            if cov.graph_name.startswith("tessera."):
                names.add(cov.graph_name[len("tessera."):])
    return frozenset(names)


class Gap:
    """One reconciliation gap: the runtime dispatches ``op`` on ``target`` but the
    manifest does not call it native."""

    __slots__ = ("op", "target", "manifest_status")

    def __init__(self, op: str, target: str, manifest_status: str | None) -> None:
        self.op = op
        self.target = target
        self.manifest_status = manifest_status


def reconciliation_gaps() -> list[Gap]:
    """``(op, target)`` the runtime dispatches natively but the manifest declares
    non-native (``reference`` / ``artifact_only`` / missing / …) — sorted.

    Only **canonical primitives** are reported (see :func:`_canonical_primitives`)
    — numpy-style aliases the runtime also accepts are dropped, since their
    canonical form is what the manifest tracks."""
    dispatch = runtime_dispatch_map()
    canonical = _canonical_primitives()
    gaps: list[Gap] = []
    for target in _TARGETS:
        for op in sorted(dispatch.get(target, ())):
            if op not in canonical:
                continue
            status = _manifest_status(op, target)
            if status not in _NATIVE_STATUSES:
                gaps.append(Gap(op, target, status))
    gaps.sort(key=lambda g: (g.target, g.op))
    return gaps


def render_markdown() -> str:
    dispatch = runtime_dispatch_map()
    gaps = reconciliation_gaps()
    by_target: dict[str, list[Gap]] = {t: [] for t in _TARGETS}
    for g in gaps:
        by_target[g.target].append(g)
    lines = [
        "<!-- AUTO-GENERATED by tessera.compiler.manifest_runtime_reconciliation "
        "— do not edit. Regenerate via scripts/check_generated_docs.sh --write -->",
        "",
        "# Manifest-vs-Runtime Reconciliation (generated)",
        "",
        "Ops the **runtime** dispatches to a native device lane (parsed from the "
        "`op_name not in _*_OPS` gates in `runtime.py`) whose **manifest** row is "
        "*not* native (`reference` / `artifact_only` / missing). Each is a "
        "candidate **cheap close** (declare the manifest row + a fixture) or a "
        "runtime-overclaims bug. The manifest stays audit truth; this only "
        "surfaces the drift (Decision #26).",
        "",
        "## Summary",
        "",
        "- Runtime native-dispatch lanes parsed: "
        + ", ".join(f"{t}={len(dispatch.get(t, ()))}" for t in _TARGETS) + ".",
        f"- **Reconciliation gaps (runtime native, manifest not): {len(gaps)}**.",
        "",
    ]
    for t in _TARGETS:
        rows = by_target[t]
        lines.append(f"## `{t}` — {len(rows)} gap(s)")
        lines.append("")
        if not rows:
            lines.append("_None — manifest matches the runtime for this target._")
            lines.append("")
            continue
        lines.append("| Op | Manifest status |")
        lines.append("|----|-----------------|")
        for g in rows:
            lines.append(f"| `{g.op}` | {g.manifest_status or '— (missing)'} |")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - manual inspection
    print(render_markdown())
