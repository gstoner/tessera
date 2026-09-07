"""Gap analysis for pruning the bootstrap (Python) backend compiler.

Direction (2026-08-30): the original Python->backend path was a **bootstrap
compiler**. The architecture is core MLIR/LLVM -- Graph -> Schedule -> Tile ->
Target driven by ``tessera-opt`` -- and the per-backend ``package_*`` families
are the prune target. This dashboard answers the question that has to be
settled *before* any of that is deleted: **which families does the mainline
compiler already cover, and which would lose their only lowering?**

It is deliberately a generated dashboard rather than a hand-written table.
The whole point is to watch a gap close, and a hand table would be stale by
the second landing (Decision #26).

What is derived vs declared
---------------------------
Everything countable is **derived by AST** from the live sources: the
``package_*`` inventory per backend, the family names ``native_package_kind``
returns, and the ``supports_scheduled_*`` predicates the driver consults.
Nothing here is a transcribed number.

The one thing that cannot be derived is *which compiled predicate serves which
family*, because that correspondence lives in the driver's control flow rather
than in any table. It is therefore **declared** in ``_FAMILY_TO_COMPILED``
below and **verified**: if a named module or predicate stops existing, the
generator raises rather than silently reporting a family as covered. A
mis-declared mapping would produce exactly the false "already covered" that
would make a prune lossy, so it fails closed.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

_COMPILER = Path(__file__).resolve().parent

#: Backend bootstrap packager modules, in fleet-lead order.
_BACKEND_MODULES: tuple[tuple[str, str], ...] = (
    ("nvidia_sm120", "nvidia_native.py"),
    ("rocm_gfx1151", "rocm_native.py"),
    ("x86", "x86_native.py"),
    ("apple_cpu", "apple_cpu_native.py"),
    ("apple_gpu", "apple_native.py"),
)

#: family name -> (scheduled module stem, admission predicate).
#: DECLARED, not derived -- the correspondence lives in driver.py control flow.
#: Verified for existence below; a rename fails the generator rather than
#: silently marking a family covered.
_FAMILY_TO_COMPILED: dict[str, tuple[str, str]] = {
    "matmul": ("scheduled_matmul", "supports_scheduled_matmul"),
    "attention": ("scheduled_attention", "supports_scheduled_attention"),
    "attention_backward": (
        "scheduled_attention_backward",
        "supports_scheduled_attention_backward",
    ),
    "depth_attention": (
        "scheduled_depth_attention",
        "supports_scheduled_depth_attention",
    ),
}

#: The generic compiled fallback the driver tries last. It admits by op
#: structure rather than by family name, so it is reported separately: a
#: family it happens to accept is covered, but not *by that family's name*.
_GENERIC_COMPILED = ("scheduled_kernel", "supports_scheduled_kernel")


@dataclass(frozen=True)
class BackendInventory:
    """What one backend's bootstrap module contains."""

    target: str
    module: str
    #: (name, first-parameter type) for every package_* in the module.
    packagers: tuple[tuple[str, str], ...]
    families: tuple[str, ...]
    lines: int
    unresolved_returns: tuple[str, ...] = ()
    #: bootstrap packager name -> what it actually does.
    kinds: dict[str, str] = field(default_factory=dict)

    @property
    def bootstrap(self) -> tuple[str, ...]:
        """Packagers that re-enter Graph IR — the prune target."""
        return tuple(n for n, t in self.packagers if _is_bootstrap(t))

    @property
    def compiled_packagers(self) -> tuple[str, ...]:
        """Packagers declaring scheduled artifact inputs; body replay still needs proof."""
        return tuple(n for n, t in self.packagers if _is_artifact(t))

    @property
    def unclassified_packagers(self) -> tuple[str, ...]:
        """Unknown/raw inputs are not proof of artifact consumption."""
        return tuple(n for n, t in self.packagers
                     if not _is_bootstrap(t) and not _is_artifact(t))


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None


def _first_param_type(node: ast.FunctionDef) -> str:
    """Annotation of the first positional parameter, or '' if unannotated."""
    args = node.args.args or node.args.posonlyargs
    if not args or args[0].annotation is None:
        return ""
    try:
        return ast.unparse(args[0].annotation)
    except Exception:  # pragma: no cover - defensive on exotic annotations
        return ""


def _packagers(tree: ast.Module, source: str = "") -> tuple[tuple[str, str], ...]:
    """(name, first-parameter type) for every ``package_*`` function.

    The first parameter partitions the declared input types; it does not prove
    what the body consumes. Unknown types remain a separate population:

    * ``GraphIRModule`` -- the function reads Graph IR and emits target code
      itself, bypassing Schedule and Tile. That is the bootstrap compiler.
    * ``Scheduled*Artifact`` -- the function packages an artifact the compiled
      route already lowered ("without Graph re-entry"). That is the mainline
      compiler's packaging step and is NOT a prune target.

    An earlier version of this audit classified by name suffix and wrongly
    counted six compiled-route packagers as bootstrap surface.
    """
    return tuple(
        sorted(
            (node.name, _first_param_type(node))
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("package_")
        )
    )


def _packager_kinds(tree: ast.Module, source: str) -> dict[str, str]:
    """name -> kind, for every bootstrap packager in the module."""
    lines = source.split("\n")
    out: dict[str, str] = {}
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("package_")):
            continue
        if not _is_bootstrap(_first_param_type(node)):
            continue
        body = "\n".join(lines[node.lineno - 1:node.end_lineno])
        out[node.name] = _packager_kind(body)
    return out


def _is_artifact(param_type: str) -> bool:
    # Inventory of typed inputs only, not a semantic data-flow proof.
    return bool(re.fullmatch(r"Scheduled[A-Za-z0-9_]*Artifact", param_type.strip("'\"")))


def _computed_classifier_families(tree: ast.Module) -> dict[str, str]:
    """Resolve only the two known, table-bounded Apple classifier expressions.

    Read literal producer tables; never execute backend modules or guess the
    domain of an arbitrary computed return. Unknown syntax remains unresolved.
    """
    tables = {}
    for node in tree.body:
        name = (node.targets[0].id if isinstance(node, ast.Assign)
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                else node.target.id if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name) else None)
        value = getattr(node, "value", None)
        if name and isinstance(value, ast.Dict):
            try:
                tables[name] = ast.literal_eval(value)
            except (ValueError, TypeError):
                pass
    classifier = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == 'native_package_kind'), None)
    if classifier is None:
        return {}
    expressions = {ast.unparse(n.value) for n in ast.walk(classifier)
                   if isinstance(n, ast.Return) and n.value is not None}
    source = ast.unparse(classifier)
    symbols = tables.get('_VALUE_SYMBOLS')
    if not isinstance(symbols, dict):
        return {}
    cpu = "op.op_name.removeprefix('tessera.')"
    gpu = "'value_' + op.op_name.removeprefix('tessera.').replace('.', '_')"
    if cpu in expressions and '_entry_for(op.op_name, dtype)' in source:
        lowp = tables.get('_LOW_PRECISION_MATMUL_SYMBOLS', {})
        names = set(symbols) | {key[0] for key in lowp if isinstance(key, tuple)}
        return {name.removeprefix('tessera.'): name for name in sorted(names)}
    if gpu in expressions and 'value_descriptor_state(op.op_name)' in source:
        states = tables.get('APPLE_VALUE_DESCRIPTOR_STATES', {})
        excluded = set()
        for branch in ast.walk(classifier):
            if (isinstance(branch, ast.If) and isinstance(branch.test, ast.Compare)
                    and ast.unparse(branch.test.left) == 'op.op_name'
                    and len(branch.test.ops) == 1 and isinstance(branch.test.ops[0], ast.In)
                    and isinstance(branch.test.comparators[0], ast.Set)
                    and len(branch.body) == 1 and isinstance(branch.body[0], ast.Return)
                    and isinstance(branch.body[0].value, ast.Constant)
                    and branch.body[0].value.value is None):
                excluded.update(ast.literal_eval(branch.test.comparators[0]))
        names = {name for name, state in states.items() if state == 'descriptor_ready'} - excluded
        return {'value_' + name.removeprefix('tessera.').replace('.', '_'): name
                for name in sorted(names)}
    return {}


def _unresolved_classifier_returns(tree: ast.Module) -> tuple[str, ...]:
    known = _computed_classifier_families(tree)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "native_package_kind":
            expressions = tuple(ast.unparse(sub.value) for sub in ast.walk(node)
                                if isinstance(sub, ast.Return) and sub.value is not None
                                and not isinstance(sub.value, ast.Constant))
            bounded = {"op.op_name.removeprefix('tessera.')",
                       "'value_' + op.op_name.removeprefix('tessera.').replace('.', '_')"}
            return tuple(e for e in expressions if not known or e not in bounded)
    return ()


def _is_bootstrap(param_type: str) -> bool:
    """Whether a packager re-enters Graph IR rather than consuming an artifact."""
    return "GraphIRModule" in param_type


#: Substrings that mean a packager reaches outside the compiler for its code:
#: a runtime compiler, a vendor library, a shipped shared object, or raw device
#: source. Absence of all of them is what distinguishes "constructs IR" from
#: "delegates".
#: Any of these means the packager built IR and ran it through the tool.
#: A regex because each backend spells the helper differently.
_IR_CONSTRUCTION_RE = re.compile(
    r"_compile_\w*_ir\b|emit_\w*_(?:tile|graph)_ir\b|tessera-opt|run_tessera_opt"
)

_DELEGATION_MARKERS: tuple[str, ...] = (
    "nvrtc", "NVRTC", "hiprtc", "HIPRTC",
    "cublas", "cudnn", "cutlass", "rocblas", "hipblas",
    ".so", "__global__", "ptx_emit", "libtessera",
)


def _packager_kind(body: str) -> str:
    """What a bootstrap packager actually does, which decides how it is retired.

    The distinction matters because the two kinds take opposite treatment and
    a plan that conflates them scopes the wrong work:

    * ``constructs_tile_ir`` -- builds Tile IR in Python and then compiles it
      through ``tessera-opt``. The MLIR pipeline *is* running from Tile
      onward; what bypasses it is Graph -> Schedule -> Tile. These are
      **absorbed** by growing the compiled route, not re-expressed as
      delegates. There is no fast path here to preserve.
    * ``delegates`` -- reaches a runtime compiler, vendor library, or shipped
      object. This is the genuine fast path, and the one the Target IR
      delegation boundary (`kernel_call` / `inline_ptx`) exists to carry.
    * ``both`` -- constructs IR *and* reaches outside; reported distinctly
      rather than forced into one bucket, because it needs both treatments.
    * ``other`` -- typically a thin dtype wrapper or dispatcher over one of
      the above; it retires with whatever it forwards to.

    Measured 2026-08-30, this overturned a planning assumption: the NVIDIA
    bootstrap packagers were described as containing "vendor libraries,
    hand-tuned kernels, inline PTX". None of them delegate -- 13 of 19
    construct Tile IR and compile it. NVIDIA's real delegation surface is
    `ptx_emit.py`, `emit/nvidia_cuda.py` and `runtime.py`: different files,
    different work.

    The IR-construction markers are deliberately a regex rather than one
    literal. Each backend spells its helper differently -- NVIDIA
    ``_compile_tile_ir``, ROCm ``_compile_attention_tile_ir``, x86
    ``emit_matmul_tile_ir`` plus a direct ``tessera-opt`` invocation -- and a
    detector keyed to one spelling silently classified the other two as
    ``other``, which is a taxonomy that reports mostly nothing.
    """
    constructs = bool(_IR_CONSTRUCTION_RE.search(body))
    delegates = any(marker in body for marker in _DELEGATION_MARKERS)
    if constructs and delegates:
        return "both"
    if constructs:
        return "constructs_tile_ir"
    if delegates:
        return "delegates"
    return "other"

def _classified_families(tree: ast.Module) -> tuple[str, ...]:
    """Literal and known table-bounded families from ``native_package_kind``.

    These are the families the backend's own classifier recognises, which is
    the set the driver dispatches on.
    """
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "native_package_kind":
            continue
        names = [
            sub.value.value
            for sub in ast.walk(node)
            if isinstance(sub, ast.Return)
            and isinstance(sub.value, ast.Constant)
            and isinstance(sub.value.value, str)
        ]
        return tuple(dict.fromkeys([*names, *_computed_classifier_families(tree)]))
    return ()


def _predicate_exists(module_stem: str, predicate: str) -> bool:
    tree = _parse(_COMPILER / f"{module_stem}.py")
    if tree is None:
        return False
    return any(
        isinstance(node, ast.FunctionDef) and node.name == predicate
        for node in tree.body
    )


def collect_inventories() -> tuple[BackendInventory, ...]:
    """Derive each backend's bootstrap surface from its live source."""
    out: list[BackendInventory] = []
    for target, filename in _BACKEND_MODULES:
        path = _COMPILER / filename
        tree = _parse(path)
        if tree is None:
            raise ValueError(f"cannot inventory backend source: {path}")
        text = path.read_text(encoding="utf-8")
        out.append(
            BackendInventory(
                target=target,
                module=filename,
                packagers=_packagers(tree),
                families=_classified_families(tree),
                lines=text.count("\n") + 1,
                kinds=_packager_kinds(tree, text),
                unresolved_returns=_unresolved_classifier_returns(tree),
            )
        )
    return tuple(out)


def verify_declared_mapping() -> None:
    """Fail closed if a declared compiled route no longer exists.

    A stale entry here would report a family as already covered by the
    mainline compiler when it is not -- the single error that would make a
    prune silently lossy.
    """
    missing = [
        f"{family} -> {mod}.{pred}"
        for family, (mod, pred) in _FAMILY_TO_COMPILED.items()
        if not _predicate_exists(mod, pred)
    ]
    generic_mod, generic_pred = _GENERIC_COMPILED
    if not _predicate_exists(generic_mod, generic_pred):
        missing.append(f"<generic> -> {generic_mod}.{generic_pred}")
    if missing:
        raise RuntimeError(
            "bootstrap_prune_audit: declared compiled routes no longer exist: "
            + "; ".join(sorted(missing))
            + ". Update _FAMILY_TO_COMPILED rather than letting a family be "
            "reported as covered when it is not."
        )


def _target_family_route(target: str, family: str) -> tuple[str, str] | None:
    """A family mapping alone cannot establish a consumer on another target."""
    route = _FAMILY_TO_COMPILED.get(family)
    filename = dict(_BACKEND_MODULES).get(target)
    if route is None or filename is None:
        return None
    tree = _parse(_COMPILER / filename)
    consumer = 'package_' + route[0]
    if tree is None or not any(isinstance(n, ast.FunctionDef) and n.name == consumer
                               for n in tree.body):
        return None
    return route


def family_rows() -> tuple[tuple[str, str, str, str], ...]:
    """(target, family, compiled_route, status) for every classified family."""
    verify_declared_mapping()
    rows: list[tuple[str, str, str, str]] = []
    for inv in collect_inventories():
        for family in inv.families:
            route = _target_family_route(inv.target, family)
            if route is None:
                rows.append((inv.target, family, "—", "gap"))
            else:
                rows.append(
                    (inv.target, family, f"{route[0]}.{route[1]}", "compiled")
                )
    return tuple(rows)


def orphan_packagers() -> tuple[tuple[str, str], ...]:
    """Packagers whose name matches no classified family.

    ``package_<family>`` is the naming convention, so a packager with no
    matching family is reached some other way -- a sibling entry point, a
    dtype specialisation, or dead code. Each needs its own disposition before
    the prune; none can be assumed covered.
    """
    out: list[tuple[str, str]] = []
    for inv in collect_inventories():
        families = set(inv.families)
        for packager in inv.bootstrap:
            suffix = packager.removeprefix("package_")
            if suffix in families or suffix == "native":
                continue
            out.append((inv.target, packager))
    return tuple(out)


def summary() -> dict[str, int]:
    rows = family_rows()
    inventories = collect_inventories()
    return {
        "backends": len(inventories),
        "packagers": sum(len(i.packagers) for i in inventories),
        "bootstrap": sum(len(i.bootstrap) for i in inventories),
        "compiled_packagers": sum(len(i.compiled_packagers) for i in inventories),
        "unclassified_packagers": sum(len(i.unclassified_packagers) for i in inventories),
        "lines": sum(i.lines for i in inventories),
        "families": len(rows),
        "compiled": sum(1 for r in rows if r[3] == "compiled"),
        "gap": sum(1 for r in rows if r[3] == "gap"),
        "orphan_packagers": len(orphan_packagers()),
        "constructs_tile_ir": sum(
            sum(1 for k in i.kinds.values() if k == "constructs_tile_ir")
            for i in inventories),
        "delegates": sum(
            sum(1 for k in i.kinds.values() if k == "delegates")
            for i in inventories),
        "both": sum(
            sum(1 for k in i.kinds.values() if k == "both") for i in inventories),
        "other_kind": sum(
            sum(1 for k in i.kinds.values() if k == "other") for i in inventories),
    }


def render_markdown() -> str:
    inventories = collect_inventories()
    rows = family_rows()
    orphans = orphan_packagers()
    s = summary()

    out: list[str] = [
        "# Bootstrap Prune — Mainline Coverage Gap",
        "",
        "**Generated. Do not hand-edit.** Regenerate with",
        "`python -m tessera.compiler.generated_docs --write`.",
        "",
        "The Python per-backend `package_*` inventory includes bootstrap and",
        "artifact packagers; the architecture is core MLIR/LLVM (Graph → Schedule →",
        "Tile → Target via `tessera-opt`). This dashboard answers what must be",
        "settled before any of it is deleted: **which families does the",
        "mainline compiler already cover, and which would lose their only",
        "lowering?** Decision #31's ordering caveat is the rule — a duplicate",
        "authority is removed only after the survivor is proven to carry what",
        "it carried.",
        "",
        "A `gap` row is *not* a defect. It is scope: work the mainline",
        "compiler must absorb, or a fast path that must be re-expressed",
        "through a declared Target IR boundary (Decision #28 Tier 3) before",
        "the bootstrap row can go.",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|---|---|",
        f"| Backends with a bootstrap module | {s['backends']} |",
        f"| `package_*` functions total | {s['packagers']} |",
        f"| — **bootstrap** (re-enter Graph IR; prune target) | {s['bootstrap']} |",
        f"|   ·  of the bootstrap, construct Tile IR then run `tessera-opt` | {s['constructs_tile_ir']} |",
        f"|   ·  of the bootstrap, **delegate** (runtime compiler / library / object) | {s['delegates']} |",
        f"|   ·  of the bootstrap, both | {s['both']} |",
        f"|   ·  of the bootstrap, other (wrapper / dispatcher) | {s['other_kind']} |",
        f"| — typed scheduled-artifact inputs (consumption needs verification) | {s['compiled_packagers']} |",
        f"| — unclassified/raw inputs (not assumed compiled) | {s['unclassified_packagers']} |",
        f"| Lines in those modules | {s['lines']} |",
        f"| Classified family/target candidates (shape admission not implied) | {s['families']} |",
        f"| — covered by a compiled route | {s['compiled']} |",
        f"| — **gap (no declared family route)** | {s['gap']} |",
        f"| Packagers matching no family | {s['orphan_packagers']} |",
        "",
        "## Per-backend bootstrap surface",
        "",
        "| Target | Module | Graph input | Typed artifact input | Unknown/raw input | Family candidates | Lines |",
        "|---|---|---|---|---|---|---|",
    ]
    for inv in inventories:
        out.append(
            f"| `{inv.target}` | `{inv.module}` | {len(inv.bootstrap)} "
            f"| {len(inv.compiled_packagers)} | {len(inv.unclassified_packagers)} | {len(inv.families)} | {inv.lines} |"
        )

    out += ["", "## Census limits", "",
            "Input annotations are inventory evidence, not proof of semantic authority.",
            "Any, unannotated and raw-IR inputs remain unclassified; inspect their producers and consumers.",
            "Known Apple computed returns are derived from their producer tables; other computed returns remain unresolved.",
            "A missing family mapping is not proof that no generic scheduled route accepts it.",
            "", "| Target | Unresolved classifier return |", "|---|---|"]
    for inv in inventories:
        for expression in inv.unresolved_returns:
            out.append(f"| `{inv.target}` | `{expression}` |")
    out += [
        "",
        "## Family coverage",
        "",
        "`compiled` means a family has a declared admission-predicate mapping.",
        "The target module must also define the corresponding package consumer.",
        "Actual driver paths, shapes and policies require separate checks.",
        "It does **not** assert the compiled route reaches parity on",
        "every shape and dtype — that is per-family evidence the backend",
        "queues own.",
        "",
        "| Target | Family | Compiled route | Status |",
        "|---|---|---|---|",
    ]
    for target, family, route, status in rows:
        mark = "✅ compiled" if status == "compiled" else "🔴 **gap**"
        route_cell = f"`{route}`" if route != "—" else "—"
        out.append(f"| `{target}` | `{family}` | {route_cell} | {mark} |")

    out += [
        "",
        "## Packagers matching no classified family",
        "",
        "`package_<family>` is the convention, so these are reached by some",
        "other entry point — a sibling call site, a dtype specialisation, or",
        "dead code. Each needs its own disposition; none may be assumed",
        "covered because a same-named family is compiled.",
        "",
        "| Target | Packager |",
        "|---|---|",
    ]
    for target, packager in orphans:
        out.append(f"| `{target}` | `{packager}` |")

    out += [
        "",
        "## How to read a closing gap",
        "",
        "**Measured, and it redirects the work:** the bootstrap surface is",
        "overwhelmingly *IR-constructing*, not delegating. Most packagers build",
        "Tile IR in Python and then compile it through `tessera-opt`, so the",
        "MLIR pipeline already runs from Tile onward and what bypasses it is",
        "Graph → Schedule → Tile. Those retire by **absorption**, and there is",
        "no fast path in them to preserve. The genuine delegation surface —",
        "the one the Target IR boundary exists for — is elsewhere",
        "(`ptx_emit.py`, `emit/nvidia_cuda.py`, `runtime.py`). A plan that",
        "treats the whole bootstrap surface as fast paths to re-express scopes",
        "the wrong work; this table exists partly to stop that.",
        "",
        "A family leaves this table one of two ways, and only these two:",
        "",
        "1. **Absorbed** — the mainline compiler grows an admission predicate",
        "   and lowering for it, proven against the bootstrap row it replaces.",
        "2. **Re-expressed** — it stays hand-written or library-backed, but is",
        "   reached through a declared Target IR boundary",
        "   (`tessera_x86.abi_call` and its per-backend equivalents) so the",
        "   Decision #28 arbiter can score it. Chosen, never defaulted into.",
        "",
        "Deleting a `gap` row without one of those is capability loss, which",
        "is the failure mode Decision #31's ordering caveat exists to prevent.",
        "",
    ]
    return "\n".join(out)


def render_csv() -> str:
    lines = ["target,family,compiled_route,status"]
    lines += [f"{t},{f},{r},{s}" for t, f, r, s in family_rows()]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover - manual inspection aid
    print(render_markdown())
