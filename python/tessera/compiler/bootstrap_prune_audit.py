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
from dataclasses import dataclass
from pathlib import Path

_COMPILER = Path(__file__).resolve().parent

#: Backend bootstrap packager modules, in fleet-lead order.
_BACKEND_MODULES: tuple[tuple[str, str], ...] = (
    ("nvidia_sm120", "nvidia_native.py"),
    ("rocm_gfx1151", "rocm_native.py"),
    ("x86", "x86_native.py"),
    ("apple_cpu", "apple_cpu_native.py"),
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

    @property
    def bootstrap(self) -> tuple[str, ...]:
        """Packagers that re-enter Graph IR — the prune target."""
        return tuple(n for n, t in self.packagers if _is_bootstrap(t))

    @property
    def compiled_packagers(self) -> tuple[str, ...]:
        """Packagers that consume an already-lowered artifact — not a target."""
        return tuple(n for n, t in self.packagers if not _is_bootstrap(t))


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


def _packagers(tree: ast.Module) -> tuple[tuple[str, str], ...]:
    """(name, first-parameter type) for every ``package_*`` function.

    The first parameter is what separates the two populations, and it is a
    real data-flow fact rather than a naming convention:

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


def _is_bootstrap(param_type: str) -> bool:
    """Whether a packager re-enters Graph IR rather than consuming an artifact."""
    return "GraphIRModule" in param_type


def _classified_families(tree: ast.Module) -> tuple[str, ...]:
    """String literals returned by ``native_package_kind``.

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
        return tuple(dict.fromkeys(names))
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
            continue
        text = path.read_text(encoding="utf-8")
        out.append(
            BackendInventory(
                target=target,
                module=filename,
                packagers=_packagers(tree),
                families=_classified_families(tree),
                lines=text.count("\n") + 1,
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


def family_rows() -> tuple[tuple[str, str, str, str], ...]:
    """(target, family, compiled_route, status) for every classified family."""
    verify_declared_mapping()
    rows: list[tuple[str, str, str, str]] = []
    for inv in collect_inventories():
        for family in inv.families:
            route = _FAMILY_TO_COMPILED.get(family)
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
        "lines": sum(i.lines for i in inventories),
        "families": len(rows),
        "compiled": sum(1 for r in rows if r[3] == "compiled"),
        "gap": sum(1 for r in rows if r[3] == "gap"),
        "orphan_packagers": len(orphan_packagers()),
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
        "The Python per-backend `package_*` families are the **bootstrap",
        "compiler**; the architecture is core MLIR/LLVM (Graph → Schedule →",
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
        f"| — compiled-route packagers (consume a lowered artifact) | {s['compiled_packagers']} |",
        f"| Lines in those modules | {s['lines']} |",
        f"| Classified families | {s['families']} |",
        f"| — covered by a compiled route | {s['compiled']} |",
        f"| — **gap (no compiled route)** | {s['gap']} |",
        f"| Packagers matching no family | {s['orphan_packagers']} |",
        "",
        "## Per-backend bootstrap surface",
        "",
        "| Target | Module | bootstrap | compiled-route | Families | Lines |",
        "|---|---|---|---|---|---|",
    ]
    for inv in inventories:
        out.append(
            f"| `{inv.target}` | `{inv.module}` | {len(inv.bootstrap)} "
            f"| {len(inv.compiled_packagers)} | {len(inv.families)} | {inv.lines} |"
        )

    out += [
        "",
        "## Family coverage",
        "",
        "`compiled` means a compiled-route admission predicate serves that",
        "family. It does **not** assert the compiled route reaches parity on",
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
