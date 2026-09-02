"""The join between per-primitive coverage and the MLIR/LLVM route.

`standalone_primitive_coverage.md` tracks 515 primitives across 12 contract
axes. `bootstrap_prune_gap.md` tracks which *families* the mainline compiler
covers and which would lose their only lowering if the Python bootstrap were
deleted. Both are generated, both are drift-gated, and **they had no join**:
you could not read the primitive dashboard and learn whether a primitive is
MLIR-native or bootstrap-only on a given target.

That gap is not cosmetic. `lowering_rule` reads 463/515 complete, and it is
route-BLIND -- it asks whether a lowering rule is *specified*, so a primitive
served only by a Python packager scores identically to one that descends
Graph -> Schedule -> Tile -> Target through `tessera-opt`. `backend_kernel` is
the axis that could discriminate and it had **0 completes out of 515**, which
makes it uninformative rather than wrong: an axis whose value never varies
carries no information, which is Decision #29 wearing different clothes.

**Why this is a relation and not a column.** A family is a SUBGRAPH PATTERN,
not an op: `_attention_contract` inspects `module.functions[0]` -- a whole op
sequence. "attention" is a composition of many primitives. Flattening that into
a per-primitive `route` value would answer "does `softmax` have a compiled
route?" with something meaningless outside the pattern it appeared in. So the
membership is modelled explicitly, per family, and the per-primitive answer is
DERIVED from it.

**Why membership is declared rather than derived.** Three derivations were
tried and rejected on evidence:

* scanning packager bodies for `tessera.*` literals finds 0-1 ops per backend
  -- the names live in the `supports_*` contract helpers, not the packagers;
* resolving those helpers transitively models the dispatcher instead of running
  it, so a refactor that moves a literal silently changes the map;
* probing the real `native_package_kind` with a synthesized single-op module
  under-reports every multi-op family, since a one-op probe cannot match a
  subgraph pattern.

A plausible-but-wrong route map is worse than none: it asserts route facts
nobody verified, in a dashboard people act on. So membership is declared where
the judgment genuinely lives, and **verified in both directions** against two
independent sources -- the same shape as `bootstrap_prune_audit._FAMILY_TO_COMPILED`,
which already raises rather than silently reporting a family as covered.
"""

from __future__ import annotations

from typing import Mapping

#: Route kinds, most-preferred first. `compiled` is the mainline
#: Graph -> Schedule -> Tile -> Target descent through `tessera-opt`;
#: `bootstrap` is a Python `package_*` packager (the prune target);
#: `none` means neither serves this primitive on that target.
ROUTE_COMPILED = "compiled"
ROUTE_BOOTSTRAP = "bootstrap"
ROUTE_NONE = "none"

#: Families whose name is a *benchmark cohort*, not a Graph IR pattern.
#:
#: `x86_native.native_package_kind` returns these alongside real families. They
#: have no member primitives because they are not op patterns, and declaring
#: them empty is the honest record -- the alternative is for the two-way guard
#: below to fail forever on families that can never have members. Listed
#: explicitly rather than skipped silently, so a future reader sees that the
#: x86 classifier mixes two vocabularies.
NON_PATTERN_FAMILIES: frozenset[str] = frozenset({"breadth", "cohort2"})

#: family -> the coverage primitives whose Graph IR op the family's contract
#: recognises. Grounded in the `tessera.*` literals each backend's native
#: module names; reviewed, because which primitives *compose* a pattern is a
#: judgment the source cannot state on its own.
FAMILY_MEMBERS: Mapping[str, tuple[str, ...]] = {
    # The attention pattern serves the flash kernel and the head-layout
    # variants that lower onto it -- exactly the set the gfx1151 ragged
    # head_dim proof exercised (multi_head / GQA / MQA / sliding window).
    "attention": (
        "flash_attn", "multi_head_attention", "gqa_attention",
        "mqa_attention", "attn_sliding_window",
    ),
    "attention_lse": ("flash_attn", "attn_with_stats"),
    # There is no separate `*_bwd` primitive: a backward is the VJP axis of the
    # forward one, so the backward families claim the same member. This is why
    # the route map deliberately does NOT try to answer "is the backward
    # compiled" -- that question belongs to the vjp axis, not to a route.
    "attention_backward": ("flash_attn",),
    "attention_backward_lse": ("flash_attn", "attn_with_stats"),
    "depth_attention": ("depth_attn",),
    "softmax": ("softmax",),
    "norm": ("layer_norm", "rmsnorm"),
    "reduction": ("sum", "mean", "max", "min", "amax", "amin"),
    "matmul": ("matmul",),
    "int4_matmul": ("matmul",),
    "nvfp4_matmul": ("matmul",),
    "mx_matmul": ("matmul",),
    "moe_dispatch": ("moe_dispatch",),
    "paged_kv": ("flash_attn",),
    "elementwise": (
        "add", "sub", "mul", "div", "exp", "log", "sqrt", "rsqrt", "tanh",
        "sigmoid", "gelu", "silu", "erf", "abs", "sign", "where", "pow",
        "maximum", "minimum",
    ),
}


def _coverage_primitives() -> "Mapping[str, str | None]":
    """name -> graph_name, from the RAW registry.

    Deliberately not `all_primitive_coverages()`: that function consults this
    module to derive `backend_kernel`, so reading it back would recurse. It is
    also the right layering -- a route map has no business reading contract
    statuses.
    """
    from .primitive_coverage import primitive_graph_names
    return primitive_graph_names()


def declared_families() -> frozenset[str]:
    return frozenset(FAMILY_MEMBERS) | NON_PATTERN_FAMILIES


def discovered_families() -> "dict[str, tuple[str, ...]]":
    """family -> the targets whose own classifier returns it.

    Read from `bootstrap_prune_audit`, which AST-derives it from each
    backend's `native_package_kind`. Deriving rather than declaring this half
    is what makes the guard below meaningful: a family added to a backend
    shows up here without anyone remembering to record it.
    """
    from .bootstrap_prune_audit import collect_inventories
    out: dict[str, list[str]] = {}
    for inventory in collect_inventories():
        for family in inventory.families:
            out.setdefault(family, []).append(inventory.target)
    return {family: tuple(targets) for family, targets in sorted(out.items())}


def compiled_route_families() -> frozenset[str]:
    """Families the mainline compiler already covers."""
    from .bootstrap_prune_audit import _FAMILY_TO_COMPILED
    return frozenset(_FAMILY_TO_COMPILED)


def verify_family_membership() -> None:
    """Fail closed on either half of the relation going stale.

    Raises rather than returning a status, because every consumer of this
    module reports route facts: a silently-degraded map would put unverified
    claims into a dashboard, which is the specific failure this whole join
    exists to prevent.
    """
    coverage = _coverage_primitives()
    problems: list[str] = []

    # (1) A declared member must be a real coverage primitive with a Graph IR
    #     name. Without this the map can name an op that no longer exists and
    #     keep reporting a route for it.
    for family, members in sorted(FAMILY_MEMBERS.items()):
        if not members:
            problems.append(f"{family}: declared with no members")
        for member in members:
            if member not in coverage:
                problems.append(
                    f"{family}: member {member!r} is not a coverage primitive")
            elif not coverage[member]:
                problems.append(
                    f"{family}: member {member!r} has no graph_name, so it "
                    f"cannot be reached by a Graph IR pattern")

    # (2) Every family a backend actually classifies must be declared. Without
    #     this a new family silently has zero members and reads as 'no
    #     primitive is served by it', which is indistinguishable from a family
    #     nobody uses.
    for family, targets in discovered_families().items():
        if family not in declared_families():
            problems.append(
                f"{family}: classified by {', '.join(targets)} but has no "
                f"membership declaration in FAMILY_MEMBERS "
                f"(add it, or list it in NON_PATTERN_FAMILIES with a reason)")

    # (3) Same for a family with a compiled route: it is the half that
    #     upgrades a primitive's status, so an undeclared one loses coverage
    #     rather than gaining it -- a silent under-report.
    for family in sorted(compiled_route_families()):
        if family not in declared_families():
            problems.append(
                f"{family}: has a compiled route but no membership declaration")

    # (4) A declaration for a family nobody classifies and nothing compiles is
    #     dead weight that will drift.
    live = set(discovered_families()) | set(compiled_route_families())
    for family in sorted(FAMILY_MEMBERS):
        if family not in live:
            problems.append(
                f"{family}: declared, but no backend classifies it and no "
                f"compiled route claims it -- stale declaration")

    if problems:
        raise ValueError(
            "primitive_route_map membership is stale:\n  "
            + "\n  ".join(problems))


def primitive_routes() -> "dict[str, dict[str, str]]":
    """primitive -> {target: route kind}.

    A primitive's route on a target is the BEST route any family it belongs to
    has there: `compiled` beats `bootstrap` beats `none`. "Best" is the right
    reduction because the question the prune plan asks is whether deleting the
    bootstrap would strand this primitive -- one compiled path is enough to say
    no.
    """
    verify_family_membership()
    discovered = discovered_families()
    compiled = compiled_route_families()
    targets = sorted({t for ts in discovered.values() for t in ts})

    routes: dict[str, dict[str, str]] = {}
    for family, members in FAMILY_MEMBERS.items():
        family_targets = discovered.get(family, ())
        for member in members:
            per_target = routes.setdefault(member, {t: ROUTE_NONE for t in targets})
            if family in compiled:
                # A compiled route is not per-target in `_FAMILY_TO_COMPILED`:
                # it is a scheduled-module predicate, which is target-agnostic
                # by construction. Recording it on every target this family is
                # classified for -- and on all targets when no backend
                # classifies it -- keeps the claim no stronger than the source.
                for t in (family_targets or targets):
                    per_target[t] = ROUTE_COMPILED
            for t in family_targets:
                if per_target[t] == ROUTE_NONE:
                    per_target[t] = ROUTE_BOOTSTRAP
    return routes


def derived_backend_kernel(primitive: str) -> str:
    """The `backend_kernel` axis value, derived rather than hand-set.

    * `complete`  -- a compiled (mainline MLIR) route serves it on some target.
    * `partial`   -- only a bootstrap packager serves it: it runs, and deleting
                     the bootstrap would strand it.
    * `planned`   -- no route on any target.

    Deliberately says nothing about exact-device evidence. Whether a route was
    *proven on silicon* is a different claim with its own surface
    (`runtime_execution_matrix.md`, Decision #26), and folding two independent
    facts into one enum is how the old axis became uninformative.
    """
    per_target = primitive_routes().get(primitive)
    if not per_target:
        return "planned"
    if ROUTE_COMPILED in per_target.values():
        return "complete"
    if ROUTE_BOOTSTRAP in per_target.values():
        return "partial"
    return "planned"


__all__ = [
    "FAMILY_MEMBERS",
    "NON_PATTERN_FAMILIES",
    "ROUTE_BOOTSTRAP",
    "ROUTE_COMPILED",
    "ROUTE_NONE",
    "compiled_route_families",
    "declared_families",
    "derived_backend_kernel",
    "discovered_families",
    "primitive_routes",
    "verify_family_membership",
]


def render_markdown() -> str:
    """The join surface: which compiler serves each primitive, per target."""
    verify_family_membership()
    routes = primitive_routes()
    discovered = discovered_families()
    compiled = compiled_route_families()
    targets = sorted({t for ts in discovered.values() for t in ts})

    compiled_count = sum(
        1 for per in routes.values() if ROUTE_COMPILED in per.values())
    bootstrap_only = sum(
        1 for per in routes.values()
        if ROUTE_COMPILED not in per.values() and ROUTE_BOOTSTRAP in per.values())

    lines = [
        "# Primitive Route Map — which compiler serves each primitive",
        "",
        "**Generated. Do not hand-edit.** Regenerate with",
        "`python -m tessera.compiler.generated_docs --write primitive_route_map`.",
        "",
        "`standalone_primitive_coverage.md` tracks per-primitive *contracts*;",
        "`bootstrap_prune_gap.md` tracks which *families* the mainline compiler",
        "covers. This is the join. It exists because `lowering_rule` is",
        "**route-blind** -- it asks whether a lowering rule is *specified*, so a",
        "primitive served only by a Python packager scores the same as one that",
        "descends Graph -> Schedule -> Tile -> Target through `tessera-opt`.",
        "",
        "**This is not a kernel-quality claim and not a device claim.** It says",
        "which compiler produces the code, nothing about whether that code was",
        "proven on silicon -- that is `runtime_execution_matrix.md`",
        "(Decision #26). It is deliberately NOT folded into the",
        "`backend_kernel` axis: that axis means *hardware proofs across every",
        "declared target*, and its zero is an honest statement that nothing has",
        "met that bar, not an axis carrying no information.",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Primitives reached by any route | {len(routes)} |",
        f"| — served by a **compiled** route on some target | {compiled_count} |",
        f"| — **bootstrap only** (deleting the packager strands them) | {bootstrap_only} |",
        f"| Families a backend classifies | {len(discovered)} |",
        f"| Families with a compiled route | {len(compiled)} |",
        "",
        "## Per primitive",
        "",
        "| Primitive | " + " | ".join(targets) + " |",
        "|---" * (len(targets) + 1) + "|",
    ]
    for primitive in sorted(routes):
        cells = []
        for target in targets:
            kind = routes[primitive].get(target, ROUTE_NONE)
            cells.append({ROUTE_COMPILED: "✅ compiled",
                          ROUTE_BOOTSTRAP: "🟡 bootstrap",
                          ROUTE_NONE: "—"}[kind])
        lines.append(f"| `{primitive}` | " + " | ".join(cells) + " |")

    lines += ["", "## Family membership", "",
              "Declared, and verified in both directions against",
              "`bootstrap_prune_audit`: a member must be a real coverage",
              "primitive with a Graph IR name, and a family any backend",
              "classifies must be declared here. Either half going stale",
              "raises rather than degrading quietly.",
              "",
              "| Family | Compiled route | Classified by | Members |",
              "|---|---|---|---|"]
    for family in sorted(set(discovered) | set(compiled) | set(FAMILY_MEMBERS)):
        members = FAMILY_MEMBERS.get(family, ())
        note = ("*benchmark cohort, not an op pattern*"
                if family in NON_PATTERN_FAMILIES
                else ", ".join(f"`{m}`" for m in members) or "—")
        lines.append(
            f"| `{family}` | {'✅' if family in compiled else '—'} | "
            f"{', '.join(discovered.get(family, ())) or '—'} | {note} |")
    return "\n".join(lines) + "\n"


def render_csv() -> str:
    verify_family_membership()
    routes = primitive_routes()
    targets = sorted({t for per in routes.values() for t in per})
    out = ["primitive,target,route"]
    for primitive in sorted(routes):
        for target in targets:
            out.append(f"{primitive},{target},{routes[primitive].get(target, ROUTE_NONE)}")
    return "\n".join(out) + "\n"
