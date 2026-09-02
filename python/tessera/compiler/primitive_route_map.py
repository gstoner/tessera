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

import re
from pathlib import Path

from typing import Mapping

#: Route kinds, most-preferred first. `compiled` is the mainline
#: Graph -> Schedule -> Tile -> Target descent through `tessera-opt`;
#: `bootstrap` is a Python `package_*` packager (the prune target);
#: `none` means neither serves this primitive on that target.
ROUTE_COMPILED = "compiled"
ROUTE_BOOTSTRAP = "bootstrap"
ROUTE_NONE = "none"
#: The audit cannot see this target's families, so no route claim is possible.
#:
#: NOT the same as `none`, and the distinction is the whole point: `none` says
#: "measured, nothing serves it"; this says "not measured". Collapsing them is
#: how a backend disappears from a dashboard and reads as unserved -- the same
#: failure `unmeasured` exists to prevent in the autotune corpus.
ROUTE_UNCLASSIFIED = "unclassified"

#: Families whose name is a *benchmark cohort*, not a Graph IR pattern.
#:
#: `x86_native.native_package_kind` returns these alongside real families. They
#: have no member primitives because they are not op patterns, and declaring
#: them empty is the honest record -- the alternative is for the two-way guard
#: below to fail forever on families that can never have members. Listed
#: explicitly rather than skipped silently, so a future reader sees that the
#: x86 classifier mixes two vocabularies.
NON_PATTERN_FAMILIES: frozenset[str] = frozenset({"breadth", "cohort2"})

#: Targets whose families the audit CANNOT derive, with the reason. Declared so
#: the omission is visible in the rendered map instead of silent.
#:
#: `apple_cpu` is in `bootstrap_prune_audit._BACKEND_MODULES` but its
#: `native_package_kind` returns a computed expression
#: (`op.op_name.removeprefix(...)`) rather than string literals, so the AST
#: walker yields nothing. `apple_gpu` is not in that audit at all, yet
#: `driver.py` has a live scheduled dispatch for it. Both have real routes this
#: map cannot yet see.
UNCLASSIFIABLE_TARGETS: Mapping[str, str] = {
    "apple_cpu": ("native_package_kind returns a computed expression, not "
                  "string literals, so the AST walker derives no families"),
    "apple_gpu": ("not in bootstrap_prune_audit._BACKEND_MODULES, though "
                  "driver.py has a live scheduled dispatch for it"),
}

#: Families whose compiled route is gated to specific targets in `driver.py`.
#: A family absent here is treated as available on every target that classifies
#: it -- which is only sound when the dispatch is not target-gated.
#:
#: `depth_attention` is the reason this exists: `driver.py` dispatches it under
#: `target_kind == "rocm_gfx1151"`, and an earlier version of this module fanned
#: its compiled route across every discovered target, publishing `compiled` for
#: NVIDIA and x86 in the generated CSV. The comment justifying that fan-out
#: claimed it "keeps the claim no stronger than the source"; the source is
#: target-aware, so it made the claim strictly stronger (review on #677).
COMPILED_ROUTE_TARGETS: Mapping[str, tuple[str, ...]] = {
    "depth_attention": ("rocm_gfx1151",),
}

#: family -> target -> the coverage primitives that target's contract accepts.
#:
#: **Per target, because the contracts differ.** A single global tuple was
#: applied to every backend and published `min`/`amin` as routes on ROCm and
#: x86, whose reduction contracts accept only sum/mean/max/amax
#: (`rocm_native`/`x86_native`). The guard below now checks every declared
#: member's Graph IR name against the `tessera.*` literals of that target's own
#: native module, which is what would have caught it.
#:
#: `"*"` declares a member set for every target that classifies the family.
FAMILY_MEMBERS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    # Only `flash_attn`. An earlier draft also claimed multi_head_attention,
    # gqa_attention, mqa_attention and attn_sliding_window because the gfx1151
    # ragged head_dim proof exercised them -- but that is the RUNTIME lane
    # (`_rocm_flash_attn`), not the packager's contract, and those op names
    # appear in no native module. Working on the device and being named by the
    # family classifier are different facts; the literal check refuses the
    # conflation.
    "attention": {"*": ("flash_attn",)},
    "attention_lse": {"*": ("flash_attn",)},
    # There is no separate `*_bwd` coverage primitive: a backward is the VJP
    # axis of its forward, so the backward families claim the same member. The
    # route map therefore does NOT answer "is the backward compiled" -- that
    # belongs to the vjp axis.
    "attention_backward": {"*": ("flash_attn",)},
    "attention_backward_lse": {"*": ("flash_attn",)},
    "depth_attention": {"*": ("depth_attn",)},
    "softmax": {"*": ("softmax",)},
    "norm": {"nvidia_sm120": ("layer_norm", "rmsnorm"),
             "x86": ("layer_norm", "rmsnorm"),
             "*": ("layer_norm", "rmsnorm")},
    # Per target, because the contracts differ. NVIDIA names amin/min; ROCm and
    # x86 do not, and one global tuple published both as routes there.
    "reduction": {
        "nvidia_sm120": ("sum", "mean", "max", "min", "amax", "amin"),
        "rocm_gfx1151": ("sum", "mean", "max", "amax"),
        "x86": ("sum", "mean", "max", "amax"),
    },
    "matmul": {"*": ("matmul",)},
    "int4_matmul": {"*": ("matmul",)},
    "nvfp4_matmul": {"*": ("matmul",)},
    "mx_matmul": {"*": ("matmul",)},
    "moe_dispatch": {"*": ("moe_dispatch",)},
    # The paged-KV family reads the same flash kernel's cache path.
    "paged_kv": {"*": ("flash_attn",)},
    # x86 is the only classifier of this family, and its module names a wide
    # pointwise vocabulary. Restricted to ops it actually names.
    "elementwise": {"*": (
        "add", "sub", "mul", "div", "exp", "log", "sqrt", "rsqrt", "tanh",
        "sigmoid", "gelu", "silu", "erf", "sign", "where", "pow",
        "maximum", "minimum",
    )},
}


#: Native module per target, for the literal cross-check below.
_NATIVE_MODULE: Mapping[str, str] = {
    "nvidia_sm120": "nvidia_native.py",
    "rocm_gfx1151": "rocm_native.py",
    "x86": "x86_native.py",
    "apple_cpu": "apple_cpu_native.py",
}

_OP_LITERAL = re.compile(r'"(tessera\.[a-z_0-9]+)"')


def target_op_literals(target: str) -> frozenset[str]:
    """The `tessera.*` op names a target's native module actually mentions.

    This is the half that makes a declared membership checkable. A member whose
    Graph IR name never appears in the module is not a route that target can
    serve -- which is exactly how `min`/`amin` were published for ROCm and x86.
    A literal appearing is necessary, not sufficient; the check is deliberately
    one-directional and catches over-claims only.
    """
    module = _NATIVE_MODULE.get(target)
    if module is None:
        return frozenset()
    path = Path(__file__).resolve().parent / module
    try:
        return frozenset(_OP_LITERAL.findall(path.read_text(encoding="utf-8")))
    except OSError:
        return frozenset()


def members_for(family: str, target: str) -> tuple[str, ...]:
    """Declared members of `family` on `target` (`"*"` is the default)."""
    per_target = FAMILY_MEMBERS.get(family, {})
    return tuple(per_target.get(target, per_target.get("*", ())))


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
    discovered = discovered_families()
    for family, per_target in sorted(FAMILY_MEMBERS.items()):
        if not per_target or not any(per_target.values()):
            problems.append(f"{family}: declared with no members")
        for member in {m for members in per_target.values() for m in members}:
            if member not in coverage:
                problems.append(
                    f"{family}: member {member!r} is not a coverage primitive")
            elif not coverage[member]:
                problems.append(
                    f"{family}: member {member!r} has no graph_name, so it "
                    f"cannot be reached by a Graph IR pattern")

    # (1b) A member must appear in the native module of the target it is
    #      declared for. This is the check that would have caught `min`/`amin`
    #      being published as ROCm and x86 reduction routes: their contracts
    #      name only sum/mean/max/amax, so those literals are simply absent.
    #      One-directional on purpose -- a literal appearing is necessary, not
    #      sufficient, so this catches over-claims and never manufactures one.
    for family, targets in sorted(discovered.items()):
        for target in targets:
            literals = target_op_literals(target)
            if not literals:
                continue          # target has no derivable literals; (5) covers it
            for member in members_for(family, target):
                graph_name = coverage.get(member)
                if not graph_name:
                    continue
                # A backend may name EITHER the canonical Graph IR op or the
                # public alias: coverage records `sum -> tessera.reduce`, while
                # `x86_native` names `tessera.sum`. Both are the same primitive,
                # so accepting one spelling only would reject a real route.
                spellings = {graph_name, f"tessera.{member}"}
                if not (spellings & literals):
                    problems.append(
                        f"{family}/{target}: member {member!r} "
                        f"(neither {graph_name} nor tessera.{member}) appears "
                        f"in {_NATIVE_MODULE.get(target, '?')} -- that "
                        f"target's contract does not name it")

    # (1c) A compiled-route target restriction must name a real target.
    for family, targets in sorted(COMPILED_ROUTE_TARGETS.items()):
        if family not in compiled_route_families():
            problems.append(
                f"{family}: COMPILED_ROUTE_TARGETS names it but it has no "
                f"compiled route -- stale restriction")
        for target in targets:
            if target not in _NATIVE_MODULE and target not in UNCLASSIFIABLE_TARGETS:
                problems.append(
                    f"{family}: compiled-route target {target!r} is unknown")

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

    # (5) A target this map reports on must either have derivable families or
    #     be named as unclassifiable WITH a reason. Silence is what let Apple
    #     drop out of the map entirely while `driver.py` had live routes for it.
    for target in sorted(set(_NATIVE_MODULE) | set(UNCLASSIFIABLE_TARGETS)):
        derivable = any(target in ts for ts in discovered.values())
        named = target in UNCLASSIFIABLE_TARGETS
        if not derivable and not named:
            problems.append(
                f"{target}: no derivable families and no entry in "
                f"UNCLASSIFIABLE_TARGETS -- it would vanish from the map "
                f"silently, which reads as 'unserved' rather than 'unmeasured'")
        if derivable and named:
            problems.append(
                f"{target}: listed as unclassifiable but the audit does derive "
                f"families for it -- remove the entry")

    if problems:
        raise ValueError(
            "primitive_route_map membership is stale:\n  "
            + "\n  ".join(problems))


def all_targets() -> tuple[str, ...]:
    """Every target the map reports on, derivable or not.

    Includes the unclassifiable ones on purpose: a target missing from the
    columns reads as "no route", and Apple vanished from an earlier version of
    this map for exactly that reason while `driver.py` had live scheduled
    dispatch for `apple_gpu`.
    """
    derived = {t for ts in discovered_families().values() for t in ts}
    return tuple(sorted(derived | set(UNCLASSIFIABLE_TARGETS)))


def primitive_routes() -> "dict[str, dict[str, str]]":
    """primitive -> {target: route kind}.

    A primitive's route on a target is the BEST route any family it belongs to
    has there: `compiled` beats `bootstrap` beats `none`. "Best" is the right
    reduction because the question the prune plan asks is whether deleting the
    bootstrap would strand this primitive -- one compiled path answers no.

    A target in `UNCLASSIFIABLE_TARGETS` gets `unclassified` for every
    primitive rather than `none`: this map cannot see its families, and
    "not measured" must not render as "nothing serves it".
    """
    verify_family_membership()
    discovered = discovered_families()
    compiled = compiled_route_families()
    targets = all_targets()

    routes: dict[str, dict[str, str]] = {}

    def _blank() -> dict[str, str]:
        return {t: (ROUTE_UNCLASSIFIED if t in UNCLASSIFIABLE_TARGETS
                    else ROUTE_NONE) for t in targets}

    for family in FAMILY_MEMBERS:
        family_targets = discovered.get(family, ())
        # A compiled route is only claimed where `driver.py` actually
        # dispatches it. Absent a restriction the route follows the targets
        # that classify the family; it is NEVER fanned across all targets,
        # which is what published `depth_attn` as compiled on NVIDIA and x86.
        compiled_targets: tuple[str, ...] = ()
        if family in compiled:
            compiled_targets = COMPILED_ROUTE_TARGETS.get(family, family_targets)
        for target in set(family_targets) | set(compiled_targets):
            if target in UNCLASSIFIABLE_TARGETS:
                continue                      # no claim is possible there
            for member in members_for(family, target):
                per_target = routes.setdefault(member, _blank())
                if target in compiled_targets:
                    per_target[target] = ROUTE_COMPILED
                elif per_target[target] == ROUTE_NONE:
                    per_target[target] = ROUTE_BOOTSTRAP
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
    targets = list(all_targets())

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
        "**`❔ unclassified` is not `—`.** A dash means measured and nothing",
        "serves it; `unclassified` means this map cannot see that target's",
        "families at all, so no claim is possible. Collapsing the two is how a",
        "whole backend reads as unserved.",
        "",
        "| Target | Why unclassified |",
        "|---|---|",
    ] + [f"| `{t}` | {why} |" for t, why in sorted(UNCLASSIFIABLE_TARGETS.items())] + [
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
                          ROUTE_NONE: "—",
                          ROUTE_UNCLASSIFIED: "❔ unclassified"}[kind])
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
        per_target = FAMILY_MEMBERS.get(family, {})
        if family in NON_PATTERN_FAMILIES:
            note = "*benchmark cohort, not an op pattern*"
        elif set(per_target) - {"*"}:
            note = "; ".join(
                f"**{t}**: " + ", ".join(f"`{m}`" for m in members)
                for t, members in sorted(per_target.items()))
        else:
            note = ", ".join(f"`{m}`" for m in per_target.get("*", ())) or "—"
        scope = COMPILED_ROUTE_TARGETS.get(family)
        lines.append(
            f"| `{family}` | "
            f"{('✅ ' + ', '.join(scope)) if (family in compiled and scope) else ('✅' if family in compiled else '—')} | "
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
