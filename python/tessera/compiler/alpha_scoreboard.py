"""Functional-complete alpha scoreboard: lane x family x stage.

MASTER_AUDIT's "Functional-complete alpha" section defines the release: every
in-scope family runs frontend -> typed Graph IR -> AD/optimization -> Schedule
IR -> Tile IR -> Target IR + native lowering -> native image + checked ABI,
each stage produced by MLIR passes, on all eight fleet lanes. This module
measures that definition from sources that already exist, so progress is
derived rather than narrated.

Two things are **declared**, because they are policy rather than facts about
the code: the eight lanes (the owner's definition) and the alpha family set
with its per-backend names. Everything else is derived:

* ``frontend``       -- whether the AST ``_OpExtractor`` still exists
                        (E2E-REAL-6: the tracer must be the only frontend).
* ``graph_opt``      -- not yet derivable per family; reported ``unmeasured``
                        so the gap is visible rather than assumed closed.
* ``schedule_tile``  -- :func:`bootstrap_prune_audit.family_rows`: a family is
                        ``native`` when a compiled (family-named or generic)
                        Schedule->Tile route serves it, ``bypass`` when only a
                        Python ``package_*`` constructor does, ``unrouted``
                        when the backend module does not classify it at all.
* ``native_lowering``-- the lane target's Level C in the compilation spine.
* ``execution``      -- the E2E fleet release packet for the lane.

Fail closed throughout: anything not proven native counts against alpha, and a
declared alias that names a family its module does not classify raises rather
than reporting coverage.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_COMPILER = Path(__file__).resolve().parent

STAGES: tuple[str, ...] = (
    "frontend", "graph_opt", "schedule_tile", "native_lowering", "execution",
)
DONE = "native"


@dataclass(frozen=True)
class Lane:
    """One of the eight alpha lanes (owner definition, MASTER_AUDIT)."""

    lane: str
    host: str
    device: str
    #: key in bootstrap_prune_audit._BACKEND_MODULES
    route_target: str
    #: target in the compilation spine inventory
    spine_target: str
    #: (target, architecture) in the E2E fleet registry, or None if unregistered
    fleet: tuple[str, str] | None


LANES: tuple[Lane, ...] = (
    Lane("mac_cpu", "Mac M1 Max", "arm64 CPU", "apple_cpu", "apple_cpu",
         ("apple_cpu", "apple_m1_max")),
    Lane("mac_gpu", "Mac M1 Max", "Apple7 GPU", "apple_gpu", "apple_gpu",
         ("apple_gpu", "apple7")),
    Lane("luna_cpu", "Princess-Luna", "Zen 5 AVX-512", "x86", "x86",
         ("x86", "x86_64_avx512_strix_halo")),
    Lane("luna_gpu", "Princess-Luna", "gfx1151", "rocm_gfx1151", "rocm_gfx1151",
         ("rocm_gfx1151", "gfx1151")),
    # Zen 2 has no AVX-512; the only x86 lane it can run today is the
    # portable baseline, which is why its coverage is so narrow.
    Lane("bear_cpu", "The-Super-Bear", "Zen 2 AVX2", "x86", "x86",
         ("x86", "x86_64_base")),
    Lane("bear_gpu", "The-Super-Bear", "sm_120", "nvidia_sm120", "nvidia_sm120",
         ("nvidia_sm120", "sm_120a")),
    Lane("taj_cpu", "Tajasarus", "Zen 5 AVX-512", "x86", "x86",
         ("x86", "x86_64_avx512_granite_ridge")),
    # rocm_native.py serves both RDNA parts; chip-specific admission is what
    # the spine and fleet columns check, and gfx1201 has no fleet packet.
    Lane("taj_gpu", "Tajasarus", "gfx1201", "rocm_gfx1151", "rocm_gfx1201", None),
)

#: Alpha family set -> per-route-target family names where they differ from
#: the canonical name. A tuple means several module families together make up
#: the alpha family; its stage is the worst of them. Verified at render time.
ALPHA_FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "matmul": {"apple_gpu": ("batched_gemm",)},
    "softmax": {},
    "reduction": {},
    "norm": {},
    "attention": {},
    "paged_kv": {},
    "moe": {"rocm_gfx1151": ("moe_dispatch",)},
    "linalg": {
        "apple_cpu": ("cholesky", "cholesky_solve", "lu", "qr", "svd", "tri_solve"),
        "apple_gpu": ("svd", "value_cholesky", "value_cholesky_solve", "value_tri_solve"),
    },
    "epilogue": {},
    "replay_ssm": {},
    "ppo": {"apple_gpu": ("value_rl_ppo_policy_loss",)},
    "ebm": {"apple_gpu": ("value_ebm_energy_quadratic", "value_ebm_langevin_step",
                          "value_ebm_partition_exact", "value_ebm_refinement")},
    "clifford": {"apple_gpu": ("value_clifford_geometric_product",)},
}

_ROUTE_RANK = {"native": 0, "bypass": 1, "unrouted": 2}


class AlphaScoreboardError(ValueError):
    """A declared lane or alias no longer matches the sources it reads."""


def _op_extractor_present() -> bool:
    tree = ast.parse((_COMPILER / "graph_ir.py").read_text(encoding="utf-8"))
    return any(isinstance(n, ast.ClassDef) and n.name == "_OpExtractor"
               for n in ast.walk(tree))


def _route_status() -> dict[tuple[str, str], str]:
    from .bootstrap_prune_audit import family_rows
    out: dict[tuple[str, str], str] = {}
    for target, family, _route, status in family_rows():
        out[(target, family)] = "bypass" if status == "gap" else "native"
    return out


def _verify_aliases(routes: dict[tuple[str, str], str]) -> None:
    known = {t for t, _ in routes}
    missing = []
    for family, per_target in ALPHA_FAMILIES.items():
        for target, names in per_target.items():
            if target not in known:
                missing.append(f"{family}: unknown route target {target!r}")
            for name in names:
                if (target, name) not in routes:
                    missing.append(f"{family}: {target} has no family {name!r}")
    if missing:
        raise AlphaScoreboardError(
            "alpha_scoreboard: declared aliases no longer match the backend "
            "modules: " + "; ".join(missing))


def _schedule_tile(routes: dict[tuple[str, str], str], lane: Lane, family: str) -> str:
    names = ALPHA_FAMILIES[family].get(lane.route_target, (family,))
    states = [routes.get((lane.route_target, n), "unrouted") for n in names]
    return max(states, key=_ROUTE_RANK.__getitem__)


def _spine_levels() -> dict[str, str]:
    from .pipeline_registry import compilation_spine_inventory
    return {r.target: r.level_c for r in compilation_spine_inventory()}


def _fleet_states() -> dict[tuple[str, str, str], str]:
    from .e2e_fleet import fleet_dashboard_rows
    return {(r.target, r.architecture, r.family): r.state
            for r in fleet_dashboard_rows()}


@dataclass(frozen=True)
class Cell:
    lane: str
    family: str
    stages: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return all(s == DONE for s in self.stages)


def cells() -> tuple[Cell, ...]:
    routes = _route_status()
    _verify_aliases(routes)
    spine = _spine_levels()
    fleet = _fleet_states()
    frontend = "bypass" if _op_extractor_present() else DONE
    out: list[Cell] = []
    for lane in LANES:
        if lane.spine_target not in spine:
            raise AlphaScoreboardError(
                f"alpha_scoreboard: lane {lane.lane} names spine target "
                f"{lane.spine_target!r}, which the spine inventory lacks")
        level_c = spine[lane.spine_target]
        lowering = DONE if level_c == "native" else level_c
        for family in ALPHA_FAMILIES:
            if lane.fleet is None:
                execution = "absent"
            else:
                state = fleet.get((*lane.fleet, family))
                execution = {"release_ready": DONE}.get(state or "", state or "absent")
            out.append(Cell(lane.lane, family, (
                frontend,
                "unmeasured",
                _schedule_tile(routes, lane, family),
                lowering,
                execution,
            )))
    return tuple(out)


def summary() -> dict[str, Any]:
    """Counts the ratchet tests pin. Everything here may only improve."""
    from .bootstrap_prune_audit import summary as prune_summary
    from .target_ir_membership import summary as membership_summary

    rows = cells()
    prune = prune_summary()
    membership = membership_summary()
    emit_dir = _COMPILER / "emit"
    emitters = 0
    for path in sorted(emit_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        emitters += sum(
            1 for n in tree.body if isinstance(n, ast.ClassDef)
            and any(isinstance(b, ast.Name) and b.id == "KernelEmitter" for b in n.bases))
    stage_native = {
        stage: sum(1 for c in rows if c.stages[i] == DONE)
        for i, stage in enumerate(STAGES)
    }
    lane_native = {
        lane.lane: sum(1 for c in rows if c.lane == lane.lane
                       for s in c.stages if s == DONE)
        for lane in LANES
    }
    return {
        "cells": len(rows),
        "complete": sum(1 for c in rows if c.complete),
        "stage_native": stage_native,
        "lane_native": lane_native,
        # Guard rail 1: bypass surfaces, which may only shrink.
        "graph_input_packagers": prune["bootstrap"],
        "delegating_packagers": prune["delegates"] + prune["both"],
        "source_emitters": emitters,
        "target_ops_without_required_contract": sum(
            b["optional-only"] + b["no-contract"] for b in membership.values()),
    }


def render_csv() -> str:
    lines = ["lane,host,device,family," + ",".join(STAGES) + ",alpha_complete"]
    by_lane = {lane.lane: lane for lane in LANES}
    for c in cells():
        lane = by_lane[c.lane]
        lines.append(",".join((c.lane, lane.host, lane.device, c.family, *c.stages,
                               "yes" if c.complete else "no")))
    return "\n".join(lines) + "\n"


_MARK = {"native": "✅", "bypass": "🔴", "unrouted": "⬛", "absent": "⬛",
         "partial": "🟡", "packet_pending": "🟡", "unmeasured": "❔"}


def render_markdown() -> str:
    s = summary()
    rows = cells()
    out = [
        "# Functional-Complete Alpha Scoreboard",
        "",
        "**Generated. Do not hand-edit.** Regenerate with",
        "`python -m tessera.compiler.generated_docs --write alpha_scoreboard`.",
        "",
        "Measures the release definition in",
        "[MASTER_AUDIT §Functional-complete alpha](../MASTER_AUDIT.md#functional-complete-alpha-definition-and-guard-rails):",
        "every alpha family, on every fleet lane, through every stage, produced by",
        "MLIR passes. A cell is alpha-complete only when all five stages are",
        "`native`; anything unproven counts against it (fail closed).",
        "",
        f"**Alpha-complete cells: {s['complete']} of {s['cells']}.**",
        "",
        "## Stages",
        "",
        "| Stage | Source | Cells native |",
        "|---|---|---|",
        f"| `frontend` | AST: does `graph_ir._OpExtractor` still exist (E2E-REAL-6) | {s['stage_native']['frontend']} |",
        f"| `graph_opt` | not yet derivable per family — shown `unmeasured` | {s['stage_native']['graph_opt']} |",
        f"| `schedule_tile` | [`bootstrap_prune_gap`](bootstrap_prune_gap.md) family routes | {s['stage_native']['schedule_tile']} |",
        f"| `native_lowering` | [spine](compilation_spine_inventory.md) Level C for the lane target | {s['stage_native']['native_lowering']} |",
        f"| `execution` | [E2E fleet](e2e_fleet.md) release packet for the lane | {s['stage_native']['execution']} |",
        "",
        "## Guard rail 1 — bypass surfaces (may only shrink)",
        "",
        "| Surface | Count |",
        "|---|---|",
        f"| Graph-input `package_*` constructors | {s['graph_input_packagers']} |",
        f"| Packagers that delegate to a runtime compiler / library | {s['delegating_packagers']} |",
        f"| `emit/*` source emitters (`KernelEmitter` subclasses) | {s['source_emitters']} |",
        f"| Target IR ops without a required contract | {s['target_ops_without_required_contract']} |",
        "",
        "`tests/unit/test_alpha_scoreboard.py` ratchets these counts and the",
        "per-stage / per-lane native counts against",
        "`tests/unit/alpha_ratchet_baseline.json`: a regression fails CI, and an",
        "improvement fails until the baseline is tightened, so gains are locked.",
        "",
        "## Lanes",
        "",
        "| Lane | Host | Device | Native stage-cells |",
        "|---|---|---|---|",
    ]
    for lane in LANES:
        out.append(f"| `{lane.lane}` | {lane.host} | {lane.device} "
                   f"| {s['lane_native'][lane.lane]} |")
    out += [
        "",
        "Known limits: `taj_gpu` shares the ROCm route module with gfx1151, so",
        "its chip-specific state comes from the spine and fleet columns.",
        "",
        "## Cells",
        "",
        "Legend: ✅ native · 🔴 bypass · 🟡 partial / pending · ⬛ absent / unrouted · ❔ unmeasured",
        "",
        "| Lane | Family | " + " | ".join(f"`{st}`" for st in STAGES) + " | Alpha |",
        "|---|---|" + "---|" * len(STAGES) + "---|",
    ]
    for c in rows:
        marks = " | ".join(f"{_MARK.get(st, '')} {st}" for st in c.stages)
        out.append(f"| `{c.lane}` | `{c.family}` | {marks} | {'✅' if c.complete else '—'} |")
    out.append("")
    return "\n".join(out)
