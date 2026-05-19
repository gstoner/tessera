"""``tessera.solvers.tpp`` — Python-side surface for the Tensor
Parallel Primitives (TPP) solver dialect.

The dialect itself lives in C++ MLIR under ``src/solvers/tpp/`` with:

  - **Ops & types** (``Dialect/TPP/``): ``!tpp.field``, ``!tpp.mesh``,
    ``#tpp.units``, ``#tpp.bc``; stencil + boundary-condition ops.
  - **Passes** (``lib/Passes/``): ``-tpp-halo-infer``,
    ``-tpp-legalize-space-time``, ``-tpp-fuse-stencil-time``,
    ``-tpp-async-prefetch``, ``-tpp-vectorize``,
    ``-tpp-distribute-halo``, ``-lower-tpp-to-target-ir``.
  - **Pipeline alias** (``lib/Passes/PassPipeline.cpp``):
    ``tpp-space-time`` chains all seven passes for the canonical
    space-time stencil lowering path.
  - **Lit fixtures** (``test/TPP/``): halo_infer.mlir,
    shallow_water_smoke.mlir, bc_lowering.mlir, pipeline_alias.mlir.

This Python module surfaces the **pass-pipeline alias names** and the
**dialect metadata** so Python tooling (audit dashboards, JIT routing,
support tables) can reason about TPP without linking against
``tessera-opt``.  Actual lowering / execution requires
``tessera-opt`` built against MLIR 21.

Status: **scaffold + Python frontmatter** (2026-05-18).  The C++
dialect + passes ship; the Python ``solve(...)`` driver is not yet
wired — it would dispatch through ``tessera-opt`` once that build
lands.  See ``docs/audit/compiler_improvement_milestone_plan_2026_05_18.md``
M5 / M8 follow-ups.
"""

from __future__ import annotations

from dataclasses import dataclass


#: Stable name of the canonical TPP pass-pipeline alias as
#: registered in ``src/solvers/tpp/lib/Passes/PassPipeline.cpp``.
TPP_PIPELINE_ALIAS: str = "tpp-space-time"


#: Names of every individual pass exposed by the TPP solver.  Order
#: matches the chain in the ``tpp-space-time`` alias.
TPP_PASS_NAMES: tuple[str, ...] = (
    "tpp-legalize-space-time",
    "tpp-halo-infer",
    "tpp-fuse-stencil-time",
    "tpp-async-prefetch",
    "tpp-vectorize",
    "tpp-distribute-halo",
    "lower-tpp-to-target-ir",
)


#: Type names defined by the TPP dialect (without the ``!`` sigil).
TPP_TYPE_NAMES: tuple[str, ...] = (
    "tpp.field",
    "tpp.mesh",
)


#: Attribute names defined by the TPP dialect (without ``#``).
TPP_ATTR_NAMES: tuple[str, ...] = (
    "tpp.units",
    "tpp.bc",
)


@dataclass(frozen=True)
class TPPStatus:
    """One-shot snapshot of TPP's current build/wiring state."""
    dialect_present: bool
    passes_present: bool
    pipeline_alias_present: bool
    python_driver_wired: bool
    lit_fixtures_runnable: bool
    notes: str


def status() -> TPPStatus:
    """Report what's wired and what isn't.

    The C++ side ships (dialect + 7 passes + pipeline alias + 4 lit
    fixtures); the Python driver dispatch and ``tessera-opt``-based
    lit runs are gated on the MLIR 21 build pass landing.
    """
    return TPPStatus(
        dialect_present=True,
        passes_present=True,
        pipeline_alias_present=True,
        python_driver_wired=False,
        lit_fixtures_runnable=False,
        notes=(
            "TPP scaffold + Python frontmatter shipped 2026-05-18; "
            "Python driver dispatch and `tessera-opt` lit runs gated "
            "on the MLIR 21 build pass landing."
        ),
    )


def pipeline_command(input_mlir: str) -> list[str]:
    """Construct the ``tessera-opt`` command line for the canonical
    TPP pipeline alias.

    Returns a list suitable for ``subprocess.run([...])`` *once*
    ``tessera-opt`` exists on ``PATH``.  Until then this helper
    documents the expected invocation::

        tessera-opt --pass-pipeline="builtin.module(tpp-space-time)" input.mlir

    Parameters
    ----------
    input_mlir
        Path to the input MLIR file containing TPP ops.
    """
    return [
        "tessera-opt",
        f"--pass-pipeline=builtin.module({TPP_PIPELINE_ALIAS})",
        input_mlir,
    ]


__all__ = [
    "TPP_PIPELINE_ALIAS",
    "TPP_PASS_NAMES",
    "TPP_TYPE_NAMES",
    "TPP_ATTR_NAMES",
    "TPPStatus",
    "status",
    "pipeline_command",
]
