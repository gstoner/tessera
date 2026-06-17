"""M5 follow-on — register the displaced synthesizer lanes as grader tasks.

The displacement gate (``fusion_equivalence.displacement_verdict``) proves one
fused MSL codegen lane equals its unfused reference on hidden inputs. This module
lifts that gate into the *scored environment* (``compiler_grader``): each shipped
lane becomes a grader task graded on hidden inputs, so the synthesizer
displacements are scored by the same anti-cheat, hidden-input harness as
everything else — and a lane that ever diverges fails the grade, not just a unit
test. Importing this module registers the tasks (the standard import-to-register
pattern); nothing in the grader imports it, so it stays optional.

See docs/audit/backend/apple/APPLE_GPU_CODEGEN_PLAN.md (M5) and
docs/audit/compiler/EVALUATOR_PLAN.md §9.
"""

from __future__ import annotations

from typing import Any

from tessera.compiler import fusion_equivalence as FE
from tessera.compiler.compiler_grader import CheckResult, task

# Each lane graded across a small shape matrix — breadth matters (a lane can be
# correct at one shape and wrong at another), so we sweep more than one.
_SHAPES: dict[str, tuple[tuple[int, ...], ...]] = {
    "matmul_epilogue": ((16, 64, 128), (32, 32, 64)),
    "norm_chain": ((8, 64), (4, 128)),
    "attention": ((8, 32, 16), (16, 16, 8)),
    "pointwise": ((8, 64), (16, 32)),
    "gated_matmul": ((16, 32, 48), (8, 16, 12)),
}


def _grade_lane(kind: str, rng: Any) -> list[CheckResult]:
    # Derive a per-shape seed from the hidden RNG so the inputs are fresh draws
    # the codegen never saw, while staying reproducible within a grade.
    checks: list[CheckResult] = []
    any_on_metal = False
    for shape in _SHAPES[kind]:
        seed = int(rng.integers(0, 2**31 - 1))
        v = FE.displacement_verdict(kind, shape, seed=seed)
        # Hard invariant: a displaced lane must never diverge from its reference.
        checks.append(CheckResult(
            f"{kind}/{shape}/not_divergent",
            v.relation != "divergent",
            v.detail))
        if v.executed == "metal_runtime":
            any_on_metal = True
            checks.append(CheckResult(
                f"{kind}/{shape}/equivalent_on_metal",
                v.relation == "equivalent",
                v.detail))
    # Provenance: at least one shape must have genuinely run on Metal here, or
    # there is no displacement to credit (a numpy fallback can't pass the lane).
    checks.append(CheckResult(
        f"{kind}/executes_on_metal",
        any_on_metal,
        "no Metal runtime on this host — displacement not exercised"))
    return checks


@task("displacement/matmul_epilogue")
def _t_disp_matmul_epilogue(rng: Any) -> list[CheckResult]:
    return _grade_lane("matmul_epilogue", rng)


@task("displacement/norm_chain")
def _t_disp_norm_chain(rng: Any) -> list[CheckResult]:
    return _grade_lane("norm_chain", rng)


@task("displacement/attention")
def _t_disp_attention(rng: Any) -> list[CheckResult]:
    return _grade_lane("attention", rng)


@task("displacement/pointwise")
def _t_disp_pointwise(rng: Any) -> list[CheckResult]:
    return _grade_lane("pointwise", rng)


@task("displacement/gated_matmul")
def _t_disp_gated_matmul(rng: Any) -> list[CheckResult]:
    return _grade_lane("gated_matmul", rng)
