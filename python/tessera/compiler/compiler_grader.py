"""Phase E3 / Pillar 3 (precursor) — internal TensorBench-style compiler grader
(docs/audit/compiler/EVALUATOR_PLAN.md §6, §8).

TensorBench grades a unit of compiler work pass/fail with a reliable oracle and
an adversarial audit against vacuous passes. This wires Tessera's Evaluator
oracles into that contract: a **task** is a named, self-contained compiler-
correctness check graded over **hidden inputs** (a fresh RNG at grade time, never
seen at authoring), and a task only passes when a real oracle agrees — there is
no "it ran, therefore pass" credit.

This is the scored-environment substrate an agent / human change is judged
against. The seed task set below is the pattern, not an exhaustive suite; it
composes the vertical / cross-path / horizontal / metamorphic oracles so each
hardens a distinct compiler property. Anti-cheat is structural:

  * **hidden inputs** — every task is re-exercised on fresh RNG draws;
  * **no silent-fallback credit** — checks ride the Evaluator's provenance gate
    (a numpy fallback can't earn a pass);
  * **no vacuous task** — a task with zero checks is a FAIL, and every seed task
    asserts a numerical/equivalence relation, not mere execution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tessera.compiler.evaluator import (
    Rung,
    cross_path_equivalence,
    evaluate,
    horizontal_equivalence,
    metamorphic_equivalence,
    run_native,
)

import tessera as ts


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class Grade:
    task: str
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        # No vacuous pass: a task with zero checks fails by construction.
        return len(self.checks) > 0 and all(c.passed for c in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed)


# A task: hidden-input RNG → its checks.
TaskFn = Callable[[Any], list[CheckResult]]
_TASKS: dict[str, TaskFn] = {}


def task(name: str) -> Callable[[TaskFn], TaskFn]:
    """Register a graded compiler task."""
    def deco(fn: TaskFn) -> TaskFn:
        if name in _TASKS:
            raise ValueError(f"duplicate task {name!r}")
        _TASKS[name] = fn
        return fn
    return deco


def task_names() -> list[str]:
    return sorted(_TASKS)


def grade(name: str, rng: Any) -> Grade:
    """Grade one task on hidden inputs (the passed-in RNG)."""
    checks = _TASKS[name](rng)
    return Grade(name, tuple(checks))


def grade_all(rng: Any) -> list[Grade]:
    """Grade every registered task. Caller supplies the hidden-input RNG."""
    return [grade(n, rng) for n in task_names()]


# ── module-level jitted programs (source-inspectable for @jit) ───────────────

def _mm(a, b):
    return ts.ops.matmul(a, b)


def _mm_gelu(a, b):
    return ts.ops.gelu(ts.ops.matmul(a, b))


def _sm(a):
    return ts.ops.softmax(a, axis=-1)


def _gelu(a):
    return ts.ops.gelu(a)


_MM_GPU = ts.jit(target="apple_gpu")(_mm)
_MM_CPU = ts.jit(target="apple_cpu")(_mm)
_MM_GELU_GPU = ts.jit(target="apple_gpu")(_mm_gelu)
_SM_GPU = ts.jit(target="apple_gpu")(_sm)
_GELU_GPU = ts.jit(target="apple_gpu")(_gelu)


def _f32(rng: Any, shape: tuple[int, ...]) -> Any:
    import numpy as np

    return (rng.standard_normal(shape) / 4).astype(np.float32)


# ── seed task set (each hardens a distinct compiler property) ────────────────

@task("matmul/apple_gpu/rung7")
def _t_matmul_rung7(rng: Any) -> list[CheckResult]:
    """Vertical + provenance: matmul executes natively on Metal and matches numpy."""
    a, b = _f32(rng, (64, 64)), _f32(rng, (64, 64))
    v = evaluate("apple_gpu", _MM_GPU, (a, b), a @ b, rtol=2e-3, atol=1e-4)
    return [
        CheckResult("rung==device_verified_abi", v.rung is Rung.HARDWARE_VERIFIED, v.detail),
        CheckResult("provenance_ok", v.provenance_ok, "native execution, not a fallback"),
    ]


@task("matmul/cross_path/metal_vs_accelerate")
def _t_matmul_cross_path(rng: Any) -> list[CheckResult]:
    """DESIL: the same matmul agrees across two independent lowering paths."""
    a, b = _f32(rng, (64, 64)), _f32(rng, (64, 64))
    v = cross_path_equivalence([("apple_gpu", _MM_GPU), ("apple_cpu", _MM_CPU)], (a, b))
    return [
        CheckResult("two paths native", len(v.paths) == 2, f"paths={v.paths}"),
        CheckResult("paths equivalent", v.relation == "equivalent", v.detail),
    ]


@task("matmul_gelu/fusion/horizontal")
def _t_matmul_gelu_fusion(rng: Any) -> list[CheckResult]:
    """PolyJuice: the fused matmul→gelu equals its unfused native composition."""
    a, b = _f32(rng, (64, 64)), _f32(rng, (64, 64))

    def unfused(args: tuple[Any, ...]) -> tuple[Any, bool]:
        x, y = args
        mm, n1 = run_native("apple_gpu", _MM_GPU, (x, y))
        if not n1:
            return None, False
        g, n2 = run_native("apple_gpu", _GELU_GPU, (mm,))
        return g, (n1 and n2)

    v = horizontal_equivalence("apple_gpu", _MM_GELU_GPU, (a, b), unfused, rtol=3e-3, atol=1e-3)
    return [CheckResult("fused≡unfused", v.relation == "equivalent", v.detail)]


@task("softmax/metamorphic/shift_invariant")
def _t_softmax_shift(rng: Any) -> list[CheckResult]:
    """Metamorphic: softmax(x) ≡ softmax(x+c) — numerical-stability invariant."""
    import numpy as np

    x = _f32(rng, (16, 16))
    v = metamorphic_equivalence("apple_gpu", _SM_GPU, (x,), (x + np.float32(3.0),))
    return [CheckResult("shift invariant", v.relation == "equivalent", v.detail)]
