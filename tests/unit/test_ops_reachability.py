"""A catalog op must be reachable from `tessera.ops`, or no `@jit` body can emit it.

Measured before W2.2: **24 catalog ops had no `tessera.ops` entry point**, and
19 of them were the complex family — `complex_mul`, `complex_exp`, `mobius`,
`stereographic`, `cross_ratio` and friends. Several already had Apple GPU MSL
kernels and per-target capability entries. The compiler declared them, backends
implemented them, dashboards counted them, and no Python program could reach
them.

That gap also hid ordinary defects, because an unreachable op is an unprobed
one. Making the family reachable immediately surfaced four:

  * `mobius` declared 2 operands; the Möbius transform `(az+b)/(cz+d)` takes 5.
  * `conformal_energy_on_sphere` declared 1; it is an energy BETWEEN two point
    sets, so 2.
  * `_as_pair`, the family's declared "single ingest gate", promoted every real
    input to f64 — so `complex_exp(f32)` returned complex128, setting the
    precision for all 16 ops from one line.
  * `complex_abs` was on the `elementwise` default (`same_as_first`), and the
    storage-dtype wrapper ENFORCES declarations: it cast the float32 magnitude
    back to complex64. A wrong declaration is worse than none once something
    acts on it.

The five ops still unreachable are exactly the callable-operand family, which
is a different problem (they take a Python function, not a tensor) and is what
the region work addresses.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tessera.compiler.op_catalog import _SPECS

#: Ops whose operand 0 is a Python CALLABLE rather than a value. A shape rule
#: reads `operand_types`, and there is no tensor type to read — see the
#: recorded reasons in `DELIBERATELY_UNDECLARED`.
_CALLABLE_OPERAND_OPS = frozenset({
    "dz", "dbar", "conformal_jacobian", "check_cauchy_riemann",
    "ebm_langevin_step",
})


def _unreachable() -> list[str]:
    from tessera import ops

    seen: set[str] = set()
    missing: list[str] = []
    for spec in _SPECS:
        if spec.graph_name in seen:
            continue
        seen.add(spec.graph_name)
        if getattr(ops, spec.public_name, None) is None:
            missing.append(spec.public_name)
    return sorted(missing)


def test_only_the_callable_operand_family_is_unreachable():
    """Ratchet: 24 -> 5, and the 5 are a named, understood class."""
    unexpected = [n for n in _unreachable() if n not in _CALLABLE_OPERAND_OPS]
    assert not unexpected, (
        f"catalog ops with no tessera.ops entry point: {unexpected}. "
        "A declared op no @jit body can emit is a contract with no way to "
        "exercise it."
    )


def test_the_complex_family_is_reachable():
    """The single largest reachability gap — 19 of the original 24."""
    from tessera import ops

    for name in (
        "complex_abs", "complex_arg", "complex_conjugate", "complex_div",
        "complex_exp", "complex_log", "complex_mul", "complex_pow",
        "complex_sqrt", "cross_ratio", "is_concyclic", "mobius",
        "mobius_from_three_points", "stereographic", "laplacian_2d",
        "conformal_energy_on_sphere",
    ):
        assert getattr(ops, name, None) is not None, f"ops.{name} missing"


def test_complex_scalar_is_array_like():
    """`np.asarray(ComplexScalar)` must interleave, not box.

    It produced a rank-0 `object` scalar: the value carried both halves and
    exposed neither, so every consumer reaching for `np.asarray` — the registry
    gates, the storage wrapper, benchmark harnesses — saw an opaque box.
    """
    from tessera.complex import ComplexScalar

    value = ComplexScalar(np.array([1.0, 2.0], dtype=np.float32),
                          np.array([3.0, 4.0], dtype=np.float32))
    arr = np.asarray(value)
    assert arr.dtype == np.dtype("complex64"), arr.dtype
    assert arr.shape == (2,)
    np.testing.assert_allclose(arr, np.array([1 + 3j, 2 + 4j], dtype=np.complex64))


@pytest.mark.parametrize(
    ("storage", "expected"),
    [("float32", "complex64"), ("float64", "complex128"),
     ("float16", "complex64")],
)
def test_real_input_keeps_its_width_through_the_ingest_gate(storage, expected):
    """`_as_pair` promoted every real input to f64 unconditionally.

    It is the family's declared single ingest gate, so that one line set the
    precision for all 16 ops — the same "host library picks the precision"
    defect as `np.fft.*`, at the most leveraged point in the surface. f64 stays
    f64 (the oracle path); anything narrower takes the declared compute float.
    """
    from tessera import ops

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = np.asarray(ops.complex_exp(np.array([0.5, 1.5], dtype=storage)))
    assert out.dtype == np.dtype(expected), f"{storage} -> {out.dtype}"


def test_magnitude_and_angle_are_real_not_complex():
    """`complex_abs` returned complex64 — the wrapper enforcing a wrong rule.

    These sit in the `elementwise` kind, whose default is `same_as_first`, and
    the storage-dtype wrapper casts the result back to the operand's dtype. So
    the float32 magnitude became complex64: not a cosmetic mislabel, an
    actively wrong value produced BY the enforcement machinery.
    """
    from tessera import ops

    z = np.array([3 + 4j, 1 + 0j], dtype=np.complex64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        magnitude = np.asarray(ops.complex_abs(z))
        angle = np.asarray(ops.complex_arg(z))
    assert magnitude.dtype == np.dtype("float32"), magnitude.dtype
    assert angle.dtype == np.dtype("float32"), angle.dtype
    np.testing.assert_allclose(magnitude, [5.0, 1.0], atol=1e-6)
