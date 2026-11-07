"""Effect inference by call-site scan (E2E-REAL-6F, effect inference off
``_OpExtractor``).

``EffectLattice.infer`` no longer lowers the function through the AST Graph IR
front end. It scans the body's call sites, resolves each by object identity,
and joins an unresolved call to ``top``. These tests pin the resolution rules,
and the ``deterministic=True`` fail-open the scan closes: a host RNG call that
never touches a traced value is invisible to a trace, so before this change a
deterministic program could call ``np.random`` and still be accepted and run.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.effect_scan import scan_call_effects
from tessera.compiler.effects import Effect, EffectLattice, TesseraEffectError

ops_alias = ts.ops
relu_alias = ts.ops.relu


def _pure_helper(x):
    return ts.ops.relu(x)


def _numpy_helper(x):
    return np.exp(x)


def _nested_pure_helper(x):
    return _pure_helper(ts.ops.sigmoid(x))


def _recursive_helper(x):
    return _recursive_helper(x)


# ── resolution ───────────────────────────────────────────────────────────────


def test_resolved_ops_carry_their_registered_effect():
    def pure(x, w):
        return ts.ops.relu(ts.ops.matmul(x, w))

    def random(x):
        return ts.ops.dropout(x)

    def collective(x):
        return ts.ops.all_reduce(x)

    assert scan_call_effects(pure).effect == Effect.pure
    assert scan_call_effects(random).effect == Effect.random
    assert scan_call_effects(random).effectful_ops == ("dropout",)
    assert scan_call_effects(collective).effect == Effect.collective


def test_aliases_resolve_by_identity():
    def via_namespace_alias(x):
        return ops_alias.relu(x)

    def via_function_alias(x):
        return relu_alias(x)

    def via_local_import(x):
        import tessera as T
        from tessera.ops import sigmoid as sig

        return sig(T.ops.relu(x))

    for fn in (via_namespace_alias, via_function_alias, via_local_import):
        scan = scan_call_effects(fn)
        assert scan.effect == Effect.pure, (fn.__name__, scan)
        assert scan.unresolved == ()


def test_plain_python_helpers_are_scanned_through():
    def outer(x):
        return _nested_pure_helper(x)

    def nested_def(x):
        def inner(y):
            return ts.ops.relu(y)

        return inner(x)

    def bound_lambda(x):
        act = lambda y: ts.ops.relu(y)  # noqa: E731
        return act(x)

    for fn in (outer, nested_def, bound_lambda):
        assert scan_call_effects(fn).effect == Effect.pure, fn.__name__


@pytest.mark.parametrize(
    "fn,unresolved",
    [
        (lambda x: np.exp(x), "np.exp"),
        (lambda x: _numpy_helper(x), "_numpy_helper->np.exp"),
        (lambda x: _recursive_helper(x), "_recursive_helper->_recursive_helper->_recursive_helper"),
        (lambda x: getattr(ts.ops, "relu")(x), "<dynamic>"),
        (lambda x: x.reshape(4), "x.reshape"),
        (lambda x: print(x), "print"),
    ],
)
def test_unresolved_calls_fail_closed_to_top(fn, unresolved):
    scan = scan_call_effects(fn)
    assert scan.effect == Effect.top
    assert any(name.startswith(unresolved.split("->")[0]) for name in scan.unresolved), scan


def test_dict_dispatch_is_unresolved():
    def dispatch(x):
        table = {"act": ts.ops.relu}
        return table["act"](x)

    assert scan_call_effects(dispatch).effect == Effect.top


def test_decorators_and_constraints_are_not_body_calls():
    @ts.jit
    def constrained(x):
        ts.require(ts.constraint.Divisible("D", 4))
        return ts.ops.relu(x)

    assert scan_call_effects(constrained._fn).effect == Effect.pure


def test_pure_builtins_resolve_and_shadowing_does_not():
    def loops(x):
        for _ in range(len(x.shape)):
            x = ts.ops.relu(x)
        return x

    assert scan_call_effects(loops).effect == Effect.pure

    def shadowed(x):
        return range(x)

    # A module global that shadows a builtin is resolved like any other name.
    globals()["range"] = np.arange
    try:
        assert scan_call_effects(shadowed).effect == Effect.top
    finally:
        del globals()["range"]


def test_in_place_write_is_memory_not_unknown():
    def write(a, b):
        a[:] = ts.ops.relu(b)
        return a

    scan = scan_call_effects(write)
    assert scan.effect == Effect.memory
    assert scan.unresolved == ()


def test_unavailable_source_is_top():
    namespace: dict = {}
    exec("def f(x):\n    return x\n", namespace)
    assert scan_call_effects(namespace["f"]).effect == Effect.top
    assert scan_call_effects(namespace["f"], "def f(x):\n    return x\n").effect == Effect.pure


def test_lattice_reports_unresolved_calls_as_offending():
    def host_rng(x):
        noise = np.random.rand(4)
        return ts.ops.relu(x)

    effect, offending = EffectLattice().infer_with_ops(host_rng)
    assert effect == Effect.top
    assert "np.random.rand" in offending


# ── deterministic contract ───────────────────────────────────────────────────


def test_deterministic_refuses_unseeded_host_rng_at_decoration():
    def host_rng(x):
        noise = np.random.rand(4).astype(np.float32)
        return ts.ops.relu(x)

    with pytest.raises(TesseraEffectError, match="np.random.rand"):
        ts.jit(deterministic=True)(host_rng)
    # A seed does not help: Tessera's seed does not seed numpy's global RNG.
    with pytest.raises(TesseraEffectError, match="np.random.rand"):
        ts.jit(deterministic=True, seed=7)(host_rng)


def test_deterministic_admits_pure_helpers_and_in_place_writes():
    def composed(x):
        return _nested_pure_helper(x)

    def write(a, b):
        a[:] = ts.ops.relu(b)
        return a

    ts.jit(deterministic=True)(composed)
    ts.jit(deterministic=True)(write)


def test_deterministic_random_needs_a_seed():
    def dropout(x):
        return ts.ops.dropout(x)

    with pytest.raises(TesseraEffectError, match="without a seed"):
        ts.jit(deterministic=True)(dropout)
    ts.jit(deterministic=True, seed=3)(dropout)


# ── call-time authority ──────────────────────────────────────────────────────


def test_unresolved_call_keeps_the_tracer_from_claiming_authority():
    """The trace of this body is pure (only ``relu`` touches a traced value);
    the scan keeps the program at ``top``, so the trace must not become the
    frontend authority for it."""

    def host_rng(x):
        noise = np.random.rand(4).astype(np.float32)
        return ts.ops.relu(x)

    fn = ts.jit(host_rng)
    assert fn.inferred_effect == Effect.top
    fn(np.ones(4, np.float32))
    assert fn.frontend_authority == "legacy_candidate_effectful_signature"


def test_pure_program_still_traces():
    def pure(x):
        return _pure_helper(x)

    fn = ts.jit(pure)
    assert fn.inferred_effect == Effect.pure
    np.testing.assert_allclose(fn(np.array([-1.0, 2.0], np.float32)), [0.0, 2.0])
    assert fn.frontend_authority == "tracer"
