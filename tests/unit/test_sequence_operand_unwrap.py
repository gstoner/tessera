"""Sequence operands (`ops.cat` / `ops.stack`) — unwrapping and gradient flow (MC8).

Two defects are pinned here, both found while writing
`examples/matrix_calculus/matrix_calculus_tutorial.py` and written up as MC8 in
`docs/audit/compiler/MATRIX_CALCULUS_REVIEW.md`:

1. **`_unwrap` peeled one `._data` level.** An `nn.Parameter` wraps a
   `DistributedArray` which wraps the ndarray, so one hop stops short and
   `np.asarray` produced a 0-d *object* array. Top-level arguments never showed
   it — the tape converts those before the op runs — but a **sequence** operand
   bypasses that path, so `ops.stack([Parameter, Parameter])` returned a
   `dtype=object` array and reported success, and `ops.cat` raised a raw numpy
   `ValueError` naming neither the op nor the argument.

2. **The tape could not carry a sequence operand at all.** A list argument was
   recorded as neither an operand nor a kwarg, so `vjp_cat` was called with no
   `xs` — reverse mode through `cat`/`stack` raised `TypeError` for *any*
   input, Parameter or not. Forward mode was worse: the trace never saw the
   elements, stayed inactive, and returned a silently **zero** tangent.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import ops
from tessera.autodiff import grad, jacrev, jvp, tape
from tessera.nn.module import Parameter

CODE = "E_TENSOR_UNWRAP"


def _params():
    return Parameter(np.arange(3.0)), Parameter(np.arange(3.0) + 10.0)


# ── 1. unwrapping ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("op_name", ["cat", "stack"])
def test_sequence_op_unwraps_parameters(op_name) -> None:
    p, q = _params()
    out = getattr(ops, op_name)([p, q], axis=0)
    assert isinstance(out, np.ndarray)
    assert out.dtype.kind == "f", f"got dtype {out.dtype} — the object-array bug"
    expected = (np.concatenate if op_name == "cat" else np.stack)(
        [np.arange(3.0), np.arange(3.0) + 10.0], axis=0
    )
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("op_name", ["cat", "stack"])
def test_sequence_op_accepts_mixed_wrapper_depths(op_name) -> None:
    # Parameter (two hops) next to a bare ndarray (zero hops).
    p, _q = _params()
    out = getattr(ops, op_name)([p, np.arange(3.0) + 100.0], axis=0)
    assert out.dtype.kind == "f"
    assert np.isfinite(out).all()


def test_unwrap_is_transitive_and_bounded() -> None:
    # `_unwrap` is a closure inside the ops namespace; exercise it through an op.
    # A self-referential wrapper must hit the depth limit, not spin forever.
    class _Cycle:
        @property
        def _data(self):
            return self

    with pytest.raises(TypeError, match=CODE):
        ops.cat([np.arange(3.0), _Cycle()], axis=0)


# ── 2. fail closed, with a diagnostic that names the argument ───────────────
def test_non_tensor_element_names_the_op_and_the_index() -> None:
    with pytest.raises(TypeError) as excinfo:
        ops.cat([np.arange(3.0), "not-a-tensor"], axis=0)
    message = str(excinfo.value)
    assert message.startswith(CODE)
    assert "cat" in message and "element 1" in message


def test_object_array_is_never_returned() -> None:
    p, q = _params()
    for op_name in ("cat", "stack"):
        out = getattr(ops, op_name)([p, q], axis=0)
        assert out.dtype != object, (
            f"{op_name} returned a dtype=object array — the silent-wrong-answer "
            f"shape this test exists to stop"
        )


def test_empty_sequence_fails_closed() -> None:
    with pytest.raises(ValueError, match=CODE):
        ops.stack([], axis=0)


def test_non_sequence_argument_fails_closed() -> None:
    with pytest.raises(TypeError, match=CODE):
        ops.cat(np.arange(3.0), axis=0)      # 1-d: elements are scalars
    with pytest.raises(TypeError, match=CODE):
        ops.stack(object(), axis=0)          # not iterable at all


@pytest.mark.parametrize("op_name", ["cat", "stack"])
def test_stacked_array_is_a_valid_sequence(op_name) -> None:
    """An array whose leading axis enumerates the operands is a sequence.

    `np.concatenate`/`np.stack` have always accepted this, and the autodiff law
    harness produces exactly this shape when it perturbs a sequence operand
    (`np.asarray(list_of_arrays)` in `laws.chain_check`). Rejecting it would
    break a legitimate calling convention.
    """
    stacked = np.arange(12.0).reshape(2, 2, 3)
    out = getattr(ops, op_name)(stacked, axis=0)
    expected = (np.concatenate if op_name == "cat" else np.stack)(
        list(stacked), axis=0
    )
    np.testing.assert_allclose(out, expected)


def test_diagnostic_code_is_registered() -> None:
    from tessera.compiler.diagnostic_codes import code_lookup

    assert code_lookup(CODE) is not None


# ── 3. reverse mode: gradients reach every element ──────────────────────────
@pytest.mark.parametrize("op_name", ["cat", "stack"])
def test_gradients_reach_each_parameter(op_name) -> None:
    p, q = _params()
    with tape() as t:
        out = getattr(ops, op_name)([p, q], axis=0)
        loss = ops.reduce(ops.mul(out, out), op="sum")
        t.backward(loss, accumulate_param_grad=True)

    # d/dx sum(x^2) = 2x, for each element of the sequence independently.
    np.testing.assert_allclose(np.asarray(p.grad._data), 2.0 * np.arange(3.0))
    np.testing.assert_allclose(np.asarray(q.grad._data), 2.0 * (np.arange(3.0) + 10.0))


def test_cat_gradient_splits_the_cotangent_by_operand() -> None:
    # A weighted sum makes each operand's gradient a distinct, checkable slice.
    def f(a, b):
        return ops.reduce(ops.mul(ops.cat([a, b], axis=0), np.arange(6.0)), op="sum")

    ga, gb = grad(f, argnums=(0, 1))(np.arange(3.0), np.arange(3.0) + 5.0)
    np.testing.assert_allclose(ga, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(gb, [3.0, 4.0, 5.0])


def test_stack_gradient_splits_the_cotangent_by_operand() -> None:
    def f(a, b):
        w = np.arange(6.0).reshape(2, 3)
        return ops.reduce(ops.mul(ops.stack([a, b], axis=0), w), op="sum")

    ga, gb = grad(f, argnums=(0, 1))(np.arange(3.0), np.arange(3.0) + 5.0)
    np.testing.assert_allclose(ga, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(gb, [3.0, 4.0, 5.0])


def test_repeated_operand_accumulates_rather_than_overwrites() -> None:
    # cat([a, a]) must give a gradient of 2x the single-slot contribution.
    def f(a):
        return ops.reduce(ops.mul(ops.cat([a, a], axis=0), np.ones(6)), op="sum")

    np.testing.assert_allclose(grad(f)(np.arange(3.0)), np.full(3, 2.0))


def test_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(20260820)
    a0, b0 = rng.standard_normal(4), rng.standard_normal(4)
    w = rng.standard_normal(8)

    def f(a, b):
        return ops.reduce(ops.mul(ops.cat([a, b], axis=0), w), op="sum")

    ga, gb = grad(f, argnums=(0, 1))(a0, b0)
    step = 1e-6
    da, db = rng.standard_normal(4), rng.standard_normal(4)
    directional = (
        float(f(a0 + step * da, b0 + step * db))
        - float(f(a0 - step * da, b0 - step * db))
    ) / (2 * step)
    np.testing.assert_allclose(float(ga @ da + gb @ db), directional, rtol=1e-6)


def test_jacrev_through_a_sequence_operand() -> None:
    J = jacrev(lambda a: ops.cat([a, a], axis=0))(np.arange(3.0))
    np.testing.assert_allclose(J, np.vstack([np.eye(3), np.eye(3)]))


# ── 4. forward mode: no silent zero tangent ─────────────────────────────────
@pytest.mark.parametrize("op_name", ["cat", "stack"])
def test_forward_mode_tangent_is_not_silently_zero(op_name) -> None:
    other = np.arange(3.0) + 5.0
    seed = np.ones(3)
    _value, tangent = jvp(
        lambda a: getattr(ops, op_name)([a, other], axis=0), np.arange(3.0), seed
    )
    tangent = np.asarray(tangent)
    assert tangent.any(), "forward mode returned an all-zero tangent"
    expected = (np.concatenate if op_name == "cat" else np.stack)(
        [seed, np.zeros(3)], axis=0
    )
    np.testing.assert_allclose(tangent, expected)


def test_forward_mode_tracks_both_elements() -> None:
    def f(a):
        return ops.cat([a, ops.mul(a, scalar=2.0)], axis=0)

    _value, tangent = jvp(f, np.arange(3.0), np.ones(3))
    np.testing.assert_allclose(np.asarray(tangent), np.concatenate([np.ones(3),
                                                                    np.full(3, 2.0)]))


# ── 5. the ordinary path is untouched ───────────────────────────────────────
def test_configuration_lists_are_not_mistaken_for_sequence_operands() -> None:
    # `pad_width` is a list of ints/tuples passed positionally — it must stay
    # configuration, not become a list of operands.
    x = np.arange(3.0)
    with tape() as t:
        out = ops.pad(x, [(1, 1)])
        loss = ops.reduce(ops.mul(out, out), op="sum")
        t.backward(loss, accumulate_param_grad=False)
    np.testing.assert_allclose(t.cotangent[id(x)], 2.0 * x)


def test_single_operand_ops_still_record_one_input_each() -> None:
    a = np.arange(3.0) + 1.0
    b = np.arange(3.0) + 2.0
    with tape() as t:
        out = ops.mul(a, b)
        loss = ops.reduce(out, op="sum")
        t.backward(loss, accumulate_param_grad=False)
        entry = [e for e in t.entries if e.op == "mul"][0]
        assert entry.input_groups is None, "scalar operands must keep the fast path"
    np.testing.assert_allclose(t.cotangent[id(a)], b)
    np.testing.assert_allclose(t.cotangent[id(b)], a)


def test_sequence_entry_records_its_grouping() -> None:
    a, b = np.arange(3.0), np.arange(3.0) + 1.0
    with tape() as t:
        ops.cat([a, b], axis=0)
        entry = [e for e in t.entries if e.op == "cat"][0]
    assert entry.input_groups == (2,), (
        "the sequence operand must be recorded as one group of two inputs"
    )
    assert len(entry.inputs) == 2
