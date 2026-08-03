"""A result dtype may not be a function of the input's INTEGER width.

Measured across the catalog before this landed: 29 ops returned four different
float precisions for identical mathematics, chosen only by how the input
integer was stored::

    cos(int8)  -> float16      cos(int16) -> float32
    cos(int32) -> float64      cos(int64) -> float64

That is NumPy's int->float promotion table (int8 -> f16, int16 -> f32,
int32/int64 -> f64) reaching through the reference lane and into Tessera's type
system. A compiler targeting accelerators cannot let the storage width of an
index-like input choose the compute precision of a transcendental — f64 runs at
1/64 rate on the GPUs this targets, and f16-from-int8 silently caps the op at
6.55e4.

`popcount` was the same defect in the integer direction, and had been recorded
as *unfixable*: "the exact integer width is NumPy-version dependent (uint8
under 2.x, int64 under 1.26), so pinning one would be wrong". Backwards. A
result dtype decided by which NumPy the host imported is not a contract, and
the values agreed the whole time — only the dtype moved, which is why it went
unnoticed.

Both are now declared once, in `op_catalog`: `COMPUTE_FLOAT_DTYPE` and
`INDEX_DTYPE`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tessera.compiler.op_catalog import (
    COMPUTE_FLOAT_DTYPE,
    INDEX_DTYPE,
    _SPECS,
    shape_rule_for,
)

_INT_WIDTHS = ("int8", "int16", "int32", "int64")

#: Rules whose result carries an operand's storage dtype — the set the
#: enforcement wrapper installs over. An op outside this set is `unclassified`
#: and is not yet claimed to have any dtype contract, so it is not held to one.
_DECLARED = {"same_as_first", "reduce_all", "reduce_trailing", "select_from_second"}


def _probe(fn, arity, dtype):
    x = np.arange(1, 13, dtype=dtype).reshape(3, 4)
    result = fn(*([x] * arity))
    result = result[0] if isinstance(result, tuple) else result
    return np.asarray(result).dtype


def _declared_ops():
    from tessera import ops

    seen: set[str] = set()
    for spec in _SPECS:
        if spec.graph_name in seen:
            continue
        seen.add(spec.graph_name)
        if shape_rule_for(spec.graph_name) not in _DECLARED:
            continue
        fn = getattr(ops, spec.public_name, None)
        if fn is None or spec.min_arity > 2:
            continue
        yield spec, fn


def test_no_declared_op_picks_its_float_width_from_the_input_int_width():
    """The headline invariant: same math, same result precision."""
    offenders = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for spec, fn in _declared_ops():
            widths = {}
            for name in _INT_WIDTHS:
                try:
                    widths[name] = _probe(fn, spec.min_arity, name)
                except Exception:
                    widths = {}
                    break
            if not widths:
                continue
            kinds = {d.kind for d in widths.values()}
            if kinds == {"f"} and len({d.itemsize for d in widths.values()}) > 1:
                offenders.append(
                    f"{spec.public_name}: "
                    + ", ".join(f"{k}->{v.name}" for k, v in widths.items())
                )

    assert not offenders, (
        "ops whose float result width tracks the input integer width:\n  "
        + "\n  ".join(offenders)
        + "\nAn integer input must promote to the DECLARED compute dtype "
        f"({COMPUTE_FLOAT_DTYPE}), not one derived from the integer's storage."
    )


def test_integer_input_promotes_to_the_declared_compute_dtype():
    """Uniformity is necessary but not sufficient — it must be the right one.

    A rule that promoted everything to f64 would satisfy the test above while
    pinning production to the oracle path.
    """
    from tessera import ops

    expected = np.dtype("float32")
    assert COMPUTE_FLOAT_DTYPE == "fp32", COMPUTE_FLOAT_DTYPE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ("cos", "exp", "log", "sqrt", "tanh", "sigmoid", "softmax"):
            fn = getattr(ops, name)
            for width in _INT_WIDTHS:
                got = _probe(fn, 1, width)
                assert got == expected, f"{name}({width}) -> {got}, want {expected}"


def test_float_inputs_are_untouched_by_the_integer_rule():
    """The integer path must not disturb float storage preservation.

    Without this the fix could 'succeed' by flattening every dtype to f32,
    which would undo the reduced-precision work it sits next to.
    """
    from tessera import ops

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for width in ("float32", "float64"):
            x = np.linspace(0.1, 1.0, 12, dtype=width).reshape(3, 4)
            assert np.asarray(ops.cos(x)).dtype == np.dtype(width), width


def test_popcount_returns_the_declared_index_width_on_any_numpy():
    """The values always agreed; only the dtype moved with the host NumPy.

    `np.bitwise_count` (numpy >= 2.0) returns the input's width, so
    `popcount(int8)` gave uint8 there and int64 from the 1.26 fallback.
    """
    from tessera import ops

    expected = np.dtype(INDEX_DTYPE)
    for width in (*_INT_WIDTHS, "uint8", "uint32"):
        x = np.arange(1, 9, dtype=width)
        out = np.asarray(ops.popcount(x))
        assert out.dtype == expected, f"popcount({width}) -> {out.dtype}"
        assert out.tolist() == [1, 1, 2, 1, 2, 2, 3, 1], width


def test_index_width_is_declared_in_exactly_one_place():
    """Three spellings of one convention is how they drift apart.

    `int64` was written literally in `_shape_reduce_all_index`, again in
    `_shape_same_shape_index`, and derived a third way by `popcount`.
    """
    import inspect

    from tessera.compiler import graph_ir

    source = inspect.getsource(graph_ir)
    for rule in ("_shape_reduce_all_index", "_shape_same_shape_index"):
        body = source.split(f"def {rule}")[1].split("\ndef ")[0]
        assert "INDEX_DTYPE" in body, f"{rule} does not use INDEX_DTYPE"
        assert '"int64"' not in body, f"{rule} still hard-codes int64"


@pytest.mark.parametrize("op_name", ["argmax", "argmin", "argsort", "popcount"])
def test_index_returning_ops_agree_with_the_declared_width(op_name):
    """The runtime result must match what the shape rule claims."""
    from tessera import ops

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = np.arange(1, 13, dtype=np.int32).reshape(3, 4)
        out = np.asarray(getattr(ops, op_name)(x))
    assert out.dtype == np.dtype(INDEX_DTYPE), f"{op_name} -> {out.dtype}"
