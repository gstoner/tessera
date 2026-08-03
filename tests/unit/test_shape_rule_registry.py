"""W1.2 — the one shape-rule registry, checked against reality.

`_infer_result_type` used to be a five-case if-chain ending in
``return operand_types[0]``. That fallback is correct for the 60 elementwise
ops and silently wrong for anything whose result differs from its first
operand — and `primitive_coverage` reported the ``shape_rule`` axis CLOSED
across 480 primitives the whole time. Decision #29, "declared but not
consumed", in its purest form.

The registry replaces the implicit fallback with a NAMED rule per op. The
tests below are what make it binding:

  * every declared rule has an implementation, and vice versa;
  * every declared rule AGREES WITH THE OP'S ACTUAL BEHAVIOR — this is the
    check that found `eq`/`lt`/`ge` returning the operand dtype instead of
    bool, `isnan`/`isinf`/`isfinite` doing the same, and `gelu`/`dropout`
    promoting f32 to f64;
  * the `unclassified` list is a shrink-only ratchet.

A shape-only comparison would have missed the dtype bugs entirely, so the
differential check compares BOTH.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tessera.compiler.graph_ir import _SHAPE_RULES, _infer_result_type, tensor_ir_type
from tessera.compiler.op_catalog import (
    _SPECS,
    SHAPE_RULE_NAMES,
    shape_rule_for,
    unclassified_shape_ops,
)

#: Ratchet. May shrink, never grow. Driving it to zero is what closes W1's
#: "no op reaches the `operand_types[0]` fallback".
MAX_UNCLASSIFIED = 106


def test_declared_rule_names_match_implementations():
    """A rule name with no implementation (or vice versa) is drift."""
    declared = set(SHAPE_RULE_NAMES)
    implemented = set(_SHAPE_RULES)
    assert declared == implemented, (
        f"declared-but-unimplemented={sorted(declared - implemented)}, "
        f"implemented-but-undeclared={sorted(implemented - declared)}"
    )


def test_every_catalog_op_resolves_to_a_named_rule():
    """`shape_rule_for` never returns an empty string.

    A caller must not be able to confuse "no rule declared" with "no answer";
    the absence of a rule is itself a named, countable status.
    """
    for spec in _SPECS:
        rule = shape_rule_for(spec.graph_name)
        assert rule, f"{spec.graph_name} resolved to an empty rule"
        assert rule in SHAPE_RULE_NAMES, (
            f"{spec.graph_name} resolved to undeclared rule {rule!r}"
        )


def test_deliberately_undeclared_ops_are_real_and_explained():
    """The third state must stay honest.

    "Examined and deliberately left without a rule" is not the same as "not yet
    looked at", and collapsing them makes the remaining count meaningless. Each
    entry must name a real op and carry a substantive reason — an optimizer
    keeping f32 master state with bf16 params is a *correct* design, not a gap,
    and declaring it storage-preserving would make the enforcement wrapper
    round that state back to bf16.
    """
    from tessera.compiler.op_catalog import DELIBERATELY_UNDECLARED

    real = {spec.graph_name for spec in _SPECS}
    for name, reason in DELIBERATELY_UNDECLARED.items():
        assert name in real, f"{name} is not a catalog op"
        assert len(reason) > 20, f"{name}: reason too thin to be a decision"


def test_no_phantom_declarations():
    """Every declared op key must name a REAL catalog op.

    The first version of the loss block used public names (`tessera.mse_loss`)
    where the graph names are `tessera.loss.mse`. All 16 matched nothing and
    declared nothing -- while *looking* like coverage. That is precisely the
    "declared but not consumed" failure the registry exists to remove, so it
    gets a gate rather than a comment.
    """
    from tessera.compiler.op_catalog import OP_SHAPE_RULE

    real = {spec.graph_name for spec in _SPECS}
    phantom = sorted(k for k in OP_SHAPE_RULE if k not in real)
    assert not phantom, (
        "these shape-rule declarations name no catalog op, so they silently "
        f"declare nothing: {phantom}"
    )


def test_unclassified_ratchet_does_not_grow():
    remaining = unclassified_shape_ops()
    assert len(remaining) <= MAX_UNCLASSIFIED, (
        f"{len(remaining)} ops are unclassified, above the ratchet of "
        f"{MAX_UNCLASSIFIED}. Declare a rule rather than raising the bound."
    )


def test_declared_rules_agree_with_actual_op_behavior():
    """Every DECLARED rule must match what the op really returns.

    This is the check that earns the registry its keep. It compares shape *and*
    dtype: the shapes agreed for `eq`, `isnan`, and `gelu` while their dtypes
    were wrong, so a shape-only check would have passed all three.

    Unary ops only — they can be probed with a single sample without guessing
    each op's operand contract. Extending this to n-ary ops is the natural
    follow-up and would widen the net further.
    """
    from tessera import ops
    from tessera.dtype import canonicalize_dtype

    x = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    probe = tensor_ir_type((4, 8), "f32")

    disagreements = []
    seen: set[str] = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for spec in _SPECS:
            if spec.graph_name in seen or spec.min_arity > 1:
                continue
            seen.add(spec.graph_name)
            rule = shape_rule_for(spec.graph_name)
            if rule == "unclassified":
                continue  # honest gap, covered by the ratchet above
            fn = getattr(ops, spec.public_name, None)
            if fn is None:
                continue
            try:
                actual = np.asarray(fn(x))
            except Exception:
                continue  # op needs more than a bare tensor; not probeable here
            predicted = _infer_result_type(spec.graph_name, [probe])
            try:
                actual_dtype = canonicalize_dtype(str(actual.dtype))
            except Exception:
                continue
            pred_shape = tuple(str(d) for d in predicted.shape)
            real_shape = tuple(str(d) for d in actual.shape)
            if pred_shape != real_shape:
                disagreements.append(
                    f"{spec.public_name}: rule {rule!r} predicts shape "
                    f"{pred_shape}, op returns {real_shape}"
                )
            elif predicted.dtype and predicted.dtype != actual_dtype:
                disagreements.append(
                    f"{spec.public_name}: rule {rule!r} predicts dtype "
                    f"{predicted.dtype}, op returns {actual_dtype}"
                )

    assert not disagreements, (
        "declared shape rules disagree with actual op behavior:\n  "
        + "\n  ".join(disagreements)
    )


@pytest.mark.parametrize(
    "op_name,expected_dtype",
    [
        ("tessera.eq", "bool"),
        ("tessera.lt", "bool"),
        ("tessera.ge", "bool"),
        ("tessera.logical_and", "bool"),
        ("tessera.isnan", "bool"),
        # bitwise ops are NOT predicates — they preserve the integer dtype, so
        # the `logical` kind could not carry one default for all of it.
        ("tessera.bitwise_and", "int32"),
    ],
)
def test_predicates_yield_bool_not_the_operand_dtype(op_name, expected_dtype):
    """Regression for the bug the taxonomy work surfaced.

    `tessera.eq` on two f32 tensors reported `dtype=fp32`. A comparison
    producing a value in the operand's dtype is silently wrong, and nothing
    caught it because the axis was reported closed.
    """
    operand = tensor_ir_type((4, 8), "int32" if expected_dtype == "int32" else "f32")
    result = _infer_result_type(op_name, [operand, operand])
    assert result.dtype == expected_dtype
    assert result.shape == operand.shape


def test_activations_do_not_promote_precision():
    """Regression: `gelu`/`dropout` promoted f32 to f64.

    `np.sqrt(...)` returns a float64 *scalar*, which is strong under NumPy 2
    promotion and dragged the whole array up; `binomial()/(1-p)` did the same
    for dropout. An elementwise activation silently doubling precision and
    memory contradicts Decision #15a (storage dtype lives on the tensor).
    """
    from tessera import ops

    x32 = np.ones((4,), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ("gelu", "relu", "silu", "tanh", "sigmoid", "dropout"):
            fn = getattr(ops, name, None)
            if fn is None:
                continue
            out = np.asarray(fn(x32))
            assert out.dtype == np.float32, (
                f"ops.{name} promoted f32 to {out.dtype}"
            )


def test_declared_rules_hold_at_reduced_precision():
    """Every storage-preserving rule must hold at bf16, not just fp32.

    Production runs reduced precision on accelerators; fp64 is the oracle and
    fp32 the convenient middle. An f32-only check is blind to the failure that
    matters, because the common bug is *upcasting out of* reduced precision.

    Measured before this gate existed: 35 of 96 probed unary ops did not
    preserve bf16, including gelu, silu, sigmoid, dropout, layer_norm, rmsnorm,
    weight_norm and the whole clifford family. Root cause: `ml_dtypes.bfloat16`
    does NOT follow NumPy's weak-scalar promotion, so `x * 0.5` silently yields
    float32.

    All THREE production dtypes are probed. fp16 was originally excluded on the
    grounds that bf16 caught a strict superset for *propagation* -- true at the
    time, and too narrow a conclusion. Several ops ignored the input dtype
    entirely and returned f64 for f32, bf16 and fp16 alike, and f32 is a
    production dtype rather than an oracle: a wrapper that only handled reduced
    precision left f32 callers still getting f64. Probing all three is what made
    that visible.

    fp64 remains the oracle and is not asserted here; its range/precision
    hazards live in test_fp16_range_sensitivity.py.
    """
    import ml_dtypes

    from tessera import ops

    bf16 = ml_dtypes.bfloat16
    sample = np.random.default_rng(0).standard_normal((4, 8))
    preserving = {"same_as_first", "reduce_all", "reduce_trailing"}

    failures = []
    checked = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for spec in _SPECS:
            if spec.min_arity > 1:
                continue
            if shape_rule_for(spec.graph_name) not in preserving:
                continue
            fn = getattr(ops, spec.public_name, None)
            if fn is None:
                continue
            # All five production storage widths. fp8/fp4 are canonical dtypes
            # in `tessera.dtype`, not hypotheticals -- the per-backend contracts
            # already model them (gfx1151 `unsupported`, x86 `emulated`), and an
            # op that silently leaves fp8 forfeits the entire reason an
            # accelerator pipeline chose it.
            probes = (
                ("f32", np.float32, np.dtype(np.float32)),
                ("bf16", bf16, np.dtype(bf16)),
                ("fp16", np.float16, np.dtype(np.float16)),
                ("fp8_e4m3", ml_dtypes.float8_e4m3fn, np.dtype(ml_dtypes.float8_e4m3fn)),
                ("fp4_e2m1", ml_dtypes.float4_e2m1fn, np.dtype(ml_dtypes.float4_e2m1fn)),
            )
            try:
                observed = [
                    (label, np.asarray(fn(sample.astype(dt))).dtype, expected)
                    for label, dt, expected in probes
                ]
            except Exception:
                continue
            checked += 1
            for label, got, expected in observed:
                if got != expected:
                    failures.append(
                        f"{spec.public_name}: {label} in -> {got}"
                    )

    assert checked > 40, f"only {checked} ops probed; the sweep likely broke"
    assert not failures, (
        "ops declared to preserve storage dtype do not:\n  "
        + "\n  ".join(failures)
    )


def test_bfloat16_is_not_detected_by_the_usual_numpy_float_idioms():
    """Pin the trap that caused this whole class of bug.

    Both standard float tests report False/'V' for bf16, so any guard written
    with them silently skips the dtype an accelerator pipeline actually runs
    in. This test exists so the next person meets the trap here rather than in
    production numerics.
    """
    import ml_dtypes

    bf16 = np.dtype(ml_dtypes.bfloat16)
    assert not np.issubdtype(bf16, np.floating), (
        "np.issubdtype(bf16, np.floating) started returning True — the "
        "workaround in _enforce_storage_dtype_preservation may be removable"
    )
    assert bf16.kind == "V", f"bf16.kind is {bf16.kind!r}, expected 'V' (void)"
    # ml_dtypes.finfo is the detection that actually works.
    assert ml_dtypes.finfo(bf16).max > 3e38
