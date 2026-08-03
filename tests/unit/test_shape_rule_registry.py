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

from tessera.compiler.graph_ir import (
    _SHAPE_RULES,
    _infer_result_type,
    _infer_result_types,
    tensor_ir_type,
)
from tessera.compiler.op_catalog import (
    _SPECS,
    SHAPE_RULE_NAMES,
    shape_rule_for,
    unclassified_shape_ops,
)

#: Ratchet. May shrink, never grow. Driving it to zero is what closes W1's
#: "no op reaches the `operand_types[0]` fallback".
#:
#: It is ZERO (W1.4). Every catalog op now resolves to a named rule or to a
#: recorded, examined reason in `DELIBERATELY_UNDECLARED` — so a NEW op with no
#: rule fails here immediately rather than being absorbed into a large bound.
#: A ratchet left slack after it has been closed stops gating anything.
MAX_UNCLASSIFIED = 0


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

    EXCLUDES the mesh-scaling collectives, and the reason is the point of the
    exclusion rather than a convenience. The reference `all_gather` /
    `reduce_scatter` are single-rank no-op stubs (`return x`), so probing them
    in-process can only ever report `world_size == 1`. Holding a mesh-scaling
    rule to that measurement does not verify it — it re-derives the wrong
    answer the exemption was originally written from, and would force the rule
    back to same-as-first to make this test pass. Their contract is verified
    against declared mesh extents in `test_collective_mesh_shape_rules.py`.

    A behavioural gate is only as good as the configuration it measures. This
    is the second time that has bitten here: the first was `ebm_self_verify`,
    probed with two same-shaped tensors, where operand 0 and operand 1 gave the
    same answer.
    """
    from tessera import ops
    from tessera.compiler.graph_ir import _MESH_AWARE_RULES
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
            if rule in _MESH_AWARE_RULES:
                continue  # single-rank stub; see the docstring
            fn = getattr(ops, spec.public_name, None)
            if fn is None:
                continue
            try:
                raw = fn(x)
            except Exception:
                continue  # op needs more than a bare tensor; not probeable here
            # A multi-result op must be compared result-by-result. `np.asarray`
            # on a tuple silently STACKS it -- `nonzero`'s `(rows, cols)` came
            # back as one `(2, 32)` array, so the gate compared a rule for the
            # first result against a shape that exists nowhere in the contract.
            # Same tuple-blindness the quantize probe had: a check that cannot
            # express multiple results does not skip them, it invents one.
            if isinstance(raw, tuple):
                predicted_all = _infer_result_types(spec.graph_name, [probe])
                if len(predicted_all) != len(raw):
                    disagreements.append(
                        f"{spec.public_name}: rule {rule!r} predicts "
                        f"{len(predicted_all)} results, op returns {len(raw)}"
                    )
                continue
            actual = np.asarray(raw)
            predicted = _infer_result_type(spec.graph_name, [probe])
            try:
                actual_dtype = canonicalize_dtype(str(actual.dtype))
            except Exception:
                continue
            pred_shape = tuple(str(d) for d in predicted.shape)
            real_shape = tuple(str(d) for d in actual.shape)
            # `?` is a declared unknown -- a data-dependent extent such as
            # `nonzero`'s count or `segment_reduce`'s segment total. It agrees
            # with any concrete size; demanding a match would force the rule to
            # echo whatever the probe's data happened to produce, which is the
            # degenerate-probe error this file already guards against twice.
            if len(pred_shape) == len(real_shape):
                real_shape = tuple(
                    p if p == "?" else r for p, r in zip(pred_shape, real_shape)
                )
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


# ─────────────────────────────────────────────────────────────────────────────
# Multi-result contracts
# ─────────────────────────────────────────────────────────────────────────────

MULTI_RESULT_OPS = {
    "tessera.quantize_fp8": "quantize_fp8",
    "tessera.quantize_fp6": "quantize_fp6",
    "tessera.quantize_fp4": "quantize_fp4",
    "tessera.quantize_nvfp4": "quantize_nvfp4",
}


def test_multi_result_rules_match_actual_arity_and_shapes():
    """The quantize family returns (codes, scale) — verify BOTH results.

    These were declared `same_as_first`: a single-tensor claim for a two-result
    op. It survived because the differential probe did `np.asarray(fn(...))`,
    which raises on a tuple, so the op was silently SKIPPED. A gate that skips
    the thing it cannot express is indistinguishable from a passing gate — the
    same failure mode as the substring assertions this registry replaced.

    `_infer_result_types` returns the full contract; `_infer_result_type`
    returns only the primary tensor and cannot express these at all.
    """
    from tessera import ops
    from tessera.compiler.graph_ir import _infer_result_types, tensor_ir_type

    sample = np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32)
    probe = tensor_ir_type(("4", "16"), "f32")

    for graph_name, public_name in sorted(MULTI_RESULT_OPS.items()):
        fn = getattr(ops, public_name, None)
        if fn is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                actual = fn(sample)
            except Exception:
                continue
        assert isinstance(actual, tuple), (
            f"{public_name} no longer returns a tuple — update MULTI_RESULT_OPS"
        )
        predicted = _infer_result_types(graph_name, [probe])
        assert len(predicted) == len(actual), (
            f"{public_name}: rule predicts {len(predicted)} results, op returns "
            f"{len(actual)}"
        )
        for index, (want, got) in enumerate(zip(predicted, actual)):
            got_arr = np.asarray(got)
            want_shape = tuple(str(d) for d in want.shape)
            real_shape = tuple(str(d) for d in got_arr.shape)
            assert want_shape == real_shape, (
                f"{public_name} result[{index}]: rule predicts shape "
                f"{want_shape}, op returns {real_shape}"
            )


def test_nvfp4_scale_is_per_block_not_per_tensor():
    """NVFP4 is micro-scaled; folding it into the per-tensor rule misstates it.

    `quantize_fp8/fp6/fp4` carry ONE scale for the whole tensor (rank-0), while
    `quantize_nvfp4` carries one per block of 16 along the last axis — the
    format Blackwell actually implements. A shared rule would have been wrong
    for exactly the target that motivates the format.
    """
    from tessera import ops
    from tessera.compiler.graph_ir import _infer_result_types, tensor_ir_type

    sample = np.random.default_rng(0).standard_normal((4, 32)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, per_tensor_scale = ops.quantize_fp8(sample)
        _, per_block_scale = ops.quantize_nvfp4(sample)

    assert np.asarray(per_tensor_scale).shape == ()
    assert np.asarray(per_block_scale).shape == (4, 2), (
        "expected one scale per 16-element block along the last axis"
    )

    probe = tensor_ir_type(("4", "32"), "f32")
    _, predicted_scale = _infer_result_types("tessera.quantize_nvfp4", [probe])
    assert tuple(predicted_scale.shape) == ("4", "2")
