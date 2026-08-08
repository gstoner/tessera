"""C3 — stochastic computation-graph typing.

Derives which values are random, whether a function is deterministic, and which
gradient estimator is legal — from the traced IR, not the Python AST. The
headline test contrasts with EffectLattice's documented fail-open (Decision #5):
an RNG op reached through an alias is invisible to the AST walker but visible to
this structural analysis.
"""

from __future__ import annotations

import pytest
import numpy as np

from tessera.compiler.stochastic_graph import (
    analyze_stochastic_graph,
    certify_deterministic,
    is_distribution_op,
    is_reparameterizable_op,
    DISTRIBUTION_OPS,
)


# IR nodes are (result, op_name, operands) triples — the shape a trace produces.

def test_pure_function_graph_is_deterministic():
    body = [
        ("%1", "tessera.matmul", ("%x", "%w")),
        ("%2", "tessera.add", ("%1", "%b")),
        ("%3", "tessera.relu", ("%2",)),
    ]
    a = analyze_stochastic_graph(body, ["%3"])
    assert a.is_deterministic
    assert a.estimator == "none"
    assert not a.random_values


def test_distribution_node_makes_downstream_random():
    body = [
        ("%z", "tessera.normal", ("%mu", "%sigma")),   # distribution node
        ("%y", "tessera.mul", ("%z", "%w")),           # function node, random parent
    ]
    a = analyze_stochastic_graph(body, ["%y"])
    assert "z" in a.random_values and "y" in a.random_values
    assert not a.is_deterministic
    assert "z" in a.distribution_nodes


def test_real_trace_ssa_prefixes_propagate_randomness():
    # TraceBuilder stores results as ``v0`` but operands/returns as ``%v0``.
    # The stochastic analysis must use one SSA identity space for both forms.
    body = [
        ("v0", "tessera.normal", ("%mu", "%sigma")),
        ("v1", "tessera.add", ("%v0", "%x")),
    ]
    a = analyze_stochastic_graph(body, ["%v1"])
    assert not a.is_deterministic
    assert a.estimator == "pathwise"


def test_random_draw_outside_output_cone_stays_deterministic():
    # A sampled value that never reaches the outputs does not make the function's
    # outputs non-deterministic.
    body = [
        ("%unused", "tessera.bernoulli", ("%p",)),
        ("%1", "tessera.matmul", ("%x", "%w")),
    ]
    a = analyze_stochastic_graph(body, ["%1"])
    assert a.is_deterministic
    assert "unused" in a.random_values  # still typed random...
    assert a.estimator == "none"          # ...but not on the output path


# ── estimator classification (§11.5) ─────────────────────────────────────────

def test_reparameterizable_path_is_pathwise():
    body = [
        ("%z", "tessera.normal", ("%mu", "%sigma")),
        ("%y", "tessera.add", ("%z", "%c")),
    ]
    a = analyze_stochastic_graph(body, ["%y"])
    assert a.estimator == "pathwise"


def test_discrete_distribution_forces_score_function():
    body = [
        ("%z", "tessera.categorical", ("%logits",)),
        ("%y", "tessera.embedding", ("%table", "%z")),
    ]
    a = analyze_stochastic_graph(body, ["%y"])
    assert a.estimator == "score_function"


def test_reparameterizable_op_table_is_subset_of_distributions():
    from tessera.compiler.stochastic_graph import REPARAMETERIZABLE_OPS
    assert REPARAMETERIZABLE_OPS <= DISTRIBUTION_OPS
    assert is_distribution_op("normal") and is_reparameterizable_op("normal")
    assert is_distribution_op("categorical") and not is_reparameterizable_op("categorical")


# ── the Decision #5 fix: derive from IR, not the AST ─────────────────────────

def test_ir_analysis_catches_aliased_rng_that_ast_walker_misses():
    """EffectLattice walks the Python source AST and matches dotted call names,
    so RNG reached through an alias contributes Effect.pure — fail open. The IR
    analysis sees the op by its canonical name and flags it. This is the
    substrate for a fail-closed deterministic=True gate (Decision #30)."""
    from tessera.compiler.effects import EffectLattice, Effect

    # A Python function that samples through an alias, defeating name matching.
    def f(x):
        sampler = tessera_normal_alias  # bound to the RNG op under another name
        noise = sampler(x)
        return noise + x

    def tessera_normal_alias(x):  # stand-in bound name
        return x

    ast_effect = EffectLattice().infer(f)
    # The AST walker does not recognize `sampler(...)` as random → fails open,
    # inferring pure for a function that samples. (This is the documented
    # Decision #5 hole, asserted here so the contrast is concrete.)
    assert ast_effect == Effect.pure

    # The equivalent trace, however, records the op by its canonical name.
    body = [
        ("%noise", "tessera.normal", ("%x",)),   # the op the alias actually calls
        ("%y", "tessera.add", ("%noise", "%x")),
    ]
    det, reason = certify_deterministic(body, ["%y"])
    assert det is False
    assert "normal" not in reason  # reason is about the path, not the spelling
    a = analyze_stochastic_graph(body, ["%y"])
    assert not a.is_deterministic and a.estimator == "pathwise"


def test_certificate_true_only_when_pure():
    pure = [("%1", "tessera.gelu", ("%x",))]
    det, _ = certify_deterministic(pure, ["%1"])
    assert det is True

    rng = [("%1", "tessera.dropout", ("%x",))]
    det2, _ = certify_deterministic(rng, ["%1"])
    assert det2 is False


def test_jit_deterministic_gate_rejects_aliased_rng_from_trace():
    import tessera as ts
    from tessera.compiler.effects import TesseraEffectError

    sampler = ts.ops.dropout

    @ts.jit(deterministic=True)
    def aliased_rng(x):
        return sampler(x)

    with pytest.raises(TesseraEffectError, match="random traced dependency"):
        aliased_rng(np.ones((2, 2)))
