"""MSW-3 — each new optimizer against its source definition.

The oracle is a direct transcription of the numbered definition from
Jentzen, Kuckuck & von Wurstemberger, *Mathematical Introduction to Deep
Learning* (arXiv 2310.20360v3), written in explicit loop form below and
NEVER by calling `tessera.optim`. Comparing an implementation to itself
proves only that it is deterministic.

Everything runs in fp64 on a fixed quadratic `f(x) = 1/2 x^T A x`, whose
gradient `A x` is exact — so a trajectory mismatch is a transcription
error, not accumulated arithmetic noise.

Trajectories, not endpoints: two different methods can land near the same
minimum of a convex quadratic, so an endpoint check would pass for an
optimizer transcribed as the wrong method entirely.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera import optim


A = np.array([[3.0, 0.5, 0.0],
              [0.5, 2.0, 0.25],
              [0.0, 0.25, 1.5]], dtype=np.float64)
X0 = np.array([1.0, -2.0, 0.5], dtype=np.float64)
STEPS = 12
FP64 = dict(compute_dtype="fp64", state_dtype="fp64")


def _grad(x):
    return A @ x


def _run(fn, **kw):
    """Drive a Tessera optimizer for STEPS and collect the trajectory."""
    params, state, out = {"w": X0.copy()}, None, []
    for _ in range(STEPS):
        params, state = fn(params, {"w": _grad(params["w"])}, state, **kw)
        out.append(np.asarray(params["w"], dtype=np.float64).copy())
    return np.array(out)


# --- oracles: transcriptions of the definitions, in their own terms ---------


def _ref_adagrad(lr, eps):
    """`def:determ_adagrad` eq. (1)."""
    x, m, out = X0.copy(), np.zeros(3), []
    for _ in range(STEPS):
        g = _grad(x)
        m = m + g ** 2
        x = x - lr * g / (eps + np.sqrt(m))
        out.append(x.copy())
    return np.array(out)


def _ref_rmsprop(lr, beta, eps, *, bias_adjusted):
    """`def:determ_RMSprop` eq. (1)-(2), or `def:determ_RMSprop_bias`."""
    x, m, out = X0.copy(), np.zeros(3), []
    for n in range(1, STEPS + 1):
        g = _grad(x)
        m = beta * m + (1.0 - beta) * g ** 2
        second = m / (1.0 - beta ** n) if bias_adjusted else m
        x = x - lr * g / (eps + np.sqrt(second))
        out.append(x.copy())
    return np.array(out)


def _ref_adadelta(beta, delta, eps):
    """`def:determ_adadelta` eq. (1)-(4)."""
    x, m, d, out = X0.copy(), np.zeros(3), np.zeros(3), []
    for _ in range(STEPS):
        g = _grad(x)
        m = beta * m + (1.0 - beta) * g ** 2
        step = np.sqrt((eps + d) / (eps + m)) * g
        x_next = x - step
        d = delta * d + (1.0 - delta) * np.abs(x_next - x) ** 2
        x = x_next
        out.append(x.copy())
    return np.array(out)


def _ref_shampoo(lr, eps, x0, amat):
    """`def:determ_Shampoo` eq. (1)-(4), on a matrix parameter.

    No eigenvalue flooring here: `L_0 = eps I` plus Gram matrices is
    positive definite by construction, so the transcription needs no
    numerical guard and stays independent of the one the implementation has.
    """
    def inv4(mat):
        w, v = np.linalg.eigh((mat + mat.T) * 0.5)
        return (v * (w ** -0.25)) @ v.T

    x, out = x0.copy(), []
    d1, d2 = x.shape
    left, right = eps * np.eye(d1), eps * np.eye(d2)
    for _ in range(STEPS):
        g = amat @ x
        left = left + g @ g.T
        right = right + g.T @ g
        x = x - lr * (inv4(left) @ g @ inv4(right))
        out.append(x.copy())
    return np.array(out)


# --- the checks -------------------------------------------------------------


def test_adagrad_matches_its_definition():
    got = _run(optim.adagrad, lr=0.1, eps=1e-8, **FP64)
    np.testing.assert_allclose(got, _ref_adagrad(0.1, 1e-8), rtol=0, atol=1e-12)


@pytest.mark.parametrize("bias_adjusted", [False, True])
def test_rmsprop_matches_its_definition(bias_adjusted):
    got = _run(optim.rmsprop, lr=0.05, beta=0.9, eps=1e-8,
               bias_adjusted=bias_adjusted, **FP64)
    ref = _ref_rmsprop(0.05, 0.9, 1e-8, bias_adjusted=bias_adjusted)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


def test_rmsprop_bias_adjustment_actually_changes_the_trajectory():
    """Guards the parametrization above from being vacuous.

    If the flag were ignored, both cases would compare against references
    that are themselves identical only if the reference ignored it too —
    so pin that the two trajectories genuinely differ, most visibly on the
    first step where the correction is largest.
    """
    plain = _run(optim.rmsprop, lr=0.05, beta=0.9, eps=1e-8,
                 bias_adjusted=False, **FP64)
    adjusted = _run(optim.rmsprop, lr=0.05, beta=0.9, eps=1e-8,
                    bias_adjusted=True, **FP64)
    assert not np.allclose(plain[0], adjusted[0])


def test_adadelta_matches_its_definition():
    got = _run(optim.adadelta, beta=0.9, delta=0.9, eps=1e-6, **FP64)
    np.testing.assert_allclose(got, _ref_adadelta(0.9, 0.9, 1e-6),
                               rtol=0, atol=1e-12)


def test_adadelta_default_lr_reproduces_the_definition():
    """`lr` is a Tessera extension; its default must be the identity.

    A default of anything but 1.0 would mean `optim.adadelta(...)` is not
    the method the docstring cites.
    """
    import inspect
    assert inspect.signature(optim.adadelta).parameters["lr"].default == 1.0


def test_shampoo_matches_its_definition():
    rng = np.random.default_rng(3)
    x0 = rng.standard_normal((3, 2))
    amat = A.copy()
    params, state, got = {"w": x0.copy()}, None, []
    for _ in range(STEPS):
        params, state = optim.shampoo(
            params, {"w": amat @ np.asarray(params["w"], dtype=np.float64)},
            state, lr=0.05, eps=1e-4, **FP64)
        got.append(np.asarray(params["w"], dtype=np.float64).copy())
    np.testing.assert_allclose(np.array(got), _ref_shampoo(0.05, 1e-4, x0, amat),
                               rtol=0, atol=1e-10)


def test_shampoo_accepts_a_vector_as_a_d_by_1_matrix():
    params, state = {"w": X0.copy()}, None
    params, state = optim.shampoo(params, {"w": _grad(params["w"])}, state,
                                  lr=0.05, **FP64)
    assert np.asarray(params["w"]).shape == (3,)


@pytest.mark.parametrize("shape", [(), (2, 3, 4)])
def test_shampoo_refuses_ranks_with_no_canonical_matrix_split(shape):
    """Rank 0 and rank >= 3 fail closed rather than being flattened (#21a)."""
    p = {"w": np.zeros(shape)}
    with pytest.raises(ValueError, match="not a matrix|rank"):
        optim.shampoo(p, {"w": np.zeros(shape)}, None, lr=0.05)


def test_midpoint_sgd_matches_its_definition():
    """`def:midpointSGD` eq. (1)."""
    x = X0.copy()
    ref = []
    lr = 0.05
    for _ in range(STEPS):
        probe = x - 0.5 * lr * _grad(x)
        x = x - lr * _grad(probe)
        ref.append(x.copy())

    params, state, got = {"w": X0.copy()}, None, []
    for _ in range(STEPS):
        params, state = optim.midpoint_sgd(
            params, lambda p: {"w": _grad(np.asarray(p["w"], dtype=np.float64))},
            state, lr=lr, compute_dtype="fp64")
        got.append(np.asarray(params["w"], dtype=np.float64).copy())
    np.testing.assert_allclose(np.array(got), np.array(ref), rtol=0, atol=1e-12)


def test_midpoint_sgd_is_not_plain_sgd():
    """The second gradient evaluation is the method; pin that it happens.

    Reusing the first gradient at the probe point would degrade this to SGD
    with a half-step and still converge on a quadratic — passing any
    endpoint check while being the wrong method.
    """
    lr = 0.05
    mid, _ = optim.midpoint_sgd(
        {"w": X0.copy()}, lambda p: {"w": _grad(np.asarray(p["w"], dtype=np.float64))},
        None, lr=lr, compute_dtype="fp64")
    plain = X0 - lr * _grad(X0)
    assert not np.allclose(np.asarray(mid["w"]), plain)


def test_midpoint_sgd_refuses_state_it_would_silently_drop():
    with pytest.raises(ValueError, match="stateless"):
        optim.midpoint_sgd({"w": X0.copy()}, lambda p: {"w": _grad(p["w"])},
                           {"velocity": 1}, lr=0.05)


def test_midpoint_sgd_preserves_parameter_storage_dtype():
    """fp16 in, fp16 out (review on #695).

    The step is computed in fp32; returning it uncast silently widens the
    whole parameter tree after one step, which nothing notices until memory
    or a dtype assertion does.
    """
    params = {"w": np.array([1.0, -2.0, 0.5], dtype=np.float16)}
    out, _ = optim.midpoint_sgd(
        params, lambda q: {"w": np.asarray(q["w"], dtype=np.float64) * 0.1},
        None, lr=0.1)
    assert out["w"].dtype == np.float16


def test_midpoint_sgd_is_not_a_graph_op():
    """It must not be advertised at a Graph boundary it cannot honour.

    Its second operand is a callable, and `TraceBuilder.record_op` requires
    every positional Graph operand to be a Tracer — so a catalog entry would
    fail by construction on every compiled use. Pinned so a later "the
    registries should agree" sweep re-adds it deliberately rather than by
    reflex.
    """
    from tessera.compiler.op_catalog import OP_SPECS
    import tessera

    assert "midpoint_sgd" not in OP_SPECS
    assert getattr(tessera.ops, "midpoint_sgd", None) is None
    assert callable(optim.midpoint_sgd)


def test_new_optimizers_honour_the_state_contract():
    """The MSW-3 state contract (#693) covers the new arrivals too."""
    for name, slots in (("adagrad", ("m",)), ("rmsprop", ("m", "step")),
                        ("adadelta", ("m", "delta")), ("shampoo", ("left", "right"))):
        fn = getattr(optim, name)
        with pytest.raises(ValueError) as excinfo:
            fn({"w": X0.copy()}, {"w": _grad(X0)}, {})
        message = str(excinfo.value)
        assert name in message and "state=None" in message
        for slot in slots:
            assert slot in message


# --- muon against `def:determ_ideal_Muon` (declared oracle, #31) ------------
#
# The definition specifies the projection by a PROPERTY, not a formula:
#
#     P(A) = argmin { ||O - A||_HS : O O* = I  or  O* O = I }
#
# Tessera computes it as the orthogonal polar factor `U V*` from the SVD,
# which is the closed form of that argmin. So checking `u @ vh` against
# `u @ vh` would prove nothing. These tests check the DEFINING PROPERTY
# instead — membership in the feasible set, and minimality against sampled
# competitors — which is what makes the closed form the right one.


def _random_semi_orthogonal(rng, d1, d2):
    """A uniformly-drawn competitor from the feasible set of the definition."""
    q, _ = np.linalg.qr(rng.standard_normal((max(d1, d2), min(d1, d2))))
    return q if d1 >= d2 else q.T


def test_muon_projection_lands_in_the_feasible_set():
    """`O* O = I` (tall) or `O O* = I` (wide) — the constraint in the def."""
    rng = np.random.default_rng(11)
    for d1, d2 in ((5, 3), (3, 5), (4, 4)):
        a = rng.standard_normal((d1, d2))
        o = optim._orthogonalize_if_matrix(a)
        gram = o.T @ o if d1 >= d2 else o @ o.T
        np.testing.assert_allclose(gram, np.eye(min(d1, d2)), atol=1e-5)


def test_muon_projection_is_the_minimiser_not_merely_feasible():
    """Minimality over sampled competitors — the argmin half of the def.

    Feasibility alone is satisfied by *any* semi-orthogonal matrix, so
    without this a projection that returned a fixed orthogonal matrix and
    ignored `A` entirely would pass the test above.
    """
    rng = np.random.default_rng(12)
    for d1, d2 in ((5, 3), (3, 5)):
        a = rng.standard_normal((d1, d2))
        best = np.linalg.norm(optim._orthogonalize_if_matrix(a) - a)
        for _ in range(200):
            q = _random_semi_orthogonal(rng, d1, d2)
            assert best <= np.linalg.norm(q - a) + 1e-6


def test_muon_trajectory_matches_the_ideal_definition():
    """`def:determ_ideal_Muon` eq. (1)-(3), transcribed independently.

    The recursion is the part this pins: momentum accumulates as
    `m_n = alpha m_{n-1} + g` (the `def:momentum_two` form), and the
    projection is applied to `m_n` — NOT to the raw gradient, which is the
    plausible transcription error the trajectory would otherwise hide.
    """
    rng = np.random.default_rng(13)
    x0 = rng.standard_normal((4, 3))
    amat = np.diag([2.0, 1.5, 1.0, 0.5])
    lr, alpha = 0.05, 0.95

    x, m, ref = x0.copy(), np.zeros_like(x0), []
    for _ in range(STEPS):
        g = amat @ x
        m = alpha * m + g
        u, _s, vh = np.linalg.svd(m, full_matrices=False)
        x = x - lr * (u @ vh)
        ref.append(x.copy())

    params, state, got = {"w": x0.copy()}, None, []
    for _ in range(STEPS):
        params, state = optim.muon(
            params, {"w": amat @ np.asarray(params["w"], dtype=np.float64)},
            state, lr=lr, momentum=alpha)
        got.append(np.asarray(params["w"], dtype=np.float64).copy())

    # fp32 tolerance: `_orthogonalize_if_matrix` computes the SVD in float32.
    np.testing.assert_allclose(np.array(got), np.array(ref), rtol=0, atol=1e-5)


def test_muon_projects_the_momentum_not_the_gradient():
    """Guards the trajectory test above from a transcription that agrees.

    Projecting `g` instead of `m_n` coincides on the FIRST step (where
    `m_1 = g`), so a one-step check cannot separate them. Pin that they
    diverge by the second.
    """
    rng = np.random.default_rng(14)
    x0 = rng.standard_normal((4, 3))
    amat = np.diag([2.0, 1.5, 1.0, 0.5])
    lr, alpha = 0.05, 0.95

    params, state = {"w": x0.copy()}, None
    for _ in range(2):
        params, state = optim.muon(
            params, {"w": amat @ np.asarray(params["w"], dtype=np.float64)},
            state, lr=lr, momentum=alpha)

    x = x0.copy()
    for _ in range(2):
        g = amat @ x
        u, _s, vh = np.linalg.svd(g, full_matrices=False)   # WRONG: projects g
        x = x - lr * (u @ vh)
    assert not np.allclose(np.asarray(params["w"], dtype=np.float64), x, atol=1e-4)


def test_recorded_momentum_formulation_is_the_one_implemented():
    """MSW-3's audit obligation, as an executable claim.

    The source gives FOUR momentum recursions sharing
    `Theta_n = Theta_{n-1} - gamma_n m_n`:

        def:momentum        m = alpha m + (1 - alpha) g
        def:momentum_two    m = alpha m + g            <- Tessera
        def:momentum_three  m = alpha m + (1 - alpha) gamma g
        def:momentum_four   m = alpha m + gamma g

    Recorded in prose, this would rot the first time someone "fixed" the
    missing `(1 - alpha)`. Here it fails instead.
    """
    lr, alpha = 0.1, 0.9
    x = X0.copy()
    params, state = {"w": x.copy()}, None
    got = []
    for _ in range(3):
        params, state = optim.momentum(
            params, {"w": _grad(np.asarray(params["w"], dtype=np.float64))},
            state, lr=lr, momentum=alpha, **FP64)
        got.append(np.asarray(params["w"], dtype=np.float64).copy())

    variants = {
        "def:momentum": lambda m, g: alpha * m + (1 - alpha) * g,
        "def:momentum_two": lambda m, g: alpha * m + g,
        "def:momentum_three": lambda m, g: alpha * m + (1 - alpha) * lr * g,
        "def:momentum_four": lambda m, g: alpha * m + lr * g,
    }
    matched = []
    for label, recur in variants.items():
        y, m, traj = X0.copy(), np.zeros(3), []
        for _ in range(3):
            m = recur(m, _grad(y))
            y = y - lr * m
            traj.append(y.copy())
        if np.allclose(np.array(traj), np.array(got), atol=1e-12):
            matched.append(label)
    assert matched == ["def:momentum_two"], (
        f"momentum matched {matched}, expected exactly ['def:momentum_two']")


# --- the flat (compiler-visible) ABIs --------------------------------------
#
# Catalog optimizers carry state as explicit TENSOR operands, not as a Python
# dict — adafactor states the convention: "compiler-visible flat ABIs keep
# optimizer state as explicit tensor operands". The first version of this PR
# registered these four in the catalog while the wrappers forwarded straight
# to the tree API, so the flat call the catalog advertised handed an ndarray
# to `_resolve_state` and was rejected (review on #695).


def test_flat_abis_agree_with_the_tree_form():
    """The flat ABI must be the SAME method, not a second implementation.

    This is the property that keeps two entry points from drifting into two
    optimizers (#31): one step of each, from identical inputs, must agree.
    """
    import tessera

    p0 = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    g0 = np.array([0.1, 0.2, -0.3], dtype=np.float32)
    z = np.zeros(3, dtype=np.float32)

    flat_p, flat_m = tessera.ops.adagrad(p0, g0, z, lr=0.1, eps=1e-8)
    tree_p, tree_s = optim.adagrad({"w": p0}, {"w": g0}, None, lr=0.1, eps=1e-8)
    np.testing.assert_allclose(flat_p, tree_p["w"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(flat_m, tree_s["m"]["w"], rtol=0, atol=1e-6)

    flat_p, flat_m = tessera.ops.rmsprop(p0, g0, z, lr=0.05, beta=0.9, eps=1e-8)
    tree_p, tree_s = optim.rmsprop({"w": p0}, {"w": g0}, None, lr=0.05,
                                   beta=0.9, eps=1e-8)
    np.testing.assert_allclose(flat_p, tree_p["w"], rtol=0, atol=1e-6)

    flat_p, flat_m, flat_d = tessera.ops.adadelta(p0, g0, z, z, beta=0.9,
                                                  delta=0.9, eps=1e-6)
    tree_p, tree_s = optim.adadelta({"w": p0}, {"w": g0}, None, beta=0.9,
                                    delta=0.9, eps=1e-6)
    np.testing.assert_allclose(flat_p, tree_p["w"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(flat_d, tree_s["delta"]["w"], rtol=0, atol=1e-6)

    m0 = np.eye(3, dtype=np.float32) * 1e-4
    r0 = np.eye(1, dtype=np.float32) * 1e-4
    flat_p, _l, _r = tessera.ops.shampoo(p0, g0, m0, r0, lr=0.05, eps=1e-4)
    tree_p, _ = optim.shampoo({"w": p0}, {"w": g0}, None, lr=0.05, eps=1e-4)
    np.testing.assert_allclose(flat_p, tree_p["w"], rtol=0, atol=1e-5)


def test_flat_abi_arities_match_the_catalog():
    """Every state tensor must fit inside the declared operand count.

    Adadelta and Shampoo carry TWO state tensors; declaring max-arity 3 made
    the only executable flat call exceed the catalog's own declaration.
    """
    from tessera.compiler.op_catalog import OP_SPECS

    for name, expected_max in (("adagrad", 3), ("rmsprop", 3),
                               ("adadelta", 4), ("shampoo", 4)):
        spec = OP_SPECS[name]
        assert spec.max_arity == expected_max, (
            f"{name} declares max {spec.max_arity} operands but its flat "
            f"ABI passes {expected_max}")


def test_flat_rmsprop_refuses_bias_adjustment_without_a_step():
    """ABSENT is not 1 — the lesson the flat adafactor ABI already records.

    A stateful caller that never passes `step` would take the first-step
    correction forever, inflating every update by 1/(1 - beta): 10x at the
    default. Refusing is the only option that cannot silently mis-train.
    """
    import tessera

    p0 = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    g0 = np.array([0.1, 0.2, -0.3], dtype=np.float32)
    with pytest.raises(ValueError, match="step"):
        tessera.ops.rmsprop(p0, g0, np.zeros(3, dtype=np.float32),
                            bias_adjusted=True)


def test_flat_abis_preserve_parameter_storage_dtype():
    import tessera

    p0 = np.array([1.0, -2.0, 0.5], dtype=np.float16)
    g0 = np.array([0.1, 0.2, -0.3], dtype=np.float16)
    z = np.zeros(3, dtype=np.float32)
    assert tessera.ops.adagrad(p0, g0, z, lr=0.1)[0].dtype == np.float16
    assert tessera.ops.rmsprop(p0, g0, z, lr=0.1)[0].dtype == np.float16
    assert tessera.ops.adadelta(p0, g0, z, z)[0].dtype == np.float16


def test_two_state_flat_abis_refuse_half_a_state():
    """Arity 2 or 4, never 3 — enforced, not just asserted in the catalog.

    `test_op_arity_contract` records these as decodable from position
    *because* the two state tensors are all-or-nothing. Before this guard a
    three-operand call fell through to the tree form and complained about a
    missing dictionary slot, pointing at `state=None` — true of the tree API
    and useless to a flat caller whose actual mistake was omitting the second
    tensor.
    """
    import tessera

    p0 = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    g0 = np.array([0.1, 0.2, -0.3], dtype=np.float32)
    z = np.zeros(3, dtype=np.float32)
    for fn, name in ((tessera.ops.adadelta, "adadelta"),
                     (tessera.ops.shampoo, "shampoo")):
        with pytest.raises(ValueError, match="BOTH state tensors"):
            fn(p0, g0, z)
