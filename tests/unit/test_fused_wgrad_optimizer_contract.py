"""Numeric contract for fusing an optimizer into the weight-gradient epilogue.

This is a **declared oracle** (Decision #31b) for work that has not landed yet:
the `matmul -> optimizer` fusion proposed as W3/W4 in
``docs/audit/compiler/FORGE_ASSESSMENT.md``.  Every proposition the design rests
on is stated here as an executable check, so the pass that eventually implements
it inherits a gate instead of an argument.  Identifiers P1..P6 match §2 of that
document.

Pure numpy on purpose.  These are properties of the *update rule*, not of any
backend, so they must hold on every host with no build and no device.  When W3
lands, its lit fixtures prove the IR-level residency claim (Decision #19 /
``LOWER-COUNT-1``); these prove the arithmetic the fusion is allowed to assume.

Source: FORGE, arXiv:2606.22932v2.  Propositions 1 and 2 of that paper are P1
and P3 here; P2/P4/P5/P6 are conditions the paper states loosely, omits, or
measures inside a recipe that masks them.
"""
from __future__ import annotations

import numpy as np
import pytest

B1, B2, EPS, WD, LR = 0.9, 0.999, 1e-8, 0.01, 1e-3


def _bf16(a):
    """Round-to-nearest-even to bf16 and back to f32.

    Mirrors ``fusion_core._round_to_storage``'s fallback so the test does not
    depend on ``ml_dtypes`` being installed.
    """
    u = np.ascontiguousarray(a, np.float32).view(np.uint32).astype(np.uint64)
    return ((((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16)
            .astype(np.uint32).view(np.float32))


def _adamw(W, G, m, v, t, lr=LR, quant=None):
    """One AdamW step; returns (W, m, v, |dW|). ``quant`` models state storage."""
    q = quant or (lambda x: x)
    m = q((B1 * m + (1 - B1) * G).astype(np.float32))
    v = q((B2 * v + (1 - B2) * G * G).astype(np.float32))
    dW = (np.float32(lr) * (m / (1 - B1 ** t))
          / (np.sqrt(v / (1 - B2 ** t)) + np.float32(EPS)))
    return (W * np.float32(1 - lr * WD) - dW).astype(np.float32), m, v, dW


def _wgrad(dY, X, chunk):
    """dY^T X accumulated in fp32 over ``chunk``-sized token slices.

    This is the token-axis (K) reduction order the fused epilogue uses.  The
    Graph-IR precondition is that K stays whole inside the tile — what
    ``MatmulOp::getLoopIteratorTypes`` encodes as {parallel, parallel} over
    (M, N) with ``tessera.full_k`` stamped on the tiled clone.

    DETERMINISM (the CI flake this replaces, observed on PR #566): the chunk
    partial used to be a BLAS GEMM (``dY_chunk.T @ X_chunk``), whose per-element
    K-dot reduction order is a shape- and hardware-dependent kernel choice —
    the full-matrix GEMM and a 64x64 tile GEMM can legitimately differ in ULPs,
    and DID differ across GitHub's heterogeneous runner CPUs (AVX-512 kernels
    vs not), intermittently breaking the bitwise assertions below.  The
    per-token outer-product accumulation used here gives every output element
    an IDENTICAL sequential fp32 chain (t ascending within the chunk, chunks
    ascending) on both the monolithic and the tiled side, on any hardware —
    which is the accumulation-order pin the bitwise contract needs.
    """
    acc = np.zeros((dY.shape[1], X.shape[1]), np.float32)
    for k in range(0, dY.shape[0], chunk):
        partial = np.zeros_like(acc)
        for t in range(k, min(k + chunk, dY.shape[0])):
            partial += dY[t][:, None] * X[t][None, :]
        acc += partial
    return acc


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# P1 — per-tile exactness, and the legality predicate that protects it
# ---------------------------------------------------------------------------

def test_p1_tile_fused_step_is_bitwise_identical_to_one_shot(rng):
    """A tile-local state update fused per tile equals the monolithic step.

    Bitwise, not approximately: the token-axis summation order is identical, so
    there is no reassociation to explain away.  This is what licenses the
    rewrite to elide the gradient value entirely.
    """
    BT, V, H, TK = 512, 128, 256, 64
    dY = rng.standard_normal((BT, V), np.float32)
    X = rng.standard_normal((BT, H), np.float32)
    W = rng.standard_normal((V, H), np.float32) * 0.02
    m = rng.standard_normal((V, H), np.float32) * 1e-3
    v = np.abs(rng.standard_normal((V, H), np.float32)) * 1e-6

    W_ref, m_ref, v_ref, _ = _adamw(W, _wgrad(dY, X, TK), m, v, 7)

    W_f, m_f, v_f = W.copy(), m.copy(), v.copy()
    for i in range(0, V, 64):
        for j in range(0, H, 64):
            si, sj = slice(i, i + 64), slice(j, j + 64)
            g = _wgrad(dY[:, si], X[:, sj], TK)
            W_f[si, sj], m_f[si, sj], v_f[si, sj], _ = _adamw(
                W[si, sj], g, m[si, sj], v[si, sj], 7)

    assert np.array_equal(W_f, W_ref)
    assert np.array_equal(m_f, m_ref)
    assert np.array_equal(v_f, v_ref)


def test_p1_exactness_is_independent_of_tile_partition(rng):
    """Any partition of the weight index set works — the tiling is free.

    Prop. 1 quantifies over *any* partition, so the autotuner may pick tile
    geometry per shape and per architecture (W7) without touching correctness.
    """
    BT, V, H, TK = 256, 96, 128, 32
    dY = rng.standard_normal((BT, V), np.float32)
    X = rng.standard_normal((BT, H), np.float32)
    W = rng.standard_normal((V, H), np.float32) * 0.02
    z = np.zeros((V, H), np.float32)
    ref = _adamw(W, _wgrad(dY, X, TK), z, z, 3)[0]

    for tr, tc in ((32, 128), (96, 16), (16, 32), (V, H)):
        out = W.copy()
        for i in range(0, V, tr):
            for j in range(0, H, tc):
                si, sj = slice(i, i + tr), slice(j, j + tc)
                out[si, sj] = _adamw(
                    W[si, sj], _wgrad(dY[:, si], X[:, sj], TK),
                    z[si, sj], z[si, sj], 3)[0]
        assert np.array_equal(out, ref), f"partition {tr}x{tc} diverged"


def test_p1_multi_consumer_weight_must_not_fuse(rng):
    """A weight on two backward paths (tied embedding) breaks the fusion.

    Justifies the single-use guard.  In value-semantic Graph IR the true
    gradient is ``add(%g1, %g2)``, so ``%g1``'s consumer is the add rather than
    the optimizer and a `producer=matmul, consumer=optimizer, one-use` match
    cannot fire — the guard is structural, needing no aliasing analysis.  This
    test pins the cost of getting it wrong.
    """
    BT, V, H = 256, 96, 64
    W = rng.standard_normal((V, H), np.float32) * 0.02
    z = np.zeros((V, H), np.float32)
    g1 = rng.standard_normal((BT, V), np.float32).T @ rng.standard_normal((BT, H), np.float32)
    g2 = rng.standard_normal((BT, V), np.float32).T @ rng.standard_normal((BT, H), np.float32)

    correct = _adamw(W, g1 + g2, z, z, 1)[0]
    W1, m1, v1, _ = _adamw(W, g1, z, z, 1)          # stepping each path in turn
    wrong = _adamw(W1, g2, m1, v1, 2)[0]

    assert np.abs(wrong - correct).max() / np.abs(correct).max() > 1e-3


# ---------------------------------------------------------------------------
# P2 — global-norm clipping has no safe fused form
# ---------------------------------------------------------------------------

def test_p2_exact_clipping_cannot_be_recovered_post_hoc(rng):
    """m_t(c) - c*m_t(1) = (1-c)*b1*m_{t-1}: recovery needs G or m_{t-1}.

    Both are O(P), so an exactly-clipped step cannot be reconstructed from an
    unclipped fused step plus a scalar.  This is why ``grad_clip_scope =
    "global"`` must fail closed rather than fall back (Decision #21a).
    """
    V, H, c = 96, 64, 0.37
    # The residual below is a difference of near-equal quantities, so evaluate
    # the identity in f64; the f32 magnitude check that follows is the part
    # that matters operationally.
    G = rng.standard_normal((V, H))
    m_prev = rng.standard_normal((V, H)) * 1e-3

    m_clipped = B1 * m_prev + (1 - B1) * c * G
    m_unclipped = B1 * m_prev + (1 - B1) * G
    residual = m_clipped - c * m_unclipped

    # Exactly the term that carries m_{t-1}: no scalar applied to the unclipped
    # state can produce it, so an O(P) tensor is required either way.
    np.testing.assert_allclose(residual, (1 - c) * B1 * m_prev, rtol=1e-9)
    assert np.abs(residual).max() / np.abs(m_clipped).max() > 1e-3


def test_p2c_clipping_is_exact_under_sgd(rng):
    """Under SGD, clipping the gradient and scaling lr are the same operation.

    The exception that names the cause: the obstruction is A's nonlinearity,
    not the fusion.  An affine A may carry the clip scalar into U for free.
    """
    V, H, c = 96, 64, 0.37
    W = rng.standard_normal((V, H), np.float32) * 0.02
    g = rng.standard_normal((V, H), np.float32)
    np.testing.assert_allclose(W - LR * (c * g), W - (LR * c) * g, rtol=0, atol=1e-7)


def test_p2ab_approximate_clipping_policies_do_not_contain_a_spike():
    """Neither 'clip the update' nor 'delayed norm' does what clipping is for.

    Metric note: final-weight distance is useless here — Adam trajectories
    decorrelate, so every policy including *no clipping* scores alike.  What
    clipping exists to bound is a spike, so measure the spike: the single-step
    displacement, and the second-moment inflation that persists ~1/(1-b2)
    steps afterwards.

    'clip-the-update' bounds the displacement only because Adam was already
    approximately scale-invariant, and lets the spike into v unchanged.
    'delayed' bounds nothing on the step the spike arrives.
    """
    V, H, SPIKE, T = 96, 64, 300, 320
    rg = np.random.default_rng(4)
    W0 = rg.standard_normal((V, H), np.float32) * 0.02
    z = lambda: np.zeros((V, H), np.float32)
    tau = 1.15 * np.sqrt(V * H)

    gs = [rg.standard_normal((V, H), np.float32) for _ in range(T + 1)]
    gs[SPIKE] *= 50.0

    arms = {k: [W0.copy(), z(), z()] for k in
            ("exact", "update", "delayed", "none")}
    disp = {k: [] for k in arms}
    vrms = {k: [] for k in arms}
    prev_n = None
    for t in range(1, T + 1):
        g = gs[t]
        c = min(1.0, tau / float(np.linalg.norm(g)))
        cd = 1.0 if prev_n is None else min(1.0, tau / prev_n)
        for k, grad, lr in (("exact", c * g, LR), ("update", g, LR * c),
                            ("delayed", cd * g, LR), ("none", g, LR)):
            W, m, v = arms[k]
            W, m, v, dW = _adamw(W, grad, m, v, t, lr=lr)
            arms[k][:] = [W, m, v]
            disp[k].append(float(np.abs(dW).max()))
            vrms[k].append(float(np.sqrt(v.mean())))
        prev_n = float(np.linalg.norm(g))

    i = SPIKE - 1
    # 'delayed' is indistinguishable from no clipping on the arriving spike.
    np.testing.assert_allclose(disp["delayed"][i], disp["none"][i], rtol=1e-6)
    # Both approximations inflate v as much as not clipping at all.
    for k in ("update", "delayed"):
        assert vrms[k][i + 10] / vrms["exact"][i + 10] > 2.5, k
        np.testing.assert_allclose(
            vrms[k][i + 10], vrms["none"][i + 10], rtol=1e-3)


# ---------------------------------------------------------------------------
# P3 — affine state updates reduce exactly across ranks
# ---------------------------------------------------------------------------

def test_p3_nonlinear_state_update_blocks_stepping_on_partials(rng):
    """AdamW's second moment is quadratic, so per-rank partials cannot step.

    The obstruction belongs to the update rule, not to the fusion: it binds any
    scheme that would step before the cross-rank sum.
    """
    V, H, R, BT = 64, 48, 4, 256
    z = np.zeros((V, H), np.float32)
    parts = [(rng.standard_normal((BT // R, V), np.float32),
              rng.standard_normal((BT // R, H), np.float32)) for _ in range(R)]
    G_r = [dY.T @ X for dY, X in parts]

    v_true = B2 * z + (1 - B2) * sum(G_r) ** 2
    v_seq = z.copy()
    for Gr in G_r:
        v_seq = B2 * v_seq + (1 - B2) * Gr * Gr
    assert np.abs(v_seq - v_true).max() / np.abs(v_true).max() > 0.1


@pytest.mark.parametrize("collective", ["all_reduce", "reduce_scatter"])
def test_p3_affine_state_update_reduces_exactly(collective, rng):
    """Seed A(S) on one rank and 0 on the others; reducing the STATE is exact.

    Holds for both collectives, so it composes with Decision #16 (ZeRO-2),
    where the owner of a shard is unambiguous.  No rank ever forms a gradient
    and no staging buffer is needed; wire volume is unchanged.
    """
    V, H, R, mu = 64, 48, 4, 0.95
    B = rng.standard_normal((V, H), np.float32) * 1e-3
    G_r = [rng.standard_normal((V, H), np.float32) for _ in range(R)]
    expected = mu * B + sum(G_r)

    if collective == "all_reduce":
        contrib = [mu * B + G_r[0]] + G_r[1:]
        np.testing.assert_allclose(sum(contrib), expected, rtol=1e-5, atol=1e-6)
        return

    for owner, idx in enumerate(np.array_split(np.arange(V), R)):
        contrib = [(mu * B[idx] + G_r[r][idx]) if r == owner else G_r[r][idx]
                   for r in range(R)]
        np.testing.assert_allclose(
            sum(contrib), expected[idx], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# P4 — micro-batch accumulation
# ---------------------------------------------------------------------------

def test_p4_affine_state_absorbs_micro_batch_accumulation(rng):
    """Prop. 2(iii) across time: accumulate into momentum, not into a buffer.

    Exact for affine A, so the gradient-shaped accumulator disappears for
    SGD/momentum/Muon.  It does NOT disappear for AdamW: v is quadratic, and
    sum(g_k^2) != (sum g_k)^2.  The paper is right that the pool returns there.
    """
    V, H, K, mu = 64, 48, 8, 0.95
    B = rng.standard_normal((V, H), np.float32) * 1e-3
    gk = [rng.standard_normal((V, H), np.float32) for _ in range(K)]

    acc = mu * B.copy()
    for g in gk:
        acc = acc + g
    np.testing.assert_allclose(acc, mu * B + sum(gk), rtol=1e-5, atol=1e-5)

    z = np.zeros((V, H), np.float32)
    v_ref = B2 * z + (1 - B2) * sum(gk) ** 2
    v_acc = z.copy()
    for g in gk:
        v_acc = B2 * v_acc + (1 - B2) * g * g
    assert np.abs(v_acc - v_ref).max() / np.abs(v_ref).max() > 0.1


# ---------------------------------------------------------------------------
# P5 — conditional routing breaks decay folded into the GEMM's beta
# ---------------------------------------------------------------------------

def test_p5_conditional_routing_needs_a_gradient_free_decay_pass():
    """Folding state decay into the accumulate's beta skips untouched tiles.

    What makes the fusion free — carrying the decay as the GEMM's beta — applies
    alpha only where a contribution arrives.  An expert nothing routes to on a
    step stops decaying while every other expert's state decays.  Tessera has
    `tessera.moe_swiglu_block` and MoE routing, so this is a verifier condition
    (`routing = "conditional"`), not a comment.
    """
    E, T = 8, 200
    routed = np.random.default_rng(1).random((T, E)) < 0.5
    m_ok = np.zeros(E)
    m_beta_folded = np.zeros(E)
    for t in range(T):
        for e in range(E):
            g = 1.0 if routed[t, e] else 0.0
            m_ok[e] = B1 * m_ok[e] + (1 - B1) * g       # decay always applied
            if routed[t, e]:                            # decay folded into beta
                m_beta_folded[e] = B1 * m_beta_folded[e] + (1 - B1) * g

    assert m_beta_folded.mean() / m_ok.mean() > 1.5


# ---------------------------------------------------------------------------
# P6 — the precision benefit is conditional on state dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "weight_q, state_q, min_gain, max_gain",
    [
        ("f32", "f32", 100.0, None),   # unmasked: the fp32 accumulator dominates
        ("f32", "bf16", None, 2.0),    # masked by bf16 state rounding
        ("bf16", "bf16", None, 2.0),   # the paper's BF16-everywhere recipe
    ],
    ids=["fp32_master_fp32_state", "fp32_master_bf16_state", "bf16_everywhere"],
)
def test_p6_fp32_accumulator_gain_depends_on_state_dtype(
        weight_q, state_q, min_gain, max_gain):
    """Reading the fp32 register accumulator is worth ~3 orders of magnitude —
    but only when the optimizer state is not itself quantized.

    This is the compile-time decision W5 exposes as a diagnostic: whether the
    fusion's numerical benefit is realizable is a function of
    `numeric_policy.accum` x state dtype.  Measuring it inside a
    BF16-everywhere recipe (as the source paper does) reports ~1x and hides the
    real effect.
    """
    q = {"f32": lambda x: x, "bf16": _bf16}
    wq, sq = q[weight_q], q[state_q]
    V, H, BT, STEPS = 128, 256, 512, 20

    rg = np.random.default_rng(5)
    W0 = rg.standard_normal((V, H), np.float32) * 0.02
    ref = W0.astype(np.float64)
    mR = np.zeros_like(ref)
    vR = np.zeros_like(ref)
    A = wq(W0.copy()); mA = np.zeros_like(A); vA = np.zeros_like(A)
    F = wq(W0.copy()); mF = np.zeros_like(F); vF = np.zeros_like(F)

    for t in range(1, STEPS + 1):
        dY = rg.standard_normal((BT, V), np.float32)
        X = rg.standard_normal((BT, H), np.float32)
        g32 = (dY.T @ X).astype(np.float32)          # fp32 register accumulator
        g64 = dY.T.astype(np.float64) @ X.astype(np.float64)

        mR = B1 * mR + (1 - B1) * g64
        vR = B2 * vR + (1 - B2) * g64 ** 2
        ref = ref * (1 - LR * WD) - LR * (mR / (1 - B1 ** t)) / (
            np.sqrt(vR / (1 - B2 ** t)) + EPS)

        A, mA, vA, _ = _adamw(A, _bf16(g32), mA, vA, t, quant=sq); A = wq(A)
        F, mF, vF, _ = _adamw(F, g32, mF, vF, t, quant=sq); F = wq(F)

    err = lambda a: float(np.sqrt(((a - ref) ** 2).mean())
                          / np.sqrt((ref ** 2).mean()))
    gain = err(A) / err(F)
    if min_gain is not None:
        assert gain > min_gain, f"expected >{min_gain}x, got {gain:.2f}x"
    if max_gain is not None:
        assert gain < max_gain, f"expected <{max_gain}x, got {gain:.2f}x"
