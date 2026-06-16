"""Phase 0 production-lane oracle test (docs/spec/PRODUCTION_COMPILER_PLAN.md).

Proves the MLIR/LLVM lane *executed its own codegen* — not just that the answer
matches numpy. The Python lane (numpy) is the ORACLE; the production lane is the
thing under test. Key guardrail (ratified): if the path ever fell back to numpy,
these tests must FAIL — enforced via the unfakeable C++ invocation counter.

Skips only when libtessera_jit is not built (e.g. a non-MLIR CI runner).
"""

import numpy as np
import pytest

from tessera import _jit_boundary as jb

pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`",
)


def test_jit_add_matches_numpy_oracle():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)

    out = jb.jit_add(a, b)

    # Numpy is the oracle, not the executor.
    np.testing.assert_allclose(out, a + b, rtol=1e-6, atol=1e-6)


def test_jit_add_actually_executed_the_compiled_function():
    """Proof-of-execution: the JIT invocation counter MUST advance by exactly 1.

    If jit_add had silently fallen back to numpy, the counter would not move and
    this test would fail — which is the whole point of Phase 0 (prove the
    MLIR/LLVM lane ran, not merely that numbers match).
    """
    a = np.ones((3, 5), dtype=np.float32)
    b = np.full((3, 5), 2.0, dtype=np.float32)

    before = jb.invocation_count()
    out = jb.jit_add(a, b)
    after = jb.invocation_count()

    assert after == before + 1, "compiled function did not execute (silent fallback?)"
    np.testing.assert_array_equal(out, np.full((3, 5), 3.0, dtype=np.float32))


def test_jit_add_is_destination_passing_writes_into_fresh_output():
    # Each call allocates a fresh output; distinct shapes round-trip correctly.
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(6, dtype=np.float32).reshape(2, 3) * 10.0
    out = jb.jit_add(a, b)
    np.testing.assert_array_equal(out, a + b)


@pytest.mark.parametrize(
    "a, b",
    [
        # int32 — the lowering is float-only (f32/f64/f16/bf16); integer dtypes
        # are outside the boundary table (f64 is now wired, see test below).
        (np.ones((4, 4), np.int32), np.ones((4, 4), np.int32)),
        # Shape mismatch — elementwise requires equal shapes.
        (np.ones((4, 8), np.float32), np.ones((8, 4), np.float32)),
        # Scalar (rank-0) — Phase 1 boundary requires rank >= 1.
        (np.float32(1.0), np.float32(2.0)),
    ],
)
def test_jit_add_rejects_out_of_envelope_instead_of_falling_back(a, b):
    """Out-of-envelope inputs must RAISE, never silently compute via numpy.

    A silent numpy fallback here would be a correctness-masking bug: it would
    return the right number while bypassing the compiled lane entirely.

    Note: Phase 1 generalized the elementwise lowering, so rank-3+ adds now
    legitimately execute through the lane (previously a Phase 0 envelope guard);
    Phase 4 added f16 to the boundary. Negatives here are unsupported-dtype +
    shape-mismatch + rank-0, which the boundary still rejects.
    """
    with pytest.raises(jb.TesseraJitError):
        jb.jit_add(np.asarray(a), np.asarray(b))


def test_jit_add_f16_now_executes():
    # Phase 4: f16 (native M1 Max NEON, ARMv8.2-A FP16) is in the boundary table.
    a = np.ones((4, 4), np.float16)
    b = (2.0 * np.ones((4, 4))).astype(np.float16)
    before = jb.invocation_count()
    out = jb.jit_add(a, b)
    assert jb.invocation_count() == before + 1
    assert np.asarray(out).dtype == np.float16
    np.testing.assert_array_equal(np.asarray(out).astype(np.float32),
                                  np.full((4, 4), 3.0, np.float32))


def test_jit_add_phase1_higher_rank_now_executes():
    """Phase 1 generalized elementwise: rank-3 add now executes through the lane.

    This was a deliberate Phase 0 guard ("rank-2 only"); Phase 1 removed it.
    The test pins that the broadening is real (oracle match + counter advance).
    """
    rng = np.random.default_rng(7)
    a = rng.standard_normal((2, 3, 4)).astype(np.float32)
    b = rng.standard_normal((2, 3, 4)).astype(np.float32)
    before = jb.invocation_count()
    out = jb.jit_add(a, b)
    assert jb.invocation_count() == before + 1
    np.testing.assert_allclose(out, a + b, rtol=1e-6, atol=1e-6)
