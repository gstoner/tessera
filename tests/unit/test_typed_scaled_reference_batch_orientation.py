"""Independent public eager oracle for named batched operand orientations."""
import itertools

import ml_dtypes
import numpy as np
import pytest
import tessera as ts


@pytest.mark.parametrize("policy", ["shared_rhs_rows", "shared_lhs", "independent_rhs"])
@pytest.mark.parametrize("ta,tb,encoded", list(itertools.product((False, True), repeat=3)))
def test_named_batches_preserve_operand_orientation(policy, ta, tb, encoded):
    rng = np.random.default_rng(8942)
    prefix = (2, 3)
    m, n, k = 3, 5, 7
    block_n, block_k = (1, 32) if encoded else (2, 4)
    groups, columns = (k + block_k - 1) // block_k, (n + block_n - 1) // block_n
    lhs_prefix = () if policy == "shared_lhs" else prefix
    rhs_prefix = () if policy == "shared_rhs_rows" else prefix
    lhs = rng.uniform(-1, 1, (*lhs_prefix, m, k)).astype(ml_dtypes.float8_e4m3fn)
    rhs = rng.uniform(-1, 1, (*rhs_prefix, k, n)).astype(ml_dtypes.float8_e4m3fn)
    if encoded:
        sa = rng.integers(125, 130, (*lhs_prefix, m, groups), dtype=np.uint8)
        sb = rng.integers(125, 130, (*rhs_prefix, groups, columns), dtype=np.uint8)
        sa_real = np.exp2(sa.astype(np.float64) - 127)
        sb_real = np.exp2(sb.astype(np.float64) - 127)
    else:
        sa = rng.uniform(.25, 2, (*lhs_prefix, m, groups)).astype(np.float32)
        sb = rng.uniform(.25, 2, (*rhs_prefix, groups, columns)).astype(np.float32)
        sa_real, sb_real = sa.astype(np.float64), sb.astype(np.float64)
    a = lhs.swapaxes(-1, -2) if ta else lhs
    b = rhs.swapaxes(-1, -2) if tb else rhs
    actual = ts.ops.scaled_matmul(
        a, b, sa, sb, transposeA=ta, transposeB=tb, batching=policy,
        numeric_policy={"accum": "fp32", "execution_mode": "exact_per_block"},
        scale_layout={"granularity": "block", "block": [block_n, block_k],
                      "format": "e8m0" if encoded else "fp32"})
    # Scalar indexing avoids using either the reference's matrix multiply or
    # transpose implementation. Shared scales follow their operand's owner.
    expected = np.empty((*prefix, m, n), np.float64)
    for batch in np.ndindex(prefix):
        ai = () if policy == "shared_lhs" else batch
        bi = () if policy == "shared_rhs_rows" else batch
        for row in range(m):
            for col in range(n):
                expected[(*batch, row, col)] = sum(
                    float(lhs[(*ai, row, inner)]) * float(rhs[(*bi, inner, col)])
                    * sa_real[(*ai, row, inner // block_k)]
                    * sb_real[(*bi, inner // block_k, col // block_n)]
                    for inner in range(k))
    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
