"""Independent finite differences for ragged scale groups and shared storage."""
import numpy as np
import pytest
from tests.support.scaled_product_transpose_oracle import scale_adjoint


def objective(a, b, sa, sb, dy, sk, sn, policy, nk):
    if nk:
        b = b.swapaxes(-1, -2)
    value = 0.0
    for batch in np.ndindex(dy.shape[:-2]):
        aa = a[batch] if policy != "shared_lhs" else a
        bb = b[batch] if policy != "shared_rhs_rows" else b
        ss = sa[batch] if policy != "shared_lhs" else sa
        tt = sb[batch] if policy != "shared_rhs_rows" else sb
        for row in range(aa.shape[0]):
            for col in range(bb.shape[1]):
                for contraction in range(aa.shape[1]):
                    value += (dy[batch][row, col] * aa[row, contraction] *
                              bb[contraction, col] * ss[row, contraction // sk] *
                              tt[contraction // sk, col // sn])
    return value


@pytest.mark.parametrize("prefix", [(2,), (2, 3)])
@pytest.mark.parametrize("policy", ["shared_rhs_rows", "independent_rhs", "shared_lhs"])
@pytest.mark.parametrize("nk", [False, True])
def test_scale_adjoint_matches_every_coordinate_finite_difference(prefix, policy, nk):
    rng = np.random.default_rng(703)
    m, n, k, sk, sn = 2, 5, 7, 3, 2
    lhs = prefix if policy != "shared_lhs" else ()
    rhs = prefix if policy != "shared_rhs_rows" else ()
    a = rng.normal(size=(*lhs, m, k))
    b = rng.normal(size=(*rhs, k, n))
    if nk:
        b = b.swapaxes(-1, -2)
    sa = rng.normal(size=(*lhs, m, 3))
    sb = rng.normal(size=(*rhs, 3, 3))
    dy = rng.normal(size=(*prefix, m, n))
    gradients = scale_adjoint(a, b, sa, sb, dy, scale_k=sk, scale_n=sn,
                             batching=policy, transpose_b=nk)
    for role, gradient in enumerate(gradients):
        for index in np.ndindex(gradient.shape):
            scales = [sa.copy(), sb.copy()]
            scales[role][index] += 1e-5
            plus = objective(a, b, *scales, dy, sk, sn, policy, nk)
            scales[role][index] -= 2e-5
            minus = objective(a, b, *scales, dy, sk, sn, policy, nk)
            np.testing.assert_allclose(gradient[index], (plus-minus)/2e-5,
                                       rtol=2e-8, atol=2e-8)
