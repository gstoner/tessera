import numpy as np
import pytest
from tessera_numerics.validators import finite_difference_grad, check_allclose
from tessera_numerics import tessera_adapter as A

def test_matmul_grad_small():
    # f(x) = sum((x @ B)^2), check grad wrt x by finite differences
    M,K,N = 8, 6, 4
    rng = A.rng(123)
    x = rng.standard_normal((M,K)).astype(np.float64) * 0.1
    B = rng.standard_normal((K,N)).astype(np.float64) * 0.1

    def f_np(xx):
        y = xx @ B
        return np.sum(y*y)

    # Analytic grad: d/dx sum((xB)^2) = 2 * (x B) B^T
    ref_grad = 2.0 * (x @ B) @ B.T
    num_grad = finite_difference_grad(f_np, x, eps=1e-4)
    check_allclose("matmul_grad", num_grad, ref_grad, dtype="float64")
