import numpy as np
import pytest
from scipy import stats
from tessera_numerics import tessera_adapter as A

def test_uniform_ks():
    rng = A.rng(2024)
    xs = rng.random(10_000)
    # Kolmogorov-Smirnov test for U(0,1)
    D, pval = stats.kstest(xs, "uniform")
    assert pval > 1e-3, f"Uniform KS failed: p={pval}"

def test_normal_moments():
    rng = A.rng(2025)
    xs = rng.standard_normal(20_000)
    mu = xs.mean()
    sigma = xs.std()
    assert abs(mu) < 0.02
    assert abs(sigma - 1.0) < 0.03
