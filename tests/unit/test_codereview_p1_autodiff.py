"""P1 autodiff-stability regressions from the full-source code review.

Two real fixes and two guards for findings that turned out to be false
positives on inspection (kept as guards so the behavior can't silently
regress):

  * vjp_pow — log was evaluated over non-positive bases (divide-by-zero /
    invalid RuntimeWarning + masked NaN).  [FIX]
  * Parameter.grad / tape._accumulate_param_grad — fp16/bf16 parameters
    accumulated gradients in fp16, losing precision.  [FIX]
  * relu/clip — numpy already promotes (fp32 dout x fp16 mask -> fp32), so
    no downcast occurs.  [GUARD]
  * cosine_embedding / nt_xent VJPs — gradients at zero vectors are finite
    (eps protects the denominator), not NaN.  [GUARD]
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import tessera.losses as losses
from tessera.autodiff.tape import _accumulate_param_grad
from tessera.autodiff.vjp import get_vjp
from tessera.nn.module import Parameter


# ── vjp_pow: no spurious warning / NaN on non-positive bases [FIX] ──────────
def test_vjp_pow_no_warning_and_finite_on_nonpositive_base():
    a = np.array([-2.0, 0.0, 3.0])
    b = np.array([2.0, 2.0, 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        da, db = get_vjp("pow")(np.ones(3), a, b)
    # db = dout * a**b * log(a); defined as 0 for a<=0, finite everywhere.
    assert np.all(np.isfinite(db)), db
    np.testing.assert_array_equal(db[:2], [0.0, 0.0])  # a<=0 -> 0 contribution
    # positive base matches the analytic value
    np.testing.assert_allclose(db[2], (3.0**2) * np.log(3.0), rtol=1e-12)


# ── fp16/bf16 param grads accumulate in fp32 [FIX] ──────────────────────────
def test_param_grad_accumulates_in_fp32_for_fp16_param():
    p = Parameter(np.ones((1,), dtype=np.float16))
    for _ in range(4096):
        _accumulate_param_grad(p, np.array([1e-3], dtype=np.float16))
    assert p.grad._data.dtype == np.float32, "fp16 param grad must accumulate in fp32"
    # fp32 accumulation keeps the small contributions; fp16 saturates at 4.0
    assert float(p.grad._data[0]) > 4.05, float(p.grad._data[0])


def test_param_grad_setter_follows_gradient_dtype_not_param_dtype():
    p = Parameter(np.ones((3,), dtype=np.float16))
    p.grad = np.ones((3,), dtype=np.float32)
    assert p.grad._data.dtype == np.float32


# ── relu/clip: cotangent dtype follows dout, not the activation [GUARD] ──────
@pytest.mark.parametrize("op,kw", [("relu", {}), ("clip", {"min": 0.0, "max": 1.5})])
def test_activation_vjp_grad_dtype_follows_dout(op, kw):
    x = np.array([-1.0, 0.5, 2.0], dtype=np.float16)
    (g,) = get_vjp(op)(np.ones(3, dtype=np.float32), x, **kw)
    assert g.dtype == np.float32, f"{op} grad should follow dout dtype, got {g.dtype}"


# ── cosine_embedding / nt_xent VJPs: finite (not NaN) at zero vectors [GUARD]─
def test_cosine_and_ntxent_vjp_finite_at_zero_vectors():
    (gc, *_rest) = get_vjp("cosine_embedding_loss")(
        1.0, np.zeros((1, 4)), np.ones((1, 4)), np.array([1.0])
    )
    assert np.all(np.isfinite(np.asarray(gc)))

    (gn, *_rest2) = get_vjp("nt_xent_loss")(1.0, np.zeros((2, 4)), np.array([0, 0]))
    assert np.all(np.isfinite(np.asarray(gn)))
