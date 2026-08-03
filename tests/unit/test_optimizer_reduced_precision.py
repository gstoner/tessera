"""Optimizer behavior at reduced precision — fp16 AND bf16 are both needed.

They fail differently, and neither substitutes for the other:

  * **fp16 tests RANGE.** Adam's second moment squares the gradient
    (``v = b2*v + (1-b2)*g**2``). fp16's smallest normal is 6.1e-5, so a
    gradient of 1e-4 squares to 1e-8 and **underflows to zero** — the classic
    mixed-precision training failure that loss scaling exists to prevent.
    bf16 cannot see this: its smallest normal is 1.2e-38.

  * **bf16 tests RESOLUTION.** bf16 buys its range by spending mantissa bits —
    8, versus fp16's 10 and fp32's 23. Near 1.0 the representable step is
    ~0.0039, so a 1e-3 Adam update **rounds away to nothing** and the parameter
    never moves. fp16 (step ~9.8e-4) survives the same test.

So bf16 — the dtype usually preferred for activations because of its range — is
the WORST choice for parameter *storage*. That is precisely why real
mixed-precision training keeps an f32 master copy of the weights, and it is why
`ops.adam` is deliberately excluded from the storage-dtype-preservation rule in
`op_catalog.DELIBERATELY_UNDECLARED`: forcing its state back to the parameter
dtype would destroy the update.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest


def _step(dtype, steps, lr=1e-3, grad=1e-4):
    from tessera import ops

    param = np.full((4,), 1.0, dtype=dtype)
    m = np.zeros((4,), dtype=dtype)
    v = np.zeros((4,), dtype=dtype)
    g = np.full((4,), grad, dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(1, steps + 1):
            out = ops.adam(param, g, m, v, step=i, lr=lr)
            if isinstance(out, tuple):
                param, m, v = (np.asarray(x) for x in out[:3])
            else:
                param = np.asarray(out)
    return param, m, v


def test_fp16_second_moment_does_not_underflow():
    """The f32 compute/state defaults protect Adam from fp16 g**2 underflow.

    In raw fp16, `(1e-4)**2` is exactly zero. If Adam accumulated the second
    moment at the parameter dtype, `v` would collapse to zero and the update
    would divide by `sqrt(0)+eps` — i.e. explode or stall. `ops.adam` defaults
    to `compute_dtype='fp32'` / `state_dtype='fp32'` precisely to avoid this,
    and this test is what makes that default load-bearing rather than
    incidental.
    """
    raw = np.asarray(1e-4, dtype=np.float16) ** 2
    assert float(raw) == 0.0, "fp16 g**2 no longer underflows; premise changed"

    _, _, v = _step(np.float16, steps=1)
    v_max = float(np.asarray(v).astype(np.float32).max())
    assert v_max > 0.0, (
        "Adam's second moment collapsed to zero at fp16 — the f32 state "
        "default is not being applied"
    )


def test_fp16_parameters_still_learn():
    """fp16 has enough mantissa (10 bits) for a 1e-3 update near 1.0."""
    param, _, _ = _step(np.float16, steps=100)
    moved = float(np.max(np.abs(np.asarray(param).astype(np.float32) - 1.0)))
    assert moved > 1e-2, f"fp16 parameters barely moved: {moved:.3e}"


def test_bf16_parameters_stagnate_at_realistic_learning_rates():
    """PINNED BEHAVIOR, not a bug to fix here: bf16 weights do not move.

    bf16 has 8 mantissa bits, so near 1.0 the representable step is ~0.0039.
    A 1e-3 Adam update rounds to nothing and 100 steps produce EXACTLY zero
    change — silently, with no error.

    This is inherent to the format, not a defect in `ops.adam`; the remedy is
    an f32 master copy of the parameters held by the caller. The test exists so
    that (a) nobody assumes bf16 parameter storage trains, and (b) if a master
    -weight path is added, this turns into an expected failure that has to be
    revisited rather than silently passing.
    """
    import ml_dtypes

    bf16 = ml_dtypes.bfloat16
    param, _, _ = _step(bf16, steps=100)
    moved = float(np.max(np.abs(np.asarray(param).astype(np.float32) - 1.0)))
    assert moved == 0.0, (
        f"bf16 parameters moved by {moved:.3e} — if a master-weight path was "
        "added, update this test and the DELIBERATELY_UNDECLARED rationale "
        "for the optimizer ops"
    )

    # The mechanism is resolution, not a broken update: raise the step size
    # above bf16's spacing and the very same call moves the parameter.
    param_big, _, _ = _step(bf16, steps=100, lr=1e-1)
    moved_big = float(np.max(np.abs(np.asarray(param_big).astype(np.float32) - 1.0)))
    assert moved_big > 1.0, (
        "bf16 parameters did not move even at lr=1e-1; the stagnation is not "
        "simple round-off and needs investigating"
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_full_precision_and_fp16_both_train(dtype_name):
    """fp32 and fp16 both make real progress under identical settings."""
    dtype = getattr(np, dtype_name)
    param, _, _ = _step(dtype, steps=100)
    moved = float(np.max(np.abs(np.asarray(param).astype(np.float32) - 1.0)))
    assert moved > 1e-2, f"{dtype_name} parameters barely moved: {moved:.3e}"
