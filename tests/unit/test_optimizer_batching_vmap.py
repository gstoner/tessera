"""S-series #3 (2026-06-01) — optimizer-step batching rule.

Closes the ``batching_rule`` axis for the functional optimizers by
proving ``vmap`` composes with each optimizer step. Two invariants,
chosen per optimizer's math:

* **Elementwise optimizers** (sgd / momentum / nesterov / adam /
  adamw / lion): every parameter element updates independently of the
  others, so a per-row ``vmap`` over a batch of independent models
  must equal applying the optimizer to the whole stacked batch at
  once. We assert ``vmap(row_step)(batch) ≈ optimizer(batch)``.

* **Reduction / matrix optimizers** (lamb — per-layer trust ratio is
  a norm over the whole param; muon — Newton–Schulz orthogonalization
  over a 2-D matrix): the whole-batch call would fold the reduction
  across the batch axis (wrong per-model semantics), so we assert
  ``vmap(row_step)(batch) ≈ manual per-row loop`` — which is exactly
  the per-model answer vmap is supposed to give.

Both forms run the REAL ``tessera.autodiff.vmap`` (scan-then-stack) over
the REAL ``tessera.optim`` functional steps — no mocks. Passing this is
the evidence that flips ``optimizer`` / ``functional_optimizer_step``
from ``batching_rule = partial`` → ``complete`` in
``primitive_coverage.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.autodiff import vmap
from tessera import optim


_B = 4  # batch of independent models
_D = 8  # param vector length


def _batch(seed: int, shape=(_B, _D)):
    rng = np.random.default_rng(seed)
    params = rng.standard_normal(shape, dtype=np.float32)
    grads = rng.standard_normal(shape, dtype=np.float32) * 0.1
    return params, grads


# ---- Elementwise optimizers: vmap(row) == whole-batch -----------------

def _row_new_params(step_fn):
    """Wrap a (params, grads) -> (new_params[, state]) optimizer into a
    function returning just new_params (for vmap stacking)."""
    def row(p, g):
        out = step_fn(p, g)
        return out[0] if isinstance(out, tuple) else out
    return row


ELEMENTWISE = {
    "sgd": lambda p, g: optim.sgd(p, g, lr=0.1),
    "momentum": lambda p, g: optim.momentum(p, g, None, lr=0.1),
    "nesterov": lambda p, g: optim.nesterov(p, g, None, lr=0.1),
    "adam": lambda p, g: optim.adam(p, g, None, lr=0.01),
    "adamw": lambda p, g: optim.adamw(p, g, None, lr=0.01, weight_decay=0.01),
    "lion": lambda p, g: optim.lion(p, g, None, lr=0.01),
}


@pytest.mark.parametrize("name", sorted(ELEMENTWISE))
def test_elementwise_optimizer_vmap_equals_whole_batch(name):
    step = ELEMENTWISE[name]
    # Deterministic per-name seed (avoid hash() — it's salted per run).
    seed = 0xAA00 + sum(ord(c) for c in name)
    params, grads = _batch(seed)
    row = _row_new_params(step)

    # Per-row vmap over the batch axis.
    vmapped = vmap(row, in_axes=0)(params, grads)
    vmapped = np.asarray(vmapped)

    # Whole-batch call (elementwise → must match per-row exactly).
    whole = step(params, grads)
    whole = np.asarray(whole[0] if isinstance(whole, tuple) else whole)

    assert vmapped.shape == params.shape
    np.testing.assert_allclose(vmapped, whole, rtol=1e-5, atol=1e-6,
                                err_msg=f"{name}: vmap != whole-batch")


# ---- Reduction / matrix optimizers: vmap(row) == per-row loop ---------

def test_lamb_vmap_equals_per_row_loop():
    """LAMB's trust ratio is a norm over the whole param — per-model
    semantics require vmap; verify against an explicit per-row loop."""
    params, grads = _batch(0x1A3B)
    row = _row_new_params(lambda p, g: optim.lamb(p, g, None, lr=0.01))

    vmapped = np.asarray(vmap(row, in_axes=0)(params, grads))
    manual = np.stack([row(params[i], grads[i]) for i in range(_B)])

    assert vmapped.shape == params.shape
    np.testing.assert_allclose(vmapped, manual, rtol=1e-5, atol=1e-6)


def test_muon_vmap_equals_per_row_loop():
    """Muon orthogonalizes each 2-D matrix param (Newton–Schulz/SVD);
    per-model semantics require vmap over a batch of matrices."""
    # Batch of B independent (m, n) matrices.
    rng = np.random.default_rng(0x9C0FFEE % 2**31)
    P = rng.standard_normal((_B, 6, 4), dtype=np.float32)
    G = rng.standard_normal((_B, 6, 4), dtype=np.float32) * 0.1
    row = _row_new_params(lambda p, g: optim.muon(p, g, None, lr=0.1))

    vmapped = np.asarray(vmap(row, in_axes=0)(P, G))
    manual = np.stack([row(P[i], G[i]) for i in range(_B)])

    assert vmapped.shape == P.shape
    np.testing.assert_allclose(vmapped, manual, rtol=1e-4, atol=1e-5)


# ---- Registry contract: batching axis is now complete -----------------

def test_optimizer_batching_axis_is_complete():
    """The proof above is what flips these categories complete. Pin the
    registry so a regression that reverts the classifier is caught."""
    from tessera.compiler import primitive_coverage as pc
    # A representative sample of the functional optimizers must report
    # batching_rule = complete now.
    for op_name in ("sgd", "adam", "adamw", "lion", "lamb", "muon",
                    "momentum", "nesterov"):
        cov = pc.coverage_for(op_name)
        if cov is None:
            continue  # not all optimizers are registry primitives
        assert cov.contract_status.get("batching_rule") == "complete", (
            f"{op_name}: batching_rule is "
            f"{cov.contract_status.get('batching_rule')!r}, expected "
            f"'complete' after the vmap-over-optimizer-step proof")
