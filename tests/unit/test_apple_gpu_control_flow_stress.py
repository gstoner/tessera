"""Regression guard for the Apple GPU control-flow bulk-run segfault.

``tessera_apple_gpu_cf_while_generate_f32`` used to be lowered onto an MPSGraph
``-whileWithInitialInputs:before:after:`` loop. That route ran fine in
isolation but crashed (SIGSEGV) inside MPSGraph's own ``GPU::WhileOpHandler``
constructor during lazy graph specialization once enough MPSGraph executables
had churned through the process — so ``test_apple_gpu_control_flow.py`` passed
alone yet segfaulted the interpreter in a larger Apple slice / the full unit
sweep. The fix (2026-06-04) moves the bounded generate loop to a single
hand-written MSL kernel with a native ``while`` loop, avoiding the fragile
MPSGraph while-op handler entirely.

This test reproduces the original interaction — interleaving bulk ``bmm``
MPSGraph dispatches with ``cf_while_generate`` — so a regression back onto the
MPSGraph ``while`` route (or any other bulk-ordering crash) trips here instead
of taking down the whole suite. Override the iteration count with
``TESSERA_APPLE_GPU_CF_STRESS_ITERS`` when investigating.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from tessera import runtime as R


@pytest.mark.skipif(sys.platform != "darwin", reason="MPSGraph stress is Darwin-only")
@pytest.mark.skipif(
    not R.DeviceTensor.is_metal(), reason="needs a real Metal device"
)
def test_cf_while_generate_after_bulk_bmm_dispatches():
    iterations = int(os.environ.get("TESSERA_APPLE_GPU_CF_STRESS_ITERS", "25"))
    rng = np.random.default_rng(20260612)

    for i in range(iterations):
        a = rng.standard_normal((2, 4, 8)).astype(np.float32) * 0.2
        b = rng.standard_normal((2, 8, 4)).astype(np.float32) * 0.2
        bmm = R._dispatch_gpu_batched_matmul(
            [a, b], {"symbol": "tessera_apple_gpu_bmm_f32"}, np)
        np.testing.assert_allclose(bmm, a @ b, rtol=1e-4, atol=1e-4)

        d, vocab = 8, 16
        W = rng.standard_normal((d, d)).astype(np.float32) * 0.4
        lm = rng.standard_normal((d, vocab)).astype(np.float32) * 0.4
        h0 = rng.standard_normal(d).astype(np.float32) * 0.2
        toks, n = R.apple_gpu_cf_while_generate(
            W, lm, h0, 0, -1, 6, d, vocab, np)
        assert n == 6
        assert len(toks) == 6, f"iteration {i}: unexpected token count"
