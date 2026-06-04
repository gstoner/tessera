"""Opt-in diagnostic harness for the Apple GPU control-flow bulk-run segfault.

This file is intentionally skipped by default. Enable it with
``TESSERA_RUN_APPLE_GPU_CF_STRESS=1`` when investigating the pre-existing
MPSGraph bulk ordering issue where ``test_apple_gpu_control_flow.py`` passes in
isolation but can segfault in a larger Apple slice.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from tessera import runtime as R


pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_RUN_APPLE_GPU_CF_STRESS") != "1",
    reason="set TESSERA_RUN_APPLE_GPU_CF_STRESS=1 to run Apple CF stress harness",
)


@pytest.mark.skipif(sys.platform != "darwin", reason="MPSGraph stress is Darwin-only")
def test_cf_while_generate_after_bulk_bmm_dispatches():
    iterations = int(os.environ.get("TESSERA_APPLE_GPU_CF_STRESS_ITERS", "20"))
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
