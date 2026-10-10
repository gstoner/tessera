"""Canonical reverse-family dispatch on the owning SM120 GPU."""
from __future__ import annotations

import pytest
from benchmarks.nvidia.benchmark_public_attention_vjp import run
from tests._support.nvidia import nvidia_cuda_host_ready


@pytest.mark.parametrize(
    "order,wrt,sk,causal,bias",
    [
        (("v", "q", "k"), ("k", "q"), 129, True, None),
        (("bias", "v", "q", "k"), ("bias", "v", "k", "q"), 5, False, (1, 4, 1, 1)),
    ],
)
def test_public_saved_lse_reverse(tmp_path, order, wrt, sk, causal, bias):
    if not nvidia_cuda_host_ready():
        pytest.skip("owning SM120 CUDA device/toolchain unavailable")
    row = run(order, wrt, sk, causal, tmp_path, bias)
    assert row["correctness"] == "independent_fp64_gradients"
    assert row["warm_compiler_subprocesses"] == "forbidden"
    assert row["receipt"]["execution_certificate"]["evidence_scope"] == "exact_device"
