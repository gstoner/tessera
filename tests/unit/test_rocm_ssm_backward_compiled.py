"""ROCm (gfx1151) backward pass for Mamba2 selective_ssm.

The compiler generates the reverse-mode adjoint kernel
(generate-rocm-selective-ssm-bwd-kernel): one thread per (b,d), forward-fill
h_traj then a reverse scan accumulating (dx, dA, dB, dC, ddelta) — dx/ddelta are
unique per (b,t,d), while dC/dB (reduce over channels d) and dA (reduce over b,t)
use memref.atomic_rmw addf. Validated on real gfx1151 vs the numpy VJP
(autodiff.vjp.vjp_selective_ssm, forced onto the numpy body via f64 inputs).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tessera.autodiff.vjp import vjp_selective_ssm


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _numpy_vjp(dout, x, A, B, C, delta, gate, state):
    """The pure-numpy adjoint (f64 skips the device fast path)."""
    f = lambda v: None if v is None else np.asarray(v, np.float64)
    return vjp_selective_ssm(
        f(dout), f(x), f(A), f(B), f(C), f(delta), gate=f(gate), state=f(state))


@pytest.mark.parametrize("n", [16, 10])
@pytest.mark.parametrize("a_1d", [False, True])
@pytest.mark.parametrize("gate_on,state_on", [(False, False), (True, True)])
def test_rocm_ssm_backward_matches_vjp(n, a_1d, gate_on, state_on):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(n * 7 + a_1d + (gate_on << 2) + (state_on << 3))
    b, s, d = 2, 7, 4
    x = rng.standard_normal((b, s, d)).astype(np.float32)
    A = (-np.abs(rng.standard_normal((d,) if a_1d else (d, n)))).astype(np.float32)
    B = rng.standard_normal((b, s, n)).astype(np.float32)
    C = rng.standard_normal((b, s, n)).astype(np.float32)
    delta = np.abs(rng.standard_normal((b, s, d)) * 0.1).astype(np.float32)
    dout = rng.standard_normal((b, s, d)).astype(np.float32)
    gate = rng.standard_normal((b, s, d)).astype(np.float32) if gate_on else None
    state = rng.standard_normal((b, d, n)).astype(np.float32) if state_on else None

    ref = _numpy_vjp(dout, x, A, B, C, delta, gate, state)
    got = rt._rocm_selective_ssm_bwd(x, A, B, C, delta, dout, gate, state, np)
    for name, g, r in zip(("dx", "dA", "dB", "dC", "ddelta"), got, ref):
        np.testing.assert_allclose(np.asarray(g, np.float64), r, rtol=0,
                                   atol=5e-3, err_msg=f"{name} mismatch")


def test_rocm_ssm_backward_codegen_lowers():
    # GPU-free: the bwd kernel emits the atomic cross-channel reductions and
    # lowers cleanly through ROCDL (atomic_rmw -> llvm.atomicrmw).
    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    directive = ('module {\n  "tessera_rocm.selective_ssm_bwd"() {name = "ssb"} '
                 ': () -> ()\n}\n')
    gen = subprocess.run(
        [str(opt), "-", "--generate-rocm-selective-ssm-bwd-kernel"],
        input=directive, capture_output=True, text=True)
    assert gen.returncode == 0, gen.stderr
    assert gen.stdout.count("memref.atomic_rmw") == 3   # dC, dB, dA2d
    low = subprocess.run(
        [str(opt), "-",
         "--pass-pipeline=builtin.module(generate-rocm-selective-ssm-bwd-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=directive, capture_output=True, text=True)
    assert low.returncode == 0, low.stderr
    assert "llvm.atomicrmw" in low.stdout
