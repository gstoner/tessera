"""Compiler-generated binary-cross-entropy losses on gfx1151 — the ROCm mirror
of the x86_binary_loss lane (bce / asymmetric_bce). The
`tessera_rocm.binary_loss` directive expands (generate-rocm-binary-loss-kernel)
into a flat 2-operand per-element kernel; the runtime reduces.

Reachable via `compiler_path="rocm_binary_loss_compiled"`. Validated vs
tessera.losses on gfx1151. Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import losses


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_binary_loss_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["z", "t"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["z", "t"],
                 "kwargs": kwargs}],
    })


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("shape", [(64,), (8, 33)])
def test_bce_matches_reference(reduction, shape):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(5 + len(shape) + int(np.prod(shape)))
    z = (rng.standard_normal(shape) * 3).astype(np.float32)
    t = rng.integers(0, 2, size=shape).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.binary_cross_entropy_loss",
                              {"reduction": reduction}), (z, t))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_binary_loss_compiled"
    ref = np.asarray(losses.binary_cross_entropy_loss(z, t, reduction=reduction),
                     dtype=np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32), ref,
                               atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("pw,nw", [(1.0, 1.0), (2.0, 0.5)])
def test_asymmetric_bce_matches_reference(pw, nw):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(11 + int(pw * 10 + nw))
    z = (rng.standard_normal((6, 20)) * 4).astype(np.float32)
    t = rng.integers(0, 2, size=(6, 20)).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.asymmetric_bce",
                              {"pos_weight": pw, "neg_weight": nw,
                               "reduction": "mean"}), (z, t))
    assert res["ok"] is True, res.get("reason")
    ref = np.float32(losses.asymmetric_bce(z, t, pos_weight=pw, neg_weight=nw))
    np.testing.assert_allclose(np.float32(res["output"]), ref, atol=2e-5,
                               rtol=2e-5)


# ── GPU-free codegen gate ────────────────────────────────────────────────────
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

_OPT = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"


@pytest.mark.parametrize("kind", [0, 1])
def test_binary_loss_codegen_and_lowers(kind):
    if not _OPT.is_file():
        pytest.skip("build tessera-opt")
    d = ('module {\n  "tessera_rocm.binary_loss"() {name = "bl", dtype = "f32", '
         f'kind = {kind} : i64}} : () -> ()\n}}\n')
    ir = subprocess.run([str(_OPT), "-", "--generate-rocm-binary-loss-kernel"],
                        input=d, capture_output=True, text=True)
    assert ir.returncode == 0 and "gpu.func @bl" in ir.stdout, ir.stderr
    low = subprocess.run(
        [str(_OPT), "-",
         "--pass-pipeline=builtin.module(generate-rocm-binary-loss-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=d, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout
