"""Compiler-generated pointwise regression losses on gfx1151 — the ROCm mirror
of the x86_loss lane. The `tessera_rocm.pointwise_loss` directive expands (via
generate-rocm-pointwise-loss-kernel) into a flat per-element kernel computing
mse/mae/huber/smooth_l1/log_cosh; the runtime reduces (none/mean/sum). exp/log1p
lower through math->ROCDL.

Reachable through `runtime.launch()` via `compiler_path="rocm_loss_compiled"`.
Validated vs tessera.losses on gfx1151. Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import losses


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_loss_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["pred", "target"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o",
                 "operands": ["pred", "target"], "kwargs": kwargs}],
    })


_REF = {
    "tessera.mse_loss": (losses.mse_loss, {}),
    "tessera.mae_loss": (losses.mae_loss, {}),
    "tessera.huber_loss": (losses.huber_loss, {"delta": 1.0}),
    "tessera.smooth_l1_loss": (losses.smooth_l1_loss, {"beta": 1.0}),
    "tessera.log_cosh_loss": (losses.log_cosh_loss, {}),
}


@pytest.mark.parametrize("op_name", list(_REF))
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("shape", [(64,), (8, 33)])
def test_loss_matches_reference(op_name, reduction, shape):
    rt = _rocm_or_skip()
    ref_fn, params = _REF[op_name]
    rng = np.random.default_rng(5 + len(shape) + int(np.prod(shape)))
    pred = (rng.standard_normal(shape) * 2).astype(np.float32)
    target = (rng.standard_normal(shape) * 2).astype(np.float32)
    kwargs = dict(params)
    kwargs["reduction"] = reduction
    res = rt.launch(_artifact(rt, op_name, kwargs), (pred, target))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_loss_compiled"
    ref = np.asarray(ref_fn(pred, target, **{**params, "reduction": reduction}),
                     dtype=np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32), ref,
                               atol=2e-5, rtol=2e-5)


# ── GPU-free codegen gate (needs only tessera-opt) ───────────────────────────
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

_OPT = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"


@pytest.mark.parametrize("kind", [0, 1, 2, 3, 4])
def test_loss_codegen_and_lowers(kind):
    if not _OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    d = ('module {\n  "tessera_rocm.pointwise_loss"() {name = "pl", '
         f'dtype = "f32", kind = {kind} : i64, param = 1.0 : f32}} '
         ': () -> ()\n}\n')
    ir = subprocess.run([str(_OPT), "-", "--generate-rocm-pointwise-loss-kernel"],
                        input=d, capture_output=True, text=True)
    assert ir.returncode == 0, ir.stderr
    assert "gpu.func @pl" in ir.stdout
    low = subprocess.run(
        [str(_OPT), "-",
         "--pass-pipeline=builtin.module(generate-rocm-pointwise-loss-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=d, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout
