"""Owning-device ordinary matmul/GEMM native package and warm binding proof."""
from __future__ import annotations

import os
import numpy as np
import pytest
import tessera


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_E2E_DEVICE_TEST") != "1"
    and os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="requires an explicit owning ROCm device proof lane",
)
@pytest.mark.parametrize("architecture", ["gfx1151", "gfx1201"])
@pytest.mark.parametrize("generic", [True, False])
@pytest.mark.parametrize("operation", ["matmul", "gemm"])
@pytest.mark.parametrize("lhs,rhs", [
    ("fp16", "fp16"), ("bf16", "bf16"),
    ("fp8_e4m3", "fp8_e4m3"), ("fp8_e4m3", "fp8_e5m2"),
    ("fp8_e5m2", "fp8_e4m3"), ("fp8_e5m2", "fp8_e5m2"),
])
def test_public_matmul_executes_owning_native_profile(architecture, generic, operation, lhs, rhs):
    from tessera import runtime

    if runtime._rocm_live_arch() != architecture:
        pytest.skip("requires the owning architecture")
    if architecture == "gfx1151" and lhs.startswith("fp8"):
        pytest.skip("RDNA3.5 has no FP8 WMMA")
    assert runtime._rocm_chip() == architecture
    ml_dtypes = pytest.importorskip("ml_dtypes")
    dtypes = {"fp16": np.float16, "bf16": ml_dtypes.bfloat16,
              "fp8_e4m3": ml_dtypes.float8_e4m3fn, "fp8_e5m2": ml_dtypes.float8_e5m2}
    target = "rocm" if generic else "rocm_" + architecture
    public = tessera.from_text(
        f'def product(a,b):\n    return tessera.ops.{operation}(a,b,output_dtype="fp32")\n',
        target=target,
    )
    compiled, retained = {}, []
    for iteration, shape in enumerate(((16, 32, 16), (17, 31, 19), (48, 33, 16), (16, 32, 16))):
        m, k, n = shape
        rng = np.random.default_rng(9481151 + iteration)
        a = rng.uniform(-0.5, 0.5, (m, k)).astype(dtypes[lhs])
        b = rng.uniform(-0.5, 0.5, (k, n)).astype(dtypes[rhs])
        original_a, original_b = a.copy(), b.copy()
        a.flags.writeable = b.flags.writeable = False
        output = public(a, b)
        np.testing.assert_allclose(output, a.astype(np.float32) @ b.astype(np.float32), rtol=1e-4, atol=1e-3)
        np.testing.assert_array_equal(a, original_a)
        np.testing.assert_array_equal(b, original_b)
        assert public.execution_kind == "native_gpu"
        assert public.compile_result.executable and public.compile_result.target == "rocm_" + architecture
        assert public._native_descriptor_last_receipt["compiler_path"] == "rocm_" + architecture + "_native_descriptor"
        bundle = public.compile_bundle
        assert "schedule.matmul" in bundle.schedule.text
        assert "tile.matmul_kernel" in bundle.tile.text
        assert "tessera_rocm.wmma_gemm" in bundle.target_ir.text
        if shape in compiled:
            assert public.compile_result is compiled[shape]
        compiled[shape] = public.compile_result
        retained.append((output, output.copy()))
    assert len(public._native_descriptor_specializations) == 3
    for actual, original in retained:
        np.testing.assert_array_equal(actual, original)
