import time

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.stdlib import quant


def _artifact(op_name, arg_names):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_dequant_gemm_compiled",
        "executable": True,
        "arg_names": list(arg_names),
        "ops": [{"op_name": op_name}],
    })


def _median_ms(fn, reps=7):
    vals = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(vals))


def test_rocm_dequant_matmul_runtime_matches_packed_oracle_int4():
    rng = np.random.default_rng(31)
    x = rng.standard_normal((5, 16)).astype(np.float32)
    w = rng.standard_normal((16, 9)).astype(np.float32)
    packed = quant.quantize_weight(w, "int4", group_size=4)

    res = rt.launch(_artifact("tessera.dequant_matmul", ["x", "packed_w"]), (x, packed))

    assert res["ok"]
    assert res["compiler_path"] == "rocm_dequant_gemm_compiled"
    assert res["execution_kind"] in {"native_gpu", "reference_cpu"}
    np.testing.assert_allclose(
        res["output"],
        quant.dequant_matmul(x, packed, backend="reference"),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        res["output"],
        quant.reference_dequant_then_matmul(x, packed).astype(np.float32),
        rtol=2e-6,
        atol=2e-6,
    )


def test_rocm_dequant_grouped_gemm_runtime_matches_per_expert_oracle():
    rng = np.random.default_rng(32)
    group_sizes = np.array([2, 0, 3], dtype=np.int64)
    x = rng.standard_normal((int(group_sizes.sum()), 12)).astype(np.float32)
    experts = [
        quant.quantize_weight(rng.standard_normal((12, 7)).astype(np.float32),
                              "int8", group_size=3)
        for _ in range(3)
    ]
    art = _artifact(
        "tessera.dequant_grouped_gemm",
        ["x", "packed_experts", "group_sizes"],
    )

    res = rt.launch(art, {
        "x": x,
        "packed_experts": experts,
        "group_sizes": group_sizes,
    })

    assert res["ok"]
    assert res["execution_kind"] in {"native_gpu", "reference_cpu"}
    np.testing.assert_allclose(
        res["output"],
        quant.dequant_grouped_gemm(x, experts, group_sizes),
        rtol=2e-6,
        atol=2e-6,
    )


def test_dk4_rocm_dequant_gemm_perf_baseline_is_bounded():
    rng = np.random.default_rng(33)
    x = rng.standard_normal((16, 64)).astype(np.float32)
    w = rng.standard_normal((64, 32)).astype(np.float32)
    packed = quant.quantize_weight(w, "int4", group_size=8)
    art = _artifact("tessera.dequant_matmul", ["x", "packed_w"])

    direct_ms = _median_ms(lambda: quant.dequant_matmul(x, packed), reps=9)
    launch_ms = _median_ms(lambda: rt.launch(art, (x, packed)), reps=9)

    # Hardware-free runs fall back to the oracle; native ROCm runs still need
    # bounded launch overhead for small packed-GEMM shapes.
    assert launch_ms < max(75.0, direct_ms * 4.0)


def test_rocm_dequant_gemm_native_gpu_matches_reference_on_hardware():
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")

    rng = np.random.default_rng(34)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    w = rng.standard_normal((32, 11)).astype(np.float32)
    packed = quant.quantize_weight(w, "int4", group_size=8)

    res = rt.launch(_artifact("tessera.dequant_matmul", ["x", "packed_w"]), (x, packed))

    assert res["ok"], res.get("reason")
    assert res["execution_kind"] == "native_gpu"
    np.testing.assert_allclose(
        res["output"],
        quant.dequant_matmul(x, packed, backend="reference"),
        rtol=1e-5,
        atol=1e-5,
    )


def test_rocm_dequant_gemm_codegen_lowers():
    import subprocess
    from pathlib import Path

    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    directive = (
        'module {\n'
        '  "tessera_rocm.dequant_gemm"() {name = "dq"} : () -> ()\n'
        '}\n')
    low = subprocess.run(
        [str(opt), "-",
         "--pass-pipeline=builtin.module(generate-rocm-dequant-gemm-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=directive, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout
