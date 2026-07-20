from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.x86_native import (
    X86_MATMUL_BF16_F32_ABI,
    X86_MATMUL_F64_ABI,
    X86_MATMUL_U8S8_S32_ABI,
    package_matmul,
    supports_matmul,
    tools_available,
)


def _module(a_dtype: str, b_dtype: str, out_dtype: str, shape=(5, 7, 9)) -> GraphIRModule:
    m, n, k = shape
    spelling = {"bf16": "bf16", "fp32": "f32", "fp64": "f64", "uint8": "i8", "int8": "i8", "int32": "i32"}
    a = IRType(f"tensor<{m}x{k}x{spelling[a_dtype]}>", (str(m), str(k)), a_dtype)
    b = IRType(f"tensor<{k}x{n}x{spelling[b_dtype]}>", (str(k), str(n)), b_dtype)
    o = IRType(f"tensor<{m}x{n}x{spelling[out_dtype]}>", (str(m), str(n)), out_dtype)
    return GraphIRModule(functions=[GraphIRFunction(
        name="x86_dtype_matmul", args=[IRArg("a", a), IRArg("b", b)], result_types=[o],
        body=[IROp(result="o", op_name="tessera.matmul", operands=["%a", "%b"],
                   operand_types=[str(a), str(b)], result_type=str(o), kwargs={})],
        return_values=["%o"],
    )])


@pytest.mark.parametrize(
    "dtypes,abi",
    [(("bf16", "bf16", "fp32"), X86_MATMUL_BF16_F32_ABI),
     (("uint8", "int8", "int32"), X86_MATMUL_U8S8_S32_ABI),
     (("fp64", "fp64", "fp64"), X86_MATMUL_F64_ABI)],
)
def test_dtype_matmul_packages_have_distinct_abis(monkeypatch, dtypes, abi) -> None:
    monkeypatch.setattr(
        "tessera.compiler.x86_native._lower",
        lambda tile_ir, symbol: (
            f"module {{ func.call @{symbol}() : () -> () }}",
            b"\x7fELF-x86-e2e2-dtype",
            "compiler",
            "toolchain",
        ),
    )
    module = _module(*dtypes)
    assert supports_matmul(module)
    package = package_matmul(module, pipeline_name="tessera-lower-to-x86")
    assert package.descriptor.abi_id == abi
    assert package.descriptor.provenance["work_item"] == "X86-E2E-2"
    assert f"call @{package.descriptor.entry_symbol}" in package.target_ir


def _launch(module: GraphIRModule, a: np.ndarray, b: np.ndarray, output: np.ndarray) -> np.ndarray:
    package = package_matmul(module, pipeline_name="tessera-lower-to-x86")
    m, n, k = package.descriptor.provenance["shape"]
    values = {"a": a, "b": b, "o": output, "M": m, "N": n, "K": k}
    result = rt.launch(rt.RuntimeArtifact(
        metadata={"target": "x86"}, tile_ir=package.tile_ir, target_ir=package.target_ir,
        native_image=package.image, launch_descriptor=package.descriptor,
    ), values)
    assert result["ok"] is True, result.get("reason")
    return output


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
def test_bf16_vnni_and_fp64_descriptors_execute_on_exact_host() -> None:
    rng = np.random.default_rng(8641)
    ml = pytest.importorskip("ml_dtypes")
    shape = (5, 7, 9)
    m, n, k = shape

    af = rng.uniform(-1, 1, (m, k)).astype(np.float32)
    bf = rng.uniform(-1, 1, (k, n)).astype(np.float32)
    ab, bb = af.astype(ml.bfloat16), bf.astype(ml.bfloat16)
    got_bf16 = _launch(_module("bf16", "bf16", "fp32", shape), ab, bb, np.zeros((m, n), np.float32))
    np.testing.assert_allclose(got_bf16, ab.astype(np.float32) @ bb.astype(np.float32), rtol=2e-5, atol=2e-5)

    au = rng.integers(0, 32, (m, k), dtype=np.uint8)
    bs = rng.integers(-16, 16, (k, n), dtype=np.int8)
    got_i8 = _launch(_module("uint8", "int8", "int32", shape), au, bs, np.zeros((m, n), np.int32))
    np.testing.assert_array_equal(got_i8, au.astype(np.int32) @ bs.astype(np.int32))

    ad = rng.uniform(-1, 1, (m, k)).astype(np.float64)
    bd = rng.uniform(-1, 1, (k, n)).astype(np.float64)
    got_f64 = _launch(_module("fp64", "fp64", "fp64", shape), ad, bd, np.zeros((m, n), np.float64))
    np.testing.assert_allclose(got_f64, ad @ bd, rtol=1e-13, atol=1e-13)
