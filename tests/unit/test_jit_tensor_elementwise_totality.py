"""Production JIT closure for tensor-valued pointwise operations.

The contract is mathematical rather than an allow-list: ranked tensor ops
carrying MLIR's elementwise trait are converted to the same-index-space
``linalg.generic`` form before one-shot bufferization. These execution tests
cover the arithmetic families emitted by AD and commonly written directly in
Graph IR, including dynamic extents and a tensor predicate.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import _jit_boundary as jb
from tests._support.compiler_tool import require_tessera_opt, run_tessera_opt


pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`",
)


def _binary_module(op: str, *, dynamic: bool = False) -> str:
    tensor = "tensor<?xf32>" if dynamic else "tensor<7xf32>"
    return f"""
module {{
  func.func @pointwise(%a: {tensor}, %b: {tensor}) -> {tensor} {{
    %r = arith.{op} %a, %b : {tensor}
    return %r : {tensor}
  }}
}}
"""


def _unary_math_module(op: str) -> str:
    return f"""
module {{
  func.func @pointwise_math(%x: tensor<7xf32>) -> tensor<7xf32> {{
    %r = math.{op} %x : tensor<7xf32>
    return %r : tensor<7xf32>
  }}
}}
"""


@pytest.mark.parametrize(
    ("op", "oracle"),
    [
        ("addf", np.add),
        ("subf", np.subtract),
        ("mulf", np.multiply),
        ("divf", np.divide),
        ("maximumf", np.maximum),
        ("minimumf", np.minimum),
        ("maxnumf", np.fmax),
        ("minnumf", np.fmin),
    ],
)
def test_tensor_binary_arithmetic_family_executes_natively(op, oracle):
    handle = jb.compile_module(_binary_module(op))
    a = np.array([-4.0, -1.5, -0.0, 0.25, 1.0, 3.0, 8.0], dtype=np.float32)
    b = np.array([2.0, -2.0, 0.5, 4.0, -0.5, 1.5, 0.25], dtype=np.float32)
    out = np.empty_like(a)
    before = jb.invocation_count()
    jb.invoke(handle, "pointwise", [a, b], out)
    assert jb.invocation_count() - before == 1
    np.testing.assert_allclose(out, oracle(a, b), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("op", "oracle"),
    [
        ("maximumf", np.maximum),
        ("minimumf", np.minimum),
        ("maxnumf", np.fmax),
        ("minnumf", np.fmin),
    ],
)
def test_tensor_minmax_preserves_nan_and_signed_zero_semantics(op, oracle):
    handle = jb.compile_module(_binary_module(op))
    a = np.array([np.nan, 1.0, 0.0, -0.0, -3.0, 7.0, 2.0], dtype=np.float32)
    b = np.array([2.0, np.nan, -0.0, 0.0, 4.0, -8.0, 2.0], dtype=np.float32)
    expected = oracle(a, b)
    out = np.empty_like(a)
    jb.invoke(handle, "pointwise", [a, b], out)

    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(out[finite], expected[finite])
    zero = finite & (expected == 0.0)
    np.testing.assert_array_equal(np.signbit(out[zero]), np.signbit(expected[zero]))


@pytest.mark.parametrize(
    ("op", "oracle"),
    [
        ("exp", np.exp),
        ("log", np.log),
        ("sqrt", np.sqrt),
        ("sin", np.sin),
        ("cos", np.cos),
        ("tanh", np.tanh),
    ],
)
def test_tensor_math_family_executes_natively(op, oracle):
    handle = jb.compile_module(_unary_math_module(op))
    x = np.array([0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0], dtype=np.float32)
    out = np.empty_like(x)
    jb.invoke(handle, "pointwise_math", [x], out)
    np.testing.assert_allclose(out, oracle(x), rtol=2e-6, atol=2e-7)


def test_dynamic_tensor_compare_select_and_negate_execute_natively():
    module = """
module {
  func.func @choose_negated(%a: tensor<?xf32>, %b: tensor<?xf32>)
      -> tensor<?xf32> {
    %predicate = arith.cmpf olt, %a, %b : tensor<?xf32>
    %negative = arith.negf %a : tensor<?xf32>
    %result = arith.select %predicate, %negative, %b
        : tensor<?xi1>, tensor<?xf32>
    return %result : tensor<?xf32>
  }
}
"""
    handle = jb.compile_module(module)
    a = np.array([-2.0, 4.0, 1.5, -0.25, 9.0], dtype=np.float32)
    b = np.array([1.0, 3.0, 2.0, -1.0, 9.0], dtype=np.float32)
    out = np.empty_like(a)
    before = jb.invocation_count()
    jb.invoke(handle, "choose_negated", [a, b], out)
    assert jb.invocation_count() - before == 1
    np.testing.assert_allclose(out, np.where(a < b, -a, b))


def test_dynamic_tensor_subtraction_executes_at_multiple_extents():
    handle = jb.compile_module(_binary_module("subf", dynamic=True))
    for size in (1, 5, 19):
        a = np.linspace(-2.0, 3.0, size, dtype=np.float32)
        b = np.linspace(4.0, -1.0, size, dtype=np.float32)
        out = np.empty_like(a)
        jb.invoke(handle, "pointwise", [a, b], out)
        np.testing.assert_allclose(out, a - b, rtol=1e-6, atol=1e-7)


def test_scalar_condition_select_over_dynamic_tensors_executes_natively():
    module = """
module {
  func.func @choose_by_extent(%a: tensor<?xf32>, %b: tensor<?xf32>)
      -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %extent = tensor.dim %a, %c0 : tensor<?xf32>
    %large = arith.cmpi sgt, %extent, %c4 : index
    %result = arith.select %large, %a, %b : tensor<?xf32>
    return %result : tensor<?xf32>
  }
}
"""
    handle = jb.compile_module(module)
    for size in (3, 7):
        a = np.arange(size, dtype=np.float32)
        b = -np.arange(size, dtype=np.float32)
        out = np.empty_like(a)
        jb.invoke(handle, "choose_by_extent", [a, b], out)
        np.testing.assert_array_equal(out, a if size > 4 else b)


def test_forward_ad_generated_tensor_arithmetic_executes_natively():
    source = """
module {
  func.func @affine_product(%x: tensor<7xf32>, %y: tensor<7xf32>)
      -> tensor<7xf32> attributes {tessera.autodiff = "forward"} {
    %product = arith.mulf %x, %y : tensor<7xf32>
    %result = arith.subf %product, %x : tensor<7xf32>
    return %result : tensor<7xf32>
  }
}
"""
    require_tessera_opt()
    transformed = run_tessera_opt(source, "--tessera-autodiff-forward")
    assert transformed.returncode == 0, transformed.stderr
    assert "@affine_product__jvp" in transformed.stdout
    # The forward rule deliberately emits tensor arith mul/add/sub. This is
    # the compiler-produced path whose JIT closure originally had no proof.
    assert "arith.mulf" in transformed.stdout
    assert "arith.addf" in transformed.stdout
    assert "arith.subf" in transformed.stdout

    handle = jb.compile_module(transformed.stdout)
    x = np.linspace(-2.0, 2.0, 7, dtype=np.float32)
    y = np.linspace(0.5, 3.5, 7, dtype=np.float32)
    dx = np.linspace(1.0, 2.0, 7, dtype=np.float32)
    dy = np.linspace(-0.75, 0.25, 7, dtype=np.float32)
    primal = np.empty_like(x)
    tangent = np.empty_like(x)
    before = jb.invocation_count()
    jb.invoke(handle, "affine_product__jvp", [x, y, dx, dy], [primal, tangent])
    assert jb.invocation_count() - before == 1
    np.testing.assert_allclose(primal, x * y - x, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        tangent, dx * y + x * dy - dx, rtol=1e-6, atol=1e-7
    )
