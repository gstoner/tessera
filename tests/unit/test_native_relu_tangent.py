"""Native ReLU tangents are a Graph-IR proof, distinct from device packaging."""

import numpy as np
import ml_dtypes
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_autodiff_paired_cpu_oracle import _parse_functions


@pytest.fixture
def tool():
    value = find_tessera_opt()
    if value is None:
        pytest.skip("requires production MLIR compiler")
    return value


def graph(storage, shape):
    return f"""module {{
func.func @relu(%x: tensor<{shape}x{storage}>) -> tensor<{shape}x{storage}>
    attributes {{tessera.autodiff = "forward"}} {{
  %y = "tessera.relu"(%x) : (tensor<{shape}x{storage}>) -> tensor<{shape}x{storage}>
  return %y : tensor<{shape}x{storage}>
}}
}}"""


@pytest.mark.compiler_route
@pytest.mark.parametrize(
    "storage,dtype", [("f16", np.float16), ("bf16", ml_dtypes.bfloat16), ("f32", np.float32), ("f64", np.float64)]
)
@pytest.mark.parametrize("shape_type,shape", [("10", (10,)), ("2x5", (2, 5)), ("?x5", (3, 5))])
def test_native_relu_forward_emitted_ir_matches_independent_oracle(tool, storage, dtype, shape_type, shape):
    ir = run_tessera_opt(tool, graph(storage, shape_type), "--tessera-autodiff-forward")
    assert "tessera.compare_scalar" in ir and "tessera.masked_fill" in ir
    assert 'predicate = "gt"' in ir
    assert "custom_adjoint" not in ir and "custom_jvp" not in ir
    patterns = [-2.0, -0.0, 0.0, 1.5, np.inf, -np.inf, np.nan, 3.0, 0.0, -3.0]
    x = np.resize(np.asarray(patterns, dtype=dtype), shape)
    seed = np.resize(
        np.asarray([0.125, -0.25, 4.0, 1.0 + 2**-40, 0.75, 0.5, -0.125, -2.0 + 2**-42, 1.0, -1.0], dtype=dtype), shape
    )
    primal, tangent = _parse_functions(ir)["relu__jvp"].run([x, seed])
    expected_primal = np.maximum(x, 0)
    # NumPy does not classify ml_dtypes BF16 as np.floating for NaN equality.
    # Conversion is exact for all tested real storage values.
    np.testing.assert_array_equal(primal.astype(np.float64), expected_primal.astype(np.float64))
    assert primal.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(tangent, np.where(x > 0, seed, 0).astype(dtype))
    assert tangent.dtype == np.dtype(dtype)


@pytest.mark.compiler_route
def test_native_relu_directional_derivative_and_reverse_duality(tool):
    from tessera.compiler.autodiff_proof import (
        DERIVATIVE_PROOF_MATRIX,
        ProofKind,
        assert_directional_close,
        assert_forward_reverse_duality,
        central_directional_difference,
    )

    rng = np.random.default_rng(151077)
    x = rng.uniform(0.25, 2.0, (3, 8))
    x[:, ::2] *= -1
    direction = rng.normal(size=x.shape)
    cotangent = rng.normal(size=x.shape)
    source = graph("f64", "3x8")
    paired = run_tessera_opt(tool, source, "--tessera-autodiff-forward")
    _, tangent = _parse_functions(paired)["relu__jvp"].run([x, direction])
    reverse_source = source.replace("forward", "reverse")
    reverse = run_tessera_opt(tool, reverse_source, "--tessera-autodiff-paired")
    (input_cotangent,) = _parse_functions(reverse)["relu__bwd"].run([x, cotangent])
    finite = central_directional_difference(lambda value: np.maximum(value, 0), (x,), (direction,))
    assert_directional_close(tangent, finite, rtol=1e-10, atol=1e-11)
    assert_forward_reverse_duality(
        tangent, cotangent, (direction,), (input_cotangent,), rtol=1e-12, atol=1e-12
    )
    contract = DERIVATIVE_PROOF_MATRIX["relu"]
    assert contract.kind is ProofKind.DIRECTIONAL_AND_DUALITY
    assert "signed zeros" in contract.boundary_policy and "NaN" in contract.boundary_policy


@pytest.mark.compiler_route
@pytest.mark.parametrize("storage", ["i32", "f8E4M3FN", "complex<f32>"])
def test_relu_tangent_does_not_admit_new_storage_domains(tool, storage):
    with pytest.raises(RuntimeError):
        run_tessera_opt(tool, graph(storage, "10"), "--tessera-autodiff-forward")


@pytest.mark.compiler_route
def test_symbolic_relu_admission_does_not_widen_other_dynamic_programs(tool):
    source = graph("f32", "?x5")
    old = '%y = "tessera.relu"(%x)'
    new = '%n = "tessera.rmsnorm"(%x) {eps = 1.0E-5 : f64} : (tensor<?x5xf32>) -> tensor<?x5xf32>\n  %y = "tessera.relu"(%n)'
    assert old in source
    source = source.replace(old, new)
    with pytest.raises(RuntimeError, match="requires static ranked"):
        run_tessera_opt(tool, source, "--tessera-autodiff-forward")


@pytest.mark.compiler_route
@pytest.mark.parametrize("wrt", [(0,), (1,), (0, 1)])
def test_native_rmsnorm_relu_matmul_tangent_matches_directional_oracle(tool, wrt):
    """Evaluate native paired SSA; this does not claim a device AD package."""
    indices = ", ".join(str(index) + " : i64" for index in wrt)
    source = f"""module {{
  func.func @product(%a: tensor<3x8xf32>, %b: tensor<8x5xf32>) -> tensor<3x5xf32>
    attributes {{tessera.autodiff = "forward", tessera.autodiff.wrt_indices = [{indices}]}} {{
    %n = "tessera.rmsnorm"(%a) {{eps = 1.0E-5 : f64}} : (tensor<3x8xf32>) -> tensor<3x8xf32>
    %r = "tessera.relu"(%n) : (tensor<3x8xf32>) -> tensor<3x8xf32>
    %p = "tessera.matmul"(%r, %b) : (tensor<3x8xf32>, tensor<8x5xf32>) -> tensor<3x5xf32>
    return %p : tensor<3x5xf32>
  }}
}}"""
    ir = run_tessera_opt(tool, source, "--tessera-autodiff-forward")
    rng = np.random.default_rng(151061)
    # Stay away from the ReLU kink for the independent finite difference.
    a = rng.uniform(0.5, 1.5, (3, 8)).astype(np.float32)
    a[:, ::2] *= -1
    b = rng.uniform(-0.25, 0.25, (8, 5)).astype(np.float32)
    da = rng.uniform(-0.1, 0.1, a.shape).astype(np.float32) if 0 in wrt else np.zeros_like(a)
    db = rng.uniform(-0.1, 0.1, b.shape).astype(np.float32) if 1 in wrt else np.zeros_like(b)
    seeds = [(da, db)[index] for index in wrt]
    primal, tangent = _parse_functions(ir)["product__jvp"].run([a, b, *seeds])
    a64, b64, da64, db64 = (value.astype(np.float64) for value in (a, b, da, db))
    inverse = 1 / np.sqrt(np.mean(a64 * a64, axis=-1, keepdims=True) + 1e-5)
    norm = a64 * inverse
    dnorm = da64 * inverse - a64 * inverse**3 * np.mean(a64 * da64, axis=-1, keepdims=True)
    expected = np.maximum(norm, 0) @ b64
    expected_tangent = np.where(norm > 0, dnorm, 0) @ b64 + np.maximum(norm, 0) @ db64

    def oracle(x, y):
        return np.maximum(x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5), 0) @ y

    epsilon = 1e-4
    finite_difference = (
        oracle(a64 + epsilon * da64, b64 + epsilon * db64) - oracle(a64 - epsilon * da64, b64 - epsilon * db64)
    ) / (2 * epsilon)
    np.testing.assert_allclose(primal, expected, rtol=3e-6, atol=2e-7)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=3e-6, atol=2e-7)
    np.testing.assert_allclose(tangent, finite_difference, rtol=3e-6, atol=2e-7)


@pytest.mark.parametrize("target", ["apple_gpu", "x86", "rocm", "nvidia_sm120"])
def test_relu_graph_tangent_does_not_claim_native_target_execution(target):
    from tessera.compiler.autodiff_request import (
        build_request,
        resolve_differentiation_provenance,
        DifferentiationStatus,
    )

    def identity(x):
        return x

    request = build_request(identity, autodiff="forward", wrt=("x",))
    proof = resolve_differentiation_provenance(request, target=target, op_families=("relu",))
    assert proof.status is DifferentiationStatus.IR_TRANSFORMED
    assert not proof.native
    required = build_request(identity, autodiff="forward", wrt=("x",), native_required=True)
    proof = resolve_differentiation_provenance(required, target=target, op_families=("relu",))
    assert proof.status is DifferentiationStatus.UNSUPPORTED
    assert not proof.native
