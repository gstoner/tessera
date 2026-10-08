"""Logical transpose dimensions agree between frontend typing and MLIR."""
from itertools import product
import subprocess

import pytest

from tessera.compiler.graph_ir import _infer_result_types, tensor_ir_type
from tessera.compiler.scheduled_matmul import find_tessera_opt


@pytest.mark.parametrize("transpose_a,transpose_b", list(product((False, True), repeat=2)))
@pytest.mark.parametrize("m,k,n", [(3, 7, 5), ("?", 7, 5), (3, "?", 5)])
def test_scaled_shape_preserves_logical_free_axes(transpose_a, transpose_b, m, k, n):
    a = (k, m) if transpose_a else (m, k)
    b = (n, k) if transpose_b else (k, n)
    operands = [tensor_ir_type(a, "fp16"), tensor_ir_type(b, "fp16"),
                tensor_ir_type((1,), "fp32"), tensor_ir_type((1,), "fp32")]
    result = _infer_result_types("tessera.scaled_matmul", operands,
                                {"transposeA": transpose_a, "transposeB": transpose_b})[0]
    assert str(result) == f"tensor<{m}x{n}xf32>"


@pytest.mark.parametrize("attribute", ["transposeA", "transposeB"])
def test_scaled_shape_rejects_nonboolean_transpose(attribute):
    operands = [tensor_ir_type((3, 7), "fp16"), tensor_ir_type((7, 5), "fp16"),
                tensor_ir_type((1,), "fp32"), tensor_ir_type((1,), "fp32")]
    with pytest.raises(ValueError, match="must be boolean"):
        _infer_result_types("tessera.scaled_matmul", operands, {attribute: "false"})


@pytest.mark.parametrize("transpose_a,transpose_b", list(product((False, True), repeat=2)))
@pytest.mark.parametrize("result_shape,error", [("3x5", None), ("4x5", "result M"), ("3x6", "result N")])
def test_mlir_scaled_result_checks_free_axes_before_optional_scale_exit(
        transpose_a, transpose_b, result_shape, error):
    compiler = find_tessera_opt()
    if compiler is None:
        pytest.skip("requires matching tessera-opt")
    a = "7x3" if transpose_a else "3x7"
    b = "5x7" if transpose_b else "7x5"
    attrs = f"transposeA = {str(transpose_a).lower()}, transposeB = {str(transpose_b).lower()}"
    source = f"""module {{
      func.func @scaled(%a: tensor<{a}xf16>, %b: tensor<{b}xf16>,
                        %sa: tensor<1xf32>, %sb: tensor<1xf32>) -> tensor<{result_shape}xf32> {{
        %d = tessera.scaled_matmul %a, %b scales (%sa, %sb) {{{attrs}}}
          : (tensor<{a}xf16>, tensor<{b}xf16>, tensor<1xf32>, tensor<1xf32>) -> tensor<{result_shape}xf32>
        return %d : tensor<{result_shape}xf32>
      }}
    }}"""
    process = subprocess.run([str(compiler)], input=source, capture_output=True, text=True)
    if error is None:
        assert process.returncode == 0, process.stderr
    else:
        assert process.returncode != 0
        assert error in process.stderr


def test_sm120_scaled_capability_admits_normalized_nvfp4_without_sibling_promotion():
    from tessera.compiler.capabilities import supports_op
    assert supports_op("nvidia_sm120", "tessera.scaled_matmul", dtype="nvfp4", rank=2).supported
    assert supports_op("nvidia_sm120", "tessera.scaled_matmul", dtype="nvfp4", rank=4).supported
    for target in ("nvidia_sm90", "nvidia_sm100", "apple_gpu", "x86", "rocm_gfx1151"):
        assert not supports_op(target, "tessera.scaled_matmul", dtype="nvfp4", rank=2).supported


@pytest.mark.parametrize("m,n,k", [(16, 8, 64), (33, 19, 129), (7, 5, 31)])
def test_logical_nvfp4_scaled_shape_keeps_unpacked_graph_dimensions(m, n, k):
    operands = [tensor_ir_type((m, k), "nvfp4"), tensor_ir_type((k, n), "nvfp4"),
                tensor_ir_type((m, (k + 15) // 16), "uint8"),
                tensor_ir_type(((k + 15) // 16, n), "uint8")]
    attrs = {"physical_contract": "nvidia_sm120_nvfp4_blockscale_v1"}
    assert str(_infer_result_types("tessera.scaled_matmul", operands, attrs)[0]) == f"tensor<{m}x{n}xf32>"
    with pytest.raises(ValueError, match="logical K differs"):
        _infer_result_types("tessera.scaled_matmul", operands, {**attrs, "transposeB": True})


@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
def test_logical_nvfp4_transpose_infers_free_dimensions(ta, tb):
    m, n, k, sk = 7, 5, 31, 2
    a_shape = (k, m) if ta else (m, k)
    b_shape = (n, k) if tb else (k, n)
    sa_shape = (sk, m) if ta else (m, sk)
    sb_shape = (n, sk) if tb else (sk, n)
    operands = [tensor_ir_type(a_shape, "nvfp4"), tensor_ir_type(b_shape, "nvfp4"),
                tensor_ir_type(sa_shape, "uint8"), tensor_ir_type(sb_shape, "uint8")]
    attrs = {"physical_contract": "nvidia_sm120_nvfp4_blockscale_v1",
             "transposeA": ta, "transposeB": tb}
    result = _infer_result_types("tessera.scaled_matmul", operands, attrs)[0]
    assert result.shape == (str(m), str(n))
