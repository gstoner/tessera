"""Shared semantic shape/type inference; backend execution remains independent."""
import pytest

from tessera.compiler.graph_ir import _shape_matmul_2d, tensor_ir_type
from tessera.compiler.legality import TensorContract, check_op_legality


@pytest.mark.parametrize("ta,tb", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("a_dtype,b_dtype", [("fp32", "fp16"), ("bf16", "fp32")])
def test_matmul_semantic_shapes_and_wide_result(ta, tb, a_dtype, b_dtype):
    a_shape = ("35", "17") if ta else ("17", "35")
    b_shape = ("19", "35") if tb else ("35", "19")
    args = [tensor_ir_type(a_shape, a_dtype), tensor_ir_type(b_shape, b_dtype)]
    attrs = {"transposeA": ta, "transposeB": tb}
    result = _shape_matmul_2d(args, attrs)
    assert result.shape == ("17", "19")
    assert result.dtype == "fp32"
    legality = check_op_legality("tessera.matmul", [
        TensorContract(shape=a_shape, dtype=a_dtype),
        TensorContract(shape=b_shape, dtype=b_dtype)], attrs=attrs)
    assert not any(d.rule in {"matmul_shape", "matmul_rank"} for d in legality.diagnostics)


@pytest.mark.parametrize("name", ["transposeA", "transposeB"])
@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_matmul_rejects_nonboolean_orientation(name, value):
    args = [tensor_ir_type(("17", "35"), "fp32"), tensor_ir_type(("35", "19"), "fp16")]
    with pytest.raises(ValueError, match="transpose flags must be boolean"):
        _shape_matmul_2d(args, {name: value})
    result = check_op_legality("tessera.matmul", [
        TensorContract(shape=("17", "35"), dtype="fp32"),
        TensorContract(shape=("35", "19"), dtype="fp16")], attrs={name: value})
    assert any(d.code == "LEGALITY_MATMUL_TRANSPOSE_TYPE" for d in result.diagnostics)
