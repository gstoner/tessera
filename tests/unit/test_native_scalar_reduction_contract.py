from tessera.compiler.graph_ir import tensor_ir_type, IRType


def test_scalar_tensor_is_rank_zero_not_unranked():
    scalar = tensor_ir_type((), 'fp32')
    assert str(scalar) == 'tensor<f32>'
    assert scalar.rank == 0
    assert tensor_ir_type(('*',), 'fp32').rank is None
    assert IRType('f32').rank is None


def test_traced_reduce_uses_ods_kind_and_scalar_result():
    import numpy as np
    import tessera as ts
    @ts.jit(autodiff='forward')
    def total(x):
        return ts.ops.reduce(x, op='sum', axis=0)
    text = total._specialized_autodiff_module((np.ones(32, np.float32),), {}).to_mlir()
    assert 'kind = "sum"' in text
    assert 'tensor<f32>' in text
    assert 'op = "sum"' not in text
