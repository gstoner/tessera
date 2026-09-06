from pathlib import Path
import pytest
from tessera.compiler.native_tape_products import export_native_tape_products
from tessera.compiler.scheduled_matmul import find_tessera_opt
ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('fixture,expected',[
    ('autodiff_saved_if_intermediates.mlir',('i1','tensor<4xf32>')),
    ('autodiff_saved_while_state_tape.mlir',('index','tensor<3x4xf32>'))])
def test_split_product_preserves_residual_types_and_native_regions(fixture,expected):
    if find_tessera_opt() is None:
        pytest.skip('requires native compiler')
    source=(ROOT/'tests/tessera-ir/phase_f4'/fixture).read_text()
    pair=export_native_tape_products(source)
    assert set(expected)<=set(pair.residual_types)
    assert 'scf.' in pair.forward_ir and 'scf.' in pair.backward_ir
    assert 'tessera.autodiff.paired = @' not in pair.forward_ir.split('\n  func.func',1)[1]
    assert len(pair.residual_types)==len(pair.residual_sources)
    assert pair.digest==export_native_tape_products(source).digest


def nested_source(inner=3,outer=2):
    return f'''module {{
  func.func @nested(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32>
    attributes {{tessera.autodiff = "reverse"}} {{
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %outer = arith.constant {outer} : index
    %inner = arith.constant {inner} : index
    %out = scf.for %i = %zero to %outer step %one iter_args(%state = %x) -> tensor<4xf32> {{
      %in = scf.for %j = %zero to %inner step %one iter_args(%carry = %state) -> tensor<4xf32> {{
        %next = "tessera.mul"(%carry,%w) : (tensor<4xf32>,tensor<4xf32>) -> tensor<4xf32>
        scf.yield %next : tensor<4xf32>
      }} {{tessera.autodiff.checkpoint_policy = "save", tessera.autodiff.checkpoint_indices = array<i64: 1, 2>}}
      scf.yield %in : tensor<4xf32>
    }} {{tessera.autodiff.checkpoint_policy = "save", tessera.autodiff.checkpoint_indices = array<i64: 1>}}
    return %out : tensor<4xf32>
  }}
}}'''


@pytest.mark.parametrize('inner,outer',[(3,2),(0,2),(3,0)])
def test_nested_and_zero_trip_products_keep_the_typed_boundary(inner,outer):
    if find_tessera_opt() is None:
        pytest.skip('requires native compiler')
    pair=export_native_tape_products(nested_source(inner,outer))
    if outer==0:
        assert pair.residual_types==()
    else:
        assert 'tensor<1x4xf32>' in pair.residual_types
    assert len(pair.digest)==64
