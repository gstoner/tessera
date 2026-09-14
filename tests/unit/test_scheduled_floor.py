"""Floor migration must replay IR and bypass the historical Tile emitter."""
from dataclasses import replace

import pytest

from tessera.compiler.scheduled_absolute import lower_floor, package_unary
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_scheduled_absolute import absolute_module

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason='native compiler required')


def floor_module():
    module = absolute_module()
    module.functions[0].body[0].op_name = 'tessera.floor'
    return module


def test_floor_uses_serialized_policy_and_never_legacy_emitter(monkeypatch):
    from tessera.compiler import x86_native
    monkeypatch.setattr(x86_native,'emit_elementwise_tile_ir',lambda **kw: pytest.fail('Graph constructor'))
    monkeypatch.setattr(x86_native,'_lower',lambda *a: ('target',b'image','compiler','toolchain'))
    package = x86_native.package_elementwise(floor_module(),pipeline_name='tessera-lower-to-x86')
    assert package.descriptor.provenance['kind'] == 'floor'
    assert package.descriptor.provenance['numeric_policy'] == 'ieee_floor'
    assert 'tessera.floor_contract' in package.tile_ir


def test_floor_rejects_operand_swap_before_compilation(monkeypatch):
    from tessera.compiler import x86_native
    artifact = lower_floor(floor_module())
    assert artifact.project()[:2] == (['x','o'],(3,17))
    altered = artifact.tile_ir.replace('tile.elementwise_kernel %arg0, %arg1','tile.elementwise_kernel %arg1, %arg0')
    assert altered != artifact.tile_ir
    monkeypatch.setattr(x86_native,'_lower',lambda *a: pytest.fail('corrupt artifact compiled'))
    with pytest.raises(ValueError,match='replay'):
        package_unary(replace(artifact,tile_ir=altered),pipeline_name='tessera-lower-to-x86')


def test_floor_rejects_wrong_family_and_numeric_policy():
    artifact = lower_floor(floor_module())
    for altered in (artifact.schedule_ir.replace('family=floor','family=absolute'),
                    artifact.schedule_ir.replace('ieee_floor','ieee_abs_clear_sign')):
        assert altered != artifact.schedule_ir
        with pytest.raises(RuntimeError,match='altered'):
            run_tessera_opt(find_tessera_opt(),altered,'--tessera-schedule-to-tile')


def test_floor_graph_verifier_rejects_shape_changing_result():
    graph = '''module {
      func.func @bad(%x: tensor<4xf32>) -> tensor<5xf32> {
        %o = tessera.floor %x : (tensor<4xf32>) -> tensor<5xf32>
        return %o : tensor<5xf32>
      }
    }'''
    with pytest.raises(RuntimeError,match='unchanged tensor'):
        run_tessera_opt(find_tessera_opt(),graph,'--tessera-graph-to-schedule')
