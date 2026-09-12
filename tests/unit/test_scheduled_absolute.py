from dataclasses import replace
import pytest
from tests.unit.test_x86_e2e_spine import _softmax_module
from tessera.compiler.scheduled_absolute import lower_absolute, package_absolute
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason='native compiler required')


def absolute_module(shape=(3,17)):
    module = _softmax_module(shape)
    module.functions[0].body[0].op_name = 'tessera.absolute'
    module.functions[0].body[0].kwargs = {}
    return module


def test_absolute_projects_serialized_shape_and_rejects_tile_operand_swap(monkeypatch):
    from tessera.compiler import x86_native
    artifact = lower_absolute(absolute_module())
    assert artifact.project()[:2] == (['x','o'],(3,17))
    monkeypatch.setattr(x86_native,'_lower',lambda *a: pytest.fail('tampering reached compilation'))
    altered = artifact.tile_ir.replace('tile.elementwise_kernel %arg0, %arg1', 'tile.elementwise_kernel %arg1, %arg0')
    assert altered != artifact.tile_ir
    with pytest.raises(ValueError,match='replay'):
        package_absolute(replace(artifact,tile_ir=altered),pipeline_name='tessera-lower-to-x86')


def test_absolute_graph_packager_never_uses_legacy_constructor(monkeypatch):
    from tessera.compiler import x86_native
    monkeypatch.setattr(x86_native,'emit_elementwise_tile_ir',lambda **kw: pytest.fail('Graph constructor'))
    monkeypatch.setattr(x86_native,'_lower',lambda *a: ('target',b'image','compiler','toolchain'))
    packet = x86_native.package_elementwise(absolute_module(),pipeline_name='tessera-lower-to-x86')
    assert packet.descriptor.provenance['numeric_policy'] == 'ieee_abs_clear_sign'


def test_absolute_schedule_rejects_changed_shape_record():
    artifact = lower_absolute(absolute_module())
    changed = artifact.schedule_ir.replace('shape = array<i64: 3, 17>', 'shape = array<i64: 3, 18>')
    assert changed != artifact.schedule_ir
    with pytest.raises(RuntimeError,match='altered'):
        run_tessera_opt(find_tessera_opt(),changed,'--tessera-schedule-to-tile')


def test_absolute_accepts_frontend_dimension_names():
    module = absolute_module()
    module.functions[0].args[0].dim_names = ('3','17')
    assert lower_absolute(module).project()[1] == (3,17)
