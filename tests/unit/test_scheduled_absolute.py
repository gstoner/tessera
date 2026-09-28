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


def test_absolute_exact_package_cache_reuses_native_image_and_refuses_drift(monkeypatch):
    from tessera.compiler import x86_native as x

    if not x.tools_available_for_architecture(x.X86_AVX512_ARCHITECTURE):
        pytest.skip('x86 native image toolchain required')
    artifact = lower_absolute(absolute_module())
    x._SCHEDULED_UNARY_PACKAGE_CACHE.clear()
    real_lower = x._lower
    calls = []

    def recorded_lower(*args, **kwargs):
        calls.append(1)
        return real_lower(*args, **kwargs)

    monkeypatch.setattr(x, '_lower', recorded_lower)
    first = package_absolute(artifact, pipeline_name='tessera-lower-to-x86')
    first.descriptor.provenance['caller_mutation'] = True
    repeated = package_absolute(artifact, pipeline_name='tessera-lower-to-x86')
    assert len(calls) == 1
    assert first.image.image_digest == repeated.image.image_digest
    assert 'caller_mutation' not in repeated.descriptor.provenance
    package_absolute(artifact, pipeline_name='tessera-x86-executable')
    assert len(calls) == 2
    with pytest.raises(ValueError, match='replay'):
        package_absolute(replace(artifact, tile_ir=artifact.tile_ir + '\n'),
                         pipeline_name='tessera-lower-to-x86')
    assert len(calls) == 2


def test_absolute_lower_cache_keys_exact_graph_and_compiler(monkeypatch):
    from tessera.compiler import scheduled_absolute as scheduled

    scheduled._LOWER_CACHE.clear()
    real_run = scheduled.run_tessera_opt
    calls = []

    def recorded_run(*args):
        calls.append(args[-1])
        return real_run(*args)

    monkeypatch.setattr(scheduled, 'run_tessera_opt', recorded_run)
    first = scheduled.lower_absolute(absolute_module())
    second = scheduled.lower_absolute(absolute_module())
    assert first == second and len(calls) == 2
    scheduled.lower_absolute(absolute_module((2, 19)))
    assert len(calls) == 4


def test_absolute_graph_packager_never_uses_legacy_constructor(monkeypatch):
    from tessera.compiler import x86_native
    monkeypatch.setattr('tests._support.x86_kernel_baseline.emit_elementwise_tile_ir',lambda **kw: pytest.fail('Graph constructor'))
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


def test_absolute_preserves_explicit_row_major_packaging(monkeypatch):
    from tessera.compiler import x86_native
    module = absolute_module()
    module.functions[0].args[0].layout = 'row_major'
    module.functions[0].args[0].dim_names = ('3', '17')
    artifact = lower_absolute(module)
    assert 'tessera.layout = "row_major"' in artifact.graph_ir
    assert artifact.project()[1] == (3, 17)
    monkeypatch.setattr('tests._support.x86_kernel_baseline.emit_elementwise_tile_ir', lambda **kw: pytest.fail('Graph constructor'))
    monkeypatch.setattr(x86_native, '_lower', lambda *a: ('target', b'image', 'compiler', 'toolchain'))
    packet = x86_native.package_elementwise(module, pipeline_name='tessera-lower-to-x86')
    assert packet.descriptor.provenance['numeric_policy'] == 'ieee_abs_clear_sign'


@pytest.mark.parametrize('layout', ['"col_major"', '42'])
def test_absolute_rejects_incompatible_or_malformed_argument_layout(layout):
    module = absolute_module()
    module.functions[0].args[0].layout = 'row_major'
    graph = lower_absolute(module).graph_ir.replace('tessera.layout = "row_major"', f'tessera.layout = {layout}')
    with pytest.raises(RuntimeError, match='row_major argument layout'):
        run_tessera_opt(find_tessera_opt(), graph, '--tessera-graph-to-schedule')


@pytest.mark.parametrize('family', ['floor', 'ceil', 'trunc'])
def test_rounding_unary_projects_and_replays_without_graph_constructor(family, monkeypatch):
    from tessera.compiler import x86_native
    from tessera.compiler.scheduled_absolute import lower_floor, lower_ceil, lower_trunc, package_unary
    module = absolute_module()
    module.functions[0].body[0].op_name = 'tessera.' + family
    module.functions[0].args[0].layout = 'row_major'
    artifact = {'floor': lower_floor, 'ceil': lower_ceil, 'trunc': lower_trunc}[family](module)
    assert artifact.project()[1] == (3,17)
    monkeypatch.setattr('tests._support.x86_kernel_baseline.emit_elementwise_tile_ir',lambda **kw: pytest.fail('Graph constructor'))
    monkeypatch.setattr(x86_native,'_lower',lambda *a: ('target',b'image','compiler','toolchain'))
    packet = x86_native.package_elementwise(module,pipeline_name='tessera-lower-to-x86')
    assert packet.descriptor.provenance['numeric_policy'] == 'ieee_' + family
    altered = artifact.tile_ir.replace('tile.elementwise_kernel %arg0, %arg1','tile.elementwise_kernel %arg1, %arg0')
    assert altered != artifact.tile_ir
    with pytest.raises(ValueError,match='replay'):
        package_unary(replace(artifact,tile_ir=altered),pipeline_name='tessera-lower-to-x86')


@pytest.mark.hardware_avx512
@pytest.mark.parametrize('shape', [(51,), (3, 17), (2, 3, 17)])
def test_trunc_replayed_image_executes_on_owning_cpu(shape):
    import math
    import numpy as np
    from tessera import runtime as rt
    from tessera.compiler import x86_native
    from tests._support.environment import avx512_is_plausibly_present

    if x86_native._library_path() is None or not avx512_is_plausibly_present():
        pytest.skip('owning AVX-512 runtime required')
    module = absolute_module(shape)
    module.functions[0].body[0].op_name = 'tessera.trunc'
    package = x86_native.package_elementwise(module, pipeline_name='tessera-lower-to-x86')
    bound = rt.RuntimeArtifact(metadata={'target': 'x86'}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=package.tile_ir, target_ir=package.target_ir)
    values = np.resize(np.array([0.0, -0.0, 1.9, -1.9, np.nextafter(np.float32(0), np.float32(1)),
        np.inf, -np.inf, np.nan], np.float32), math.prod(shape)).reshape(shape)
    out = np.empty_like(values)
    result = rt.launch(bound, {'x': values, 'o': out, 'N': values.size})
    assert result.get('ok') and result.get('execution_kind') == 'native_cpu', result
    expected = np.trunc(values)
    np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))
