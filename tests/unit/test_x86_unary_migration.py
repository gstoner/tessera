"""Native parent authority for x86 direct unary packages."""
from dataclasses import replace
import pytest
from tessera.compiler import x86_native, scheduled_kernel
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.unit.test_scheduled_kernel_consumers import _module

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason='native compiler required')


@pytest.mark.parametrize('family', ['softmax', 'reduce'])
def test_direct_unary_uses_native_schedule_not_graph_constructor(monkeypatch, family):
    module = _module(family=family, target='x86')
    def forbidden(*args, **kwargs):
        pytest.fail('legacy Graph-to-Tile constructor called')
    monkeypatch.setattr(x86_native, 'emit_softmax_tile_ir', forbidden)
    monkeypatch.setattr(x86_native, 'emit_reduce_tile_ir', forbidden)
    monkeypatch.setattr(x86_native, '_lower', lambda tile, symbol, family: ('target', b'image', 'compiler', 'toolchain'))
    call = x86_native.package_softmax if family == 'softmax' else x86_native.package_reduction
    package = call(module, pipeline_name='tessera-lower-to-x86')
    assert package.descriptor.provenance['route'] == 'canonical_scheduled_tile_consumer'
    assert package.descriptor.provenance['shape'] == [2, 3, 5]


@pytest.mark.parametrize('field,value', [('rows', 5), ('input_shape', (3, 2, 5)), ('workgroup_size', True), ('kind', 'max')])
def test_descriptor_forgery_refuses_before_native_compile(monkeypatch, field, value):
    artifact = scheduled_kernel.lower_scheduled_kernel(_module(family='softmax', target='x86'), target='x86')
    monkeypatch.setattr(x86_native, '_lower', lambda *args: pytest.fail('forgery reached compilation'))
    with pytest.raises(ValueError):
        x86_native.package_scheduled_kernel(replace(artifact, **{field: value}), pipeline_name='test')


def test_tile_dataflow_forgery_refuses_before_compile(monkeypatch):
    artifact = scheduled_kernel.lower_scheduled_kernel(_module(family='softmax', target='x86'), target='x86')
    import re
    changed = re.sub(r'(tile.softmax_kernel )(%\w+), (%\w+)', r'\1\3, \2', artifact.tile_ir)
    assert changed != artifact.tile_ir
    monkeypatch.setattr(x86_native, '_lower', lambda *args: pytest.fail('forgery reached compilation'))
    with pytest.raises(ValueError, match='replay'):
        x86_native.package_scheduled_kernel(replace(artifact, tile_ir=changed), pipeline_name='test')


def test_relabelled_foreign_native_parent_refuses(monkeypatch):
    artifact = scheduled_kernel.lower_scheduled_kernel(_module(family='softmax', target='x86'), target='apple_gpu')
    forged = replace(artifact, target='x86', architecture='zen5-avx512')
    monkeypatch.setattr(x86_native, '_lower', lambda *args: pytest.fail('foreign parent reached compilation'))
    with pytest.raises(ValueError, match='target'):
        x86_native.package_scheduled_kernel(forged, pipeline_name='test')


@pytest.mark.skipif(not x86_native.tools_available(), reason='x86 native image toolchain required')
@pytest.mark.parametrize('family', ['softmax', 'reduce'])
def test_direct_migrated_package_executes(family):
    import numpy as np
    from tessera import runtime as rt
    module = _module(family=family, target='x86')
    call = x86_native.package_softmax if family == 'softmax' else x86_native.package_reduction
    package = call(module, pipeline_name='tessera-lower-to-x86')
    x = np.arange(30, dtype=np.float32).reshape(2, 3, 5) / 17
    output = np.zeros_like(x) if family == 'softmax' else np.zeros((2, 3), np.float32)
    args = {'x': x, 'o': output}
    if family == 'softmax':
        args.update(Rows=6, K=5)
        exp = np.exp(x - x.max(axis=-1, keepdims=True))
        expected = exp / exp.sum(axis=-1, keepdims=True)
    else:
        args.update(Outer=6, AxisExtent=5, Inner=1)
        expected = x.mean(axis=-1)
    artifact = rt.RuntimeArtifact(metadata={'target': 'x86'}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=package.tile_ir, target_ir=package.target_ir)
    result = rt.launch(artifact, args)
    assert result['ok'], result
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('family', ['softmax', 'reduce'])
def test_direct_entry_refuses_wrong_family_before_lowering(monkeypatch, family):
    module = _module(family=family, target='x86')
    monkeypatch.setattr(scheduled_kernel, 'lower_scheduled_kernel', lambda *args, **kwargs: pytest.fail('wrong family lowered'))
    wrong_call = x86_native.package_reduction if family == 'softmax' else x86_native.package_softmax
    with pytest.raises(ValueError):
        wrong_call(module, pipeline_name='tessera-lower-to-x86')


@pytest.mark.parametrize('kind', ['sum', 'mean', 'max'])
@pytest.mark.skipif(not x86_native.tools_available(), reason='x86 native image toolchain required')
def test_keepdims_reduction_executes_native_parent(kind, monkeypatch):
    import numpy as np
    from tessera import runtime as rt
    from tessera.compiler.graph_ir import IRType
    module = _module(family='reduce', target='x86')
    fn = module.functions[0]
    output_type = IRType('tensor<2x3x1xf32>', ('2', '3', '1'), 'fp32')
    fn.result_types = [output_type]
    fn.body[0].result_type = str(output_type)
    fn.body[0].kwargs['keepdims'] = True
    fn.body[0].op_name = 'tessera.' + kind
    monkeypatch.setattr(x86_native, 'emit_reduce_tile_ir', lambda *a, **kw: pytest.fail('Graph constructor used'))
    artifact = scheduled_kernel.lower_scheduled_kernel(module, target='x86')
    assert artifact.keepdims and artifact.output_shape == (2, 3, 1)
    with pytest.raises(ValueError):
        x86_native.package_scheduled_kernel(replace(artifact, keepdims=False), pipeline_name='test')
    package = x86_native.package_reduction(module, pipeline_name='tessera-lower-to-x86')
    x = np.arange(30, dtype=np.float32).reshape(2, 3, 5) / 17
    output = np.zeros((2, 3, 1), np.float32)
    runtime = rt.RuntimeArtifact(metadata={'target': 'x86'}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=package.tile_ir, target_ir=package.target_ir)
    result = rt.launch(runtime, dict(x=x, o=output, Outer=6, AxisExtent=5, Inner=1))
    assert result['ok'], result
    np.testing.assert_allclose(output, getattr(np, kind)(x, axis=-1, keepdims=True), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('family',['softmax','reduce'])
@pytest.mark.skipif(not x86_native.tools_available_for_architecture(x86_native.X86_BASE_ARCHITECTURE), reason='baseline x86 toolchain required')
def test_baseline_unary_uses_its_native_parent(family,monkeypatch):
    module=_module(family=family,target='x86')
    monkeypatch.setattr(x86_native,'emit_softmax_tile_ir',lambda **kw:pytest.fail('Graph constructor used'))
    monkeypatch.setattr(x86_native,'emit_reduce_tile_ir',lambda **kw:pytest.fail('Graph constructor used'))
    call=x86_native.package_softmax if family=='softmax' else x86_native.package_reduction
    package=call(module,pipeline_name='tessera-lower-to-x86',architecture=x86_native.X86_BASE_ARCHITECTURE)
    assert package.descriptor.entry_symbol.startswith('tessera_x86_base_')
    assert package.descriptor.provenance['required_features']==[]
    assert 'tessera.arch = "x86_64_base"' in package.tile_ir
    import numpy as np
    from tessera import runtime as rt
    x=np.arange(30,dtype=np.float32).reshape(2,3,5)/17
    output=np.zeros_like(x) if family=='softmax' else np.zeros((2,3),np.float32)
    args=dict(x=x,o=output)
    if family=='softmax':
        args.update(Rows=6,K=5)
        exp=np.exp(x-x.max(axis=-1,keepdims=True));expected=exp/exp.sum(axis=-1,keepdims=True)
    else:
        args.update(Outer=6,AxisExtent=5,Inner=1);expected=x.mean(axis=-1)
    artifact=rt.RuntimeArtifact(metadata={'target':'x86'},native_image=package.image,
        launch_descriptor=package.descriptor,tile_ir=package.tile_ir,target_ir=package.target_ir)
    assert rt.launch(artifact,args)['ok']
    np.testing.assert_allclose(output,expected,rtol=2e-5,atol=2e-5)


def test_direct_f32_matmul_replays_native_schedule(monkeypatch):
    from tests.unit.test_x86_e2e_spine import _matmul_module
    monkeypatch.setattr(x86_native,'emit_matmul_tile_ir',lambda **kw: pytest.fail('legacy matmul constructor called'))
    monkeypatch.setattr(x86_native,'_lower',lambda *a: ('target',b'image','compiler','toolchain'))
    package=x86_native.package_matmul(_matmul_module(),pipeline_name='tessera-lower-to-x86')
    assert package.descriptor.provenance['route']=='canonical_scheduled_tile_consumer'


def test_x86_matmul_native_projection_rejects_forged_shape(monkeypatch):
    from tests.unit.test_x86_e2e_spine import _matmul_module
    from tessera.compiler.scheduled_matmul import lower_scheduled_matmul
    artifact=lower_scheduled_matmul(_matmul_module(),target='x86')
    monkeypatch.setattr(x86_native,'_lower',lambda *a: pytest.fail('forgery reached compilation'))
    with pytest.raises(ValueError):
        x86_native.package_scheduled_matmul(replace(artifact,m=artifact.m+1),pipeline_name='test')
