"""Cohort scan migration: compiler replay and owning x86 execution."""
from dataclasses import replace
import math
import numpy as np
import pytest
from tessera.compiler.scheduled_absolute import lower_cumsum, package_cumsum
from tessera.compiler.scheduled_matmul import find_tessera_opt
from benchmarks.record_absolute_migration import module

pytestmark = pytest.mark.skipif(find_tessera_opt() is None,reason='native compiler required')


def graph(shape):
    result = module(shape)
    result.functions[0].body[0].op_name = 'tessera.cumsum'
    result.functions[0].body[0].kwargs = {'axis':-1}
    return result


def test_cumsum_cohort_route_bypasses_old_emitter_and_refuses_tampering(monkeypatch):
    from tessera.compiler import x86_native
    source = graph((3,17))
    artifact = lower_cumsum(source)
    monkeypatch.setattr(x86_native,'emit_cohort2_tile_ir',lambda **kw: pytest.fail('Graph constructor'))
    monkeypatch.setattr(x86_native,'_lower',lambda *a: ('target',b'image','compiler','toolchain'))
    package = x86_native.package_cohort2(source,pipeline_name='tessera-lower-to-x86')
    assert package.descriptor.provenance['inclusive'] is True
    changed = artifact.tile_ir.replace('tile.scan_kernel %arg0, %arg1','tile.scan_kernel %arg1, %arg0')
    assert changed != artifact.tile_ir
    with pytest.raises(ValueError,match='replay'):
        package_cumsum(replace(artifact,tile_ir=changed),pipeline_name='tessera-lower-to-x86')


@pytest.mark.parametrize('shape',[(51,),(3,17),(2,3,17)])
def test_cumsum_native_descriptor_numerical(shape):
    from tessera import runtime as rt
    from tessera.compiler.x86_native import _library_path
    if _library_path() is None:
        pytest.skip('owning x86 runtime required')
    package = package_cumsum(lower_cumsum(graph(shape)),pipeline_name='tessera-lower-to-x86')
    bound = rt.RuntimeArtifact(metadata={'target':'x86'},native_image=package.image,
        launch_descriptor=package.descriptor,tile_ir=package.tile_ir,target_ir=package.target_ir)
    for values in (np.resize(np.array([-2,1,.5,-.25],np.float32),math.prod(shape)).reshape(shape),
                   np.resize(np.array([0,-0.,np.inf,-np.inf,np.nan],np.float32),math.prod(shape)).reshape(shape)):
        out = np.empty_like(values)
        result = rt.launch(bound,dict(x=values,out=out,Rows=math.prod(shape[:-1]),Cols=shape[-1]))
        assert result.get('ok') and result.get('execution_kind') == 'native_cpu',result
        with np.errstate(invalid='ignore'):
            expected = np.cumsum(values,axis=-1)
        np.testing.assert_allclose(out,expected,rtol=0,atol=0,equal_nan=True)
