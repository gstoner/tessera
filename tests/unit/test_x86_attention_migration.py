"""Direct attention must replay its native parent before consuming an image."""
from dataclasses import replace
import pytest
from tessera.compiler import x86_native
from tessera.compiler.scheduled_attention import lower_scheduled_attention
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.unit.test_x86_e2e_spine import _attention_module

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason='native compiler required')


@pytest.mark.parametrize('extended', [False, True])
def test_direct_attention_has_no_graph_constructor(monkeypatch, extended):
    def forbidden(*args, **kwargs): pytest.fail('legacy Graph constructor used')
    monkeypatch.setattr(x86_native, 'emit_attention_graph_ir', forbidden)
    monkeypatch.setattr(x86_native, 'emit_attention_tile_ir', forbidden)
    monkeypatch.setattr(x86_native, '_lower', lambda *a: ('target', b'image', 'compiler', 'toolchain'))
    package = x86_native.package_attention(_attention_module(extended=extended), pipeline_name='tessera-lower-to-x86')
    assert package.descriptor.provenance['route'] == 'canonical_scheduled_tile_consumer'
    assert package.descriptor.provenance['window'] == (3 if extended else -1)


@pytest.mark.parametrize('field,value', [('scale', 0.25), ('dims', (1,2,2,6,7,4,3)), ('causal', True)])
def test_forged_attention_descriptor_refuses_before_compilation(monkeypatch, field, value):
    artifact = lower_scheduled_attention(_attention_module(), target='x86')
    monkeypatch.setattr(x86_native, '_lower', lambda *a: pytest.fail('forgery reached compiler'))
    with pytest.raises(ValueError):
        x86_native.package_scheduled_attention(replace(artifact, **{field: value}), pipeline_name='tessera-lower-to-x86')


def test_forged_attention_tile_refuses_before_compilation(monkeypatch):
    artifact = lower_scheduled_attention(_attention_module(), target='x86')
    changed = artifact.tile_ir.replace('scale = 5.000000e-01', 'scale = 2.500000e-01')
    assert changed != artifact.tile_ir
    monkeypatch.setattr(x86_native, '_lower', lambda *a: pytest.fail('forgery reached compiler'))
    with pytest.raises(ValueError, match='replay'):
        x86_native.package_scheduled_attention(replace(artifact, tile_ir=changed), pipeline_name='tessera-lower-to-x86')
