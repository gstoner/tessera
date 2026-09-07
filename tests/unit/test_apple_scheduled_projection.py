"""Native parser/replay contracts, not Metal execution or timing evidence."""
import dataclasses
import re

import pytest

from tessera.compiler import apple_native, scheduled_kernel, scheduled_matmul
from tessera.compiler import scheduled_attention, scheduled_attention_backward
from test_scheduled_kernel_consumers import _module as unary_module
from test_scheduled_matmul_consumers import _module as matmul_module
from test_scheduled_attention_consumers import _apple_module as attention_module
from test_scheduled_attention_backward_consumers import _apple_module as backward_module

pytestmark = pytest.mark.skipif(scheduled_matmul.find_tessera_opt() is None,
                                reason='requires native tessera-opt')


@pytest.fixture(scope='module', params=['softmax', 'reduce'])
def unary(request):
    return scheduled_kernel.lower_scheduled_kernel(
        unary_module(family=request.param, target='x86'), target='apple_gpu')


def test_unary_projection_accepts_native_product(unary):
    apple_native._verify_kernel_schedule_ancestry(unary)


@pytest.mark.parametrize('field', ['rows', 'columns', 'outer', 'axis_extent', 'inner',
                                  'input_shape', 'output_shape', 'axis', 'keepdims',
                                  'function_name', 'kind', 'schedule', 'epsilon'])
def test_unary_descriptor_mutations_reject_before_runtime(unary, field, monkeypatch):
    old = getattr(unary, field)
    value = ((999,) if isinstance(old, tuple) else not old if type(old) is bool else
             old + 1 if type(old) in (int, float) else 'forged')
    forged = dataclasses.replace(unary, **{field: value})
    monkeypatch.setattr(apple_native, '_runtime_library_path',
                        lambda: pytest.fail('runtime lookup before descriptor rejection'))
    with pytest.raises(ValueError, match='disagrees with native IR'):
        apple_native.package_scheduled_kernel(forged, pipeline_name='test')


@pytest.mark.parametrize('family', ['matmul', 'attention', 'backward'])
def test_native_apple_family_replay_and_operand_tamper(family, monkeypatch):
    if family == 'matmul':
        artifact = scheduled_matmul.lower_scheduled_matmul(
            matmul_module(target='apple_gpu'), target='apple_gpu')
        package = apple_native.package_scheduled_matmul
    elif family == 'attention':
        artifact = scheduled_attention.lower_scheduled_attention(attention_module(), target='apple_gpu')
        package = apple_native.package_scheduled_attention
    else:
        artifact = scheduled_attention_backward.lower_scheduled_attention_backward(
            backward_module(), target='apple_gpu')
        package = apple_native.package_scheduled_attention_backward
    apple_native._verify_schedule_ancestry(artifact)
    op = 'attention_backward' if family == 'backward' else family
    changed, count = re.subn(r'(tile\.' + op + r'_kernel) (%\w+), (%\w+)',
                            r'\1 \3, \2', artifact.tile_ir)
    assert count == 1
    forged = dataclasses.replace(artifact, tile_ir=changed)
    forged.validate()
    monkeypatch.setattr(apple_native, '_runtime_library_path',
                        lambda: pytest.fail('runtime lookup before native replay rejection'))
    with pytest.raises(ValueError, match='native Schedule replay'):
        package(forged, pipeline_name='test')


@pytest.fixture(scope='module', params=['apple_gpu', 'apple_gpu_f16'])
def matmul(request):
    return scheduled_matmul.lower_scheduled_matmul(matmul_module(target=request.param), target='apple_gpu')


def test_matmul_projection_accepts_native_product(matmul):
    apple_native._verify_matmul_schedule_ancestry(matmul)


@pytest.mark.parametrize('field', ['m', 'n', 'k', 'function_name', 'bias_name', 'dynamic_m'])
def test_matmul_descriptor_mutation_rejects_before_runtime(matmul, field, monkeypatch):
    old = getattr(matmul, field)
    value = not old if type(old) is bool else old + 1 if type(old) is int else 'forged'
    monkeypatch.setattr(apple_native, '_runtime_library_path',
                        lambda: pytest.fail('runtime lookup before descriptor rejection'))
    with pytest.raises(ValueError, match='disagrees with native IR|dropped its bias epilogue'):
        apple_native.package_scheduled_matmul(dataclasses.replace(matmul, **{field: value}),
                                              pipeline_name='test')


@pytest.fixture(scope='module', params=['forward', 'backward'])
def attention(request):
    if request.param == 'forward':
        return scheduled_attention.lower_scheduled_attention(attention_module(), target='apple_gpu')
    return scheduled_attention_backward.lower_scheduled_attention_backward(backward_module(), target='apple_gpu')


def test_attention_projection_accepts_native_product(attention):
    apple_native._verify_attention_schedule_ancestry(attention)


@pytest.mark.parametrize('field', ['dims', 'scale', 'dropout_seed'])
def test_attention_projection_rejects_descriptor_mutation(attention, field, monkeypatch):
    old = getattr(attention, field)
    value = (old[0] + 1,) + old[1:] if isinstance(old, tuple) else old + 1 if type(old) in (int, float) else 'forged'
    monkeypatch.setattr(apple_native, '_runtime_library_path', lambda: pytest.fail('runtime before projection'))
    package = (apple_native.package_scheduled_attention_backward
               if isinstance(attention, scheduled_attention_backward.ScheduledAttentionBackwardArtifact)
               else apple_native.package_scheduled_attention)
    with pytest.raises(ValueError, match='disagrees with native IR'):
        package(dataclasses.replace(attention, **{field: value}), pipeline_name='test')


def test_legacy_f32_softmax_uses_native_artifact(tmp_path, monkeypatch):
    from test_apple_e2e_native_spine import _softmax_module
    module = _softmax_module()
    library = tmp_path / 'runtime.dylib'
    library.write_bytes(b'contract-test')
    monkeypatch.setattr(apple_native, '_runtime_library_path', lambda: library)
    package = apple_native.package_softmax(module, pipeline_name='tessera-lower-to-apple_gpu-runtime')
    assert 'tile.softmax_kernel' in package.tile_ir
    assert 'schedule_digest' in package.descriptor.provenance


def test_attention_backward_rounded_scale():
    artifact = scheduled_attention_backward.lower_scheduled_attention_backward(
        backward_module(scale=0.123456789, causal=True), target='apple_gpu')
    apple_native._verify_attention_schedule_ancestry(artifact)


def test_legacy_f32_softmax_cannot_fall_back_without_native_compiler(monkeypatch):
    from test_apple_e2e_native_spine import _softmax_module
    monkeypatch.setattr(scheduled_kernel, 'find_tessera_opt', lambda: None)
    monkeypatch.setattr(apple_native, '_runtime_library_path', lambda: pytest.fail('bootstrap fallback'))
    with pytest.raises(RuntimeError, match='production tessera-opt'):
        apple_native.package_softmax(_softmax_module(), pipeline_name='tessera-lower-to-apple_gpu-runtime')
