"""Native low-precision package contracts and owning-Metal differential proof."""
import sys

import numpy as np
import pytest

from tessera.compiler import apple_native, scheduled_attention_backward, scheduled_kernel
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import find_tessera_opt
from test_scheduled_attention_backward_consumers import _apple_module, _apple_backward_reference

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason='native compiler required')


def softmax_module(dtype):
    element = {'fp16': 'f16', 'bf16': 'bf16'}[dtype]
    ty = IRType(f'tensor<3x17x{element}>', ('3', '17'), dtype)
    return GraphIRModule(functions=[GraphIRFunction(
        name='softmax_lowp', args=[IRArg('x', ty)], result_types=[ty],
        body=[IROp(result='out', op_name='tessera.softmax', operands=['%x'],
                   operand_types=[str(ty)], result_type=str(ty), kwargs={'axis': -1})],
        return_values=['%out'])])


@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_lowp_native_projection(dtype):
    unary = scheduled_kernel.lower_scheduled_kernel(softmax_module(dtype), target='apple_gpu')
    apple_native._verify_kernel_schedule_ancestry(unary)
    backward = scheduled_attention_backward.lower_scheduled_attention_backward(
        _apple_module(dtype=dtype, causal=True, scale=0.123456789), target='apple_gpu')
    apple_native._verify_attention_schedule_ancestry(backward)


@pytest.mark.hardware_apple_gpu
@pytest.mark.skipif(sys.platform != 'darwin', reason='requires owning Metal host')
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_lowp_metal_differential(dtype):
    from tessera.runtime import RuntimeArtifact, launch
    from ml_dtypes import bfloat16
    storage = np.float16 if dtype == 'fp16' else bfloat16
    rng = np.random.default_rng(791)

    def run(package, buffers):
        result = launch(RuntimeArtifact(metadata={'target': 'apple_gpu'},
            native_image=package.image, launch_descriptor=package.descriptor,
            tile_ir=package.tile_ir, target_ir=package.target_ir), buffers)
        assert result['ok'], result
        assert result['execution_kind'] == 'native_gpu', result

    x = rng.standard_normal((3, 17)).astype(storage)
    out = np.zeros_like(x)
    run(apple_native.package_softmax(softmax_module(dtype),
        pipeline_name='tessera-lower-to-apple_gpu-runtime'), {'x': x, 'out': out})
    scores = x.astype(np.float64)
    expected = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected /= expected.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(out.astype(np.float32), expected, rtol=0.01, atol=0.001)

    artifact = scheduled_attention_backward.lower_scheduled_attention_backward(
        _apple_module(dtype=dtype, causal=True), target='apple_gpu')
    package = apple_native.package_scheduled_attention_backward(artifact,
        pipeline_name='tessera-lower-to-apple_gpu')
    q = rng.standard_normal((1, 4, 8, 16)).astype(storage)
    k = rng.standard_normal((1, 2, 8, 16)).astype(storage)
    v = rng.standard_normal((1, 2, 8, 16)).astype(storage)
    do = rng.standard_normal(q.shape).astype(storage)
    grads = [np.zeros(a.shape, dtype=np.float32) for a in (q, k, v)]
    run(package, dict(q=q, k=k, v=v, do=do, dq=grads[0], dk=grads[1], dv=grads[2]))
    expected_grads = _apple_backward_reference(*[a.astype(np.float64) for a in (do, q, k, v)],
                                               scale=0.25, causal=True)
    for actual, expected in zip(grads, expected_grads):
        np.testing.assert_allclose(actual, expected, rtol=0.005, atol=0.001)


BROAD_CASES = [
    ('fp16', 2, 4, 2, 7, 19, 32, False, False),
    ('bf16', 1, 8, 1, 9, 33, 64, True, False),
    ('fp16', 1, 2, 2, 17, 65, 128, False, False),
    ('bf16', 1, 4, 2, 19, 7, 16, True, False),
    ('fp32', 2, 4, 2, 7, 19, 32, False, True),
    ('fp32', 1, 8, 1, 9, 33, 64, True, True),
    ('fp16', 2, 4, 2, 7, 19, 32, False, True),
    ('bf16', 1, 8, 1, 9, 33, 64, True, True),
]


def broader_module(case):
    dtype, b, hq, hkv, sq, sk, d, causal, bias = case
    module = _apple_module(dtype=dtype, b=b, hq=hq, hkv=hkv, sq=sq, sk=sk, d=d, causal=causal)
    if bias:
        ty = IRType(f'tensor<{b}x{hq}x{sq}x{sk}xf32>', tuple(map(str, (b, hq, sq, sk))), 'fp32')
        fn = module.functions[0]
        fn.args.append(IRArg('bias', ty))
        fn.body[0].operands.append('%bias')
        fn.body[0].operand_types.append(str(ty))
    return module


@pytest.mark.parametrize('case', BROAD_CASES)
def test_broader_attention_projection(case):
    artifact = scheduled_attention_backward.lower_scheduled_attention_backward(
        broader_module(case), target='apple_gpu')
    apple_native._verify_attention_schedule_ancestry(artifact)


@pytest.mark.hardware_apple_gpu
@pytest.mark.skipif(sys.platform != 'darwin', reason='requires owning Metal host')
@pytest.mark.parametrize('case', BROAD_CASES)
def test_broader_metal_differential(case):
    from ml_dtypes import bfloat16
    from tessera.runtime import RuntimeArtifact, launch
    dtype, b, hq, hkv, sq, sk, d, causal, has_bias = case
    storage = {'fp16': np.float16, 'bf16': bfloat16, 'fp32': np.float32}[dtype]
    rng = np.random.default_rng(793)
    arrays = [rng.normal(size=shape).astype(storage) for shape in
              [(b, hq, sq, d), (b, hkv, sk, d), (b, hkv, sk, d), (b, hq, sq, d)]]
    q, k, v, do = arrays
    grads = [np.zeros(a.shape, np.float32) for a in (q, k, v)]
    buffers = dict(q=q, k=k, v=v, do=do, dq=grads[0], dk=grads[1], dv=grads[2])
    bias = rng.normal(scale=0.2, size=(b, hq, sq, sk)).astype(np.float32) if has_bias else None
    if bias is not None:
        buffers['bias'] = bias
    artifact = scheduled_attention_backward.lower_scheduled_attention_backward(
        broader_module(case), target='apple_gpu')
    package = apple_native.package_scheduled_attention_backward(artifact,
        pipeline_name='tessera-lower-to-apple_gpu')
    result = launch(RuntimeArtifact(metadata={'target': 'apple_gpu'}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=package.tile_ir, target_ir=package.target_ir), buffers)
    assert result['ok'] and result['execution_kind'] == 'native_gpu', result
    expected = _apple_backward_reference(do, q, k, v, scale=0.25, causal=causal, bias=bias)
    for actual, reference in zip(grads, expected):
        np.testing.assert_allclose(actual, reference, rtol=0.005, atol=0.001)


@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_lowp_fp32_bias_projects_without_narrowing(dtype, monkeypatch, tmp_path):
    artifact = scheduled_attention_backward.lower_scheduled_attention_backward(
        broader_module((dtype, 1, 4, 2, 7, 19, 32, False, True)), target='apple_gpu')
    library = tmp_path / 'runtime.dylib'
    library.write_bytes(b'contract fixture')
    monkeypatch.setattr(apple_native, '_runtime_library_path', lambda: library)
    package = apple_native.package_scheduled_attention_backward(artifact, pipeline_name='tessera-lower-to-apple_gpu')
    assert '_bias_f32.' in package.descriptor.abi_id
    assert package.descriptor.buffers[4].dtype == 'fp32'
    assert package.descriptor.buffers[4].alignment == 4
