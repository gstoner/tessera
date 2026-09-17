"""Native low-precision package contracts and owning-Metal differential proof."""

import numpy as np
import pytest
from pathlib import Path

from tessera.compiler import apple_native, scheduled_attention_backward, scheduled_kernel
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import find_tessera_opt
from test_scheduled_attention_backward_consumers import _apple_module, _apple_backward_reference

def _apple_backend_reason() -> str | None:
    """Why this module cannot run here, or None when it can.

    `find_tessera_opt() is None` was the only guard, and it is the wrong
    question on a Linux box: tessera-opt exists there, built without
    TESSERA_BUILD_APPLE_BACKEND, so every `--tessera-*-to-apple_gpu` pass is
    unknown and 39 tests failed with "Unknown command line argument" on both
    Princess-Luna and Super-Bear — a host reporting "no Apple backend here" as a
    broken contract. Ask the binary what it registered.
    """
    from tests._support.compiler_tool import registered_passes

    tool = find_tessera_opt()
    if tool is None:
        return 'native compiler required'
    if 'tessera-lower-to-apple_gpu' not in registered_passes(Path(tool)):
        return ("this host's tessera-opt was built without the Apple backend "
                "(no tessera-lower-to-apple_gpu pipeline)")
    return None


_SKIP_REASON = _apple_backend_reason()
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or '')


def softmax_module(dtype):
    element = {'fp32': 'f32', 'fp16': 'f16', 'bf16': 'bf16'}[dtype]
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
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_lowp_metal_differential(dtype):
    from tessera.runtime import RuntimeArtifact, launch
    from ml_dtypes import bfloat16
    storage = {'fp32': np.float32, 'fp16': np.float16, 'bf16': bfloat16}[dtype]
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


@pytest.mark.parametrize('dtype', ['fp32', 'fp16', 'bf16'])
@pytest.mark.parametrize('status', [0, -1, None, 'missing'])
def test_lowp_softmax_non_native_status_refuses(dtype, status, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from ml_dtypes import bfloat16
    from tessera import runtime
    library = tmp_path / 'runtime.dylib'
    library.write_bytes(b'contract fixture')
    monkeypatch.setattr(apple_native, '_runtime_library_path', lambda: library)
    package = apple_native.package_softmax(softmax_module(dtype), pipeline_name='tessera-lower-to-apple_gpu')
    assert package.descriptor.entry_symbol.endswith('_status')
    assert package.descriptor.abi_id.endswith('.v2')
    fake = SimpleNamespace()
    if status != 'missing':
        setattr(fake, package.descriptor.entry_symbol, lambda *args: status)
    monkeypatch.setattr(runtime, '_load_apple_gpu_runtime', lambda: fake)
    storage = {'fp32': np.float32, 'fp16': np.float16, 'bf16': bfloat16}[dtype]
    buffers = {'x': np.zeros((3, 17), storage), 'out': np.zeros((3, 17), storage)}
    with pytest.raises(RuntimeError, match='did not execute on Metal|runtime is missing'):
        runtime._submit_apple_gpu_native(package.image, package.descriptor, buffers, {}, None)


def apple_recompute_pair():
    from tessera.compiler.rocm_native import emit_attention_backward_graph_ir
    return emit_attention_backward_graph_ir(
        forward_entry='primal', backward_entry='vjp', storage='f16',
        dims=(1, 4, 2, 7, 19, 32, 32), scale=.25, causal=False, bias=False,
        window_left=-1, window_right=-1, softcap=0., save_lse=False,
    ).replace('module {', 'module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {', 1)


@pytest.mark.parametrize('mutation', ['standalone', 'missing_link', 'wrong_primal', 'wrong_scale', 'wrong_bias'])
def test_apple_lowp_recompute_requires_matching_vjp(mutation):
    from tessera.compiler.scheduled_matmul import run_tessera_opt
    graph = apple_recompute_pair()
    if mutation == 'standalone':
        graph = graph.split('  func.func @vjp(')[0] + '}\n'
    elif mutation == 'missing_link':
        graph = graph.replace('tessera.vjp = @vjp, ', '')
    elif mutation == 'wrong_primal':
        graph = graph.replace('tessera.primal = @primal', 'tessera.primal = @vjp')
    elif mutation == 'wrong_scale':
        forward, backward = graph.split('  func.func @vjp(', 1)
        import re
        backward, count = re.subn(r'scale = [^ ]+ : f32', 'scale = 5.000000e-01 : f32', backward)
        assert count == 1
        graph = forward + '  func.func @vjp(' + backward
    else:
        graph = graph.replace('dense<0.000000e+00>', 'dense<1.000000e+00>')
    with pytest.raises(RuntimeError, match='scheduled compiler boundary'):
        run_tessera_opt(find_tessera_opt(), graph, '--tessera-graph-to-schedule')


def test_apple_lowp_recompute_accepts_verified_pair():
    from tessera.compiler.scheduled_matmul import run_tessera_opt
    result = run_tessera_opt(find_tessera_opt(), apple_recompute_pair(), '--tessera-graph-to-schedule')
    assert 'schedule.attention_backward' in result


@pytest.mark.parametrize('dtype', ['fp32', 'fp16', 'bf16'])
@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('status', [0, -1, None, 'missing', 1])
def test_gelu_native_status_boundary(dtype, dynamic, status, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from ml_dtypes import bfloat16
    from tessera import runtime
    from tests.unit.test_apple_e2e_native_spine import _gelu_contract_module
    library = tmp_path / 'runtime.dylib'
    library.write_bytes(b'contract fixture')
    monkeypatch.setattr(apple_native, '_runtime_library_path', lambda: library)
    shape = ('?', '?') if dynamic else ('3', '17')
    module = _gelu_contract_module(dtype, dtype, shape, shape)
    package_fn = apple_native.package_dynamic_gelu if dynamic else apple_native.package_gelu
    package = package_fn(module, pipeline_name='tessera-lower-to-apple_gpu')
    assert package.descriptor.entry_symbol.endswith('_status')
    assert package.descriptor.abi_id.endswith('.v2')
    fake = SimpleNamespace()
    if status != 'missing':
        setattr(fake, package.descriptor.entry_symbol, lambda *args: status)
    monkeypatch.setattr(runtime, '_load_apple_gpu_runtime', lambda: fake)
    storage = {'fp32': np.float32, 'fp16': np.float16, 'bf16': bfloat16}[dtype]
    buffers = {'a0': np.zeros((3, 17), storage), 'out': np.zeros((3, 17), storage)}
    scalars = {'Elements': 51} if dynamic else {}
    if status == 1:
        assert runtime._submit_apple_gpu_native(package.image, package.descriptor, buffers, scalars, None) is buffers['out']
    else:
        with pytest.raises(RuntimeError, match='did not execute on Metal|runtime is missing'):
            runtime._submit_apple_gpu_native(package.image, package.descriptor, buffers, scalars, None)


@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('mutation', ['status', 'operand', 'target', 'shape'])
def test_gelu_artifact_replays_native_parent(mutation, dynamic, monkeypatch, tmp_path):
    from dataclasses import replace
    from tests.unit.test_apple_e2e_native_spine import _gelu_contract_module
    shape = ('?', '?') if dynamic else ('3', '17')
    module = _gelu_contract_module('fp32', 'fp32', shape, shape)
    artifact = apple_native.lower_gelu_artifact(module)
    if mutation == 'status':
        artifact = replace(artifact, native_ir=artifact.native_ir.replace('arith.cmpi eq', 'arith.cmpi ne'))
    elif mutation == 'operand':
        import re
        swapped = re.sub(r'(call @tessera_apple_gpu_gelu\w*\()(%[^,]+), (%[^,]+)', r'\1\3, \2', artifact.native_ir)
        assert swapped != artifact.native_ir
        artifact = replace(artifact, native_ir=swapped)
    elif mutation == 'target':
        artifact = replace(artifact, parent_ir=artifact.parent_ir.replace('apple7', 'apple8'))
    else:
        artifact = replace(artifact, parent_ir=artifact.parent_ir.replace('?x?', '?x19') if dynamic else artifact.parent_ir.replace('3x17', '3x19'))
    monkeypatch.setattr(apple_native, '_runtime_library_path', lambda: pytest.fail('runtime touched before replay rejection'))
    with pytest.raises(ValueError, match='parent|replay'):
        apple_native.package_gelu_artifact(artifact, pipeline_name='tessera-lower-to-apple_gpu')


def test_dynamic_gelu_native_dimensions_and_named_argument(monkeypatch, tmp_path):
    from tests.unit.test_apple_e2e_native_spine import _gelu_contract_module
    module = _gelu_contract_module('fp32', 'fp32', ('?', '?'), ('?', '?'))
    module.functions[0].args[0].dim_names = ('M', 'K')
    artifact = apple_native.lower_gelu_artifact(module)
    assert 'tensor.dim' in artifact.native_ir
    assert 'memref.alloc(' in artifact.native_ir
    # Capacity and positivity checks dominate the output allocation.
    prefix = artifact.native_ir.split('memref.alloc(', 1)[0]
    assert prefix.count('cf.assert') >= 3
    library = tmp_path / 'runtime.dylib'
    library.write_bytes(b'contract fixture')
    monkeypatch.setattr(apple_native, '_runtime_library_path', lambda: library)
    package = apple_native.package_gelu_artifact(artifact, pipeline_name='tessera-lower-to-apple_gpu')
    assert package.descriptor.provenance['dynamic_shape'] is True
    assert package.descriptor.provenance['shape'] == [None, None]
    assert package.descriptor.scalars[0].name == 'Elements'
