"""Host-free identity/ABI checks; execution evidence lives on the owning Mac."""
from dataclasses import replace
import json
import pytest
from tessera.compiler.apple_native_arena import AppleNativeArena, AppleArenaPackage, _apple_abi


def artifact():
    return AppleNativeArena('kernel void scratch(\ndevice uchar* v0 [[buffer(0)]],\nconstant long& v1 [[buffer(1)]],\nthreadgroup uchar* arena [[threadgroup(0)]]',
                            'host LLVM', '__tessera_shared_bytes_scratch', 'native IR', 'compiler')


def package():
    p = AppleArenaPackage(artifact(), b'library', 'bridge', '')
    return replace(p, binding_digest=p._digest())


def test_apple_package_roundtrip_is_pinned():
    p = package()
    assert AppleArenaPackage.from_json(p.to_json(), expected_digest=p.binding_digest) == p
    with pytest.raises(ValueError, match='pinned'):
        AppleArenaPackage.from_json(p.to_json(), expected_digest='another package')


@pytest.mark.parametrize('field', ['msl', 'host_llvm_ir', 'sizer', 'arena_ir', 'compiler_digest'])
def test_apple_package_rejects_shader_companion_or_lineage_edits(field):
    p = package()
    data = json.loads(p.to_json())
    data['artifact'][field] += ' changed'
    with pytest.raises(ValueError, match='identity'):
        AppleArenaPackage.from_json(json.dumps(data), expected_digest=p.binding_digest)


@pytest.mark.parametrize('before,after', [('buffer(1)', 'buffer(0)'), ('constant long&', 'constant int&'),
                                         ('threadgroup(0)', 'threadgroup(1)')])
def test_apple_package_rejects_unsupported_binding_abi(before, after):
    a = artifact()
    with pytest.raises(ValueError):
        _apple_abi(replace(a, msl=a.msl.replace(before, after)))


def test_typed_apple_manifest_matches_signature_and_dtype():
    import inspect
    from tessera.compiler.apple_native_arena import AppleTensorCall
    from tessera.compiler.native_storage_contract import attach_tensor_contract
    from tessera.compiler.native_gpu_tensor import TensorSpec, IndexSpec
    p = package()
    source = attach_tensor_contract('module {\n}', (TensorSpec('output', 'fp32', ('n',), True),
        IndexSpec('n', 1, 256)), grid=(1, 1, 1), block=('n', 1, 1))
    p = replace(p, artifact=replace(p.artifact, arena_ir=source), binding_digest='')
    p = replace(p, binding_digest=p._digest())
    signature = inspect.signature(lambda output, n: None)
    binding = AppleTensorCall(p, signature)
    with pytest.raises(ValueError, match='index bounds'):
        binding.prepare(None, True)
    with pytest.raises(ValueError, match='signature'):
        AppleTensorCall(p, inspect.signature(lambda different, n: None))
    changed = replace(p, artifact=replace(p.artifact, arena_ir=source.replace('fp32', 'fp16')), binding_digest='')
    changed = replace(changed, binding_digest=changed._digest())
    with pytest.raises(ValueError, match='f32'):
        AppleTensorCall(changed, signature)


def test_apple_jit_reports_explicit_binding_without_eager_fallback():
    import tessera as ts
    from tessera.compiler.native_storage_contract import attach_tensor_contract
    from tessera.compiler.native_gpu_tensor import TensorSpec, IndexSpec
    p = package()
    source = attach_tensor_contract('module {\n}', (TensorSpec('output', 'fp32', ('n',), True),
        IndexSpec('n', 1, 256)), grid=(1, 1, 1), block=('n', 1, 1))
    p = replace(p, artifact=replace(p.artifact, arena_ir=source), binding_digest='')
    p = replace(p, binding_digest=p._digest())
    @ts.jit
    def dispatch(output, n):
        return output
    dispatch.bind_apple_native_arena(p)
    assert dispatch.execution_kind == 'native_gpu'
    assert dispatch.runtime_artifact().metadata['package_digest'] == p.binding_digest
    with pytest.raises(ValueError, match='index bounds'):
        dispatch(None, True)
    dispatch.close_native_storage()


def test_apple_binding_cannot_silently_drop_requested_ad():
    import tessera as ts
    @ts.jit(autodiff='forward')
    def differentiable(x):
        return x
    with pytest.raises(ValueError, match='paired differentiation'):
        differentiable.bind_apple_native_arena(package())
