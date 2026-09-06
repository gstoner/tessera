"""Generated ABI contracts retain package identity and cannot bypass the arbiter."""
from dataclasses import replace
import inspect
from types import SimpleNamespace
import pytest
from tessera.compiler.native_gpu_storage import NativeGPUStoragePackage
from tessera.compiler.native_gpu_tensor import TensorSpec, IndexSpec
from tessera.compiler.native_storage_contract import attach_tensor_contract, generate_tensor_binding


def package():
    source = attach_tensor_contract('module {\n}', (TensorSpec('output', 'float32', ('n',), True), IndexSpec('n')),
                                   grid=(1, 1, 1), block=('n', 1, 1))
    p = NativeGPUStoragePackage('rocm', 'gfx1151', 'entry', 'size', ('pointer', 'index'), source,
                                b'image', b'host', 'c' * 64, 'd' * 64, '')
    return replace(p, binding_digest=p._digest())


def test_generate_binding_from_compiler_preserved_manifest():
    p = package()
    binding = generate_tensor_binding(p, inspect.signature(lambda output, n: None))
    assert binding.block == ('n', 1, 1)
    assert binding.specs == (TensorSpec('output', 'float32', ('n',), True), IndexSpec('n'))
    assert generate_tensor_binding(p, binding.signature).binding_digest == binding.binding_digest


def test_missing_or_edited_abi_cannot_reuse_pinned_package():
    p = package()
    with pytest.raises(ValueError, match='binding'):
        generate_tensor_binding(replace(p, arena_ir='module {}'), inspect.signature(lambda output, n: None))
    p = replace(p, arena_ir='module {}')
    p = replace(p, binding_digest=p._digest())
    with pytest.raises(ValueError, match='manifest'):
        generate_tensor_binding(p, inspect.signature(lambda output, n: None))


def test_generated_arbiter_requires_real_candidate_execution(monkeypatch):
    from tessera.compiler.emit.native_storage_candidate import NativeStorageCandidate, _verify
    p = package()
    candidate = NativeStorageCandidate(p, inspect.signature(lambda output, n: None), lambda *a, **kw: True)
    assert not _verify(candidate, p.binding_digest)
    # An oracle returning truth without a completed run never qualifies.
    assert not _verify(candidate, p.binding_digest)
    assert candidate.native_runs == 0


def test_paired_storage_rejects_noncompiler_rule():
    from tessera.compiler.native_storage_jvp import NativeStorageJVP
    with pytest.raises(ValueError, match='compiler'):
        NativeStorageJVP((lambda x: x, lambda dx: dx))


def test_paired_storage_binds_children_and_preserves_primal_tangent_order(monkeypatch):
    from tessera.compiler.native_jvp import build_native_jvp_artifact
    from tessera.compiler.native_storage_jvp import NativeStorageJVP
    from tessera.compiler.native_gpu_tensor import NativeTensorCall
    p = package()
    artifact = build_native_jvp_artifact(target='rocm', architecture='gfx1151', family='native_storage',
        source_graph_ir='module attributes {tessera.frontend.authority = "tracer"} {}',
        paired_jvp_ir='func.func @program__jvp()', wrt_indices=[0], arg_names=['out', 'dout', 'n'],
        steps=[dict(id=name, child_digest=p.binding_digest, child_artifact={'native_storage_json': p.to_json()},
                    inputs=[value, 'n'], outputs=[name]) for name, value in [('primal', 'out'), ('tangent', 'dout')]])
    pair = NativeStorageJVP(artifact)
    def array(pointer):
        return SimpleNamespace(__cuda_array_interface__=dict(version=3, shape=(32,), typestr='<f4', data=(pointer, False)))
    out, dout = array(4096), array(8192)
    calls = []
    monkeypatch.setattr(NativeTensorCall, '__call__', lambda self, *args: calls.append(args))
    assert pair(out, dout, 32) == (out, dout)
    assert calls == [(out, 32), (dout, 32)]
    # A bad tangent ABI must not execute even the valid primal step.
    calls.clear()
    with pytest.raises(ValueError):
        pair(out, array(8193), 32)
    assert not calls


def test_retiring_old_candidate_preserves_same_name_replacement():
    from tessera.compiler.emit.candidate import candidates_for
    from tessera.compiler.emit.native_storage_candidate import register_native_storage_candidate
    p = package()
    signature = inspect.signature(lambda output, n: None)
    first = register_native_storage_candidate(p, signature, lambda *a, **kw: False)
    second = register_native_storage_candidate(p, signature, lambda *a, **kw: False)
    first.close()
    assert second in candidates_for(second.target, second.op)
    second.close()
    assert second not in candidates_for(second.target, second.op)
