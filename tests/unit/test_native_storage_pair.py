"""AD lineage and JIT binding refuse malformed or mismatched pair contracts."""
import hashlib
import json
from types import SimpleNamespace
import pytest
import tessera as ts
from tessera.compiler import native_storage_pair as pair_module
from tessera.compiler import native_storage_contract


def package(monkeypatch, **updates):
    data = dict(schema=1, mode='forward', source_digest='a'*64,
                paired_ir='paired program', paired_digest=hashlib.sha256(b'paired program').hexdigest(),
                output_order=['primal', 'tangent'])
    data.update(updates)
    encoded = json.dumps(data).replace('"', '\\22')
    monkeypatch.setattr(native_storage_contract, 'read_tensor_contract', lambda _: dict(arguments=[
        dict(name='arg0'), dict(name='arg1'), dict(name='primal', writable=True),
        dict(name='derivative', writable=True), dict(name='n')]))
    binding = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(pair_module, 'generate_tensor_binding', lambda *_: binding)
    return SimpleNamespace(validate=lambda: None, arena_ir='tessera.native_ad_contract = "'+encoded+'"')


@pytest.mark.parametrize('change,match', [
    ({'schema': True}, 'unsupported'),
    ({'paired_digest': '0'*64}, 'identity'),
    ({'output_order': ['tangent', 'primal']}, 'order'),
    ({'mode': 'reverse'}, 'order'),
])
def test_pair_rejects_invalid_lineage(monkeypatch, change, match):
    with pytest.raises(ValueError, match=match):
        pair_module.NativeStoragePair(package(monkeypatch, **change))


def test_contract_is_immutable(monkeypatch):
    pair = pair_module.NativeStoragePair(package(monkeypatch))
    with pytest.raises(TypeError):
        pair.contract['mode'] = 'reverse'
    assert pair.contract['output_order'] == ('primal', 'tangent')


def test_jit_refuses_wrong_physical_signature(monkeypatch):
    @ts.jit
    def wrong(x):
        return x
    with pytest.raises(ValueError, match='physical signature'):
        wrong.bind_native_storage_pair(package(monkeypatch))


def test_jit_refuses_wrong_ad_mode(monkeypatch):
    @ts.jit(autodiff='reverse')
    def wrong(arg0, arg1, primal, derivative, n):
        return primal, derivative
    with pytest.raises(ValueError, match='differentiation mode'):
        wrong.bind_native_storage_pair(package(monkeypatch))
