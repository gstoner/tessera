"""ROCm matmul replay reuse retains per-call ABI and Tile validation."""
from dataclasses import replace
import pytest
from tessera.compiler import rocm_pass_cache as cache, scheduled_matmul as sm
from tests.unit.test_scheduled_matmul_consumers import _module, requires_nvidia_target_ir

pytestmark = pytest.mark.compiler_route

@pytest.fixture
def native_matmul(production_compiler, monkeypatch):
    artifact = sm.lower_scheduled_matmul(
        _module(target="rocm", shape=(17, 31, 19)), target="rocm_gfx1201")
    execute = sm.run_tessera_opt
    calls = []
    def observed(*args):
        calls.append(args[2])
        return execute(*args)
    monkeypatch.setattr(sm, "run_tessera_opt", observed)
    cache.clear()
    def prime():
        sm.verify_matmul_projection(artifact)
        assert calls == ["--tessera-schedule-to-tile", "--canonicalize"]
        calls.clear()
        return artifact, calls
    yield prime
    cache.clear()

def test_repeated_matmul_projection_reuses_passes(native_matmul):
    artifact, calls = native_matmul()
    sm.verify_matmul_projection(artifact)
    assert calls == []

@pytest.mark.parametrize("field,value", [
    ("m", 18), ("k", 32), ("n", 20), ("a_dtype", "bf16"),
    ("output_dtype", "fp16"), ("activation", "relu"),
    ("architecture", "gfx1151"),
])
def test_cache_prime_does_not_admit_mutated_descriptor(native_matmul, field, value):
    artifact, calls = native_matmul()
    with pytest.raises(ValueError, match="disagrees"):
        sm.verify_matmul_projection(replace(artifact, **{field: value}))
    assert calls == []
    sm.verify_matmul_projection(artifact)
    assert calls == []

def test_cache_prime_does_not_admit_changed_tile(native_matmul):
    artifact, calls = native_matmul()
    with pytest.raises(ValueError, match="Tile product disagrees"):
        sm.verify_matmul_projection(replace(artifact, tile_ir=artifact.tile_ir+"\n"))
    assert calls == []

def test_changed_native_parent_requires_new_replay(native_matmul):
    artifact, calls = native_matmul()
    changed = replace(artifact, schedule_ir=artifact.schedule_ir+"\n")
    sm.verify_matmul_projection(changed)
    assert calls == ["--tessera-schedule-to-tile", "--canonicalize"]

@requires_nvidia_target_ir
def test_nvidia_projection_does_not_enter_rocm_cache(production_compiler, monkeypatch):
    artifact = sm.lower_scheduled_matmul(
        _module(target="nvidia_sm120", shape=(16, 16, 16)), target="nvidia_sm120")
    def forbidden(*args, **kwargs):
        raise AssertionError("NVIDIA projection entered ROCm cache")
    monkeypatch.setattr(cache, "run", forbidden)
    sm.verify_matmul_projection(artifact)
