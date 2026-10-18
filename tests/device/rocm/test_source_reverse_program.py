"""Exact owning ROCm source VJP with private native scratch and portable replay."""
from functools import lru_cache
import os

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.native_rocm_source_reverse import package_source_reverse, RocmSourceReverseProgram
from benchmarks.math.source_reverse_reference import inputs, check
from tests.unit.test_native_rocm_source_reverse import source

pytestmark = [
    pytest.mark.hardware_rocm,
    pytest.mark.skipif(os.environ.get("TESSERA_ROCM_SOURCE_REVERSE_DEVICE_PROOF") != "1",
                       reason="explicit owning ROCm source-pullback proof required"),
]


@lru_cache(maxsize=16)
def packaged(arch, storage, permuted, shape=(17, 19, 35)):
    return package_source_reverse(source(arch, storage, permuted, shape))


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("permuted", [False, True])
def test_native_source_pullback_replays_current_roots_and_preserves_results(storage, permuted, monkeypatch):
    arch = os.environ["TESSERA_ROCM_CHIP"]
    assert rt._rocm_live_arch() == arch
    package = packaged(arch, storage, permuted)
    with package.prepare(inputs(storage, permuted)) as owner:
        generation, _ = owner.invoke()
        first = owner.read(generation)
        check(first, inputs(storage, permuted), permuted)
        retained = first[0].copy()
        from tessera.compiler import scheduled_matmul, rocm_native
        with monkeypatch.context() as warm:
            warm.setattr(scheduled_matmul, "find_tessera_opt", lambda: pytest.fail("warm VJP requested compiler"))
            warm.setattr(rocm_native, "_run_opt", lambda *a, **k: pytest.fail("warm VJP compiled"))
            warm.setattr(rt, "_launch_native_descriptor", lambda *a, **k: pytest.fail("Python launched a member"))
            for seed in (193, 427):
                roots = inputs(storage, permuted, seed)
                owner.update(roots)
                generation, _ = owner.invoke()
                actual = owner.read(generation)
                check(actual, roots, permuted)
                assert not np.shares_memory(first[0], actual[0])
        np.testing.assert_array_equal(first[0], retained)
        generation, elapsed = owner.invoke(repeats=16, timed=True)
        assert np.isfinite(elapsed) and elapsed > 0
        check(owner.read(generation), roots, permuted)
        generation, stages = owner.profile_members(repeats=16)
        assert len(stages) == 2 and all(np.isfinite(v) and v > 0 for v in stages)
        check(owner.read(generation), roots, permuted)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("permuted", [False, True])
def test_source_pullback_preserves_fp32_seed_cancellation(storage, permuted):
    arch = os.environ["TESSERA_ROCM_CHIP"]
    assert rt._rocm_live_arch() == arch
    roots = inputs(storage, permuted, witness=True)
    with packaged(arch, storage, permuted).prepare(roots) as owner:
        generation, _ = owner.invoke()
        actual = owner.read(generation)
        check(actual, roots, permuted)
        assert np.any(actual[0] != 0)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_portable_source_pullback_validates_and_executes_without_compiler(storage, monkeypatch):
    arch = os.environ["TESSERA_ROCM_CHIP"]
    package = packaged(arch, storage, False)
    encoded = package.to_json()
    from tessera.compiler import scheduled_matmul, rocm_native
    with monkeypatch.context() as offline:
        offline.setattr(scheduled_matmul, "find_tessera_opt", lambda: pytest.fail("portable VJP requested compiler"))
        offline.setattr(rocm_native, "_run_opt", lambda *a, **k: pytest.fail("portable VJP compiled"))
        restored = RocmSourceReverseProgram.from_json(encoded)
        roots = inputs(storage, False, 761)
        with restored.prepare(roots) as owner:
            generation, _ = owner.invoke()
            check(owner.read(generation), roots, False)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_fresh_process_source_pullback_has_no_compiler_dependency(storage, tmp_path):
    import subprocess
    import sys
    arch = os.environ["TESSERA_ROCM_CHIP"]
    path = tmp_path / "source-vjp.json"
    path.write_text(packaged(arch, storage, False).to_json())
    code = """
import sys
from pathlib import Path
from tessera.compiler.native_rocm_source_reverse import RocmSourceReverseProgram
from benchmarks.math.source_reverse_reference import inputs, check
package = RocmSourceReverseProgram.from_json(Path(sys.argv[1]).read_text())
roots = inputs(sys.argv[2], False, 911)
with package.prepare(roots) as owner:
    generation, _ = owner.invoke()
    check(owner.read(generation), roots, False)
print(package.manifest["architecture"] + " compiler-free source VJP passed")
"""
    env = dict(os.environ, TESSERA_OPT="/missing/compiler", TESSERA_ROCM_OPT="/missing/compiler",
               TESSERA_LLVM_BIN="/missing/llvm", TESSERA_NVIDIA_OPT="/missing/compiler")
    result = subprocess.run([sys.executable, "-c", code, str(path), storage],
                            env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert arch + " compiler-free source VJP passed" in result.stdout
