"""Exact gfx1201 resident expansion/product/BF16 activation ownership."""
import ctypes as C
import os

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_expansion_program import package_native_nvfp4_expansion_program
from tests.device.rocm.test_prepared_nvfp4_expansion_program import inputs_and_oracle
from tests.device.rocm.test_native_parent_view_bf16_jit import oracle, assert_result
from tests.unit.test_native_nvfp4_parent_view_program import resident_source

pytestmark = pytest.mark.skipif(
    os.getenv("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="requires owning gfx1201",
)


@pytest.mark.parametrize("rows,width,k", [(3, 17, 48), (16, 64, 80)])
def test_resident_bf16_parent_remains_private_across_activation_updates(rows, width, k, monkeypatch):
    assert rt._rocm_live_arch() == "gfx1201"
    package = package_native_nvfp4_expansion_program(resident_source(rows, width, k))
    roots, parent = inputs_and_oracle(rows, 2 * width, k, (0, 1, 2, 3, 4), "bf16")
    original = [value.copy() for value in roots]
    expected = oracle(parent, width, False)
    expected_changed = oracle(-parent, width, False)
    import subprocess
    def forbidden(*args, **kwargs):
        pytest.fail("resident BF16 invocation called compiler or eager SiLU arithmetic")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(np, "exp", forbidden)
    monkeypatch.setattr(rt, "_execute_rocm_compiled_silu_mul", forbidden)
    with package.prepare(roots, runtime_library=os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]) as owner:
        with pytest.raises(RuntimeError, match="status 10"):
            owner.invoke_consumer()
        imported, _ = owner.import_weights()
        with pytest.raises(RuntimeError):
            owner.read(imported)
        generation, _ = owner.invoke_consumer()
        result, = owner.read(generation)
        assert_result(result, expected)
        retained = result.copy()
        # The matmul parent is private storage, never an exported host result.
        probe = np.full(parent.shape, 123, dtype=parent.dtype)
        status = owner.lib.tessera_rocm_program_read(
            owner.handle, 7, generation, C.c_void_p(probe.ctypes.data), probe.nbytes)
        assert status != 0
        np.testing.assert_array_equal(probe, np.full_like(probe, 123))
        changed = -roots[3]
        owner.update_activations(changed, roots[4])
        with pytest.raises(RuntimeError):
            owner.read(generation)
        updated, _ = owner.invoke_consumer()
        again, = owner.read(updated)
        assert_result(again, expected_changed)
        np.testing.assert_array_equal(result, retained)
        assert not np.shares_memory(result, again)
        for actual, before in zip(roots, original, strict=True):
            np.testing.assert_array_equal(actual, before)
    assert owner.handle.value == 0


@pytest.mark.parametrize("rows,width,k", [(3, 17, 48), (16, 64, 80)])
@pytest.mark.parametrize("reverse", [False, True])
def test_python_composed_nvfp4_activation_jit_and_replay(rows, width, k, reverse, monkeypatch):
    import subprocess
    import tessera as ts
    from tessera.compiler import nvfp4_expansion, reference_typed_scaled_matmul
    from tests.unit.test_native_nvfp4_parent_view_program import frontend_activation

    assert rt._rocm_live_arch() == "gfx1201"
    roots, parent = inputs_and_oracle(rows, 2 * width, k, (0, 1, 2, 3, 4), "bf16")
    before = [value.copy() for value in roots]
    expected, changed_expected = oracle(parent, width, reverse), oracle(-parent, width, reverse)
    def forbidden(*args, **kwargs):
        pytest.fail("composed native execution invoked compiler or eager arithmetic")
    monkeypatch.setattr(nvfp4_expansion, "reference_nvfp4_expand", forbidden)
    monkeypatch.setattr(reference_typed_scaled_matmul, "reference_typed_scaled_matmul", forbidden)
    monkeypatch.setattr(rt, "_execute_rocm_compiled_silu_mul", forbidden)
    compiled = ts.jit(target="rocm_gfx1201")(frontend_activation(rows, width, reverse))
    with monkeypatch.context() as local:
        local.setattr(np, "exp", forbidden)
        first = compiled(*roots)
    assert compiled.execution_kind == "native_gpu"
    assert_result(first, expected)
    assert len(compiled.native_nvfp4_packages()) == 3
    artifact = compiled.runtime_artifact()
    assert artifact.metadata["native_program"]["schema"] == "tessera.rocm.nvfp4_source_frontend_program.v2"
    assert "3 ordered native packages" in compiled.compile_report().target_decision["rocm_gfx1201"]
    changed = list(roots)
    changed[3] = -roots[3]
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(np, "exp", forbidden)
    second = compiled(*changed)
    assert_result(second, changed_expected)
    replay = rt.launch(rt.RuntimeArtifact.from_json(artifact.to_json()), roots)
    assert replay["ok"] and replay["execution_kind"] == "native_gpu", replay
    assert_result(replay["output"], expected)
    assert_result(first, expected)
    for actual, original in zip(roots, before, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_composed_nvfp4_fresh_process_replay_and_schema_separation(tmp_path):
    import subprocess
    import sys
    from copy import deepcopy
    import tessera as ts
    from tessera.compiler.rocm_nvfp4_source_frontend import source_program_from_manifest
    from tessera.compiler.rocm_nvfp4_resident import _program_digest
    from tests.unit.test_native_nvfp4_parent_view_program import frontend_activation

    assert rt._rocm_live_arch() == "gfx1201"
    roots, parent = inputs_and_oracle(3, 34, 48, (0, 1, 2, 3, 4), "bf16")
    expected = oracle(parent, 17, False)
    compiled = ts.jit(target="rocm_gfx1201")(frontend_activation(3, 17))
    assert_result(compiled(*roots), expected)
    artifact = compiled.runtime_artifact()
    manifest = artifact.metadata["native_program"]
    for downgrade in (True, False):
        broken = deepcopy(manifest)
        if downgrade:
            broken["schema"] = "tessera.rocm.nvfp4_source_frontend_program.v1"
        else:
            broken["native_program"]["stages"].pop()
        broken["contract_digest"] = _program_digest(
            {key: value for key, value in broken.items() if key != "contract_digest"})
        with pytest.raises(ValueError, match="record types differ"):
            source_program_from_manifest(broken)
    path = tmp_path / "program.json"
    path.write_text(artifact.to_json())
    arrays = tmp_path / "inputs.npz"
    np.savez(arrays, **{f"a{i}": value for i, value in enumerate(roots)},
             expected_bits=expected.view(np.uint16))
    script = """
import sys, subprocess
import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tessera.compiler import nvfp4_expansion, reference_typed_scaled_matmul
from tessera.compiler import rocm_nvfp4_expansion_program
def forbidden(*a, **kw):
    raise AssertionError("fresh replay invoked compiler or eager arithmetic")
subprocess.run = forbidden
np.exp = forbidden
nvfp4_expansion.reference_nvfp4_expand = forbidden
reference_typed_scaled_matmul.reference_typed_scaled_matmul = forbidden
rocm_nvfp4_expansion_program.package_native_nvfp4_expansion_program = forbidden
rt._execute_rocm_compiled_silu_mul = forbidden
assert rt._rocm_live_arch() == "gfx1201"
artifact = rt.RuntimeArtifact.from_json(open(sys.argv[1]).read())
data = np.load(sys.argv[2])
args = [data[f"a{i}"] for i in range(5)]
for i in (1,3):
    args[i] = args[i].view(ml_dtypes.float8_e4m3fn)
result = rt.launch(artifact, args)
assert result["ok"] and result["execution_kind"] == "native_gpu", result
assert result["output"].dtype == np.dtype(ml_dtypes.bfloat16)
expected = data["expected_bits"].view(ml_dtypes.bfloat16)
np.testing.assert_allclose(result["output"].astype(np.float32), expected.astype(np.float32),
                           rtol=1/128, atol=1e-5)
"""
    child = subprocess.run([sys.executable, "-c", script, str(path), str(arrays)],
        env=dict(os.environ, TESSERA_OPT="/no/compiler", TESSERA_ROCM_OPT="/no/compiler"),
        text=True, capture_output=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr
