"""Ordinary public JIT and compiler-free replay of native expansion/product."""
import os
import subprocess
import sys

import numpy as np
import pytest
import tessera as ts

from tests.device.rocm.test_prepared_nvfp4_expansion_program import frontend_product, inputs_and_oracle

pytestmark = pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
                               reason="requires owning gfx1201")


@pytest.mark.parametrize("k", [16, 48, 80])
def test_public_source_program_jit_and_replay(k, monkeypatch, tmp_path):
    from tessera import runtime as rt
    from tessera.compiler import nvfp4_expansion, reference_typed_scaled_matmul
    def forbidden(*args, **kwargs):
        pytest.fail("native JIT/replay called an eager numerical oracle")
    monkeypatch.setattr(nvfp4_expansion, "reference_nvfp4_expand", forbidden)
    monkeypatch.setattr(reference_typed_scaled_matmul, "reference_typed_scaled_matmul", forbidden)
    fn = ts.jit(target="rocm_gfx1201")(frontend_product)
    roots, expected = inputs_and_oracle(19, 7, k, (0, 1, 2, 3, 4), "f32")
    original = [x.copy() for x in roots]
    first = fn(*roots)
    np.testing.assert_array_equal(first, expected)
    assert fn.execution_kind == "native_gpu"
    assert len(fn.native_nvfp4_packages()) == 2
    artifact = fn.runtime_artifact()
    assert artifact.metadata["native_program"]["schema"] == "tessera.rocm.nvfp4_source_frontend_program.v1"
    assert artifact.metadata["native_graph_verified"]
    report = fn.compile_report()
    assert report.plan_hash == artifact.artifact_hash
    assert set(report.ir_hashes) == {"graph_ir", "schedule_ir", "tile_ir", "target_ir"}
    assert "2 ordered native packages" in report.target_decision["rocm_gfx1201"]
    assert all(p.descriptor.provenance.get("route") for p in fn.native_nvfp4_packages())
    warm = fn(codes=roots[0], scales=roots[1], globals_=roots[2],
              activation=roots[3], activation_scales=roots[4])
    np.testing.assert_array_equal(warm, expected)
    for a, b in zip(roots, original, strict=True):
        np.testing.assert_array_equal(a, b)
    restored = rt.RuntimeArtifact.from_json(artifact.to_json())
    replay = rt.launch(restored, roots)
    assert replay["ok"] and replay["execution_kind"] == "native_gpu", replay
    np.testing.assert_array_equal(replay["output"], expected)
    # Same pointers/shapes with changed bytes must never reuse expanded values.
    roots[0].fill(0)
    changed = fn(*roots)
    np.testing.assert_array_equal(changed, np.zeros_like(expected))
    np.testing.assert_array_equal(first, expected)
    # Fresh-process replay has neither compiler path nor packaging/oracle access.
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(artifact.to_json())
    inputs_path = tmp_path / "inputs.npz"
    np.savez(inputs_path, **{f"a{i}": a for i, a in enumerate(original)}, expected=expected)
    script = """
import os, sys
import numpy as np
from tessera import runtime as rt
from tessera.compiler import nvfp4_expansion, rocm_nvfp4_expansion_program, reference_typed_scaled_matmul
def forbidden(*a, **kw): raise AssertionError("replay invoked compiler/oracle")
nvfp4_expansion.reference_nvfp4_expand = forbidden
reference_typed_scaled_matmul.reference_typed_scaled_matmul = forbidden
rocm_nvfp4_expansion_program.package_native_nvfp4_expansion_program = forbidden
assert rt._rocm_live_arch() == "gfx1201"
artifact = rt.RuntimeArtifact.from_json(open(sys.argv[1]).read())
data = np.load(sys.argv[2])
# NPZ stores the original FP8 bit patterns as a byte-view dtype. Restore dtype.
import ml_dtypes
args = [data[f"a{i}"] for i in range(5)]
for i in (1,3): args[i] = args[i].view(ml_dtypes.float8_e4m3fn)
result = rt.launch(artifact, args)
assert result.get("ok") and result["execution_kind"] == "native_gpu", result
np.testing.assert_array_equal(result["output"], data["expected"])
"""
    env = dict(os.environ, TESSERA_OPT="/no/compiler", TESSERA_ROCM_OPT="/no/compiler")
    child = subprocess.run([sys.executable, "-c", script, str(artifact_path), str(inputs_path)],
                           env=env, text=True, capture_output=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr


def test_public_source_manifest_corruption_rejected_before_provider(monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler.rocm_nvfp4_source_frontend import source_program_from_manifest, package_traced_source
    from tessera.compiler.trace import trace, to_graph_ir_module
    roots, _ = inputs_and_oracle(19, 7, 48, (0, 1, 2, 3, 4), "f32")
    program = package_traced_source(to_graph_ir_module(trace(frontend_product, *roots), target="rocm_gfx1201"))
    manifest = program.manifest()
    manifest["native_program"]["stages"][0]["target_ir"] += "\n// forged"
    monkeypatch.setattr(rt, "_load_rocm_native_movement_runtime",
                        lambda: pytest.fail("corrupt program reached provider"))
    with pytest.raises(ValueError, match="identity differs"):
        source_program_from_manifest(manifest)


def test_parent_target_cannot_relabel_a_native_program(monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler.native_artifact import ArtifactContractError
    fn = ts.jit(target="rocm_gfx1201")(frontend_product)
    roots, _ = inputs_and_oracle(19, 7, 48, (0, 1, 2, 3, 4), "f32")
    fn(*roots)
    artifact = fn.runtime_artifact()
    artifact.metadata["target"] = "rocm_gfx1151"
    monkeypatch.setattr(rt, "_load_rocm_native_movement_runtime",
                        lambda: pytest.fail("foreign parent reached provider"))
    with pytest.raises(ArtifactContractError, match="exact gfx1201"):
        rt._execute_rocm_nvfp4_program_artifact(artifact, roots)
