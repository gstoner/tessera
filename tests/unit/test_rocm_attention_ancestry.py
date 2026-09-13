"""ROCm attention packages consume replayed native contracts before compiling."""
from dataclasses import replace
import pytest
from tessera.compiler import rocm_native
from tessera.compiler.native_attention_contract import verify_attention_ancestry
from tessera.compiler.scheduled_attention import lower_scheduled_attention
from tessera.compiler.scheduled_attention_backward import lower_scheduled_attention_backward
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.unit.test_scheduled_attention_consumers import _module
from benchmarks.rocm.benchmark_rocm_attention_backward_program import _module as backward_module

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason="native compiler required")

@pytest.mark.parametrize("backward", [False, True])
def test_native_rocm_attention_ancestry(backward, monkeypatch):
    if backward:
        artifact = lower_scheduled_attention_backward(backward_module(1, 2, 2, 17, 19, 64), target="rocm_gfx1151")
        package = rocm_native.package_scheduled_attention_backward
    else:
        artifact = lower_scheduled_attention(_module(target="rocm"), target="rocm_gfx1151")
        package = rocm_native.package_scheduled_attention
    verify_attention_ancestry(artifact, target="rocm", architecture="gfx1151")
    monkeypatch.setattr(rocm_native, "_compile_native_tile_ir", lambda *a, **k: pytest.fail("compiled corrupt contract"))
    for changed in (replace(artifact, scale=0.75), replace(artifact, workgroup_size=1024),
                    replace(artifact, tile_ir=artifact.tile_ir.replace("gfx1151", "gfx1201"))):
        with pytest.raises(ValueError):
            package(changed, pipeline_name="tessera-lower-to-rocm")
