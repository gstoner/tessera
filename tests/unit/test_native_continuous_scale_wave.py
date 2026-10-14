"""Native member-specific continuous scale-adjoint wave scheduling."""
import json
import os
import pytest
from tests.unit.test_native_scaled_multiply_carrier import case
from tessera.compiler.native_scaled_program import package_native_scaled_vjp


pytestmark=pytest.mark.skipif(not os.environ.get("TESSERA_OPT"),reason="matching native compiler required")


@pytest.mark.parametrize("ta,tb", [(False,False),(True,False),(False,True),(True,True)])
def test_explicit_wave_selects_scale_adjoints_and_keeps_other_members_serial(ta,tb):
    _,graph,_,_=case("reverse",ta=ta,tb=tb)
    graph.module_attrs.update({"tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    package=package_native_scaled_vjp(
        graph.to_mlir(target="rocm_gfx1201",canonical=True),
        schedule="wave_per_scale_element")
    manifest=json.loads(package.program_json)
    for step,raw in zip(manifest["steps"],package.members_json,strict=True):
        member=json.loads(raw)
        if step.get("gradient_role") in {"lhs_scale","rhs_scale"}:
            assert member["scale_adjoint_schedule"]=="wave_per_scale_element"
            assert member["geometry"]==[manifest["buffers"][step["output"]]["elements"],1,1,32,1,1]
        elif step["operation"] in {"tensor.generate","tessera.scaled_matmul"}:
            assert member["scale_adjoint_schedule"]=="serial_per_scale_element"
            assert member["geometry"][3:]==[128,1,1]
    package.validate()


def test_continuous_wave_tile_requires_matching_scale_role():
    from tests.unit.test_native_floating_scaled_adjoint import floating_source
    from tests.unit.test_native_scaled_transpose_lowering import native
    prefix=("tessera-autodiff-paired{export-scaled-transpose=true "
            "select-scaled-transpose-member=0},"
            "tessera-graph-to-schedule{scale-transpose-wave=true},tessera-schedule-to-tile")
    tile=native(floating_source(),prefix)
    assert tile.returncode==0,tile.stderr
    role='tessera.autodiff.scale_adjoint = "lhs_scale"'
    assert tile.stdout.count(role)>=2
    # The first occurrence is the Tile wrapper; the GPU body remains sealed.
    before,after=tile.stdout.split(role,1)
    forged=before+role.replace("lhs_scale","lhs_matrix")+after
    refusal=native(forged,"tessera-rocm-executable{family=reduction input=tile output=target arch=gfx1201}")
    assert refusal.returncode!=0
    assert "role-bound four-f32" in refusal.stderr
