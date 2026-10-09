"""Native SSA-based gfx1201 scale-adjoint recipe selection and replay."""
import json
import pytest
from tests.unit.test_native_scaled_transpose_export import source
from tests.unit.test_native_scaled_transpose_lowering import native
from tests.unit.test_native_floating_scaled_adjoint import floating_source
from tessera.compiler.native_scaled_program import package_native_scaled_vjp


def prefix(member=0):
    return ("tessera-autodiff-paired{export-scaled-transpose=true "
            f"select-scaled-transpose-member={member}" +
            "},tessera-graph-to-schedule{scale-transpose-auto=true}")


@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("nk",[False,True])
@pytest.mark.parametrize("member",[0,1])
def test_auto_selects_wave_for_low_cardinality_wide_contributions(policy,nk,member):
    graph=source(policy,nk=nk,shape=(1,2,7,19,256))
    result=native(graph,prefix(member)+",tessera-schedule-to-tile")
    assert result.returncode==0,result.stderr
    assert 'algorithm = "wave_per_scale_element"' in result.stdout
    assert "gpu.shuffle xor" in result.stdout


@pytest.mark.parametrize("columns",[1,3])
@pytest.mark.parametrize("member",[0,1])
def test_auto_keeps_serial_for_narrow_contribution_span(columns,member):
    result=native(source("independent_rhs",shape=(1,2,7,columns,256)),
                  prefix(member)+",tessera-schedule-to-tile")
    assert result.returncode==0,result.stderr
    assert 'algorithm = "serial_per_scale_element"' in result.stdout
    assert "gpu.shuffle" not in result.stdout


def test_auto_selects_per_member_not_from_whole_program_shape():
    graph=source("independent_rhs",shape=(1,2,129,19,256))
    left=native(graph,prefix(0))
    right=native(graph,prefix(1))
    assert left.returncode==right.returncode==0,left.stderr+right.stderr
    assert 'algorithm = "serial_per_scale_element"' in left.stdout
    assert 'algorithm = "wave_per_scale_element"' in right.stdout


def test_auto_uses_actual_rhs_scale_block_not_logical_n():
    graph=source("independent_rhs",shape=(1,2,7,19,256))
    graph=graph.replace("block = [128, 128]","block = [1, 128]")
    graph=graph.replace("1x2x2x1xf32","1x2x2x19xf32")
    left=native(graph,prefix(0))
    right=native(graph,prefix(1))
    assert left.returncode==right.returncode==0,left.stderr+right.stderr
    assert 'algorithm = "wave_per_scale_element"' in left.stdout
    assert 'algorithm = "serial_per_scale_element"' in right.stdout


@pytest.mark.parametrize("member",range(4))
def test_auto_preserves_continuous_adjoint_serial_admission(member):
    result=native(floating_source(),prefix(member)+",tessera-schedule-to-tile")
    assert result.returncode==0,result.stderr
    assert 'algorithm = "serial_per_scale_element"' in result.stdout
    assert "gpu.shuffle" not in result.stdout


def test_conflicting_recipe_requests_refuse_before_schedule_artifact():
    pipeline=prefix().replace("scale-transpose-auto=true",
                             "scale-transpose-auto=true scale-transpose-wave=true")
    result=native(source("independent_rhs"),pipeline)
    assert result.returncode!=0
    assert "both explicit wave and automatic" in result.stderr


def test_auto_schedule_identity_and_member_geometry_are_replay_bound():
    graph=source("independent_rhs",roles=(3,2),shape=(1,2,7,19,256))
    scheduled=native(graph,prefix())
    assert scheduled.returncode==0,scheduled.stderr
    corrupted=scheduled.stdout.replace("wave_per_scale_element","serial_per_scale_element")
    refusal=native(corrupted,"tessera-schedule-to-tile")
    assert refusal.returncode!=0
    assert "Schedule differs" in refusal.stderr
    package=package_native_scaled_vjp(graph,schedule="auto")
    assert json.loads(package.program_json)["gradient_roles"]==[3,2]
    for raw in package.members_json:
        member=json.loads(raw)
        assert member["scale_adjoint_schedule"]=="wave_per_scale_element"
        assert member["geometry"][3:]==[32,1,1]


def test_public_auto_request_is_distinct_from_explicit_recipes(monkeypatch):
    from tessera.compiler import native_vjp_plugins as plugins
    monkeypatch.setenv("TESSERA_ROCM_SCALE_VJP_SCHEDULE","auto")
    assert plugins._scaled_transpose_schedule()=="auto"
