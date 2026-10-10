"""Native recipe and policy checks; numerical execution belongs to CUDA proof."""
from pathlib import Path
import pytest
from tessera.compiler.native_attention_jvp import source
from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt


@pytest.mark.parametrize('causal',[False,True])
def test_score_product_uses_checked_native_arena(causal):
    tool=find_tessera_opt()
    if tool is None:
        pytest.skip('requires native compiler')
    ir=source((1,2,1,3,129,4,3),.5,causal)
    assert 'tensor<' not in ir
    verified=run_tessera_opt(Path(tool),ir,'--tessera-tile-buffer-reuse')
    arena=run_tessera_opt(Path(tool),verified,'--tessera-tile-buffer-arena')
    assert 'tile.dynamic_shared_size' in arena
    assert '__tessera_shared_bytes_attention_jvp_saved_lse_jvp' in arena
    assert 'tessera.attention_checkpoint_identity' in arena


@pytest.mark.parametrize('dims,scale,causal',[
    ((1,3,2,3,5,4,3),.5,False),((1,2,1,3,5,4,3),True,False),
    ((1,2,1,3,5,4,3),.5,1),((1,2,1,0,5,4,3),.5,False),
    ((1,2,1,3,5,4,3),1e100,False),((1,2,1,3,5,4,3),1e-100,False)])
def test_invalid_score_product_policy_refuses(dims,scale,causal):
    with pytest.raises(ValueError):
        source(dims,scale,causal)


@pytest.mark.parametrize('indices,active',[('0',(True,False,False)),('1',(False,True,False)),('2',(False,False,True)),('0, 1, 2',(True,True,True))])
def test_automatic_product_controls_physical_tangent_slots(monkeypatch,indices,active):
    from tessera.compiler.native_attention_jvp import materialize_generated
    from test_native_attention_ad_products import source as graph_source
    tool=find_tessera_opt()
    if tool is None:
        pytest.skip('requires native compiler')
    monkeypatch.setattr('tessera.compiler.native_attention_jvp.build_native_gpu_storage',lambda text,**kw:text)
    ir=materialize_generated(graph_source('forward',', tessera.autodiff.wrt_indices = ['+indices+']'),
                            (1,2,1,4,6,8,8),8**-.5,True,compiler=tool,llvm_bin='/unused')
    expected="active = ["+", ".join(str(x).lower() for x in active)+"]"
    assert expected in ir
    # Q/K/V, saved LSE/output and only the active tangent slots load.
    scores_active=active[0] or active[1]
    assert ir.count("llvm.load")==(5 if scores_active else 3)+sum(active)
    if not scores_active:
        assert "cooperative_saved_lse_value_linear_v1" in ir
        assert ir.count("tile.alloc_shared")==1
    assert "tessera.attention_jvp_schedule_hash" in ir
    assert 'tessera.autodiff.generated_jvp' in ir


def _scheduled_product():
    from test_native_attention_ad_products import source as graph_source
    tool=find_tessera_opt()
    if tool is None:
        pytest.skip('requires native compiler')
    graph=run_tessera_opt(Path(tool),graph_source('forward'),
                         '--tessera-autodiff-forward=export-attention-jvp')
    graph=graph.replace('module attributes {',
        'module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120", ',1)
    return Path(tool),run_tessera_opt(Path(tool),graph,'--tessera-graph-to-schedule')


@pytest.mark.parametrize('old,new',[
    ('active = [true, true, true]','active = [false, true, true]'),
    ('argument_roles = array<i64: 0, 1, 2, 3, 4, 5>',
     'argument_roles = array<i64: 1, 0, 2, 3, 4, 5>'),
    ('algorithm = "cooperative_saved_lse_moments_v1"','algorithm = "forged"'),
])
def test_schedule_replay_seals_tangent_policy_and_roles(old,new):
    tool,schedule=_scheduled_product()
    assert old in schedule
    with pytest.raises(RuntimeError):
        run_tessera_opt(tool,schedule.replace(old,new),'--tessera-schedule-to-tile')


def test_native_product_does_not_discard_unrelated_function():
    tool,schedule=_scheduled_product()
    changed=schedule.rstrip()[:-1]+'  func.func @unrelated() { return }\n}\n'
    with pytest.raises(RuntimeError):
        run_tessera_opt(tool,changed,'--tessera-schedule-to-tile')


def test_native_product_preserves_selected_return_mapping():
    tool,schedule=_scheduled_product()
    import re
    returns = re.findall(r"return (%[A-Za-z0-9_.$]+), (%[A-Za-z0-9_.$]+) :", schedule)
    assert len(returns) == 1
    primal, tangent = returns[0]
    assert primal != tangent
    changed=schedule.replace(f"return {primal}, {tangent} :", f"return {primal}, %arg3 :")
    with pytest.raises(RuntimeError,match='must return its selected tangent'):
        run_tessera_opt(tool,changed,'--tessera-schedule-to-tile')

def test_native_jvp_export_does_not_admit_aliased_primal_roles():
    from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("requires native compiler")
    text='''module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
      func.func @aliased(%q: tensor<1x1x3x4xf32>, %k: tensor<1x1x3x4xf32>, %v: tensor<1x1x3x4xf32>) -> tensor<1x1x3x4xf32>
        attributes {tessera.autodiff = "forward", tessera.autodiff.wrt_indices = [0]} {
        %o = "tessera.flash_attn"(%q, %k, %q) {causal = false, head_dim = 4 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0>, tessera.effect_kind = "pure"}
          : (tensor<1x1x3x4xf32>, tensor<1x1x3x4xf32>, tensor<1x1x3x4xf32>) -> tensor<1x1x3x4xf32>
        return %o : tensor<1x1x3x4xf32>
      }
    }'''
    with pytest.raises(RuntimeError,match="direct Q/K/V"):
        run_tessera_opt(Path(tool),text,"--tessera-autodiff-forward=export-attention-jvp")

@pytest.mark.parametrize("bounds", [
    '"ignored"',
    "array<i64: 1, 2, 1>",
    "array<i64: 1, 2, 1, 8, 12, 8, 8>",
])
def test_jvp_schedule_replay_checks_checkpoint_capacity_policy(bounds):
    # A static tangent generation must not ignore a module capacity override
    # that its paired checkpoint serializer rejects.
    tool, schedule = _scheduled_product()
    changed = schedule.replace(
        "module attributes {",
        "module attributes {tessera.attention_shape_bounds = " + bounds + ", ",
        1,
    )
    assert changed != schedule
    with pytest.raises(RuntimeError, match="bounds|sequences"):
        run_tessera_opt(tool, changed, "--tessera-schedule-to-tile")
