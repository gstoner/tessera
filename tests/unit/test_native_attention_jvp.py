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


@pytest.mark.parametrize('indices,active',[('0',(True,False,False)),('1',(False,True,False)),('0, 1, 2',(True,True,True))])
def test_automatic_product_controls_physical_tangent_slots(monkeypatch,indices,active):
    from tessera.compiler.native_attention_jvp import materialize_generated
    from test_native_attention_ad_products import source as graph_source
    tool=find_tessera_opt()
    if tool is None:
        pytest.skip('requires native compiler')
    monkeypatch.setattr('tessera.compiler.native_attention_jvp.build_native_gpu_storage',lambda text,**kw:text)
    ir=materialize_generated(graph_source('forward',', tessera.autodiff.wrt_indices = ['+indices+']'),
                            (1,2,1,4,6,8,8),8**-.5,True,compiler=tool,llvm_bin='/unused')
    for name,enabled in zip(('dq','dk','dv'),active,strict=True):
        assert (f'%{name}v = llvm.load' in ir)==enabled
    assert 'tessera.autodiff.generated_jvp' in ir
