"""Native packed image projection retains physical policy and edge categories."""
import pytest
from tessera.compiler import rocm_native as native
from tessera.compiler.rocm_mxfp4_packed_folded import author_packed_folded_shape_graph
from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt

def projected(m,n,k):
    tool=find_tessera_opt()
    if tool is None:pytest.skip("matching native compiler required")
    schedule=run_tessera_opt(tool,author_packed_folded_shape_graph(m,n,k),"--tessera-graph-to-schedule")
    tile=run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
    target=run_tessera_opt(tool,tile,"--lower-tile-to-rocm=arch=gfx1201")
    image=native._shape_free_target_ir(target,family="folded_matmul",
                                       directive="tessera_rocm.scaled_wmma_gemm")
    return target,image

@pytest.mark.parametrize("pair",[
    ((128,32,256),(200,80,256)),
    ((256,64,256),(512,128,256)),
])
def test_packed_identity_reuses_runtime_mn(pair):
    left,right=(projected(*shape) for shape in pair)
    assert left[0]!=right[0]
    assert left[1]==right[1]
    assert "runtime_mn" in left[1]
    assert "packed_folded_prefill_v1" in left[1]

@pytest.mark.parametrize("other",[(256,32,256),(128,64,256),(128,32,512)])
def test_packed_identity_retains_edge_categories_and_k(other):
    assert projected(128,32,256)[1]!=projected(*other)[1]
