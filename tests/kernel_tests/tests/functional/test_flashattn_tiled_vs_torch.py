import pytest, torch, torch.nn.functional as F
from tessera_kernels.tiled import flashattn_fwd_tiled, flashattn_bwd_tiled
from tessera_kernels import HAS_EXT

requires_cuda = pytest.mark.skipif(not (torch.cuda.is_available() and HAS_EXT), reason="CUDA/Ext required")

@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("causal", [False, True])
def test_tiled_forward_matches_torch(dtype, causal):
    torch.manual_seed(0)
    B,H,S,D = 1, 4, 64, 64
    Q = torch.randn(B,H,S,D, device="cuda", dtype=dtype)
    K = torch.randn(B,H,S,D, device="cuda", dtype=dtype)
    V = torch.randn(B,H,S,D, device="cuda", dtype=dtype)
    mask = None
    if causal:
        mask = torch.zeros(B,H,S,S, device="cuda", dtype=torch.float32)
        tri = torch.triu(torch.ones(S,S, device="cuda"), diagonal=1)
        mask[:] = mask + tri * (-1e9)
    out = flashattn_fwd_tiled(Q,K,V,mask=mask,dropout_mask=None,is_causal=causal)
    ref = F.scaled_dot_product_attention(Q,K,V,attn_mask=mask,dropout_p=0.0,is_causal=False)
    rtol = 5e-2 if dtype==torch.float16 else 1e-3
    atol = 5e-2 if dtype==torch.float16 else 1e-4
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)

@requires_cuda
def test_tiled_backward_matches_torch():
    torch.manual_seed(0)
    B,H,S,D = 1, 2, 32, 64
    dtype = torch.float32
    Q = torch.randn(B,H,S,D, device="cuda", dtype=dtype, requires_grad=True)
    K = torch.randn(B,H,S,D, device="cuda", dtype=dtype, requires_grad=True)
    V = torch.randn(B,H,S,D, device="cuda", dtype=dtype, requires_grad=True)
    mask = None
    ref = F.scaled_dot_product_attention(Q,K,V,attn_mask=mask,dropout_p=0.0,is_causal=False)
    loss_ref = ref.sum()
    qgr, kgr, vgr = torch.autograd.grad(loss_ref, (Q,K,V), retain_graph=True)
    dOut = torch.ones_like(ref)  # dLoss/dOut = 1
    dQ,dK,dV = flashattn_bwd_tiled(Q,K,V,dOut,mask=mask,dropout_mask=None,is_causal=False)
    assert torch.allclose(dQ, qgr, rtol=3e-2, atol=3e-2)
    assert torch.allclose(dK, kgr, rtol=3e-2, atol=3e-2)
    assert torch.allclose(dV, vgr, rtol=3e-2, atol=3e-2)
