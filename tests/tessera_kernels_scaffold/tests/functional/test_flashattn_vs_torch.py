import pytest, torch, torch.nn.functional as F
from tessera_kernels import flashattn, flashattn_forward, HAS_EXT

requires_cuda = pytest.mark.skipif(not (torch.cuda.is_available() and HAS_EXT), reason="CUDA/Ext required")

@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_flashattn_forward_matches_torch(dtype):
    torch.manual_seed(0)
    B,H,S,D = 1, 4, 64, 64
    Q = torch.randn(B,H,S,D, device="cuda", dtype=dtype)
    K = torch.randn(B,H,S,D, device="cuda", dtype=dtype)
    V = torch.randn(B,H,S,D, device="cuda", dtype=dtype)
    out = flashattn_forward(Q,K,V,mask=None)
    ref = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False)
    rtol = 5e-2 if dtype==torch.float16 else 1e-3
    atol = 5e-2 if dtype==torch.float16 else 1e-4
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)

@requires_cuda
def test_flashattn_backward_basic():
    torch.manual_seed(0)
    B,H,S,D = 1, 2, 16, 16
    Q = torch.randn(B,H,S,D, device="cuda", dtype=torch.float32, requires_grad=True)
    K = torch.randn(B,H,S,D, device="cuda", dtype=torch.float32, requires_grad=True)
    V = torch.randn(B,H,S,D, device="cuda", dtype=torch.float32, requires_grad=True)
    out = flashattn(Q,K,V)
    loss = out.sum()
    loss.backward()
    assert all(torch.isfinite(g).all() for g in (Q.grad, K.grad, V.grad))
