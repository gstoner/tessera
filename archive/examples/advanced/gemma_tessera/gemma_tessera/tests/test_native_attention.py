import torch, pytest
from tessera_gemma.kernels.native_attention_tessera import native_flash_attention, _repeat_kv

@pytest.mark.parametrize("B,T,H,Hk,D", [(1,64,8,2,64), (2,128,12,4,64)])
def test_native_attention_matches_reference(B,T,H,Hk,D):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(B,T,H,D, device=device, dtype=torch.float16)
    k = torch.randn(B,T,Hk,D, device=device, dtype=torch.float16)
    v = torch.randn(B,T,Hk,D, device=device, dtype=torch.float16)
    k2, v2 = _repeat_kv(k, v, H, Hk)

    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1,2), k2.transpose(1,2), v2.transpose(1,2),
        attn_mask=None, dropout_p=0.0, is_causal=True, scale=None
    ).transpose(1,2)

    out = native_flash_attention(q, k2, v2, causal=True, dropout_p=0.0, block_size=128)
    atol = 2e-2 if device=='cuda' else 3e-2
    rtol = 2e-2
    assert torch.allclose(out, ref, atol=atol, rtol=rtol)
