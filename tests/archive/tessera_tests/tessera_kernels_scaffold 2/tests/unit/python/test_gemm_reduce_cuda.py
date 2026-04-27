import pytest, torch
from tessera_kernels import gemm_fp16, reduce_sum, HAS_EXT

requires_cuda = pytest.mark.skipif(not (torch.cuda.is_available() and HAS_EXT), reason="CUDA/Ext required")

@requires_cuda
def test_gemm_fp16_matches_torch():
    torch.manual_seed(0)
    M,N,K = 128, 128, 128
    A = torch.randn(M,K, device="cuda", dtype=torch.float16)
    B = torch.randn(K,N, device="cuda", dtype=torch.float16)
    C = gemm_fp16(A,B)
    ref = (A.float() @ B.float())
    assert torch.allclose(C, ref, rtol=1e-2, atol=1e-2)

@requires_cuda
def test_reduce_sum_matches_torch():
    X = torch.randn(256, 2048, device="cuda", dtype=torch.float32)
    out = reduce_sum(X)
    ref = X.sum(dim=1)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)
