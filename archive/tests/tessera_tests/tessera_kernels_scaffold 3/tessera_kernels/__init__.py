import torch
import torch.nn.functional as F

try:
    import tessera_ext as _C
    HAS_EXT = True
except Exception:
    _C = None
    HAS_EXT = False

class nvtx_range:
    def __init__(self, name): self.name = name
    def __enter__(self):
        try: torch.cuda.nvtx.range_push(self.name)
        except Exception: pass
    def __exit__(self, exc_type, exc, tb):
        try: torch.cuda.nvtx.range_pop()
        except Exception: pass

def gemm_fp16(A: torch.Tensor, B: torch.Tensor, alpha=1.0, beta=0.0):
    assert HAS_EXT, "Extension not built"
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.zeros((M, N), device=A.device, dtype=torch.float32)
    with nvtx_range("tessera.gemm_wmma"):
        _C.gemm_wmma(A, B, C, float(alpha), float(beta))
    return C

def reduce_sum(X: torch.Tensor):
    assert HAS_EXT, "Extension not built"
    assert X.is_cuda and X.dtype == torch.float32 and X.ndim == 2
    out = torch.empty((X.shape[0],), device=X.device, dtype=torch.float32)
    with nvtx_range("tessera.reduce_sum"):
        _C.reduce_tile_sum(X, out)
    return out

def flashattn_forward(Q, K, V, mask=None, scale=None):
    assert HAS_EXT, "Extension not built"
    assert Q.shape == K.shape == V.shape
    B, H, S, D = Q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    Out = torch.empty_like(Q)
    with nvtx_range("tessera.flashattn_fwd"):
        _C.flashattn_naive_fwd(Q, K, V, mask, float(scale), Out)
    return Out

class FlashAttnAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, mask=None, scale=None):
        out = flashattn_forward(Q, K, V, mask, scale)
        ctx.save_for_backward(Q, K, V, mask if mask is not None else torch.tensor([]))
        ctx.scale = 1.0 / (Q.shape[-1] ** 0.5) if scale is None else scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, mask = ctx.saved_tensors
        mask = None if mask.numel() == 0 else mask
        out_ref = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=0.0, is_causal=False)
        qg, kg, vg = torch.autograd.grad(out_ref, (Q, K, V), grad_out, retain_graph=False, allow_unused=False)
        return qg, kg, vg, None, None

def flashattn(Q, K, V, mask=None, scale=None):
    return FlashAttnAutograd.apply(Q, K, V, mask, scale)
