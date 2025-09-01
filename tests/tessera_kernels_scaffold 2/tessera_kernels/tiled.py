import torch
from . import nvtx_range, HAS_EXT
try:
    import tessera_ext as _C
except Exception:
    _C = None

def flashattn_fwd_tiled(Q,K,V,mask=None,dropout_mask=None,scale=None,dropout_p=0.0,is_causal=False):
    assert HAS_EXT, "Extension not built"
    B,H,S,D = Q.shape
    if scale is None: scale = 1.0 / (D ** 0.5)
    Out = torch.zeros_like(Q)
    with nvtx_range("tessera.flashattn_fwd_tiled"):
        _C.flashattn_fwd_tiled(Q,K,V,mask,dropout_mask,float(scale),float(dropout_p),bool(is_causal),Out)
    return Out

def flashattn_bwd_tiled(Q,K,V,dOut,mask=None,dropout_mask=None,scale=None,dropout_p=0.0,is_causal=False):
    assert HAS_EXT, "Extension not built"
    B,H,S,D = Q.shape
    if scale is None: scale = 1.0 / (D ** 0.5)
    dQ = torch.empty_like(Q); dK = torch.empty_like(K); dV = torch.empty_like(V)
    with nvtx_range("tessera.flashattn_bwd_tiled"):
        _C.flashattn_bwd_tiled(Q,K,V,dOut,mask,dropout_mask,float(scale),float(dropout_p),bool(is_causal),dQ,dK,dV)
    return dQ,dK,dV
