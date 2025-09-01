
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

from tessera import Tensor, kernel

FeatureMap = Literal["elu1","relu2","favor"]

@dataclass
class StreamingState:
    U: Tensor  # (B,H,M)
    S: Tensor  # (B,H,M,Dh)

def _phi(x: Tensor, kind: FeatureMap) -> Tensor:
    if kind == "elu1":
        return x.elu() + 1.0
    elif kind == "relu2":
        return (x.relu())**2 + 1e-6
    else:
        return (x * 0.70710678).exp()

@kernel.autotune(space=dict(BM=[64,128], BN=[64,128], BD=[64,128], warps=[4,8], stages=[2,3], vector=[4,8]))
def lin_attn_kernel(phi_q: Tensor["B","S","H","M"], phi_k: Tensor["B","S","H","M"],
                    v: Tensor["B","S","H","Dh"], *, causal: bool) -> Tensor["B","S","H","Dh"]:
    T = tile.context()
    acc = tile.zeros((T.m, T.d), f32)
    for nblk in tile.range_n(v.shape, T.n, prefetch=2):
        Kb = tile.load(phi_k, nblk, cols=T.m, vector=T.vector)
        Vb = tile.load(v, nblk, cols=T.d, vector=T.vector)
        if causal:
            tile.mask_causal_block(Kb, tile.row_index(), tile.col_index(nblk))
        acc += tile.dot(tile.transpose(Kb), Vb)
    Y = tile.dot(phi_q, acc)
    return Y

def lin_attn_forward(q: Tensor, k: Tensor, v: Tensor, *, feature_map: FeatureMap,
                     dropout_p: float, causal: bool, state: Optional[StreamingState],
                     streaming: bool) -> Tuple[Tensor, Optional[StreamingState]]:
    phi_q = _phi(q, feature_map)
    phi_k = _phi(k, feature_map)
    if not streaming:
        S = (phi_k.transpose(-2,-1) @ v)
        y = (phi_q @ S)
        return y, None
    else:
        if state is None:
            B,S,H,Dh = v.shape
            M = phi_k.shape[-1]
            U = v.zeros_like(shape=(B,H,M))
            Ssum = v.zeros_like(shape=(B,H,M,Dh))
            state = StreamingState(U=U, S=Ssum)
        U = state.U + phi_k.sum(dim=1)
        Ssum = state.S + (phi_k.transpose(-2,-1) @ v).sum(dim=1)
        y = (phi_q @ Ssum)
        return y, StreamingState(U=U, S=Ssum)
