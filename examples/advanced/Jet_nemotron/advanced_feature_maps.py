from dataclasses import dataclass
from typing import Optional, Tuple

from tessera import Tensor, kernel, nn
from tessera.stdlib import dropout

@dataclass
class RFState:
    # Rolling window summaries and ring-buffer of block contributions
    U: Tensor   # (B,H,M)
    S: Tensor   # (B,H,M,Dh)
    # FIFO buffers of per-block contributions for subtraction when window moves
    U_blocks: list  # list[Tensor (B,H,M)]
    S_blocks: list  # list[Tensor (B,H,M,Dh)]
    max_blocks: int
    blk_size: int

def make_gaussian_proj(head_dim: int, m_features: int, *, seed: int=0) -> Tensor:
    # Draw Ω ~ N(0, I) for FAVOR+ random feature map
    return nn.random_normal((head_dim, m_features), seed=seed)

def phi_rf(x: Tensor, W: Tensor, *, normalize: bool=True) -> Tensor:
    # FAVOR+ style: φ(x) = exp(-||x||^2/2) * exp(xW)
    # Stabilize via subtracting max per token
    e = x @ W  # (..., M)
    if normalize:
        c = e.max(dim=-1, keepdim=True)
        e = e - c
    return (e.exp()) * (-0.5 * (x**2).sum(dim=-1, keepdim=True)).exp()

@kernel.autotune(space=dict(BM=[64,128], BN=[64,128], BD=[64,128], warps=[4,8], stages=[2,3], vector=[4,8]))
def lin_attn_rf_kernel(phi_q: Tensor["B","S","H","M"],
                       phi_k: Tensor["B","S","H","M"],
                       v: Tensor["B","S","H","Dh"], *, causal: bool) -> Tensor["B","S","H","Dh"]:
    T = tile.context()
    acc = tile.zeros((T.m, T.d), f32)
    for nblk in tile.range_n(v.shape, T.n, prefetch=2):
        Kb = tile.load(phi_k, nblk, cols=T.m, vector=T.vector)
        Vb = tile.load(v, nblk, cols=T.d, vector=T.vector)
        if causal:
            tile.mask_causal_block(Kb, tile.row_index(), tile.col_index(nblk))
        acc += tile.dot(tile.transpose(Kb), Vb)  # (M x Dh)
    Y = tile.dot(phi_q, acc)
    return Y

def linear_attention_rf(q: Tensor, k: Tensor, v: Tensor, *, Wq: Tensor, Wk: Tensor,
                        dropout_p: float, causal: bool,
                        state: Optional[RFState], window_blocks: Optional[int], block_size: int
                        ) -> Tuple[Tensor, Optional[RFState]]:
    phi_q = phi_rf(q, Wq)
    phi_k = phi_rf(k, Wk)
    if state is None:
        B,S,H,Dh = v.shape
        M = phi_q.shape[-1]
        U = v.zeros_like(shape=(B,H,M))
        Ssum = v.zeros_like(shape=(B,H,M,Dh))
        state = RFState(U=U, S=Ssum, U_blocks=[], S_blocks=[], max_blocks=window_blocks or 0, blk_size=block_size)
    # Compute contributions for this block
    # Split along sequence into blocks of size block_size
    outputs = []
    for s0 in range(0, q.shape[1], block_size):
        s1 = min(q.shape[1], s0 + block_size)
        phi_q_blk = phi_q[:, s0:s1]
        phi_k_blk = phi_k[:, s0:s1]
        v_blk     = v[:, s0:s1]
        U_add = phi_k_blk.sum(dim=1)            # (B,H,M)
        S_add = phi_k_blk.transpose(-2,-1) @ v_blk  # (B,H,M,Dh)
        # Update rolling window: push new block, pop old if beyond window
        state.U_blocks.append(U_add)
        state.S_blocks.append(S_add)
        state.U += U_add
        state.S += S_add
        if state.max_blocks and len(state.U_blocks) > state.max_blocks:
            U_old = state.U_blocks.pop(0)
            S_old = state.S_blocks.pop(0)
            state.U -= U_old
            state.S -= S_old
        # Compute outputs for this block using current summaries
        y_blk = phi_q_blk @ state.S
        outputs.append(y_blk)
    y = Tensor.concat(outputs, dim=1)
    return y, state

# --- Extensions: per-head projection reuse + on-device deterministic generation across shards ---
from typing import Optional
from tessera import effects, dist

def make_gaussian_proj_per_head(head_dim: int, m_features: int, H: int, *, seed: int=0):
    """Create per-head random feature projections W with shape (H, head_dim, m_features).
    Uses a global seed for determinism; callers can vary seed per-run.
    """
    WHs = []
    for h in range(H):
        W = make_gaussian_proj(head_dim, m_features, seed=seed + h*1315423911)  # distinct but deterministic per head
        WHs.append(W)
    return Tensor.stack(WHs, dim=0)  # (H, Dh, M)

def broadcast_seed_across_mesh(base_seed: int) -> int:
    """Derive a deterministic seed shared across shards (data/tensor/pipeline ranks).
    Tessera's dist layer can broadcast a single root value to all ranks.
    """
    # Pseudocode: get mesh, have rank 0 hold base_seed, broadcast to others
    if dist.is_initialized():
        root = 0
        seed_t = Tensor([base_seed], dtype="int32")
        seed_b = dist.broadcast(seed_t, src=root)
        return int(seed_b.item())
    return base_seed

def generate_rf_proj_on_device(head_dim: int, m_features: int, H: int, *, base_seed: int=1234):
    """On-device generation of per-head RF projections with deterministic seeding across shards.
    Seed is broadcast across the mesh so all shards see the same projections.
    """
    seed = broadcast_seed_across_mesh(base_seed)
    # Pseudocode using Tessera RNG effects (Philox)
    rng = effects.rng(kind="philox", seed=seed)
    WHs = []
    for h in range(H):
        # Draw normals on device; shape (Dh, M)
        W = rng.normal((head_dim, m_features))
        WHs.append(W)
    return Tensor.stack(WHs, dim=0)  # (H, Dh, M)
