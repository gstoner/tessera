"""Host-free NVIDIA linear-attention source contracts."""
from __future__ import annotations


def test_nvidia_variant_vjp_cuda_wrapper_releases_allocations():
    from tessera.compiler.emit.nvidia_cuda import _synthesize_linear_attn_variant_bwd_cuda
    source = _synthesize_linear_attn_variant_bwd_cuda()
    for device_buffer in ("g", "q", "k", "v", "d", "dq", "dk", "dv"):
        assert f"if({device_buffer})cudaFree({device_buffer})" in source


def test_decay_factor_is_a_running_product_not_a_per_key_inner_loop():
    """`prod(dec[n+1..m])` rebuilt per key is O(S) inside an O(S) key loop, so
    the decayed lane costs an extra S factor over the undecayed one. Forward
    folds it into the accumulator (Horner, ascending n); backward carries it
    down from n = m, because there the factor feeds per-key atomics into
    dv/dk and cannot be folded into an accumulator."""
    from tessera.compiler.emit.nvidia_cuda import (
        _synthesize_linear_attn_variant_bwd_cuda,
        _synthesize_linear_attn_variant_cuda,
    )
    fwd = _synthesize_linear_attn_variant_cuda()
    bwd = _synthesize_linear_attn_variant_bwd_cuda()
    assert "for(long u=n+1;u<=m;u++)" not in fwd
    assert "for(long u=n+1;u<=m;u++)" not in bwd
    assert "if(dec&&n)y*=dec[" in fwd                    # Horner recurrence
    assert "float fac=1;for(long n=m;n>=0;n--)" in bwd   # descending product
    assert "if(de)fac*=de[" in bwd
