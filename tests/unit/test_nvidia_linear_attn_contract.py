"""Host-free NVIDIA linear-attention source contracts."""
from __future__ import annotations


def test_nvidia_variant_vjp_cuda_wrapper_releases_allocations():
    from tessera.compiler.emit.nvidia_cuda import _synthesize_linear_attn_variant_bwd_cuda
    source = _synthesize_linear_attn_variant_bwd_cuda()
    for device_buffer in ("g", "q", "k", "v", "d", "dq", "dk", "dv"):
        assert f"if({device_buffer})cudaFree({device_buffer})" in source
