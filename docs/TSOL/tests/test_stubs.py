
# tests/test_stubs.py
# Minimal smoke test for tessera.ops type stubs.
# This doesn't execute real kernels—it's for import/type-check surface.

from tessera import ops as op  # type: ignore

def test_matmul_stub_surface() -> None:
    A = op.tensor((4, 8), dtype="bf16")
    B = op.tensor((8, 16), dtype="bf16")
    y = op.matmul(A, B, epilogue={"activation":"silu"})  # type: ignore
    # We can't assert runtime values; just ensure object presence
    assert y is not None

def test_flash_attention_stub_surface() -> None:
    q = op.tensor((2, 16, 64), dtype="bf16")
    k = op.tensor((2, 16, 64), dtype="bf16")
    v = op.tensor((2, 16, 64), dtype="bf16")
    y = op.flash_attention(q, k, v, params={"causal": True, "block_q": 128})  # type: ignore
    assert y is not None

def test_moe_stub_surface() -> None:
    x = op.tensor((8, 128), dtype="bf16")
    # Experts are opaque here; IDEs will still get signature hints
    y = op.moe(x, experts=[], router="topk", k=2,
               transport={"type":"nvshmem","multi_qp":True,"pack_dtype":"fp8_e4m3"},
               deterministic={"deterministic": True})  # type: ignore
    assert y is not None
