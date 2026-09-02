"""Regression guards for the retained NVIDIA PTX GEMM staging region."""

from pathlib import Path


_SOURCE = Path(
    "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/"
    "tessera_nvidia_ptx_launch.cpp"
)


def _body(source: str, start: str, end: str) -> str:
    return source[source.index(start):source.index(end, source.index(start))]


def test_gemm_host_buffer_handlers_share_one_retained_staging_arena():
    source = _SOURCE.read_text(encoding="utf-8")

    assert "struct StagingArena" in source
    assert "bool stagingPointersLocked" in source
    assert "CUdeviceptr replacement" in source
    assert "allocation == CUDA_ERROR_OUT_OF_MEMORY && previous" in source
    assert "g_staging = {};" in source
    assert "allocation = cuMemAlloc(&replacement, total);" in source

    mma = _body(source, "int invokeMma(", "// Launch the general aligned")
    gemm = _body(source, "int invokeMmaGemm16(", "// Compiler-owned launch-level")
    fused = _body(source, "int invokeFusedMatmul16(", "// Compiler-owned stable row-softmax")
    for body in (mma, gemm, fused):
        assert "stagingPointersLocked" in body
        assert "cuMemAlloc" not in body
        assert "cuMemFree" not in body
