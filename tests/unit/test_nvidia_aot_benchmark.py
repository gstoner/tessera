import hashlib
import json
from pathlib import Path


def test_nvidia_aot_benchmark_has_nonce_and_separate_timing_domains() -> None:
    source = Path("benchmarks/nvidia/benchmark_aot_vs_jit.cu").read_text()
    assert "tessera_aot_probe_" in source
    assert "compile_nvrtc" in source
    assert "aot_offline_build_ms" in source
    assert "jit_compile_load_launch_ms" in source
    assert "cuModuleLoadData" in source


def test_sm120_f16_aot_product_artifact_is_versioned_and_single_sourced() -> None:
    root = Path("src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda")
    source = root / "aot/tessera_nvidia_mma_f16_sm120_v1.cu"
    manifest = json.loads(
        (root / "aot/tessera_nvidia_mma_f16_sm120_v1.json").read_text())
    cmake = (root / "CMakeLists.txt").read_text()
    runtime = (root / "tessera_nvidia_gemm.cpp").read_text()

    assert manifest == {
        "schema": "tessera.nvidia.aot.package.v1",
        "artifact_id": "tessera_nvidia_mma_f16_sm120",
        "artifact_version": 1,
        "abi_version": 1,
        "formats": ["fatbin", "cubin"],
        "target": "nvidia_sm120",
        "architecture": "sm_120a",
        "compute_capability": [12, 0],
        "minimum_cuda_driver": 13000,
        "entry": "gemm",
        "physical_abi": {
            "inputs": ["A:u16[M,K]", "B:u16[K,N]"],
            "output": "D:f32[M,N]",
            "arguments": ["A", "B", "D", "M", "N", "K"],
        },
        "source_identity": "generated from the canonical .cu file at configure time",
        "fallback": "NVRTC compiles the identical canonical .cu source",
    }
    assert "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" in source.read_text()
    assert "tessera_aot_source_sha256" in source.read_text()
    assert "-arch=sm_120a -cubin" in cmake
    assert "-arch=sm_120a -fatbin" in cmake
    assert "file(SHA256" in cmake
    assert "kSrcF16Aot" in runtime
    assert "sm120F16AotCompatible" in runtime
    assert "TESSERA_NVIDIA_AOT_MODE" in runtime
    assert "TESSERA_NVIDIA_AOT_ARTIFACT_DIR" in runtime
    assert "TESSERA_NVIDIA_AOT_ARTIFACT" in runtime
    assert "nvrtc_missing" in runtime
    assert "nvrtc_stale" in runtime
    assert "moduleMetadataMatches" in runtime
    assert "tessera_nvidia_mma_gemm_f16_aot_cache_key" in runtime
    assert len(hashlib.sha256(source.read_bytes()).hexdigest()) == 64
