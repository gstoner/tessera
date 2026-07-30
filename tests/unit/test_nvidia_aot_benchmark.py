from pathlib import Path


def test_nvidia_aot_benchmark_has_nonce_and_separate_timing_domains() -> None:
    source = Path("benchmarks/nvidia/benchmark_aot_vs_jit.cu").read_text()
    assert "tessera_aot_probe_" in source
    assert "compile_nvrtc" in source
    assert "aot_offline_build_ms" in source
    assert "jit_compile_load_launch_ms" in source
    assert "cuModuleLoadData" in source
