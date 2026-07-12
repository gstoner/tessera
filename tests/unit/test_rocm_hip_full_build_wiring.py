"""Source-level guards for the full ROCm/HIP CMake configuration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
_FULL_TOOLCHAIN_GUARD = (
    "TESSERA_BUILD_ROCM_BACKEND AND NOT TESSERA_ENABLE_CUDA "
    "AND NOT TESSERA_ENABLE_HIP"
)


def test_hip_is_enabled_as_a_cmake_language() -> None:
    top = (ROOT / "CMakeLists.txt").read_text()
    hip_block = top.split("if(TESSERA_ENABLE_HIP)", 1)[1].split("endif()", 1)[0]
    assert "enable_language(HIP)" in hip_block
    assert "find_package(hip REQUIRED CONFIG)" in hip_block


def test_real_hip_build_is_not_classified_as_hardware_free() -> None:
    """HIP-on builds must retain full compiler components and examples."""
    for relative in (
        "CMakeLists.txt",
        "src/CMakeLists.txt",
        "tools/tessera-opt/CMakeLists.txt",
        "tools/tessera-translate/CMakeLists.txt",
    ):
        text = (ROOT / relative).read_text()
        assert _FULL_TOOLCHAIN_GUARD in text, relative


def test_real_hip_build_keeps_neighbors_solvers_and_tpp() -> None:
    source_tree = (ROOT / "src/CMakeLists.txt").read_text()
    assert "add_subdirectory(compiler/tessera_neighbors)" in source_tree
    assert "add_subdirectory(solvers)" in source_tree

    opt = (ROOT / "tools/tessera-opt/CMakeLists.txt").read_text()
    assert "TESSERA_HAVE_NEIGHBORS" in opt
    assert "TESSERA_HAVE_TPP" in opt


def test_nvidia_lit_site_loads_tests_and_llvm_tools() -> None:
    test_root = ROOT / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test"
    site = (test_root / "lit.site.cfg.py.in").read_text()
    assert "lit_config.load_config" in site
    assert "@TESSERA_LLVM_TOOLS_DIR@" in site

    config = (test_root / "lit.cfg.py").read_text()
    assert 'config.suffixes = [".mlir"]' in config
    assert 'config.environment["PATH"]' in config
