"""Source-level guards for the full ROCm/HIP CMake configuration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
_HARDWARE_FREE_GUARD = (
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
    top = (ROOT / "CMakeLists.txt").read_text()
    assert _HARDWARE_FREE_GUARD in top
    assert "if(TESSERA_FORCE_FULL_COMPILER_DRIVER)" in top
    assert "set(TESSERA_HARDWARE_FREE_TARGET_BUILD OFF)" in top
    for relative in (
        "src/CMakeLists.txt",
        "tools/tessera-opt/CMakeLists.txt",
        "tools/tessera-translate/CMakeLists.txt",
    ):
        assert "TESSERA_HARDWARE_FREE_TARGET_BUILD" in (
            ROOT / relative
        ).read_text(), relative


def test_ci_lit_lane_requests_full_portable_target_matrix() -> None:
    workflow = (ROOT / ".github/workflows/validate.yml").read_text()
    for option in (
        "-DTESSERA_BUILD_APPLE_BACKEND=ON",
        "-DTESSERA_BUILD_X86_BACKEND=ON",
        "-DTESSERA_BUILD_NVIDIA_BACKEND=ON",
        "-DTESSERA_BUILD_ROCM_BACKEND=ON",
        "-DTESSERA_FORCE_FULL_COMPILER_DRIVER=ON",
    ):
        assert option in workflow


def test_real_hip_build_keeps_neighbors_solvers_and_tpp() -> None:
    source_tree = (ROOT / "src/CMakeLists.txt").read_text()
    assert "add_subdirectory(compiler/tessera_neighbors)" in source_tree
    assert "add_subdirectory(solvers)" in source_tree

    opt = (ROOT / "tools/tessera-opt/CMakeLists.txt").read_text()
    assert "TESSERA_HAVE_NEIGHBORS" in opt
    assert "TESSERA_HAVE_TPP" in opt


def test_rocm_serialization_registration_tracks_linked_libraries() -> None:
    cmake = (ROOT / "tools/tessera-opt/CMakeLists.txt").read_text()
    driver = (ROOT / "tools/tessera-opt/tessera-opt.cpp").read_text()
    assert "TESSERA_HAVE_ROCM_SERIALIZATION" in cmake
    assert (
        "#ifdef TESSERA_HAVE_ROCM_SERIALIZATION\n"
        "  // Stage L3: LLVM-IR translations"
    ) in driver


def test_lean_rocm_driver_excludes_ambient_fa4_targets() -> None:
    """Default-source FA4 targets must not make backend-only lean builds
    self-conflicting, while full HIP builds continue to link them."""
    opt = (ROOT / "tools/tessera-opt/CMakeLists.txt").read_text()
    assert (
        "if(TARGET TesseraAttnDialect AND NOT "
        "TESSERA_OPT_LEAN_ARTIFACT_DRIVER)"
    ) in opt
    assert "tessera_opt_feature(fa4-attn TESSERA_HAVE_FA4_ATTN)" in opt
    # The tessera.queue dialect (fa4-queue feature) was deleted 2026-08-10
    # (Decisions #29/#31); assert it stays gone from the driver wiring.
    assert "TesseraQueueDialect" not in opt
    assert "fa4-queue" not in opt.replace(
        "(and, before its 2026-08-10 deletion, fa4-queue)", ""
    )


def test_nvidia_lit_site_loads_tests_and_llvm_tools() -> None:
    test_root = ROOT / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test"
    site = (test_root / "lit.site.cfg.py.in").read_text()
    assert "lit_config.load_config" in site
    assert "@TESSERA_LLVM_TOOLS_DIR@" in site

    config = (test_root / "lit.cfg.py").read_text()
    assert 'config.suffixes = [".mlir"]' in config
    assert 'config.environment["PATH"]' in config
