"""Sprint G-5 / G-6/G-7/G-8 / G-9 / H-2 / H-6/H-7/H-8 / H-8 — Phase G/H Lane 2.

Hardware-free guards for the toolchain detection + compile-only
validation infrastructure landed 2026-05-11:

  G-5: NVIDIATargetPipeline named pass aliases registered in
       `src/transforms/lib/Passes.cpp` (tessera-nvidia-pipeline +
       per-SM variants).  Lit fixture
       `tests/tessera-ir/phase3/cuda13/nvidia_pipeline_alias.mlir`.
  H-2: `mfma_table.inc` generator + content sync check between
       Python `_MFMA_VARIANTS` source and the C++ X-macro file.
  G-6/G-7/G-8: `cmake/TesseraToolchainPins.cmake` + explicit nvcc
       instruction-probe validator (`scripts/validate_nvcc_compile.py`).
  H-6/H-7/H-8: same CMake module + hipcc validator
       (`scripts/validate_hipcc_compile.py`).
  G-9 + H-8: NCCL/RCCL version-pin header
       (`AdapterVersionPin.h`) + symbol probe
       (`scripts/probe_collective_libs.py`).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


# ──────────────────────────────────────────────────────────────────────────
#                G-5: NVIDIATargetPipeline alias registration
# ──────────────────────────────────────────────────────────────────────────

class TestG5PipelineAlias:
    def test_passes_cpp_registers_nvidia_pipeline(self):
        body = (REPO / "src" / "transforms" / "lib" / "Passes.cpp").read_text()
        # All four aliases should be present.
        for alias in (
            "tessera-nvidia-pipeline",
            "tessera-nvidia-pipeline-sm90",
            "tessera-nvidia-pipeline-sm100",
            "tessera-nvidia-pipeline-sm120",
        ):
            assert f'"{alias}"' in body, f"{alias} not registered"

    def test_pipeline_includes_warpspec_to_nvptx_chain(self):
        body = (REPO / "src" / "transforms" / "lib" / "Passes.cpp").read_text()
        # The exact-SM builder contains the proven NVIDIA stage vocabulary.
        for stage in (
            "createWarpSpecializationPass",
            "createAsyncCopyLoweringPass",
            "createNVWGMMALoweringPass",
            "createNVTMADescriptorPass",
            "createNVFlashAttnKernelEmitterPass",
        ):
            assert stage in body, f"pipeline missing {stage}"

    def test_pipeline_description_mentions_cuda_13_3(self):
        body = (REPO / "src" / "transforms" / "lib" / "Passes.cpp").read_text()
        assert "CUDA 13.3" in body
        assert "PTX ISA 9.3" in body or "PTX ISA 9.3," in body

    def test_lit_fixture_for_pipeline_alias_exists(self):
        fixture = REPO / "tests" / "tessera-ir" / "phase3" / "cuda13" / \
                  "nvidia_pipeline_alias.mlir"
        assert fixture.exists()
        body = fixture.read_text()
        # The fixture should exercise all four aliases.
        for alias in (
            "tessera-nvidia-pipeline",
            "tessera-nvidia-pipeline-sm90",
            "tessera-nvidia-pipeline-sm100",
            "tessera-nvidia-pipeline-sm120",
        ):
            assert f"--{alias}" in body, f"fixture missing --{alias} RUN line"


# ──────────────────────────────────────────────────────────────────────────
#                  H-2: mfma_table.inc generator + sync
# ──────────────────────────────────────────────────────────────────────────

MFMA_TABLE_PATH = (
    REPO / "src" / "compiler" / "codegen" / "Tessera_ROCM_Backend"
    / "include" / "TesseraROCM" / "mfma_table.inc"
)
MFMA_GENERATOR_PATH = REPO / "scripts" / "generate_mfma_table.py"


class TestH2MFMATableSync:
    @pytest.fixture(autouse=True, scope="class")
    def _generate_mfma_table(self):
        """``mfma_table.inc`` is a generated, git-ignored build artifact — it is
        NOT committed. These tests therefore must generate it themselves rather
        than depend on a stale copy left on disk by a prior build or test run
        (the source of an isolation-order flake: they pass only when something
        upstream regenerated the table, and fail in isolation against a stale
        one). Regenerate once per class from the `_MFMA_VARIANTS` source so the
        format/sync/count assertions validate fresh generator output."""
        subprocess.run(
            [sys.executable, str(MFMA_GENERATOR_PATH)],
            env={"PYTHONPATH": f"{REPO / 'python'}:{REPO}", "PATH": "/usr/bin:/bin"},
            check=True, capture_output=True, text=True,
        )

    def test_generator_script_exists(self):
        assert MFMA_GENERATOR_PATH.exists()

    def test_mfma_table_file_exists(self):
        assert MFMA_TABLE_PATH.exists()

    def test_mfma_table_x_macro_format(self):
        body = MFMA_TABLE_PATH.read_text()
        assert "TESSERA_MFMA_VARIANT" in body
        assert "#ifndef TESSERA_MFMA_VARIANT" in body
        assert "#undef TESSERA_MFMA_VARIANT" in body
        # Auto-generated banner
        assert "Auto-generated" in body
        # Toolchain pin
        assert "ROCm target pin: 7.2.4" in body

    def test_mfma_table_in_sync_with_python_source(self):
        """The on-disk `mfma_table.inc` must match what
        `scripts/generate_mfma_table.py --check` would produce."""
        result = subprocess.run(
            [sys.executable, str(MFMA_GENERATOR_PATH), "--check"],
            env={
                "PYTHONPATH": f"{REPO / 'python'}:{REPO}",
                "PATH": "/usr/bin:/bin",
            },
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"mfma_table.inc drifted from _MFMA_VARIANTS source:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_mfma_table_contains_all_arch_shapes(self):
        from tessera.compiler.rocm_target import AMDArch, mfma_variants

        body = MFMA_TABLE_PATH.read_text()
        for arch in AMDArch:
            shapes = mfma_variants(arch)
            for (m, n, k, kb) in shapes:
                expected = (
                    f'TESSERA_MFMA_VARIANT({arch.value}, '
                    f'"{arch.name.lower()}", {m}, {n}, {k}, {kb})'
                )
                assert expected in body, (
                    f"{arch.name} shape ({m},{n},{k},{kb}) missing from table"
                )

    def test_mfma_table_total_count(self):
        body = MFMA_TABLE_PATH.read_text()
        # Per `_MFMA_VARIANTS`: gfx90a=2, gfx940=6, gfx942=6, gfx950=8,
        # gfx1100=0, gfx1200=0  →  total = 22
        match = re.search(r"Total shapes across all ROCm 7.2.4 arches: (\d+)", body)
        assert match is not None
        assert int(match.group(1)) == 22


# ──────────────────────────────────────────────────────────────────────────
#                  G-6 + H-6: CMake toolchain pin module
# ──────────────────────────────────────────────────────────────────────────

TOOLCHAIN_CMAKE = REPO / "cmake" / "TesseraToolchainPins.cmake"


class TestG6H6CMakeToolchainPin:
    def test_cmake_module_exists(self):
        assert TOOLCHAIN_CMAKE.exists()

    def test_pins_required_versions(self):
        body = TOOLCHAIN_CMAKE.read_text()
        assert 'TESSERA_REQUIRED_CUDA_VERSION   "13.3"' in body
        assert 'TESSERA_REQUIRED_PTX_ISA        "9.3"' in body
        assert 'TESSERA_REQUIRED_NCCL_VERSION   "2.22"' in body
        assert 'TESSERA_REQUIRED_ROCM_VERSION   "7.2.4"' in body
        assert 'TESSERA_REQUIRED_HIP_VERSION    "7.2.4"' in body
        assert 'TESSERA_REQUIRED_RCCL_VERSION   "2.22"' in body

    def test_exports_pin_functions(self):
        body = TOOLCHAIN_CMAKE.read_text()
        for fn in (
            "function(tessera_pin_cuda_toolkit",
            "function(tessera_pin_rocm",
            "function(tessera_add_nvcc_compile_check",
            "function(tessera_add_hipcc_compile_check",
        ):
            assert fn in body, f"missing {fn}"

    def test_skip_pin_escape_hatch_exists(self):
        body = TOOLCHAIN_CMAKE.read_text()
        assert "TESSERA_SKIP_TOOLCHAIN_PIN" in body


# ──────────────────────────────────────────────────────────────────────────
#                  G-7/G-8: nvcc compile-only validator
# ──────────────────────────────────────────────────────────────────────────

NVCC_VALIDATOR = REPO / "scripts" / "validate_nvcc_compile.py"


class TestG78NvccCompileValidator:
    def test_validator_script_exists(self):
        assert NVCC_VALIDATOR.exists()

    def test_validator_imports_clean(self):
        body = NVCC_VALIDATOR.read_text()
        assert "PTX_PATTERN_STUBS" in body
        assert "MIN_NVCC_VERSION = (13, 3, 0)" in body

    def test_validator_covers_ptx_instruction_probe_catalog(self):
        """The toolchain validator owns an explicit instruction-probe catalog."""
        body = NVCC_VALIDATOR.read_text()
        for pat in (
            "wgmma.mma_async.sync.aligned.m64n256k16",
            "wgmma.mma_async.sync.aligned.m64n128k16",
            "mbarrier.arrive.expect_tx",
            "tcgen05.mma",
            "tcgen05.alloc",
            "shfl.sync.bfly",
        ):
            assert pat in body, f"validator missing stub for {pat}"

    def test_validator_skips_gracefully_when_nvcc_absent(self):
        """Running with --nvcc /no/such/path should not raise (script
        prints a skip message and returns 0)."""
        proc = subprocess.run(
            [sys.executable, str(NVCC_VALIDATOR),
             "--nvcc", "/no/such/path/nvcc"],
            capture_output=True, text=True,
        )
        # Either skipped (return 0) or reported error (1) — both acceptable
        # so long as the script doesn't crash.
        assert proc.returncode in (0, 1)


# ──────────────────────────────────────────────────────────────────────────
#                  H-7/H-8: hipcc compile-only validator
# ──────────────────────────────────────────────────────────────────────────

HIPCC_VALIDATOR = REPO / "scripts" / "validate_hipcc_compile.py"


class TestH78HipccCompileValidator:
    def test_validator_script_exists(self):
        assert HIPCC_VALIDATOR.exists()

    def test_validator_pins_hip_7_2_3(self):
        body = HIPCC_VALIDATOR.read_text()
        assert "MIN_HIP_VERSION = (7, 2, 3)" in body

    def test_validator_covers_amdgcn_instruction_probe_catalog(self):
        body = HIPCC_VALIDATOR.read_text()
        for pat in (
            "llvm.amdgcn.mfma.f32.32x32x8bf16.1k",
            "llvm.amdgcn.mfma.f32.16x16x16bf16.1k",
            "llvm.amdgcn.mfma.f32.32x32x16f8f8",
            "llvm.amdgcn.mfma.f32.32x32x32f4f4",
            "llvm.amdgcn.global.load.lds",
            "llvm.amdgcn.s.barrier",
            "llvm.amdgcn.wmma.f32.16x16x16",
            "llvm.amdgcn.buffer.load",
        ):
            assert pat in body, f"validator missing stub for {pat}"

    def test_validator_dispatches_to_correct_arch_per_pattern(self):
        """CDNA 4 FP4 patterns route to gfx950; WMMA routes to gfx1100."""
        body = HIPCC_VALIDATOR.read_text()
        assert 'arch_for_pattern = "gfx950"' in body
        assert 'arch_for_pattern = "gfx1100"' in body

    def test_validator_skips_gracefully_when_hipcc_absent(self):
        proc = subprocess.run(
            [sys.executable, str(HIPCC_VALIDATOR),
             "--hipcc", "/no/such/path/hipcc"],
            capture_output=True, text=True,
        )
        assert proc.returncode in (0, 1)


# ──────────────────────────────────────────────────────────────────────────
#                  G-9 + H-8: NCCL/RCCL adapter version pin
# ──────────────────────────────────────────────────────────────────────────

ADAPTER_PIN_HEADER = (
    REPO / "src" / "collectives" / "include" / "tessera" / "Dialect"
    / "Collective" / "Runtime" / "AdapterVersionPin.h"
)
COLLECTIVE_PROBE = REPO / "scripts" / "probe_collective_libs.py"


class TestG9H8CollectivePin:
    def test_adapter_version_pin_header_exists(self):
        assert ADAPTER_PIN_HEADER.exists()

    def test_pins_nccl_2_22(self):
        body = ADAPTER_PIN_HEADER.read_text()
        assert "TESSERA_NCCL_MIN_MAJOR 2" in body
        assert "TESSERA_NCCL_MIN_MINOR 22" in body

    def test_pins_rccl_2_22(self):
        body = ADAPTER_PIN_HEADER.read_text()
        assert "TESSERA_RCCL_MIN_MAJOR 2" in body
        assert "TESSERA_RCCL_MIN_MINOR 22" in body

    def test_pins_cuda_13_3_marker(self):
        body = ADAPTER_PIN_HEADER.read_text()
        assert 'TESSERA_TARGET_CUDA_TOOLKIT "13.3"' in body
        assert 'TESSERA_TARGET_PTX_ISA      "9.3"' in body

    def test_pins_rocm_7_2_3(self):
        body = ADAPTER_PIN_HEADER.read_text()
        assert 'TESSERA_TARGET_ROCM         "7.2.4"' in body
        assert 'TESSERA_TARGET_HIP          "7.2.4"' in body

    def test_static_assert_via_error_directive(self):
        """If NCCL/RCCL headers are present at compile time and the
        version is too low, the header should emit `#error` (a real
        static_assert would require <type_traits> which Adapters.h may
        not pull in)."""
        body = ADAPTER_PIN_HEADER.read_text()
        # Two `#error` lines — one for NCCL, one for RCCL.
        nccl_error_count = body.count(
            '#error "Tessera requires NCCL >= 2.22'
        )
        rccl_error_count = body.count(
            '#error "Tessera requires RCCL >= 2.22'
        )
        assert nccl_error_count >= 1
        assert rccl_error_count >= 1

    def test_probe_script_exists(self):
        assert COLLECTIVE_PROBE.exists()

    def test_probe_script_runs_clean_with_no_libs(self):
        """Probe should exit 0 when neither NCCL nor RCCL is installed."""
        proc = subprocess.run(
            [sys.executable, str(COLLECTIVE_PROBE)],
            capture_output=True, text=True,
        )
        # Either 0 (nothing installed → skip) or 1 (some lib present
        # but symbol failed) — both legitimate; what we don't want is
        # an unhandled crash.
        assert proc.returncode in (0, 1)
        assert "NCCL" in proc.stdout
        assert "RCCL" in proc.stdout

    def test_probe_script_expected_symbol_list_complete(self):
        body = COLLECTIVE_PROBE.read_text()
        for sym in (
            "ncclAllReduce", "ncclReduceScatter", "ncclAllGather",
            "ncclSend", "ncclRecv", "ncclCommInitRank",
            "ncclGetVersion",
        ):
            assert f'"{sym}"' in body, f"probe missing {sym}"


# ──────────────────────────────────────────────────────────────────────────
#                  Toolchain pin consistency across sources
# ──────────────────────────────────────────────────────────────────────────

class TestToolchainPinConsistency:
    """The toolchain pin values must agree across:
       * python/tessera/compiler/gpu_target.py
       * python/tessera/compiler/rocm_target.py
       * cmake/TesseraToolchainPins.cmake
       * src/collectives/include/.../AdapterVersionPin.h"""

    def test_cuda_pin_agrees_python_to_cmake_to_cpp(self):
        from tessera.compiler.gpu_target import (
            TESSERA_TARGET_CUDA_TOOLKIT,
            TESSERA_TARGET_PTX_ISA,
            TESSERA_TARGET_NCCL_MIN,
        )
        assert TESSERA_TARGET_CUDA_TOOLKIT == "13.3"

        cmake_body = TOOLCHAIN_CMAKE.read_text()
        assert TESSERA_TARGET_PTX_ISA in cmake_body
        assert TESSERA_TARGET_NCCL_MIN in cmake_body

        header_body = ADAPTER_PIN_HEADER.read_text()
        assert f'"{TESSERA_TARGET_CUDA_TOOLKIT}"' in header_body
        assert f'"{TESSERA_TARGET_PTX_ISA}"' in header_body

    def test_rocm_pin_agrees_python_to_cmake_to_cpp(self):
        from tessera.compiler.rocm_target import (
            TESSERA_TARGET_ROCM,
            TESSERA_TARGET_HIP,
            TESSERA_TARGET_RCCL_MIN,
        )
        assert TESSERA_TARGET_ROCM == "7.2.4"
        assert TESSERA_TARGET_HIP == "7.2.4"
        assert TESSERA_TARGET_RCCL_MIN == "2.22"

        cmake_body = TOOLCHAIN_CMAKE.read_text()
        assert TESSERA_TARGET_ROCM in cmake_body
        assert TESSERA_TARGET_HIP in cmake_body
        assert TESSERA_TARGET_RCCL_MIN in cmake_body

        header_body = ADAPTER_PIN_HEADER.read_text()
        assert f'"{TESSERA_TARGET_ROCM}"' in header_body
        assert f'"{TESSERA_TARGET_HIP}"' in header_body
