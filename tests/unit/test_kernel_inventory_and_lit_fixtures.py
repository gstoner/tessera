"""Sprint G-2 + G-3 + G-4 + H-3 + H-4 — Kernel inventory + schema + lit fixtures.

Hardware-free guards for the per-target kernel pre-work:

  G-2: `docs/backends/nvidia/kernel-inventory.md` — enumerates every
       planned NVIDIA fused kernel under CUDA 13.3.
  G-3: `BackendKernelEntry` schema extended with `cuda_arch_min`,
       `nvcc_version_min`, `wgmma_shape`, `cluster_size`, `mfma_shape`,
       `hipcc_version_min`, `expected_mfu`, `roofline_target`.
  G-4: NVIDIA compiler contracts are split by proof layer: core named-pipeline
       coverage in `tests/tessera-ir/phase3/cuda13/nvidia_pipeline_alias.mlir`,
       and typed Tile→Target→NVVM coverage in the NVIDIA backend lit suite.
  H-3: `docs/backends/rocm/kernel-inventory.md` — enumerates every planned
       ROCm fused kernel under ROCm 7.2.4.
  H-4: ROCm compiler contracts live in the ROCm backend lit suite, including
       architecture-keyed MFMA/WMMA selection and Target→ROCDL conversion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TESSERA_IR_FIXTURES = REPO / "tests" / "tessera-ir"
NVIDIA_FIXTURES = (
    REPO / "src" / "compiler" / "codegen" / "tessera_gpu_backend_NVIDIA"
    / "test" / "nvidia"
)
ROCM_FIXTURES = (
    REPO / "src" / "compiler" / "codegen" / "Tessera_ROCM_Backend"
    / "test" / "rocm"
)


# ──────────────────────────────────────────────────────────────────────────
#                G-3: BackendKernelEntry schema extension
# ──────────────────────────────────────────────────────────────────────────

class TestG3SchemaExtension:
    def test_cuda_arch_min_field(self):
        from tessera.compiler.backend_manifest import BackendKernelEntry
        e = BackendKernelEntry(
            target="nvidia_sm90", status="artifact_only",
            dtypes=("bf16",), cuda_arch_min="sm_90a",
        )
        assert e.cuda_arch_min == "sm_90a"
        d = e.as_dict()
        assert d["cuda_arch_min"] == "sm_90a"

    def test_invalid_cuda_arch_min_rejected(self):
        from tessera.compiler.backend_manifest import BackendKernelEntry
        with pytest.raises(ValueError, match="cuda_arch_min must be"):
            BackendKernelEntry(
                target="nvidia_sm90", status="artifact_only",
                cuda_arch_min="sm_99",
            )

    def test_wgmma_shape_field(self):
        from tessera.compiler.backend_manifest import BackendKernelEntry
        e = BackendKernelEntry(
            target="nvidia_sm90", status="artifact_only",
            wgmma_shape=(64, 256, 16),
        )
        assert e.wgmma_shape == (64, 256, 16)
        d = e.as_dict()
        assert d["wgmma_shape"] == [64, 256, 16]

    def test_wgmma_shape_rejected_on_rocm_target(self):
        from tessera.compiler.backend_manifest import BackendKernelEntry
        with pytest.raises(ValueError, match="wgmma_shape only applies to NVIDIA"):
            BackendKernelEntry(
                target="rocm_gfx942", status="artifact_only",
                wgmma_shape=(64, 256, 16),
            )

    def test_wgmma_shape_must_be_3_tuple(self):
        from tessera.compiler.backend_manifest import BackendKernelEntry
        with pytest.raises(ValueError, match="wgmma_shape must be"):
            BackendKernelEntry(
                target="nvidia_sm90", status="artifact_only",
                wgmma_shape=(64, 256),  # type: ignore[arg-type]
            )

    def test_mfma_shape_field(self):
        from tessera.compiler.backend_manifest import BackendKernelEntry
        e = BackendKernelEntry(
            target="rocm_gfx942", status="artifact_only",
            mfma_shape=(32, 32, 8, 1),
        )
        assert e.mfma_shape == (32, 32, 8, 1)
        d = e.as_dict()
        assert d["mfma_shape"] == [32, 32, 8, 1]

    def test_mfma_shape_rejected_on_nvidia_target(self):
        from tessera.compiler.backend_manifest import BackendKernelEntry
        with pytest.raises(ValueError, match="mfma_shape only applies to ROCm"):
            BackendKernelEntry(
                target="nvidia_sm90", status="artifact_only",
                mfma_shape=(32, 32, 8, 1),
            )

    def test_nvcc_version_min_field(self):
        from tessera.compiler.backend_manifest import BackendKernelEntry
        e = BackendKernelEntry(
            target="nvidia_sm90", status="artifact_only",
            nvcc_version_min="13.3",
        )
        assert e.nvcc_version_min == "13.3"
        assert e.as_dict()["nvcc_version_min"] == "13.3"

    def test_hipcc_version_min_field(self):
        from tessera.compiler.backend_manifest import BackendKernelEntry
        e = BackendKernelEntry(
            target="rocm_gfx942", status="artifact_only",
            hipcc_version_min="7.2.4",
        )
        assert e.hipcc_version_min == "7.2.4"
        assert e.as_dict()["hipcc_version_min"] == "7.2.4"

    def test_expected_mfu_in_range(self):
        from tessera.compiler.backend_manifest import BackendKernelEntry
        e = BackendKernelEntry(
            target="nvidia_sm90", status="artifact_only",
            expected_mfu=0.75,
        )
        assert e.expected_mfu == 0.75

    def test_expected_mfu_out_of_range_rejected(self):
        from tessera.compiler.backend_manifest import BackendKernelEntry
        for bad in (-0.1, 1.5, 2.0):
            with pytest.raises(ValueError, match="expected_mfu must be"):
                BackendKernelEntry(
                    target="cpu", status="reference",
                    expected_mfu=bad,
                )

    def test_optional_fields_absent_from_dict_when_none(self):
        from tessera.compiler.backend_manifest import BackendKernelEntry
        e = BackendKernelEntry(target="cpu", status="reference")
        d = e.as_dict()
        for k in ("cuda_arch_min", "nvcc_version_min", "wgmma_shape",
                  "cluster_size", "mfma_shape", "hipcc_version_min",
                  "expected_mfu", "roofline_target"):
            assert k not in d, f"{k} should not appear when None"


# ──────────────────────────────────────────────────────────────────────────
#         G-3 manifest wiring: per-kernel WGMMA/MFMA shape attached
# ──────────────────────────────────────────────────────────────────────────

class TestG3ManifestWiring:
    def test_nvidia_matmul_carries_wgmma_shape(self):
        from tessera.compiler.backend_manifest import manifest_for
        entries = {e.target: e for e in manifest_for("matmul")}
        sm90 = entries["nvidia_sm90"]
        assert sm90.wgmma_shape == (64, 256, 16)
        assert sm90.cuda_arch_min == "sm_90a"
        assert sm90.nvcc_version_min == "13.3"

    def test_nvidia_flash_attn_carries_cluster_size(self):
        from tessera.compiler.backend_manifest import manifest_for
        entries = {e.target: e for e in manifest_for("flash_attn")}
        sm90 = entries["nvidia_sm90"]
        assert sm90.wgmma_shape == (64, 128, 16)
        assert sm90.cluster_size == (2, 1, 1)

    def test_nvidia_lightning_attention_uses_small_tile(self):
        from tessera.compiler.backend_manifest import manifest_for
        entries = {e.target: e for e in manifest_for("lightning_attention")}
        assert entries["nvidia_sm90"].wgmma_shape == (32, 32, 16)
        assert entries["nvidia_sm90"].cluster_size == (1, 1, 1)

    def test_nvidia_matmul_has_mfu_target(self):
        from tessera.compiler.backend_manifest import manifest_for
        entries = {e.target: e for e in manifest_for("matmul")}
        assert entries["nvidia_sm90"].expected_mfu == 0.80
        assert entries["nvidia_sm100"].expected_mfu == 0.82

    def test_nvidia_matmul_has_roofline(self):
        from tessera.compiler.backend_manifest import manifest_for
        entries = {e.target: e for e in manifest_for("matmul")}
        rl = entries["nvidia_sm90"].roofline_target
        assert rl is not None
        assert "compute-bound" in rl

    def test_rocm_matmul_is_wmma_hardware_verified(self):
        # Strix Halo bring-up (2026-06-22): the rocm matmul row was promoted to
        # the RDNA WMMA device_verified_abi path, so it no longer carries the CDNA
        # MFMA (32,32,8,1) shape. The MFMA artifact contract still holds for the
        # other GEMM-family ops (see test_rocm_gemm_carries_mfma_shape).
        from tessera.compiler.backend_manifest import manifest_for
        rocm = {e.target: e for e in manifest_for("matmul")}["rocm"]
        assert rocm.status == "device_verified_abi"
        assert rocm.mfma_shape is None
        assert "wmma" in rocm.feature_flags
        assert rocm.runtime_symbol == "tessera_rocm_wmma_gemm_f16"
        assert rocm.hipcc_version_min == "7.2.4"

    def test_rocm_gemm_carries_mfma_shape(self):
        # The CDNA MFMA artifact contract still holds for the (still
        # artifact_only) GEMM-family ops that didn't get a WMMA proof.
        from tessera.compiler.backend_manifest import manifest_for
        rocm = {e.target: e for e in manifest_for("gemm")}["rocm"]
        assert rocm.mfma_shape == (32, 32, 8, 1)
        assert rocm.hipcc_version_min == "7.2.4"

    def test_rocm_flash_attn_is_wmma_hardware_verified(self):
        # flash_attn on rocm executes natively on gfx1151 (RDNA 3.5) via the
        # shipped libtessera_rocm_flash_attn.so WMMA kernel — the second op after
        # matmul to do so. The device_verified_abi WMMA row carries the wmma flag,
        # not the CDNA MFMA shape (WMMA != MFMA). The 16x16x16 attention MFMA
        # shape is still asserted below on the artifact-path attention family.
        from tessera.compiler.backend_manifest import manifest_for
        entries = {e.target: e for e in manifest_for("flash_attn")}
        rocm = entries["rocm"]
        assert rocm.status == "device_verified_abi"
        assert "wmma" in rocm.feature_flags
        assert rocm.runtime_symbol == "tessera_rocm_wmma_flash_attn_f16"

    def test_rocm_attention_family_promotions_are_not_mfma_artifacts(self):
        # The executing members moved off the CDNA MFMA artifact path:
        # flash_attn → WMMA device_verified_abi, multi_head_attention → WMMA
        # `device_verified_jit`, and DeepSeek/NSA → the DK2 sparse-attention compiled lane.
        from tessera.compiler.backend_manifest import manifest_for
        nsa = {e.target: e for e in manifest_for("deepseek_sparse_attention")}
        assert nsa["rocm"].status == "device_verified_jit"
        assert "sparse_attention" in nsa["rocm"].feature_flags
        assert nsa["rocm"].mfma_shape is None

        mha = {e.target: e for e in manifest_for("multi_head_attention")}
        assert mha["rocm"].status == "device_verified_jit"
        assert "wmma" in mha["rocm"].feature_flags
        assert mha["rocm"].mfma_shape is None

    def test_rocm_matmul_has_mfu_target(self):
        from tessera.compiler.backend_manifest import manifest_for
        entries = {e.target: e for e in manifest_for("matmul")}
        assert entries["rocm"].expected_mfu == 0.75


# ──────────────────────────────────────────────────────────────────────────
#                  G-2: NVIDIA kernel inventory doc
# ──────────────────────────────────────────────────────────────────────────

class TestG2NvidiaInventoryDoc:
    def test_doc_exists(self):
        doc = REPO / "docs/backends/nvidia/kernel-inventory.md"
        assert doc.exists()

    def test_doc_covers_required_sections(self):
        doc = (REPO / "docs/backends/nvidia/kernel-inventory.md").read_text()
        for section in (
            "Toolchain pin",
            "Per-SM feature matrix",
            "Per-SM dtype matrix",
            "Planned fused kernel inventory",
            "PTX assembly patterns",
            "Execution gates",
            "CUDA 13.3",
        ):
            assert section in doc, f"Section {section!r} missing"

    def test_doc_mentions_canonical_kernel_families(self):
        doc = (REPO / "docs/backends/nvidia/kernel-inventory.md").read_text()
        for kernel in (
            "flash_attn",          # FA-4
            "mla_decode",          # DeepSeek MLA
            "deepseek_sparse_attention",  # NSA
            "lightning_attention", # MiniMax
            "kimi_delta_attention",
            "swiglu_mlp",
            "matmul_softmax_matmul",
            "adamw_step",
        ):
            assert kernel in doc, f"{kernel} missing from inventory"

    def test_doc_documents_ptx_patterns(self):
        doc = (REPO / "docs/backends/nvidia/kernel-inventory.md").read_text()
        for ptx in (
            "wgmma.mma_async.sync.aligned",
            "cp.async.bulk.tensor",
            "mbarrier.arrive.expect_tx",
            "tcgen05.mma",
            "tcgen05.alloc",
        ):
            assert ptx in doc, f"PTX pattern {ptx!r} missing"


# ──────────────────────────────────────────────────────────────────────────
#                  H-3: ROCm kernel inventory doc
# ──────────────────────────────────────────────────────────────────────────

class TestH3RocmInventoryDoc:
    def test_doc_exists(self):
        doc = REPO / "docs/backends/rocm/kernel-inventory.md"
        assert doc.exists()

    def test_doc_covers_required_sections(self):
        doc = (REPO / "docs/backends/rocm/kernel-inventory.md").read_text()
        for section in (
            "Toolchain pin",
            "Per-arch feature matrix",
            "MFMA instruction shape table",
            "Per-arch dtype matrix",
            "Fused kernel inventory",
            "AMDGCN intrinsic patterns",
            "Execution gates",
            "ROCm 7.2.4",
            "HIP 7.2.4",
        ):
            assert section in doc, f"Section {section!r} missing"

    def test_doc_mentions_arch_families(self):
        doc = (REPO / "docs/backends/rocm/kernel-inventory.md").read_text()
        for arch in ("gfx90a", "gfx940", "gfx942", "gfx950", "gfx1100",
                     "gfx1200", "CDNA 2", "CDNA 3", "CDNA 4", "RDNA 3",
                     "RDNA 4",
                     "MI300X", "MI325X"):
            assert arch in doc, f"{arch} missing from inventory"

    def test_doc_documents_amdgcn_patterns(self):
        doc = (REPO / "docs/backends/rocm/kernel-inventory.md").read_text()
        for intrinsic in (
            "llvm.amdgcn.mfma.f32.32x32x8bf16",
            "llvm.amdgcn.mfma.f32.16x16x16bf16",
            "llvm.amdgcn.mfma.f32.32x32x16f8f8",
            "llvm.amdgcn.wmma.f32.16x16x16",
            "llvm.amdgcn.global.load.lds",
        ):
            assert intrinsic in doc, f"intrinsic {intrinsic!r} missing"

    def test_doc_documents_cdna4_fp4_lanes(self):
        doc = (REPO / "docs/backends/rocm/kernel-inventory.md").read_text()
        for fp4 in (
            "llvm.amdgcn.mfma.f32.32x32x32f4f4",
            "(32, 32, 32, 1)",
            "fp4_e2m1",
        ):
            assert fp4 in doc, f"{fp4} missing (CDNA 4 FP4 coverage)"


# ──────────────────────────────────────────────────────────────────────────
#     H-3 anti-drift: inventory execution-status must track the generated
#     runtime execution matrix.  Added 2026-07-10 after the doc silently
#     drifted for weeks — the structural guards above never checked *what
#     executes*, so §7/§9 kept claiming only matmul+flash_attn ran on gfx1151
#     while dozens of compiled HIP lanes had already landed.  This ties the
#     inventory's execution claims to the drift-gated matrix (Decision #26).
# ──────────────────────────────────────────────────────────────────────────

class TestH3RocmInventoryExecutionStatus:
    INVENTORY = REPO / "docs/backends/rocm/kernel-inventory.md"
    MATRIX = REPO / "docs" / "audit" / "generated" / "runtime_execution_matrix.md"

    def test_doc_points_at_generated_truth_surface(self):
        doc = self.INVENTORY.read_text()
        assert "runtime_execution_matrix" in doc, (
            "the inventory must point at the generated runtime execution matrix "
            "as status truth (Decision #26) so execution status is not "
            "hand-maintained here"
        )

    def test_doc_acknowledges_native_rocm_execution(self):
        # A revert to the stale 'only matmul + flash_attn execute' framing would
        # drop these markers of the native HIP compiled-lane program.
        doc = self.INVENTORY.read_text()
        assert "hip_runtime" in doc
        assert "device_verified_jit" in doc

    def test_every_compiled_lane_named_in_doc_exists_in_matrix(self):
        # The inventory must not name an executing rocm_*_compiled lane that the
        # drift-gated matrix does not carry.  This is the concrete anti-drift
        # tie: doc execution claims are verifiable against generated truth.
        import re

        doc = self.INVENTORY.read_text()
        matrix = self.MATRIX.read_text()
        lane_re = re.compile(r"rocm_[a-z0-9]+(?:_[a-z0-9]+)*_compiled")
        doc_lanes = set(lane_re.findall(doc))
        matrix_lanes = set(lane_re.findall(matrix))
        assert doc_lanes, "inventory should name the executing rocm compiled lanes"
        missing = sorted(doc_lanes - matrix_lanes)
        assert not missing, (
            f"inventory names compiled lanes absent from the generated matrix: "
            f"{missing}"
        )


# ──────────────────────────────────────────────────────────────────────────
#                  G-4: NVIDIA compiler contracts
# ──────────────────────────────────────────────────────────────────────────

NVIDIA_FIXTURE_NAMES = [
    "hopper_tile_to_nvidia.mlir",
    "hopper_to_nvvm_contract.mlir",
    "blackwell_tile_to_nvidia.mlir",
    "blackwell_to_nvvm_contract.mlir",
    "sm120_attention_kernel.mlir",
    "structured_kernels_tile_to_nvidia.mlir",
]


class TestG4NvidiaLitFixtures:
    @pytest.mark.parametrize("name", NVIDIA_FIXTURE_NAMES)
    def test_fixture_exists(self, name):
        assert (NVIDIA_FIXTURES / name).exists()

    @pytest.mark.parametrize("name", NVIDIA_FIXTURE_NAMES)
    def test_fixture_has_filecheck_run_line(self, name):
        body = (NVIDIA_FIXTURES / name).read_text()
        assert "RUN: %tnv" in body
        assert "FileCheck" in body

    def test_hopper_and_blackwell_contracts_are_architecture_owned(self):
        hopper = (NVIDIA_FIXTURES / "hopper_tile_to_nvidia.mlir").read_text()
        blackwell = (NVIDIA_FIXTURES / "blackwell_tile_to_nvidia.mlir").read_text()
        assert "tessera_nvidia.wgmma" in hopper
        assert 'arch = "sm_90a"' in hopper
        assert "tessera_nvidia.tcgen05_mma" in blackwell
        assert 'arch = "sm_100a"' in blackwell

    def test_core_pipeline_alias_fixture_runs_without_backend_gate(self):
        fixture = (
            TESSERA_IR_FIXTURES / "phase3" / "cuda13"
            / "nvidia_pipeline_alias.mlir"
        )
        body = fixture.read_text()
        assert "REQUIRES:" not in body
        for alias in (
            "tessera-nvidia-pipeline",
            "tessera-nvidia-pipeline-sm90",
            "tessera-nvidia-pipeline-sm100",
            "tessera-nvidia-pipeline-sm120",
        ):
            assert f"--{alias}" in body


# ──────────────────────────────────────────────────────────────────────────
#                  H-4: ROCm compiler contracts
# ──────────────────────────────────────────────────────────────────────────

ROCM_FIXTURE_NAMES = [
    "wmma_rdna3_matmul.mlir",
    "tile_matmul_to_rocm.mlir",
    "fp8_flavor_arch_keyed.mlir",
    "rocm_target_to_rocdl_contract.mlir",
    "gfx1151_tile_matmul_kernel.mlir",
    "gfx1151_tile_softmax_kernel.mlir",
]


class TestH4RocmLitFixtures:
    @pytest.mark.parametrize("name", ROCM_FIXTURE_NAMES)
    def test_fixture_exists(self, name):
        assert (ROCM_FIXTURES / name).exists()

    @pytest.mark.parametrize("name", ROCM_FIXTURE_NAMES)
    def test_fixture_has_backend_filecheck_run_line(self, name):
        body = (ROCM_FIXTURES / name).read_text()
        assert "RUN: %trop" in body
        assert "FileCheck" in body

    def test_architecture_selection_distinguishes_wmma_and_mfma(self):
        body = (ROCM_FIXTURES / "wmma_rdna3_matmul.mlir").read_text()
        assert "gfx1100" in body
        assert "llvm.amdgcn.wmma" in body
        cdna = (ROCM_FIXTURES / "tile_matmul_to_rocm.mlir").read_text()
        assert "tessera_rocm.mfma" in cdna


class TestLitFeatureHygiene:
    def test_every_required_feature_is_declared_by_lit_config(self):
        import re

        config_body = (TESSERA_IR_FIXTURES / "lit.cfg.py").read_text()
        declared = set(
            re.findall(r'available_features\.add\("([^"]+)"\)', config_body)
        )
        required: set[str] = set()
        for fixture in TESSERA_IR_FIXTURES.rglob("*.mlir"):
            for line in fixture.read_text().splitlines():
                if line.startswith("// REQUIRES:"):
                    required.update(
                        feature.strip()
                        for feature in line.split(":", 1)[1].split(",")
                    )
        assert required <= declared, (
            f"undefined lit features: {sorted(required - declared)}"
        )

    def test_obsolete_global_target_flags_do_not_return(self):
        stale_flags = ("--gpu-target=", "--rocm-target=")
        offenders: list[str] = []
        for fixture in TESSERA_IR_FIXTURES.rglob("*.mlir"):
            body = fixture.read_text()
            if any(flag in body for flag in stale_flags):
                offenders.append(str(fixture.relative_to(REPO)))
        assert not offenders, f"obsolete target flags in lit fixtures: {offenders}"


# ──────────────────────────────────────────────────────────────────────────
#                  Cross-checks: kernel coverage parity
# ──────────────────────────────────────────────────────────────────────────

class TestCrossTargetParity:
    """Both inventories should cover the same set of canonical kernel
    families — Tessera promises the same compiler surface on both
    backends modulo dtype availability."""

    @pytest.mark.parametrize("op_name", [
        "matmul", "flash_attn", "mla_decode",
        "lightning_attention", "deepseek_sparse_attention",
        "rmsnorm", "softmax",
    ])
    def test_op_has_entries_on_both_backends(self, op_name):
        from tessera.compiler.backend_manifest import manifest_for
        targets = {e.target for e in manifest_for(op_name)}
        # NVIDIA SM_90 + ROCm should both appear.
        assert "nvidia_sm90" in targets, f"{op_name} missing nvidia_sm90"
        assert "rocm" in targets, f"{op_name} missing rocm"
