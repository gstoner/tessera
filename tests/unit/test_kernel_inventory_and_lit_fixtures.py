"""Sprint G-2 + G-3 + G-4 + H-3 + H-4 — Kernel inventory + schema + lit fixtures.

Hardware-free guards for the per-target kernel pre-work:

  G-2: `docs/nvidia_cuda13_kernel_inventory.md` — enumerates every
       planned NVIDIA fused kernel under CUDA 13.2 U1.
  G-3: `BackendKernelEntry` schema extended with `cuda_arch_min`,
       `nvcc_version_min`, `wgmma_shape`, `cluster_size`, `mfma_shape`,
       `hipcc_version_min`, `expected_mfu`, `roofline_target`.
  G-4: NVIDIA lit fixtures under
       `tests/tessera-ir/phase3/cuda13/` — WGMMA matmul (bf16, fp8),
       FA-4 forward, MLA decode, DeepSeek NSA, Lightning attention,
       matmul→softmax fusion, SwiGLU MLP, Blackwell tcgen05, AdamW.
  H-3: `docs/rocm_mfma_kernel_inventory.md` — enumerates every planned
       ROCm fused kernel under ROCm 7.2.3.
  H-4: ROCm lit fixtures under `tests/tessera-ir/phase8/rocm_7_2/`
       — MFMA matmul (bf16, fp8, fp4-CDNA4), FA fwd, MLA decode,
       AdamW, RDNA3 WMMA.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NVIDIA_FIXTURES = REPO / "tests" / "tessera-ir" / "phase3" / "cuda13"
ROCM_FIXTURES = REPO / "tests" / "tessera-ir" / "phase8" / "rocm_7_2"


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
            nvcc_version_min="13.2.1",
        )
        assert e.nvcc_version_min == "13.2.1"
        assert e.as_dict()["nvcc_version_min"] == "13.2.1"

    def test_hipcc_version_min_field(self):
        from tessera.compiler.backend_manifest import BackendKernelEntry
        e = BackendKernelEntry(
            target="rocm_gfx942", status="artifact_only",
            hipcc_version_min="7.2.3",
        )
        assert e.hipcc_version_min == "7.2.3"
        assert e.as_dict()["hipcc_version_min"] == "7.2.3"

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
        assert sm90.nvcc_version_min == "13.2.1"

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

    def test_rocm_matmul_carries_mfma_shape(self):
        from tessera.compiler.backend_manifest import manifest_for
        entries = {e.target: e for e in manifest_for("matmul")}
        rocm = entries["rocm"]
        assert rocm.mfma_shape == (32, 32, 8, 1)
        assert rocm.hipcc_version_min == "7.2.3"

    def test_rocm_flash_attn_uses_16x16x16(self):
        from tessera.compiler.backend_manifest import manifest_for
        entries = {e.target: e for e in manifest_for("flash_attn")}
        assert entries["rocm"].mfma_shape == (16, 16, 16, 1)

    def test_rocm_matmul_has_mfu_target(self):
        from tessera.compiler.backend_manifest import manifest_for
        entries = {e.target: e for e in manifest_for("matmul")}
        assert entries["rocm"].expected_mfu == 0.75


# ──────────────────────────────────────────────────────────────────────────
#                  G-2: NVIDIA kernel inventory doc
# ──────────────────────────────────────────────────────────────────────────

class TestG2NvidiaInventoryDoc:
    def test_doc_exists(self):
        doc = REPO / "docs" / "nvidia_cuda13_kernel_inventory.md"
        assert doc.exists()

    def test_doc_covers_required_sections(self):
        doc = (REPO / "docs" / "nvidia_cuda13_kernel_inventory.md").read_text()
        for section in (
            "Toolchain pin",
            "Per-SM feature matrix",
            "Per-SM dtype matrix",
            "Planned fused kernel inventory",
            "PTX assembly patterns",
            "Execution gates",
            "13.2 Update 1",
        ):
            assert section in doc, f"Section {section!r} missing"

    def test_doc_mentions_canonical_kernel_families(self):
        doc = (REPO / "docs" / "nvidia_cuda13_kernel_inventory.md").read_text()
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
        doc = (REPO / "docs" / "nvidia_cuda13_kernel_inventory.md").read_text()
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
        doc = REPO / "docs" / "rocm_mfma_kernel_inventory.md"
        assert doc.exists()

    def test_doc_covers_required_sections(self):
        doc = (REPO / "docs" / "rocm_mfma_kernel_inventory.md").read_text()
        for section in (
            "Toolchain pin",
            "Per-arch feature matrix",
            "MFMA instruction shape table",
            "Per-arch dtype matrix",
            "Planned fused kernel inventory",
            "AMDGCN intrinsic patterns",
            "Execution gates",
            "ROCm 7.2.3",
            "HIP 7.2.3",
        ):
            assert section in doc, f"Section {section!r} missing"

    def test_doc_mentions_arch_families(self):
        doc = (REPO / "docs" / "rocm_mfma_kernel_inventory.md").read_text()
        for arch in ("gfx90a", "gfx940", "gfx942", "gfx950", "gfx1100",
                     "CDNA 2", "CDNA 3", "CDNA 4", "RDNA 3",
                     "MI300X", "MI325X"):
            assert arch in doc, f"{arch} missing from inventory"

    def test_doc_documents_amdgcn_patterns(self):
        doc = (REPO / "docs" / "rocm_mfma_kernel_inventory.md").read_text()
        for intrinsic in (
            "llvm.amdgcn.mfma.f32.32x32x8bf16",
            "llvm.amdgcn.mfma.f32.16x16x16bf16",
            "llvm.amdgcn.mfma.f32.32x32x16f8f8",
            "llvm.amdgcn.wmma.f32.16x16x16",
            "llvm.amdgcn.global.load.lds",
        ):
            assert intrinsic in doc, f"intrinsic {intrinsic!r} missing"

    def test_doc_documents_cdna4_fp4_lanes(self):
        doc = (REPO / "docs" / "rocm_mfma_kernel_inventory.md").read_text()
        for fp4 in (
            "llvm.amdgcn.mfma.f32.32x32x32f4f4",
            "(32, 32, 32, 1)",
            "fp4_e2m1",
        ):
            assert fp4 in doc, f"{fp4} missing (CDNA 4 FP4 coverage)"


# ──────────────────────────────────────────────────────────────────────────
#                  G-4: NVIDIA lit fixtures present + correct
# ──────────────────────────────────────────────────────────────────────────

NVIDIA_FIXTURE_NAMES = [
    "wgmma_matmul_bf16.mlir",
    "wgmma_matmul_fp8.mlir",
    "flash_attn_fwd_fa4.mlir",
    "mla_decode_fused.mlir",
    "deepseek_nsa_sparse_attention.mlir",
    "lightning_attention.mlir",
    "matmul_softmax_fused.mlir",
    "swiglu_mlp_fused.mlir",
    "tcgen05_blackwell_matmul.mlir",
    "adamw_step_fused.mlir",
]


class TestG4NvidiaLitFixtures:
    @pytest.mark.parametrize("name", NVIDIA_FIXTURE_NAMES)
    def test_fixture_exists(self, name):
        assert (NVIDIA_FIXTURES / name).exists()

    @pytest.mark.parametrize("name", NVIDIA_FIXTURE_NAMES)
    def test_fixture_has_filecheck_run_line(self, name):
        body = (NVIDIA_FIXTURES / name).read_text()
        assert "RUN: tessera-opt" in body
        assert "FileCheck" in body
        assert "REQUIRES: tessera_opt_built" in body

    @pytest.mark.parametrize("name", [
        "wgmma_matmul_bf16.mlir",
        "wgmma_matmul_fp8.mlir",
        "flash_attn_fwd_fa4.mlir",
        "mla_decode_fused.mlir",
        "deepseek_nsa_sparse_attention.mlir",
        "lightning_attention.mlir",
        "matmul_softmax_fused.mlir",
        "swiglu_mlp_fused.mlir",
    ])
    def test_fixture_asserts_wgmma_pattern(self, name):
        body = (NVIDIA_FIXTURES / name).read_text()
        assert "wgmma.mma_async.sync.aligned" in body, (
            f"{name} must assert on WGMMA PTX pattern"
        )

    def test_tcgen05_fixture_blackwell_specific(self):
        body = (NVIDIA_FIXTURES / "tcgen05_blackwell_matmul.mlir").read_text()
        assert "tcgen05.mma" in body
        assert "tcgen05.alloc" in body
        assert "sm_100a" in body
        assert "nvfp4" in body

    def test_adamw_fixture_no_wgmma(self):
        body = (NVIDIA_FIXTURES / "adamw_step_fused.mlir").read_text()
        # AdamW is elementwise — no WGMMA expected.
        assert "CHECK-NOT: wgmma.mma_async" in body


# ──────────────────────────────────────────────────────────────────────────
#                  H-4: ROCm lit fixtures present + correct
# ──────────────────────────────────────────────────────────────────────────

ROCM_FIXTURE_NAMES = [
    "mfma_matmul_bf16.mlir",
    "mfma_matmul_fp8.mlir",
    "mfma_matmul_fp4_cdna4.mlir",
    "flash_attn_fwd.mlir",
    "mla_decode.mlir",
    "adamw_step_fused.mlir",
    "wmma_rdna3_matmul.mlir",
]


class TestH4RocmLitFixtures:
    @pytest.mark.parametrize("name", ROCM_FIXTURE_NAMES)
    def test_fixture_exists(self, name):
        assert (ROCM_FIXTURES / name).exists()

    @pytest.mark.parametrize("name", ROCM_FIXTURE_NAMES)
    def test_fixture_has_filecheck_run_line(self, name):
        body = (ROCM_FIXTURES / name).read_text()
        assert "RUN: tessera-opt" in body
        assert "FileCheck" in body
        assert "REQUIRES: tessera_opt_built" in body

    @pytest.mark.parametrize("name", [
        "mfma_matmul_bf16.mlir",
        "mfma_matmul_fp8.mlir",
        "mfma_matmul_fp4_cdna4.mlir",
        "flash_attn_fwd.mlir",
        "mla_decode.mlir",
    ])
    def test_fixture_asserts_mfma_pattern(self, name):
        body = (ROCM_FIXTURES / name).read_text()
        assert "llvm.amdgcn.mfma" in body, (
            f"{name} must assert on MFMA AMDGCN intrinsic"
        )

    def test_cdna4_fp4_fixture(self):
        body = (ROCM_FIXTURES / "mfma_matmul_fp4_cdna4.mlir").read_text()
        assert "gfx950" in body
        assert "fp4_e2m1" in body
        assert "llvm.amdgcn.mfma.f32.32x32x32f4f4" in body

    def test_rdna3_uses_wmma_not_mfma(self):
        body = (ROCM_FIXTURES / "wmma_rdna3_matmul.mlir").read_text()
        assert "gfx1100" in body
        assert "llvm.amdgcn.wmma" in body
        # WMMA, not MFMA, on RDNA 3.
        assert "llvm.amdgcn.mfma" not in body

    def test_adamw_fixture_no_mfma(self):
        body = (ROCM_FIXTURES / "adamw_step_fused.mlir").read_text()
        assert "CHECK-NOT: llvm.amdgcn.mfma" in body


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
