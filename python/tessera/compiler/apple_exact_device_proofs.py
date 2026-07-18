"""Registry for the Apple exact-device proof cohort.

An execution-matrix ``native_gpu`` row states what a successful dispatch may
report.  It does not by itself prove that the registered C ABI is present, that
the exact-device test asserts placement, or that the fallback test rejects the
same native label.  This registry binds those independently owned facts for the
first native Apple vertical slices.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppleExactDeviceProof:
    compiler_path: str
    op_names: tuple[str, ...]
    native_test: str
    fallback_test: str
    runtime_symbols: tuple[str, ...]
    compiler_envelope: bool = True
    cohort: str = "APPLE-TEST-2-C1 / APPLE-REG-1-C1 sparse-and-runtime proofs"


@dataclass(frozen=True)
class AppleSynthesisProof:
    """Proof-ladder record for native synthesis outside the C1--C3 ABI cohort.

    The synthesis runners report ``metal_runtime``/``reference`` rather than
    the execution-matrix mapping used by :class:`AppleExactDeviceProof`; keep
    that distinction explicit instead of pretending a reference result is a
    native ABI launch.
    """
    family: str
    native_test: str
    fallback_test: str
    retired_symbols: tuple[str, ...] = ()


SYNTHESIS_PROOFS: tuple[AppleSynthesisProof, ...] = (
    AppleSynthesisProof(
        "matmul pointwise and reduction synthesis",
        "tests/unit/test_fusion_synthesis.py::test_synthesized_kernel_equals_unfused_on_metal",
        "tests/unit/test_fusion_synthesis.py::test_synthesized_region_forced_fallback_is_reference_and_correct",
        ("tessera_apple_gpu_matmul_gelu_f32",
         "tessera_apple_gpu_matmul_rmsnorm_f32"),
    ),
    AppleSynthesisProof(
        "threadgroup-tiled softmax synthesis",
        "tests/unit/test_fusion_synthesis.py::test_tiled_softmax_synthesis_equals_oracle_on_metal",
        "tests/unit/test_fusion_synthesis.py::test_synthesized_region_forced_fallback_is_reference_and_correct",
        ("tessera_apple_gpu_matmul_softmax_tiled_f32",),
    ),
)


# These comparisons used retired catalog ABIs.  They are deliberately not in
# the exact-device lane: their live synthesized replacements are registered in
# SYNTHESIS_PROOFS above.
RETIRED_SYNTHESIS_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("tessera_apple_gpu_matmul_gelu_f32",
     "tests/unit/test_fusion_synthesis.py::test_retired_matmul_gelu_symbol_is_not_an_exact_device_contract"),
    ("tessera_apple_gpu_matmul_rmsnorm_f32",
     "tests/unit/test_fusion_synthesis.py::test_retired_matmul_rmsnorm_symbol_is_not_an_exact_device_contract"),
    ("tessera_apple_gpu_matmul_gelu_f16",
     "tests/unit/test_fusion_synthesis.py::test_retired_matmul_gelu_f16_symbol_is_not_an_exact_device_contract"),
    ("tessera_apple_gpu_matmul_softmax_tiled_f32",
     "tests/unit/test_fusion_synthesis.py::test_retired_tiled_matmul_softmax_symbol_is_not_an_exact_device_contract"),
)


@dataclass(frozen=True)
class AppleStatefulProof:
    """Native stateful-family proof, with a distinct forced-fallback negative."""
    family: str
    native_test: str
    fallback_test: str
    stress_test: str


STATEFUL_PROOFS: tuple[AppleStatefulProof, ...] = (
    AppleStatefulProof(
        "paged KV attention",
        "tests/unit/test_paged_kv_native.py::test_native_equivalence_oracle",
        "tests/unit/test_paged_kv_native.py::test_native_oracle_is_inconclusive_without_metal",
        "tests/unit/test_apple_gpu_resident_block_paged.py::test_concurrent_sequences",
    ),
    AppleStatefulProof(
        "ReplaySSM fused decode",
        "tests/unit/test_ssm_apple_gpu_fused.py::test_fused_decode_reports_native_gpu_and_matches_eager",
        "tests/unit/test_ssm_apple_gpu_fused.py::test_fused_decode_forced_missing_binding_is_reference_and_correct",
        "tests/unit/test_ssm_apple_gpu_fused.py::test_fused_speculative_rollback_still_exact",
    ),
    AppleStatefulProof(
        "ReplaySSM resident checkpoint fold",
        "tests/unit/test_ssm_apple_gpu_resident_replay.py::test_resident_replay_native_flush_folds_and_clears_on_device",
        "tests/unit/test_ssm_apple_gpu_resident_replay.py::test_resident_replay_flush_reference_fallback_is_explicit",
        "tests/unit/test_ssm_apple_gpu_resident_replay.py::test_resident_replay_repeated_native_flush_and_cleanup",
    ),
)


EXACT_DEVICE_PROOFS: tuple[AppleExactDeviceProof, ...] = (
    AppleExactDeviceProof(
        "apple_gpu_spmm_csr_compiled", ("tessera.spmm_csr",),
        "tests/unit/test_apple_gpu_spmm_csr_compiled.py::test_spmm_csr_f32_reports_native_gpu_on_metal",
        "tests/unit/test_apple_gpu_spmm_csr_compiled.py::test_spmm_csr_non_f32_uses_reference_cpu_override",
        ("tessera_apple_gpu_spmm_csr_f32",),
    ),
    AppleExactDeviceProof(
        "apple_gpu_spmm_coo_compiled", ("tessera.spmm_coo",),
        "tests/unit/test_apple_gpu_spmm_coo_compiled.py::test_spmm_coo_f32_reports_native_gpu_on_metal",
        "tests/unit/test_apple_gpu_spmm_coo_compiled.py::test_spmm_coo_non_f32_uses_reference_cpu_override",
        ("tessera_apple_gpu_spmm_csr_f32",),
    ),
    AppleExactDeviceProof(
        "apple_gpu_sddmm_compiled", ("tessera.sddmm",),
        "tests/unit/test_apple_gpu_sddmm_compiled.py::test_sddmm_f32_reports_native_gpu_on_metal",
        "tests/unit/test_apple_gpu_sddmm_compiled.py::test_sddmm_non_f32_uses_reference_cpu_override",
        ("tessera_apple_gpu_sddmm_f32",),
    ),
    AppleExactDeviceProof(
        "apple_gpu_bsmm_compiled", ("tessera.bsmm",),
        "tests/unit/test_apple_gpu_bsmm_compiled.py::test_bsmm_f32_reports_native_gpu_on_metal",
        "tests/unit/test_apple_gpu_bsmm_compiled.py::test_bsmm_f32_demotes_to_reference_cpu_without_metal",
        ("tessera_apple_gpu_mpsgraph_bsmm_f32",),
    ),
    AppleExactDeviceProof(
        "apple_gpu_scatter_compiled",
        ("tessera.scatter", "tessera.scatter_add", "tessera.scatter_reduce"),
        "tests/unit/test_apple_gpu_scatter_compiled.py::test_scatter_f32_reports_native_gpu_on_metal",
        "tests/unit/test_apple_gpu_scatter_compiled.py::test_scatter_non_f32_uses_reference_cpu_override",
        ("tessera_apple_gpu_scatter_f32",),
    ),
    AppleExactDeviceProof(
        "apple_gpu_optimizer_compiled",
        ("tessera.sgd", "tessera.momentum", "tessera.adam", "tessera.adamw", "tessera.lion"),
        "tests/unit/test_apple_gpu_optimizer_compiled.py::test_f32_optimizer_ops_report_native_gpu_on_metal",
        "tests/unit/test_apple_gpu_optimizer_compiled.py::test_unsupported_optimizer_dtype_uses_reference_cpu_override",
        ("tessera_apple_gpu_optimizer_f32",),
    ),
    AppleExactDeviceProof(
        "apple_gpu_moe_compiled", ("tessera.moe",),
        "tests/unit/test_apple_gpu_moe_compiled.py::test_local_moe_f32_reports_native_gpu_on_metal",
        "tests/unit/test_apple_gpu_moe_compiled.py::test_local_moe_non_f32_and_strided_route_use_reference_cpu_override",
        ("tessera_apple_gpu_mps_matmul_f32",),
    ),
    AppleExactDeviceProof(
        "apple_gpu_moe_transport_compiled", ("tessera.moe_dispatch", "tessera.moe_combine"),
        "tests/unit/test_apple_gpu_moe_transport_compiled.py::test_moe_transport_reports_native_gpu_on_metal",
        "tests/unit/test_apple_gpu_moe_transport_compiled.py::test_moe_transport_demotes_to_reference_cpu_without_metal",
        ("tessera_apple_gpu_mpsgraph_gather_f32", "tessera_apple_gpu_scatter_f32"),
        compiler_envelope=False,
    ),
    AppleExactDeviceProof(
        "apple_gpu_rng_compiled", ("tessera.rng_uniform", "tessera.rng_normal", "tessera.dropout"),
        "tests/unit/test_apple_gpu_rng_compiled.py::test_rng_base_ops_report_native_gpu_on_darwin",
        "tests/unit/test_apple_gpu_rng_compiled.py::test_rng_non_base_op_uses_reference_cpu_override",
        ("tessera_apple_gpu_philox_uniform_f32", "tessera_apple_gpu_philox_normal_f32", "tessera_apple_gpu_philox_dropout_f32"),
        compiler_envelope=False,
    ),
    AppleExactDeviceProof(
        "apple_gpu_complex_compiled", ("tessera.complex_mul", "tessera.complex_exp"),
        "tests/unit/test_complex_runtime.py::test_complex_and_conformal_f32_native_routes_match_reference",
        "tests/unit/test_complex_runtime.py::test_complex_and_conformal_forced_bridge_miss_is_reference_cpu",
        ("tessera_apple_gpu_complex_mul_f32", "tessera_apple_gpu_complex_exp_f32"),
        compiler_envelope=False,
        cohort="APPLE-TEST-2-C2 / APPLE-REG-1-C2 complex-and-conformal proofs",
    ),
    AppleExactDeviceProof(
        "apple_gpu_conformal_compiled", ("tessera.mobius", "tessera.stereographic"),
        "tests/unit/test_complex_runtime.py::test_complex_and_conformal_f32_native_routes_match_reference",
        "tests/unit/test_complex_runtime.py::test_complex_and_conformal_forced_bridge_miss_is_reference_cpu",
        ("tessera_apple_gpu_complex_mobius_f32", "tessera_apple_gpu_complex_stereographic_f32"),
        compiler_envelope=False,
        cohort="APPLE-TEST-2-C2 / APPLE-REG-1-C2 complex-and-conformal proofs",
    ),
    AppleExactDeviceProof(
        "apple_gpu_reduce_compiled", ("tessera.sum",),
        "tests/unit/test_apple_gpu_reduce_compiled.py::test_sum_f32_reports_native_gpu_on_metal",
        "tests/unit/test_apple_gpu_reduce_compiled.py::test_mpsgraph_reduce_and_mse_forced_miss_is_reference_cpu",
        ("tessera_apple_gpu_mpsgraph_reduce_f32",),
        compiler_envelope=False,
        cohort="APPLE-TEST-2-C3 / APPLE-REG-1-C3 losses-and-reductions proofs",
    ),
    AppleExactDeviceProof(
        "apple_gpu_loss_compiled", ("tessera.mse_loss", "tessera.mae_loss"),
        "tests/unit/test_apple_gpu_loss_compiled.py::test_mse_and_mae_f32_report_native_gpu_on_metal",
        "tests/unit/test_apple_gpu_loss_compiled.py::test_mse_forced_mpsgraph_miss_is_reference_cpu",
        ("tessera_apple_gpu_mpsgraph_binary_f32", "tessera_apple_gpu_mpsgraph_unary_f32", "tessera_apple_gpu_mpsgraph_reduce_f32"),
        compiler_envelope=False,
        cohort="APPLE-TEST-2-C3 / APPLE-REG-1-C3 losses-and-reductions proofs",
    ),
)
