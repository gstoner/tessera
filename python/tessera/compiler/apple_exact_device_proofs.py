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
