"""E2E-REAL-6F exact-device certificate packet for every ROCm VJP family.

Each imported row remains an independent numerical test with its own oracle.
This packet composes those rows in one process so the process-level execution
registry can prove total family/target coverage rather than a collection of
unrelated successful launches. A newly declared ROCm family fails the final
set equality until it supplies an exact gfx1151 row here.
"""

from __future__ import annotations

import pytest

from tests.unit import test_autodiff_attention_plugin_binding as attention
from tests.unit import test_autodiff_norm_target_binding as normalization
from tests.unit import test_autodiff_optimizer_plugin_binding as optimizer
from tests.unit import test_autodiff_regression_loss_target_binding as regression
from tests.unit import test_autodiff_rocm_matmul_composed as matmul
from tests.unit import test_autodiff_spectral_target_binding as spectral
from tests.unit import test_autodiff_stateful_plugin_binding as stateful
from tests.unit import test_autodiff_training_series_target_binding as training
from tests.unit import test_rocm_ssm_bwd_launch_execute as selective_ssm


@pytest.mark.compiler_rocm
@pytest.mark.hardware_rocm
def test_every_declared_rocm_vjp_family_records_an_exact_certificate() -> None:
    from tessera import runtime
    from tessera.compiler.frontend_authority_audit import collect_rows
    from tessera.compiler.native_vjp_plugins import (
        native_vjp_exact_execution_coverage,
        native_vjp_execution_certificates,
        validate_native_vjp_execution_certificate,
    )

    if runtime._tessera_opt_path() is None or not runtime._rocm_wmma_runtime_available():
        pytest.skip("production tessera-opt and live gfx1151 are required")

    stateful.test_rocm_adafactor_topologies_record_exact_gfx1151_certificates(
        "full"
    )
    stateful.test_rocm_adafactor_topologies_record_exact_gfx1151_certificates(
        "factored"
    )
    attention.test_public_gfx1151_attention_vjp_consumes_prebuilt_program()
    training.test_rocm_bce_backward_runs_gfx1151()
    training.test_rocm_label_smoothed_backward_handles_ragged_runtime_rows()
    training.test_rocm_kl_backward_handles_nonfinal_axis_and_tensor_cotangent()
    training.test_rocm_lion_backward_runs_shared_stop_sign_policy_on_gfx1151()
    matmul.test_rocm_composed_matmul_backward_matches_numpy()
    normalization.test_rocm_public_native_backward(
        normalization._rocm_rmsnorm, False, "rocm_rmsnorm_bwd_compiled"
    )
    optimizer.test_rocm_sgd_momentum_variants_record_exact_gfx1151_certificates(
        optimizer._rocm_sgd, "sgd"
    )
    optimizer.test_rocm_sgd_momentum_variants_record_exact_gfx1151_certificates(
        optimizer._rocm_momentum, "momentum"
    )
    training.test_rocm_nesterov_backward_runs_one_gfx1151_launch()
    training.test_rocm_adam_backward_shares_exact_explicit_state_abi()
    training.test_rocm_adamw_backward_runs_one_gfx1151_launch()
    regression.test_rocm_public_smooth_l1_backward_runs_gfx1151()
    selective_ssm.test_public_selective_ssm_records_exact_gfx1151_certificate()
    stateful.test_rocm_sequence_mixer_records_exact_gfx1151_certificate()
    spectral.test_rocm_public_compound_spectral_backward_uses_prebuilt_image(
        spectral._rocm_spectral_filter, "filter"
    )
    spectral.test_rocm_stft_istft_backward_matches_independent_vjp("stft")
    spectral.test_rocm_stft_istft_backward_matches_independent_vjp("istft")
    spectral.test_rocm_stft_forward_and_adjoint_satisfy_inner_product_identity()

    required = {
        (row.family, "rocm")
        for row in collect_rows()
        if "rocm" in row.targets
    }
    observed = {
        pair for pair in native_vjp_exact_execution_coverage() if pair[1] == "rocm"
    }
    assert observed == required

    certificates = native_vjp_execution_certificates()
    for family, target in sorted(required):
        rows = [
            row
            for row in certificates[family]
            if row["target"] == target and row["evidence_scope"] == "exact_device"
        ]
        assert rows, f"{family}/{target} has no exact-device certificate"
        for row in rows:
            validate_native_vjp_execution_certificate(row)
            assert row["physical_attestation"]["device_arch"] == "gfx1151"
