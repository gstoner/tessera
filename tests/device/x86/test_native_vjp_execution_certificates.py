"""E2E-REAL-6F exact-host certificate packet for every x86 VJP family."""

from __future__ import annotations

import pytest

from tests.unit import test_autodiff_attention_plugin_binding as attention
from tests.unit import test_autodiff_norm_target_binding as normalization
from tests.unit import test_autodiff_regression_loss_target_binding as regression
from tests.unit import test_autodiff_spectral_target_binding as spectral
from tests.unit import test_autodiff_stateful_plugin_binding as stateful
from tests.unit import test_autodiff_training_series_target_binding as training

# This lane needs AVX-512, NOT AMX. Marking it hardware_amx would skip it on
# Princess-Luna -- the only host that can run it -- because Zen 5 has AVX-512
# and no AMX, and AMX is a retired target besides.
pytestmark = pytest.mark.hardware_avx512


def test_every_declared_x86_vjp_family_records_an_exact_certificate() -> None:
    from tessera import runtime
    from tessera.compiler.frontend_authority_audit import collect_rows
    from tessera.compiler.native_vjp_plugins import (
        native_vjp_exact_execution_coverage,
        native_vjp_execution_certificates,
        validate_native_vjp_execution_certificate,
    )

    if runtime._tessera_opt_path() is None or not runtime._x86_elementwise_available():
        pytest.skip("production tessera-opt and AVX-512 runtime are required")

    stateful.test_factored_adafactor_plugin_executes_and_records_topology_certificate()
    attention.test_public_x86_attention_vjp_consumes_exact_scheduled_package()
    training.test_x86_bce_backward_runs_avx512()
    training.test_x86_class_loss_backward_handles_axis_ignore_and_smoothing()
    training.test_x86_lion_backward_runs_shared_stop_sign_policy_on_avx512()
    normalization.test_x86_public_native_backward(
        normalization._x86_rmsnorm, False, "x86_rmsnorm_bwd_compiled"
    )
    regression.test_public_sgd_backward_composes_native_optimizer(
        regression._x86_sgd, "x86", "x86_avx512"
    )
    training.test_x86_momentum_backward_runs_one_avx512_launch()
    regression.test_x86_public_huber_backward_runs_avx512()
    stateful.test_x86_sequence_mixer_records_exact_avx512_certificate()
    spectral.test_x86_public_spectral_filter_backward_uses_family_plugin()

    required = {
        (row.family, "x86") for row in collect_rows() if "x86" in row.targets
    }
    observed = {
        pair for pair in native_vjp_exact_execution_coverage() if pair[1] == "x86"
    }
    assert observed == required

    certificates = native_vjp_execution_certificates()
    for family, target in sorted(required):
        rows = [
            row
            for row in certificates[family]
            if row["target"] == target and row["evidence_scope"] == "exact_device"
        ]
        assert rows, f"{family}/{target} has no exact-host certificate"
        for row in rows:
            validate_native_vjp_execution_certificate(row)
            assert row["physical_attestation"]["device_arch"] == "x86_avx512"
from tests.unit import test_autodiff_attention_plugin_binding as attention
from tests.unit import test_autodiff_norm_target_binding as normalization
from tests.unit import test_autodiff_regression_loss_target_binding as regression
from tests.unit import test_autodiff_spectral_target_binding as spectral
from tests.unit import test_autodiff_stateful_plugin_binding as stateful
from tests.unit import test_autodiff_training_series_target_binding as training


def test_every_declared_x86_vjp_family_records_an_exact_certificate() -> None:
    from tessera import runtime
    from tessera.compiler.frontend_authority_audit import collect_rows
    from tessera.compiler.native_vjp_plugins import (
        native_vjp_exact_execution_coverage,
        native_vjp_execution_certificates,
        validate_native_vjp_execution_certificate,
    )

    if runtime._tessera_opt_path() is None or not runtime._x86_elementwise_available():
        pytest.skip("production tessera-opt and AVX-512 runtime are required")

    stateful.test_factored_adafactor_plugin_executes_and_records_topology_certificate()
    attention.test_public_x86_attention_vjp_consumes_exact_scheduled_package()
    training.test_x86_bce_backward_runs_avx512()
    training.test_x86_class_loss_backward_handles_axis_ignore_and_smoothing()
    training.test_x86_lion_backward_runs_shared_stop_sign_policy_on_avx512()
    normalization.test_x86_public_native_backward(
        normalization._x86_rmsnorm, False, "x86_rmsnorm_bwd_compiled"
    )
    regression.test_public_sgd_backward_composes_native_optimizer(
        regression._x86_sgd, "x86", "x86_avx512"
    )
    training.test_x86_momentum_backward_runs_one_avx512_launch()
    regression.test_x86_public_huber_backward_runs_avx512()
    stateful.test_x86_sequence_mixer_records_exact_avx512_certificate()
    spectral.test_x86_public_spectral_filter_backward_uses_family_plugin()

    required = {
        (row.family, "x86") for row in collect_rows() if "x86" in row.targets
    }
    observed = {
        pair for pair in native_vjp_exact_execution_coverage() if pair[1] == "x86"
    }
    assert observed == required

    certificates = native_vjp_execution_certificates()
    for family, target in sorted(required):
        rows = [
            row
            for row in certificates[family]
            if row["target"] == target and row["evidence_scope"] == "exact_device"
        ]
        assert rows, f"{family}/{target} has no exact-host certificate"
        for row in rows:
            validate_native_vjp_execution_certificate(row)
            assert row["physical_attestation"]["device_arch"] == "x86_avx512"
