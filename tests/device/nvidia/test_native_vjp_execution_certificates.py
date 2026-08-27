"""E2E-REAL-6F exact-device certificate packet for every NVIDIA VJP family."""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tests.device.nvidia import test_spectral_autodiff as spectral
from tests.device.nvidia import test_optimizer_reverse as optimizer_reverse
from tests.device.nvidia import test_training_autodiff_native as training
from tests.unit import test_autodiff_training_series_target_binding as series


@ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=("logits", "target"))
def _nvidia_bce(logits, target):
    return ts.ops.binary_cross_entropy_loss(logits, target, reduction="none")


@ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=("logits",))
def _nvidia_cross_entropy(logits, target):
    return ts.ops.label_smoothed_cross_entropy(
        logits, target, smoothing=0.15, reduction="none", axis=-1,
        ignore_index=-9,
    )


@ts.jit(
    target="nvidia_sm120",
    autodiff="reverse",
    wrt=("q", "k", "v", "gate", "beta", "decay"),
)
def _nvidia_sequence(q, k, v, gate, beta, decay):
    return ts.ops.gated_deltanet(
        q, k, v, gate, beta, decay, causal=True,
    )


@pytest.mark.compiler_nvidia
@pytest.mark.hardware_nvidia
def test_every_declared_nvidia_vjp_family_records_an_exact_certificate() -> None:
    from tessera import runtime
    from tessera.autodiff.vjp import get_vjp
    from tessera.compiler.frontend_authority_audit import collect_rows
    from tessera.compiler.native_vjp_plugins import (
        native_vjp_exact_execution_coverage,
        native_vjp_execution_certificates,
        validate_native_vjp_execution_certificate,
    )

    if not runtime._nvidia_tile_runtime_available():
        pytest.skip("live SM120 CUDA runtime is required")

    # Normalization and regression loss use their independent device oracles.
    training.test_public_jit_native_backward_binds_sm120_training_packages()
    series.test_nvidia_lion_backward_runs_sm120_stop_sign_package()
    optimizer_reverse.test_sm120_sgd_reverse_exact_certificate()
    optimizer_reverse.test_sm120_momentum_and_nesterov_reverse_exact_certificates()
    optimizer_reverse.test_sm120_adam_and_adamw_reverse_exact_certificates()
    optimizer_reverse.test_sm120_adafactor_full_and_factored_exact_certificates()

    logits = np.linspace(-9.0, 9.0, 35, dtype=np.float32).reshape(5, 7)
    target = np.linspace(0.0, 1.0, 35, dtype=np.float32).reshape(5, 7)
    seed = np.linspace(0.25, 1.25, 35, dtype=np.float32).reshape(5, 7)
    dz, dt = _nvidia_bce.native_backward(logits, target, out_cotangents=seed)
    np.testing.assert_allclose(
        dz, (series._stable_sigmoid(logits) - target) * seed,
        atol=3e-6, rtol=3e-6,
    )
    np.testing.assert_allclose(dt, -logits * seed, atol=3e-6, rtol=3e-6)

    rng = np.random.default_rng(20260903)
    class_logits = rng.normal(size=(5, 7)).astype(np.float32)
    targets = rng.integers(0, 7, size=(5,), dtype=np.int64)
    class_seed = np.linspace(0.5, 1.0, 5, dtype=np.float32)
    (dlogits,) = _nvidia_cross_entropy.native_backward(
        class_logits, targets, out_cotangents=class_seed,
    )
    shifted = class_logits - class_logits.max(axis=-1, keepdims=True)
    probability = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    expected_target = np.full_like(class_logits, 0.15 / class_logits.shape[-1])
    expected_target[np.arange(targets.size), targets] += 0.85
    expected = probability - expected_target
    np.testing.assert_allclose(
        dlogits, expected * class_seed[:, None], atol=5e-6, rtol=5e-6,
    )

    q_shape, v_shape = (1, 1, 5, 3), (1, 1, 5, 2)
    q = rng.normal(scale=0.3, size=q_shape).astype(np.float32)
    k = rng.normal(scale=0.3, size=q_shape).astype(np.float32)
    k /= np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1.0e-6)
    v = rng.normal(scale=0.3, size=v_shape).astype(np.float32)
    gate = rng.normal(scale=0.2, size=v_shape).astype(np.float32)
    beta = rng.uniform(0.2, 0.8, q_shape[:-1]).astype(np.float32)
    decay = rng.uniform(0.7, 0.95, q_shape[:-1]).astype(np.float32)
    dy = rng.normal(scale=0.3, size=v_shape).astype(np.float32)
    actual = _nvidia_sequence.native_backward(
        q, k, v, gate, beta, decay, out_cotangents=dy,
    )
    reference = get_vjp("gated_deltanet")(
        dy, q, k, v, gate, beta, decay, erase=False,
    )
    for value, expected_value in zip(actual, reference, strict=True):
        np.testing.assert_allclose(
            value, expected_value, rtol=2e-3, atol=2e-3,
        )

    with pytest.MonkeyPatch.context() as monkeypatch:
        spectral.test_spectral_convolution_vjp_matches_direct_correlation(
            monkeypatch
        )
    spectral.test_stft_istft_vjp_matches_independent_reference("stft")
    spectral.test_stft_istft_vjp_matches_independent_reference("istft")

    required = {
        (row.family, "nvidia_sm120")
        for row in collect_rows()
        if "nvidia_sm120" in row.targets
    }
    observed = {
        pair
        for pair in native_vjp_exact_execution_coverage()
        if pair[1] == "nvidia_sm120"
    }
    assert observed == required

    certificates = native_vjp_execution_certificates()
    for family, target in sorted(required):
        rows = [
            row for row in certificates[family]
            if row["target"] == target and row["evidence_scope"] == "exact_device"
        ]
        assert rows, f"{family}/{target} has no exact-device certificate"
        for row in rows:
            validate_native_vjp_execution_certificate(row)
            assert row["physical_attestation"]["device_arch"] == "sm_120"
