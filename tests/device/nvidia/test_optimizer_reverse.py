"""Exact-SM120 certificates for content-addressed optimizer reverse packages."""

from __future__ import annotations

import numpy as np

import tessera as ts


@ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=("p", "g"))
def _sgd(p, g):
    return ts.ops.sgd(p, g, lr=0.05)


@ts.jit(
    target="nvidia_sm120", autodiff="reverse", wrt=("p", "g", "velocity")
)
def _momentum(p, g, velocity):
    return ts.ops.momentum(p, g, velocity, lr=0.04, momentum=0.7)


@ts.jit(
    target="nvidia_sm120", autodiff="reverse", wrt=("p", "g", "velocity")
)
def _nesterov(p, g, velocity):
    return ts.ops.nesterov(p, g, velocity, lr=0.04, momentum=0.7)


@ts.jit(
    target="nvidia_sm120", autodiff="reverse", wrt=("p", "g", "m1", "m2")
)
def _adam(p, g, m1, m2):
    return ts.ops.adam(
        p, g, m1, m2, lr=0.003, beta1=0.8, beta2=0.95,
        eps=1.0e-6, step=3,
    )


@ts.jit(
    target="nvidia_sm120", autodiff="reverse", wrt=("p", "g", "m1", "m2")
)
def _adamw(p, g, m1, m2):
    return ts.ops.adamw(
        p, g, m1, m2, lr=0.003, beta1=0.8, beta2=0.95,
        eps=1.0e-6, weight_decay=0.02, step=3,
    )


@ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=("p", "g", "state"))
def _adafactor_full(p, g, state):
    updated, _new_state = ts.ops.adafactor(
        p, g, state, lr=0.003, beta2=0.91, eps=1.0e-7, step=2
    )
    return updated


@ts.jit(
    target="nvidia_sm120", autodiff="reverse", wrt=("p", "g", "row", "col")
)
def _adafactor_factored(p, g, row, col):
    updated, _new_row, _new_col = ts.ops.adafactor(
        p, g, row, col, lr=0.003, beta2=0.91, eps=1.0e-7, step=2
    )
    return updated


def _assert_certificate(compiled, family: str, topology: str = "not_applicable"):
    from tessera.compiler.native_vjp_plugins import (
        native_vjp_exact_execution_coverage,
        validate_native_vjp_execution_certificate,
    )

    execution = compiled.last_backward_execution
    certificate = execution["execution_certificate"]
    validate_native_vjp_execution_certificate(certificate)
    assert execution["execution_mode"] == "cuda_driver"
    assert execution["evidence_target"] == "nvidia_sm120"
    assert certificate["family"] == family
    assert certificate["target"] == "nvidia_sm120"
    assert certificate["evidence_scope"] == "exact_device"
    assert certificate["physical_attestation"]["device_arch"] == "sm_120"
    assert certificate["topology"] == topology
    assert (family, "nvidia_sm120") in native_vjp_exact_execution_coverage()


def _momentum_reference(dp, dv, *, lr, momentum, nesterov):
    from_param = -lr * dp
    return (
        dp,
        (1.0 + momentum if nesterov else 1.0) * from_param + dv,
        momentum * ((momentum if nesterov else 1.0) * from_param + dv),
    )


def _adam_reference(g, m1, m2, dp, dm1, dm2, *, weight_decay):
    lr, beta1, beta2, eps, step = 0.003, 0.8, 0.95, 1.0e-6, 3
    correction1, correction2 = 1.0 - beta1**step, 1.0 - beta2**step
    m1_new = beta1 * m1 + (1.0 - beta1) * g
    m2_new = beta2 * m2 + (1.0 - beta2) * g * g
    normalized = m2_new / correction2
    root = np.sqrt(normalized)
    denom = root + eps
    dm1_new = dm1 + dp * (-lr / correction1) / denom
    droot = np.where(normalized > 0.0, 0.5 / (correction2 * root), 0.0)
    dm2_new = dm2 + dp * lr * (m1_new / correction1) * droot / (denom * denom)
    return (
        dp * (1.0 - lr * weight_decay),
        (1.0 - beta1) * dm1_new + 2.0 * (1.0 - beta2) * g * dm2_new,
        beta1 * dm1_new,
        beta2 * dm2_new,
    )


def test_sm120_sgd_reverse_exact_certificate():
    rng = np.random.default_rng(1301)
    p = rng.normal(size=(5, 17)).astype(np.float32)
    g = rng.normal(size=p.shape).astype(np.float32)
    dp = rng.normal(size=p.shape).astype(np.float32)
    actual = _sgd.native_backward(p, g, out_cotangents=dp)
    expected = (dp, np.float32(-0.05) * dp)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=2e-6, atol=2e-6)
    _assert_certificate(_sgd, "optimizer_vjp")


def test_sm120_momentum_and_nesterov_reverse_exact_certificates():
    rng = np.random.default_rng(1303)
    values = [rng.normal(size=(7, 13)).astype(np.float32) for _ in range(3)]
    dp = rng.normal(size=(7, 13)).astype(np.float32)
    dv = rng.normal(size=(7, 13)).astype(np.float32)
    for compiled, nesterov in ((_momentum, False), (_nesterov, True)):
        actual = compiled.native_backward(*values, out_cotangents=(dp, dv))
        expected = _momentum_reference(
            dp, dv, lr=0.04, momentum=0.7, nesterov=nesterov
        )
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(value, reference, rtol=2e-6, atol=2e-6)
        _assert_certificate(compiled, "optimizer_vjp")


def test_sm120_adam_and_adamw_reverse_exact_certificates():
    rng = np.random.default_rng(1307)
    shape = (5, 19)
    p = rng.normal(size=shape).astype(np.float32)
    g = rng.normal(scale=0.2, size=shape).astype(np.float32)
    m1 = rng.normal(scale=0.1, size=shape).astype(np.float32)
    m2 = rng.uniform(0.05, 0.25, size=shape).astype(np.float32)
    cotangents = tuple(
        rng.normal(scale=0.3, size=shape).astype(np.float32) for _ in range(3)
    )
    for compiled, decay in ((_adam, 0.0), (_adamw, 0.02)):
        actual = compiled.native_backward(
            p, g, m1, m2, out_cotangents=cotangents
        )
        expected = _adam_reference(g, m1, m2, *cotangents, weight_decay=decay)
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(value, reference, rtol=2e-5, atol=2e-5)
        _assert_certificate(compiled, "optimizer_vjp")


def test_sm120_adafactor_full_and_factored_exact_certificates():
    from tessera.autodiff.vjp import get_vjp

    rng = np.random.default_rng(1311)
    # The jit'd forwards above declare step=2, matching the `"step": 1` state
    # the reference VJPs carry (the update being differentiated is step 2).
    kwargs = {"lr": 0.003, "beta2": 0.91, "eps": 1.0e-7}
    for compiled, shape, topology in (
        (_adafactor_full, (17,), "full"),
        (_adafactor_factored, (3, 5), "factored"),
    ):
        p = rng.normal(size=shape).astype(np.float32)
        g = rng.normal(scale=0.2, size=shape).astype(np.float32)
        dy = rng.normal(size=shape).astype(np.float32)
        if topology == "full":
            state = rng.uniform(0.1, 0.3, size=shape).astype(np.float32)
            actual = compiled.native_backward(p, g, state, out_cotangents=dy)
            expected = get_vjp("adafactor")(
                dy, p, g, {"v": {"v": state, "factored": False}, "step": 1},
                **kwargs,
            )
            references = (expected[0], expected[1], expected[2]["v"]["v"])
        else:
            row = rng.uniform(0.1, 0.3, size=shape[:-1]).astype(np.float32)
            col = rng.uniform(0.1, 0.3, size=shape[-1]).astype(np.float32)
            actual = compiled.native_backward(p, g, row, col, out_cotangents=dy)
            expected = get_vjp("adafactor")(
                dy, p, g,
                {"v": {"row": row, "col": col, "factored": True}, "step": 1},
                **kwargs,
            )
            references = (
                expected[0], expected[1], expected[2]["v"]["row"],
                expected[2]["v"]["col"],
            )
        for value, reference in zip(actual, references, strict=True):
            np.testing.assert_allclose(value, reference, rtol=4e-4, atol=4e-5)
        _assert_certificate(compiled, "adafactor_vjp", topology)
