from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.nvidia_training import (
    NVIDIATrainingPackage,
    package_broadcast_gradient,
    package_class_backward,
    package_deltanet_backward,
    package_fused_loss_optimizer,
    package_loss_backward,
    package_lion_backward,
    package_norm_backward,
    package_optimizer,
)
from tessera.runtime import RuntimeArtifact, _submit_nvidia_sm120_native, launch
from tests._support.nvidia import nvidia_cuda_host_ready


@ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=("x", "gamma"))
def _nvidia_rmsnorm(x, gamma):
    return ts.ops.rmsnorm(x, gamma=gamma, eps=1.0e-5)


@ts.jit(
    target="nvidia_sm120",
    autodiff="reverse",
    wrt=("prediction", "target"),
)
def _nvidia_huber(prediction, target):
    return ts.ops.huber_loss(
        prediction, target, delta=0.75, reduction="mean"
    )


@ts.jit(
    target="nvidia_sm120", autodiff="reverse",
    wrt=("q", "k", "v", "gate", "beta", "decay"),
)
def _nvidia_affine_deltanet(q, k, v, gate, beta, decay):
    return ts.ops.gated_deltanet(q, k, v, gate, beta, decay, causal=True)


@ts.jit(
    target="nvidia_sm120", autodiff="reverse",
    wrt=("q", "k", "v", "gate", "beta", "decay"),
)
def _nvidia_modified_erase_deltanet(q, k, v, gate, beta, decay):
    return ts.ops.modified_delta_attention(
        q, k, v, gate, beta, decay, causal=True, erase=True
    )


def _require_cuda() -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")


def _invoke(
    package: NVIDIATrainingPackage,
    buffers: dict[str, np.ndarray],
    **scalars: int,
) -> object:
    return _submit_nvidia_sm120_native(
        package.image, package.descriptor, buffers, scalars, None
    )


def _storage_dtype(storage: str):
    if storage == "f16":
        return np.float16
    if storage == "bf16":
        return pytest.importorskip("ml_dtypes").bfloat16
    return np.float32


def _norm_reference(
    x: np.ndarray, gamma: np.ndarray, dy: np.ndarray, kind: str, epsilon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=1, keepdims=True) if kind == "layernorm" else 0.0
    centered = x - mean
    rstd = 1.0 / np.sqrt(np.mean(centered * centered, axis=1, keepdims=True) + epsilon)
    xhat = centered * rstd
    gdy = dy * gamma
    if kind == "layernorm":
        dx = (
            gdy
            - np.mean(gdy, axis=1, keepdims=True)
            - xhat * np.mean(gdy * xhat, axis=1, keepdims=True)
        ) * rstd
        dbeta = dy.sum(axis=0)
    else:
        dx = (gdy - xhat * np.mean(gdy * xhat, axis=1, keepdims=True)) * rstd
        dbeta = np.zeros(x.shape[1], np.float32)
    return dx, (dy * xhat).sum(axis=0), dbeta


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("kind", ["rmsnorm", "layernorm"])
@pytest.mark.parametrize("shape", [(3, 7), (5, 33)])
def test_dynamic_affine_norm_backward_executes_on_sm120(
    kind: str, shape: tuple[int, int]
) -> None:
    _require_cuda()
    rng = np.random.default_rng(19)
    x = rng.normal(size=shape).astype(np.float32)
    gamma = rng.normal(size=shape[1]).astype(np.float32)
    dy = rng.normal(size=shape).astype(np.float32)
    dx = np.empty_like(x)
    dgamma = np.empty_like(gamma)
    dbeta = np.empty_like(gamma)
    package = package_norm_backward(kind=kind, epsilon=1.0e-5)
    got = _invoke(
        package,
        {
            "x": x,
            "gamma": gamma,
            "dy": dy,
            "dx": dx,
            "dgamma": dgamma,
            "dbeta": dbeta,
        },
        Rows=shape[0],
        Columns=shape[1],
    )
    expected = _norm_reference(x, gamma, dy, kind, 1.0e-5)
    for actual, reference in zip(got, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
    metrics = package.image.resource_record.metrics
    assert metrics["spill_store_bytes"] == metrics["spill_load_bytes"] == 0


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("kind", ["rmsnorm", "layernorm"])
def test_low_precision_norm_backward_preserves_storage_and_accumulates_fp32(
    storage: str, kind: str
) -> None:
    _require_cuda()
    dtype = _storage_dtype(storage)
    rng = np.random.default_rng(119)
    x = rng.normal(size=(4, 19)).astype(dtype)
    gamma = rng.normal(size=19).astype(dtype)
    dy = rng.normal(size=x.shape).astype(dtype)
    package = package_norm_backward(
        kind=kind, epsilon=1.0e-5, storage=storage
    )
    got = _invoke(
        package,
        {
            "x": x,
            "gamma": gamma,
            "dy": dy,
            "dx": np.empty_like(x),
            "dgamma": np.empty(19, np.float32),
            "dbeta": np.empty(19, np.float32),
        },
        Rows=4,
        Columns=19,
    )
    expected = _norm_reference(
        x.astype(np.float32),
        gamma.astype(np.float32),
        dy.astype(np.float32),
        kind,
        1.0e-5,
    )
    assert got[0].dtype == dtype
    assert got[1].dtype == got[2].dtype == np.float32
    np.testing.assert_allclose(
        got[0].astype(np.float32), expected[0], rtol=8e-3, atol=8e-3
    )
    np.testing.assert_allclose(got[1], expected[1], rtol=4e-3, atol=4e-3)
    np.testing.assert_allclose(got[2], expected[2], rtol=4e-3, atol=4e-3)
    assert package.descriptor.provenance["accum"] == "fp32"


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize(
    ("kind", "parameter"),
    [("mse", 1.0), ("mae", 1.0), ("huber", 0.7), ("smooth_l1", 0.6), ("bce", 1.0)],
)
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_regression_and_binary_loss_backward_executes_on_sm120(
    kind: str, parameter: float, reduction: str
) -> None:
    _require_cuda()
    prediction = np.array([-90.0, -0.7, 0.0, 0.4, 90.0], np.float32)
    target = np.array([0.0, 0.3, 0.5, -0.2, 1.0], np.float32)
    error = prediction - target
    if kind == "mse":
        local = 2.0 * error
        target_local = -local
    elif kind == "mae":
        local = np.sign(error)
        target_local = -local
    elif kind == "huber":
        local = np.where(np.abs(error) <= parameter, error, parameter * np.sign(error))
        target_local = -local
    elif kind == "smooth_l1":
        local = np.where(np.abs(error) < parameter, error / parameter, np.sign(error))
        target_local = -local
    else:
        local = np.empty_like(prediction)
        positive = prediction >= 0
        local[positive] = 1.0 / (1.0 + np.exp(-prediction[positive]))
        exp_value = np.exp(prediction[~positive])
        local[~positive] = exp_value / (1.0 + exp_value)
        local -= target
        target_local = -prediction
    dy = (
        np.linspace(0.4, 1.2, prediction.size, dtype=np.float32)
        if reduction == "none"
        else np.array([0.75], np.float32)
    )
    upstream = dy if reduction == "none" else dy[0]
    if reduction == "mean":
        upstream = upstream / prediction.size
    dprediction = np.empty_like(prediction)
    dtarget = np.empty_like(target)
    package = package_loss_backward(
        kind=kind, parameter=parameter, reduction=reduction
    )
    got_prediction, got_target = _invoke(
        package,
        {
            "prediction": prediction,
            "target": target,
            "dy": dy,
            "dprediction": dprediction,
            "dtarget": dtarget,
        },
        N=prediction.size,
    )
    np.testing.assert_allclose(
        got_prediction, local * upstream, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        got_target, target_local * upstream, rtol=2e-6, atol=2e-6
    )
    assert np.isfinite(got_prediction).all()


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("kind", ["mse", "bce"])
def test_low_precision_loss_backward_preserves_storage(
    storage: str, kind: str
) -> None:
    _require_cuda()
    dtype = _storage_dtype(storage)
    prediction = np.array([-2.0, -0.3, 0.25, 1.5], dtype=dtype)
    target = np.array([0.0, 0.4, 0.8, 1.0], dtype=dtype)
    dy = np.array([0.75], dtype=dtype)
    package = package_loss_backward(
        kind=kind, reduction="mean", storage=storage
    )
    got_prediction, got_target = _invoke(
        package,
        {
            "prediction": prediction,
            "target": target,
            "dy": dy,
            "dprediction": np.empty_like(prediction),
            "dtarget": np.empty_like(target),
        },
        N=prediction.size,
    )
    prediction32, target32 = prediction.astype(np.float32), target.astype(np.float32)
    if kind == "mse":
        local = 2.0 * (prediction32 - target32)
        target_local = -local
    else:
        local = 1.0 / (1.0 + np.exp(-prediction32)) - target32
        target_local = -prediction32
    scale = np.float32(float(dy[0]) / prediction.size)
    assert got_prediction.dtype == got_target.dtype == dtype
    np.testing.assert_allclose(
        got_prediction.astype(np.float32), local * scale, rtol=5e-3, atol=5e-3
    )
    np.testing.assert_allclose(
        got_target.astype(np.float32),
        target_local * scale,
        rtol=5e-3,
        atol=5e-3,
    )


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("kind", ["kl", "js"])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_distribution_loss_backward_matches_shared_vjp(
    kind: str, reduction: str
) -> None:
    _require_cuda()
    p = np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]], np.float32)
    first = np.log(p) if kind == "kl" else p
    second = np.array([[0.3, 0.4, 0.3], [0.25, 0.5, 0.25]], np.float32)
    dy = (
        np.array([0.7, 1.2], np.float32)
        if reduction == "none"
        else np.array([0.9], np.float32)
    )
    package = package_loss_backward(
        kind=kind,
        reduction=reduction,
        mean_denominator=first.shape[0] if reduction == "mean" else None,
    )
    launch_dy = (
        np.broadcast_to(dy[:, None], first.shape).copy().reshape(-1)
        if reduction == "none"
        else dy
    )
    got = _invoke(
        package,
        {
            "prediction": first.reshape(-1),
            "target": second.reshape(-1),
            "dy": launch_dy,
            "dprediction": np.empty(first.size, np.float32),
            "dtarget": np.empty(first.size, np.float32),
        },
        N=first.size,
    )
    upstream = dy[:, None] if reduction == "none" else dy[0]
    if reduction == "mean":
        upstream = upstream / first.shape[0]
    if kind == "kl":
        probability = np.exp(first)
        expected_first = probability * (
            first - np.log(np.maximum(second, 1.0e-12)) + 1.0
        ) * upstream
        expected_second = (
            -probability / np.maximum(second, 1.0e-12) * upstream
        )
    else:
        midpoint = 0.5 * (first + second)
        expected_first = (
            0.5 * np.log(first / midpoint) * upstream
        )
        expected_second = (
            0.5 * np.log(second / midpoint) * upstream
        )
    np.testing.assert_allclose(
        got[0].reshape(first.shape), expected_first, rtol=3e-6, atol=3e-6
    )
    np.testing.assert_allclose(
        got[1].reshape(second.shape), expected_second, rtol=3e-6, atol=3e-6
    )


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
def test_general_broadcast_gradient_reduction_executes_on_sm120(
    storage: str,
) -> None:
    _require_cuda()
    dtype = _storage_dtype(storage)
    input_shape = (2, 3, 4)
    output_shape = (1, 3, 1)
    values = np.arange(24, dtype=np.float32).reshape(input_shape).astype(dtype)
    package = package_broadcast_gradient(
        input_shape=input_shape,
        output_shape=output_shape,
        storage=storage,
    )
    output = _invoke(
        package,
        {"input": values.reshape(-1), "output": np.empty(3, dtype=dtype)},
        InputN=24,
        OutputN=3,
    )
    expected = values.astype(np.float32).sum(axis=(0, 2), keepdims=True)
    assert output.dtype == dtype
    np.testing.assert_allclose(
        output.reshape(output_shape).astype(np.float32),
        expected,
        rtol=5e-3,
        atol=5e-3,
    )


@pytest.mark.hardware_nvidia
def test_cross_entropy_backward_is_stable_and_rejects_bad_labels() -> None:
    _require_cuda()
    logits = np.array(
        [[1000.0, 999.0, -1000.0], [-80.0, 0.0, 80.0]], np.float32
    )
    labels = np.array([1, 2], np.int64)
    dy = np.array([0.5, 1.25], np.float32)
    output = np.empty_like(logits)
    package = package_class_backward(label_smoothing=0.1)
    got = _invoke(
        package,
        {"logits": logits, "labels": labels, "dy": dy, "dlogits": output},
        Rows=2,
        Classes=3,
    )
    shifted = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    target = np.full_like(logits, 0.1 / 3.0)
    target[np.arange(2), labels] += 0.9
    np.testing.assert_allclose(got, (probability - target) * dy[:, None], atol=2e-6)
    with pytest.raises(RuntimeError, match="outside the class extent"):
        _invoke(
            package,
            {
                "logits": logits,
                "labels": np.array([1, 3], np.int64),
                "dy": dy,
                "dlogits": output,
            },
            Rows=2,
            Classes=3,
        )


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("kind", ["sgd", "momentum", "nesterov", "adam", "adamw"])
def test_optimizer_updates_execute_on_sm120(kind: str) -> None:
    _require_cuda()
    n = 17
    param = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    grad = np.linspace(0.3, -0.2, n, dtype=np.float32)
    velocity = np.linspace(-0.1, 0.1, n, dtype=np.float32)
    moment1 = np.linspace(0.02, 0.08, n, dtype=np.float32)
    moment2 = np.linspace(0.1, 0.3, n, dtype=np.float32)
    package = package_optimizer(
        kind=kind, lr=0.01, momentum=0.85, beta1=0.8, beta2=0.95,
        epsilon=1.0e-6, weight_decay=0.03, step=3,
    )
    buffers = {"param": param, "grad": grad, "output": np.empty_like(param)}
    if kind in {"momentum", "nesterov"}:
        buffers.update(
            velocity=velocity, velocity_output=np.empty_like(velocity)
        )
    elif kind in {"adam", "adamw"}:
        buffers.update(
            moment1=moment1,
            moment2=moment2,
            moment1_output=np.empty_like(moment1),
            moment2_output=np.empty_like(moment2),
        )
    got = _invoke(package, buffers, N=n)
    if kind == "sgd":
        np.testing.assert_allclose(got, param - 0.01 * grad, atol=1e-7)
    elif kind in {"momentum", "nesterov"}:
        velocity_new = 0.85 * velocity + grad
        update = grad + 0.85 * velocity_new if kind == "nesterov" else velocity_new
        np.testing.assert_allclose(got[0], param - 0.01 * update, atol=1e-7)
        np.testing.assert_allclose(got[1], velocity_new, atol=1e-7)
    else:
        m_new = 0.8 * moment1 + 0.2 * grad
        v_new = 0.95 * moment2 + 0.05 * grad * grad
        update = (m_new / (1.0 - 0.8**3)) / (
            np.sqrt(v_new / (1.0 - 0.95**3)) + 1.0e-6
        )
        if kind == "adamw":
            update += 0.03 * param
        np.testing.assert_allclose(got[0], param - 0.01 * update, atol=2e-7)
        np.testing.assert_allclose(got[1], m_new, atol=1e-7)
        np.testing.assert_allclose(got[2], v_new, atol=1e-7)


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_low_precision_class_optimizer_and_fused_routes(
    storage: str,
) -> None:
    _require_cuda()
    dtype = _storage_dtype(storage)
    logits = np.array([[2.0, -1.0, 0.5], [-0.5, 1.5, 0.25]], dtype=dtype)
    labels = np.array([0, 2], np.int64)
    dy = np.array([0.75, 1.25], dtype=dtype)
    class_package = package_class_backward(storage=storage)
    dlogits = _invoke(
        class_package,
        {
            "logits": logits,
            "labels": labels,
            "dy": dy,
            "dlogits": np.empty_like(logits),
        },
        Rows=2,
        Classes=3,
    )
    logits32 = logits.astype(np.float32)
    shifted = logits32 - logits32.max(axis=1, keepdims=True)
    probability = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    probability[np.arange(2), labels] -= 1.0
    np.testing.assert_allclose(
        dlogits.astype(np.float32),
        probability * dy.astype(np.float32)[:, None],
        rtol=5e-3,
        atol=5e-3,
    )

    param = np.array([1.0, -0.5, 0.25, 2.0], dtype=dtype)
    grad = np.array([0.2, -0.1, 0.4, -0.3], dtype=dtype)
    moment1 = np.zeros(4, np.float32)
    moment2 = np.full(4, 0.1, np.float32)
    optimizer = package_optimizer(
        kind="adamw",
        lr=0.01,
        beta1=0.9,
        beta2=0.99,
        epsilon=1.0e-6,
        weight_decay=0.02,
        step=2,
        storage=storage,
    )
    updated, next_m1, next_m2 = _invoke(
        optimizer,
        {
            "param": param,
            "grad": grad,
            "moment1": moment1,
            "moment2": moment2,
            "output": np.empty_like(param),
            "moment1_output": np.empty_like(moment1),
            "moment2_output": np.empty_like(moment2),
        },
        N=4,
    )
    assert updated.dtype == dtype
    assert next_m1.dtype == next_m2.dtype == np.float32

    fused = package_fused_loss_optimizer(
        loss_kind="mse",
        optimizer="sgd",
        parameter=1.0,
        reduction="mean",
        lr=0.02,
        storage=storage,
    )
    prediction = np.array([0.2, -0.3, 0.7, 1.1], dtype=dtype)
    target = np.array([0.0, 0.4, 0.5, 0.9], dtype=dtype)
    seed = np.array([0.8], dtype=dtype)
    fused_output, dtarget = _invoke(
        fused,
        {
            "prediction": prediction,
            "target": target,
            "dy": seed,
            "param": param,
            "output": np.empty_like(param),
            "dtarget": np.empty_like(target),
        },
        N=4,
    )
    gradient = (
        2.0
        * (prediction.astype(np.float32) - target.astype(np.float32))
        * (float(seed[0]) / 4.0)
    )
    np.testing.assert_allclose(
        fused_output.astype(np.float32),
        param.astype(np.float32) - 0.02 * gradient,
        rtol=5e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        dtarget.astype(np.float32), -gradient, rtol=5e-3, atol=5e-3
    )


@pytest.mark.hardware_nvidia
def test_fused_loss_adamw_matches_composed_update_and_cache_identity() -> None:
    _require_cuda()
    n = 257
    rng = np.random.default_rng(41)
    prediction = rng.normal(size=n).astype(np.float32)
    target = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    dy = np.array([0.8], np.float32)
    param = rng.normal(size=n).astype(np.float32)
    moment1 = rng.normal(scale=0.1, size=n).astype(np.float32)
    moment2 = rng.uniform(0.05, 0.2, size=n).astype(np.float32)
    kwargs = dict(
        loss_kind="bce", optimizer="adamw", parameter=1.0, reduction="mean",
        lr=0.003, beta1=0.9, beta2=0.99, epsilon=1.0e-6,
        weight_decay=0.02, step=4,
    )
    package = package_fused_loss_optimizer(**kwargs)
    warm = package_fused_loss_optimizer(**kwargs)
    assert warm.image.image_digest == package.image.image_digest
    assert warm.descriptor.descriptor_digest == package.descriptor.descriptor_digest
    assert warm.image.compile_state == "warm_cache"
    buffers = {
        "prediction": prediction,
        "target": target,
        "dy": dy,
        "param": param,
        "moment1": moment1,
        "moment2": moment2,
        "output": np.empty_like(param),
        "moment1_output": np.empty_like(moment1),
        "moment2_output": np.empty_like(moment2),
        "dtarget": np.empty_like(target),
    }
    output, m_output, v_output, dtarget = _invoke(package, buffers, N=n)
    sigmoid = 1.0 / (1.0 + np.exp(-prediction))
    grad = (sigmoid - target) * (dy[0] / n)
    m_expected = 0.9 * moment1 + 0.1 * grad
    v_expected = 0.99 * moment2 + 0.01 * grad * grad
    update = (m_expected / (1.0 - 0.9**4)) / (
        np.sqrt(v_expected / (1.0 - 0.99**4)) + 1.0e-6
    ) + 0.02 * param
    np.testing.assert_allclose(output, param - 0.003 * update, atol=3e-7)
    np.testing.assert_allclose(m_output, m_expected, atol=2e-7)
    np.testing.assert_allclose(v_output, v_expected, atol=2e-7)
    np.testing.assert_allclose(dtarget, -prediction * (dy[0] / n), atol=2e-7)


@pytest.mark.hardware_nvidia
def test_runtime_artifact_training_paths_use_cuda_driver() -> None:
    _require_cuda()
    rng = np.random.default_rng(73)

    x = rng.normal(size=(3, 9)).astype(np.float32)
    gamma = rng.normal(size=9).astype(np.float32)
    dy = rng.normal(size=x.shape).astype(np.float32)
    norm = RuntimeArtifact(
        metadata={
            "target": "nvidia_sm120",
            "compiler_path": "nvidia_norm_bwd_compiled",
            "executable": True,
            "execution_kind": "native_gpu",
            "execution_mode": "cuda_driver",
            "arg_names": ["x", "gamma", "dy"],
            "out_cotangent": "dy",
            "ops": [{
                "op_name": "tessera.layer_norm",
                "operands": ["x", "gamma"],
                "kwargs": {"eps": 1.0e-5},
            }],
        }
    )
    result = launch(norm, (x, gamma, dy))
    assert result["ok"], result
    assert result["execution_mode"] == "cuda_driver"
    expected = _norm_reference(x, gamma, dy, "layernorm", 1.0e-5)
    np.testing.assert_allclose(result["output"][0], expected[0], atol=2e-5)
    np.testing.assert_allclose(result["output"][1], expected[1], atol=2e-5)

    prediction = rng.normal(size=37).astype(np.float32)
    target = rng.normal(size=37).astype(np.float32)
    loss = RuntimeArtifact(
        metadata={
            "target": "nvidia_sm120",
            "compiler_path": "nvidia_regression_loss_bwd_compiled",
            "executable": True,
            "execution_kind": "native_gpu",
            "execution_mode": "cuda_driver",
            "arg_names": ["prediction", "target", "dy"],
            "out_cotangent": "dy",
            "ops": [{
                "op_name": "tessera.loss.mse",
                "operands": ["prediction", "target"],
                "kwargs": {"reduction": "mean"},
            }],
        }
    )
    result = launch(loss, (prediction, target, np.array(0.75, np.float32)))
    assert result["ok"], result
    assert result["execution_mode"] == "cuda_driver"
    scale = np.float32(1.5 / prediction.size)
    np.testing.assert_allclose(
        result["output"][0], (prediction - target) * scale, atol=2e-6
    )

    logits = rng.normal(size=(5, 7)).astype(np.float32)
    labels = np.array([0, 3, 6, 2, 1], np.int64)
    class_loss = RuntimeArtifact(
        metadata={
            "target": "nvidia_sm120",
            "compiler_path": "nvidia_class_loss_bwd_compiled",
            "executable": True,
            "execution_kind": "native_gpu",
            "execution_mode": "cuda_driver",
            "arg_names": ["logits", "labels", "dy"],
            "out_cotangent": "dy",
            "ops": [{
                "op_name": "tessera.loss.cross_entropy",
                "operands": ["logits", "labels"],
                "kwargs": {"reduction": "mean", "label_smoothing": 0.1},
            }],
        }
    )
    result = launch(
        class_loss, (logits, labels, np.array(0.5, np.float32))
    )
    assert result["ok"], result
    assert result["execution_mode"] == "cuda_driver"
    shifted = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    expected_target = np.full_like(logits, 0.1 / logits.shape[1])
    expected_target[np.arange(5), labels] += 0.9
    np.testing.assert_allclose(
        result["output"],
        (probability - expected_target) * (0.5 / logits.shape[0]),
        atol=2e-6,
    )

    fused = RuntimeArtifact(
        metadata={
            "target": "nvidia_sm120",
            "compiler_path": "nvidia_training_loss_sgd_compiled",
            "executable": True,
            "execution_kind": "native_gpu",
            "execution_mode": "cuda_driver",
            "arg_names": ["prediction", "target", "dy", "parameter"],
            "ops": [{
                "op_name": "tessera.training.loss_sgd",
                "operands": ["prediction", "target", "dy", "parameter"],
                "kwargs": {
                    "kind": "mse",
                    "parameter": 1.0,
                    "reduction": "mean",
                    "lr": 0.02,
                },
            }],
        }
    )
    parameter = rng.normal(size=prediction.shape).astype(np.float32)
    result = launch(
        fused,
        (prediction, target, np.array(0.75, np.float32), parameter),
    )
    assert result["ok"], result
    assert result["execution_mode"] == "cuda_driver"
    gradient = 2.0 * (prediction - target) * (0.75 / prediction.size)
    np.testing.assert_allclose(
        result["output"][0], parameter - 0.02 * gradient, atol=2e-6
    )
    np.testing.assert_allclose(
        result["output"][1], -gradient, atol=2e-6
    )


@pytest.mark.hardware_nvidia
def test_runtime_loss_backward_reduces_broadcasted_operand_gradients() -> None:
    _require_cuda()
    prediction = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10.0
    target = np.array([[[0.2], [0.5], [0.8]]], np.float32)
    artifact = RuntimeArtifact(
        metadata={
            "target": "nvidia_sm120",
            "compiler_path": "nvidia_regression_loss_bwd_compiled",
            "executable": True,
            "execution_kind": "native_gpu",
            "execution_mode": "cuda_driver",
            "arg_names": ["prediction", "target", "dy"],
            "out_cotangent": "dy",
            "ops": [{
                "op_name": "tessera.loss.mse",
                "operands": ["prediction", "target"],
                "kwargs": {"reduction": "sum"},
            }],
        }
    )
    result = launch(
        artifact, (prediction, target, np.array(0.75, np.float32))
    )
    assert result["ok"], result
    expanded_target = np.broadcast_to(target, prediction.shape)
    dprediction = 1.5 * (prediction - expanded_target)
    np.testing.assert_allclose(result["output"][0], dprediction, atol=2e-6)
    np.testing.assert_allclose(
        result["output"][1],
        (-dprediction).sum(axis=(0, 2), keepdims=True),
        atol=2e-6,
    )


@pytest.mark.hardware_nvidia
def test_runtime_optimizer_path_uses_compiler_owned_ptx_package() -> None:
    _require_cuda()
    n = 31
    param = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    grad = np.linspace(0.2, -0.3, n, dtype=np.float32)
    moment1 = np.zeros(n, np.float32)
    moment2 = np.full(n, 0.1, np.float32)
    artifact = RuntimeArtifact(
        metadata={
            "target": "nvidia_sm120",
            "compiler_path": "nvidia_optimizer_compiled",
            "executable": True,
            "execution_kind": "native_gpu",
            "execution_mode": "cuda_driver",
            "arg_names": ["param", "grad", "moment1", "moment2"],
            "ops": [{
                "op_name": "tessera.adamw",
                "operands": ["param", "grad", "moment1", "moment2"],
                "kwargs": {
                    "extras": ["m", "v"],
                    "lr": 0.01,
                    "beta1": 0.9,
                    "beta2": 0.99,
                    "eps": 1.0e-6,
                    "weight_decay": 0.02,
                    "step": 2,
                },
            }],
        }
    )
    result = launch(artifact, (param, grad, moment1, moment2))
    assert result["ok"], result
    assert result["execution_mode"] == "cuda_runtime"
    m_expected = 0.1 * grad
    v_expected = 0.99 * moment2 + 0.01 * grad * grad
    update = (m_expected / (1.0 - 0.9**2)) / (
        np.sqrt(v_expected / (1.0 - 0.99**2)) + 1.0e-6
    ) + 0.02 * param
    np.testing.assert_allclose(
        result["output"][0], param - 0.01 * update, atol=2e-7
    )
    np.testing.assert_allclose(result["output"][1], m_expected, atol=1e-7)
    np.testing.assert_allclose(result["output"][2], v_expected, atol=1e-7)


@pytest.mark.hardware_nvidia
def test_public_jit_native_backward_binds_sm120_training_packages() -> None:
    _require_cuda()
    rng = np.random.default_rng(91)
    x = rng.normal(size=(4, 11)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=11).astype(np.float32)
    dy = rng.normal(size=x.shape).astype(np.float32)
    dx, dgamma = _nvidia_rmsnorm.native_backward(
        x, gamma, out_cotangents=dy
    )
    expected = _norm_reference(x, gamma, dy, "rmsnorm", 1.0e-5)
    np.testing.assert_allclose(dx, expected[0], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(dgamma, expected[1], atol=2e-5, rtol=2e-5)
    assert _nvidia_rmsnorm.last_backward_execution["compiler_path"] == (
        "nvidia_norm_bwd_compiled"
    )
    assert _nvidia_rmsnorm.last_backward_execution["evidence_target"] == (
        "nvidia_sm120"
    )

    error = np.array([-1.5, -0.75, -0.2, 0.0, 0.4, 1.2], np.float32)
    target = np.zeros_like(error)
    seed = np.array(1.25, np.float32)
    dprediction, dtarget = _nvidia_huber.native_backward(
        error, target, out_cotangents=seed
    )
    local = np.where(
        np.abs(error) <= 0.75, error, 0.75 * np.sign(error)
    )
    expected_gradient = local * (seed / error.size)
    np.testing.assert_allclose(dprediction, expected_gradient, atol=2e-6)
    np.testing.assert_allclose(dtarget, -expected_gradient, atol=2e-6)
    assert _nvidia_huber.last_backward_execution["compiler_path"] == (
        "nvidia_regression_loss_bwd_compiled"
    )


@pytest.mark.hardware_nvidia
def test_lion_stop_sign_backward_executes_on_sm120() -> None:
    _require_cuda()
    rng = np.random.default_rng(725)
    shape = (5, 19)
    param, grad = (rng.normal(size=shape).astype(np.float32) for _ in range(2))
    moment = rng.normal(size=shape).astype(np.float32)
    dparam_out = rng.normal(size=shape).astype(np.float32)
    dmoment_out = rng.normal(size=shape).astype(np.float32)
    package = package_lion_backward(lr=0.004, beta2=0.93, weight_decay=0.025)
    got = _invoke(
        package,
        {
            "param": param.reshape(-1), "grad": grad.reshape(-1),
            "moment": moment.reshape(-1),
            "dparam_out": dparam_out.reshape(-1),
            "dmoment_out": dmoment_out.reshape(-1),
            "dparam": np.empty(param.size, np.float32),
            "dgrad": np.empty(param.size, np.float32),
            "dmoment": np.empty(param.size, np.float32),
        },
        N=param.size,
    )
    expected = (
        dparam_out * (1.0 - 0.004 * 0.025),
        dmoment_out * (1.0 - 0.93),
        dmoment_out * 0.93,
    )
    for actual, reference in zip(got, expected, strict=True):
        np.testing.assert_allclose(
            actual.reshape(shape), reference, rtol=2e-6, atol=2e-6
        )


@pytest.mark.hardware_nvidia
def test_plain_causal_deltanet_backward_executes_on_sm120() -> None:
    """The first CUDA DeltaNet VJP matches the shared analytic oracle."""
    _require_cuda()
    from tessera.autodiff.vjp import get_vjp
    rng = np.random.default_rng(730)
    q = rng.normal(size=(1, 1, 5, 3)).astype(np.float32)
    k = rng.normal(size=q.shape).astype(np.float32)
    v = rng.normal(size=(1, 1, 5, 2)).astype(np.float32)
    dy = rng.normal(size=v.shape).astype(np.float32)
    package = package_deltanet_backward()
    got = _invoke(package, {
        "q": q, "k": k, "v": v,
        "gate": np.zeros_like(v), "beta": np.ones((1, 1, 5), np.float32),
        "decay": np.ones((1, 1, 5), np.float32), "dy": dy,
        "dq": np.empty_like(q), "dk": np.empty_like(k), "dv": np.empty_like(v),
        "dgate": np.empty_like(v), "dbeta": np.empty((1, 1, 5), np.float32),
        "ddecay": np.empty((1, 1, 5), np.float32),
    }, B=1, H=1, S=5, Dqk=3, Dv=2,
       HasGate=0, HasBeta=0, HasDecay=0, Erase=0, Modified=0)
    expected = get_vjp("gated_deltanet")(dy, q, k, v)
    for actual, reference in zip(got[:3], expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=3e-3, atol=3e-3)
    for unused in got[3:]:
        np.testing.assert_array_equal(unused, 0.0)


@pytest.mark.hardware_nvidia
def test_affine_gated_beta_decay_deltanet_backward_executes_on_sm120() -> None:
    """The v2 package differentiates the full affine recurrence and gate."""
    _require_cuda()
    from tessera.autodiff.vjp import get_vjp
    rng = np.random.default_rng(731)
    q = (0.1 * rng.normal(size=(1, 2, 4, 3))).astype(np.float32)
    k = (0.1 * rng.normal(size=q.shape)).astype(np.float32)
    v = (0.1 * rng.normal(size=(1, 2, 4, 2))).astype(np.float32)
    gate = rng.normal(size=v.shape).astype(np.float32)
    beta = rng.uniform(0.2, 0.9, size=(1, 2, 4)).astype(np.float32)
    decay = rng.uniform(0.7, 0.99, size=(1, 2, 4)).astype(np.float32)
    dy = rng.normal(size=v.shape).astype(np.float32)
    package = package_deltanet_backward()
    got = _invoke(package, {
        "q": q, "k": k, "v": v, "gate": gate, "beta": beta,
        "decay": decay, "dy": dy,
        "dq": np.empty_like(q), "dk": np.empty_like(k), "dv": np.empty_like(v),
        "dgate": np.empty_like(v), "dbeta": np.empty_like(beta),
        "ddecay": np.empty_like(decay),
    }, B=1, H=2, S=4, Dqk=3, Dv=2,
       HasGate=1, HasBeta=1, HasDecay=1, Erase=0, Modified=0)
    expected = get_vjp("gated_deltanet")(dy, q, k, v, gate, beta, decay)
    for actual, reference in zip(got, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=4e-3, atol=4e-3)


@pytest.mark.hardware_nvidia
def test_erase_modified_deltanet_serial_fill_backward_executes_on_sm120() -> None:
    """Erase plus modified normalization uses the CUDA-owned serial-fill VJP."""
    _require_cuda()
    from tessera.autodiff.vjp import get_vjp
    rng = np.random.default_rng(734)
    q = (0.1 * rng.normal(size=(1, 1, 4, 3))).astype(np.float32)
    k = (0.1 * rng.normal(size=q.shape)).astype(np.float32)
    v = (0.1 * rng.normal(size=(1, 1, 4, 2))).astype(np.float32)
    gate = rng.normal(size=v.shape).astype(np.float32)
    beta = rng.uniform(0.2, 0.9, size=(1, 1, 4)).astype(np.float32)
    decay = rng.uniform(0.7, 0.99, size=(1, 1, 4)).astype(np.float32)
    dy = rng.normal(size=v.shape).astype(np.float32)
    package = package_deltanet_backward()
    got = _invoke(package, {
        "q": q, "k": k, "v": v, "gate": gate, "beta": beta,
        "decay": decay, "dy": dy,
        "dq": np.empty_like(q), "dk": np.empty_like(k), "dv": np.empty_like(v),
        "dgate": np.empty_like(v), "dbeta": np.empty_like(beta),
        "ddecay": np.empty_like(decay),
    }, B=1, H=1, S=4, Dqk=3, Dv=2,
       HasGate=1, HasBeta=1, HasDecay=1, Erase=1, Modified=1)
    expected = get_vjp("modified_delta_attention")(
        dy, q, k, v, gate, beta, decay, erase=True
    )
    for name, actual, reference in zip(
        ("dq", "dk", "dv", "dgate", "dbeta", "ddecay"), got, expected, strict=True
    ):
        # Tight on purpose (DELTANET-BOUNDED-VJP-2026-08-31).  These inputs
        # produce gradients whose full scale is 1e-3..3e-2, and `dgate` peaks at
        # 1.1e-03 -- so the previous atol=5e-3 exceeded two of the six
        # gradients' entire range and would have passed on all-zero output.  The
        # measured device-vs-reference deviation is 4.7e-10 abs / 2.0e-07 rel
        # (f32 round-off), so this leaves ~50x margin while still catching the
        # bounded-correction defect that ROCm and x86 carried.
        np.testing.assert_allclose(
            actual, reference, rtol=1e-5, atol=1e-8, err_msg=name
        )


@pytest.mark.hardware_nvidia
def test_jit_affine_deltanet_backward_routes_v2_package_on_sm120() -> None:
    """JIT binds all optional inputs and maps only their requested gradients."""
    _require_cuda()
    from tessera.autodiff.vjp import get_vjp
    rng = np.random.default_rng(732)
    q = rng.normal(size=(1, 1, 4, 3)).astype(np.float32)
    k = rng.normal(size=q.shape).astype(np.float32)
    v = rng.normal(size=(1, 1, 4, 2)).astype(np.float32)
    gate = rng.normal(size=v.shape).astype(np.float32)
    beta = rng.uniform(0.2, 0.9, size=(1, 1, 4)).astype(np.float32)
    decay = rng.uniform(0.7, 0.99, size=(1, 1, 4)).astype(np.float32)
    dy = rng.normal(size=v.shape).astype(np.float32)
    got = _nvidia_affine_deltanet.native_backward(
        q, k, v, gate, beta, decay, out_cotangents=dy
    )
    expected = get_vjp("gated_deltanet")(dy, q, k, v, gate, beta, decay)
    for actual, reference in zip(got, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=4e-3, atol=4e-3)
    assert _nvidia_affine_deltanet.last_backward_execution["compiler_path"] == (
        "nvidia_deltanet_bwd_compiled"
    )


@pytest.mark.hardware_nvidia
def test_jit_erase_modified_deltanet_routes_serial_fill_on_sm120() -> None:
    """The public JIT route carries erase/modified flags through the v2 ABI."""
    _require_cuda()
    from tessera.autodiff.vjp import get_vjp
    rng = np.random.default_rng(734)
    q = (0.1 * rng.normal(size=(1, 1, 4, 3))).astype(np.float32)
    k = (0.1 * rng.normal(size=q.shape)).astype(np.float32)
    v = (0.1 * rng.normal(size=(1, 1, 4, 2))).astype(np.float32)
    gate = rng.normal(size=v.shape).astype(np.float32)
    beta = rng.uniform(0.2, 0.9, size=(1, 1, 4)).astype(np.float32)
    decay = rng.uniform(0.7, 0.99, size=(1, 1, 4)).astype(np.float32)
    dy = rng.normal(size=v.shape).astype(np.float32)
    got = _nvidia_modified_erase_deltanet.native_backward(
        q, k, v, gate, beta, decay, out_cotangents=dy
    )
    expected = get_vjp("modified_delta_attention")(
        dy, q, k, v, gate, beta, decay, erase=True
    )
    for actual, reference in zip(got, expected, strict=True):
        # Same tolerance rationale as the direct-package test above.
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-8)
    assert _nvidia_modified_erase_deltanet.last_backward_execution["compiler_path"] == (
        "nvidia_deltanet_bwd_compiled"
    )
