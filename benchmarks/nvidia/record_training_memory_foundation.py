"""Record exact-SM120 CUDA training and dynamic-memory foundation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Callable

import numpy as np

from tessera.compiler.nvidia_dynamic_smem import (
    package_local_expression_probe,
    package_path_max_probe,
)
from tessera.compiler.nvidia_rematerialization import (
    resolve_nvidia_rematerialization_policy,
)
from tessera.compiler.nvidia_training import (
    NVIDIATrainingPackage,
    package_class_backward,
    package_broadcast_gradient,
    package_fused_loss_optimizer,
    package_lion_backward,
    package_loss_backward,
    package_norm_backward,
    package_optimizer,
)
import tessera.runtime as rt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "benchmarks/baselines/nvidia_sm120_training_memory_foundation.json"
)


def _two_run_medians(
    sample: Callable[[], float], *, samples: int
) -> tuple[list[float], float]:
    medians: list[float] = []
    for _ in range(2):
        values = [sample() for _ in range(samples)]
        medians.append(float(statistics.median(values)))
    delta = abs(medians[1] - medians[0]) / min(medians) * 100.0
    return medians, delta


def _measure_package(
    package: Any,
    arrays: dict[str, np.ndarray],
    scalars: dict[str, int],
    *,
    block_size: int,
    samples: int,
    device_reps: int,
    e2e_reps: int,
) -> dict[str, Any]:
    args = {**arrays, **scalars}

    def device_sample() -> float:
        return (
            rt._nvidia_native_descriptor_device_latency(
                package.image,
                package.descriptor,
                args,
                reps=device_reps,
                warmup=10,
            )
            * 1_000_000.0
        )

    def e2e_sample() -> float:
        start = time.perf_counter_ns()
        for _ in range(e2e_reps):
            rt._submit_nvidia_sm120_native(
                package.image, package.descriptor, arrays, scalars, None
            )
        return (time.perf_counter_ns() - start) / e2e_reps

    # First launch includes driver JIT and is intentionally outside both
    # repeated-median domains.
    rt._submit_nvidia_sm120_native(
        package.image, package.descriptor, arrays, scalars, None
    )
    device, device_delta = _two_run_medians(device_sample, samples=samples)
    e2e, e2e_delta = _two_run_medians(e2e_sample, samples=samples)
    resources = rt._nvidia_native_descriptor_resources(
        package.image, package.descriptor, block_size=block_size
    )
    resource_fingerprint = hashlib.sha256(
        json.dumps(resources, sort_keys=True).encode()
    ).hexdigest()
    return {
        "artifact_identity": {
            "image_digest": package.image.image_digest,
            "descriptor_digest": package.descriptor.descriptor_digest,
            "compiler_fingerprint": package.image.compiler_fingerprint,
            "toolchain_fingerprint": package.image.toolchain_fingerprint,
        },
        "device_event_ns": device,
        "device_event_delta_pct": device_delta,
        "end_to_end_ns": e2e,
        "end_to_end_delta_pct": e2e_delta,
        "stable_4pct": device_delta <= 4.0 and e2e_delta <= 4.0,
        "resources": resources,
        "resource_fingerprint": resource_fingerprint,
    }


def _training_cases(n: int) -> list[tuple[str, NVIDIATrainingPackage, dict[str, np.ndarray], dict[str, int]]]:
    rng = np.random.default_rng(20260724)
    prediction = rng.normal(size=n).astype(np.float32)
    target = rng.uniform(-0.2, 1.0, size=n).astype(np.float32)
    dy = rng.normal(size=n).astype(np.float32)
    param = rng.normal(size=n).astype(np.float32)
    grad = rng.normal(scale=0.2, size=n).astype(np.float32)
    velocity = rng.normal(scale=0.1, size=n).astype(np.float32)
    m1 = rng.normal(scale=0.05, size=n).astype(np.float32)
    m2 = rng.uniform(0.05, 0.2, size=n).astype(np.float32)
    cases: list[
        tuple[str, NVIDIATrainingPackage, dict[str, np.ndarray], dict[str, int]]
    ] = []
    for kind, parameter in (
        ("mse", 1.0), ("mae", 1.0), ("huber", 0.7),
        ("smooth_l1", 0.6), ("bce", 1.0),
    ):
        package = package_loss_backward(
            kind=kind, parameter=parameter, reduction="none"
        )
        cases.append(
            (
                f"{kind}_backward",
                package,
                {
                    "prediction": prediction,
                    "target": target,
                    "dy": dy,
                    "dprediction": np.empty_like(prediction),
                    "dtarget": np.empty_like(target),
                },
                {"N": n},
            )
        )
    for kind in ("sgd", "momentum", "nesterov", "adam", "adamw"):
        package = package_optimizer(
            kind=kind, lr=0.003, momentum=0.9, beta1=0.9, beta2=0.99,
            epsilon=1.0e-6, weight_decay=0.02, step=4,
        )
        arrays: dict[str, np.ndarray] = {
            "param": param, "grad": grad, "output": np.empty_like(param)
        }
        if kind in {"momentum", "nesterov"}:
            arrays.update(
                velocity=velocity, velocity_output=np.empty_like(velocity)
            )
        elif kind in {"adam", "adamw"}:
            arrays.update(
                moment1=m1,
                moment2=m2,
                moment1_output=np.empty_like(m1),
                moment2_output=np.empty_like(m2),
            )
        cases.append((kind, package, arrays, {"N": n}))
    lion_moment = rng.normal(scale=0.05, size=n).astype(np.float32)
    cases.append(
        (
            "lion_backward",
            package_lion_backward(lr=0.003, beta2=0.99, weight_decay=0.02),
            {
                "param": param,
                "grad": grad,
                "moment": lion_moment,
                "dparam_out": rng.normal(size=n).astype(np.float32),
                "dmoment_out": rng.normal(size=n).astype(np.float32),
                "dparam": np.empty(n, np.float32),
                "dgrad": np.empty(n, np.float32),
                "dmoment": np.empty(n, np.float32),
            },
            {"N": n},
        )
    )
    for optimizer in ("sgd", "adamw"):
        package = package_fused_loss_optimizer(
            loss_kind="mse", optimizer=optimizer, parameter=1.0,
            reduction="none", lr=0.003, beta1=0.9, beta2=0.99,
            epsilon=1.0e-6, weight_decay=0.02, step=4,
        )
        arrays = {
            "prediction": prediction,
            "target": target,
            "dy": dy,
            "param": param,
            "output": np.empty_like(param),
            "dtarget": np.empty_like(target),
        }
        if optimizer == "adamw":
            arrays.update(
                moment1=m1,
                moment2=m2,
                moment1_output=np.empty_like(m1),
                moment2_output=np.empty_like(m2),
            )
        cases.append((f"fused_mse_{optimizer}", package, arrays, {"N": n}))
    return cases


def _warm_package(package: NVIDIATrainingPackage) -> NVIDIATrainingPackage:
    provenance = package.descriptor.provenance

    def number(name: str) -> float:
        value = provenance[name]
        if not isinstance(value, (int, float)):
            raise TypeError(f"provenance {name} must be numeric")
        return float(value)

    def integer(name: str) -> int:
        value = provenance[name]
        if not isinstance(value, int):
            raise TypeError(f"provenance {name} must be integer")
        return value

    def integer_tuple(name: str) -> tuple[int, ...]:
        value = provenance[name]
        if not isinstance(value, list) or not all(
            isinstance(item, int) for item in value
        ):
            raise TypeError(f"provenance {name} must be an integer list")
        return tuple(value)

    family = str(provenance["family"])
    storage = str(provenance["storage"])
    if family == "loss_backward":
        return package_loss_backward(
            kind=str(provenance["kind"]),
            parameter=number("parameter"),
            reduction=str(provenance["reduction"]),
            storage=storage,
            mean_denominator=(
                None
                if provenance["mean_denominator"] is None
                else integer("mean_denominator")
            ),
        )
    if family == "norm_backward":
        return package_norm_backward(
            kind=str(provenance["kind"]),
            epsilon=number("epsilon"),
            storage=storage,
        )
    if family == "class_backward":
        return package_class_backward(
            label_smoothing=number("label_smoothing"),
            reduction=str(provenance["reduction"]),
            storage=storage,
        )
    if family == "broadcast_gradient":
        return package_broadcast_gradient(
            input_shape=integer_tuple("input_shape"),
            output_shape=integer_tuple("output_shape"),
            storage=storage,
        )
    if family == "optimizer":
        return package_optimizer(
            kind=str(provenance["kind"]),
            lr=number("lr"),
            momentum=number("momentum"),
            beta1=number("beta1"),
            beta2=number("beta2"),
            epsilon=number("epsilon"),
            weight_decay=number("weight_decay"),
            step=integer("step"),
            storage=storage,
        )
    if family == "optimizer_backward" and provenance.get("kind") == "lion":
        return package_lion_backward(
            lr=number("lr"), beta2=number("beta2"),
            weight_decay=number("weight_decay"), storage=storage,
        )
    return package_fused_loss_optimizer(
        loss_kind=str(provenance["loss_kind"]),
        optimizer=str(provenance["optimizer"]),
        parameter=number("parameter"),
        reduction=str(provenance["reduction"]),
        lr=number("lr"),
        momentum=number("momentum"),
        beta1=number("beta1"),
        beta2=number("beta2"),
        epsilon=number("epsilon"),
        weight_decay=number("weight_decay"),
        step=integer("step"),
        storage=storage,
    )


def _low_precision_cases(
    n: int, rng: np.random.Generator
) -> list[
    tuple[
        str,
        NVIDIATrainingPackage,
        dict[str, np.ndarray],
        dict[str, int],
    ]
]:
    import ml_dtypes

    cases = []
    for storage, dtype in (("f16", np.float16), ("bf16", ml_dtypes.bfloat16)):
        prediction = rng.normal(size=n).astype(dtype)
        target = rng.normal(size=n).astype(dtype)
        dy = rng.normal(size=n).astype(dtype)
        loss = package_loss_backward(
            kind="mse", reduction="none", storage=storage
        )
        cases.append(
            (
                f"mse_backward_{storage}",
                loss,
                {
                    "prediction": prediction,
                    "target": target,
                    "dy": dy,
                    "dprediction": np.empty_like(prediction),
                    "dtarget": np.empty_like(target),
                },
                {"N": n},
            )
        )
        rows_count, columns = 64, 256
        norm = package_norm_backward(
            kind="rmsnorm" if storage == "f16" else "layernorm",
            storage=storage,
        )
        cases.append(
            (
                f"{'rmsnorm' if storage == 'f16' else 'layernorm'}_backward_{storage}",
                norm,
                {
                    "x": rng.normal(size=(rows_count, columns)).astype(dtype),
                    "gamma": rng.normal(size=columns).astype(dtype),
                    "dy": rng.normal(size=(rows_count, columns)).astype(dtype),
                    "dx": np.empty((rows_count, columns), dtype=dtype),
                    "dgamma": np.empty(columns, np.float32),
                    "dbeta": np.empty(columns, np.float32),
                },
                {"Rows": rows_count, "Columns": columns},
            )
        )
        optimizer = package_optimizer(
            kind="adamw",
            lr=0.003,
            beta1=0.9,
            beta2=0.99,
            epsilon=1.0e-6,
            weight_decay=0.02,
            step=4,
            storage=storage,
        )
        cases.append(
            (
                f"adamw_{storage}",
                optimizer,
                {
                    "param": rng.normal(size=n).astype(dtype),
                    "grad": rng.normal(scale=0.2, size=n).astype(dtype),
                    "moment1": rng.normal(scale=0.05, size=n).astype(np.float32),
                    "moment2": rng.uniform(0.05, 0.2, size=n).astype(np.float32),
                    "output": np.empty(n, dtype=dtype),
                    "moment1_output": np.empty(n, np.float32),
                    "moment2_output": np.empty(n, np.float32),
                },
                {"N": n},
            )
        )
    return cases


def record(
    *,
    samples: int = 10,
    device_reps: int = 100,
    e2e_reps: int = 10,
) -> dict[str, Any]:
    # Keep pointwise rows large enough that WSL host-launch jitter does not
    # dominate the operation while still exercising a realistic training tile.
    n = 65_536
    rows: list[dict[str, Any]] = []
    cases = _training_cases(n)

    rng = np.random.default_rng(31)
    cases.extend(_low_precision_cases(n, rng))
    for kind in ("kl", "js"):
        prediction = rng.uniform(0.05, 0.95, size=n).astype(np.float32)
        target = rng.uniform(0.05, 0.95, size=n).astype(np.float32)
        package = package_loss_backward(kind=kind, reduction="none")
        cases.append(
            (
                f"{kind}_backward",
                package,
                {
                    "prediction": prediction,
                    "target": target,
                    "dy": rng.normal(size=n).astype(np.float32),
                    "dprediction": np.empty(n, np.float32),
                    "dtarget": np.empty(n, np.float32),
                },
                {"N": n},
            )
        )
    broadcast_input_shape = (64, 32, 128)
    broadcast_output_shape = (1, 32, 1)
    broadcast = package_broadcast_gradient(
        input_shape=broadcast_input_shape,
        output_shape=broadcast_output_shape,
    )
    cases.append(
        (
            "broadcast_gradient_reduce",
            broadcast,
            {
                "input": rng.normal(size=broadcast_input_shape)
                .astype(np.float32)
                .reshape(-1),
                "output": np.empty(int(np.prod(broadcast_output_shape)), np.float32),
            },
            {
                "InputN": int(np.prod(broadcast_input_shape)),
                "OutputN": int(np.prod(broadcast_output_shape)),
            },
        )
    )
    for kind in ("rmsnorm", "layernorm"):
        rows_count, columns = 64, 256
        package = package_norm_backward(kind=kind)
        arrays = {
            "x": rng.normal(size=(rows_count, columns)).astype(np.float32),
            "gamma": rng.normal(size=columns).astype(np.float32),
            "dy": rng.normal(size=(rows_count, columns)).astype(np.float32),
            "dx": np.empty((rows_count, columns), np.float32),
            "dgamma": np.empty(columns, np.float32),
            "dbeta": np.empty(columns, np.float32),
        }
        cases.append(
            (f"{kind}_backward", package, arrays,
             {"Rows": rows_count, "Columns": columns})
        )

    logits = rng.normal(size=(512, 128)).astype(np.float32)
    class_package = package_class_backward(label_smoothing=0.1)
    cases.append(
        (
            "cross_entropy_backward",
            class_package,
            {
                "logits": logits,
                "labels": rng.integers(0, 128, size=512, dtype=np.int64),
                "dy": rng.normal(size=512).astype(np.float32),
                "dlogits": np.empty_like(logits),
            },
            {"Rows": 512, "Classes": 128},
        )
    )

    for name, package, arrays, scalars in cases:
        warm_package = _warm_package(package)
        measured = _measure_package(
            package, arrays, scalars, block_size=128, samples=samples,
            device_reps=device_reps, e2e_reps=e2e_reps,
        )
        rows.append(
            {
                "operation": name,
                "shape": list(next(iter(arrays.values())).shape),
                "selected_route": "compiler_owned_cuda_ptx",
                "selector_changed": False,
                "compile_state": {
                    "cold": package.image.compile_state,
                    "warm": warm_package.image.compile_state,
                    "same_image": (
                        warm_package.image.image_digest == package.image.image_digest
                    ),
                },
                **measured,
            }
        )

    dynamic = package_path_max_probe(
        then_bytes=12_289, else_bytes=32_001, branch=1
    )
    dynamic_arrays = {"output": np.zeros(2, np.float32)}
    dynamic_scalars = {
        "ThenBytes": 12_289, "ElseBytes": 32_001, "Branch": 1
    }
    rows.append(
        {
            "operation": "dynamic_shared_path_max",
            "shape": [12_289, 32_001],
            "selected_route": "compiler_owned_cuda_ptx",
            "selector_changed": False,
            "launch_expression": "aligned_sum_of_slot_maxima",
            "compile_state": {
                "cold": dynamic.image.compile_state,
                "warm": package_path_max_probe(
                    then_bytes=12_289, else_bytes=32_001, branch=1
                ).image.compile_state,
                "same_image": True,
            },
            **_measure_package(
                dynamic, dynamic_arrays, dynamic_scalars, block_size=1,
                samples=samples, device_reps=device_reps,
                e2e_reps=e2e_reps,
            ),
        }
    )
    local_expression = package_local_expression_probe(
        base=4096, factor=4, bias=17, fallback=12_289, branch=1
    )
    rows.append(
        {
            "operation": "dynamic_shared_local_expression",
            "shape": [4096, 4, 17, 12_289],
            "selected_route": "compiler_owned_cuda_ptx",
            "selector_changed": False,
            "launch_expression": "cfg_forwarded_path_max_of_local_mul_add",
            "compile_state": {
                "cold": local_expression.image.compile_state,
                "warm": package_local_expression_probe(
                    base=4096,
                    factor=4,
                    bias=17,
                    fallback=12_289,
                    branch=1,
                ).image.compile_state,
                "same_image": True,
            },
            **_measure_package(
                local_expression,
                {"output": np.zeros(2, np.float32)},
                {
                    "Base": 4096,
                    "Factor": 4,
                    "Bias": 17,
                    "Fallback": 12_289,
                    "Branch": 1,
                },
                block_size=1,
                samples=samples,
                device_reps=device_reps,
                e2e_reps=e2e_reps,
            ),
        }
    )
    by_name = {row["operation"]: row for row in rows}
    for fused_name, optimizer_name in (
        ("fused_mse_sgd", "sgd"),
        ("fused_mse_adamw", "adamw"),
    ):
        fused = by_name[fused_name]
        loss = by_name["mse_backward"]
        optimizer = by_name[optimizer_name]
        composed_device = [
            loss["device_event_ns"][index]
            + optimizer["device_event_ns"][index]
            for index in range(2)
        ]
        composed_e2e = [
            loss["end_to_end_ns"][index] + optimizer["end_to_end_ns"][index]
            for index in range(2)
        ]
        device_winner = (
            "fused"
            if statistics.median(fused["device_event_ns"])
            < statistics.median(composed_device)
            else "composed"
        )
        e2e_winner = (
            "fused"
            if statistics.median(fused["end_to_end_ns"])
            < statistics.median(composed_e2e)
            else "composed"
        )
        fused["comparison"] = {
            "candidate": "fused",
            "reference": "composed_loss_backward_plus_optimizer",
            "composed_device_event_ns": composed_device,
            "composed_end_to_end_ns": composed_e2e,
            "device_winner": device_winner,
            "end_to_end_winner": e2e_winner,
            "timing_domains_agree": device_winner == e2e_winner,
            "promotion_eligible": (
                bool(fused["stable_4pct"])
                and bool(loss["stable_4pct"])
                and bool(optimizer["stable_4pct"])
                and device_winner == e2e_winner
            ),
            "disposition": "evidence_only_no_production_selector",
        }
    memory = rt._nvidia_device_memory_envelope()
    rematerialization = resolve_nvidia_rematerialization_policy(
        model_parameter_bytes=2 << 30,
        capacity_bytes=memory["capacity_bytes"],
        free_bytes=memory["free_bytes"],
        reserve_basis_points=1000,
        gradient_copies=1,
        optimizer_state_copies=2,
        persistent_bytes=512 << 20,
    )
    return {
        "schema": "tessera.nvidia.training-memory-foundation.v2",
        "work_items": [
            "NVIDIA-TRAINING-AUTODIFF",
            "E2E-SPINE3-SM120-DYNAMIC-MEMORY",
        ],
        "device": rt._nvidia_device_name(),
        "host_environment": (
            "wsl2" if "microsoft" in platform.release().lower() else "native_linux"
        ),
        "device_memory_policy": {
            "capacity_bytes": rematerialization.device_capacity_bytes,
            "free_bytes": rematerialization.device_free_bytes,
            "effective_capacity_bytes": rematerialization.effective_capacity_bytes,
            "model_parameter_bytes": rematerialization.model_parameter_bytes,
            "model_state_bytes": rematerialization.model_state_bytes,
            "activation_budget_bytes": rematerialization.activation_budget_bytes,
            "reserve_basis_points": rematerialization.reserve_basis_points,
        },
        "policy": {
            "discard_first_compile_launch": True,
            "samples_per_run": samples,
            "runs": 2,
            "device_repetitions": device_reps,
            "end_to_end_repetitions": e2e_reps,
            "stability_pct": 4.0,
        },
        "rows": rows,
        "selector_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--device-reps", type=int, default=100)
    parser.add_argument("--e2e-reps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = record(
        samples=args.samples,
        device_reps=args.device_reps,
        e2e_reps=args.e2e_reps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
