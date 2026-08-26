"""Family-owned execution for native reverse products.

``JitFn`` owns call binding and records the selected execution result.  It must
not own another operation-family dispatch table or construct backend packages.
This module is the migration boundary for reverse products: every registered
family declares the Graph, Schedule, Tile, and Target consumers that own its
physical execution.

Normalization, bounded compound spectral products, canonical rank-4 attention,
Lion, factored/full Adafactor, and causal sequence-mixer backward are migrated.
SGD, Momentum/Nesterov, and the gfx1151 Adam/AdamW consumers also use that
boundary; unsupported target pairs fail closed before package construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from threading import Lock
from typing import Any, Callable, Mapping, Sequence

from .._jit_boundary import TesseraJitError


@dataclass(frozen=True)
class NativeVJPPluginDeclaration:
    """Typed ownership declaration across the complete compiler spine."""

    family: str
    graph_consumers: tuple[str, ...]
    schedule_consumer: str
    tile_consumer: str
    target_consumers: Mapping[str, str]
    migration_state: str = "canonical_composite"
    differential_policy: str = "pure_only"

    def validate(self) -> None:
        if not self.family or not self.graph_consumers:
            raise ValueError("native VJP declaration requires family and Graph consumers")
        if not self.schedule_consumer.startswith("schedule."):
            raise ValueError("native VJP declaration requires a Schedule consumer")
        if not self.tile_consumer.startswith("tile."):
            raise ValueError("native VJP declaration requires a Tile consumer")
        if self.migration_state not in {"canonical", "canonical_composite"}:
            raise ValueError("native VJP declaration has an invalid migration state")
        if self.differential_policy not in {
            "pure_only",
            "zero_dropout_attention",
            "non_reexecuting_state_lineage",
        }:
            raise ValueError("native VJP declaration has an invalid differential policy")
        if not self.target_consumers or any(not value for value in self.target_consumers.values()):
            raise ValueError("native VJP declaration requires concrete Target consumers")


@dataclass(frozen=True)
class NativeVJPResult:
    gradients: tuple[Any, ...]
    execution: Mapping[str, Any]


_EXECUTION_SCHEMA = "tessera.native_vjp_execution.v1"
_EXECUTION_CERTIFICATES: dict[str, dict[str, Mapping[str, Any]]] = {}
_EXECUTION_CERTIFICATE_LOCK = Lock()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _value_signature(value: Any) -> Mapping[str, Any]:
    import numpy as np

    array = np.asarray(value)
    return {
        "dtype": str(array.dtype),
        "shape": [int(dim) for dim in array.shape],
    }


def _compose_physical_attestations(
    attestations: Sequence[Any],
) -> Mapping[str, Any] | None:
    if not attestations or any(not isinstance(row, Mapping) for row in attestations):
        return None
    rows = [dict(row) for row in attestations]
    identity = {
        (row.get("schema"), row.get("target"), row.get("device_arch"),
         row.get("execution_kind"), row.get("execution_mode"))
        for row in rows
    }
    if len(identity) != 1 or any(len(str(row.get("digest", ""))) != 64 for row in rows):
        return None
    schema, target, device_arch, execution_kind, execution_mode = identity.pop()
    artifact_hash = hashlib.sha256(
        "|".join(str(row["digest"]) for row in rows).encode()
    ).hexdigest()
    body = {
        "schema": schema,
        "target": target,
        "device_arch": device_arch,
        "execution_kind": execution_kind,
        "execution_mode": execution_mode,
        "artifact_hash": artifact_hash,
    }
    return {**body, "digest": _canonical_digest(body)}


def validate_native_vjp_execution_certificate(
    certificate: Mapping[str, Any],
) -> None:
    body = dict(certificate)
    actual = str(body.pop("digest", ""))
    if body.get("schema") != _EXECUTION_SCHEMA or actual != _canonical_digest(body):
        raise ValueError("native VJP execution certificate has stale identity")
    if body.get("status") != "executed":
        raise ValueError("native VJP execution certificate is not successful")
    if not str(body.get("graph_consumer", "")).startswith("tessera."):
        raise ValueError("native VJP execution certificate has no Graph consumer")
    if not str(body.get("schedule_consumer", "")).startswith("schedule."):
        raise ValueError("native VJP execution certificate has no Schedule consumer")
    if not str(body.get("tile_consumer", "")).startswith("tile."):
        raise ValueError("native VJP execution certificate has no Tile consumer")
    if not body.get("target_consumer") or not body.get("execution_mode"):
        raise ValueError("native VJP execution certificate has no physical consumer")
    if not body.get("input_signature") or not body.get("gradient_signature"):
        raise ValueError("native VJP execution certificate has no tensor signature")
    attestation = body.get("physical_attestation")
    if attestation is not None:
        if not isinstance(attestation, Mapping):
            raise ValueError("native VJP execution certificate has invalid physical proof")
        attestation_body = dict(attestation)
        attestation_digest = str(attestation_body.pop("digest", ""))
        if (
            attestation_body.get("schema")
            != "tessera.runtime_physical_execution.v1"
            or attestation_digest != _canonical_digest(attestation_body)
            or attestation_body.get("target") != body.get("target")
            or attestation_body.get("execution_kind") != body.get("execution_kind")
            or attestation_body.get("execution_mode") != body.get("execution_mode")
        ):
            raise ValueError(
                "native VJP execution certificate has stale physical identity"
            )
    expected_arch = {
        "x86": "x86_avx512",
        "rocm": "gfx1151",
        "nvidia_sm120": "sm_120",
        "apple_gpu": "apple7",
    }.get(str(body.get("target", "")))
    expected_scope = (
        "exact_device"
        if isinstance(attestation, Mapping)
        and attestation.get("device_arch") == expected_arch
        else "runtime_unattested"
    )
    if body.get("evidence_scope") != expected_scope:
        raise ValueError("native VJP execution certificate has invalid evidence scope")
    if (
        body.get("source_reexecution") == "prohibited"
        and len(str(body.get("frontend_certificate_digest", ""))) != 64
    ):
        raise ValueError(
            "non-reexecuting native VJP certificate lacks frontend identity"
        )


def _record_execution_certificate(
    *,
    declaration: NativeVJPPluginDeclaration,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    out_cotangents: Any,
    result: NativeVJPResult,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    cotangents = (
        tuple(out_cotangents)
        if isinstance(out_cotangents, (tuple, list))
        else (out_cotangents,)
    )
    execution = dict(result.execution)
    attestation = execution.get("physical_attestation")
    expected_arch = {
        "x86": "x86_avx512",
        "rocm": "gfx1151",
        "nvidia_sm120": "sm_120",
        "apple_gpu": "apple7",
    }.get(target)
    identities = {
        key: str(execution[key])
        for key in (
            "source_graph_ir_digest",
            "frontend_certificate_digest",
            "schedule_artifact_hash",
            "state_lineage_digest",
            "tile_program_digest",
            "artifact_hash",
        )
        if execution.get(key) is not None
    }
    certificate_body = {
        "schema": _EXECUTION_SCHEMA,
        "family": declaration.family,
        "graph_consumer": str(source.op_name),
        "schedule_consumer": declaration.schedule_consumer,
        "tile_consumer": declaration.tile_consumer,
        "target_consumer": declaration.target_consumers[target],
        "target": target,
        "execution_kind": str(execution.get("execution_kind", "")),
        "execution_mode": str(execution.get("execution_mode", "")),
        "evidence_target": str(execution.get("evidence_target", "")),
        "evidence_scope": (
            "exact_device"
            if isinstance(attestation, Mapping)
            and attestation.get("device_arch") == expected_arch
            else "runtime_unattested"
        ),
        "physical_attestation": dict(attestation)
        if isinstance(attestation, Mapping)
        else None,
        "proof_mode": str(execution.get("proof_mode", "")),
        "topology": str(execution.get("topology", "not_applicable")),
        "algorithm": str(execution.get("algorithm", "not_applicable")),
        "input_signature": [
            _value_signature(value) for value in (*ordered_inputs, *cotangents)
        ],
        "gradient_signature": [
            _value_signature(value) for value in result.gradients
        ],
        "frontend_certificate_digest": (
            str(getattr(frontend_certificate, "digest", ""))
        ),
        "artifact_identities": identities,
        "source_reexecution": "prohibited"
        if declaration.differential_policy == "non_reexecuting_state_lineage"
        else "policy_permitted",
        "status": "executed",
    }
    digest = _canonical_digest(certificate_body)
    certificate = {**certificate_body, "digest": digest}
    validate_native_vjp_execution_certificate(certificate)
    with _EXECUTION_CERTIFICATE_LOCK:
        _EXECUTION_CERTIFICATES.setdefault(declaration.family, {})[digest] = certificate
    execution["execution_certificate_schema"] = _EXECUTION_SCHEMA
    execution["execution_certificate_digest"] = digest
    execution["execution_certificate"] = certificate
    return NativeVJPResult(result.gradients, execution)


Executor = Callable[..., NativeVJPResult]
_PLUGINS: dict[str, tuple[NativeVJPPluginDeclaration, Executor]] = {}


def register_native_vjp_plugin(
    *op_names: str,
    family: str,
    schedule_consumer: str,
    tile_consumer: str,
    target_consumers: Mapping[str, str],
    differential_policy: str = "pure_only",
) -> Callable[[Executor], Executor]:
    """Register one explicit Graph-op owner; duplicate ownership is invalid."""
    declaration = NativeVJPPluginDeclaration(
        family=family,
        graph_consumers=tuple(f"tessera.{name}" for name in op_names),
        schedule_consumer=schedule_consumer,
        tile_consumer=tile_consumer,
        target_consumers=dict(target_consumers),
        differential_policy=differential_policy,
    )
    declaration.validate()

    def decorate(executor: Executor) -> Executor:
        for name in op_names:
            if name in _PLUGINS:
                raise RuntimeError(f"native VJP plugin already owns {name!r}")
            _PLUGINS[name] = (declaration, executor)
        return executor

    return decorate


@register_native_vjp_plugin(
    "rmsnorm",
    "rmsnorm_safe",
    "layer_norm",
    family="normalization",
    schedule_consumer="schedule.native_vjp_program",
    tile_consumer="tile.native_vjp_program",
    target_consumers={
        "x86": "x86.avx512_normalization",
        "rocm": "rocm.gfx1151_normalization",
        "nvidia_sm120": "nvidia.sm120_normalization",
        "apple_gpu": "apple.metal_normalization",
    },
)
def _execute_normalization(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    source_graph_ir: str | None,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    """Package and launch one registered normalization VJP."""
    import numpy as np

    from tessera.runtime import RuntimeArtifact, launch

    inputs = [np.ascontiguousarray(np.asarray(value)) for value in ordered_inputs]
    cotangents = (
        tuple(out_cotangents)
        if isinstance(out_cotangents, (tuple, list))
        else (out_cotangents,)
    )


    if len(cotangents) != 1:
        raise TesseraJitError("normalization backward requires one output cotangent")
    dy = np.ascontiguousarray(np.asarray(cotangents[0]))
    bare = source.op_name.removeprefix("tessera.")
    family = "layer_norm" if bare == "layer_norm" else "rmsnorm"
    operand_names = list(arg_names)
    gradient_names = [f"d_{name}" for name in operand_names]
    target_identity = {
        "x86": ("cpu_avx512", "native_cpu", "x86_avx512"),
        "rocm": ("hip_runtime", "native_gpu", "rocm_gfx1151"),
        "nvidia_sm120": ("cuda_driver", "native_gpu", "nvidia_sm120"),
        "apple_gpu": ("metal_runtime", "native_gpu", "apple7"),
    }
    try:
        execution_mode, execution_kind, evidence_target = target_identity[target]
    except KeyError as exc:
        raise TesseraJitError(
            f"native normalization VJP has no Target consumer for {target!r}"
        ) from exc
    path = (
        "nvidia_norm_bwd_compiled"
        if target == "nvidia_sm120"
        else f"{target}_{family}_bwd_compiled"
    )
    artifact = RuntimeArtifact(metadata={
        "target": target,
        "compiler_path": path,
        "executable": True,
        "execution_kind": execution_kind,
        "execution_mode": execution_mode,
        "autodiff_phase": "backward",
        "native_vjp_family": declaration.family,
        "native_vjp_schedule_consumer": declaration.schedule_consumer,
        "native_vjp_tile_consumer": declaration.tile_consumer,
        "native_vjp_target_consumer": declaration.target_consumers[target],
        "out_cotangent": "dy",
        "arg_names": operand_names + ["dy"],
        "output_names": gradient_names,
        "ops": [{
            "op_name": source.op_name,
            "result": source.result,
            "operands": operand_names,
            "kwargs": dict(source.kwargs),
        }],
    })
    result = launch(artifact, tuple([*inputs, dy]))
    if not result.get("ok") or result.get("execution_mode") != execution_mode:
        raise TesseraJitError(
            f"verified {target} normalization backward launch failed: "
            + str(result.get("reason"))
        )
    output = result["output"]
    all_gradients = (
        tuple(output) if isinstance(output, (tuple, list)) else (output,)
    )
    by_name = dict(zip(operand_names, all_gradients))
    missing = [name for name in wrt_names if name not in by_name]
    if missing:
        raise TesseraJitError(
            "normalization VJP requested unknown operands: " + ", ".join(missing)
        )
    return NativeVJPResult(
        gradients=tuple(by_name[name] for name in wrt_names),
        execution={
            "compiler_path": path,
            "execution_kind": execution_kind,
            "execution_mode": execution_mode,
            "evidence_target": evidence_target,
            "implementation": "family_plugin",
            "residual_policy": "recompute_all",
            "family": declaration.family,
            "graph_consumer": source.op_name,
            "schedule_consumer": declaration.schedule_consumer,
            "tile_consumer": declaration.tile_consumer,
            "target_consumer": declaration.target_consumers[target],
            "physical_attestation": result.get("physical_attestation"),
        },
    )


@register_native_vjp_plugin(
    "spectral_filter",
    "spectral_conv",
    "stft",
    "istft",
    family="spectral_backward",
    schedule_consumer="schedule.spectral_backward",
    tile_consumer="tile.spectral_backward_kernel",
    target_consumers={
        "x86": "x86.avx512_spectral_backward",
        "rocm": "rocm.gfx1151_spectral_backward",
        "nvidia_sm120": "nvidia.sm120_spectral_backward",
    },
)
def _execute_compound_spectral(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    source_graph_ir: str | None,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    """Build and execute one traced compound-spectral reverse package."""
    from tessera.runtime import RuntimeArtifact, launch

    from .native_spectral_vjp import (
        build_native_spectral_vjp_package,
        compile_rocm_native_spectral_vjp,
    )

    if not source_graph_ir:
        raise TesseraJitError(
            "compound spectral VJP requires tracer-owned source Graph IR"
        )
    cotangents = (
        tuple(out_cotangents)
        if isinstance(out_cotangents, (tuple, list))
        else (out_cotangents,)
    )
    if len(cotangents) != 1:
        raise TesseraJitError("compound spectral VJP requires one output cotangent")
    try:
        package = build_native_spectral_vjp_package(
            source_graph_ir=source_graph_ir,
            source=source,
            target=target,
            ordered_inputs=ordered_inputs,
            arg_names=arg_names,
            out_cotangent=cotangents[0],
        )
        if target == "rocm":
            package = compile_rocm_native_spectral_vjp(package)
    except (RuntimeError, ValueError) as exc:
        raise TesseraJitError(str(exc)) from exc
    result = launch(
        RuntimeArtifact(metadata=package.runtime_metadata()),
        tuple([cotangents[0], *ordered_inputs]),
    )
    expected_mode = ("cpu_avx512" if target == "x86" else
                     "cuda_runtime" if target == "nvidia_sm120" else
                     "hip_runtime")
    if not result.get("ok") or result.get("execution_mode") != expected_mode:
        raise TesseraJitError(
            f"verified {target} compound spectral backward launch failed: "
            + str(result.get("reason"))
        )
    outputs = result.get("output")
    gradients = tuple(outputs) if isinstance(outputs, (tuple, list)) else (outputs,)
    by_name = dict(zip(arg_names, gradients))
    missing = [name for name in wrt_names if name not in by_name]
    if missing:
        raise TesseraJitError(
            "compound spectral VJP requested unknown operands: " + ", ".join(missing)
        )
    return NativeVJPResult(
        gradients=tuple(by_name[name] for name in wrt_names),
        execution={
            "compiler_path": f"{target}_spectral_backward_compiled",
            "execution_kind": "native_cpu" if target == "x86" else "native_gpu",
            "execution_mode": expected_mode,
            "evidence_target": ("x86_avx512" if target == "x86" else
                                "nvidia_sm120" if target == "nvidia_sm120" else
                                "rocm_gfx1151"),
            "implementation": "family_plugin",
            "residual_policy": "save_inputs",
            "family": declaration.family,
            "graph_consumer": source.op_name,
            "schedule_consumer": declaration.schedule_consumer,
            "tile_consumer": declaration.tile_consumer,
            "target_consumer": declaration.target_consumers[target],
            "source_graph_ir_digest": package.source_graph_ir_digest,
            "schedule_artifact_hash": package.schedule_artifact_hash,
            "tile_program_digest": package.tile_program_digest,
            "algorithm": package.algorithm,
            "frontend_authority": "tracer",
            "physical_attestation": result.get("physical_attestation"),
        },
    )


@register_native_vjp_plugin(
    "lion",
    family="lion_vjp",
    schedule_consumer="schedule.lion_vjp",
    tile_consumer="tile.training_kernel",
    target_consumers={
        "x86": "x86.avx512_lion_backward",
        "rocm": "rocm.gfx1151_lion_backward",
        "nvidia_sm120": "nvidia.sm120_lion_backward",
    },
    differential_policy="non_reexecuting_state_lineage",
)
def _execute_lion(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    source_graph_ir: str | None,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    """Build and launch one non-reexecuting, state-lineage Lion package."""
    import numpy as np

    from tessera.runtime import RuntimeArtifact, launch

    cotangents = (
        tuple(out_cotangents)
        if isinstance(out_cotangents, (tuple, list))
        else (out_cotangents,)
    )
    if target == "nvidia_sm120":
        if len(ordered_inputs) != 3 or len(arg_names) != 3:
            raise TesseraJitError(
                "NVIDIA Lion backward requires param, grad, and moment"
            )
        if len(cotangents) != 2:
            raise TesseraJitError(
                "NVIDIA Lion backward requires parameter and moment cotangents"
            )
        cotangent_names = ["dparam_out", "dmoment_out"]
        from .native_lion_vjp import build_native_lion_vjp_package

        if not source_graph_ir:
            raise TesseraJitError("NVIDIA Lion VJP requires tracer-owned source Graph IR")
        try:
            package = build_native_lion_vjp_package(
                source_graph_ir=source_graph_ir, source=source, target=target,
                ordered_inputs=ordered_inputs, arg_names=arg_names,
                out_cotangents=cotangents,
                frontend_certificate=frontend_certificate,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise TesseraJitError(str(exc)) from exc
        path = "nvidia_lion_bwd_compiled"
        metadata = package.runtime_metadata()
        metadata.update({
            "native_vjp_family": declaration.family,
            "native_vjp_schedule_consumer": declaration.schedule_consumer,
            "native_vjp_tile_consumer": declaration.tile_consumer,
            "native_vjp_target_consumer": declaration.target_consumers[target],
            # This is a Target ABI descriptor, not source Graph IR.  The
            # executor uses the bounded native kernel descriptor below.
            "nvidia_training_descriptor": {
                "schema": "tessera.cuda.training_descriptor.v1",
                "family": declaration.family,
                "state_lineage_digest": package.contract["state_lineage_digest"],
                "tile_program_digest": package.tile_program_digest,
            },
            "ops": [{
                "op_name": source.op_name, "result": source.result,
                "operands": list(arg_names), "kwargs": dict(source.kwargs),
            }],
        })
        result = launch(
            RuntimeArtifact(metadata=metadata),
            tuple(
                np.ascontiguousarray(np.asarray(value), dtype=np.float32)
                for value in (*ordered_inputs, *cotangents)
            ),
        )
        if not result.get("ok") or result.get("execution_mode") != "cuda_driver":
            raise TesseraJitError(
                "verified SM120 Lion backward launch failed: "
                + str(result.get("reason"))
            )
        by_name = dict(zip(arg_names, tuple(result["output"]), strict=True))
        missing = [name for name in wrt_names if name not in by_name]
        if missing:
            raise TesseraJitError(
                "Lion VJP requested unknown operands: " + ", ".join(missing)
            )
        return NativeVJPResult(
            gradients=tuple(by_name[name] for name in wrt_names),
            execution={
                "compiler_path": path,
                "execution_kind": "native_gpu",
                "execution_mode": "cuda_driver",
                "evidence_target": "nvidia_sm120",
                "implementation": "family_plugin",
                "residual_policy": "none",
                "family": declaration.family,
                "graph_consumer": source.op_name,
                "schedule_consumer": declaration.schedule_consumer,
                "tile_consumer": declaration.tile_consumer,
                "target_consumer": declaration.target_consumers[target],
                "source_graph_ir_digest": package.source_graph_ir_digest,
                "frontend_certificate_digest": package.frontend_certificate.digest,
                "schedule_artifact_hash": package.scheduled.schedule_digest,
                "state_lineage_digest": package.contract["state_lineage_digest"],
                "tile_program_digest": package.tile_program_digest,
                "artifact_hash": package.artifact_hash,
                "frontend_authority": "tracer",
                "proof_mode": "structural_non_reexecuting",
                "physical_attestation": result.get("physical_attestation"),
            },
        )

    from .native_lion_vjp import build_native_lion_vjp_package

    if not source_graph_ir:
        raise TesseraJitError("Lion VJP requires tracer-owned source Graph IR")
    try:
        package = build_native_lion_vjp_package(
            source_graph_ir=source_graph_ir,
            source=source,
            target=target,
            ordered_inputs=ordered_inputs,
            arg_names=arg_names,
            out_cotangents=cotangents,
            frontend_certificate=frontend_certificate,
        )
        result = launch(
            RuntimeArtifact(metadata=package.runtime_metadata()),
            tuple(
                np.ascontiguousarray(np.asarray(value), dtype=np.float32)
                for value in (*ordered_inputs, *cotangents)
            ),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TesseraJitError(str(exc)) from exc
    execution_mode = "cpu_avx512" if target == "x86" else "hip_runtime"
    if not result.get("ok") or result.get("execution_mode") != execution_mode:
        raise TesseraJitError(
            f"verified {target} Lion backward launch failed: "
            + str(result.get("reason"))
        )
    gradients = tuple(result["output"])
    by_name = dict(zip(package.argument_names, gradients, strict=True))
    missing = [name for name in wrt_names if name not in by_name]
    if missing:
        raise TesseraJitError(
            "Lion VJP requested unknown operands: " + ", ".join(missing)
        )
    return NativeVJPResult(
        gradients=tuple(by_name[name] for name in wrt_names),
        execution={
            "compiler_path": f"{target}_lion_bwd_compiled",
            "execution_kind": "native_cpu" if target == "x86" else "native_gpu",
            "execution_mode": execution_mode,
            "evidence_target": "x86_avx512" if target == "x86" else "rocm_gfx1151",
            "implementation": "family_plugin",
            "residual_policy": "none",
            "family": declaration.family,
            "graph_consumer": source.op_name,
            "schedule_consumer": declaration.schedule_consumer,
            "tile_consumer": declaration.tile_consumer,
            "target_consumer": declaration.target_consumers[target],
            "source_graph_ir_digest": package.source_graph_ir_digest,
            "frontend_certificate_digest": package.frontend_certificate.digest,
            "schedule_artifact_hash": package.scheduled.schedule_digest,
            "state_lineage_digest": package.scheduled.state_contract["artifact_hash"],
            "tile_program_digest": package.tile_program_digest,
            "artifact_hash": package.artifact_hash,
            "frontend_authority": "tracer",
            "proof_mode": "structural_non_reexecuting",
            "physical_attestation": result.get("physical_attestation"),
        },
    )


def _execute_stateful_package(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    source_graph_ir: str | None,
    frontend_certificate: Any | None,
    builder: Callable[..., Any],
    residual_policy: str,
) -> NativeVJPResult:
    """Build and launch one state-lineage package without replaying Graph IR."""
    import numpy as np

    from tessera.runtime import RuntimeArtifact, launch

    if not source_graph_ir:
        raise TesseraJitError(
            f"{declaration.family} requires tracer-owned source Graph IR"
        )
    cotangents = (
        tuple(out_cotangents)
        if isinstance(out_cotangents, (tuple, list))
        else (out_cotangents,)
    )
    try:
        package = builder(
            source_graph_ir=source_graph_ir,
            source=source,
            target=target,
            ordered_inputs=ordered_inputs,
            arg_names=arg_names,
            out_cotangents=cotangents,
            frontend_certificate=frontend_certificate,
        )
        result = launch(
            RuntimeArtifact(metadata=package.runtime_metadata()),
            tuple(
                np.ascontiguousarray(np.asarray(value), dtype=np.float32)
                for value in (*ordered_inputs, *cotangents)
            ),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TesseraJitError(str(exc)) from exc
    execution_mode = "cpu_avx512" if target == "x86" else "hip_runtime"
    if not result.get("ok") or result.get("execution_mode") != execution_mode:
        raise TesseraJitError(
            f"verified {target} {declaration.family} launch failed: "
            + str(result.get("reason"))
        )
    gradients = tuple(result["output"])
    by_name = dict(zip(package.argument_names, gradients, strict=True))
    missing = [name for name in wrt_names if name not in by_name]
    if missing:
        raise TesseraJitError(
            f"{declaration.family} requested unknown operands: " + ", ".join(missing)
        )
    return NativeVJPResult(
        gradients=tuple(by_name[name] for name in wrt_names),
        execution={
            "compiler_path": package.compiler_path,
            "execution_kind": "native_cpu" if target == "x86" else "native_gpu",
            "execution_mode": execution_mode,
            "evidence_target": "x86_avx512" if target == "x86" else "rocm_gfx1151",
            "implementation": "family_plugin",
            "residual_policy": residual_policy,
            "family": declaration.family,
            "graph_consumer": source.op_name,
            "schedule_consumer": declaration.schedule_consumer,
            "tile_consumer": declaration.tile_consumer,
            "target_consumer": declaration.target_consumers[target],
            "source_graph_ir_digest": package.source_graph_ir_digest,
            "frontend_certificate_digest": package.frontend_certificate.digest,
            "schedule_artifact_hash": package.scheduled.schedule_digest,
            "state_lineage_digest": package.state_lineage_digest,
            "tile_program_digest": package.tile_program_digest,
            "artifact_hash": package.artifact_hash,
            "frontend_authority": "tracer",
            "proof_mode": "structural_non_reexecuting",
            "physical_attestation": result.get("physical_attestation"),
            "topology": str(getattr(package.scheduled, "topology", "not_applicable")),
        },
    )


@register_native_vjp_plugin(
    "sgd",
    family="optimizer_vjp",
    schedule_consumer="schedule.optimizer_vjp",
    tile_consumer="tile.training_kernel",
    target_consumers={
        "x86": "x86.avx512_sgd_backward",
        "rocm": "rocm.gfx1151_sgd_backward",
    },
    differential_policy="non_reexecuting_state_lineage",
)
@register_native_vjp_plugin(
    "momentum", "nesterov",
    family="optimizer_vjp",
    schedule_consumer="schedule.optimizer_vjp",
    tile_consumer="tile.training_kernel",
    target_consumers={
        "x86": "x86.avx512_momentum_backward",
        "rocm": "rocm.gfx1151_momentum_backward",
    },
    differential_policy="non_reexecuting_state_lineage",
)
@register_native_vjp_plugin(
    "adam", "adamw",
    family="optimizer_vjp",
    schedule_consumer="schedule.optimizer_vjp",
    tile_consumer="tile.training_kernel",
    target_consumers={"rocm": "rocm.gfx1151_adam_backward"},
    differential_policy="non_reexecuting_state_lineage",
)
def _execute_optimizer_vjp(
    *, source: Any, target: str, ordered_inputs: Sequence[Any],
    arg_names: Sequence[str], source_arg_names: Sequence[str],
    out_cotangents: Any, wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration, source_graph_ir: str | None,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    from .native_stateful_vjp import build_native_optimizer_vjp_package

    return _execute_stateful_package(
        source=source, target=target, ordered_inputs=ordered_inputs,
        arg_names=arg_names, out_cotangents=out_cotangents,
        wrt_names=wrt_names, declaration=declaration,
        source_graph_ir=source_graph_ir,
        frontend_certificate=frontend_certificate,
        builder=build_native_optimizer_vjp_package,
        residual_policy="save_explicit_optimizer_state",
    )


@register_native_vjp_plugin(
    "adafactor",
    family="adafactor_vjp",
    schedule_consumer="schedule.adafactor_vjp",
    tile_consumer="tile.training_kernel",
    target_consumers={
        "x86": "x86.avx512_adafactor_backward",
        "rocm": "rocm.gfx1151_adafactor_backward",
    },
    differential_policy="non_reexecuting_state_lineage",
)
def _execute_adafactor(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    source_graph_ir: str | None,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    from .native_stateful_vjp import build_native_adafactor_vjp_package

    return _execute_stateful_package(
        source=source,
        target=target,
        ordered_inputs=ordered_inputs,
        arg_names=arg_names,
        out_cotangents=out_cotangents,
        wrt_names=wrt_names,
        declaration=declaration,
        source_graph_ir=source_graph_ir,
        frontend_certificate=frontend_certificate,
        builder=build_native_adafactor_vjp_package,
        residual_policy="recompute_optimizer_state",
    )


@register_native_vjp_plugin(
    "gated_deltanet",
    "kimi_delta_attention",
    "modified_delta_attention",
    family="sequence_mixer_backward",
    schedule_consumer="schedule.sequence_mixer_backward",
    tile_consumer="tile.training_kernel",
    target_consumers={
        "x86": "x86.avx512_sequence_mixer_backward",
        "rocm": "rocm.gfx1151_sequence_mixer_backward",
        "nvidia_sm120": "nvidia.sm120_sequence_mixer_backward",
    },
    differential_policy="non_reexecuting_state_lineage",
)
def _execute_sequence_mixer(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    source_graph_ir: str | None,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    if target == "nvidia_sm120":
        import numpy as np

        from tessera.runtime import RuntimeArtifact, launch

        flags = dict(source.kwargs)
        source_operands = tuple(getattr(source, "operands", ()))
        optional_labels = {
            str(name).lstrip("%").lower() for name in source_operands[3:]
        }
        for key, label in (
            ("has_gate", "gate"),
            ("has_beta", "beta"),
            ("has_decay", "decay"),
        ):
            if key not in flags:
                flags[key] = label in optional_labels
        if not optional_labels.intersection({"gate", "beta", "decay"}) and len(
            source_operands
        ) > 3:
            for index, key in enumerate(("has_gate", "has_beta", "has_decay")):
                flags[key] = index < len(source_operands) - 3
        expected_inputs = 3 + sum(
            int(bool(flags.get(key, False)))
            for key in ("has_gate", "has_beta", "has_decay")
        )
        cotangents = (
            tuple(out_cotangents)
            if isinstance(out_cotangents, (tuple, list))
            else (out_cotangents,)
        )
        if len(ordered_inputs) != expected_inputs or not bool(
            flags.get("causal", True)
        ):
            raise TesseraJitError(
                "SM120 DeltaNet backward requires causal Q/K/V with optional "
                "gate/beta/decay inputs"
            )
        if len(cotangents) != 1:
            raise TesseraJitError(
                "SM120 DeltaNet backward requires one output cotangent"
            )
        path = "nvidia_deltanet_bwd_compiled"
        result = launch(
            RuntimeArtifact(metadata={
                "target": target,
                "compiler_path": path,
                "executable": True,
                "execution_kind": "native_gpu",
                "execution_mode": "cuda_driver",
                "autodiff_phase": "backward",
                "native_vjp_family": declaration.family,
                "native_vjp_schedule_consumer": declaration.schedule_consumer,
                "native_vjp_tile_consumer": declaration.tile_consumer,
                "native_vjp_target_consumer": declaration.target_consumers[target],
                "out_cotangents": ["dy"],
                "arg_names": [*arg_names, "dy"],
                "output_names": [f"d_{name}" for name in arg_names],
                "ops": [{
                    "op_name": source.op_name,
                    "result": source.result,
                    "operands": list(arg_names),
                    "kwargs": flags,
                }],
            }),
            tuple(
                np.ascontiguousarray(np.asarray(value), dtype=np.float32)
                for value in (*ordered_inputs, cotangents[0])
            ),
        )
        if not result.get("ok") or result.get("execution_mode") != "cuda_driver":
            raise TesseraJitError(
                "verified SM120 DeltaNet backward launch failed: "
                + str(result.get("reason"))
            )
        by_name = dict(zip(arg_names, tuple(result["output"]), strict=True))
        missing = [name for name in wrt_names if name not in by_name]
        if missing:
            raise TesseraJitError(
                "sequence-mixer VJP requested unknown operands: "
                + ", ".join(missing)
            )
        return NativeVJPResult(
            gradients=tuple(by_name[name] for name in wrt_names),
            execution={
                "compiler_path": path,
                "execution_kind": "native_gpu",
                "execution_mode": "cuda_driver",
                "evidence_target": "nvidia_sm120",
                "implementation": "family_plugin",
                "residual_policy": "recompute",
                "family": declaration.family,
                "graph_consumer": source.op_name,
                "schedule_consumer": declaration.schedule_consumer,
                "tile_consumer": declaration.tile_consumer,
                "target_consumer": declaration.target_consumers[target],
                "frontend_authority": "tracer",
                "proof_mode": "structural_non_reexecuting",
                "physical_attestation": result.get("physical_attestation"),
            },
        )

    from .native_stateful_vjp import build_native_sequence_mixer_vjp_package

    return _execute_stateful_package(
        source=source,
        target=target,
        ordered_inputs=ordered_inputs,
        arg_names=arg_names,
        out_cotangents=out_cotangents,
        wrt_names=wrt_names,
        declaration=declaration,
        source_graph_ir=source_graph_ir,
        frontend_certificate=frontend_certificate,
        builder=build_native_sequence_mixer_vjp_package,
        residual_policy="launch_owned_checkpoint_workspace",
    )


@register_native_vjp_plugin(
    "flash_attn",
    "gqa_attention",
    "mqa_attention",
    family="attention_backward",
    schedule_consumer="schedule.attention_backward",
    tile_consumer="tile.attention_backward_kernel",
    target_consumers={
        "x86": "x86.avx512_attention_backward",
        "rocm": "rocm.gfx1151_attention_backward_program",
    },
    differential_policy="zero_dropout_attention",
)
def _execute_attention(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    source_graph_ir: str | None,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    """Build and execute one traced canonical attention reverse package."""
    from .native_attention_vjp import (
        build_native_attention_vjp_package,
        execute_native_attention_vjp_package,
    )

    if not source_graph_ir:
        raise TesseraJitError("attention VJP requires tracer-owned source Graph IR")
    cotangents = (
        tuple(out_cotangents)
        if isinstance(out_cotangents, (tuple, list))
        else (out_cotangents,)
    )
    if len(cotangents) != 1:
        raise TesseraJitError("attention VJP requires one output cotangent")
    try:
        package = build_native_attention_vjp_package(
            source_graph_ir=source_graph_ir,
            source=source,
            target=target,
            ordered_inputs=ordered_inputs,
            arg_names=arg_names,
            source_arg_names=source_arg_names,
            out_cotangent=cotangents[0],
        )
        gradients, physical_attestation = execute_native_attention_vjp_package(
            package,
            ordered_inputs=ordered_inputs,
            arg_names=arg_names,
            out_cotangent=cotangents[0],
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TesseraJitError(str(exc)) from exc
    by_name = dict(zip(package.operand_names, gradients, strict=True))
    missing = [name for name in wrt_names if name not in by_name]
    if missing:
        raise TesseraJitError(
            "attention VJP has no physical gradient for operands: "
            + ", ".join(missing)
        )
    execution_mode = "cpu_avx512" if target == "x86" else "hip_runtime"
    return NativeVJPResult(
        gradients=tuple(by_name[name] for name in wrt_names),
        execution={
            "compiler_path": f"{target}_flash_attn_bwd_compiled",
            "execution_kind": "native_cpu" if target == "x86" else "native_gpu",
            "execution_mode": execution_mode,
            "evidence_target": "x86_avx512" if target == "x86" else "rocm_gfx1151",
            "implementation": "family_plugin",
            "residual_policy": package.scheduled.lse_checkpoint_selection,
            "family": declaration.family,
            "graph_consumer": source.op_name,
            "schedule_consumer": declaration.schedule_consumer,
            "tile_consumer": declaration.tile_consumer,
            "target_consumer": declaration.target_consumers[target],
            "source_graph_ir_digest": package.source_graph_ir_digest,
            "schedule_artifact_hash": package.schedule_artifact_hash,
            "tile_program_digest": package.tile_program_digest,
            "native_image_digest": package.native_image_digest,
            "artifact_hash": package.artifact_hash,
            "frontend_authority": "tracer",
            "physical_attestation": physical_attestation,
        },
    )


def _execute_loss_family(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    loss_kind: str,
    source_arg_names: Sequence[str] | None = None,
    source_graph_ir: str | None = None,
    frontend_certificate: Any | None = None,
) -> NativeVJPResult:
    """Package a traced loss product without returning to ``JitFn`` dispatch."""
    del source_arg_names, source_graph_ir, frontend_certificate
    import numpy as np

    from tessera.runtime import RuntimeArtifact, launch

    if len(ordered_inputs) != 2:
        raise TesseraJitError(f"{loss_kind} backward requires two operands")
    cotangents = (
        tuple(out_cotangents)
        if isinstance(out_cotangents, (tuple, list))
        else (out_cotangents,)
    )
    if len(cotangents) != 1:
        raise TesseraJitError(f"{loss_kind} backward requires one output cotangent")
    if loss_kind == "class_loss" and any(name != arg_names[0] for name in wrt_names):
        raise TesseraJitError(
            "integer class-index targets are explicitly nondifferentiable"
        )

    nvidia = target == "nvidia_sm120"
    gpu = target in {"rocm", "nvidia_sm120"}
    execution_mode = (
        "cuda_driver"
        if nvidia
        else "hip_runtime" if target == "rocm" else "cpu_avx512"
    )
    path_stem = {
        "regression_loss": "regression_loss_bwd_compiled",
        "binary_loss": "binary_loss_bwd_compiled",
        "class_loss": "class_loss_bwd_compiled",
        "distribution_loss": "distribution_loss_bwd_compiled",
    }[loss_kind]
    path = f"nvidia_{path_stem}" if nvidia else f"{target}_{path_stem}"
    source_kwargs = dict(source.kwargs)
    if loss_kind == "class_loss" and "smoothing" in source_kwargs:
        source_kwargs["label_smoothing"] = source_kwargs.pop("smoothing")
    output_names = (
        [f"d_{arg_names[0]}"]
        if loss_kind == "class_loss"
        else [f"d_{name}" for name in arg_names]
    )
    inputs = [np.ascontiguousarray(np.asarray(value)) for value in ordered_inputs]
    dy = np.ascontiguousarray(np.asarray(cotangents[0]))
    if loss_kind == "distribution_loss":
        inputs = [np.ascontiguousarray(value, dtype=np.float32) for value in inputs]
        dy = np.ascontiguousarray(dy, dtype=np.float32)
    artifact = RuntimeArtifact(
        metadata={
            "target": target,
            "compiler_path": path,
            "executable": True,
            "execution_kind": "native_gpu" if gpu else "native_cpu",
            "execution_mode": execution_mode,
            "autodiff_phase": "backward",
            "native_vjp_family": declaration.family,
            "native_vjp_schedule_consumer": declaration.schedule_consumer,
            "native_vjp_tile_consumer": declaration.tile_consumer,
            "native_vjp_target_consumer": declaration.target_consumers[target],
            "out_cotangent": "dy",
            "arg_names": [*arg_names, "dy"],
            "output_names": output_names,
            "ops": [
                {
                    "op_name": source.op_name,
                    "result": source.result,
                    "operands": list(arg_names),
                    "kwargs": source_kwargs,
                }
            ],
        }
    )
    launched = launch(artifact, tuple([*inputs, dy]))
    if not launched.get("ok") or launched.get("execution_mode") != execution_mode:
        raise TesseraJitError(
            f"verified {target} {loss_kind} backward launch failed: "
            + str(launched.get("reason"))
        )
    raw_output = launched["output"]
    gradients = (
        tuple(raw_output)
        if isinstance(raw_output, (tuple, list))
        else (raw_output,)
    )
    gradient_names = arg_names[:1] if loss_kind == "class_loss" else arg_names
    by_name = dict(zip(gradient_names, gradients, strict=True))
    missing = [name for name in wrt_names if name not in by_name]
    if missing:
        raise TesseraJitError(
            f"{loss_kind} VJP has no physical gradient for: " + ", ".join(missing)
        )
    return NativeVJPResult(
        gradients=tuple(by_name[name] for name in wrt_names),
        execution={
            "compiler_path": path,
            "execution_kind": "native_gpu" if gpu else "native_cpu",
            "execution_mode": execution_mode,
            "evidence_target": (
                "nvidia_sm120"
                if nvidia
                else "rocm_gfx1151" if target == "rocm" else "x86_avx512"
            ),
            "implementation": "family_plugin",
            "residual_policy": "save_inputs",
            "family": declaration.family,
            "graph_consumer": source.op_name,
            "schedule_consumer": declaration.schedule_consumer,
            "tile_consumer": declaration.tile_consumer,
            "target_consumer": declaration.target_consumers[target],
            "frontend_authority": "tracer",
            "physical_attestation": launched.get("physical_attestation"),
        },
    )


@register_native_vjp_plugin(
    "loss.mse",
    "mse_loss",
    "loss.mae",
    "mae_loss",
    "loss.huber",
    "huber_loss",
    "loss.smooth_l1",
    "smooth_l1_loss",
    family="regression_loss_backward",
    schedule_consumer="schedule.loss_backward",
    tile_consumer="tile.loss_backward_kernel",
    target_consumers={
        "x86": "x86.avx512_regression_loss_backward",
        "rocm": "rocm.gfx1151_regression_loss_backward",
        "nvidia_sm120": "nvidia.sm120_regression_loss_backward",
    },
)
def _execute_regression_loss(**kwargs: Any) -> NativeVJPResult:
    return _execute_loss_family(**kwargs, loss_kind="regression_loss")


@register_native_vjp_plugin(
    "loss.binary_cross_entropy",
    "binary_cross_entropy_loss",
    family="binary_loss_backward",
    schedule_consumer="schedule.loss_backward",
    tile_consumer="tile.loss_backward_kernel",
    target_consumers={
        "x86": "x86.avx512_binary_loss_backward",
        "rocm": "rocm.gfx1151_binary_loss_backward",
        "nvidia_sm120": "nvidia.sm120_binary_loss_backward",
    },
)
def _execute_binary_loss(**kwargs: Any) -> NativeVJPResult:
    return _execute_loss_family(**kwargs, loss_kind="binary_loss")


@register_native_vjp_plugin(
    "loss.cross_entropy",
    "cross_entropy_loss",
    "label_smoothed_cross_entropy",
    family="class_loss_backward",
    schedule_consumer="schedule.loss_backward",
    tile_consumer="tile.loss_backward_kernel",
    target_consumers={
        "x86": "x86.avx512_class_loss_backward",
        "rocm": "rocm.gfx1151_class_loss_backward",
        "nvidia_sm120": "nvidia.sm120_class_loss_backward",
    },
)
def _execute_class_loss(**kwargs: Any) -> NativeVJPResult:
    return _execute_loss_family(**kwargs, loss_kind="class_loss")


@register_native_vjp_plugin(
    "loss.kl_divergence",
    "kl_divergence",
    "loss.js_divergence",
    "js_divergence",
    family="distribution_loss_backward",
    schedule_consumer="schedule.loss_backward",
    tile_consumer="tile.loss_backward_kernel",
    target_consumers={"rocm": "rocm.gfx1151_distribution_loss_backward"},
)
def _execute_distribution_loss(**kwargs: Any) -> NativeVJPResult:
    return _execute_loss_family(**kwargs, loss_kind="distribution_loss")


@register_native_vjp_plugin(
    "matmul",
    family="matmul_backward",
    schedule_consumer="schedule.matmul_backward",
    tile_consumer="tile.matmul",
    target_consumers={"rocm": "rocm.gfx1151_composed_matmul_backward"},
)
def _execute_rocm_matmul_backward(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    source_graph_ir: str | None,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    """Compose dA/dB from the two target-owned forward GEMM packages."""
    del source, source_arg_names, source_graph_ir, frontend_certificate
    import numpy as np

    from tessera.runtime import RuntimeArtifact, launch

    if target != "rocm" or len(ordered_inputs) != 2 or len(arg_names) != 2:
        raise TesseraJitError("ROCm matmul VJP requires exactly A and B")
    cotangents = (
        tuple(out_cotangents)
        if isinstance(out_cotangents, (tuple, list))
        else (out_cotangents,)
    )
    if len(cotangents) != 1:
        raise TesseraJitError("ROCm matmul VJP requires one output cotangent")
    a, b = (np.ascontiguousarray(np.asarray(value)) for value in ordered_inputs)
    dout = np.ascontiguousarray(np.asarray(cotangents[0]))

    def launch_gemm(lhs: Any, rhs: Any) -> tuple[Any, Any]:
        artifact = RuntimeArtifact(
            metadata={
                "target": "rocm",
                "compiler_path": "rocm_compiled",
                "executable": True,
                "execution_kind": "native_gpu",
                "execution_mode": "hip_runtime",
                "arg_names": ["a", "b"],
                "output_name": "c",
                "ops": [
                    {
                        "op_name": "tessera.matmul",
                        "result": "c",
                        "operands": ["a", "b"],
                        "kwargs": {},
                    }
                ],
            }
        )
        result = launch(
            artifact,
            (np.ascontiguousarray(lhs), np.ascontiguousarray(rhs)),
        )
        if not result.get("ok") or result.get("execution_mode") != "hip_runtime":
            raise TesseraJitError(
                "ROCm composed matmul backward failed: " + str(result.get("reason"))
            )
        return result["output"], result.get("physical_attestation")

    da, da_attestation = launch_gemm(dout, b.T)
    db, db_attestation = launch_gemm(a.T, dout)
    gradients = (da, db)
    physical_attestation = _compose_physical_attestations(
        (da_attestation, db_attestation)
    )
    by_name = dict(zip(arg_names, gradients, strict=True))
    return NativeVJPResult(
        gradients=tuple(by_name[name] for name in wrt_names),
        execution={
            "compiler_path": "rocm_compiled+rocm_compiled",
            "execution_kind": "native_gpu",
            "execution_mode": "hip_runtime",
            "evidence_target": "rocm_gfx1151",
            "implementation": "family_plugin_composition",
            "residual_policy": "save_inputs",
            "family": declaration.family,
            "graph_consumer": "tessera.matmul",
            "schedule_consumer": declaration.schedule_consumer,
            "tile_consumer": declaration.tile_consumer,
            "target_consumer": declaration.target_consumers[target],
            "frontend_authority": "tracer",
            "physical_attestation": physical_attestation,
        },
    )


@register_native_vjp_plugin(
    "selective_ssm",
    family="selective_ssm_backward",
    schedule_consumer="schedule.sequence_mixer_backward",
    tile_consumer="tile.training_kernel",
    target_consumers={"rocm": "rocm.gfx1151_selective_ssm_backward"},
    differential_policy="non_reexecuting_state_lineage",
)
def _execute_rocm_selective_ssm_backward(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str],
    out_cotangents: Any,
    wrt_names: Sequence[str],
    declaration: NativeVJPPluginDeclaration,
    source_graph_ir: str | None,
    frontend_certificate: Any | None,
) -> NativeVJPResult:
    """Bind the retained physical kernel to the non-reexecuting plugin seam."""
    del source, source_arg_names, source_graph_ir
    import numpy as np

    from tessera.runtime import RuntimeArtifact, launch

    if target != "rocm" or frontend_certificate is None:
        raise TesseraJitError(
            "ROCm selective-SSM VJP requires non-reexecuting frontend proof"
        )
    cotangents = (
        tuple(out_cotangents)
        if isinstance(out_cotangents, (tuple, list))
        else (out_cotangents,)
    )
    launch_values = [
        *(np.ascontiguousarray(np.asarray(value)) for value in cotangents),
        *(np.ascontiguousarray(np.asarray(value)) for value in ordered_inputs),
    ]
    names = [f"arg{index}" for index in range(len(launch_values))]
    artifact = RuntimeArtifact(
        metadata={
            "target": "rocm",
            "compiler_path": "rocm_selective_ssm_bwd_compiled",
            "executable": True,
            "execution_kind": "native_gpu",
            "execution_mode": "hip_runtime",
            "native_vjp_family": declaration.family,
            "native_vjp_schedule_consumer": declaration.schedule_consumer,
            "native_vjp_tile_consumer": declaration.tile_consumer,
            "native_vjp_target_consumer": declaration.target_consumers[target],
            "arg_names": names,
            "output_name": "grads",
            "ops": [
                {
                    "op_name": "tessera.selective_ssm_bwd",
                    "result": "grads",
                    "operands": names,
                    "kwargs": {},
                }
            ],
        }
    )
    result = launch(artifact, tuple(launch_values))
    if not result.get("ok") or result.get("execution_mode") != "hip_runtime":
        raise TesseraJitError(
            "verified ROCm selective-SSM backward launch failed: "
            + str(result.get("reason"))
        )
    output = result["output"]
    gradients = tuple(output) if isinstance(output, (tuple, list)) else (output,)
    by_name = dict(zip(arg_names, gradients, strict=True))
    return NativeVJPResult(
        gradients=tuple(by_name[name] for name in wrt_names),
        execution={
            "compiler_path": "rocm_selective_ssm_bwd_compiled",
            "execution_kind": "native_gpu",
            "execution_mode": "hip_runtime",
            "evidence_target": "rocm_gfx1151",
            "implementation": "family_plugin",
            "residual_policy": "recompute_all",
            "family": declaration.family,
            "graph_consumer": "tessera.selective_ssm",
            "schedule_consumer": declaration.schedule_consumer,
            "tile_consumer": declaration.tile_consumer,
            "target_consumer": declaration.target_consumers[target],
            "frontend_authority": "tracer",
            "physical_attestation": result.get("physical_attestation"),
        },
    )


def execute_native_vjp_family(
    *,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    source_arg_names: Sequence[str] | None = None,
    out_cotangents: Any,
    wrt_names: Sequence[str],
    source_graph_ir: str | None = None,
    frontend_certificate: Any | None = None,
) -> NativeVJPResult | None:
    """Execute the sole registered owner for ``source``, or return no match."""
    bare = source.op_name.removeprefix("tessera.")
    entry = _PLUGINS.get(bare)
    if entry is None:
        return None
    declaration, executor = entry
    if target not in declaration.target_consumers:
        return None
    result = executor(
        source=source,
        target=target,
        ordered_inputs=ordered_inputs,
        arg_names=arg_names,
        source_arg_names=source_arg_names or arg_names,
        out_cotangents=out_cotangents,
        wrt_names=wrt_names,
        declaration=declaration,
        source_graph_ir=source_graph_ir,
        frontend_certificate=frontend_certificate,
    )
    return _record_execution_certificate(
        declaration=declaration,
        source=source,
        target=target,
        ordered_inputs=ordered_inputs,
        out_cotangents=out_cotangents,
        result=result,
        frontend_certificate=frontend_certificate,
    )


def native_vjp_plugin_declarations() -> Mapping[str, NativeVJPPluginDeclaration]:
    """Return the explicit Graph/Schedule/Tile/Target ownership registry."""
    return {name: declaration for name, (declaration, _) in sorted(_PLUGINS.items())}


def native_vjp_execution_certificates() -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    """Return successful content-addressed executions, grouped by family.

    This process-level registry outlives individual ``JitFn`` objects and each
    row is directly serializable into an evidence packet.
    """
    with _EXECUTION_CERTIFICATE_LOCK:
        return {
            family: tuple(records[digest] for digest in sorted(records))
            for family, records in sorted(_EXECUTION_CERTIFICATES.items())
        }


def native_vjp_exact_execution_coverage() -> frozenset[tuple[str, str]]:
    """Family/target rows backed by a live physical runtime attestation."""
    with _EXECUTION_CERTIFICATE_LOCK:
        return frozenset(
            (family, str(certificate["target"]))
            for family, records in _EXECUTION_CERTIFICATES.items()
            for certificate in records.values()
            if certificate.get("evidence_scope") == "exact_device"
        )


def native_vjp_plugin_owners() -> Mapping[str, str]:
    """Stable inspection surface for totality and audit tests."""
    return {name: executor.__name__ for name, (_, executor) in sorted(_PLUGINS.items())}


def native_vjp_plugin_available(op_name: str, target: str) -> bool:
    """Whether one exact Graph op/Target pair has migrated plugin authority."""
    bare = op_name.removeprefix("tessera.")
    entry = _PLUGINS.get(bare)
    return entry is not None and target in entry[0].target_consumers


def native_vjp_frontend_proof_policy(op_name: str, target: str) -> str | None:
    """Return the declared frontend proof policy for one owned target pair."""
    bare = op_name.removeprefix("tessera.")
    entry = _PLUGINS.get(bare)
    if entry is None or target not in entry[0].target_consumers:
        return None
    return entry[0].differential_policy


def native_vjp_differential_safe(source: Any, target: str, effect: str) -> bool:
    """Whether the plugin permits the concrete call to run in a parity gate."""
    bare = source.op_name.removeprefix("tessera.")
    entry = _PLUGINS.get(bare)
    if entry is None or target not in entry[0].target_consumers:
        return False
    policy = entry[0].differential_policy
    if policy == "non_reexecuting_state_lineage":
        return False
    if effect == "pure":
        return True
    if policy == "zero_dropout_attention":
        dropout = source.kwargs.get("dropout_p", source.kwargs.get("dropout", 0.0))
        return float(dropout or 0.0) == 0.0
    return False


def native_vjp_differential_effect_exemptions(
    source: Any, target: str, effect: str
) -> tuple[str, ...]:
    """Return exact Graph ops admitted by a validated plugin effect policy."""
    bare = source.op_name.removeprefix("tessera.")
    entry = _PLUGINS.get(bare)
    if entry is None or target not in entry[0].target_consumers:
        return ()
    if (
        entry[0].differential_policy == "zero_dropout_attention"
        and native_vjp_differential_safe(source, target, effect)
    ):
        return (source.op_name,)
    return ()


__all__ = [
    "NativeVJPPluginDeclaration",
    "NativeVJPResult",
    "execute_native_vjp_family",
    "native_vjp_differential_effect_exemptions",
    "native_vjp_plugin_available",
    "native_vjp_differential_safe",
    "native_vjp_execution_certificates",
    "native_vjp_exact_execution_coverage",
    "native_vjp_frontend_proof_policy",
    "native_vjp_plugin_declarations",
    "native_vjp_plugin_owners",
    "validate_native_vjp_execution_certificate",
]
