"""Non-reexecuting native packages for stateful training reverse products.

E2E-REAL-6E moves Adafactor and sequence-mixer package construction out of
``JitFn``.  The runtime sees a content-addressed physical package, never the
source Graph program, while the package binds the one-execution frontend proof
to the existing state, Schedule, and exact Tile identities.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from .frontend_authority import NonReexecutingFrontendCertificate
from .stateful_training import (
    ScheduledAdafactorVJPArtifact,
    ScheduledSequenceMixerBackwardArtifact,
    lower_scheduled_adafactor_vjp,
    lower_scheduled_sequence_mixer_backward,
    validate_scheduled_adafactor_vjp_metadata,
    validate_scheduled_sequence_mixer_backward_metadata,
)


SCHEMA = "tessera.native_stateful_vjp.v1"
_SEQUENCE_FAMILIES = {
    "gated_deltanet",
    "kimi_delta_attention",
    "modified_delta_attention",
}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    text = value if isinstance(value, str) else _canonical(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NativeStatefulVJPPackage:
    source_graph_ir: str
    source_graph_ir_digest: str
    frontend_certificate: NonReexecutingFrontendCertificate
    scheduled: ScheduledAdafactorVJPArtifact | ScheduledSequenceMixerBackwardArtifact
    target: str
    graph_consumer: str
    argument_names: tuple[str, ...]
    cotangent_names: tuple[str, ...]
    output_names: tuple[str, ...]
    compiler_path: str

    @property
    def tile_program_digest(self) -> str:
        return _digest(self.scheduled.tile_ir)

    @property
    def state_lineage_digest(self) -> str:
        return str(self.scheduled.state_contract["artifact_hash"])

    @property
    def contract(self) -> Mapping[str, Any]:
        body = {
            "schema": SCHEMA,
            "target": self.target,
            "graph_consumer": self.graph_consumer,
            "source_graph_ir_digest": self.source_graph_ir_digest,
            "frontend_certificate_digest": self.frontend_certificate.digest,
            "schedule_artifact_hash": self.scheduled.schedule_digest,
            "state_lineage_digest": self.state_lineage_digest,
            "tile_program_digest": self.tile_program_digest,
            "argument_names": list(self.argument_names),
            "cotangent_names": list(self.cotangent_names),
            "output_names": list(self.output_names),
            "compiler_path": self.compiler_path,
            "mutation_mode": "functional_no_alias",
        }
        return {**body, "artifact_hash": _digest(body)}

    @property
    def artifact_hash(self) -> str:
        return str(self.contract["artifact_hash"])

    def validate(self) -> None:
        if self.target not in {"x86", "rocm"}:
            raise ValueError("stateful native VJP has no physical target owner")
        if _digest(self.source_graph_ir) != self.source_graph_ir_digest:
            raise ValueError("stateful native VJP source Graph digest is stale")
        self.frontend_certificate.validate()
        if self.frontend_certificate.contract["graph_consumers"] != [
            self.graph_consumer
        ]:
            raise ValueError("stateful native VJP frontend owner is stale")
        self.scheduled.validate()
        expected = "x86" if self.target == "x86" else "rocm_gfx1151"
        if self.scheduled.target != expected:
            raise ValueError("stateful native VJP Schedule target is stale")
        if not self.argument_names or len(set(self.argument_names)) != len(
            self.argument_names
        ):
            raise ValueError("stateful native VJP arguments must be distinct")

    def runtime_metadata(self) -> dict[str, Any]:
        self.validate()
        execution_mode = "cpu_avx512" if self.target == "x86" else "hip_runtime"
        metadata = {
            "target": self.target,
            "compiler_path": self.compiler_path,
            "executable": True,
            "execution_kind": "native_cpu" if self.target == "x86" else "native_gpu",
            "execution_mode": execution_mode,
            "autodiff_phase": "backward",
            "arg_names": [*self.argument_names, *self.cotangent_names],
            "output_names": list(self.output_names),
            "state_contract": dict(self.scheduled.state_contract),
            "scheduled_training": self.scheduled.metadata(),
            "frontend_certificate": dict(self.frontend_certificate.contract),
            "native_stateful_vjp_package": dict(self.contract),
        }
        if len(self.cotangent_names) == 1:
            metadata["out_cotangent"] = self.cotangent_names[0]
        return metadata


def _require_common(
    *,
    source_graph_ir: str,
    source: Any,
    target: str,
    frontend_certificate: Any,
) -> NonReexecutingFrontendCertificate:
    if not source_graph_ir:
        raise ValueError("stateful native VJP requires tracer-owned source Graph IR")
    if target not in {"x86", "rocm"}:
        raise ValueError(f"stateful native VJP has no Target consumer for {target!r}")
    if not isinstance(frontend_certificate, NonReexecutingFrontendCertificate):
        raise ValueError("stateful native VJP requires non-reexecuting frontend proof")
    if not str(source.op_name).startswith("tessera."):
        raise ValueError("stateful native VJP requires a registered Graph operation")
    return frontend_certificate


def build_native_adafactor_vjp_package(
    *,
    source_graph_ir: str,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    out_cotangents: Sequence[Any],
    frontend_certificate: Any,
) -> NativeStatefulVJPPackage:
    certificate = _require_common(
        source_graph_ir=source_graph_ir,
        source=source,
        target=target,
        frontend_certificate=frontend_certificate,
    )
    if source.op_name != "tessera.adafactor":
        raise ValueError("Adafactor VJP requires one tessera.adafactor Graph op")
    if len(ordered_inputs) not in {3, 4} or len(arg_names) != len(ordered_inputs):
        raise ValueError("Adafactor VJP requires p/g plus full or factored state")
    if len(out_cotangents) != 1:
        raise ValueError("Adafactor VJP requires one parameter-output cotangent")
    shapes = [tuple(int(dim) for dim in getattr(value, "shape", ())) for value in ordered_inputs]
    parameter_shape = shapes[0]
    if not parameter_shape or shapes[1] != parameter_shape:
        raise ValueError("Adafactor parameter and gradient shapes must match")
    topology = "factored" if len(ordered_inputs) == 4 else "full"
    expected_states = (
        [parameter_shape[:-1], (parameter_shape[-1],)]
        if topology == "factored"
        else [parameter_shape]
    )
    if shapes[2:] != expected_states:
        raise ValueError("Adafactor state shapes disagree with selected topology")
    if tuple(int(dim) for dim in getattr(out_cotangents[0], "shape", ())) != parameter_shape:
        raise ValueError("Adafactor parameter cotangent shape must match parameter")
    dtypes = {str(getattr(value, "dtype", "")) for value in (*ordered_inputs, *out_cotangents)}
    if dtypes != {"float32"}:
        raise ValueError("Adafactor native VJP currently requires f32 storage")
    scheduled = lower_scheduled_adafactor_vjp(
        target="x86" if target == "x86" else "rocm_gfx1151",
        parameter_shape=parameter_shape,
        topology=topology,
        kwargs=dict(source.kwargs),
    )
    package = NativeStatefulVJPPackage(
        source_graph_ir=source_graph_ir,
        source_graph_ir_digest=_digest(source_graph_ir),
        frontend_certificate=certificate,
        scheduled=scheduled,
        target=target,
        graph_consumer=source.op_name,
        argument_names=tuple(str(name) for name in arg_names),
        cotangent_names=("dy",),
        output_names=tuple(f"d_{name}" for name in arg_names),
        compiler_path=f"{target}_adafactor_bwd_compiled",
    )
    package.validate()
    return package


def build_native_sequence_mixer_vjp_package(
    *,
    source_graph_ir: str,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    out_cotangents: Sequence[Any],
    frontend_certificate: Any,
) -> NativeStatefulVJPPackage:
    certificate = _require_common(
        source_graph_ir=source_graph_ir,
        source=source,
        target=target,
        frontend_certificate=frontend_certificate,
    )
    family = source.op_name.removeprefix("tessera.")
    if family not in _SEQUENCE_FAMILIES or not bool(source.kwargs.get("causal", True)):
        raise ValueError("sequence-mixer VJP requires one causal DeltaNet Graph op")
    if len(ordered_inputs) != 6 or len(arg_names) != 6:
        raise ValueError("sequence-mixer VJP requires Q/K/V/gate/beta/decay")
    if len(out_cotangents) != 1:
        raise ValueError("sequence-mixer VJP requires one output cotangent")
    shapes = [tuple(int(dim) for dim in getattr(value, "shape", ())) for value in ordered_inputs]
    q_shape, k_shape, v_shape, gate_shape, beta_shape, decay_shape = shapes
    if (
        len(q_shape) != 4
        or k_shape != q_shape
        or v_shape[:3] != q_shape[:3]
        or gate_shape != v_shape
        or beta_shape != q_shape[:3]
        or decay_shape != q_shape[:3]
        or tuple(int(dim) for dim in getattr(out_cotangents[0], "shape", ())) != v_shape
    ):
        raise ValueError("sequence-mixer VJP operand shapes violate the typed ABI")
    dtypes = {str(getattr(value, "dtype", "")) for value in (*ordered_inputs, *out_cotangents)}
    if dtypes != {"float32"}:
        raise ValueError("sequence-mixer native VJP currently requires f32 storage")
    scheduled = lower_scheduled_sequence_mixer_backward(
        target="x86" if target == "x86" else "rocm_gfx1151",
        family=family,
        q_shape=q_shape,
        v_shape=v_shape,
        erase=bool(source.kwargs.get("erase", False)),
        chunk_size=int(source.kwargs.get("chunk_size", 64)),
        parallel_chunks=bool(source.kwargs.get("parallel_chunks", True)),
    )
    package = NativeStatefulVJPPackage(
        source_graph_ir=source_graph_ir,
        source_graph_ir_digest=_digest(source_graph_ir),
        frontend_certificate=certificate,
        scheduled=scheduled,
        target=target,
        graph_consumer=source.op_name,
        argument_names=tuple(str(name) for name in arg_names),
        cotangent_names=("dy",),
        output_names=tuple(f"d_{name}" for name in arg_names),
        compiler_path=f"{target}_deltanet_bwd_compiled",
    )
    package.validate()
    return package


def validate_native_stateful_vjp_runtime_metadata(metadata: Mapping[str, Any]) -> None:
    if "source_graph_ir" in metadata or "ops" in metadata:
        raise ValueError("stateful native VJP runtime must not receive source Graph IR")
    contract = dict(metadata.get("native_stateful_vjp_package") or {})
    body = dict(contract)
    actual = str(body.pop("artifact_hash", ""))
    if body.get("schema") != SCHEMA or actual != _digest(body):
        raise ValueError("stateful native VJP package identity is stale")
    target = str(metadata.get("target", ""))
    if body.get("target") != target or body.get("compiler_path") != metadata.get("compiler_path"):
        raise ValueError("stateful native VJP target identity is stale")
    certificate = NonReexecutingFrontendCertificate(
        dict(metadata.get("frontend_certificate") or {})
    )
    certificate.validate()
    if body.get("frontend_certificate_digest") != certificate.digest:
        raise ValueError("stateful native VJP frontend certificate is stale")
    state_contract = dict(metadata.get("state_contract") or {})
    scheduled = dict(metadata.get("scheduled_training") or {})
    if body.get("state_lineage_digest") != state_contract.get("artifact_hash"):
        raise ValueError("stateful native VJP state lineage is stale")
    if body.get("schedule_artifact_hash") != scheduled.get("schedule_digest"):
        raise ValueError("stateful native VJP Schedule identity is stale")
    if body.get("tile_program_digest") != _digest(str(scheduled.get("tile_ir", ""))):
        raise ValueError("stateful native VJP Tile identity is stale")
    if list(metadata.get("arg_names") or []) != [
        *body.get("argument_names", []),
        *body.get("cotangent_names", []),
    ]:
        raise ValueError("stateful native VJP argument ABI is stale")
    graph_consumer = str(body.get("graph_consumer", ""))
    physical_target = "x86" if target == "x86" else "rocm_gfx1151"
    if graph_consumer == "tessera.adafactor":
        validate_scheduled_adafactor_vjp_metadata(
            scheduled,
            target=physical_target,
            parameter_shape=tuple(scheduled.get("parameter_shape", ())),
            topology=str(scheduled.get("topology", "")),
            state_contract=state_contract,
        )
    elif graph_consumer.removeprefix("tessera.") in _SEQUENCE_FAMILIES:
        validate_scheduled_sequence_mixer_backward_metadata(
            scheduled,
            target=physical_target,
            family=graph_consumer.removeprefix("tessera."),
            q_shape=tuple(scheduled.get("q_shape", ())),
            v_shape=tuple(scheduled.get("v_shape", ())),
            state_contract=state_contract,
        )
    else:
        raise ValueError("stateful native VJP package has an unknown Graph consumer")


__all__ = [
    "NativeStatefulVJPPackage",
    "SCHEMA",
    "build_native_adafactor_vjp_package",
    "build_native_sequence_mixer_vjp_package",
    "validate_native_stateful_vjp_runtime_metadata",
]
