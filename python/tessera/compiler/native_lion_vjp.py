"""Content-addressed, non-reexecuting Lion reverse packages.

E2E-REAL-6D treats the functional flat Lion step as one state-lineage family.
The tracer executes the source exactly once, the retained AST graph is used
only for structural comparison, and the runtime receives the already-lowered
Schedule/Tile package without source Graph IR.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from .frontend_authority import NonReexecutingFrontendCertificate
from .stateful_training import (
    ScheduledLionVJPArtifact,
    lower_scheduled_lion_vjp,
    validate_scheduled_lion_vjp_metadata,
)


SCHEMA = "tessera.native_lion_vjp.v1"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    text = value if isinstance(value, str) else _canonical(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NativeLionVJPPackage:
    source_graph_ir: str
    source_graph_ir_digest: str
    frontend_certificate: NonReexecutingFrontendCertificate
    scheduled: ScheduledLionVJPArtifact
    target: str
    argument_names: tuple[str, str, str]

    @property
    def tile_program_digest(self) -> str:
        return _digest(self.scheduled.tile_ir)

    @property
    def contract(self) -> Mapping[str, Any]:
        body = {
            "schema": SCHEMA,
            "target": self.target,
            "source_graph_ir_digest": self.source_graph_ir_digest,
            "frontend_certificate_digest": self.frontend_certificate.digest,
            "schedule_artifact_hash": self.scheduled.schedule_digest,
            "state_lineage_digest": str(
                self.scheduled.state_contract["artifact_hash"]
            ),
            "tile_program_digest": self.tile_program_digest,
            "argument_names": list(self.argument_names),
            "cotangent_names": ["dparam_out", "dmoment_out"],
            "output_names": [f"d_{name}" for name in self.argument_names],
            "mutation_mode": "functional_no_alias",
        }
        return {**body, "artifact_hash": _digest(body)}

    @property
    def artifact_hash(self) -> str:
        return str(self.contract["artifact_hash"])

    def validate(self) -> None:
        if self.target not in {"x86", "rocm", "nvidia_sm120"}:
            raise ValueError("native Lion VJP has no physical target owner")
        if _digest(self.source_graph_ir) != self.source_graph_ir_digest:
            raise ValueError("native Lion VJP source Graph digest is stale")
        self.frontend_certificate.validate()
        if self.frontend_certificate.contract["graph_consumers"] != [
            "tessera.lion"
        ]:
            raise ValueError("native Lion VJP frontend owner is stale")
        self.scheduled.validate()
        expected = {"x86": "x86", "rocm": "rocm_gfx1151", "nvidia_sm120": "nvidia_sm120"}[self.target]
        if self.scheduled.target != expected:
            raise ValueError("native Lion VJP Schedule target is stale")
        if len(self.argument_names) != 3 or len(set(self.argument_names)) != 3:
            raise ValueError("native Lion VJP requires three distinct arguments")

    def runtime_metadata(self) -> dict[str, Any]:
        """Serialize the physical package while deliberately omitting Graph IR."""
        self.validate()
        execution_mode = {"x86": "cpu_avx512", "rocm": "hip_runtime", "nvidia_sm120": "cuda_driver"}[self.target]
        path = f"{self.target}_lion_bwd_compiled"
        return {
            "target": self.target,
            "compiler_path": path,
            "executable": True,
            "execution_kind": "native_cpu" if self.target == "x86" else "native_gpu",
            "execution_mode": execution_mode,
            "autodiff_phase": "backward",
            "arg_names": [
                *self.argument_names,
                "dparam_out",
                "dmoment_out",
            ],
            "out_cotangents": ["dparam_out", "dmoment_out"],
            "output_names": [f"d_{name}" for name in self.argument_names],
            "state_contract": dict(self.scheduled.state_contract),
            "scheduled_training": self.scheduled.metadata(),
            "frontend_certificate": dict(self.frontend_certificate.contract),
            "native_vjp_package": dict(self.contract),
        }


def validate_native_lion_vjp_runtime_metadata(metadata: Mapping[str, Any]) -> None:
    """Reject stale lineage before either architecture launches native code."""
    if "source_graph_ir" in metadata:
        raise ValueError("native Lion runtime must not receive source Graph IR")
    contract = dict(metadata.get("native_vjp_package") or {})
    body = dict(contract)
    actual = str(body.pop("artifact_hash", ""))
    if body.get("schema") != SCHEMA or actual != _digest(body):
        raise ValueError("native Lion VJP package identity is stale")
    target = str(metadata.get("target", ""))
    if body.get("target") != target:
        raise ValueError("native Lion VJP package target is stale")
    certificate = NonReexecutingFrontendCertificate(
        dict(metadata.get("frontend_certificate") or {})
    )
    certificate.validate()
    if body.get("frontend_certificate_digest") != certificate.digest:
        raise ValueError("native Lion VJP frontend certificate is stale")
    state_contract = dict(metadata.get("state_contract") or {})
    if body.get("state_lineage_digest") != state_contract.get("artifact_hash"):
        raise ValueError("native Lion VJP state lineage is stale")
    scheduled = dict(metadata.get("scheduled_training") or {})
    if body.get("schedule_artifact_hash") != scheduled.get("schedule_digest"):
        raise ValueError("native Lion VJP Schedule identity is stale")
    if body.get("tile_program_digest") != _digest(str(scheduled.get("tile_ir", ""))):
        raise ValueError("native Lion VJP Tile identity is stale")
    argument_names = tuple(str(name) for name in body.get("argument_names", ()))
    if list(metadata.get("arg_names") or []) != [
        *argument_names,
        "dparam_out",
        "dmoment_out",
    ]:
        raise ValueError("native Lion VJP argument ABI is stale")
    descriptor = dict(metadata.get("nvidia_training_descriptor") or {})
    if target == "nvidia_sm120" and (
        descriptor.get("schema") != "tessera.cuda.training_descriptor.v1"
        or descriptor.get("state_lineage_digest") != body.get("state_lineage_digest")
        or descriptor.get("tile_program_digest") != body.get("tile_program_digest")
    ):
        raise ValueError("native Lion CUDA training descriptor is stale")
    shape = tuple(int(dim) for dim in scheduled.get("shape", ()))
    validate_scheduled_lion_vjp_metadata(
        scheduled,
        target={"x86": "x86", "rocm": "rocm_gfx1151", "nvidia_sm120": "nvidia_sm120"}[target],
        shape=shape,
        state_contract=state_contract,
    )


def build_native_lion_vjp_package(
    *,
    source_graph_ir: str,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    out_cotangents: Sequence[Any],
    frontend_certificate: Any,
) -> NativeLionVJPPackage:
    if not source_graph_ir:
        raise ValueError("native Lion VJP requires tracer-owned source Graph IR")
    if source.op_name != "tessera.lion":
        raise ValueError("native Lion VJP requires one tessera.lion Graph op")
    if target not in {"x86", "rocm", "nvidia_sm120"}:
        raise ValueError(f"native Lion VJP has no Target consumer for {target!r}")
    if len(ordered_inputs) != 3 or len(arg_names) != 3:
        raise ValueError("native Lion VJP requires parameter, gradient, and moment")
    if len(out_cotangents) != 2:
        raise ValueError("native Lion VJP requires two output cotangents")
    shapes = [tuple(int(dim) for dim in getattr(value, "shape", ())) for value in ordered_inputs]
    shapes.extend(
        tuple(int(dim) for dim in getattr(value, "shape", ()))
        for value in out_cotangents
    )
    if not shapes[0] or any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError("native Lion VJP operands and cotangents must match")
    dtypes = {str(getattr(value, "dtype", "")) for value in (*ordered_inputs, *out_cotangents)}
    if dtypes != {"float32"}:
        raise ValueError("native Lion VJP currently requires f32 storage")
    if not isinstance(frontend_certificate, NonReexecutingFrontendCertificate):
        raise ValueError("native Lion VJP requires non-reexecuting frontend proof")
    scheduled = lower_scheduled_lion_vjp(
        target={"x86": "x86", "rocm": "rocm_gfx1151", "nvidia_sm120": "nvidia_sm120"}[target],
        shape=shapes[0],
        kwargs=dict(source.kwargs),
    )
    package = NativeLionVJPPackage(
        source_graph_ir=source_graph_ir,
        source_graph_ir_digest=_digest(source_graph_ir),
        frontend_certificate=frontend_certificate,
        scheduled=scheduled,
        target=target,
        argument_names=(
            str(arg_names[0]),
            str(arg_names[1]),
            str(arg_names[2]),
        ),
    )
    package.validate()
    return package


__all__ = [
    "NativeLionVJPPackage",
    "SCHEMA",
    "build_native_lion_vjp_package",
    "validate_native_lion_vjp_runtime_metadata",
]
