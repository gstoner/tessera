"""Content-addressed native forward-product packages.

The C++ forward autodiff pass remains the semantic authority.  This module
binds its paired Graph IR product to already-typed physical child packages; it
does not rediscover a derivative from the original Graph operation at launch.
The first executable envelope is intentionally single-operation and exact-
architecture gated so a successful host-free transform cannot be mistaken for
native evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence


_SCHEMA = "tessera.native_jvp.v3"
_TARGET_ARCHITECTURES = {
    "x86": "zen5_avx512",
    "rocm": "gfx1151",
    "nvidia_sm120": "sm120",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: object) -> str:
    return _digest_text(_canonical_json(value))


@dataclass(frozen=True)
class NativeJVPArtifact:
    """One paired primal/tangent package and its immutable launch lineage."""

    contract: Mapping[str, Any]

    @property
    def artifact_hash(self) -> str:
        return str(self.contract["artifact_hash"])

    def validate(self) -> None:
        body = dict(self.contract)
        actual = str(body.pop("artifact_hash", ""))
        if body.get("schema") != _SCHEMA or actual != _digest(body):
            raise ValueError("native JVP artifact has stale content identity")
        target = str(body.get("target", ""))
        if body.get("architecture") != _TARGET_ARCHITECTURES.get(target):
            raise ValueError("native JVP artifact is not bound to an exact architecture")
        steps = body.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("native JVP artifact requires an ordered launch plan")
        ids = [str(step.get("id", "")) for step in steps if isinstance(step, dict)]
        if len(ids) != len(steps) or len(ids) != len(set(ids)):
            raise ValueError("native JVP launch step identities must be unique")
        for step in steps:
            if not isinstance(step.get("child_digest"), str) or len(step["child_digest"]) != 64:
                raise ValueError("native JVP launch step lacks a child artifact digest")
            has_metadata = isinstance(step.get("child_metadata"), dict)
            has_artifact = isinstance(step.get("child_artifact"), dict)
            if has_metadata == has_artifact:
                raise ValueError(
                    "native JVP step requires exactly one metadata or descriptor child"
                )
            dependencies = step.get("depends_on", [])
            if not isinstance(dependencies, list) or any(
                not isinstance(item, str) or item not in ids for item in dependencies
            ):
                raise ValueError("native JVP step has invalid dependencies")
            if any(ids.index(item) >= ids.index(str(step["id"])) for item in dependencies):
                raise ValueError("native JVP dependencies must precede their consumer")
            outputs = step.get("outputs")
            if outputs is not None and (
                not isinstance(outputs, list)
                or not outputs
                or any(not isinstance(name, str) or not name for name in outputs)
            ):
                raise ValueError("native JVP step outputs must be named")
        schedule = body.get("schedule_program")
        tile = body.get("tile_program")
        if not isinstance(schedule, dict) or not isinstance(tile, dict):
            raise ValueError("native JVP artifact requires Schedule and Tile programs")
        schedule_body = dict(schedule)
        schedule_digest = str(schedule_body.pop("digest", ""))
        if schedule_digest != _digest(schedule_body):
            raise ValueError("native JVP Schedule program has stale identity")
        tile_body = dict(tile)
        tile_digest = str(tile_body.pop("digest", ""))
        if tile_digest != _digest(tile_body):
            raise ValueError("native JVP Tile program has stale identity")
        if tile_body.get("schedule_digest") != schedule_digest:
            raise ValueError("native JVP Tile program has stale Schedule lineage")

    def runtime_metadata(self) -> dict[str, Any]:
        self.validate()
        target = str(self.contract["target"])
        return {
            "target": target,
            "compiler_path": f"{target}_jvp_compiled",
            "executable": True,
            "execution_kind": "native_cpu" if target == "x86" else "native_gpu",
            "execution_mode": ("cpu_avx512" if target == "x86" else
                               "cuda_runtime" if target == "nvidia_sm120" else
                               "hip_runtime"),
            "autodiff_phase": "forward",
            "native_jvp": dict(self.contract),
        }


def build_native_jvp_artifact(
    *,
    target: str,
    architecture: str,
    family: str,
    source_graph_ir: str,
    paired_jvp_ir: str,
    wrt_indices: Sequence[int],
    arg_names: Sequence[str],
    steps: Sequence[Mapping[str, Any]],
    consumer_declaration: Mapping[str, Any] | None = None,
    schedule_program: Mapping[str, Any] | None = None,
    tile_program: Mapping[str, Any] | None = None,
) -> NativeJVPArtifact:
    """Bind a compiler JVP product to exact physical child packages.

    ``gfx1200`` and ``gfx1250`` are intentionally rejected here, even if a
    caller labels them as ROCm: neither has a native JVP evidence packet.
    """
    expected = _TARGET_ARCHITECTURES.get(target)
    if expected is None or architecture != expected:
        raise ValueError(
            "native JVP packages support only zen5_avx512, gfx1151, and sm120; "
            "gfx1200/gfx1250 and other unmeasured architectures fail closed"
        )
    if not source_graph_ir or 'tessera.frontend.authority = "tracer"' not in source_graph_ir:
        raise ValueError("native JVP package requires tracer-owned source Graph IR")
    if not paired_jvp_ir or "__jvp" not in paired_jvp_ir:
        raise ValueError("native JVP package requires compiler-emitted paired JVP IR")
    normalized_steps = [dict(step) for step in steps]
    schedule_body = dict(schedule_program or {
        "schema": "tessera.native_jvp_schedule.v1",
        "family": str(family),
        "actions": [
            {
                "id": str(step["id"]),
                "depends_on": list(step.get("depends_on", [])),
            }
            for step in normalized_steps
        ],
    })
    schedule_body.pop("digest", None)
    normalized_schedule = {**schedule_body, "digest": _digest(schedule_body)}
    tile_body = dict(tile_program or {
        "schema": "tessera.native_jvp_tile.v1",
        "family": str(family),
        "schedule_digest": normalized_schedule["digest"],
        "actions": [
            {"id": str(step["id"]), "child_digest": str(step["child_digest"])}
            for step in normalized_steps
        ],
    })
    tile_body.pop("digest", None)
    tile_body["schedule_digest"] = normalized_schedule["digest"]
    normalized_tile = {**tile_body, "digest": _digest(tile_body)}
    body: dict[str, Any] = {
        "schema": _SCHEMA,
        "target": target,
        "architecture": architecture,
        "family": str(family),
        "source_graph_ir_digest": _digest_text(source_graph_ir),
        "paired_jvp_ir_digest": _digest_text(paired_jvp_ir),
        "wrt_indices": [int(index) for index in wrt_indices],
        "arg_names": [str(name) for name in arg_names],
        "output_order": ["primal", "tangent"],
        "steps": normalized_steps,
        "schedule_program": normalized_schedule,
        "tile_program": normalized_tile,
    }
    if consumer_declaration is not None:
        body["consumer_declaration"] = dict(consumer_declaration)
    artifact = NativeJVPArtifact({**body, "artifact_hash": _digest(body)})
    artifact.validate()
    return artifact


def child_digest(metadata: Mapping[str, Any]) -> str:
    """Canonical digest used to bind an existing runtime child package."""
    return _digest(dict(metadata))
