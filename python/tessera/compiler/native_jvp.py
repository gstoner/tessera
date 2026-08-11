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


_SCHEMA = "tessera.native_jvp.v1"
_TARGET_ARCHITECTURES = {
    "x86": "zen5_avx512",
    "rocm": "gfx1151",
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
            outputs = step.get("outputs")
            if outputs is not None and (
                not isinstance(outputs, list)
                or not outputs
                or any(not isinstance(name, str) or not name for name in outputs)
            ):
                raise ValueError("native JVP step outputs must be named")

    def runtime_metadata(self) -> dict[str, Any]:
        self.validate()
        target = str(self.contract["target"])
        return {
            "target": target,
            "compiler_path": f"{target}_jvp_compiled",
            "executable": True,
            "execution_kind": "native_cpu" if target == "x86" else "native_gpu",
            "execution_mode": "cpu_avx512" if target == "x86" else "hip_runtime",
            "autodiff_phase": "forward",
            "native_jvp": dict(self.contract),
        }


def build_native_jvp_artifact(
    *,
    target: str,
    architecture: str,
    family: str,
    paired_jvp_ir: str,
    wrt_indices: Sequence[int],
    arg_names: Sequence[str],
    steps: Sequence[Mapping[str, Any]],
) -> NativeJVPArtifact:
    """Bind a compiler JVP product to exact physical child packages.

    ``gfx1200`` and ``gfx1250`` are intentionally rejected here, even if a
    caller labels them as ROCm: neither has a native JVP evidence packet.
    """
    expected = _TARGET_ARCHITECTURES.get(target)
    if expected is None or architecture != expected:
        raise ValueError(
            "native JVP packages support only zen5_avx512 and gfx1151; "
            "gfx1200/gfx1250 and other unmeasured architectures fail closed"
        )
    if not paired_jvp_ir or "__jvp" not in paired_jvp_ir:
        raise ValueError("native JVP package requires compiler-emitted paired JVP IR")
    normalized_steps = [dict(step) for step in steps]
    body: dict[str, Any] = {
        "schema": _SCHEMA,
        "target": target,
        "architecture": architecture,
        "family": str(family),
        "paired_jvp_ir_digest": _digest_text(paired_jvp_ir),
        "wrt_indices": [int(index) for index in wrt_indices],
        "arg_names": [str(name) for name in arg_names],
        "output_order": ["primal", "tangent"],
        "steps": normalized_steps,
    }
    artifact = NativeJVPArtifact({**body, "artifact_hash": _digest(body)})
    artifact.validate()
    return artifact


def child_digest(metadata: Mapping[str, Any]) -> str:
    """Canonical digest used to bind an existing runtime child package."""
    return _digest(dict(metadata))
