"""Portable native-image and launch-descriptor contracts.

E2E-SPINE-1 defines the compiler/plugin boundary without selecting a backend
or changing runtime dispatch.  Backend-owned schedules stay in Target IR and
native payloads; this module owns only content identity, ABI bindings, launch
requirements, and deterministic serialization.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from tessera.dtype import TesseraDtypeError, canonicalize_dtype

from .capabilities import normalize_target
from .pipeline_registry import pipeline_lookup


NATIVE_IMAGE_SCHEMA_VERSION = "tessera.native_image.v1"
LAUNCH_DESCRIPTOR_SCHEMA_VERSION = "tessera.launch_descriptor.v1"

NATIVE_IMAGE_FORMATS: frozenset[str] = frozenset({
    "ptx", "cubin", "hsaco", "elf", "object", "shared_object",
    "metallib", "msl_package",
})
COMPILE_STATES: frozenset[str] = frozenset({
    "cold", "warm_cache", "prepackaged",
})
DEVICE_LIBRARY_LINK_MODES: frozenset[str] = frozenset({
    "llvm_link_only_needed", "compiler_driver", "embedded",
})

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL_RE = re.compile(r"^[A-Za-z_.$][A-Za-z0-9_.$@-]*$")
_DIRECTIONS = frozenset({"input", "output", "inout"})
_SHAPE_PREDICATES = frozenset({"eq", "min", "max", "multiple_of"})
_RESIDENCY = frozenset({"none", "inputs", "outputs", "all"})
_WORKSPACE_LIFETIMES = frozenset({"launch", "session"})
_WORKSPACE_INITIALIZATION = frozenset({"undefined", "zero", "preserve"})


class ArtifactContractError(ValueError):
    """A stable, registered native-artifact contract failure."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _error(code: str, detail: str) -> ArtifactContractError:
    return ArtifactContractError(code, detail)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(data: object) -> str:
    return _sha256_bytes(_canonical_json(data).encode("utf-8"))


def _validate_digest(value: str, field_name: str, code: str) -> None:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise _error(code, f"{field_name} must be a lowercase SHA-256 digest")


def _json_value(value: object, field_name: str, code: str) -> object:
    """Return a deterministic JSON-compatible copy or reject the value."""
    if isinstance(value, float) and not math.isfinite(value):
        raise _error(code, f"{field_name} contains a non-finite float")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise _error(code, f"{field_name} keys must be strings")
        out: dict[str, object] = {}
        for key in sorted(value):
            out[key] = _json_value(value[key], f"{field_name}.{key}", code)
        return out
    if isinstance(value, (list, tuple)):
        return [_json_value(item, field_name, code) for item in value]
    raise _error(code, f"{field_name} contains non-JSON value {type(value).__name__}")


def _canonical_dtype(value: str, code: str, field_name: str) -> str:
    try:
        return canonicalize_dtype(value, allow_planned_gated=True)
    except (TesseraDtypeError, TypeError) as exc:
        raise _error(code, f"{field_name} has invalid dtype {value!r}: {exc}") from exc


def _string_field(data: Mapping[str, object], name: str, code: str) -> str:
    if name not in data:
        raise _error(code, f"missing required field {name}")
    value = data[name]
    if not isinstance(value, str):
        raise _error(code, f"{name} must be a string")
    return value


def _int_field(
    data: Mapping[str, object], name: str, code: str, *, default: int | None = None,
) -> int:
    if name not in data:
        if default is not None:
            return default
        raise _error(code, f"missing required field {name}")
    value = data[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise _error(code, f"{name} must be an integer")
    return value


def _bool_field(
    data: Mapping[str, object], name: str, code: str, *, default: bool,
) -> bool:
    if name not in data:
        return default
    value = data[name]
    if not isinstance(value, bool):
        raise _error(code, f"{name} must be a boolean")
    return value


def _object_field(
    data: Mapping[str, object], name: str, code: str, *, required: bool = True,
) -> Mapping[str, object]:
    if name not in data:
        if required:
            raise _error(code, f"missing required field {name}")
        return {}
    value = data[name]
    if not isinstance(value, Mapping):
        raise _error(code, f"{name} must be an object")
    return value


def _array_field(
    data: Mapping[str, object], name: str, code: str, *, required: bool = True,
) -> Sequence[object]:
    if name not in data:
        if required:
            raise _error(code, f"missing required field {name}")
        return ()
    value = data[name]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _error(code, f"{name} must be an array")
    return value


@dataclass(frozen=True)
class NativeEntryPoint:
    symbol: str
    abi_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or _SYMBOL_RE.fullmatch(self.symbol) is None:
            raise _error("E_NATIVE_IMAGE_SCHEMA", f"invalid entry symbol {self.symbol!r}")
        if not isinstance(self.abi_id, str) or not self.abi_id.strip():
            raise _error("E_NATIVE_IMAGE_SCHEMA", "entry ABI identifier must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"symbol": self.symbol, "abi_id": self.abi_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "NativeEntryPoint":
        return cls(
            symbol=_string_field(data, "symbol", "E_NATIVE_IMAGE_SCHEMA"),
            abi_id=_string_field(data, "abi_id", "E_NATIVE_IMAGE_SCHEMA"),
        )


@dataclass(frozen=True)
class ResourceRecord:
    provenance: str
    metrics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise _error("E_NATIVE_IMAGE_SCHEMA", "resource provenance must be non-empty")
        clean = _json_value(self.metrics, "resource metrics", "E_NATIVE_IMAGE_SCHEMA")
        object.__setattr__(self, "metrics", clean)

    def to_dict(self) -> dict[str, object]:
        return {"provenance": self.provenance, "metrics": dict(self.metrics)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ResourceRecord":
        return cls(
            provenance=_string_field(data, "provenance", "E_NATIVE_IMAGE_SCHEMA"),
            metrics=_object_field(data, "metrics", "E_NATIVE_IMAGE_SCHEMA", required=False),
        )


@dataclass(frozen=True)
class DeviceLibraryRecord:
    """Content identity for one device library consumed at the LLVM stage."""

    logical_name: str
    content_digest: str
    link_mode: str

    def __post_init__(self) -> None:
        if not isinstance(self.logical_name, str) or not self.logical_name.strip():
            raise _error("E_NATIVE_IMAGE_SCHEMA", "device library name must be non-empty")
        _validate_digest(
            self.content_digest, "device library content_digest", "E_NATIVE_IMAGE_SCHEMA"
        )
        if self.link_mode not in DEVICE_LIBRARY_LINK_MODES:
            raise _error(
                "E_NATIVE_IMAGE_SCHEMA",
                f"device library link mode {self.link_mode!r} is not registered",
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "logical_name": self.logical_name,
            "content_digest": self.content_digest,
            "link_mode": self.link_mode,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "DeviceLibraryRecord":
        return cls(
            logical_name=_string_field(data, "logical_name", "E_NATIVE_IMAGE_SCHEMA"),
            content_digest=_string_field(
                data, "content_digest", "E_NATIVE_IMAGE_SCHEMA"
            ),
            link_mode=_string_field(data, "link_mode", "E_NATIVE_IMAGE_SCHEMA"),
        )


@dataclass(frozen=True)
class NativeImageArtifact:
    target: str
    architecture: str
    pipeline_name: str
    compiler_fingerprint: str
    toolchain_fingerprint: str
    target_ir_digest: str
    binary_format: str
    payload: bytes
    entry_points: tuple[NativeEntryPoint, ...]
    compile_state: str
    resource_record: ResourceRecord | None = None
    device_libraries: tuple[DeviceLibraryRecord, ...] = ()
    schema_version: str = NATIVE_IMAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != NATIVE_IMAGE_SCHEMA_VERSION:
            raise _error(
                "E_NATIVE_IMAGE_SCHEMA",
                f"unsupported native-image schema {self.schema_version!r}",
            )
        try:
            target = normalize_target(self.target)
        except (TypeError, ValueError) as exc:
            raise _error("E_NATIVE_IMAGE_SCHEMA", f"invalid target {self.target!r}") from exc
        object.__setattr__(self, "target", target)
        if not isinstance(self.architecture, str) or not self.architecture.strip():
            raise _error("E_NATIVE_IMAGE_SCHEMA", "exact architecture must be non-empty")
        pipeline = pipeline_lookup(self.pipeline_name)
        if pipeline is None:
            raise _error(
                "E_NATIVE_IMAGE_SCHEMA",
                f"pipeline {self.pipeline_name!r} is not registered",
            )
        if target not in pipeline.targets:
            raise _error(
                "E_NATIVE_IMAGE_SCHEMA",
                f"pipeline {self.pipeline_name!r} does not declare target {target!r}",
            )
        for name, value in (
            ("compiler_fingerprint", self.compiler_fingerprint),
            ("toolchain_fingerprint", self.toolchain_fingerprint),
        ):
            if not isinstance(value, str) or not value.strip():
                raise _error("E_NATIVE_IMAGE_SCHEMA", f"{name} must be non-empty")
        _validate_digest(self.target_ir_digest, "target_ir_digest", "E_NATIVE_IMAGE_SCHEMA")
        if self.binary_format not in NATIVE_IMAGE_FORMATS:
            raise _error(
                "E_NATIVE_IMAGE_SCHEMA",
                f"binary format {self.binary_format!r} is not registered",
            )
        if not isinstance(self.payload, bytes) or not self.payload:
            raise _error("E_NATIVE_IMAGE_SCHEMA", "native payload must be non-empty bytes")
        object.__setattr__(self, "entry_points", tuple(self.entry_points))
        if not self.entry_points:
            raise _error("E_NATIVE_IMAGE_SCHEMA", "native image needs at least one entry point")
        if any(not isinstance(entry, NativeEntryPoint) for entry in self.entry_points):
            raise _error("E_NATIVE_IMAGE_SCHEMA", "entry_points must contain NativeEntryPoint values")
        if self.resource_record is not None and not isinstance(self.resource_record, ResourceRecord):
            raise _error("E_NATIVE_IMAGE_SCHEMA", "resource_record must be a ResourceRecord")
        object.__setattr__(self, "device_libraries", tuple(self.device_libraries))
        if any(
            not isinstance(library, DeviceLibraryRecord)
            for library in self.device_libraries
        ):
            raise _error(
                "E_NATIVE_IMAGE_SCHEMA",
                "device_libraries must contain DeviceLibraryRecord values",
            )
        library_names = [library.logical_name for library in self.device_libraries]
        if len(library_names) != len(set(library_names)):
            raise _error("E_NATIVE_IMAGE_SCHEMA", "native image has duplicate device libraries")
        symbols = [entry.symbol for entry in self.entry_points]
        if len(symbols) != len(set(symbols)):
            raise _error("E_NATIVE_IMAGE_SCHEMA", "native image has duplicate entry symbols")
        if self.compile_state not in COMPILE_STATES:
            raise _error(
                "E_NATIVE_IMAGE_SCHEMA",
                f"compile state {self.compile_state!r} is not registered",
            )

    @property
    def payload_digest(self) -> str:
        return _sha256_bytes(self.payload)

    def _identity_dict(self) -> dict[str, object]:
        identity: dict[str, object] = {
            "schema_version": self.schema_version,
            "target": self.target,
            "architecture": self.architecture,
            "pipeline_name": self.pipeline_name,
            "compiler_fingerprint": self.compiler_fingerprint,
            "toolchain_fingerprint": self.toolchain_fingerprint,
            "target_ir_digest": self.target_ir_digest,
            "binary_format": self.binary_format,
            "payload_digest": self.payload_digest,
            "entry_points": [entry.to_dict() for entry in self.entry_points],
        }
        if self.device_libraries:
            identity["device_libraries"] = [
                library.to_dict() for library in self.device_libraries
            ]
        return identity

    @property
    def image_digest(self) -> str:
        return _sha256_json(self._identity_dict())

    @property
    def cache_key(self) -> str:
        """Pre-compilation cache key; compile state and measured resources do not affect it."""
        identity: dict[str, object] = {
            "schema_version": self.schema_version,
            "target": self.target,
            "architecture": self.architecture,
            "pipeline_name": self.pipeline_name,
            "compiler_fingerprint": self.compiler_fingerprint,
            "toolchain_fingerprint": self.toolchain_fingerprint,
            "target_ir_digest": self.target_ir_digest,
            "binary_format": self.binary_format,
        }
        if self.device_libraries:
            identity["device_libraries"] = [
                library.to_dict() for library in self.device_libraries
            ]
        return _sha256_json(identity)

    def entry_point(self, symbol: str) -> NativeEntryPoint | None:
        return next((entry for entry in self.entry_points if entry.symbol == symbol), None)

    def to_dict(self) -> dict[str, object]:
        return {
            **self._identity_dict(),
            "device_libraries": [
                library.to_dict() for library in self.device_libraries
            ],
            "payload_b64": base64.b64encode(self.payload).decode("ascii"),
            "image_digest": self.image_digest,
            "compile_state": self.compile_state,
            "cache_key": self.cache_key,
            "resource_record": (
                self.resource_record.to_dict() if self.resource_record is not None else None
            ),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "NativeImageArtifact":
        try:
            payload_text = _string_field(data, "payload_b64", "E_NATIVE_IMAGE_SCHEMA")
            payload = base64.b64decode(payload_text, validate=True)
            entries_raw = _array_field(data, "entry_points", "E_NATIVE_IMAGE_SCHEMA")
            entries_list: list[NativeEntryPoint] = []
            for entry in entries_raw:
                if not isinstance(entry, Mapping):
                    raise _error("E_NATIVE_IMAGE_SCHEMA", "entry point must be an object")
                entries_list.append(NativeEntryPoint.from_dict(entry))
            libraries_raw = _array_field(
                data, "device_libraries", "E_NATIVE_IMAGE_SCHEMA", required=False
            )
            libraries_list: list[DeviceLibraryRecord] = []
            for library in libraries_raw:
                if not isinstance(library, Mapping):
                    raise _error("E_NATIVE_IMAGE_SCHEMA", "device library must be an object")
                libraries_list.append(DeviceLibraryRecord.from_dict(library))
            resource_raw = data.get("resource_record")
            if resource_raw is not None and not isinstance(resource_raw, Mapping):
                raise _error("E_NATIVE_IMAGE_SCHEMA", "resource_record must be an object")
            resource = None if resource_raw is None else ResourceRecord.from_dict(resource_raw)
            artifact = cls(
                schema_version=_string_field(data, "schema_version", "E_NATIVE_IMAGE_SCHEMA"),
                target=_string_field(data, "target", "E_NATIVE_IMAGE_SCHEMA"),
                architecture=_string_field(data, "architecture", "E_NATIVE_IMAGE_SCHEMA"),
                pipeline_name=_string_field(data, "pipeline_name", "E_NATIVE_IMAGE_SCHEMA"),
                compiler_fingerprint=_string_field(data, "compiler_fingerprint", "E_NATIVE_IMAGE_SCHEMA"),
                toolchain_fingerprint=_string_field(data, "toolchain_fingerprint", "E_NATIVE_IMAGE_SCHEMA"),
                target_ir_digest=_string_field(data, "target_ir_digest", "E_NATIVE_IMAGE_SCHEMA"),
                binary_format=_string_field(data, "binary_format", "E_NATIVE_IMAGE_SCHEMA"),
                payload=payload,
                entry_points=tuple(entries_list),
                compile_state=_string_field(data, "compile_state", "E_NATIVE_IMAGE_SCHEMA"),
                resource_record=resource,
                device_libraries=tuple(libraries_list),
            )
        except ArtifactContractError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise _error("E_NATIVE_IMAGE_SCHEMA", f"malformed native image: {exc}") from exc
        for field_name, actual in (
            ("payload_digest", artifact.payload_digest),
            ("image_digest", artifact.image_digest),
            ("cache_key", artifact.cache_key),
        ):
            persisted = data.get(field_name)
            if persisted != actual:
                raise _error(
                    "E_NATIVE_IMAGE_DIGEST_MISMATCH",
                    f"{field_name} does not match serialized native-image content",
                )
        return artifact

    @classmethod
    def from_json(cls, payload: str | bytes) -> "NativeImageArtifact":
        try:
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            data = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise _error("E_NATIVE_IMAGE_SCHEMA", f"invalid native-image JSON: {exc}") from exc
        if not isinstance(data, Mapping):
            raise _error("E_NATIVE_IMAGE_SCHEMA", "native-image JSON root must be an object")
        return cls.from_dict(data)


@dataclass(frozen=True)
class BufferBinding:
    ordinal: int
    name: str
    direction: str
    dtype: str
    rank: int
    layout: str
    alignment: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.ordinal, int) or isinstance(self.ordinal, bool) or self.ordinal < 0:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "buffer ordinal must be non-negative")
        if not isinstance(self.name, str) or not self.name:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "buffer name must be non-empty")
        if self.direction not in _DIRECTIONS:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", f"invalid buffer direction {self.direction!r}")
        object.__setattr__(self, "dtype", _canonical_dtype(
            self.dtype, "E_LAUNCH_DESCRIPTOR_SCHEMA", f"buffer {self.name}",
        ))
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 0:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "buffer rank must be non-negative")
        if not isinstance(self.layout, str) or not self.layout:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "buffer layout must be non-empty")
        if (not isinstance(self.alignment, int) or isinstance(self.alignment, bool)
                or self.alignment <= 0 or self.alignment & (self.alignment - 1)):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "buffer alignment must be a positive power of two")

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal, "name": self.name, "direction": self.direction,
            "dtype": self.dtype, "rank": self.rank, "layout": self.layout,
            "alignment": self.alignment,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "BufferBinding":
        return cls(
            ordinal=_int_field(data, "ordinal", "E_LAUNCH_DESCRIPTOR_SCHEMA"),
            name=_string_field(data, "name", "E_LAUNCH_DESCRIPTOR_SCHEMA"),
            direction=_string_field(data, "direction", "E_LAUNCH_DESCRIPTOR_SCHEMA"),
            dtype=_string_field(data, "dtype", "E_LAUNCH_DESCRIPTOR_SCHEMA"),
            rank=_int_field(data, "rank", "E_LAUNCH_DESCRIPTOR_SCHEMA"),
            layout=_string_field(data, "layout", "E_LAUNCH_DESCRIPTOR_SCHEMA"),
            alignment=_int_field(data, "alignment", "E_LAUNCH_DESCRIPTOR_SCHEMA", default=1),
        )


@dataclass(frozen=True)
class ScalarArgument:
    ordinal: int
    name: str
    dtype: str

    def __post_init__(self) -> None:
        if not isinstance(self.ordinal, int) or isinstance(self.ordinal, bool) or self.ordinal < 0:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "scalar ordinal must be non-negative")
        if not isinstance(self.name, str) or not self.name:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "scalar name must be non-empty")
        object.__setattr__(self, "dtype", _canonical_dtype(
            self.dtype, "E_LAUNCH_DESCRIPTOR_SCHEMA", f"scalar {self.name}",
        ))

    def to_dict(self) -> dict[str, object]:
        return {"ordinal": self.ordinal, "name": self.name, "dtype": self.dtype}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ScalarArgument":
        return cls(
            ordinal=_int_field(data, "ordinal", "E_LAUNCH_DESCRIPTOR_SCHEMA"),
            name=_string_field(data, "name", "E_LAUNCH_DESCRIPTOR_SCHEMA"),
            dtype=_string_field(data, "dtype", "E_LAUNCH_DESCRIPTOR_SCHEMA"),
        )


@dataclass(frozen=True)
class ShapeGuard:
    binding: str
    dimension: int
    predicate: str
    value: int

    def __post_init__(self) -> None:
        if not self.binding:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "shape guard binding must be non-empty")
        if not isinstance(self.dimension, int) or isinstance(self.dimension, bool) or self.dimension < 0:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "shape guard dimension must be non-negative")
        if self.predicate not in _SHAPE_PREDICATES:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", f"invalid shape predicate {self.predicate!r}")
        if not isinstance(self.value, int) or isinstance(self.value, bool) or self.value <= 0:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "shape guard value must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "binding": self.binding, "dimension": self.dimension,
            "predicate": self.predicate, "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ShapeGuard":
        return cls(
            binding=_string_field(data, "binding", "E_LAUNCH_DESCRIPTOR_SCHEMA"),
            dimension=_int_field(data, "dimension", "E_LAUNCH_DESCRIPTOR_SCHEMA"),
            predicate=_string_field(data, "predicate", "E_LAUNCH_DESCRIPTOR_SCHEMA"),
            value=_int_field(data, "value", "E_LAUNCH_DESCRIPTOR_SCHEMA"),
        )


def _dimension3(value: tuple[int, int, int] | None, field_name: str) -> None:
    if value is None:
        return
    if len(value) != 3 or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value
    ):
        raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", f"{field_name} must contain three positive integers")


@dataclass(frozen=True)
class LaunchGeometry:
    grid: tuple[int, int, int] | None = None
    workgroup: tuple[int, int, int] | None = None
    policy: str | None = None

    def __post_init__(self) -> None:
        _dimension3(self.grid, "grid")
        _dimension3(self.workgroup, "workgroup")
        if self.policy is not None and not self.policy.strip():
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "geometry policy must be non-empty")
        fixed = self.grid is not None or self.workgroup is not None
        if fixed == (self.policy is not None):
            raise _error(
                "E_LAUNCH_DESCRIPTOR_SCHEMA",
                "launch geometry must use either fixed dimensions or one runtime policy",
            )
        if fixed and (self.grid is None or self.workgroup is None):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "fixed geometry needs both grid and workgroup")

    def to_dict(self) -> dict[str, object]:
        return {
            "grid": list(self.grid) if self.grid is not None else None,
            "workgroup": list(self.workgroup) if self.workgroup is not None else None,
            "policy": self.policy,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "LaunchGeometry":
        def dims(name: str) -> tuple[int, int, int] | None:
            raw = data.get(name)
            if raw is None:
                return None
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
                raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", f"{name} must be a three-element array")
            if any(not isinstance(item, int) or isinstance(item, bool) for item in raw):
                raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", f"{name} entries must be integers")
            return (raw[0], raw[1], raw[2])

        policy_raw = data.get("policy")
        if policy_raw is not None and not isinstance(policy_raw, str):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "policy must be a string or null")
        return cls(grid=dims("grid"), workgroup=dims("workgroup"), policy=policy_raw)


@dataclass(frozen=True)
class WorkspaceRequirement:
    bytes: int = 0
    alignment: int = 1
    lifetime: str = "launch"
    initialization: str = "undefined"

    def __post_init__(self) -> None:
        if not isinstance(self.bytes, int) or isinstance(self.bytes, bool) or self.bytes < 0:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "workspace bytes must be non-negative")
        if (not isinstance(self.alignment, int) or isinstance(self.alignment, bool)
                or self.alignment <= 0 or self.alignment & (self.alignment - 1)):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "workspace alignment must be a positive power of two")
        if self.lifetime not in _WORKSPACE_LIFETIMES:
            raise _error(
                "E_LAUNCH_DESCRIPTOR_SCHEMA",
                f"invalid workspace lifetime {self.lifetime!r}",
            )
        if self.initialization not in _WORKSPACE_INITIALIZATION:
            raise _error(
                "E_LAUNCH_DESCRIPTOR_SCHEMA",
                f"invalid workspace initialization {self.initialization!r}",
            )
        if self.lifetime == "launch" and self.initialization == "preserve":
            raise _error(
                "E_LAUNCH_DESCRIPTOR_SCHEMA",
                "preserved workspace requires session lifetime",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "bytes": self.bytes,
            "alignment": self.alignment,
            "lifetime": self.lifetime,
            "initialization": self.initialization,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "WorkspaceRequirement":
        return cls(
            bytes=_int_field(data, "bytes", "E_LAUNCH_DESCRIPTOR_SCHEMA", default=0),
            alignment=_int_field(
                data, "alignment", "E_LAUNCH_DESCRIPTOR_SCHEMA", default=1,
            ),
            lifetime=(
                _string_field(data, "lifetime", "E_LAUNCH_DESCRIPTOR_SCHEMA")
                if "lifetime" in data else "launch"
            ),
            initialization=(
                _string_field(data, "initialization", "E_LAUNCH_DESCRIPTOR_SCHEMA")
                if "initialization" in data else "undefined"
            ),
        )


@dataclass(frozen=True)
class OrderingSemantics:
    ordered_submission: bool = True
    residency: str = "none"
    synchronization: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.residency not in _RESIDENCY:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", f"invalid residency {self.residency!r}")
        if any(not isinstance(item, str) or not item for item in self.synchronization):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "synchronization tokens must be non-empty strings")

    def to_dict(self) -> dict[str, object]:
        return {
            "ordered_submission": self.ordered_submission,
            "residency": self.residency,
            "synchronization": list(self.synchronization),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "OrderingSemantics":
        raw = _array_field(
            data, "synchronization", "E_LAUNCH_DESCRIPTOR_SCHEMA", required=False,
        )
        if any(not isinstance(item, str) for item in raw):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "synchronization entries must be strings")
        synchronization = tuple(item for item in raw if isinstance(item, str))
        return cls(
            ordered_submission=_bool_field(
                data, "ordered_submission", "E_LAUNCH_DESCRIPTOR_SCHEMA", default=True,
            ),
            residency=(
                _string_field(data, "residency", "E_LAUNCH_DESCRIPTOR_SCHEMA")
                if "residency" in data else "none"
            ),
            synchronization=synchronization,
        )


@dataclass(frozen=True)
class BufferArgument:
    dtype: str
    shape: tuple[int, ...]
    layout: str
    address_alignment: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", _canonical_dtype(
            self.dtype, "E_LAUNCH_BINDING_MISMATCH", "runtime buffer",
        ))
        object.__setattr__(self, "shape", tuple(self.shape))
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in self.shape):
            raise _error("E_LAUNCH_BINDING_MISMATCH", "runtime shape dimensions must be non-negative")
        if not self.layout:
            raise _error("E_LAUNCH_BINDING_MISMATCH", "runtime layout must be non-empty")
        if (not isinstance(self.address_alignment, int)
                or isinstance(self.address_alignment, bool)
                or self.address_alignment <= 0):
            raise _error("E_LAUNCH_BINDING_MISMATCH", "runtime alignment must be positive")


@dataclass(frozen=True)
class LaunchDescriptor:
    image_digest: str
    entry_symbol: str
    abi_id: str
    buffers: tuple[BufferBinding, ...]
    scalars: tuple[ScalarArgument, ...] = ()
    shape_guards: tuple[ShapeGuard, ...] = ()
    geometry: LaunchGeometry = field(default_factory=lambda: LaunchGeometry(policy="runtime_default"))
    dynamic_local_memory_bytes: int = 0
    workspace: WorkspaceRequirement = field(default_factory=WorkspaceRequirement)
    ordering: OrderingSemantics = field(default_factory=OrderingSemantics)
    provenance: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = LAUNCH_DESCRIPTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LAUNCH_DESCRIPTOR_SCHEMA_VERSION:
            raise _error(
                "E_LAUNCH_DESCRIPTOR_SCHEMA",
                f"unsupported launch-descriptor schema {self.schema_version!r}",
            )
        _validate_digest(self.image_digest, "image_digest", "E_LAUNCH_DESCRIPTOR_SCHEMA")
        if _SYMBOL_RE.fullmatch(self.entry_symbol) is None:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", f"invalid entry symbol {self.entry_symbol!r}")
        if not self.abi_id:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "ABI identifier must be non-empty")
        object.__setattr__(self, "buffers", tuple(self.buffers))
        object.__setattr__(self, "scalars", tuple(self.scalars))
        object.__setattr__(self, "shape_guards", tuple(self.shape_guards))
        if any(not isinstance(item, BufferBinding) for item in self.buffers):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "buffers must contain BufferBinding values")
        if any(not isinstance(item, ScalarArgument) for item in self.scalars):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "scalars must contain ScalarArgument values")
        if any(not isinstance(item, ShapeGuard) for item in self.shape_guards):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "shape_guards must contain ShapeGuard values")
        if not isinstance(self.geometry, LaunchGeometry):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "geometry must be LaunchGeometry")
        if not isinstance(self.workspace, WorkspaceRequirement):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "workspace must be WorkspaceRequirement")
        if not isinstance(self.ordering, OrderingSemantics):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "ordering must be OrderingSemantics")
        ordinals = [item.ordinal for item in self.buffers]
        ordinals.extend(item.ordinal for item in self.scalars)
        names = [item.name for item in self.buffers]
        names.extend(item.name for item in self.scalars)
        if len(ordinals) != len(set(ordinals)):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "argument ordinals must be unique")
        if ordinals and sorted(ordinals) != list(range(len(ordinals))):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "argument ordinals must be contiguous from zero")
        if len(names) != len(set(names)):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "argument names must be unique")
        by_name = {binding.name: binding for binding in self.buffers}
        for guard in self.shape_guards:
            binding = by_name.get(guard.binding)
            if binding is None:
                raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", f"shape guard references unknown buffer {guard.binding!r}")
            if guard.dimension >= binding.rank:
                raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", f"shape guard dimension exceeds rank of {guard.binding!r}")
        if (not isinstance(self.dynamic_local_memory_bytes, int)
                or isinstance(self.dynamic_local_memory_bytes, bool)
                or self.dynamic_local_memory_bytes < 0):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "dynamic local memory must be non-negative")
        clean = _json_value(self.provenance, "launch provenance", "E_LAUNCH_DESCRIPTOR_SCHEMA")
        object.__setattr__(self, "provenance", clean)

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "image_digest": self.image_digest,
            "entry_symbol": self.entry_symbol,
            "abi_id": self.abi_id,
            "buffers": [binding.to_dict() for binding in self.buffers],
            "scalars": [scalar.to_dict() for scalar in self.scalars],
            "shape_guards": [guard.to_dict() for guard in self.shape_guards],
            "geometry": self.geometry.to_dict(),
            "dynamic_local_memory_bytes": self.dynamic_local_memory_bytes,
            "workspace": self.workspace.to_dict(),
            "ordering": self.ordering.to_dict(),
            "provenance": dict(self.provenance),
        }

    @property
    def descriptor_digest(self) -> str:
        return _sha256_json(self._content_dict())

    @property
    def cache_fingerprint(self) -> str:
        return _sha256_json({
            "schema_version": self.schema_version,
            "image_digest": self.image_digest,
            "descriptor_digest": self.descriptor_digest,
        })

    def to_dict(self) -> dict[str, object]:
        return {
            **self._content_dict(),
            "descriptor_digest": self.descriptor_digest,
            "cache_fingerprint": self.cache_fingerprint,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "LaunchDescriptor":
        def objects(name: str) -> list[Mapping[str, object]]:
            raw = _array_field(
                data, name, "E_LAUNCH_DESCRIPTOR_SCHEMA", required=False,
            )
            if any(not isinstance(item, Mapping) for item in raw):
                raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", f"{name} entries must be objects")
            return list(raw)  # type: ignore[arg-type]

        try:
            geometry_raw = _object_field(
                data, "geometry", "E_LAUNCH_DESCRIPTOR_SCHEMA",
            )
            workspace_raw = _object_field(
                data, "workspace", "E_LAUNCH_DESCRIPTOR_SCHEMA", required=False,
            )
            ordering_raw = _object_field(
                data, "ordering", "E_LAUNCH_DESCRIPTOR_SCHEMA", required=False,
            )
            provenance = _object_field(
                data, "provenance", "E_LAUNCH_DESCRIPTOR_SCHEMA", required=False,
            )
            descriptor = cls(
                schema_version=_string_field(
                    data, "schema_version", "E_LAUNCH_DESCRIPTOR_SCHEMA",
                ),
                image_digest=_string_field(
                    data, "image_digest", "E_LAUNCH_DESCRIPTOR_SCHEMA",
                ),
                entry_symbol=_string_field(
                    data, "entry_symbol", "E_LAUNCH_DESCRIPTOR_SCHEMA",
                ),
                abi_id=_string_field(data, "abi_id", "E_LAUNCH_DESCRIPTOR_SCHEMA"),
                buffers=tuple(BufferBinding.from_dict(item) for item in objects("buffers")),
                scalars=tuple(ScalarArgument.from_dict(item) for item in objects("scalars")),
                shape_guards=tuple(ShapeGuard.from_dict(item) for item in objects("shape_guards")),
                geometry=LaunchGeometry.from_dict(geometry_raw),
                dynamic_local_memory_bytes=_int_field(
                    data, "dynamic_local_memory_bytes",
                    "E_LAUNCH_DESCRIPTOR_SCHEMA", default=0,
                ),
                workspace=WorkspaceRequirement.from_dict(workspace_raw),
                ordering=OrderingSemantics.from_dict(ordering_raw),
                provenance=provenance,
            )
        except ArtifactContractError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", f"malformed launch descriptor: {exc}") from exc
        for field_name, actual in (
            ("descriptor_digest", descriptor.descriptor_digest),
            ("cache_fingerprint", descriptor.cache_fingerprint),
        ):
            if data.get(field_name) != actual:
                raise _error(
                    "E_LAUNCH_DESCRIPTOR_SCHEMA",
                    f"{field_name} does not match serialized descriptor content",
                )
        return descriptor

    @classmethod
    def from_json(cls, payload: str | bytes) -> "LaunchDescriptor":
        try:
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            data = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", f"invalid launch JSON: {exc}") from exc
        if not isinstance(data, Mapping):
            raise _error("E_LAUNCH_DESCRIPTOR_SCHEMA", "launch JSON root must be an object")
        return cls.from_dict(data)

    def validate_image(self, image: NativeImageArtifact) -> None:
        if self.image_digest != image.image_digest:
            raise _error("E_LAUNCH_STALE_IMAGE", "launch descriptor references a different native image")
        entry = image.entry_point(self.entry_symbol)
        if entry is None:
            raise _error("E_LAUNCH_STALE_IMAGE", f"entry symbol {self.entry_symbol!r} is absent from native image")
        if entry.abi_id != self.abi_id:
            raise _error("E_LAUNCH_STALE_IMAGE", "launch ABI identifier does not match native-image entry point")

    def validate_invocation(
        self,
        image: NativeImageArtifact,
        buffers: Mapping[str, BufferArgument],
        scalars: Mapping[str, object],
    ) -> None:
        """Validate invocation metadata before any backend submission."""
        self.validate_image(image)
        expected_buffers = {binding.name for binding in self.buffers}
        if set(buffers) != expected_buffers:
            raise _error(
                "E_LAUNCH_BINDING_MISMATCH",
                f"buffer names differ: expected {sorted(expected_buffers)}, got {sorted(buffers)}",
            )
        expected_scalars = {scalar.name for scalar in self.scalars}
        if set(scalars) != expected_scalars:
            raise _error(
                "E_LAUNCH_BINDING_MISMATCH",
                f"scalar names differ: expected {sorted(expected_scalars)}, got {sorted(scalars)}",
            )
        for binding in self.buffers:
            actual = buffers[binding.name]
            if actual.dtype != binding.dtype:
                raise _error("E_LAUNCH_BINDING_MISMATCH", f"buffer {binding.name!r} dtype mismatch")
            if len(actual.shape) != binding.rank:
                raise _error("E_LAUNCH_BINDING_MISMATCH", f"buffer {binding.name!r} rank mismatch")
            if actual.layout != binding.layout:
                raise _error("E_LAUNCH_BINDING_MISMATCH", f"buffer {binding.name!r} layout mismatch")
            if actual.address_alignment < binding.alignment or actual.address_alignment % binding.alignment:
                raise _error("E_LAUNCH_BINDING_MISMATCH", f"buffer {binding.name!r} alignment mismatch")
        for scalar in self.scalars:
            if scalar.name not in scalars:
                continue
            value = scalars[scalar.name]
            if scalar.dtype.startswith("int") and (not isinstance(value, int) or isinstance(value, bool)):
                raise _error("E_LAUNCH_BINDING_MISMATCH", f"scalar {scalar.name!r} requires an integer")
            if scalar.dtype.startswith(("fp", "bf")) and not isinstance(value, (int, float)):
                raise _error("E_LAUNCH_BINDING_MISMATCH", f"scalar {scalar.name!r} requires a number")
            if scalar.dtype == "bool" and not isinstance(value, bool):
                raise _error("E_LAUNCH_BINDING_MISMATCH", f"scalar {scalar.name!r} requires bool")
        for guard in self.shape_guards:
            dim = buffers[guard.binding].shape[guard.dimension]
            passes = {
                "eq": dim == guard.value,
                "min": dim >= guard.value,
                "max": dim <= guard.value,
                "multiple_of": dim % guard.value == 0,
            }[guard.predicate]
            if not passes:
                raise _error(
                    "E_LAUNCH_BINDING_MISMATCH",
                    f"shape guard failed for {guard.binding}[{guard.dimension}]: "
                    f"{guard.predicate} {guard.value}, got {dim}",
                )


__all__ = [
    "ArtifactContractError",
    "BufferArgument",
    "BufferBinding",
    "COMPILE_STATES",
    "DEVICE_LIBRARY_LINK_MODES",
    "DeviceLibraryRecord",
    "LAUNCH_DESCRIPTOR_SCHEMA_VERSION",
    "LaunchDescriptor",
    "LaunchGeometry",
    "NATIVE_IMAGE_FORMATS",
    "NATIVE_IMAGE_SCHEMA_VERSION",
    "NativeEntryPoint",
    "NativeImageArtifact",
    "OrderingSemantics",
    "ResourceRecord",
    "ScalarArgument",
    "ShapeGuard",
    "WorkspaceRequirement",
]
