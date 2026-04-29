"""Tessera Inference Server foundation.

This module defines the lightweight Python contracts for Tessera inference
packages, scheduler/KV-cache metadata, route registration, and health/metrics.
It is not a network server yet; production HTTP/gRPC transports will bind to
these validated objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional


SUPPORTED_BACKENDS = ("ptx", "rocm", "level_zero", "cpu")
SUPPORTED_API_COMPAT = ("openai-v1", "tessera-v1")
SUPPORTED_SCHEDULERS = ("continuous_batch", "sequence_batch", "priority")
SUPPORTED_KV_SPILL = ("host_pinned", "none", "host")


class ServerConfigError(ValueError):
    """Raised when an inference server/package config is invalid."""


@dataclass(frozen=True)
class KVCacheConfig:
    pages: int
    page_size: str
    swap: str = "host_pinned"

    def __post_init__(self) -> None:
        if self.pages <= 0:
            raise ServerConfigError("kv_cache.pages must be > 0")
        if not self.page_size:
            raise ServerConfigError("kv_cache.page_size is required")
        if self.swap not in SUPPORTED_KV_SPILL:
            raise ServerConfigError(f"kv_cache.swap must be one of {SUPPORTED_KV_SPILL}")

    @property
    def page_size_bytes(self) -> int:
        return _parse_size(self.page_size)


@dataclass(frozen=True)
class ModelManifest:
    """Normative .tspkg manifest fields."""

    name: str
    version: str
    entry: str
    mesh: Mapping[str, int]
    dtypes: tuple[str, ...]
    kv_cache: KVCacheConfig
    autotune: Mapping[str, object] = field(default_factory=dict)
    compat: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.version or not self.entry:
            raise ServerConfigError("manifest requires name, version, and entry")
        if not self.mesh:
            raise ServerConfigError("manifest mesh is required")
        for axis, size in self.mesh.items():
            if not axis or int(size) <= 0:
                raise ServerConfigError(f"invalid mesh axis {axis!r}={size!r}")
        if not self.dtypes:
            raise ServerConfigError("manifest dtypes must be non-empty")
        api = self.compat.get("api", "tessera-v1")
        if api not in SUPPORTED_API_COMPAT:
            raise ServerConfigError(f"compat.api must be one of {SUPPORTED_API_COMPAT}")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ModelManifest":
        kv = data.get("kv_cache", {})
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            entry=str(data["entry"]),
            mesh={str(k): int(v) for k, v in dict(data.get("mesh", {})).items()},
            dtypes=tuple(str(v) for v in data.get("dtypes", ())),
            kv_cache=KVCacheConfig(
                pages=int(kv.get("pages", 0)),
                page_size=str(kv.get("page_size", "")),
                swap=str(kv.get("swap", kv.get("spill", "host_pinned"))),
            ),
            autotune=dict(data.get("autotune", {})),
            compat=dict(data.get("compat", {})),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "entry": self.entry,
            "mesh": dict(self.mesh),
            "dtypes": list(self.dtypes),
            "kv_cache": {
                "pages": self.kv_cache.pages,
                "page_size": self.kv_cache.page_size,
                "swap": self.kv_cache.swap,
            },
            "autotune": dict(self.autotune),
            "compat": dict(self.compat),
        }


@dataclass(frozen=True)
class TesseraPackage:
    """Loaded .tspkg package metadata."""

    path: Path
    manifest: ModelManifest

    @property
    def entry(self) -> str:
        return self.manifest.entry

    def validate_layout(self) -> None:
        required = ["manifest.yaml"]
        missing = [name for name in required if not (self.path / name).exists()]
        if missing:
            raise ServerConfigError(f"package missing required file(s): {missing}")


@dataclass(frozen=True)
class RuntimeCapabilities:
    backend: str
    arch: str = "generic"
    tensor_cores: bool = False
    fp8: bool = False
    int8: bool = True
    hbm_gb: float = 0.0
    nvlink: bool = False

    def supports_dtype(self, dtype: str) -> bool:
        if dtype.startswith("fp8"):
            return self.fp8
        if dtype == "int8":
            return self.int8
        return dtype in ("bf16", "fp16", "fp32", "fp64")


@dataclass(frozen=True)
class SchedulerConfig:
    policy: str = "continuous_batch"
    target_p99_ms: int = 150
    max_batch_tokens: int = 2048
    priority_lanes: tuple[str, ...] = ("p0", "p1", "p2")
    speculative: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.policy not in SUPPORTED_SCHEDULERS:
            raise ServerConfigError(f"scheduler policy must be one of {SUPPORTED_SCHEDULERS}")
        if self.target_p99_ms <= 0 or self.max_batch_tokens <= 0:
            raise ServerConfigError("target_p99_ms and max_batch_tokens must be > 0")


class SchedulerSession:
    def __init__(self, config: SchedulerConfig, model: str):
        self.config = config
        self.model = model

    def __enter__(self) -> "SchedulerSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def generate(self, messages: Iterable[Mapping[str, str]], *, max_tokens: int):
        if max_tokens < 0:
            raise ServerConfigError("max_tokens must be >= 0")
        prompt = " ".join(str(m.get("content", "")) for m in messages)
        for i in range(max_tokens):
            yield {"index": i, "delta": "", "model": self.model, "prompt_len": len(prompt)}


class Scheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config

    def session(self, *, model: str) -> SchedulerSession:
        return SchedulerSession(self.config, model=model)


class KVCacheManager:
    """Paged KV cache capacity/accounting helper."""

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self._allocated_pages = 0

    @property
    def capacity_bytes(self) -> int:
        return self.config.pages * self.config.page_size_bytes

    @property
    def allocated_pages(self) -> int:
        return self._allocated_pages

    def allocate(self, pages: int) -> None:
        if pages < 0:
            raise ServerConfigError("pages must be >= 0")
        if self._allocated_pages + pages > self.config.pages:
            raise MemoryError("KV cache capacity exceeded")
        self._allocated_pages += pages

    def evict(self, pages: int) -> None:
        if pages < 0:
            raise ServerConfigError("pages must be >= 0")
        self._allocated_pages = max(0, self._allocated_pages - pages)

    def hit_rate(self, hits: int, misses: int) -> float:
        total = hits + misses
        return hits / total if total else 0.0


@dataclass(frozen=True)
class Route:
    path: str
    handler: Callable
    stream: bool = False


class App:
    """In-process registry for inference server routes and models."""

    def __init__(self, config: str | Mapping[str, object] | None = None):
        self.config = config
        self.models: dict[str, Callable] = {}
        self.routes: dict[str, Route] = {}

    def model(self, path: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.models[path] = fn
            return fn

        return decorator

    def route(self, path: str, *, stream: bool = False) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.routes[path] = Route(path=path, handler=fn, stream=stream)
            return fn

        return decorator

    def healthz(self) -> Mapping[str, object]:
        return {
            "ready": True,
            "models": sorted(self.models),
            "routes": sorted(self.routes),
        }

    def metrics(self) -> Mapping[str, float]:
        return {
            "tis_models_loaded": float(len(self.models)),
            "tis_routes_registered": float(len(self.routes)),
        }

    def run(self) -> Mapping[str, object]:
        return self.healthz()


def load_package(path: str | Path) -> TesseraPackage:
    pkg_path = Path(path)
    manifest_path = pkg_path / "manifest.yaml"
    if not manifest_path.exists():
        raise ServerConfigError(f"package manifest not found: {manifest_path}")
    manifest = ModelManifest.from_dict(_load_mapping(manifest_path))
    package = TesseraPackage(path=pkg_path, manifest=manifest)
    package.validate_layout()
    return package


def scheduler(
    *,
    policy: str = "continuous_batch",
    target_p99_ms: int = 150,
    max_batch_tokens: int = 2048,
    speculative: Optional[Mapping[str, object]] = None,
) -> Scheduler:
    return Scheduler(SchedulerConfig(
        policy=policy,
        target_p99_ms=target_p99_ms,
        max_batch_tokens=max_batch_tokens,
        speculative=dict(speculative or {}),
    ))


def capabilities(
    *,
    backend: str = "ptx",
    arch: str = "generic",
    tensor_cores: bool = True,
    fp8: bool = False,
    int8: bool = True,
    hbm_gb: float = 0.0,
    nvlink: bool = False,
) -> RuntimeCapabilities:
    if backend not in SUPPORTED_BACKENDS:
        raise ServerConfigError(f"backend must be one of {SUPPORTED_BACKENDS}")
    return RuntimeCapabilities(
        backend=backend,
        arch=arch,
        tensor_cores=tensor_cores,
        fp8=fp8,
        int8=int8,
        hbm_gb=hbm_gb,
        nvlink=nvlink,
    )


def _load_mapping(path: Path) -> Mapping[str, object]:
    text = path.read_text()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ServerConfigError("manifest must be a mapping")
    return data


def _parse_size(text: str) -> int:
    s = text.strip()
    units = {
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
    }
    for suffix, scale in units.items():
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * scale)
    return int(s)


__all__ = [
    "App",
    "KVCacheConfig",
    "KVCacheManager",
    "ModelManifest",
    "Route",
    "RuntimeCapabilities",
    "Scheduler",
    "SchedulerConfig",
    "SchedulerSession",
    "ServerConfigError",
    "TesseraPackage",
    "capabilities",
    "load_package",
    "scheduler",
]
