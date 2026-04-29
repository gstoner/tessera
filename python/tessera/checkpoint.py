"""Runtime checkpoint manifest helpers for Tessera."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional


class CheckpointError(RuntimeError):
    """Raised when checkpoint save/load fails."""


@dataclass(frozen=True)
class TensorShard:
    """Checkpoint metadata for one logical tensor or shard."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    sharding: Mapping[str, object] = field(default_factory=dict)
    path: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "sharding": dict(self.sharding),
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TensorShard":
        return cls(
            name=str(data["name"]),
            shape=tuple(int(v) for v in data.get("shape", ())),
            dtype=str(data.get("dtype", "unknown")),
            sharding=dict(data.get("sharding", {})),
            path=str(data.get("path", "")),
        )


@dataclass(frozen=True)
class CheckpointManifest:
    """Atomic sharded checkpoint manifest."""

    version: int
    tag: str
    step: Optional[int] = None
    backend: str = "python"
    mesh: Mapping[str, int] = field(default_factory=dict)
    numerics: str = "fast"
    rng: Mapping[str, object] = field(default_factory=dict)
    reduce_tree_id: str = ""
    parameters: tuple[TensorShard, ...] = ()
    optimizer: Mapping[str, object] = field(default_factory=dict)
    autotune_cache: Mapping[str, object] = field(default_factory=dict)
    committed: bool = False

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "tag": self.tag,
            "step": self.step,
            "backend": self.backend,
            "mesh": dict(self.mesh),
            "numerics": self.numerics,
            "rng": dict(self.rng),
            "reduce_tree_id": self.reduce_tree_id,
            "parameters": [p.to_dict() for p in self.parameters],
            "optimizer": dict(self.optimizer),
            "autotune_cache": dict(self.autotune_cache),
            "committed": self.committed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CheckpointManifest":
        return cls(
            version=int(data.get("version", 1)),
            tag=str(data["tag"]),
            step=data.get("step"),
            backend=str(data.get("backend", "python")),
            mesh=dict(data.get("mesh", {})),
            numerics=str(data.get("numerics", "fast")),
            rng=dict(data.get("rng", {})),
            reduce_tree_id=str(data.get("reduce_tree_id", "")),
            parameters=tuple(TensorShard.from_dict(p) for p in data.get("parameters", ())),
            optimizer=dict(data.get("optimizer", {})),
            autotune_cache=dict(data.get("autotune_cache", {})),
            committed=bool(data.get("committed", False)),
        )


@dataclass(frozen=True)
class CheckpointState:
    """Loaded checkpoint state."""

    manifest: CheckpointManifest
    tensors: Mapping[str, TensorShard]
    optimizer: Mapping[str, object]


@dataclass
class AsyncCheckpointConfig:
    """Async checkpointing policy."""

    enabled: bool = False
    max_bandwidth_gbps: float = 0.0
    flush_interval_s: float = 0.0


_ASYNC_CONFIG = AsyncCheckpointConfig()


def enable_async(*, max_bandwidth_gbps: float, flush_interval_s: float) -> AsyncCheckpointConfig:
    if max_bandwidth_gbps <= 0 or flush_interval_s <= 0:
        raise ValueError("max_bandwidth_gbps and flush_interval_s must be > 0")
    _ASYNC_CONFIG.enabled = True
    _ASYNC_CONFIG.max_bandwidth_gbps = max_bandwidth_gbps
    _ASYNC_CONFIG.flush_interval_s = flush_interval_s
    return _ASYNC_CONFIG


def save(
    *,
    tag: str,
    tensors: Optional[Mapping[str, object]] = None,
    optimizer: Optional[Mapping[str, object]] = None,
    mesh: Optional[Mapping[str, int]] = None,
    root: str | os.PathLike = ".tessera_ckpt",
    atomic: bool = True,
    step: Optional[int] = None,
    backend: str = "python",
    numerics: str = "fast",
    rng: Optional[Mapping[str, object]] = None,
    reduce_tree_id: str = "",
    autotune_cache: Optional[Mapping[str, object]] = None,
) -> CheckpointManifest:
    """Write a manifest and placeholder shard files for a checkpoint."""

    if not tag:
        raise ValueError("checkpoint tag is required")
    ckpt_dir = Path(root) / tag
    tmp_dir = Path(root) / f".{tag}.tmp"
    work_dir = tmp_dir if atomic else ckpt_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    shards = tuple(_tensor_shard(name, value, work_dir) for name, value in (tensors or {}).items())
    manifest = CheckpointManifest(
        version=1,
        tag=tag,
        step=step,
        backend=backend,
        mesh=dict(mesh or {}),
        numerics=numerics,
        rng=dict(rng or {}),
        reduce_tree_id=reduce_tree_id,
        parameters=shards,
        optimizer=dict(optimizer or {}),
        autotune_cache=dict(autotune_cache or {}),
        committed=False,
    )
    (work_dir / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    if atomic:
        committed = CheckpointManifest(**{**manifest.__dict__, "committed": True})
        (work_dir / "manifest.json").write_text(json.dumps(committed.to_dict(), indent=2, sort_keys=True))
        (work_dir / "COMMITTED").write_text(str(time.time()))
        if ckpt_dir.exists():
            _remove_tree(ckpt_dir)
        work_dir.rename(ckpt_dir)
        return committed
    return manifest


def load(
    tag: str,
    *,
    root: str | os.PathLike = ".tessera_ckpt",
    remap_to: Optional[Mapping[str, int]] = None,
) -> CheckpointState:
    ckpt_dir = Path(root) / tag
    manifest_path = ckpt_dir / "manifest.json"
    if not manifest_path.exists():
        raise CheckpointError(f"checkpoint manifest not found: {manifest_path}")
    manifest = CheckpointManifest.from_dict(json.loads(manifest_path.read_text()))
    if not manifest.committed or not (ckpt_dir / "COMMITTED").exists():
        raise CheckpointError(f"checkpoint {tag!r} is not committed")
    if remap_to is not None:
        manifest = CheckpointManifest(**{**manifest.__dict__, "mesh": dict(remap_to)})
    return CheckpointState(
        manifest=manifest,
        tensors={p.name: p for p in manifest.parameters},
        optimizer=manifest.optimizer,
    )


def last_committed(*, root: str | os.PathLike = ".tessera_ckpt") -> Optional[str]:
    base = Path(root)
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if (p / "COMMITTED").exists() and (p / "manifest.json").exists()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p / "COMMITTED").stat().st_mtime, reverse=True)
    return candidates[0].name


def async_config() -> AsyncCheckpointConfig:
    return _ASYNC_CONFIG


def _tensor_shard(name: str, value: object, directory: Path) -> TensorShard:
    shape = tuple(int(v) for v in getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", type(value).__name__))
    safe_name = name.replace("/", "_").replace(".", "_")
    shard_path = directory / f"{safe_name}.json"
    shard_path.write_text(json.dumps({"name": name, "shape": shape, "dtype": dtype}))
    return TensorShard(name=name, shape=shape, dtype=dtype, path=shard_path.name)


def _remove_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            _remove_tree(child)
        else:
            child.unlink()
    path.rmdir()


__all__ = [
    "AsyncCheckpointConfig",
    "CheckpointError",
    "CheckpointManifest",
    "CheckpointState",
    "TensorShard",
    "async_config",
    "enable_async",
    "last_committed",
    "load",
    "save",
]
