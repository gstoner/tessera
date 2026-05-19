"""Runtime checkpoint manifest helpers for Tessera."""

from __future__ import annotations

import json
import os
import pickle
import hashlib
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, cast

import numpy as np

from .state import tree_flatten, tree_unflatten


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
        # Cast helpers — JSON deserialization produces ``object`` and
        # mypy can't narrow that to iterables / mappings without the
        # explicit cast.  Runtime validation still happens through
        # the ``str`` / ``int`` / ``dict`` constructors below.
        shape_raw = cast(Iterable[Any], data.get("shape", ()))
        sharding_raw = cast(Mapping[str, Any], data.get("sharding", {}))
        return cls(
            name=str(data["name"]),
            shape=tuple(int(v) for v in shape_raw),
            dtype=str(data.get("dtype", "unknown")),
            sharding=dict(sharding_raw),
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
        # Same Mapping[str, object] → narrower-types narrowing pattern
        # as ``TensorShard.from_dict`` above.
        version_raw = cast(Any, data.get("version", 1))
        step_raw = cast(Optional[int], data.get("step"))
        mesh_raw = cast(Mapping[str, Any], data.get("mesh", {}))
        rng_raw = cast(Mapping[str, Any], data.get("rng", {}))
        params_raw = cast(Iterable[Mapping[str, Any]], data.get("parameters", ()))
        optimizer_raw = cast(Mapping[str, Any], data.get("optimizer", {}))
        autotune_raw = cast(Mapping[str, Any], data.get("autotune_cache", {}))
        return cls(
            version=int(version_raw),
            tag=str(data["tag"]),
            step=step_raw,
            backend=str(data.get("backend", "python")),
            mesh=dict(mesh_raw),
            numerics=str(data.get("numerics", "fast")),
            rng=dict(rng_raw),
            reduce_tree_id=str(data.get("reduce_tree_id", "")),
            parameters=tuple(TensorShard.from_dict(p) for p in params_raw),
            optimizer=dict(optimizer_raw),
            autotune_cache=dict(autotune_raw),
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
_STATE_FORMAT_VERSION = 1
_STATE_MIGRATIONS: dict[tuple[int, int], Callable[[Any], Any]] = {}


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


def save_state(
    tree: Any,
    path: str | os.PathLike,
    *,
    version: int = _STATE_FORMAT_VERSION,
    atomic: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save a Tessera state tree to a typed, self-describing binary file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp") if atomic else target
    leaves, treedef = tree_flatten(tree)
    arrays = {f"leaf_{i}": _to_serializable_array(leaf) for i, leaf in enumerate(leaves)}
    leaf_meta = [
        {
            "name": f"leaf_{i}",
            "shape": list(arrays[f"leaf_{i}"].shape),
            "dtype": str(arrays[f"leaf_{i}"].dtype),
            "sha256": _sha256_array(arrays[f"leaf_{i}"]),
        }
        for i in range(len(leaves))
    ]
    manifest = {
        "format": "tessera.state.v1",
        "version": int(version),
        "leaf_count": len(leaves),
        "leaves": leaf_meta,
        "metadata": dict(metadata or {}),
    }
    payload = {
        "__manifest__": np.frombuffer(json.dumps(manifest, sort_keys=True).encode("utf-8"), dtype=np.uint8),
        "__treedef__": np.frombuffer(pickle.dumps(treedef), dtype=np.uint8),
        **arrays,
    }
    with tmp.open("wb") as f:
        # numpy's typeshed stubs declare ``savez(file, *args, **kwds)``
        # with ``**kwds: bool`` because of the legacy ``compress=``
        # keyword; in practice every other ``**kwds`` entry is an
        # ndarray (the intended use).  Cast to ``Any`` so mypy
        # accepts the polymorphic call.
        np.savez(f, **payload)  # type: ignore[arg-type]
    if atomic:
        os.replace(tmp, target)
    return target


def load_state(
    path: str | os.PathLike,
    *,
    target_version: int | None = None,
    collections: tuple[str, ...] | list[str] | set[str] | None = None,
    trust_treedef: bool = True,
) -> Any:
    """Load a Tessera state tree saved by :func:`save_state`.

    .. warning::

       The ``__treedef__`` blob in a state file is a pickled Python
       object — loading it via :func:`pickle.loads` will execute
       arbitrary code embedded in a malicious checkpoint.  Tessera
       defaults to ``trust_treedef=True`` for ergonomic
       same-process save/load, but you should pass
       ``trust_treedef=False`` whenever the source of the file is
       not under your control (downloaded checkpoints, shared
       artifacts, multi-tenant storage, etc.).  When
       ``trust_treedef=False`` the function returns a checksum-
       verified leaf bundle and refuses to materialize the treedef.

       Long-term, treedef persistence should move to a declarative
       format that can be deserialized without code execution;
       until then, ``trust_treedef`` is the explicit opt-out.

    Parameters
    ----------
    path, target_version, collections
        Standard load-state knobs.
    trust_treedef
        **Default: True** (preserves the existing save/load
        ergonomics in same-process pipelines).  Set to ``False``
        on any untrusted source.
    """
    source = Path(path)
    if not source.exists():
        raise CheckpointError(f"state file not found: {source}")
    with np.load(source, allow_pickle=False) as data:
        manifest = json.loads(bytes(data["__manifest__"].tolist()).decode("utf-8"))
        if not trust_treedef:
            raise CheckpointError(
                "load_state(trust_treedef=False): refusing to "
                "deserialize the pickled treedef from an untrusted "
                "source.  Returning the raw checksum-verified leaves "
                "is not yet supported; track a declarative treedef "
                "format in the milestone plan.  Pass "
                "trust_treedef=True only for checkpoints from a "
                "trusted source (e.g., your own training pipeline)."
            )
        treedef = pickle.loads(bytes(data["__treedef__"].tolist()))
        leaves = []
        for meta in manifest["leaves"]:
            arr = np.array(data[meta["name"]])
            if _sha256_array(arr) != meta["sha256"]:
                raise CheckpointError(f"checksum mismatch for {meta['name']}")
            leaves.append(arr)
    tree = tree_unflatten(treedef, leaves)
    if collections is not None:
        wanted = set(collections)
        if not isinstance(tree, dict):
            raise CheckpointError("partial collection loading requires a top-level dict")
        tree = {k: v for k, v in tree.items() if k in wanted}
    version = int(manifest.get("version", _STATE_FORMAT_VERSION))
    if target_version is not None:
        tree = _apply_migrations(tree, version, int(target_version))
    return tree


def save_sharded(
    tree: Any,
    path: str | os.PathLike,
    mesh: Any,
    *,
    version: int = _STATE_FORMAT_VERSION,
) -> Path:
    """Save a state tree in a directory with mesh metadata and one payload shard."""
    root = Path(path)
    tmp = root.with_name(f".{root.name}.tmp")
    if tmp.exists():
        _remove_tree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    shard_path = tmp / "shard_00000.tessera_state.npz"
    save_state(tree, shard_path, version=version, atomic=False)
    manifest = {
        "format": "tessera.sharded_state.v1",
        "version": int(version),
        "mesh": _mesh_metadata(mesh),
        "shards": [{"rank": 0, "path": shard_path.name}],
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (tmp / "COMMITTED").write_text(str(time.time()))
    if root.exists():
        _remove_tree(root)
    os.replace(tmp, root)
    return root


def load_sharded(path: str | os.PathLike, mesh: Any | None = None) -> Any:
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists() or not (root / "COMMITTED").exists():
        raise CheckpointError(f"sharded checkpoint is not committed: {root}")
    manifest = json.loads(manifest_path.read_text())
    if mesh is not None and manifest.get("mesh", {}).get("size") not in (None, _mesh_metadata(mesh).get("size")):
        raise CheckpointError("mesh size mismatch while loading sharded checkpoint")
    return load_state(root / manifest["shards"][0]["path"])


def register_state_migration(from_version: int, to_version: int, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    if to_version != from_version + 1:
        raise ValueError("state migrations must advance by exactly one version")
    _STATE_MIGRATIONS[(int(from_version), int(to_version))] = fn
    return fn


def state_migration(from_version: int, to_version: int):
    """Decorator form of :func:`register_state_migration`."""
    def decorate(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        register_state_migration(from_version, to_version, fn)
        return fn
    return decorate


def _apply_migrations(tree: Any, version: int, target_version: int) -> Any:
    if target_version < version:
        raise CheckpointError("downgrade migrations are not supported")
    out = tree
    cur = version
    while cur < target_version:
        key = (cur, cur + 1)
        if key not in _STATE_MIGRATIONS:
            raise CheckpointError(f"missing state migration {cur}->{cur + 1}")
        out = _STATE_MIGRATIONS[key](out)
        cur += 1
    return out


def _tensor_shard(name: str, value: object, directory: Path) -> TensorShard:
    shape = tuple(int(v) for v in getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", type(value).__name__))
    safe_name = name.replace("/", "_").replace(".", "_")
    shard_path = directory / f"{safe_name}.json"
    shard_path.write_text(json.dumps({"name": name, "shape": shape, "dtype": dtype}))
    return TensorShard(name=name, shape=shape, dtype=dtype, path=shard_path.name)


def _to_serializable_array(value: Any) -> np.ndarray:
    if hasattr(value, "_data"):
        value = value._data
    if hasattr(value, "_data"):
        value = value._data
    arr = np.asarray(value)
    if arr.dtype == object:
        raise CheckpointError("object dtype leaves are not supported by save_state")
    return arr


def _sha256_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(json.dumps(list(arr.shape)).encode("utf-8"))
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _mesh_metadata(mesh: Any) -> dict[str, Any]:
    if mesh is None:
        return {}
    if hasattr(mesh, "axis_names") and hasattr(mesh, "shape"):
        shape = mesh.shape
        return {
            "axis_names": list(mesh.axis_names),
            "shape": dict(shape) if isinstance(shape, Mapping) else list(shape),
            "size": int(getattr(mesh, "size", 1)),
        }
    if isinstance(mesh, Mapping):
        return {"shape": dict(mesh), "size": int(np.prod(list(mesh.values()))) if mesh else 1}
    return {"repr": repr(mesh)}


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
    "load_sharded",
    "load_state",
    "register_state_migration",
    "save",
    "save_sharded",
    "save_state",
    "state_migration",
]


class _CallableCheckpointModule(types.ModuleType):
    """Let ``tessera.checkpoint`` double as an activation-checkpoint decorator.

    The module still exposes runtime checkpoint helpers such as
    ``save_state``/``load_state``. Calling the module delegates to
    ``tessera.autodiff.checkpoint`` for compatibility with older examples that
    used ``@tessera.checkpoint`` for rematerialization.
    """

    def __call__(self, fn=None):
        from .autodiff import checkpoint as activation_checkpoint

        if fn is None:
            return activation_checkpoint
        return activation_checkpoint(fn)


sys.modules[__name__].__class__ = _CallableCheckpointModule
