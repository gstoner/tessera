"""S14 AOT export and persistent compilation-cache references."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .runtime import RuntimeArtifact


def _json_hash(data: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass
class AOTArtifact:
    runtime_artifact: RuntimeArtifact
    metadata: dict[str, Any] = field(default_factory=dict)
    fn: Callable[..., Any] | None = None

    @property
    def artifact_hash(self) -> str:
        return self.runtime_artifact.artifact_hash

    def run(self, *inputs, **kwargs):
        if self.fn is None:
            raise RuntimeError("AOTArtifact has no Python reference callable; export with a picklable function for reference execution")
        return self.fn(*inputs, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "tessera.aot.v1",
            "runtime_artifact": self.runtime_artifact.to_dict(),
            "metadata": self.metadata,
        }

    def save(self, path: str | os.PathLike) -> Path:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "artifact.json").write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        if self.fn is not None:
            try:
                (root / "callable.pkl").write_bytes(pickle.dumps(self.fn))
            except Exception:
                pass
        return root


def export(fn: Callable[..., Any], *example_inputs: Any, path: str | os.PathLike | None = None, target: str = "cpu") -> AOTArtifact:
    """Export a callable or JIT wrapper to a self-describing AOT artifact."""
    runtime_artifact = _runtime_artifact_for(fn, target=target)
    input_meta = [
        {"shape": list(np.asarray(x).shape), "dtype": str(np.asarray(x).dtype)}
        for x in example_inputs
    ]
    metadata = {
        "name": getattr(fn, "__name__", type(fn).__name__),
        "target": target,
        "example_inputs": input_meta,
        "cache_key": compilation_cache_key(runtime_artifact, target=target, input_meta=input_meta),
    }
    artifact = AOTArtifact(runtime_artifact=runtime_artifact, metadata=metadata, fn=fn)
    if path is not None:
        artifact.save(path)
    return artifact


def load(path: str | os.PathLike, *, allow_pickle: bool = False) -> AOTArtifact:
    """Load an AOT artifact.

    Parameters
    ----------
    path
        Directory containing ``artifact.json`` (and optionally a
        ``callable.pkl`` sidecar with a pickled callable).
    allow_pickle
        **Default: False.**  When ``True``, also load the
        ``callable.pkl`` sidecar via :func:`pickle.loads`.

        .. warning::

           ``pickle.loads`` will execute arbitrary code embedded in
           a maliciously-crafted artifact.  Tessera defaults to
           ``allow_pickle=False`` so an artifact received from an
           untrusted source is safe to inspect — the IR, metadata,
           and ABI signature load via JSON only.  Set
           ``allow_pickle=True`` only when the artifact came from
           a trusted source (e.g., your own build pipeline).

           The long-term plan is to move ``callable`` persistence
           to a declarative format that can be deserialized without
           code execution; until then ``allow_pickle`` is the
           opt-in gate.
    """
    root = Path(path)
    data = json.loads((root / "artifact.json").read_text())
    rt_data = data["runtime_artifact"]
    runtime_artifact = RuntimeArtifact(
        graph_ir=rt_data.get("graph_ir", ""),
        schedule_ir=rt_data.get("schedule_ir", ""),
        tile_ir=rt_data.get("tile_ir", ""),
        target_ir=rt_data.get("target_ir", ""),
        metadata=rt_data.get("metadata") or {},
        abi_signature=rt_data.get("abi_signature", ""),
    )
    fn = None
    callable_path = root / "callable.pkl"
    if callable_path.exists() and allow_pickle:
        try:
            fn = pickle.loads(callable_path.read_bytes())
        except Exception:
            fn = None
    return AOTArtifact(runtime_artifact=runtime_artifact, metadata=data.get("metadata") or {}, fn=fn)


def stablehlo_export(artifact_or_fn: AOTArtifact | Callable[..., Any], *example_inputs: Any) -> str:
    artifact = artifact_or_fn if isinstance(artifact_or_fn, AOTArtifact) else export(artifact_or_fn, *example_inputs)
    name = artifact.metadata.get("name", "main")
    return "\n".join([
        f"module @tessera_{name} {{",
        f"  // stablehlo reference export; artifact_hash={artifact.artifact_hash}",
        "  func.func @main() { return }",
        "}",
    ])


def gguf_export(artifact_or_fn: AOTArtifact | Callable[..., Any], path: str | os.PathLike, *example_inputs: Any) -> Path:
    artifact = artifact_or_fn if isinstance(artifact_or_fn, AOTArtifact) else export(artifact_or_fn, *example_inputs)
    target = Path(path)
    target.write_text(json.dumps({"format": "gguf.reference.json", "artifact": artifact.to_dict()}, indent=2, sort_keys=True))
    return target


def safetensors_export(state: Mapping[str, Any], path: str | os.PathLike) -> Path:
    """Write a safetensors-like numpy container plus JSON metadata."""
    target = Path(path)
    arrays = {name: np.asarray(value) for name, value in state.items()}
    with target.open("wb") as f:
        np.savez(f, **arrays)
    (target.with_suffix(target.suffix + ".json")).write_text(
        json.dumps(
            {
                "format": "safetensors.reference.npz",
                "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return target


def compilation_cache_key(runtime_artifact: RuntimeArtifact, *, target: str, input_meta: Any = None, dtype_policy: str = "default", mesh_spec: Any = None) -> str:
    return _json_hash(
        {
            "graph_ir": runtime_artifact.graph_ir,
            "target": target,
            "dtype_policy": dtype_policy,
            "mesh_spec": mesh_spec,
            "input_meta": input_meta,
            "tessera_version": "reference",
        }
    )


class CompilationCache:
    """Tiny persistent artifact cache keyed by compiler-visible metadata."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, artifact: AOTArtifact) -> Path:
        return artifact.save(self.root / key)

    def get(self, key: str) -> AOTArtifact | None:
        path = self.root / key
        if not (path / "artifact.json").exists():
            return None
        return load(path)

    def invalidate(self, key: str) -> None:
        path = self.root / key
        if path.exists():
            _remove_tree(path)


def compilation_cache(root: str | os.PathLike) -> CompilationCache:
    return CompilationCache(root)


def _runtime_artifact_for(fn: Callable[..., Any], *, target: str) -> RuntimeArtifact:
    if hasattr(fn, "runtime_artifact"):
        try:
            return fn.runtime_artifact()  # type: ignore[misc]
        except TypeError:
            pass
    name = getattr(fn, "__name__", type(fn).__name__)
    graph_ir = f'tessera.aot.reference @{name} target="{target}"'
    return RuntimeArtifact(graph_ir=graph_ir, metadata={"name": name, "target": target}, abi_signature=f"{name}(...)")


def _remove_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            _remove_tree(child)
        else:
            child.unlink()
    path.rmdir()


__all__ = [
    "AOTArtifact",
    "CompilationCache",
    "compilation_cache",
    "compilation_cache_key",
    "export",
    "gguf_export",
    "load",
    "safetensors_export",
    "stablehlo_export",
]
