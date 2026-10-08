"""Bounded retention of fully validated static NVFP4 program contracts."""
from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rocm_nvfp4_program import TracedNVFP4Program
import os
from threading import RLock

from .native_artifact import ArtifactContractError

_OWNER_PID = os.getpid()
_CACHE: OrderedDict[tuple[str, object], TracedNVFP4Program] = OrderedDict()
_LOCK = RLock()
_LIMIT = 24


def _freeze(value):
    # Type tags preserve list/tuple and bool/int distinctions. Float hex keeps
    # signed zero and exact binary values. Unknown Python objects are uncached.
    kind = type(value)
    if kind is dict:
        return ("dict", tuple(sorted((_freeze(k), _freeze(v)) for k, v in value.items())))
    if kind is list:
        return ("list", tuple(map(_freeze, value)))
    if kind is tuple:
        return ("tuple", tuple(map(_freeze, value)))
    if kind is float:
        return ("float", value.hex())
    if kind in (str, int, bool, type(None)):
        return (kind.__name__, value)
    raise TypeError("uncacheable program metadata type")


def _thaw(value):
    kind, data = value
    if kind == "dict":
        return {_thaw(k): _thaw(v) for k, v in data}
    if kind == "list":
        return list(map(_thaw, data))
    if kind == "tuple":
        return tuple(map(_thaw, data))
    if kind == "float":
        return float.fromhex(data)
    return data


def resolve_program(artifact):
    from .rocm_nvfp4_program import program_from_manifest

    metadata = artifact.metadata or {}
    if os.getpid() != _OWNER_PID:
        raise RuntimeError("native NVFP4 program retention cannot cross fork")
    fields = (metadata.get("native_program"), artifact.graph_ir, metadata.get("arg_names"))
    try:
        key = _freeze(fields)
    except TypeError:
        key = None
    if key is not None:
        with _LOCK:
            program = _CACHE.get(key)
            if program is not None:
                _CACHE.move_to_end(key)
                return program
        # Parse exactly the detached snapshot used for the cache key. Caller
        # mutation after freezing cannot alter the admitted compiled contract.
        fields = _thaw(key)
    manifest, graph, names = fields
    program = program_from_manifest(manifest)
    if graph != program.graph_ir or names != list(program.argument_names):
        raise ArtifactContractError(
            "E_LAUNCH_BINDING_MISMATCH", "resident program parent Graph/arguments differ")
    if key is not None:
        with _LOCK:
            _CACHE[key] = program
            _CACHE.move_to_end(key)
            while len(_CACHE) > _LIMIT:
                _CACHE.popitem(last=False)
    return program
