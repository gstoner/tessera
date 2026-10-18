"""Portable native source-attention replay; compiler-free after packaging."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import json
import os
import threading
import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .prepared_source_attention import PreparedSourceAttention

from .portable_source_attention import COMPILER_PATH, restore

_pid = os.getpid()
_lock = threading.RLock()
_live: weakref.WeakValueDictionary[tuple[int, str], PreparedSourceAttention] = weakref.WeakValueDictionary()
_certificates: OrderedDict[str, tuple] = OrderedDict()


def _checked(data):
    encoded = json.dumps(data, sort_keys=True, allow_nan=False)
    fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
    with _lock:
        checked = _certificates.get(fingerprint)
        if checked is None:
            checked = restore(json.loads(encoded))
            if len(_certificates) >= 8:
                _certificates.popitem(last=False)
            _certificates[fingerprint] = checked
        _certificates.move_to_end(fingerprint)
        return checked


def runtime_artifact(owner):
    from tessera.runtime import RuntimeArtifact
    if os.getpid() != _pid or owner.pid != _pid:
        raise ValueError("source attention owner belongs to another process")
    data = owner.manifest()
    with _lock:
        if not owner.closed:
            _live[(_pid, data["contract_digest"])] = owner
    return RuntimeArtifact(graph_ir=data["plan"]["source_graph_ir"], metadata={
        "target": "nvidia_sm120", "compiler_path": COMPILER_PATH,
        "execution_kind": "native_gpu", "runtime_status": "ready", "executable": True,
        "native_graph_verified": True, "arg_names": list(owner.argument_names),
        "native_program": data})


def execute(artifact, args):
    from tessera.runtime import NativeProgramExecution
    from .prepared_source_attention import PreparedSourceAttention
    if os.getpid() != _pid:
        raise ValueError("source attention runtime cannot cross fork")
    metadata = artifact.metadata or {}
    data, plan, packages, _ = _checked(metadata.get("native_program"))
    if (metadata.get("target") != "nvidia_sm120" or metadata.get("compiler_path") != COMPILER_PATH
            or metadata.get("arg_names") != data["argument_names"]
            or artifact.graph_ir != plan["source_graph_ir"]
            or metadata.get("result_placement", "host") != "host"):
        raise ValueError("source attention parent Graph/target/arguments differ")
    if isinstance(args, Mapping):
        if set(args) != set(data["argument_names"]):
            raise ValueError("source attention named root set differs")
        roots = tuple(args[name] for name in data["argument_names"])
    elif isinstance(args, (tuple, list)) and len(args) == 3:
        roots = tuple(args)
    else:
        raise ValueError("source attention requires three original roots")
    with _lock:
        owner = _live.get((_pid, data["contract_digest"]))
        owned = owner is None or owner.closed
        if owned:
            owner = PreparedSourceAttention.from_manifest(data)
        assert owner is not None
        try:
            outputs, receipt = owner(*roots)
            components = tuple({
                **receipt, "compiler_path": COMPILER_PATH,
                "image_digest": package.image.image_digest,
                "descriptor_digest": package.descriptor.descriptor_digest,
                "entry": package.descriptor.entry_symbol, "abi_id": package.descriptor.abi_id,
                "member": step["member"],
            } for package, step in zip(packages, plan["steps"], strict=True))
            return NativeProgramExecution(outputs, components)
        finally:
            if owned:
                owner.close()
