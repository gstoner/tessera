"""Portable source-attention VJP execution through the retained native owner."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import copy
import json
import os
import threading
import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .prepared_source_attention_reverse import PreparedSourceAttentionReverse

from .portable_source_attention_reverse import COMPILER_PATH, digest, restore

_pid = os.getpid()
_lock = threading.RLock()
_live: weakref.WeakValueDictionary[tuple[int, str], PreparedSourceAttentionReverse] = weakref.WeakValueDictionary()
_certificates: OrderedDict[str, tuple] = OrderedDict()
ARGUMENTS = ("primal_0", "primal_1", "primal_2", "cotangent_0", "cotangent_1")


def runtime_artifact(owner):
    from tessera.runtime import RuntimeArtifact
    owner._guard()
    if os.getpid() != _pid:
        raise ValueError("source attention reverse runtime cannot cross fork")
    data = owner.manifest()
    with _lock:
        _live[(_pid, data["contract_digest"])] = owner
    return RuntimeArtifact(graph_ir=data["plan"]["source_graph_ir"], metadata={
        "target": "nvidia_sm120", "compiler_path": COMPILER_PATH,
        "execution_kind": "native_gpu", "runtime_status": "ready", "executable": True,
        "native_graph_verified": True, "arg_names": list(ARGUMENTS), "native_program": data})


def _checked(data):
    encoded = json.dumps(data, sort_keys=True, allow_nan=False)
    identity = digest(data)
    with _lock:
        checked = _certificates.get(identity)
        if checked is None:
            checked = restore(json.loads(encoded))
            if len(_certificates) >= 8:
                _certificates.popitem(last=False)
            _certificates[identity] = checked
        _certificates.move_to_end(identity)
        return checked


def execute(artifact, args):
    from tessera.runtime import NativeProgramExecution
    from .prepared_source_attention_reverse import PreparedSourceAttentionReverse
    if os.getpid() != _pid:
        raise ValueError("source attention reverse runtime cannot cross fork")
    metadata = artifact.metadata or {}
    data, plan, _, _ = _checked(metadata.get("native_program"))
    if (metadata.get("target") != "nvidia_sm120" or metadata.get("compiler_path") != COMPILER_PATH
            or metadata.get("arg_names") != list(ARGUMENTS)
            or metadata.get("result_placement", "host") != "host"
            or artifact.graph_ir != plan["source_graph_ir"]):
        raise ValueError("source attention reverse parent Graph/target/ABI differs")
    if isinstance(args, Mapping):
        if set(args) != set(ARGUMENTS):
            raise ValueError("source attention reverse named root set differs")
        values = tuple(args[name] for name in ARGUMENTS)
    elif isinstance(args, (tuple, list)) and len(args) == 5:
        values = tuple(args)
    else:
        raise ValueError("source attention reverse requires five explicit roots")
    with _lock:
        owner = _live.get((_pid, data["contract_digest"]))
        owned = owner is None or owner._closed
        if owned:
            owner = PreparedSourceAttentionReverse.from_manifest(data)
        assert owner is not None
        try:
            outputs = owner(*values)
            receipt = owner.receipt
            if receipt is None:
                raise RuntimeError("source attention reverse completed without a native receipt")
            receipts = tuple(dict(
                ok=True, **receipt, compiler_path=COMPILER_PATH,
                member=step["member"], image_digest=record["image_digest"],
                descriptor_digest=record["member_digest"], abi=copy.deepcopy(record["abi"]))
                for step, record in zip(plan["steps"], data["members"], strict=True))
            return NativeProgramExecution(outputs, receipts)
        finally:
            if owned:
                owner.close()
