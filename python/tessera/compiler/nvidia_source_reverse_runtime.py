"""Serialized source pullback execution through the same checked native owner."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import copy
import hashlib
import json
import os
import threading
import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .prepared_nvidia_source_reverse import PreparedSourceReverse

from .portable_nvidia_source_reverse import COMPILER_PATH, restore

_pid = os.getpid()
_lock = threading.RLock()
_live: weakref.WeakValueDictionary[tuple[int, str], PreparedSourceReverse] = weakref.WeakValueDictionary()
_certificates: OrderedDict[str, tuple] = OrderedDict()
_CERTIFICATE_LIMIT = 8


def _checked(data):
    # Hash the entire current payload, not the caller's displayed digest.
    # On a miss validate an exact JSON snapshot, so later caller mutation
    # cannot change cached witnesses or member receipts.
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
    with _lock:
        checked = _certificates.get(fingerprint)
        if checked is None:
            snapshot = json.loads(encoded)
            native, members = restore(snapshot)
            checked = (native, native.manifest, members, snapshot["contract_digest"], snapshot)
            if len(_certificates) >= _CERTIFICATE_LIMIT:
                _certificates.popitem(last=False)
            _certificates[fingerprint] = checked
        _certificates.move_to_end(fingerprint)
        return checked



def register(owner):
    if os.getpid() != _pid:
        raise RuntimeError("source reverse replay cannot cross fork")
    with _lock:
        _live[(os.getpid(), owner.manifest()["contract_digest"])] = owner


def runtime_artifact(owner):
    from tessera import runtime as rt
    owner._guard()
    cached = getattr(owner, "_public_runtime_artifact", None)
    if cached is not None:
        return cached
    data = owner.manifest()
    plan = owner.native.manifest
    artifact = rt.RuntimeArtifact(graph_ir=plan["source_graph_ir"], metadata={
        "target": "nvidia_sm120", "compiler_path": COMPILER_PATH,
        "execution_kind": "native_gpu", "runtime_status": "ready", "executable": True,
        "native_graph_verified": True, "arg_names": ["primal_0", "primal_1", "cotangent"],
        "native_program": data})
    owner._public_runtime_artifact = artifact
    return artifact


def execute(artifact, args):
    from tessera import runtime as rt
    from .prepared_nvidia_source_reverse import PreparedSourceReverse
    if os.getpid() != _pid:
        raise RuntimeError("source reverse replay cannot cross fork")
    metadata = artifact.metadata or {}
    data = metadata.get("native_program")
    if not isinstance(data, dict):
        raise ValueError("source reverse requires a portable program manifest")
    # Validate every portable witness before loading a provider or borrowing a lease.
    native, plan, members, contract_digest, validated_snapshot = _checked(data)
    if (metadata.get("target") != "nvidia_sm120" or metadata.get("compiler_path") != COMPILER_PATH
            or metadata.get("arg_names") != ["primal_0", "primal_1", "cotangent"]
            or artifact.graph_ir != native.source_ir or metadata.get("result_placement", "host") != "host"):
        raise ValueError("source reverse parent Graph/target/ABI differs")
    if isinstance(args, Mapping):
        names = ["primal_0", "primal_1", "cotangent"]
        if set(args) != set(names):
            raise ValueError("source reverse named root set differs")
        values = tuple(args[name] for name in names)
    elif isinstance(args, (tuple, list)) and len(args) == 3:
        values = tuple(args)
    else:
        raise ValueError("source reverse requires three explicit roots")
    with _lock:
        owner = _live.get((os.getpid(), contract_digest))
        owned = owner is None or not owner.finalizer.alive
        if owned:
            owner = PreparedSourceReverse.from_manifest(validated_snapshot)
        assert owner is not None
        try:
            outputs = owner(values)
            receipts = tuple(dict(
                ok=True, execution_kind="native_gpu", compiler_path=COMPILER_PATH,
                evidence_target="nvidia_sm120", native_call_binding="prepared_cpp_source_reverse_program",
                input_binding=owner.receipt["input_binding"], root_uploads=owner.receipt["root_uploads"],
                member=step["member"], image_digest=record["image_digest"],
                descriptor_digest=record["member_digest"], abi=copy.deepcopy(record["abi"]),
            ) for step, record in zip(plan["steps"], members, strict=True))
            return rt.NativeProgramExecution(outputs, receipts)
        finally:
            if owned:
                owner.close()
