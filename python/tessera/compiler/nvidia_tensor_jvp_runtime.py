"""Public artifact binding for the compiler-owned static paired tensor owner."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import copy
import os
import threading
import weakref
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .prepared_nvidia_tensor_jvp import PreparedTensorJVP

from .portable_nvidia_tensor_jvp import digest, restore
from .portable_nvidia_active_source import SCHEMA as ACTIVE_SCHEMA, COMPILER_PATH as ACTIVE_PATH, restore as restore_active

_lock = threading.RLock()
_live: weakref.WeakValueDictionary[tuple[int, str], PreparedTensorJVP] = weakref.WeakValueDictionary()
_certificates: OrderedDict[str, dict[str, Any]] = OrderedDict()
_CERTIFICATE_LIMIT = 8


def _validated_plan(data):
    # Hash the entire typed JSON payload, including its declared integrity key.
    # Reusing a display digest never bypasses admission after a metadata edit.
    signature = digest(data)
    with _lock:
        plan = _certificates.get(signature)
        if plan is not None:
            _certificates.move_to_end(signature)
            return plan
        native, _, _ = (restore_active(data) if data.get("schema") == ACTIVE_SCHEMA else restore(data))
        plan = native.manifest
        if len(_certificates) >= _CERTIFICATE_LIMIT:
            _certificates.popitem(last=False)
        _certificates[signature] = plan
        return plan


def register(owner):
    """Reuse a live JIT lease when public artifact replay targets its images."""
    manifest = owner.manifest()
    with _lock:
        _live[(os.getpid(), manifest["contract_digest"])] = owner


def runtime_artifact(owner):
    from tessera import runtime as rt

    owner._guard()
    cached = getattr(owner, "_public_runtime_artifact", None)
    if cached is not None:
        return cached
    manifest = owner.manifest()
    plan = owner.plan
    active = manifest["schema"] == ACTIVE_SCHEMA
    path = ACTIVE_PATH if active else "nvidia_sm120_tensor_jvp_compiled"
    names = ["primal_0", "primal_1", f"tangent_{plan['role_indices'][0 if active else 1]}"]
    artifact = rt.RuntimeArtifact(
        graph_ir=plan["source_graph_ir"],
        metadata={
            "target": "nvidia_sm120",
            "compiler_path": path,
            "execution_kind": "native_gpu",
            "runtime_status": "ready",
            "executable": True,
            "native_graph_verified": True,
            "arg_names": names,
            "native_program": manifest,
        },
    )
    owner._public_runtime_artifact = artifact
    owner.receipt["artifact_hash"] = artifact.artifact_hash
    return artifact


def execute(artifact, args):
    from tessera import runtime as rt
    from .prepared_nvidia_tensor_jvp import PreparedTensorJVP
    from .prepared_nvidia_active_source import PreparedActiveSourceJVP

    metadata = artifact.metadata or {}
    if metadata.get("target") != "nvidia_sm120":
        raise ValueError("paired tensor artifact requires exact nvidia_sm120")
    placement = metadata.get("result_placement", "host")
    if type(placement) is not str or placement not in {"host", "cuda"}:
        raise ValueError("paired tensor result placement differs")
    data = metadata.get("native_program")
    if not isinstance(data, dict):
        raise ValueError("paired tensor artifact requires a program manifest")
    plan = _validated_plan(data)  # Validate all witnesses before provider access.
    active = data["schema"] == ACTIVE_SCHEMA
    path = ACTIVE_PATH if active else "nvidia_sm120_tensor_jvp_compiled"
    if metadata.get("compiler_path") != path:
        raise ValueError("paired tensor parent compiler path differs")
    names = ["primal_0", "primal_1", f"tangent_{plan['role_indices'][0 if active else 1]}"]
    if artifact.graph_ir != plan["source_graph_ir"] or metadata.get("arg_names") != names:
        raise ValueError("paired tensor parent Graph/argument ABI differs")
    if isinstance(args, Mapping):
        if set(args) != set(names):
            raise ValueError("paired tensor named input set differs")
        values = tuple(args[name] for name in names)
    elif isinstance(args, (tuple, list)) and len(args) == 3:
        values = tuple(args)
    else:
        raise ValueError("paired tensor artifact requires three explicit roots")
    # Borrow a live matching JIT owner rather than preparing a second lease
    # that would invalidate it. A standalone replay owns and retires its owner.
    with _lock:
        owner = _live.get((os.getpid(), data["contract_digest"]))
        if owner is None or not owner._finalizer.alive:
            owner = (PreparedActiveSourceJVP if active else PreparedTensorJVP).from_manifest(copy.deepcopy(data))
            owned = True
        else:
            owned = False
        assert owner is not None
        try:
            if placement == "cuda":
                from .native_tensor_result import execute as resident_result
                outputs = resident_result(owner, values)
            else:
                outputs = owner(values)
                owner.receipt["result_placement"] = "host"
                owner.receipt.pop("result_ownership", None)
            if active:
                receipts = tuple({
                    "ok": True, "execution_kind": "native_gpu", "compiler_path": path,
                    "evidence_target": "nvidia_sm120",
                    "native_call_binding": "prepared_cpp_active_source_program",
                    "input_binding": owner.receipt["input_binding"],
                    "root_uploads": owner.receipt["root_uploads"],
                    "result_placement": placement,
                    "member": step["member"], "image_digest": record["image_digest"],
                    "descriptor_digest": record["member_digest"], "abi": copy.deepcopy(record["abi"]),
                } for step, record in zip(plan["steps"], data["members"], strict=True))
            else:
                assert isinstance(owner, PreparedTensorJVP)
                receipts = tuple({
                    "ok": True, "execution_kind": "native_gpu",
                    "compiler_path": "nvidia_sm120_tensor_jvp_compiled",
                    "evidence_target": "nvidia_sm120",
                    "native_call_binding": "prepared_cpp_tensor_program",
                    "result_placement": placement,
                    "member": step["member"],
                    "image_digest": package.image.image_digest,
                    "descriptor_digest": package.descriptor.descriptor_digest,
                } for step, package in zip(plan["steps"], owner.members, strict=True))
            return rt.NativeProgramExecution(outputs, receipts)
        finally:
            if owned:
                owner.close()
