"""Public host ABI for the native ROCm source pullback and portable replay."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import json
import os
import threading
import weakref

from .native_rocm_source_reverse import RocmSourceReverseProgram, package_source_reverse

COMPILER_PATH = "rocm_source_reverse_compiled"
_NAMES = ["primal_0", "primal_1", "cotangent"]
_pid = os.getpid()
_lock = threading.RLock()
_live: weakref.WeakValueDictionary[tuple[int, str], PreparedRocmSourceReverse] = weakref.WeakValueDictionary()
_checked_packages: OrderedDict[str, RocmSourceReverseProgram] = OrderedDict()


def _guard_process():
    if os.getpid() != _pid:
        raise RuntimeError("ROCm source pullback cannot cross fork")


def _checked(data):
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    identity = hashlib.sha256(encoded.encode()).hexdigest()
    with _lock:
        package = _checked_packages.get(identity)
        if package is None:
            package = RocmSourceReverseProgram.from_json(encoded)
            if len(_checked_packages) >= 8:
                _checked_packages.popitem(last=False)
            _checked_packages[identity] = package
        _checked_packages.move_to_end(identity)
    return identity, package


class PreparedRocmSourceReverse:
    """Retain one checked native owner; warm calls only update/invoke/read."""
    def __init__(self, source, values, *, package=None):
        _guard_process()
        self.package = package_source_reverse(source) if package is None else package
        self.owner = self.package.prepare(values)
        self.pid = os.getpid()
        self.lock = threading.RLock()
        self.finalizer = weakref.finalize(self, self.owner.close)
        self.data = json.loads(self.package.to_json())
        self.identity = hashlib.sha256(json.dumps(
            self.data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.receipt = {
            "compiler_path": COMPILER_PATH, "execution_kind": "native_gpu",
            "execution_mode": "hip_runtime", "evidence_target": self.package.manifest["target"],
            "native_call_binding": "prepared_cpp_source_reverse_program",
            "source_graph_ir_digest": hashlib.sha256(
                self.package.manifest["source_graph_ir"].encode()).hexdigest(),
        }
        with _lock:
            _live[(self.pid, self.identity)] = self
            _checked_packages[self.identity] = self.package
            while len(_checked_packages) > 8:
                _checked_packages.popitem(last=False)

    def __call__(self, values):
        with self.lock:
            if os.getpid() != self.pid or not self.finalizer.alive:
                raise RuntimeError("ROCm source pullback owner is closed or crossed fork")
            self.package.validate_inputs(values)
            self.owner.update(values)
            generation, _ = self.owner.invoke()
            return self.owner.read(generation)

    def close(self):
        with self.lock:
            if os.getpid() != self.pid:
                raise RuntimeError("ROCm source pullback owner cannot cross fork")
            self.finalizer()


def runtime_artifact(owner):
    from tessera import runtime as rt
    if not owner.finalizer.alive or owner.pid != os.getpid():
        raise RuntimeError("ROCm source pullback owner is closed or crossed fork")
    cached = getattr(owner, "_public_runtime_artifact", None)
    if cached is None:
        plan = owner.package.manifest
        cached = rt.RuntimeArtifact(graph_ir=plan["source_graph_ir"], metadata={
            "target": plan["target"], "compiler_path": COMPILER_PATH,
            "execution_kind": "native_gpu", "runtime_status": "ready", "executable": True,
            "native_graph_verified": True, "arg_names": list(_NAMES),
            "native_program": json.loads(json.dumps(owner.data)),
        })
        owner._public_runtime_artifact = cached
    return cached


def execute(artifact, args):
    from tessera import runtime as rt
    _guard_process()
    metadata = artifact.metadata or {}
    if not isinstance(metadata.get("native_program"), dict):
        raise ValueError("ROCm source pullback requires its portable program")
    identity, package = _checked(metadata["native_program"])
    plan = package.manifest
    if (metadata.get("target") != plan["target"] or metadata.get("compiler_path") != COMPILER_PATH
            or metadata.get("arg_names") != _NAMES or artifact.graph_ir != plan["source_graph_ir"]
            or metadata.get("result_placement", "host") != "host"):
        raise ValueError("ROCm source pullback parent Graph/target/ABI differs")
    if isinstance(args, Mapping):
        if set(args) != set(_NAMES):
            raise ValueError("ROCm source pullback named root set differs")
        values = tuple(args[name] for name in _NAMES)
    elif isinstance(args, (tuple, list)) and len(args) == 3:
        values = tuple(args)
    else:
        raise ValueError("ROCm source pullback requires three original roots")
    package.validate_inputs(values)
    with _lock:
        owner = _live.get((os.getpid(), identity))
        borrowed = owner is not None and owner.finalizer.alive
        if not borrowed:
            owner = PreparedRocmSourceReverse(None, values, package=package)
        assert owner is not None
        try:
            outputs = owner(values)
            receipts = tuple(dict(
                ok=True, execution_kind="native_gpu", compiler_path=COMPILER_PATH,
                evidence_target=plan["target"], member=step["member"],
                image_digest=hashlib.sha256(image).hexdigest(),
                native_call_binding="prepared_cpp_source_reverse_program",
            ) for step, image in zip(plan["steps"], package.images, strict=True))
            return rt.NativeProgramExecution(outputs, receipts)
        finally:
            if not borrowed:
                owner.close()
