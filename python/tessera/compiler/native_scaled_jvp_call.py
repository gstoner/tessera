"""Owned execution of a sealed native scaled JVP program.

This binding prepares an existing compiler artifact once. Numerical operations,
device storage, launch geometry and generations belong to the checked C++ ABI.
"""
from __future__ import annotations

import copy
import os
import threading
import weakref
from types import MappingProxyType
from typing import Any

from .native_jvp import NativeJVPArtifact, child_digest
from .native_scaled_program import NativeScaledProgram, PreparedScaledProgram


class NativeScaledJVPCall:
    """One caller-owned checked native program and independent output generations."""

    def __init__(self, artifact: NativeJVPArtifact, inputs: Any):
        from tessera import runtime

        metadata = artifact.runtime_metadata()
        contract = copy.deepcopy(metadata["native_jvp"])
        NativeJVPArtifact(contract).validate()
        if (contract.get("target"), contract.get("architecture"), contract.get("family")) != (
                "rocm", "gfx1201", "scaled_product_program"):
            raise ValueError("retained scaled JVP requires its exact gfx1201 program")
        for name in ("artifact_hash", "paired_jvp_ir_digest", "source_graph_ir_digest"):
            value = contract.get(name)
            if (not isinstance(value, str) or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)):
                raise ValueError("retained scaled JVP lacks sealed compiler lineage")
        steps = contract["steps"]
        if len(steps) != 1:
            raise ValueError("retained scaled JVP requires one compiled SSA program")
        step = steps[0]
        child = step.get("child_metadata")
        names = contract.get("arg_names")
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names) or len(inputs) != len(names)
                or not isinstance(child, dict) or child_digest(child) != step["child_digest"]
                or step.get("depends_on") or step.get("outputs") != ["primal", "tangent"]
                or step.get("inputs") != names or child.get("arg_names") != names
                or child.get("target") != "rocm"
                or child.get("compiler_path") != "rocm_scaled_jvp_program_compiled"
                or child.get("execution_kind") != "native_gpu"
                or child.get("execution_mode") != "hip_runtime"
                or child.get("executable") is not True):
            raise ValueError("retained scaled JVP child binding differs from its sealed product")
        row = runtime._exec_row_for_metadata(child)
        if row is None or not row.executable or row.executor_id != "rocm_scaled_jvp_program_compiled":
            raise ValueError("retained scaled JVP has no executable native consumer")
        package = NativeScaledProgram.from_manifest(child.get("native_scaled_program"))
        library = runtime._load_rocm_native_movement_runtime()
        if library is None:
            raise RuntimeError("retained scaled JVP requires its native HIP provider")
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._owner = PreparedScaledProgram(package, inputs, runtime_library=library._name)
        # The callback owns only the native function and numeric handle, not
        # this Python object. Native close retains PID/context/completion guards.
        self._finalizer = weakref.finalize(
            self, self._owner.lib.tessera_rocm_program_close, self._owner.handle.value)
        self._closed = False
        self.receipt = MappingProxyType({
            "compiler_path": "rocm_jvp_compiled", "execution_kind": "native_gpu",
            "execution_mode": "hip_runtime", "evidence_target": "rocm_gfx1201",
            "artifact_hash": contract["artifact_hash"],
            "paired_jvp_ir_digest": contract["paired_jvp_ir_digest"],
            "source_graph_ir_digest": contract["source_graph_ir_digest"],
            "schedule_program_digest": contract["schedule_program"]["digest"],
            "tile_program_digest": contract["tile_program"]["digest"],
            "frontend_authority": "tracer", "family": "scaled_product_program",
            "host_preparation": "native_checked_view_pack",
            "retained_native_owner": True,
        })

    def check_process(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError("retained native JVP cannot cross fork")

    def invoke(self, inputs: Any):
        self.check_process()
        # ctypes releases the GIL. Keep update/dispatch/read one transaction
        # for this private owner, including competing calls and close.
        with self._lock:
            if self._closed:
                raise RuntimeError("retained native JVP owner is closed")
            self._owner.update(inputs)
            generation, _ = self._owner.invoke()
            outputs = self._owner.read(generation)
            if len(outputs) != 2:
                raise RuntimeError("retained native JVP lost its paired outputs")
            return outputs

    def close(self) -> None:
        self.check_process()
        with self._lock:
            if not self._closed:
                self._owner.close()
                self._finalizer.detach()
                self._closed = True

