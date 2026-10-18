"""Retained native owner for the compiler's original-source attention VJP."""
from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
import hashlib
import json
import os
import re
import threading
import time
from types import SimpleNamespace
from typing import Any
import weakref

import numpy as np


@dataclass(frozen=True)
class ReverseMember:
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target_ir: str
    native_ir: str
    entry: str
    image: bytes
    image_digest: str


def _retire(lib, handle, pid):
    if os.getpid() == pid:
        lib.tessera_nvidia_source_attention_reverse_close(handle)


class PreparedSourceAttentionReverse:
    """Compile actual AD members once; execute its typed members in one C++ owner."""

    def __init__(self, program):
        from .nvidia_native import _compile_tile_ir
        from .scheduled_checkpoint import _decode_checkpoint
        program.validate()
        plan = program.manifest
        stages = program.lower_members()
        forward = _decode_checkpoint(*stages[1][:3], backward=False)
        if plan["schema"] == "tessera.native.source_attention_reverse.v2":
            if forward.output_storage != forward.storage or forward.bias or forward.shape_bounds:
                raise ValueError("source attention reverse member contracts disagree")
        else:
            backward = _decode_checkpoint(*stages[2][:3], backward=True)
            if (forward.dims != backward.dims or forward.scale != backward.scale
                    or forward.causal != backward.causal or forward.storage != backward.storage
                    or forward.output_storage != forward.storage
                    or backward.output_storage != forward.storage or forward.bias or backward.bias
                    or not backward.lse_cotangent or forward.shape_bounds or backward.shape_bounds):
                raise ValueError("source attention reverse member contracts disagree")
        members = []
        for graph, schedule, tile, target, native in stages:
            entries = re.findall(r'\bllvm.func @(\w+)\(', tile)
            if len(entries) != 1:
                raise ValueError("source attention reverse requires one entry per member")
            lowered, ptx, *_ = _compile_tile_ir(tile, entries[0])
            image = ptx.encode()
            members.append(ReverseMember(graph, schedule, tile, target, lowered, entries[0],
                                         image, hashlib.sha256(image).hexdigest()))
        program.validate()
        from .portable_source_attention_reverse import restore, seal
        self._initialize(*restore(seal(plan, tuple(members))))

    @classmethod
    def from_manifest(cls, data):
        from .portable_source_attention_reverse import restore
        validated = restore(data)
        owner = cls.__new__(cls)
        owner._initialize(*validated)
        return owner

    def _initialize(self, data, plan, members, dims):
        self._members = members
        self._rows = plan["schema"] == "tessera.native.source_attention_reverse.v2"
        self._stage_count = len(members)
        self._api = ("tessera_nvidia_source_attention_reverse_rows_" if self._rows
                     else "tessera_nvidia_source_attention_reverse_")
        self._portable_text = json.dumps(data, sort_keys=True, allow_nan=False)
        self._contract_digest = data["contract_digest"]
        self._source_graph_digest = plan["source_graph_sha256"]
        self._plan_text = json.dumps(plan, sort_keys=True, allow_nan=False)
        self._dims = dims
        self._roles = tuple(plan["role_indices"])
        self._seeds = tuple(plan["cotangent_indices"])
        self._selected = tuple(self._roles.index(i) for i in plan["requested_gradients"])
        self._input_specs = tuple((tuple(b["shape"]), b["storage"], b["bytes"])
                                  for b in plan["buffers"][:5])
        self._output_specs = tuple((tuple(plan["buffers"][i]["shape"]),
                                    plan["buffers"][i]["storage"])
                                   for i in plan["outputs"])
        self._storage = {"f16": 2, "bf16": 3}[plan["buffers"][5]["storage"]]
        self._cooperative = "_cooperative_128_" in self._members[0].entry
        self._pid, self._closed, self._handle = os.getpid(), False, 0
        self._lib: ct.CDLL | None = None
        self._finalizer: weakref.finalize | None = None
        self._lock = threading.RLock()
        self.receipt = None

    def manifest(self):
        self._guard()
        return json.loads(self._portable_text)

    def runtime_artifact(self):
        from .source_attention_reverse_runtime import runtime_artifact
        return runtime_artifact(self)

    def _guard(self):
        if self._pid != os.getpid():
            raise ValueError("source attention reverse cannot cross fork")
        if self._closed:
            raise ValueError("source attention reverse is closed")

    def _check(self, status):
        if status:
            message = self._library().tessera_nvidia_attention_vjp_last_error()
            raise RuntimeError(message.decode() if message else "source attention reverse failed")

    def _library(self) -> ct.CDLL:
        if self._lib is None:
            raise RuntimeError("source attention reverse provider is uninitialized")
        return self._lib

    def _prepare(self):
        from tessera.runtime import _load_nvidia_ptx_launch
        lib = _load_nvidia_ptx_launch()
        if lib is None or any(not hasattr(lib, self._api + role)
                              for role in ("prepare", "invoke", "last_times")):
            raise RuntimeError("source attention reverse requires the matching native provider")
        self._lib = lib
        prepare = getattr(lib, self._api + "prepare")
        invoke = getattr(lib, self._api + "invoke")
        last_times = getattr(lib, self._api + "last_times")
        P, S, I, L, U = ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_int64, ct.c_uint64
        lib.tessera_nvidia_attention_vjp_last_error.argtypes = []
        lib.tessera_nvidia_attention_vjp_last_error.restype = ct.c_char_p
        signature: list[Any] = (
            [P, S, ct.c_char_p, I] + [P, S, ct.c_char_p] * (len(self._members) - 1)
            + [ct.POINTER(L), ct.POINTER(I), ct.POINTER(I), ct.POINTER(I), S, I, ct.POINTER(U)])
        prepare.argtypes = signature
        prepare.restype = I
        invoke.argtypes = [
            U, ct.POINTER(P), ct.POINTER(S), S, ct.POINTER(U), S,
            ct.POINTER(P), ct.POINTER(S), S]
        invoke.restype = I
        last_times.argtypes = [U, ct.POINTER(ct.c_float), S]
        last_times.restype = I
        lib.tessera_nvidia_source_attention_reverse_close.argtypes = [U]
        lib.tessera_nvidia_source_attention_reverse_close.restype = I
        images = tuple(ct.create_string_buffer(m.image) for m in self._members)
        args: list[Any] = []
        for i, member in enumerate(self._members):
            args.extend((images[i], len(member.image), member.entry.encode()))
            if i == 0:
                args.append(int(self._cooperative))
        handle = U()
        self._check(prepare(
            *args, (L * 7)(*self._dims), (I * 3)(*self._roles), (I * 2)(*self._seeds),
            (I * len(self._selected))(*self._selected), len(self._selected),
            self._storage, ct.byref(handle)))
        self._handle = handle.value
        self._finalizer = weakref.finalize(self, _retire, lib, self._handle, self._pid)

    def _frame(self, values):
        if len(values) != 5:
            raise ValueError("source attention reverse needs three roots and two output cotangents")
        interfaces = tuple(getattr(v, "__cuda_array_interface__", None) for v in values)
        resident = tuple(interface is not None for interface in interfaces)
        if any(resident) and not all(resident):
            raise ValueError("source attention reverse requires one input placement")
        names = {"f16": "float16", "bf16": "bfloat16", "f32": "float32"}
        if all(resident):
            from .resident_nvidia_tensor import cuda_frontend_specs
            snapshots = []
            for interface in interfaces:
                if not isinstance(interface, dict):
                    raise ValueError("source attention reverse CUDA metadata is malformed")
                snapshot = dict(interface)
                for key in ("shape", "strides", "data"):
                    if isinstance(snapshot.get(key), (tuple, list)):
                        snapshot[key] = tuple(snapshot[key])
                snapshots.append(snapshot)
            projected = tuple(SimpleNamespace(__cuda_array_interface__=s, dtype=getattr(v, "dtype", None))
                              for s, v in zip(snapshots, values, strict=True))
            specs = cuda_frontend_specs(projected, ranks=(3, 4))
            if any(shape != expected[0] or dtype.name != names[expected[1]]
                   for (shape, dtype), expected in zip(specs, self._input_specs, strict=True)):
                raise ValueError("source attention reverse CUDA shape or storage differs")
            return values, tuple(s["data"][0] for s in snapshots), tuple(s["stream"] for s in snapshots)
        if any(not isinstance(v, np.ndarray) or not v.dtype.isnative or tuple(v.shape) != spec[0]
               or v.dtype.name != names[spec[1]] for v, spec in zip(values, self._input_specs, strict=True)):
            raise ValueError("source attention reverse host shape or storage differs")
        roots = tuple(np.ascontiguousarray(v) for v in values)
        return roots, tuple(v.ctypes.data for v in roots), None

    def __call__(self, *values):
        from .native_tensor_result import input_scope
        self._guard()
        with self._lock, input_scope(values):
            self._guard()
            self.receipt = None
            roots, addresses, streams = self._frame(values)
            import ml_dtypes
            dtypes = {"f16": np.float16, "bf16": ml_dtypes.bfloat16, "f32": np.float32}
            outputs = tuple(np.empty(shape, dtypes[storage]) for shape, storage in self._output_specs)
            prepare_ms = 0.0
            if not self._handle:
                prepare_start = time.perf_counter()
                self._prepare()
                prepare_ms = (time.perf_counter() - prepare_start) * 1000
            start = time.perf_counter()
            lib = self._library()
            self._check(getattr(lib, self._api + "invoke")(
                self._handle, (ct.c_void_p * 5)(*addresses),
                (ct.c_size_t * 5)(*(s[2] for s in self._input_specs)), 5,
                (ct.c_uint64 * 5)(*streams) if streams is not None else None,
                5 if streams is not None else 0,
                (ct.c_void_p * len(outputs))(*(v.ctypes.data for v in outputs)),
                (ct.c_size_t * len(outputs))(*(v.nbytes for v in outputs)), len(outputs)))
            host_ms = (time.perf_counter() - start) * 1000
            times = (ct.c_float * (self._stage_count + 1))()
            self._check(getattr(lib, self._api + "last_times")(self._handle, times, len(times)))
            self.receipt = dict(execution_kind="native_gpu", evidence_target="nvidia_sm120",
                                native_call_binding=("prepared_cpp_source_attention_reverse_rows" if self._rows
                                                     else "prepared_cpp_source_attention_reverse"),
                                image_digests=tuple(m.image_digest for m in self._members),
                                stage_device_ms=tuple(times)[:self._stage_count], program_device_ms=times[-1],
                                completed_host_ms=host_ms, native_prepare_ms=prepare_ms,
                                input_binding="cuda_resident_roots" if streams else "host_roots")
            # roots retains the host snapshots through synchronous native completion.
            _ = roots
            return outputs

    def close(self):
        if self._pid != os.getpid():
            raise ValueError("source attention reverse cannot cross fork")
        with self._lock:
            if self._closed:
                return
            if self._handle:
                self._check(self._library().tessera_nvidia_source_attention_reverse_close(self._handle))
            if self._finalizer is not None:
                self._finalizer.detach()
            self._handle, self._closed = 0, True

    def __enter__(self):
        self._guard()
        return self

    def __exit__(self, *exc):
        self.close()
