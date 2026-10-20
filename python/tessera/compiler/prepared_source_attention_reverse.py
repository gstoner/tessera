"""Retained native owner for the compiler's original-source attention VJP."""
from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
import hashlib
import json
import math
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
        from .portable_source_attention_reverse import _rows
        expected_bounds = tuple(plan.get("shape_bounds", ()))
        if _rows(plan):
            if (forward.output_storage != forward.storage or forward.bias
                    or tuple(forward.shape_bounds) != expected_bounds):
                raise ValueError("source attention reverse member contracts disagree")
        else:
            backward = _decode_checkpoint(*stages[2][:3], backward=True)
            if (forward.dims != backward.dims or forward.scale != backward.scale
                    or forward.causal != backward.causal or forward.storage != backward.storage
                    or forward.output_storage != forward.storage
                    or backward.output_storage != forward.storage or forward.bias or backward.bias
                    or not backward.lse_cotangent
                    or tuple(forward.shape_bounds) != expected_bounds
                    or tuple(backward.shape_bounds) != expected_bounds):
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
        from .portable_source_attention_reverse import _rows, _bounded, _symbolic_dims
        self._members = members
        self._rows, self._bounded = _rows(plan), _bounded(plan)
        self._stage_count = len(members)
        self._times_api = ("tessera_nvidia_source_attention_reverse_rows_" if self._rows
                           else "tessera_nvidia_source_attention_reverse_")
        self._api = self._times_api + ("bounded_" if self._bounded else "")
        self._symbolic_dims = tuple(_symbolic_dims(plan, dims))
        self._requested = tuple(plan["requested_gradients"])
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
        if lib is None or any(not hasattr(lib, symbol) for symbol in (
                self._api + "prepare", self._api + "invoke", self._times_api + "last_times")):
            raise RuntimeError("source attention reverse requires the matching native provider")
        self._lib = lib
        prepare = getattr(lib, self._api + "prepare")
        invoke = getattr(lib, self._api + "invoke")
        last_times = getattr(lib, self._times_api + "last_times")
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
        if self._bounded:
            invoke.argtypes = [U, ct.POINTER(L), *invoke.argtypes[1:]]
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

    def _validate_specs(self, specs):
        names = {"f16": "float16", "bf16": "bfloat16", "f32": "float32"}
        for (shape, dtype), (expected, storage, _) in zip(specs, self._input_specs, strict=True):
            if len(shape) != len(expected) or dtype.name != names[storage]:
                raise ValueError("source attention reverse shape or storage differs")
        dims = self._dims
        if self._bounded:
            from .attention_shape_contract import attention_dimensions, DYNAMIC_DIM
            q, k, v = (specs[i][0] for i in self._roles)
            dims = attention_dimensions(self._symbolic_dims, self._dims,
                                       (q[0], q[1], k[1], q[2], k[2], q[3], v[3]))
            key_roots = set(self._roles[1:])
            for i, ((shape, _), (logical, _, _)) in enumerate(zip(specs, self._input_specs, strict=True)):
                wanted = tuple((dims[4] if i in key_roots else dims[3]) if extent == DYNAMIC_DIM
                               else extent for extent in logical)
                if shape != wanted:
                    raise ValueError("source attention reverse actual sequence/storage roles differ")
        elif any(shape != expected[0] for (shape, _), expected in zip(specs, self._input_specs, strict=True)):
            raise ValueError("source attention reverse shape or storage differs")
        sizes = tuple(math.prod(shape) * dtype.itemsize for shape, dtype in specs)
        if any(size > spec[2] for size, spec in zip(sizes, self._input_specs, strict=True)):
            raise ValueError("source attention reverse frame exceeds byte capacity")
        outputs = tuple((specs[index][0], storage) for index, (_, storage)
                        in zip(self._requested, self._output_specs, strict=True))
        return dims, sizes, outputs

    def _frame(self, values):
        if len(values) != 5:
            raise ValueError("source attention reverse needs three roots and two output cotangents")
        interfaces = tuple(getattr(v, "__cuda_array_interface__", None) for v in values)
        resident = tuple(interface is not None for interface in interfaces)
        if any(resident) and not all(resident):
            raise ValueError("source attention reverse requires one input placement")
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
            frame = self._validate_specs(specs)
            return (values, tuple(s["data"][0] for s in snapshots), tuple(s["stream"] for s in snapshots), *frame)
        if any(not isinstance(v, np.ndarray) or not v.dtype.isnative for v in values):
            raise ValueError("source attention reverse host shape or storage differs")
        frame = self._validate_specs(tuple((tuple(v.shape), v.dtype) for v in values))
        roots = tuple(np.ascontiguousarray(v) for v in values)
        return roots, tuple(v.ctypes.data for v in roots), None, *frame

    def __call__(self, *values):
        from .native_tensor_result import input_scope
        self._guard()
        with self._lock, input_scope(values):
            self._guard()
            self.receipt = None
            roots, addresses, streams, actual_dims, input_sizes, output_specs = self._frame(values)
            import ml_dtypes
            dtypes = {"f16": np.float16, "bf16": ml_dtypes.bfloat16, "f32": np.float32}
            outputs = tuple(np.empty(shape, dtypes[storage]) for shape, storage in output_specs)
            prepare_ms = 0.0
            if not self._handle:
                prepare_start = time.perf_counter()
                self._prepare()
                prepare_ms = (time.perf_counter() - prepare_start) * 1000
            start = time.perf_counter()
            lib = self._library()
            frame_args: list[Any] = [self._handle]
            if self._bounded:
                frame_args.append((ct.c_int64 * 7)(*actual_dims))
            frame_args.extend(((ct.c_void_p * 5)(*addresses),
                (ct.c_size_t * 5)(*input_sizes), 5,
                (ct.c_uint64 * 5)(*streams) if streams is not None else None,
                5 if streams is not None else 0,
                (ct.c_void_p * len(outputs))(*(v.ctypes.data for v in outputs)),
                (ct.c_size_t * len(outputs))(*(v.nbytes for v in outputs)), len(outputs)))
            self._check(getattr(lib, self._api + "invoke")(*frame_args))
            host_ms = (time.perf_counter() - start) * 1000
            times = (ct.c_float * (self._stage_count + 1))()
            self._check(getattr(lib, self._times_api + "last_times")(self._handle, times, len(times)))
            self.receipt = dict(execution_kind="native_gpu", evidence_target="nvidia_sm120",
                                native_call_binding=("prepared_cpp_source_attention_reverse_rows" if self._rows
                                                     else "prepared_cpp_source_attention_reverse"),
                                image_digests=tuple(m.image_digest for m in self._members),
                                stage_device_ms=tuple(times)[:self._stage_count], program_device_ms=times[-1],
                                completed_host_ms=host_ms, native_prepare_ms=prepare_ms,
                                input_binding="cuda_resident_roots" if streams else "host_roots")
            if self._bounded:
                self.receipt.update(shape_policy="bounded_sequences_v1", shape_bounds=self._dims,
                                    runtime_dims=actual_dims)
                self.receipt["native_call_binding"] += "_bounded"
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
