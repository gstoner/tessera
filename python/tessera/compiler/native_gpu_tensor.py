"""Typed tensor/scalar adapter for an explicitly selected native GPU package."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import ctypes as ct
import hashlib
import inspect
import json
import math
import threading
from typing import Any
import numpy as np
from tessera.dtype import canonicalize_dtype
from .native_gpu_storage import NativeGPUStoragePackage


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int | str, ...]
    writable: bool = False


@dataclass(frozen=True)
class IndexSpec:
    name: str
    minimum: int = 1
    maximum: int = (1 << 31) - 1


def validate_tensor_signature(abi, signature, specs, grid, block):
    if len(specs) != len(abi) or len({s.name for s in specs}) != len(specs):
        raise ValueError('native tensor ABI argument count or names disagree')
    if set(signature.parameters) != {s.name for s in specs}:
        raise ValueError('native tensor ABI must cover the JIT signature exactly')
    names = {s.name for s in specs if isinstance(s, IndexSpec)}
    for kind, spec in zip(abi, specs, strict=True):
        if (kind == 'pointer') != isinstance(spec, TensorSpec):
            raise ValueError('native tensor ABI kind disagrees')
        if isinstance(spec, TensorSpec):
            canonicalize_dtype(spec.dtype)
            for dim in spec.shape:
                if not ((type(dim) is int and dim > 0) or (isinstance(dim, str) and dim in names)):
                    raise ValueError('tensor shape must use positive constants or declared indices')
        elif type(spec.minimum) is not int or type(spec.maximum) is not int or not 0 <= spec.minimum <= spec.maximum < (1 << 63):
            raise ValueError('invalid native index bounds')
    if len(grid) != 3 or len(block) != 3:
        raise ValueError('native launch geometry requires three dimensions')
    for dim in grid + block:
        if not ((type(dim) is int and dim > 0) or (isinstance(dim, str) and dim in names)):
            raise ValueError('launch geometry must use constants or declared indices')


class NativeTensorCall:
    """Explicit ABI binding, never a claim that arbitrary Python math matches IR.

    Device allocations and output tensors remain caller-owned. Calls synchronize
    producer work; explicit submissions use stream events and retained owners.
    """
    def __init__(self, package: NativeGPUStoragePackage, signature: inspect.Signature,
                 specs: tuple[TensorSpec | IndexSpec, ...], *,
                 grid: tuple[int | str, int | str, int | str],
                 block: tuple[int | str, int | str, int | str]):
        package.validate()
        validate_tensor_signature(package.abi, signature, specs, grid, block)
        self.package, self.signature, self.specs = package, signature, specs
        self.grid, self.block = grid, block
        data = {'package': package.binding_digest, 'specs': [asdict(s) for s in specs], 'grid': grid, 'block': block}
        self.binding_digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        self._bound: Any = None
        self._lock = threading.RLock()
        self._inflight: list[tuple[Any, list[tuple[int, int, bool]]]] = []

    def prepare(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        indices = {}
        for spec in self.specs:
            if isinstance(spec, IndexSpec):
                v = values[spec.name]
                if type(v) is not int or not spec.minimum <= v <= spec.maximum:
                    raise ValueError(f'{spec.name} violates native index bounds')
                indices[spec.name] = v
        def resolve(d):
            return indices[d] if isinstance(d, str) else d
        raw: list[int] = []
        allocations: list[tuple[int, int]] = []
        outputs: list[Any] = []
        for spec in self.specs:
            value = values[spec.name]
            if isinstance(spec, IndexSpec):
                raw.append(value)
                continue
            interface = getattr(value, '__cuda_array_interface__', None)
            if not isinstance(interface, dict) or type(interface.get('version')) is not int or interface.get('version') not in (2, 3):
                raise ValueError(f'{spec.name} requires a resident CUDA/HIP array interface')
            dtype = np.dtype(interface['typestr'])
            shape = tuple(interface['shape'])
            if canonicalize_dtype(str(dtype)) != canonicalize_dtype(spec.dtype) or not dtype.isnative:
                raise ValueError(f'{spec.name} native dtype disagrees')
            if shape != tuple(resolve(d) for d in spec.shape) or any(type(d) is not int for d in shape):
                raise ValueError(f'{spec.name} native shape disagrees')
            strides = interface.get('strides')
            stride = dtype.itemsize
            expected: list[int] = []
            for dim in reversed(shape):
                expected.insert(0, stride)
                stride *= dim
            if strides is not None and tuple(strides) != tuple(expected):
                raise ValueError('native tensor ABI requires contiguous row-major storage')
            pointer, readonly = interface['data']
            if type(pointer) is not int or pointer <= 0 or pointer % dtype.alignment:
                raise ValueError('invalid or misaligned native tensor pointer')
            if type(readonly) is not bool or (spec.writable and readonly):
                raise ValueError('native output tensor is read-only')
            size = math.prod(shape) * dtype.itemsize
            if any(pointer < p + n and p < pointer + size for p, n in allocations):
                raise ValueError('native tensor bindings overlap')
            allocations.append((pointer, size))
            raw.append(pointer)
            if spec.writable:
                outputs.append(value)
        return tuple(raw), allocations, tuple(resolve(d) for d in self.grid), tuple(resolve(d) for d in self.block), tuple(outputs)

    def _resident(self, *args, **kwargs):
        raw, allocations, grid, block, outputs = self.prepare(*args, **kwargs)
        if self._bound is None:
            self._bound = self.package.bind()
        driver = self._bound._driver
        name = 'cuMemGetAddressRange_v2' if self.package.backend == 'nvidia' else 'hipMemGetAddressRange'
        query = getattr(driver, name)
        query.argtypes = [ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_size_t), ct.c_void_p]
        query.restype = ct.c_int
        device = ct.c_int()
        current = driver.cuCtxGetDevice if self.package.backend == 'nvidia' else driver.hipGetDevice
        current.argtypes = [ct.POINTER(ct.c_int)]
        current.restype = ct.c_int
        self._bound._check(current(ct.byref(device)))
        for pointer, size in allocations:
            if self.package.backend == 'nvidia':
                ordinal = ct.c_int()
                attribute = driver.cuPointerGetAttribute
                attribute.argtypes = [ct.c_void_p, ct.c_int, ct.c_uint64]
                attribute.restype = ct.c_int
                error = attribute(ct.byref(ordinal), 9, pointer)
                owner = ordinal.value
            else:
                class Attributes(ct.Structure):
                    _fields_ = [('type', ct.c_int), ('device', ct.c_int),
                                ('devicePointer', ct.c_void_p), ('hostPointer', ct.c_void_p),
                                ('isManaged', ct.c_int), ('allocationFlags', ct.c_uint)]
                attributes = Attributes()
                attribute = driver.hipPointerGetAttributes
                attribute.argtypes = [ct.POINTER(Attributes), ct.c_void_p]
                attribute.restype = ct.c_int
                error = attribute(ct.byref(attributes), ct.c_void_p(pointer))
                owner = attributes.device
            if error or owner != device.value:
                raise ValueError('native tensor belongs to another device or is not resident')
            base, length = ct.c_void_p(), ct.c_size_t()
            if query(ct.byref(base), ct.byref(length), ct.c_void_p(pointer)) or base.value is None or not base.value <= pointer or pointer + size > base.value + length.value:
                raise ValueError('native tensor exceeds its resident device allocation')
        return raw, allocations, grid, block, outputs

    def __call__(self, *args, **kwargs):
        with self._lock:
            raw, _, grid, block, outputs = self._resident(*args, **kwargs)
            self._bound._check(self._bound._sync())
            self._bound.launch(raw, grid=grid, block=block)
            return outputs[0] if len(outputs) == 1 else outputs

    def submit(self, stream: int, /, *args, **kwargs):
        """Enqueue without a device-wide wait and retain tensor owners until wait.

        Producer streams come from the array interface. Conflicting submissions
        through this binding are event-ordered; independent allocations can run
        concurrently. External writers must publish their stream in the interface.
        """
        if type(stream) is not int or not 0 < stream < (1 << 64):
            raise ValueError('submission requires a non-null native stream')
        with self._lock:
            raw, allocations, grid, block, outputs = self._resident(*args, **kwargs)
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            producers = []
            for spec in self.specs:
                if isinstance(spec, TensorSpec):
                    producer = bound.arguments[spec.name].__cuda_array_interface__.get('stream')
                    if producer is not None:
                        if type(producer) is not int or not 0 < producer < (1 << 64):
                            raise ValueError('invalid array producer stream')
                        producers.append(producer)
            accesses = [(p, n, spec.writable) for (p, n), spec in zip(allocations,
                (spec for spec in self.specs if isinstance(spec, TensorSpec)), strict=True)]
            self._inflight = [(t, a) for t, a in self._inflight if not t.done]
            for ticket, previous in self._inflight:
                if any((writes or prior_writes) and p < q + m and q < p + n
                       for p, n, writes in accesses for q, m, prior_writes in previous):
                    ticket.wait_on(stream)
            ticket = self._bound.submit(raw, grid=grid, block=block, stream=stream,
                producer_streams=tuple(producers), keepalive=tuple(bound.arguments.values()))
            self._inflight.append((ticket, accesses))
            return TensorSubmission(ticket, outputs)

    def close(self):
        with self._lock:
            if self._bound is not None:
                self._bound.close()
                self._bound = None
            self._inflight = []


class TensorSubmission:
    def __init__(self, ticket, outputs):
        self.ticket, self.outputs = ticket, outputs

    def wait(self):
        self.ticket.wait()
        return self.outputs[0] if len(self.outputs) == 1 else self.outputs
