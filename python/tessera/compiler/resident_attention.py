"""Synchronous CUDA ownership for generated, saved-LSE attention pairs.

The frame retains private Q/K/V and LSE allocations from one forward generation.
No host tensor bridge or Graph IR reconstruction participates in backward.
"""
from __future__ import annotations

import ctypes as ct
import math
import threading
import numpy as np
from .native_device_tape import _Buffer


def checkpoint_shapes(pair):
    """Validate the bounded physical ABI before loading a driver or image."""
    from .nvidia_native import SM120_ATTN_LSE_F32_ABI, SM120_ATTN_BWD_LSE_F32_ABI, _checkpoint_identity
    from .native_artifact import NativeImageArtifact, LaunchDescriptor
    shapes = None
    result_dims: tuple[int, ...] = ()
    for package, backward in ((pair.forward, False), (pair.backward, True)):
        # Round-trip also detects mutations to nested provenance dictionaries.
        image = NativeImageArtifact.from_dict(package.image.to_dict())
        desc = LaunchDescriptor.from_dict(package.descriptor.to_dict())
        desc.validate_image(image)
        p = desc.provenance
        dims = p.get('shape')
        scale, causal = p.get('scale'), p.get('causal')
        if (not isinstance(dims, (list, tuple)) or len(dims) != 7 or
                any(type(d) is not int or not 0 < d < (1 << 63) for d in dims) or
                not isinstance(causal, bool) or isinstance(scale, bool) or not isinstance(scale, (int, float)) or
                not math.isfinite(scale) or scale <= 0 or scale > float(np.finfo(np.float32).max) or float(np.float32(scale)) == 0):
            raise ValueError('resident attention requires a concrete checkpoint policy')
        b, hq, hkv, sq, sk, d, dv = dims
        identity = _checkpoint_identity(tuple(dims), scale, causal)
        if hq % hkv or pair.contract_digest != identity or p.get('checkpoint_contract') != identity:
            raise ValueError('resident attention producer/consumer identity disagrees')
        expected = ((b,hq,sq,d), (b,hkv,sk,d), (b,hkv,sk,dv), (b,hq,sq,dv), (b,hq,sq))
        if (any(math.prod(shape) > ((1 << 63) - 1) // 4 for shape in expected) or
                (max(sum(math.prod(shape) for shape in expected[:3]), math.prod(expected[3])) + 127) // 128 > (1 << 31) - 1):
            raise ValueError('resident attention shape exceeds allocation or launch bounds')
        roles = (expected[3], *expected[:3], expected[4], *expected[:3]) if backward else expected
        count = 8 if backward else 5
        policy = ('sm120_attention_backward_lse_deterministic_direct_128' if backward
                  else 'sm120_attention_lse_thread_per_output_128')
        if (image.target != 'nvidia_sm120' or image.architecture != 'sm_120a' or image.binary_format != 'ptx' or
                desc.abi_id != (SM120_ATTN_BWD_LSE_F32_ABI if backward else SM120_ATTN_LSE_F32_ABI) or
                len(desc.buffers) != count or len(desc.scalars) != 7 or
                desc.geometry.policy != policy or desc.workspace.bytes != 0 or
                desc.dynamic_local_memory_bytes or desc.dynamic_local_memory_expression is not None or
                p.get('mask_alignment') != 'end_aligned_v1' or p.get('lse_checkpoint') != 'saved'):
            raise ValueError('unsupported resident attention image or launch contract')
        for i, (binding, shape) in enumerate(zip(desc.buffers, roles, strict=True)):
            expected_access = 'input' if i < (5 if backward else 3) else 'output'
            if (binding.ordinal != i or binding.dtype != 'fp32' or binding.rank != len(shape) or
                    binding.layout != 'row_major' or binding.alignment != 4 or binding.direction != expected_access):
                raise ValueError('resident attention buffer ABI disagrees')
            guards = [(g.dimension,g.predicate,g.value) for g in desc.shape_guards if g.binding == binding.name]
            if sorted(guards) != [(axis,'eq',extent) for axis,extent in enumerate(shape)]:
                raise ValueError('resident attention shape guards disagree')
        for i, scalar in enumerate(desc.scalars):
            if scalar.ordinal != count+i or scalar.dtype != 'int64' or scalar.name != ('B','Hq','Hkv','Sq','Sk','D','Dv')[i]:
                raise ValueError('resident attention scalar ABI disagrees')
        if shapes is not None and shapes != expected:
            raise ValueError('resident attention saved shapes disagree')
        shapes = expected
        result_dims = tuple(dims)
    return result_dims, shapes


class ResidentAttentionTape:
    """Own a forward generation until explicit close; backward results persist.

    Calls synchronize the owning context. Stream-overlap and higher-order AD
    are separate contracts. Callers must not write through read-only views.
    """
    def __init__(self, pair, q, k, v):
        self.dims, self.shapes = checkpoint_shapes(pair)
        self.closed = False
        self._jvp_binding = None
        self._scale = float(pair.forward.descriptor.provenance["scale"])
        self._causal = pair.forward.descriptor.provenance["causal"]
        self.buffers = []
        self._modules = []
        self._lock = threading.RLock()
        self._driver = ct.CDLL('libcuda.so.1')
        P, S, U = ct.c_void_p, ct.c_size_t, ct.c_uint
        def bind(name, args):
            fn = getattr(self._driver, name)
            fn.argtypes, fn.restype = args, ct.c_int
            return fn
        self.alloc = bind('cuMemAlloc_v2', [ct.POINTER(P),S])
        self.free = bind('cuMemFree_v2', [P])
        self.sync = bind('cuCtxSynchronize', [])
        self.copy = bind('cuMemcpyDtoD_v2', [P,P,S])
        self._current = bind('cuCtxGetCurrent', [ct.POINTER(P)])
        self._range = bind('cuMemGetAddressRange_v2', [ct.POINTER(P),ct.POINTER(S),P])
        self._attribute = bind('cuPointerGetAttribute', [P,ct.c_int,ct.c_uint64])
        self._unload = bind('cuModuleUnload', [P])
        load = bind('cuModuleLoadData', [ct.POINTER(P),P])
        entry = bind('cuModuleGetFunction', [ct.POINTER(P),P,ct.c_char_p])
        self._launch = bind('cuLaunchKernel', [P]+[U]*7+[P,ct.POINTER(P),ct.POINTER(P)])
        self.context = P()
        self.check(self._current(ct.byref(self.context)))
        if not self.context.value:
            raise ValueError('resident attention requires a current CUDA context')
        self._functions = []
        self.contract_digest = pair.contract_digest
        try:
            for package in (pair.forward, pair.backward):
                module, function = P(), P()
                blob = ct.create_string_buffer(package.image.payload)
                self.check(load(ct.byref(module), ct.cast(blob,P)))
                self._modules.append(module)
                self.check(entry(ct.byref(function),module,package.descriptor.entry_symbol.encode()))
                self._functions.append(function)
            pointers = [self._resident(value, shape) for value,shape in zip((q,k,v),self.shapes[:3],strict=True)]
            self.check(self.sync())
            self._saved = [_Buffer(self,shape) for shape in self.shapes]
            for pointer, saved in zip(pointers,self._saved[:3],strict=True):
                self.check(self.copy(saved.pointer,P(pointer),saved.nbytes))
            self._execute(False,self._saved)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f'resident attention CUDA status {status}')

    def _ready(self):
        if self.closed:
            raise ValueError('resident attention tape is closed')
        current = ct.c_void_p()
        self.check(self._current(ct.byref(current)))
        if current.value != self.context.value:
            raise ValueError('resident attention requires its owning CUDA context')

    def _resident(self, value, shape):
        interface = getattr(value,'__cuda_array_interface__',None)
        if not isinstance(interface,dict) or type(interface.get('version')) is not int or interface['version'] not in (2,3):
            raise ValueError('resident attention requires a CUDA tensor')
        dtype = np.dtype(interface.get('typestr'))
        actual = interface.get('shape',())
        pointer, readonly = interface.get('data',(None,None))
        strides: list[int] = []
        stride = 4
        for dim in reversed(shape):
            strides.insert(0,stride)
            stride *= dim
        if (dtype != np.dtype('float32') or not dtype.isnative or tuple(actual) != shape or
                any(type(d) is not int for d in actual) or
                (interface.get('strides') is not None and tuple(interface['strides']) != tuple(strides)) or
                type(pointer) is not int or pointer <= 0 or pointer % 4 or type(readonly) is not bool):
            raise ValueError('resident attention tensor shape, dtype or storage disagrees')
        owner, base, size = ct.c_void_p(), ct.c_void_p(), ct.c_size_t()
        # Attribute 1 is the allocation's CUDA context, not merely its device.
        self.check(self._attribute(ct.byref(owner),1,pointer))
        if owner.value != self.context.value:
            raise ValueError('resident attention input belongs to another context')
        self.check(self._range(ct.byref(base),ct.byref(size),ct.c_void_p(pointer)))
        if not base.value or pointer < base.value or pointer + stride > base.value + size.value:
            raise ValueError('resident attention exceeds device allocation')
        return pointer

    def _execute(self, backward, buffers):
        b,hq,hkv,sq,sk,d,dv = self.dims
        total = b*hq*sq*d + b*hkv*sk*d + b*hkv*sk*dv if backward else b*hq*sq*dv
        arguments = [ct.c_void_p(buf.pointer.value) for buf in buffers] + [ct.c_int64(v) for v in self.dims]
        pointers = (ct.c_void_p*len(arguments))(*(ct.cast(ct.pointer(a),ct.c_void_p) for a in arguments))
        self.check(self._launch(self._functions[int(backward)],(total+127)//128,1,1,128,1,1,0,None,pointers,None))
        self.check(self.sync())

    @property
    def primal(self):
        with self._lock:
            self._ready()
            return _ReadOnly(self._saved[3])

    def backward(self, cotangent):
        with self._lock:
            self._ready()
            pointer = self._resident(cotangent,self.shapes[3])
            self.check(self.sync())
            start = len(self.buffers)
            try:
                do = _Buffer(self,self.shapes[3])
                self.check(self.copy(do.pointer,ct.c_void_p(pointer),do.nbytes))
                gradients = [_Buffer(self,shape) for shape in self.shapes[:3]]
                self._execute(True,[do,*self._saved[:3],self._saved[4],*gradients])
            except BaseException:
                self._release(start)
                raise
            # Cotangent is needed only until the synchronous launch completes.
            self.check(self.free(do.pointer))
            do.pointer = ct.c_void_p()
            self.buffers.remove(do)
            return tuple(_ReadOnly(g) for g in gradients)

    def prepare_jvp(self, *, compiler, llvm_bin, source=None):
        """Compile a cooperative native score-tangent consumer for this frame."""
        import inspect
        from .native_attention_jvp import materialize, materialize_generated
        from .native_storage_contract import generate_tensor_binding
        with self._lock:
            self._ready()
            if self._jvp_binding is not None:
                raise ValueError('resident attention JVP is already prepared')
            package = (materialize(self.dims,self._scale,self._causal,compiler=compiler,llvm_bin=llvm_bin)
                       if source is None else
                       materialize_generated(source,self.dims,self._scale,self._causal,compiler=compiler,llvm_bin=llvm_bin))
            names = ('q','k','v','primal','lse','dq','dk','dv','tangent','scratch')
            signature = inspect.Signature([inspect.Parameter(n,inspect.Parameter.POSITIONAL_ONLY) for n in names])
            self._jvp_binding = generate_tensor_binding(package,signature)
            return package.binding_digest

    def jvp(self, dq, dk, dv):
        """Execute a Q/K/V direction against this captured O/LSE generation."""
        with self._lock:
            self._ready()
            if self._jvp_binding is None:
                raise ValueError('resident attention JVP requires prepare_jvp')
            for value,shape in zip((dq,dk,dv),self.shapes[:3],strict=True):
                self._resident(value,shape)
            start = len(self.buffers)
            try:
                result = _Buffer(self,self.shapes[3])
                self._jvp_binding(*self._saved,dq,dk,dv,result,128)
            except BaseException:
                self._release(start)
                raise
            return _ReadOnly(result)

    def _release(self, start=0):
        self.check(self.sync())
        while len(self.buffers) > start:
            buf = self.buffers[-1]
            self.check(self.free(buf.pointer))
            buf.pointer = ct.c_void_p()
            self.buffers.pop()

    def close(self):
        with self._lock:
            if self.closed:
                return
            self._ready()
            if self._jvp_binding is not None:
                self._jvp_binding.close()
                self._jvp_binding = None
            self._release()
            while self._modules:
                self.check(self._unload(self._modules[-1]))
                self._modules.pop()
            self.closed = True

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self,*_exc):
        self.close()


class _ReadOnly:
    def __init__(self,buffer):
        self._buffer = buffer

    @property
    def __cuda_array_interface__(self):
        interface = self._buffer.__cuda_array_interface__.copy()
        interface['data'] = (interface['data'][0],True)
        return interface
