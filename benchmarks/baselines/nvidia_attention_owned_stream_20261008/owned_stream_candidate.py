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
    from .nvidia_native import (SM120_ATTN_LSE_F32_ABI, SM120_ATTN_BWD_LSE_F32_ABI,
                                SM120_ATTN_LSE_BIAS_F32_ABI, SM120_ATTN_BWD_LSE_BIAS_F32_ABI,
                                SM120_ATTN_BWD_LSE_BIAS_GRAD_F32_ABI,
                                SM120_ATTN_LSE_BCAST_F32_ABI, SM120_ATTN_BWD_LSE_BCAST_F32_ABI,
                                SM120_ATTN_BWD_LSE_BCAST_GRAD_F32_ABI,
                                SM120_ATTN_BWD_LSE_COMPACT_F32_ABI, SM120_ATTN_BWD_LSE_COTANGENT_F32_ABI,
                                _checkpoint_identity)
    from .native_artifact import NativeImageArtifact, LaunchDescriptor
    shapes = None
    prior_bias_storage = None
    result_dims: tuple[int, ...] = ()
    from .nvidia_native import AttentionCheckpointPair, AttentionForwardCheckpoint, NVIDIANativePackage
    packages: tuple[tuple[NVIDIANativePackage, bool], ...]
    if isinstance(pair, AttentionForwardCheckpoint):
        packages = ((pair.forward, False),)
    elif isinstance(pair, AttentionCheckpointPair):
        packages = ((pair.forward, False), (pair.backward, True))
    else:
        raise ValueError("resident attention requires a native checkpoint product")
    for package, backward in packages:
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
        bias = p.get("bias", False)
        if type(bias) is not bool:
            raise ValueError("resident attention bias policy must be boolean")
        bias_gradient = p.get("bias_gradient", False)
        if type(bias_gradient) is not bool or (bias_gradient and (not backward or not bias)):
            raise ValueError("resident attention bias gradient policy disagrees")
        seeded = p.get("lse_cotangent", False)
        if type(seeded) is not bool or (seeded and not backward):
            raise ValueError("resident attention row seed policy disagrees")
        if seeded:
            from .lse_cotangent_contract import lse_cotangent_contract
            lse_cotangent_contract(desc)
        compact = p.get("gradient_output", "complete_v1") == "compact_v1"
        if p.get("gradient_output", "complete_v1") not in ("complete_v1", "compact_v1"):
            raise ValueError("resident attention gradient output policy disagrees")
        if compact and not backward:
            raise ValueError("compact output requires a backward checkpoint")
        activity = p.get("gradient_activity", ())
        if not isinstance(activity, (list, tuple)):
            raise ValueError("resident attention gradient activity must be a sequence")
        if activity:
            if (not backward or not isinstance(activity, (list, tuple)) or
                    len(activity) != 3 + int(bias_gradient) or
                    any(type(x) is not int or x not in (0, 1) for x in activity) or
                    not any(activity) or p.get("inactive_gradient") != ("absent_v1" if compact else "zero_fill_v1")):
                raise ValueError("resident attention gradient activity disagrees")
            text = "gradient_activity = array<i64: " + ", ".join(map(str, activity)) + ">"
            if text not in package.tile_ir:
                raise ValueError("resident attention gradient activity disagrees with native Tile contract")
        elif compact or p.get("inactive_gradient", "none") != "none":
            raise ValueError("resident attention inactive gradient policy disagrees")
        gradient_roles = tuple(i for i in range(3 + int(bias_gradient)) if not compact or activity[i])
        physical_roles = p.get("physical_gradient_roles", gradient_roles)
        if not isinstance(physical_roles, (list, tuple)):
            raise ValueError("resident attention physical gradient roles must be a sequence")
        if backward and tuple(physical_roles) != gradient_roles:
            raise ValueError("resident attention physical gradient roles disagree")
        if compact:
            from .compact_attention_contract import compact_attention_contract
            if not seeded:
                compact_attention_contract(desc)
            import json
            import re
            contract = re.search(r"tessera.native_contract = \{([^\n]*)\}", package.tile_ir)
            physical = re.search(r"physical_results = (\[[^\]]*\])", contract[1]) if contract else None
            if ('gradient_output = "compact_v1"' not in package.tile_ir or physical is None or
                    json.loads(physical[1]) != [x.name for x in desc.buffers if x.direction == "output"]):
                raise ValueError("resident attention compact output differs from native contract")
        bias_dims = p.get("bias_shape", ())
        if (not isinstance(bias_dims, (list, tuple)) or
                (bias_dims and (len(bias_dims) != 4 or
                 any(type(dim) is not int or dim <= 0 for dim in bias_dims)))):
            raise ValueError("resident attention bias shape must contain four positive dimensions")
        physical_bias: tuple[int, ...] = tuple(bias_dims)
        identity = _checkpoint_identity(tuple(dims), scale, causal, bias=bias, bias_shape=physical_bias)
        if physical_bias and p.get("bias_gradient_reduction") != "physical_owner_lexicographic_bhqk_v1":
            raise ValueError("resident attention bias reduction policy disagrees")
        bias_storage = physical_bias or (b,hq,sq,sk)
        if hq % hkv or pair.contract_digest != identity or p.get('checkpoint_contract') != identity:
            raise ValueError('resident attention producer/consumer identity disagrees')
        expected = ((b,hq,sq,d), (b,hkv,sk,d), (b,hkv,sk,dv), (b,hq,sq,dv), (b,hq,sq))
        if (any(math.prod(shape) > ((1 << 63) - 1) // 4 for shape in expected) or
                (max(sum(math.prod(shape) for shape in expected[:3]), math.prod(expected[3])) + 127) // 128 > (1 << 31) - 1):
            raise ValueError('resident attention shape exceeds allocation or launch bounds')
        roles: tuple[tuple[int, ...], ...] = (expected[3], *expected[:3], expected[3], expected[4], *expected[:3]) if backward else expected
        if bias:
            index = 5 if backward else 3
            roles = (*roles[:index], bias_storage, *roles[index:])
        if seeded:
            index = 6 + int(bias)
            roles = (*roles[:index], expected[4], *roles[index:])
        if bias_gradient:
            roles = (*roles, bias_storage)
        input_count = (6 if backward else 3) + int(bias) + int(seeded)
        if backward and compact:
            output_shapes = roles[input_count:]
            roles = (*roles[:input_count], *(output_shapes[i] for i in gradient_roles))
        gradient_elements = sum(math.prod(shape) for shape in
            (*expected[:3], *((bias_storage,) if bias_gradient else ())))
        if compact and p.get("gradient_launch") == "packed_v1":
            gradient_elements = sum(math.prod(shape) for shape in roles[input_count:])
        threads = p.get("gradient_block_threads",128) if compact else 128
        if (gradient_elements + threads - 1) // threads > (1 << 31) - 1:
            raise ValueError("resident attention gradient launch exceeds bounds")
        if any(math.prod(shape) > ((1 << 63) - 1) // 4 for shape in roles):
            raise ValueError("resident attention bias allocation exceeds bounds")
        count = (9 if backward else 5) + int(bias) + int(bias_gradient) + int(seeded)
        expected_abi = ((SM120_ATTN_BWD_LSE_BIAS_F32_ABI if backward else SM120_ATTN_LSE_BIAS_F32_ABI)
                        if bias else (SM120_ATTN_BWD_LSE_F32_ABI if backward else SM120_ATTN_LSE_F32_ABI))
        if bias_gradient:
            expected_abi = SM120_ATTN_BWD_LSE_BIAS_GRAD_F32_ABI
        if physical_bias:
            expected_abi = (SM120_ATTN_BWD_LSE_BCAST_GRAD_F32_ABI if bias_gradient else
                            SM120_ATTN_BWD_LSE_BCAST_F32_ABI if backward else SM120_ATTN_LSE_BCAST_F32_ABI)
        if compact:
            count = input_count + len(gradient_roles)
            expected_abi = SM120_ATTN_BWD_LSE_COMPACT_F32_ABI
        if seeded:
            expected_abi = SM120_ATTN_BWD_LSE_COTANGENT_F32_ABI
        scalar_names = ('B','Hq','Hkv','Sq','Sk','D','Dv') + (
            ('BiasB','BiasH','BiasQ','BiasK') if physical_bias else ())
        policy = (f'sm120_attention_backward_lse_deterministic_direct_{threads}' if backward
                  else 'sm120_attention_lse_thread_per_output_128')
        if (image.target != 'nvidia_sm120' or image.architecture != 'sm_120a' or image.binary_format != 'ptx' or
                desc.abi_id != expected_abi or
                len(desc.buffers) != count or len(desc.scalars) != len(scalar_names) or
                desc.geometry.policy != policy or desc.workspace.bytes != 0 or
                desc.dynamic_local_memory_bytes or desc.dynamic_local_memory_expression is not None or
                p.get('mask_alignment') != 'end_aligned_v1' or p.get('lse_checkpoint') != 'saved'):
            raise ValueError('unsupported resident attention image or launch contract')
        for i, (binding, shape) in enumerate(zip(desc.buffers, roles, strict=True)):
            expected_access = 'input' if i < input_count else 'output'
            if (binding.ordinal != i or binding.dtype != 'fp32' or binding.rank != len(shape) or
                    binding.layout != 'row_major' or binding.alignment != 4 or binding.direction != expected_access):
                raise ValueError('resident attention buffer ABI disagrees')
            guards = [(g.dimension,g.predicate,g.value) for g in desc.shape_guards if g.binding == binding.name]
            if sorted(guards) != [(axis,'eq',extent) for axis,extent in enumerate(shape)]:
                raise ValueError('resident attention shape guards disagree')
        for i, scalar in enumerate(desc.scalars):
            if scalar.ordinal != count+i or scalar.dtype != 'int64' or scalar.name != scalar_names[i]:
                raise ValueError('resident attention scalar ABI disagrees')
        if shapes is not None and (shapes != expected or prior_bias_storage != bias_storage):
            raise ValueError('resident attention saved shapes disagree')
        prior_bias_storage = bias_storage
        shapes = expected
        result_dims = tuple(dims)
    return result_dims, shapes


class ResidentAttentionTape:
    """Own a forward generation until explicit close; backward results persist.

    Calls synchronize a private stream after ordering producer dependencies.
    Asynchronous return and higher-order AD
    are separate contracts. Callers must not write through read-only views.
    """
    def __init__(self, pair, q, k, v, *, bias=None):
        self.dims, self.shapes = checkpoint_shapes(pair)
        self._has_bias = pair.forward.descriptor.provenance.get("bias", False)
        backward = getattr(pair, "backward", None)
        self._has_backward = backward is not None
        reverse_policy = backward.descriptor.provenance if backward is not None else {}
        self._has_bias_gradient = reverse_policy.get("bias_gradient", False)
        self._has_lse_cotangent = reverse_policy.get("lse_cotangent", False)
        self._gradient_launch = reverse_policy.get("gradient_launch", "logical_v1")
        self._backward_threads = reverse_policy.get("gradient_block_threads",128)
        self._gradient_roles = tuple(reverse_policy.get(
            "physical_gradient_roles", range(3 + int(self._has_bias_gradient))))
        if (bias is not None) != self._has_bias:
            raise ValueError("resident attention bias input disagrees with checkpoint policy")
        b,hq,_,sq,sk,_,_ = self.dims
        self._bias_shape = tuple(pair.forward.descriptor.provenance.get("bias_shape", ())) or (b,hq,sq,sk)
        compact = reverse_policy.get("gradient_output") == "compact_v1"
        physical = tuple(reverse_policy.get("bias_shape", ()))
        self._backward_scalars = self.dims + (physical if compact else ())
        self._bias = None
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
        self._stream = P()
        self._stream_create = bind('cuStreamCreate', [ct.POINTER(P),U])
        self._stream_destroy = bind('cuStreamDestroy_v2', [P])
        self._stream_sync = bind('cuStreamSynchronize', [P])
        self._stream_context = bind('cuStreamGetCtx', [P,ct.POINTER(P)])
        self._event_create = bind('cuEventCreate', [ct.POINTER(P),U])
        self._event_record = bind('cuEventRecord', [P,P])
        self._stream_wait = bind('cuStreamWaitEvent', [P,P,U])
        self._event_destroy = bind('cuEventDestroy_v2', [P])
        self._copy_async = bind('cuMemcpyDtoDAsync_v2', [P,P,S,P])
        self.sync = lambda: self._stream_sync(self._stream) if self._stream.value else 0
        self.copy = lambda dst,src,size: self._copy_async(dst,src,size,self._stream)
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
            self.check(self._stream_create(ct.byref(self._stream),1))  # CU_STREAM_NON_BLOCKING
            for package in ((pair.forward, backward) if backward is not None else (pair.forward,)):
                module, function = P(), P()
                blob = ct.create_string_buffer(package.image.payload)
                self.check(load(ct.byref(module), ct.cast(blob,P)))
                self._modules.append(module)
                self.check(entry(ct.byref(function),module,package.descriptor.entry_symbol.encode()))
                self._functions.append(function)
            pointers = [self._resident(value, shape) for value,shape in zip((q,k,v),self.shapes[:3],strict=True)]
            self._saved = [_Buffer(self,shape) for shape in self.shapes]
            for pointer, saved in zip(pointers,self._saved[:3],strict=True):
                self.check(self.copy(saved.pointer,P(pointer),saved.nbytes))
            if self._has_bias:
                b,hq,_,sq,sk,_,_ = self.dims
                pointer = self._resident(bias,self._bias_shape)
                self._bias = _Buffer(self,self._bias_shape)
                self.check(self.copy(self._bias.pointer,P(pointer),self._bias.nbytes))
            forward_buffers = [*self._saved[:3], *([self._bias] if self._has_bias else []), *self._saved[3:]]
            self._execute(False,forward_buffers)
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
        self._order_producer_stream(interface.get('stream'))
        return pointer

    def _order_producer_stream(self, stream):
        if stream is None:
            return  # The CUDA Array Interface declares no producer wait.
        if type(stream) is not int or not 0 < stream < (1 << 64):
            raise ValueError('resident attention CUDA producer stream is invalid')
        producer = ct.c_void_p(stream)
        context = ct.c_void_p()
        self.check(self._stream_context(producer,ct.byref(context)))
        if context.value != self.context.value:
            raise ValueError('resident attention producer stream belongs to another context')
        if stream == self._stream.value:
            return
        event = ct.c_void_p()
        self.check(self._event_create(ct.byref(event),2))  # CU_EVENT_DISABLE_TIMING
        try:
            self.check(self._event_record(event,producer))
            self.check(self._stream_wait(self._stream,event,0))
        finally:
            # CUDA retains the queued dependency after event destruction.
            self.check(self._event_destroy(event))

    def _execute(self, backward, buffers):
        b,hq,hkv,sq,sk,d,dv = self.dims
        total = b*hq*sq*d + b*hkv*sk*d + b*hkv*sk*dv if backward else b*hq*sq*dv
        if backward:
            shapes = (*self.shapes[:3], *((self._bias_shape,) if self._has_bias_gradient else ()))
            roles = self._gradient_roles if self._gradient_launch == "packed_v1" else range(len(shapes))
            total = sum(math.prod(shapes[i]) for i in roles)
        dimensions = self._backward_scalars if backward else self.dims
        arguments = [ct.c_void_p(buf.pointer.value) for buf in buffers] + [ct.c_int64(v) for v in dimensions]
        pointers = (ct.c_void_p*len(arguments))(*(ct.cast(ct.pointer(a),ct.c_void_p) for a in arguments))
        threads = self._backward_threads if backward else 128
        self.check(self._launch(self._functions[int(backward)],(total+threads-1)//threads,1,1,threads,1,1,0,self._stream,pointers,None))
        self.check(self.sync())

    @property
    def primal(self):
        with self._lock:
            self._ready()
            if self._has_lse_cotangent:
                return (_ReadOnly(self._saved[3]), _ReadOnly(self._saved[4]))
            return _ReadOnly(self._saved[3])

    def backward(self, cotangent):
        with self._lock:
            self._ready()
            if not self._has_backward:
                raise ValueError("forward checkpoint has no reverse executable")
            seed_shapes: tuple[tuple[int, ...], ...]
            if self._has_lse_cotangent:
                if not isinstance(cotangent, (tuple,list)) or len(cotangent)!=2:
                    raise ValueError("saved O/LSE attention requires two result cotangents")
                seed_shapes = (self.shapes[3],self.shapes[4])
                pointers = [self._resident(value,shape) for value,shape in zip(cotangent,seed_shapes,strict=True)]
            else:
                seed_shapes = (self.shapes[3],)
                pointers = [self._resident(cotangent,self.shapes[3])]

            start = len(self.buffers)
            try:
                seeds = [_Buffer(self,shape) for shape in seed_shapes]
                for buffer,pointer in zip(seeds,pointers,strict=True):
                    self.check(self.copy(buffer.pointer,ct.c_void_p(pointer),buffer.nbytes))
                gradient_shapes = self.shapes[:3]
                if self._has_bias_gradient:
                    b,hq,_,sq,sk,_,_ = self.dims
                    gradient_shapes = (*gradient_shapes, getattr(self, "_bias_shape", (b,hq,sq,sk)))
                gradients = [_Buffer(self,gradient_shapes[i]) for i in self._gradient_roles]
                self._execute(True,[seeds[0],*self._saved[:4], *([self._bias] if self._has_bias else []), self._saved[4],*seeds[1:],*gradients])
            except BaseException:
                self._release(start)
                raise
            # Cotangent is needed only until the synchronous launch completes.
            for seed in seeds:
                self.check(self.free(seed.pointer))
                seed.pointer = ct.c_void_p()
                self.buffers.remove(seed)
            return tuple(_ReadOnly(g) for g in gradients)

    def prepare_jvp(self, *, compiler, llvm_bin, source=None):
        """Compile a cooperative native score-tangent consumer for this frame."""
        import inspect
        from .native_attention_jvp import materialize, materialize_generated
        from .native_storage_contract import generate_tensor_binding
        with self._lock:
            self._ready()
            if self._has_bias and source is None:
                raise ValueError("resident bias JVP requires its traced native AD source")
            if self._jvp_binding is not None:
                raise ValueError('resident attention JVP is already prepared')
            package = (materialize(self.dims,self._scale,self._causal,compiler=compiler,llvm_bin=llvm_bin)
                       if source is None else
                       materialize_generated(source,self.dims,self._scale,self._causal,compiler=compiler,llvm_bin=llvm_bin,
                           bias_shape=self._bias_shape if self._has_bias else ()))
            if f'tessera.attention_checkpoint_identity = "{self.contract_digest}"' not in package.arena_ir:
                raise ValueError("resident JVP product differs from its captured forward generation")
            names = ('q','k','v','primal','lse','dq','dk','dv') + (
                ('bias','dbias') if self._has_bias else ()) + ('tangent','scratch')
            signature = inspect.Signature([inspect.Parameter(n,inspect.Parameter.POSITIONAL_ONLY) for n in names])
            self._jvp_binding = generate_tensor_binding(package,signature)
            return package.binding_digest

    def jvp(self, dq, dk, dv, *, dbias=None):
        """Execute directions against this captured O/LSE and optional bias."""
        with self._lock:
            self._ready()
            if self._jvp_binding is None:
                raise ValueError('resident attention JVP requires prepare_jvp')
            if (dbias is not None)!=self._has_bias:
                raise ValueError("resident JVP bias direction differs from its captured policy")
            if self._has_bias:
                self._resident(dbias,self._bias_shape)
            for value,shape in zip((dq,dk,dv),self.shapes[:3],strict=True):
                self._resident(value,shape)
            self.check(self.sync())  # The separate JVP binding consumes on its own stream.
            start = len(self.buffers)
            try:
                result = _Buffer(self,self.shapes[3])
                extra=(self._bias,dbias) if self._has_bias else ()
                self._jvp_binding(*self._saved,dq,dk,dv,*extra,result,128)
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
            if self._stream.value:
                self.check(self._stream_destroy(self._stream))
                self._stream = ct.c_void_p()
            self.closed = True

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self,*_exc):
        self.close()


class _ReadOnly:
    tessera_layout = "row_major"

    @property
    def dtype(self):
        return np.dtype(self._buffer.typestr)

    @property
    def shape(self):
        return self._buffer.shape

    def __init__(self,buffer):
        self._buffer = buffer

    @property
    def __cuda_array_interface__(self):
        interface = self._buffer.__cuda_array_interface__.copy()
        interface['data'] = (interface['data'][0],True)
        return interface
