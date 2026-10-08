"""Checked physical roles for native saved-LSE output cotangents."""
from __future__ import annotations

import math


def lse_cotangent_contract(descriptor, *, runtime_dims=None):
    from .native_artifact import BufferBinding, ScalarArgument, ShapeGuard, LaunchGeometry, OrderingSemantics, WorkspaceRequirement
    p = descriptor.provenance
    dims = tuple(p.get("shape", ()))
    symbolic_dims=dims
    bounds=p.get("shape_bounds",())
    from .attention_shape_contract import attention_dimensions,attention_guards
    dims=attention_dimensions(symbolic_dims,bounds,runtime_dims)
    bias = p.get("bias")
    bias_gradient = p.get("bias_gradient")
    if (len(dims) != 7 or any(type(x) is not int or not 0 < x < (1 << 31) for x in dims)
            or type(bias) is not bool or p.get("lse_cotangent") is not True
            or type(bias_gradient) is not bool or (bias_gradient and not bias)
            or p.get("route") != "deterministic_direct" or p.get("deterministic") is not True
            or p.get("lse_checkpoint") != "saved" or p.get("storage") != "f32"
            or p.get("gradient_storage") != "f32"):
        raise ValueError("seeded attention requires verified native full-gradient roles")
    for key in ("graph_ir_digest","schedule_ir_digest","tile_ir_digest","target_ir_digest"):
        value = p.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("seeded attention is missing compiler ancestry")
    scale,causal = p.get("scale"),p.get("causal")
    if (type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0 or type(causal) is not bool
            or p.get("checkpoint_role")!="backward_load" or p.get("mask_alignment")!="end_aligned_v1"
            or p.get("row_lse")!="f32[B,Hq,Sq]" or p.get("workspace_bytes")!=0):
        raise ValueError("seeded attention numerical checkpoint policy disagrees")
    from .nvidia_native import _checkpoint_identity
    physical = tuple(p.get("bias_shape", ()))
    identity = _checkpoint_identity(symbolic_dims,scale,causal,bias=bias,bias_shape=physical,shape_bounds=tuple(bounds))
    if p.get("checkpoint_contract")!=identity:
        raise ValueError("seeded attention numerical identity differs from saved checkpoint")
    if p.get("bias_gradient_reduction")!=("physical_owner_lexicographic_bhqk_v1" if physical else "none"):
        raise ValueError("seeded physical bias reduction policy disagrees")
    if p.get("gradient_output") == "compact_v1":
        from .compact_attention_contract import compact_attention_contract
        return compact_attention_contract(descriptor, seeded=True, runtime_dims=runtime_dims)
    if (p.get("gradient_output") != "complete_v1" or p.get("gradient_launch") != "logical_v1"
            or p.get("gradient_block_threads") != 128
            or p.get("physical_gradient_roles") != list(range(3+int(bias_gradient)))):
        raise ValueError("seeded full-gradient physical roles disagree")
    b,hq,hkv,sq,sk,d,dv = dims
    if hq % hkv:
        raise ValueError("seeded attention grouped heads disagree")
    physical = tuple(p.get("bias_shape", ()))
    logical = (b,hq,sq,sk)
    from .attention_shape_contract import physical_attention_bias_shape
    if physical and not bias:
        raise ValueError("seeded attention bias envelope disagrees")
    physical_attention_bias_shape(symbolic_dims,physical)
    bias_shape=physical_attention_bias_shape(dims,physical)
    shapes: tuple[tuple[int, ...], ...] = ((b,hq,sq,dv),(b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,dv))
    if bias:
        shapes += (bias_shape,)
    shapes += ((b,hq,sq),(b,hq,sq))
    input_count = len(shapes)
    shapes += ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv))
    if bias_gradient:
        shapes += (bias_shape,)
    if any(math.prod(shape) > ((1 << 63)-1)//4 for shape in shapes):
        raise ValueError("seeded attention allocation exceeds bounds")
    if (sum(math.prod(shape) for shape in shapes[input_count:])+127)//128 > (1 << 31)-1:
        raise ValueError("seeded attention launch exceeds bounds")
    digest = p.get("schedule_digest", "")
    prefix = "tessera_tile_attention_backward_lse_output_" + ("bias_gradient_" if bias_gradient else "")
    if (not isinstance(digest,str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            or descriptor.entry_symbol != f"{prefix}cotangent_{digest[:10]}"):
        raise ValueError("seeded entry differs from sealed native Schedule")
    bindings = descriptor.buffers
    if len(bindings)!=len(shapes) or len({x.name for x in bindings})!=len(bindings):
        raise ValueError("seeded attention buffer role count disagrees")
    names = tuple(x.name for x in bindings)
    expected = tuple(BufferBinding(i,name,"input" if i<input_count else "output","fp32",len(shape),"row_major",4)
                     for i,(name,shape) in enumerate(zip(names,shapes,strict=True)))
    guards = tuple(ShapeGuard(name,axis,"eq",extent) for name,shape in zip(names,shapes,strict=True)
                   for axis,extent in enumerate(shape))
    if bounds:
        guards=attention_guards(expected,symbolic_dims,bounds,backward=True,bias=bias,
            bias_shape=physical,bias_gradient=bias_gradient,lse_cotangent=True)
    scalar_names = ("B","Hq","Hkv","Sq","Sk","D","Dv") + (("BiasB","BiasH","BiasQ","BiasK") if physical else ())
    scalars = tuple(ScalarArgument(len(names)+i,name,"int64") for i,name in enumerate(scalar_names))
    if (bindings!=expected or descriptor.scalars!=scalars or len(descriptor.shape_guards)!=len(guards)
            or set(descriptor.shape_guards)!=set(guards)
            or descriptor.geometry!=LaunchGeometry(policy="sm120_attention_backward_lse_deterministic_direct_128")
            or descriptor.workspace!=WorkspaceRequirement(bytes=0,alignment=4)
            or descriptor.ordering!=OrderingSemantics(ordered_submission=True,residency="none",synchronization=("completion",))
            or descriptor.dynamic_local_memory_bytes or descriptor.dynamic_local_memory_expression is not None):
        raise ValueError("seeded descriptor differs from native physical ABI")
    return dims+(bias_shape if physical else ()), shapes, input_count


def validate_lse_cotangent_invocation(descriptor, buffers, scalars):
    import numpy as np
    actual=tuple(scalars[name] for name in ("B","Hq","Hkv","Sq","Sk","D","Dv")) if descriptor.provenance.get("shape_bounds") else None
    dims,shapes,input_count = lse_cotangent_contract(descriptor,runtime_dims=actual)
    supplied = tuple(scalars[x.name] for x in descriptor.scalars)
    if any(type(x) is not int for x in supplied) or supplied!=dims:
        raise ValueError("seeded attention scalars differ from the compiled envelope")
    spans = []
    for binding,shape in zip(descriptor.buffers,shapes,strict=True):
        value = buffers[binding.name]
        interface = getattr(value,"__cuda_array_interface__",None)
        if interface is not None:
            actual_shape = tuple(interface.get("shape",()))
            dtype = np.dtype(interface.get("typestr"))
            strides = interface.get("strides")
            address = int(interface["data"][0])
            readonly = bool(interface["data"][1])
        else:
            actual_shape = tuple(value.shape)
            dtype = value.dtype
            strides = value.strides
            address = int(value.ctypes.data)
            readonly = not value.flags.writeable
        if actual_shape!=shape or dtype!=np.dtype(np.float32):
            raise ValueError("seeded attention storage differs from the compiled envelope")
        expected = tuple(4*math.prod(shape[i+1:]) for i in range(len(shape)))
        if strides is not None and any(n>1 and a!=e for n,a,e in zip(shape,strides,expected,strict=True)):
            raise ValueError("seeded attention requires contiguous native storage")
        if not address or address%4 or (binding.direction=="output" and readonly):
            raise ValueError("seeded attention requires aligned writable output storage")
        spans.append((address,address+4*math.prod(shape)))
    # Inputs may share read-only storage; outputs cannot alias any other role.
    for i,(lo,hi) in enumerate(spans):
        for j,(other_lo,other_hi) in enumerate(spans[:i]):
            if (i>=input_count or j>=input_count) and lo<other_hi and other_lo<hi:
                raise ValueError("seeded attention output aliases another physical role")
    return dims,shapes,input_count
