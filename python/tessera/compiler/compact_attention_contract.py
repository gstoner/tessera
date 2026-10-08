"""Checked projection of the native compact saved-LSE attention ABI."""
from __future__ import annotations
import math

def compact_attention_contract(descriptor, *, seeded=False, runtime_dims=None):
    from .native_artifact import BufferBinding, ScalarArgument, ShapeGuard, LaunchGeometry, OrderingSemantics, WorkspaceRequirement
    p = descriptor.provenance
    dims = tuple(p.get("shape", ()))
    symbolic_dims=dims
    bounds=p.get("shape_bounds",())
    from .attention_shape_contract import attention_dimensions,attention_guards
    dims=attention_dimensions(symbolic_dims,bounds,runtime_dims)
    bias, bias_gradient = p.get("bias"), p.get("bias_gradient")
    activity = tuple(p.get("gradient_activity", ()))
    if (len(dims) != 7 or any(type(x) is not int or not 0 < x < (1 << 31) for x in dims)
            or type(bias) is not bool or type(bias_gradient) is not bool
            or (bias_gradient and not bias) or len(activity) != 3 + int(bias_gradient)
            or any(type(x) is not int or x not in (0, 1) for x in activity) or not any(activity)
            or p.get("gradient_output") != "compact_v1" or p.get("inactive_gradient") != "absent_v1"):
        raise ValueError("compact attention requires concrete native gradient roles")
    b,hq,hkv,sq,sk,d,dv = dims
    if hq % hkv:
        raise ValueError("compact attention grouped heads disagree")
    roles = tuple(i for i, x in enumerate(activity) if x)
    if tuple(p.get("physical_gradient_roles", ())) != roles:
        raise ValueError("compact physical output roles disagree")
    physical = tuple(p.get("bias_shape", ()))
    logical = (b,hq,sq,sk)
    from .attention_shape_contract import physical_attention_bias_shape
    if physical and not bias:
        raise ValueError("compact physical bias shape disagrees")
    physical_attention_bias_shape(symbolic_dims,physical)
    bias_shape = physical_attention_bias_shape(dims,physical)
    shapes: tuple[tuple[int, ...], ...] = ((b,hq,sq,dv), (b,hq,sq,d), (b,hkv,sk,d), (b,hkv,sk,dv), (b,hq,sq,dv))
    names: tuple[str, ...] = ("dO", "q", "k", "v", "output")
    if bias:
        shapes += (bias_shape,)
        names += ("bias",)
    shapes += ((b,hq,sq),)
    names += ("lse",)
    if seeded:
        if p.get("lse_cotangent") is not True:
            raise ValueError("compact row seed policy disagrees")
        shapes += ((b,hq,sq),)
        names += ("row_seed",)
    input_count = len(names)
    gradients = ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),bias_shape)
    shapes += tuple(gradients[i] for i in roles)
    names += tuple(("dq","dk","dv","dbias")[i] for i in roles)
    if any(math.prod(shape) > ((1 << 63)-1)//4 for shape in shapes):
        raise ValueError("compact attention allocation exceeds bounds")
    threads = p.get("gradient_block_threads")
    if type(threads) is not int or threads not in (64,128):
        raise ValueError("compact native thread count disagrees")
    elements = sum(math.prod(gradients[i]) for i in roles)
    if (elements+threads-1)//threads > (1 << 31)-1:
        raise ValueError("compact attention launch exceeds bounds")
    launch = p.get("gradient_launch")
    if launch not in ("packed_v1","logical_v1"):
        raise ValueError("compact attention launch layout disagrees")
    if launch == "logical_v1":
        elements = sum(math.prod(shape) for shape in gradients[:3+int(bias_gradient)])
        if (elements+threads-1)//threads > (1 << 31)-1:
            raise ValueError("compact logical attention launch exceeds bounds")
    mask = sum(x << i for i,x in enumerate(activity))
    digest = p.get("schedule_digest", "")
    if (not isinstance(digest,str) or len(digest) != 64 or
            any(c not in "0123456789abcdef" for c in digest) or
            descriptor.entry_symbol != (f"tessera_tile_attention_backward_lse_output_compact_m{mask}_b{int(bias)}_g{int(bias_gradient)}_l{int(launch == 'logical_v1')}_t{threads}_{digest[:10]}" + ("_cotangent_" if seeded else ""))):
        raise ValueError("compact entry symbol disagrees with native physical roles")
    if seeded:
        if len(descriptor.buffers)!=len(shapes):
            raise ValueError("compact seed physical role count disagrees")
        names = tuple(binding.name for binding in descriptor.buffers)
        if len(set(names))!=len(names):
            raise ValueError("compact seed buffer names must be distinct")
    expected = tuple(BufferBinding(i,name,"input" if i<input_count else "output","fp32",len(shape),"row_major",4)
        for i,(name,shape) in enumerate(zip(names,shapes,strict=True)))
    guards = tuple(ShapeGuard(name,axis,"eq",extent)
        for name,shape in zip(names,shapes,strict=True) for axis,extent in enumerate(shape))
    if bounds:
        guards=attention_guards(expected,symbolic_dims,bounds,backward=True,bias=bias,
            bias_shape=physical,bias_gradient=bias_gradient,lse_cotangent=seeded,
            gradient_roles=roles)
    scalar_names = ("B","Hq","Hkv","Sq","Sk","D","Dv") + (("BiasB","BiasH","BiasQ","BiasK") if physical else ())
    scalars = tuple(ScalarArgument(len(names)+i,name,"int64") for i,name in enumerate(scalar_names))
    if (descriptor.buffers != expected or descriptor.scalars != scalars
            or set(descriptor.shape_guards) != set(guards) or len(descriptor.shape_guards) != len(guards)
            or descriptor.geometry != LaunchGeometry(policy=f"sm120_attention_backward_lse_deterministic_direct_{threads}")
            or descriptor.workspace != WorkspaceRequirement(bytes=0,alignment=4)
            or descriptor.ordering != OrderingSemantics(ordered_submission=True,residency="none",synchronization=("completion",))
            or descriptor.dynamic_local_memory_bytes or descriptor.dynamic_local_memory_expression is not None):
        raise ValueError("compact attention descriptor differs from its native physical ABI")
    return dims + (bias_shape if physical else ()), shapes, input_count
