"""Checked symbolic sequence capacities for native attention metadata."""
from __future__ import annotations
import math
from .native_artifact import ShapeGuard

# MLIR 23 ShapedType::kDynamic, preserved from the native sealed contract.
DYNAMIC_DIM = -(1 << 63)

def attention_dimensions(dims, bounds=(), actual=None):
    if not isinstance(dims,(tuple,list)) or len(dims)!=7 or any(type(x) is not int for x in dims):
        raise ValueError("attention shape requires seven integer dimensions")
    dynamic=any(x==DYNAMIC_DIM for x in dims)
    if any((x==DYNAMIC_DIM and i not in (3,4)) or (x!=DYNAMIC_DIM and x<=0) for i,x in enumerate(dims)):
        raise ValueError("only attention sequence dimensions may be dynamic")
    if dynamic:
        if (not isinstance(bounds,(tuple,list)) or len(bounds)!=7 or
                any(type(x) is not int or x<=0 for x in bounds) or
                any(x!=DYNAMIC_DIM and x!=cap for x,cap in zip(dims,bounds,strict=True))):
            raise ValueError("attention bounds must preserve positive fixed dimensions")
        caps=tuple(bounds)
    else:
        if bounds:raise ValueError("static attention has no sequence capacity policy")
        caps=tuple(dims)
    b,hq,hkv,sq,sk,d,dv=caps
    if hq%hkv:raise ValueError("attention head groups disagree")
    for shape in ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,dv),(b,hq,sq,sk)):
        if math.prod(shape)>((1<<63)-1)//4:
            raise ValueError("attention capacity exceeds byte-address ABI")
    if actual is None:return caps
    if (not isinstance(actual,(tuple,list)) or len(actual)!=7 or
            any(type(x) is not int or x<=0 for x in actual) or
            any((symbol!=DYNAMIC_DIM and x!=symbol) or x>cap
                for x,symbol,cap in zip(actual,dims,caps,strict=True))):
        raise ValueError("attention runtime dimensions exceed the compiled envelope")
    return tuple(actual)

def physical_attention_bias_shape(dims, physical=()):
    """Resolve the native broadcast policy against symbolic or actual extents."""
    b,hq,_,sq,sk,_,_=dims
    logical=(b,hq,sq,sk)
    if not isinstance(physical,(tuple,list)):
        raise ValueError("attention physical bias shape must be an integer sequence")
    if not physical:return logical
    if len(physical)!=4 or any(type(x) is not int for x in physical):
        raise ValueError("attention physical bias shape requires four integer dimensions")
    result=[]
    for axis,(extent,wanted) in enumerate(zip(physical,logical,strict=True)):
        if extent==DYNAMIC_DIM:
            if axis<2:
                raise ValueError("attention physical bias batch and heads must remain fixed")
            result.append(wanted)
        elif extent in (1,wanted) and extent>0:
            result.append(extent)
        else:
            raise ValueError("attention physical bias shape disagrees with logical dimensions")
    return tuple(result)

def attention_buffer_shapes(dims, *, backward=False,bias=False,bias_shape=(),
                            bias_gradient=False,lse_cotangent=False,gradient_roles=None):
    b,hq,hkv,sq,sk,d,dv=dims
    q,k,v=(b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv)
    o,lse=(b,hq,sq,dv),(b,hq,sq)
    physical=physical_attention_bias_shape(dims,bias_shape)
    inputs: list[tuple[int, ...]] = [o,q,k,v,o,lse] if backward else [q,k,v]
    outputs: list[tuple[int, ...]] = [q,k,v] if backward else [o,lse]
    if bias:inputs.insert(5 if backward else 3,physical)
    if lse_cotangent:inputs.insert(6+int(bias),lse)
    if bias_gradient:outputs.append(physical)
    if gradient_roles is not None:outputs=[outputs[i] for i in gradient_roles]
    return tuple(inputs+outputs)

def attention_guards(buffers,dims,bounds=(),**policies):
    caps=attention_dimensions(dims,bounds)
    symbolic=attention_buffer_shapes(dims,**policies)
    capacity=attention_buffer_shapes(caps,**policies)
    if len(buffers)!=len(symbolic):raise ValueError("attention buffer role count disagrees")
    guards: list[ShapeGuard] = []
    for binding,shape,limit in zip(buffers,symbolic,capacity,strict=True):
        for axis,(extent,cap) in enumerate(zip(shape,limit,strict=True)):
            if extent==DYNAMIC_DIM:
                guards.extend((ShapeGuard(binding.name,axis,"min",1),
                               ShapeGuard(binding.name,axis,"max",cap)))
            else:guards.append(ShapeGuard(binding.name,axis,"eq",extent))
    return tuple(guards)

def descriptor_attention_shapes(descriptor,actual=None):
    p=descriptor.provenance
    dims=p.get("shape");bounds=p.get("shape_bounds",())
    if (bool(bounds) != (p.get("shape_policy")=="bounded_sequences_v1") or
            ("shape_policy" in p and p["shape_policy"]!="bounded_sequences_v1")):
        raise ValueError("attention sequence capacity policy disagrees")
    concrete=attention_dimensions(dims,bounds,actual)
    if p.get("checkpoint_role") not in ("forward_save","backward_load"):
        raise ValueError("attention checkpoint direction is missing")
    backward=p.get("checkpoint_role")=="backward_load"
    policies=dict(backward=backward,bias=p.get("bias",False),
        bias_shape=p.get("bias_shape",()),bias_gradient=p.get("bias_gradient",False),
        lse_cotangent=p.get("lse_cotangent",False))
    if p.get("gradient_output")=="compact_v1":
        policies["gradient_roles"]=tuple(p.get("physical_gradient_roles",()))
    wanted=attention_guards(descriptor.buffers,dims,bounds,**policies)
    key=lambda g:(g.binding,g.dimension,g.predicate,g.value)
    if sorted(map(key,wanted))!=sorted(map(key,descriptor.shape_guards)):
        raise ValueError("attention guards disagree with the native shape envelope")
    return concrete,attention_buffer_shapes(concrete,**policies)
