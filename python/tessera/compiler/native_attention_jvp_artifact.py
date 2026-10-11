"""Portable binding of native attention checkpoint and tangent images."""
from __future__ import annotations
import hashlib
import json
import re

def payload(program):
    from .resident_attention import checkpoint_shapes
    from .native_storage_contract import read_tensor_contract
    dims,shapes=checkpoint_shapes(program.pair)
    p=program.pair.forward.descriptor.provenance
    biased=p.get("bias",False)
    count=3+int(biased)
    mapping=program.input_indices or tuple(range(count))
    native_mapping=tuple(program.pair.forward.descriptor.provenance.get("frontend_argument_indices",()))
    if (len(mapping)!=count or any(type(i) is not int for i in mapping) or sorted(mapping)!=list(range(count))
            or mapping!=native_mapping):
        raise ValueError("portable JVP frontend mapping disagrees with native checkpoint")
    backward = getattr(program.pair, "backward", None)
    if backward is not None and tuple(backward.descriptor.provenance.get("frontend_argument_indices", ())) != mapping:
        raise ValueError("portable JVP reverse frontend mapping disagrees with native checkpoint")
    names=program.input_names
    if names and (not isinstance(names,tuple) or len(names)!=count or len(set(names))!=count or
            any(not isinstance(n,str) or not n.isidentifier() for n in names)):
        raise ValueError("portable JVP frontend parameter names disagree")
    if not names and mapping!=tuple(range(count)):
        raise ValueError("portable JVP reordered capture requires frontend parameter names")
    active=program.active
    if (not isinstance(active,tuple) or not active or len(set(active))!=len(active) or
            any(type(i) is not int or i not in range(count) for i in active)):
        raise ValueError("portable JVP requires unique physical tangent roles")
    tangent=program.tangent
    tangent.validate()
    if (tangent.backend,tangent.chip)!=("nvidia","sm_120"):
        raise ValueError("portable JVP tangent target differs from its checkpoint")
    identity=re.findall(r'tessera.attention_checkpoint_identity = "([0-9a-f]{64})"',tangent.arena_ir)
    if identity!=[program.pair.contract_digest]:
        raise ValueError("portable JVP forward/tangent generations disagree")
    contract=re.findall(r'tessera.attention_jvp_contract = \{([^\n]*?)\}',tangent.arena_ir)
    activity="active = ["+", ".join(str(i in active).lower() for i in range(count))+"]"
    if len(contract)!=1 or activity not in contract[0]:
        raise ValueError("portable JVP tangent activity disagrees")
    bias_shape=tuple(p.get("bias_shape",())) or (dims[0],dims[1],dims[3],dims[4])
    bounds=tuple(p.get("shape_bounds",()))
    if bounds:
        from .attention_shape_contract import DYNAMIC_DIM
        symbolic=tuple(p["shape"])
        b,hq,hkv,sq,sk,d,dv=symbolic
        sq="query_size" if sq==DYNAMIC_DIM else sq
        sk="key_size" if sk==DYNAMIC_DIM else sk
        shapes=((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,dv),(b,hq,sq))
        symbolic_bias=tuple(p.get("bias_shape",())) or (b,hq,symbolic[3],symbolic[4])
        bias_shape=tuple(("query_size" if axis==2 else "key_size") if x==DYNAMIC_DIM else x
                         for axis,x in enumerate(symbolic_bias))
    expected_shapes=(*shapes[:3],shapes[3],shapes[4],*shapes[:3],
        *((bias_shape,bias_shape) if biased else ()),shapes[3])
    tensor_names=("q","k","v","primal","lse","dq","dk","dv") + (
        ("bias","dbias") if biased else ()) + ("tangent",)
    expected=[dict(kind="tensor",name=n,dtype="fp32",shape=list(s),writable=i==len(tensor_names)-1)
              for i,(n,s) in enumerate(zip(tensor_names,expected_shapes,strict=True))]
    expected.append(dict(kind="index",name="scratch",minimum=128,maximum=128))
    grid=[dims[0]*dims[1]*dims[3],1,1]
    if bounds:
        from .attention_shape_contract import DYNAMIC_DIM
        for axis,name in ((3,"query_size"),(4,"key_size")):
            expected.append(dict(kind="index",name=name,
                minimum=1 if symbolic[axis]==DYNAMIC_DIM else symbolic[axis],maximum=bounds[axis]))
        grid=[dict(product=[dims[0],dims[1],"query_size"]),1,1]
    manifest=read_tensor_contract(tangent)
    if manifest!={"schema":2 if bounds else 1,"arguments":expected,"grid":grid,"block":[128,1,1]}:
        raise ValueError("portable JVP native tensor manifest differs from checkpoint")
    if tangent.abi!=("pointer",)*len(tensor_names)+("index",)*(3 if bounds else 1):
        raise ValueError("portable JVP pointer/scalar ABI differs")
    def package(x):
        return dict(tile_ir=x.tile_ir,target_ir=x.target_ir,backend_ir=x.backend_ir,
                    image=x.image.to_dict(),descriptor=x.descriptor.to_dict())
    from .nvidia_native import AttentionCheckpointPair, AttentionForwardCheckpoint
    result = dict(forward=package(program.pair.forward),
                  checkpoint_digest=program.pair.contract_digest,tangent=json.loads(tangent.to_json()),
                  active=list(active),frontend_argument_indices=list(mapping),frontend_parameter_names=list(names))
    if isinstance(program.pair, AttentionCheckpointPair):
        result.update(schema="tessera.native_attention_jvp_program.v1", backward=package(program.pair.backward))
    elif isinstance(program.pair, AttentionForwardCheckpoint):
        result.update(schema="tessera.native_attention_jvp_program.v2")
    else:
        raise ValueError("portable JVP requires a native checkpoint product")
    return result

def canonical(data):
    return json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False)

def digest(program):
    return hashlib.sha256(canonical(payload(program)).encode()).hexdigest()

def to_json(program):
    data=payload(program)
    pin=hashlib.sha256(canonical(data).encode()).hexdigest()
    return canonical(dict(program=data,program_digest=pin))

def from_json(text,*,expected_digest):
    from .native_attention_program import NativeAttentionJVPProgram
    from .native_gpu_storage import NativeGPUStoragePackage
    from .nvidia_native import NVIDIANativePackage,AttentionCheckpointPair,AttentionForwardCheckpoint
    from .native_artifact import NativeImageArtifact,LaunchDescriptor
    envelope=json.loads(text)
    if set(envelope)!={"program","program_digest"}:
        raise ValueError("portable JVP envelope fields disagree")
    data=envelope["program"]
    actual=hashlib.sha256(canonical(data).encode()).hexdigest()
    if actual!=expected_digest or actual!=envelope["program_digest"]:
        raise ValueError("portable JVP differs from pinned identity")
    fields={"schema","forward","backward","checkpoint_digest","tangent","active","frontend_argument_indices","frontend_parameter_names"}
    if not isinstance(data,dict):
        raise ValueError("unsupported portable attention JVP schema")
    schema = data.get("schema")
    if schema == "tessera.native_attention_jvp_program.v2":
        fields.remove("backward")
    elif schema != "tessera.native_attention_jvp_program.v1":
        raise ValueError("unsupported portable attention JVP schema")
    if set(data) != fields:
        raise ValueError("unsupported portable attention JVP schema")
    def package(raw):
        if set(raw)!={"tile_ir","target_ir","backend_ir","image","descriptor"}:
            raise ValueError("portable JVP image fields disagree")
        return NVIDIANativePackage(raw["tile_ir"],raw["target_ir"],raw["backend_ir"],
            NativeImageArtifact.from_dict(raw["image"]),LaunchDescriptor.from_dict(raw["descriptor"]))
    tangent=data["tangent"]
    checkpoint = (AttentionCheckpointPair(package(data["forward"]),package(data["backward"]),data["checkpoint_digest"])
                  if schema == "tessera.native_attention_jvp_program.v1" else
                  AttentionForwardCheckpoint(package(data["forward"]),data["checkpoint_digest"]))
    result=NativeAttentionJVPProgram(
        checkpoint,
        NativeGPUStoragePackage.from_json(canonical(tangent),expected_digest=tangent["binding_digest"]),
        tuple(data["active"]),tuple(data["frontend_argument_indices"]),tuple(data["frontend_parameter_names"]))
    payload(result)
    return result
