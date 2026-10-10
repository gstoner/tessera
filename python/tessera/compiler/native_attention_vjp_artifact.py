"""Pinned native saved-LSE reverse program; restoration needs no compiler."""
from __future__ import annotations
import hashlib
import json

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)

def payload(program):
    from .resident_attention import checkpoint_shapes
    dims,_=checkpoint_shapes(program.pair)
    p=program.pair.backward.descriptor.provenance
    count=3+int(program.pair.forward.descriptor.provenance.get("bias",False))
    mapping=program.input_indices
    active=program.active
    if (not isinstance(mapping,tuple) or len(mapping)!=count
            or any(type(i) is not int for i in mapping) or sorted(mapping)!=list(range(count))
            or any(tuple(x.descriptor.provenance.get("frontend_argument_indices",()))!=mapping
                   for x in (program.pair.forward,program.pair.backward))):
        raise ValueError("portable VJP frontend mapping disagrees with native checkpoint")
    if (not isinstance(active,tuple) or not active or len(set(active))!=len(active)
            or any(type(i) is not int or i not in range(count) for i in active)):
        raise ValueError("portable VJP requires unique requested physical roles")
    roles=tuple(p.get("physical_gradient_roles",()))
    activity=tuple(p.get("gradient_activity",()))
    expected=tuple(int(i in active) for i in range(3+int(p.get("bias_gradient",False))))
    if activity!=expected or not set(active)<=set(roles):
        raise ValueError("portable VJP activity disagrees with native output lineage")
    def package(x):
        if hashlib.sha256(x.target_ir.encode()).hexdigest()!=x.image.target_ir_digest:
            raise ValueError("portable VJP target IR differs from native image")
        if hashlib.sha256(x.tile_ir.encode()).hexdigest()!=x.descriptor.provenance.get("tile_ir_digest"):
            raise ValueError("portable VJP Tile IR differs from native descriptor")
        return dict(tile_ir=x.tile_ir,target_ir=x.target_ir,backend_ir=x.backend_ir,
                    image=x.image.to_dict(),descriptor=x.descriptor.to_dict())
    return dict(schema="tessera.native_attention_vjp_program.v1",
                forward=package(program.pair.forward),backward=package(program.pair.backward),
                checkpoint_digest=program.pair.contract_digest,
                active=list(active),frontend_argument_indices=list(mapping))

def to_json(program):
    data=payload(program)
    pin=hashlib.sha256(canonical(data).encode()).hexdigest()
    return canonical(dict(program=data,program_digest=pin))

def from_json(text,*,expected_digest):
    from .native_attention_program import NativeAttentionVJPProgram
    from .nvidia_native import NVIDIANativePackage,AttentionCheckpointPair
    from .native_artifact import NativeImageArtifact,LaunchDescriptor
    envelope=json.loads(text)
    if set(envelope)!={"program","program_digest"}:
        raise ValueError("portable VJP envelope fields disagree")
    data=envelope["program"]
    actual=hashlib.sha256(canonical(data).encode()).hexdigest()
    if actual!=expected_digest or actual!=envelope["program_digest"]:
        raise ValueError("portable VJP differs from pinned identity")
    fields={"schema","forward","backward","checkpoint_digest","active","frontend_argument_indices"}
    if not isinstance(data,dict) or set(data)!=fields or data.get("schema")!="tessera.native_attention_vjp_program.v1":
        raise ValueError("unsupported portable attention VJP schema")
    def package(raw):
        if set(raw)!={"tile_ir","target_ir","backend_ir","image","descriptor"}:
            raise ValueError("portable VJP package fields disagree")
        return NVIDIANativePackage(raw["tile_ir"],raw["target_ir"],raw["backend_ir"],
            NativeImageArtifact.from_dict(raw["image"]),LaunchDescriptor.from_dict(raw["descriptor"]))
    result=NativeAttentionVJPProgram(
        AttentionCheckpointPair(package(data["forward"]),package(data["backward"]),data["checkpoint_digest"]),
        tuple(data["active"]),tuple(data["frontend_argument_indices"]))
    payload(result)
    return result
