"""Native row-product metadata packaging and compiler-free portable validation.

All arithmetic and Schedule/Tile creation stay in the production MLIR compiler.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
import re
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nvidia_native import AttentionCheckpointPair, NVIDIANativePackage

from .native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
    NativeImageArtifact, OrderingSemantics, ResourceRecord, ScalarArgument,
    ShapeGuard, WorkspaceRequirement,
)
from .typed_attention_backward import _native_dictionary

ROW_ABI = "tessera.nvidia.attention.row_correction.member.v1"
BACKWARD_ROWS_ABI = "tessera.nvidia.attention.backward_rows.member.v1"
_DTYPE = {"f16": "fp16", "bf16": "bf16", "f32": "fp32"}
_SCALARS = ("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parts(text):
    parts, start, depth, quoted, escaped = [], 0, 0, False, False
    for i, c in enumerate(text):
        if escaped:
            escaped = False
        elif quoted and c == "\\":
            escaped = True
        elif c == '"':
            quoted = not quoted
        elif not quoted:
            if c in "[<{(": depth += 1
            elif c in "]>})": depth -= 1
            elif c == "," and depth == 0:
                parts.append(text[start:i].strip()); start = i+1
        if depth < 0: raise ValueError("row native dictionary delimiters disagree")
    if quoted or depth: raise ValueError("row native dictionary is truncated")
    parts.append(text[start:].strip())
    return parts


def contract(tile):
    raw = _native_dictionary(tile)
    result = {}
    for item in _parts(raw[1:-1]):
        key, separator, value = item.partition(" = ")
        if not separator or key in result:
            raise ValueError("row native dictionary keys disagree")
        if value in ("true", "false"):
            parsed = value == "true"
        elif value.startswith('"') or value.startswith("["):
            parsed = json.loads(value)
        elif value.startswith("array<i64:") and value.endswith(">"):
            parsed = [int(x) for x in value[len("array<i64:"):-1].split(",") if x.strip()]
        elif value.endswith(" : i64"):
            parsed = int(value[:-6])
        elif value.endswith(" : f32"):
            spelling = value[:-6]
            parsed = (struct.unpack(">d", int(spelling, 16).to_bytes(8, "big"))[0]
                      if spelling.startswith("0x") else float(spelling))
            parsed = struct.unpack("!f", struct.pack("!f", parsed))[0]
        else:
            raise ValueError("unsupported row native dictionary value: " + key)
        result[key] = parsed
    return raw, result


_WITNESS_CACHE_ENTRIES = 32
_WITNESS_CACHE_BYTE_LIMIT = 256*1024
_PAIR_WITNESS = re.compile(r'tessera\.attention_ad_pair = "((?:[^"\\]|\\.)*)"')


def _decode_pair_witness(encoded: str) -> str:
    data = bytearray(); index = 0
    while index < len(encoded):
        if encoded[index] != "\\":
            data.extend(encoded[index].encode()); index += 1
        elif re.fullmatch(r"[0-9a-fA-F]{2}", encoded[index+1:index+3]):
            data.append(int(encoded[index+1:index+3], 16)); index += 3
        elif encoded[index+1:index+2] in ('"', "\\"):
            data.extend(encoded[index+1].encode()); index += 2
        else:
            raise ValueError("row paired AD witness has unsupported MLIR escaping")
    return hashlib.sha256(data).hexdigest()



_WITNESS_DIGEST_CACHE = lru_cache(maxsize=_WITNESS_CACHE_ENTRIES)(_decode_pair_witness)
_WITNESS_CACHE_PROCESS = os.getpid()


def pair_witness(text):
    # Literal attribute name, with no match inside another qualified name.
    found = [match.group(1) for match in _PAIR_WITNESS.finditer(text)
             if match.start() == 0 or not (
                 text[match.start()-1].isalnum() or text[match.start()-1] in "_.$-")]
    if len(found) != 1:
        raise ValueError("row program requires one retained native paired AD witness")
    encoded = found[0]
    global _WITNESS_DIGEST_CACHE, _WITNESS_CACHE_PROCESS
    process = os.getpid()
    if process != _WITNESS_CACHE_PROCESS:
        # Replace the CPU-only cache before accessing inherited cache state.
        _WITNESS_DIGEST_CACHE = lru_cache(maxsize=_WITNESS_CACHE_ENTRIES)(_decode_pair_witness)
        _WITNESS_CACHE_PROCESS = process
    if (len(encoded) > _WITNESS_CACHE_BYTE_LIMIT or
            (not encoded.isascii() and len(encoded.encode("utf-8")) > _WITNESS_CACHE_BYTE_LIMIT)):
        return _decode_pair_witness(encoded)
    # Only immutable encoded source -> immutable digest is memoized. All
    # package/provenance/ABI/shape/device checks still execute for each call.
    return _WITNESS_DIGEST_CACHE(encoded)


def _facts(c, backward, runtime_dims=None):
    common = {"accum", "arch", "arguments", "bias", "block_threads", "causal", "family",
              "frontend_argument_indices", "gradient_activity", "lse_cotangent", "mask_alignment",
              "output_storage", "results", "row_bytes", "row_correction", "row_storage", "scale",
              "shape", "source_ad_pair_digest", "storage", "target"}
    if c.get("bias") is True: common.add("bias_shape")
    if "shape_bounds" in c or "shape_policy" in c: common |= {"shape_bounds", "shape_policy"}
    if backward: common |= {"bias_gradient", "gradient_output", "gradient_launch", "row_correction_usage"}
    if set(c) != common:
        raise ValueError("row native policy fields disagree")
    for key in ("bias", "causal", "lse_cotangent"):
        if type(c[key]) is not bool: raise ValueError("row native boolean policy disagrees")
    if (c["family"] != ("attention_backward_rows" if backward else "attention_row_correction") or
            c["target"] != "nvidia_sm120" or c["arch"] != "sm_120" or
            c["storage"] not in ("f16", "bf16") or c["output_storage"] not in (c["storage"], "f32") or
            c["accum"] != "f32" or c["row_storage"] != "f32" or
            c["mask_alignment"] != "end_aligned_v1" or c["row_correction"] != "probability_dot_v_f32_v1"):
        raise ValueError("row native storage/numerical policy disagrees")
    from .attention_shape_contract import attention_dimensions, physical_attention_bias_shape
    bounds = c.get("shape_bounds", ())
    if (bool(bounds) != (c.get("shape_policy") == "bounded_sequences_v1") or
            ("shape_policy" in c and c["shape_policy"] != "bounded_sequences_v1")):
        raise ValueError("row native sequence capacity policy disagrees")
    capacity = attention_dimensions(c["shape"], bounds)
    shape = attention_dimensions(c["shape"], bounds, runtime_dims)
    if any(x >= 2**31 for x in capacity):
        raise ValueError("row native dimensions disagree")
    b, hq, hkv, sq, sk, d, dv = shape
    if (type(c["row_bytes"]) is not int or c["row_bytes"] != math.prod(capacity[:2])*capacity[3]*4 or
            type(c["block_threads"]) is not int or
            c["block_threads"] not in ((64, 128) if backward else (32,))):
        raise ValueError("row native byte span/thread policy disagrees")
    scale = c["scale"]
    if type(scale) is not float or not math.isfinite(scale) or scale <= 0:
        raise ValueError("row native scale disagrees")
    mapping, activity = c["frontend_argument_indices"], c["gradient_activity"]
    count = 3+int(c["bias"])
    if (not isinstance(mapping, list) or len(mapping) != count or
            any(type(x) is not int for x in mapping) or sorted(mapping) != list(range(count)) or
            not isinstance(activity, list) or len(activity) != count or
            any(type(x) is not int or x not in (0, 1) for x in activity) or not any(activity)):
        raise ValueError("row native mapping/activity disagrees")
    digest = c["source_ad_pair_digest"]
    if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
        raise ValueError("row native paired lineage is missing")
    q, k, v, out, row = (b,hq,sq,d), (b,hkv,sk,d), (b,hkv,sk,dv), (b,hq,sq,dv), (b,hq,sq)
    shapes: list[tuple[int, ...]] = [out,q,k,v]
    names = ["dO","q","k","v"]
    storages = [c["output_storage"]] + [c["storage"]]*3
    bias_shape = c.get("bias_shape", [])
    if c["bias"]:
        if not isinstance(bias_shape, list) or len(bias_shape) != 4:
            raise ValueError("row physical bias shape disagrees")
        physical_attention_bias_shape(c["shape"], bias_shape)
        bias_shape = list(physical_attention_bias_shape(shape, bias_shape))
        shapes.append(tuple(bias_shape)); names.append("bias"); storages.append("f32")
    shapes.append(row); names.append("lse"); storages.append("f32")
    if backward:
        if (c["bias_gradient"] is not c["bias"] or c["gradient_output"] != "compact_v1" or
                c["gradient_launch"] not in ("packed_v1", "logical_v1") or
                c["row_correction_usage"] != "external_input_v1"):
            raise ValueError("row consumer compact/residual policy disagrees")
        names.append("row_correction"); shapes.append(row); storages.append("f32")
        result_names = ["dq","dk","dv"] + (["dbias"] if c["bias"] else [])
        result_shapes = [q,k,v] + ([tuple(bias_shape)] if c["bias"] else [])
        roles = tuple(i for i,x in enumerate(activity) if x)
        output_names = [result_names[i] for i in roles]
        output_shapes = [result_shapes[i] for i in roles]
        output_storages = [c["storage"] if i < 3 else "f32" for i in roles]
    else:
        if c["lse_cotangent"]:
            shapes.append(row); names.append("row_seed"); storages.append("f32")
        result_names = ["row_correction"]; output_names = result_names
        output_shapes = [row]; output_storages = ["f32"]; roles = ()
    if c["arguments"] != names or c["results"] != result_names:
        raise ValueError("row native semantic binding names disagree")
    if any(math.prod(x)*(4 if t == "f32" else 2) > 2**63-1
           for x,t in zip(shapes+output_shapes,storages+output_storages,strict=True)):
        raise ValueError("row native physical byte span overflows")
    return names,shapes,storages,output_names,output_shapes,output_storages,roles


def _entry(c, digest, backward):
    if not backward:
        return (f'tessera_tile_attention_row_correction_b{int(c["bias"])}_s{int(c["lse_cotangent"])}_'
                f'{c["storage"]}_out_{c["output_storage"]}_{digest[:10]}')
    mask = sum(1 << i for i,x in enumerate(c["gradient_activity"]) if x)
    return (f'tessera_tile_attention_backward_rows_compact_m{mask}_b{int(c["bias"])}_g{int(c["bias"])}_'
            f'l{int(c["gradient_launch"] == "logical_v1")}_t{c["block_threads"]}_'
            f'{c["storage"]}_out_{c["output_storage"]}_{digest[:10]}' +
            ('_cotangent_' if c["lse_cotangent"] else ''))


def _descriptor(image, c, tile, target, backward):
    from .nvidia_native import _checkpoint_identity
    names, shapes, types, outputs, out_shapes, out_types, roles = _facts(c,backward)
    digest = hashlib.sha256(_native_dictionary(tile).encode()).hexdigest()
    buffers = tuple(BufferBinding(i,name,"input" if i < len(names) else "output",_DTYPE[t],len(shape),"row_major",4 if t == "f32" else 2)
                    for i,(name,shape,t) in enumerate(zip(names+outputs,shapes+out_shapes,types+out_types,strict=True)))
    bounds = c.get("shape_bounds", ())
    symbolic = c if not bounds else {**c, "shape": list(bounds)}
    # The compiler owns role order and the query/key sequence relationships.
    # Preserve physical extent-one broadcast axes when deriving guards.
    if bounds:
        from .attention_shape_contract import DYNAMIC_DIM
        q, k = c["shape"][3:5]
        sq_symbol = DYNAMIC_DIM if q == DYNAMIC_DIM else q
        sk_symbol = DYNAMIC_DIM if k == DYNAMIC_DIM else k
        b,hq,hkv,_,_,d,dv = bounds
        row = (b,hq,sq_symbol)
        role_shapes = {"dO": (b,hq,sq_symbol,dv), "q": (b,hq,sq_symbol,d),
            "k": (b,hkv,sk_symbol,d), "v": (b,hkv,sk_symbol,dv),
            "lse": row, "row_seed": row, "row_correction": row,
            "dq": (b,hq,sq_symbol,d), "dk": (b,hkv,sk_symbol,d),
            "dv": (b,hkv,sk_symbol,dv), "bias": tuple(c.get("bias_shape", ())),
            "dbias": tuple(c.get("bias_shape", ()))}
        symbolic_shapes = [role_shapes[name] for name in names+outputs]
        guards = tuple(guard for name, shape, cap in zip(names+outputs,symbolic_shapes,shapes+out_shapes,strict=True)
            for axis,(extent,limit) in enumerate(zip(shape,cap,strict=True))
            for guard in ((ShapeGuard(name,axis,"min",1), ShapeGuard(name,axis,"max",limit))
                          if extent == DYNAMIC_DIM else (ShapeGuard(name,axis,"eq",extent),)))
    else:
        guards = tuple(ShapeGuard(name,axis,"eq",extent) for name,shape in zip(names+outputs,shapes+out_shapes,strict=True)
                       for axis,extent in enumerate(shape))
    b,hq,hkv,sq,sk,d,dv = symbolic["shape"]
    count = (sum(math.prod(x) for x in out_shapes) if c.get("gradient_launch") != "logical_v1" else
             b*hq*sq*d+b*hkv*sk*(d+dv)+(math.prod(shapes[names.index("bias")]) if c["bias"] else 0)) if backward else b*hq*sq
    threads = c["block_threads"]
    if count > (2**31-1)*threads: raise ValueError("row launch grid exceeds native capacity")
    identity = _checkpoint_identity(tuple(c["shape"]),c["scale"],c["causal"],bias=c["bias"],
        bias_shape=tuple(c.get("bias_shape",())),shape_bounds=tuple(c.get("shape_bounds",())),storage=c["storage"],output_storage=c["output_storage"])
    return LaunchDescriptor(image_digest=image.image_digest,entry_symbol=_entry(c,digest,backward),
        abi_id=BACKWARD_ROWS_ABI if backward else ROW_ABI,buffers=buffers,
        scalars=tuple(ScalarArgument(len(buffers)+i,name,"int64") for i,name in enumerate(_SCALARS)),
        shape_guards=guards,geometry=LaunchGeometry(grid=((count+threads-1)//threads,1,1),workgroup=(threads,1,1)),
        workspace=WorkspaceRequirement(bytes=0,alignment=4),
        ordering=OrderingSemantics(ordered_submission=True,residency="none",synchronization=("completion",)),
        provenance=dict(row_contract=c,schedule_digest=digest,tile_ir_digest=hashlib.sha256(tile.encode()).hexdigest(),
            target_ir_digest=hashlib.sha256(target.encode()).hexdigest(),source_ad_pair_digest=c["source_ad_pair_digest"],
            checkpoint_contract=identity,frontend_argument_indices=c["frontend_argument_indices"],
            gradient_activity=c["gradient_activity"],physical_gradient_roles=list(roles),bias=c["bias"],
            bias_gradient=c["bias"] if backward else False,lse_cotangent=c["lse_cotangent"],
            storage=c["storage"],output=c["output_storage"],shape=c["shape"],scale=c["scale"],causal=c["causal"],
            **({"shape_bounds": list(bounds), "shape_policy": c["shape_policy"]} if bounds else {}),
            bias_shape=c.get("bias_shape",[]),checkpoint_role="backward_rows_load" if backward else "row_correction_save"))


def validate_member(package, backward):
    image=NativeImageArtifact.from_dict(package.image.to_dict())
    descriptor=LaunchDescriptor.from_dict(package.descriptor.to_dict())
    descriptor.validate_image(image)
    if (image.target,image.architecture,image.binary_format) != ("nvidia_sm120","sm_120a","ptx"):
        raise ValueError("row member requires owning SM120 PTX")
    if (hashlib.sha256(package.target_ir.encode()).hexdigest()!=image.target_ir_digest or
            package.backend_ir.encode()!=image.payload):
        raise ValueError("row member target/backend image provenance disagrees")
    raw,c=contract(package.tile_ir)
    if (_native_dictionary(package.target_ir)!=raw or
            pair_witness(package.tile_ir)!=c["source_ad_pair_digest"] or
            pair_witness(package.target_ir)!=c["source_ad_pair_digest"]):
        raise ValueError("row member source pair and retained native contracts disagree")
    expected=_descriptor(image,c,package.tile_ir,package.target_ir,backward)
    digest=expected.provenance["schedule_digest"]
    if image.entry_points!=(NativeEntryPoint(expected.entry_symbol,expected.abi_id),):
        raise ValueError("row member image entry set differs from its sealed ABI")
    if (re.findall(r'tessera.schedule_hash = "([a-f0-9]{64})"',package.tile_ir)!=[digest] or
            re.findall(r'llvm.func @([\w]+)\(',package.tile_ir)!=[expected.entry_symbol] or
            _canonical(expected.to_dict())!=_canonical(descriptor.to_dict())):
        raise ValueError("row member descriptor differs from sealed native policy/roles")
    return c


def package_member(source, *, backward, threads=128, launch="packed_v1"):
    from .nvidia_native import NVIDIANativePackage, _compile_tile_ir
    from .scheduled_matmul import find_tessera_opt, run_tessera_opt
    tool=find_tessera_opt()
    if tool is None: raise RuntimeError("row member requires production compiler")
    flags="--tessera-autodiff-paired=checkpoint-product="+("backward_rows" if backward else "row_correction")
    flags+=" prune-checkpoint-gradients=true"
    if backward:
        if type(threads) is not int or threads not in (64,128) or launch not in ("packed_v1","logical_v1"):
            raise ValueError("row member requires native compact launch geometry")
        flags+=f" compact-checkpoint-gradients=true compact-checkpoint-threads={threads} compact-checkpoint-launch={launch}"
    graph=run_tessera_opt(tool,source,flags)
    schedule=run_tessera_opt(tool,graph,"--tessera-graph-to-schedule")
    tile=run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
    _,c=contract(tile);_facts(c,backward)
    digest=hashlib.sha256(_native_dictionary(tile).encode()).hexdigest();entry=_entry(c,digest,backward)
    target,ptx,metrics,compiler_fp,toolchain_fp,libraries,state=_compile_tile_ir(tile,entry)
    abi=BACKWARD_ROWS_ABI if backward else ROW_ABI
    image=NativeImageArtifact(target="nvidia_sm120",architecture="sm_120a",pipeline_name="tessera-nvidia-pipeline-sm120",
        compiler_fingerprint=compiler_fp,toolchain_fingerprint=toolchain_fp,target_ir_digest=hashlib.sha256(target.encode()).hexdigest(),
        binary_format="ptx",payload=ptx.encode(),entry_points=(NativeEntryPoint(entry,abi),),compile_state=state,
        device_libraries=libraries,resource_record=ResourceRecord(provenance="ptxas --arch=sm_120a -v",metrics=metrics))
    package=NVIDIANativePackage(tile,target,ptx,image,_descriptor(image,c,tile,target,backward))
    validate_member(package,backward)
    return package


@dataclass(frozen=True)
class NativeAttentionRowsProgram:
    pair: AttentionCheckpointPair
    correction: NVIDIANativePackage | None
    active: tuple[int,...]
    input_indices: tuple[int,...]
    row_program: bool = True

    @property
    def members(self):
        return (self.pair.forward,)+((self.correction,) if self.correction is not None else ())+ (self.pair.backward,)

    def validate(self):
        validate_program(self)

    def capture(self, *args, bias=None, asynchronous=False):
        from .native_attention_program import capture_attention_roles,AutomaticAttentionVJPFrame
        from .resident_attention_rows import ResidentAttentionRowsTape
        roles=capture_attention_roles(self,args,bias)
        return AutomaticAttentionVJPFrame(ResidentAttentionRowsTape(
            self,*roles[:3],bias=roles[3] if len(roles)==4 else None,
            asynchronous=asynchronous),self.active)

    @property
    def program_digest(self):
        from .native_attention_vjp_artifact import canonical,payload
        return hashlib.sha256(canonical(payload(self)).encode()).hexdigest()

    def to_json(self):
        from .native_attention_vjp_artifact import to_json
        return to_json(self)


def validate_program(program, *, runtime_dims=None):
    from .typed_attention_backward import _verify_package
    from .typed_attention_checkpoint import validate_typed_saved_lse_descriptor
    from .nvidia_native import _checkpoint_identity
    forward=_verify_package(program.pair.forward,False)
    storage,output=validate_typed_saved_lse_descriptor(forward)
    c=validate_member(program.pair.backward,True)
    fimage=program.pair.forward.image
    if (fimage.entry_points!=(NativeEntryPoint(forward.entry_symbol,forward.abi_id),) or
            program.pair.forward.backend_ir.encode()!=fimage.payload or
            _native_dictionary(program.pair.forward.tile_ir)!=_native_dictionary(program.pair.forward.target_ir)):
        raise ValueError("row program forward executable provenance differs")
    fingerprints=lambda image:(image.target,image.architecture,image.binary_format,image.pipeline_name,
                               image.compiler_fingerprint,image.toolchain_fingerprint)
    if any(fingerprints(member.image)!=fingerprints(fimage) for member in program.members):
        raise ValueError("row program member compiler/toolchain identity differs")
    f=forward.provenance
    if (tuple(f.get("shape_bounds", ())) != tuple(c.get("shape_bounds", ())) or
            f.get("shape_policy") != c.get("shape_policy")):
        raise ValueError("row program forward/consumer sequence capacity differs")
    b,hq,hkv,sq,sk,d,dv=c["shape"]
    physical_bias=list(f.get("bias_shape",())) or ([b,hq,sq,sk] if c["bias"] else [])
    for key,value in (("shape",list(f["shape"])),("storage",storage),("output_storage",output),
                      ("bias",f["bias"]),("causal",f["causal"]),("bias_shape",physical_bias),
                      ("frontend_argument_indices",list(program.input_indices))):
        if _canonical(c.get(key,[]))!=_canonical(value):
            raise ValueError("row program forward/consumer policy differs: "+key)
    if struct.pack("!f",c["scale"])!=struct.pack("!f",f["scale"]):raise ValueError("row program scale differs")
    if pair_witness(program.pair.forward.tile_ir)!=c["source_ad_pair_digest"] or pair_witness(program.pair.forward.target_ir)!=c["source_ad_pair_digest"]:
        raise ValueError("row program forward/consumer source generation differs")
    mapping=program.input_indices;active=program.active
    if (not isinstance(mapping,tuple) or tuple(f["frontend_argument_indices"])!=mapping or
            not isinstance(active,tuple) or not active or len(set(active))!=len(active) or
            any(type(x) is not int or x not in range(3+int(c["bias"])) for x in active) or
            c["gradient_activity"]!=[int(i in active) for i in range(3+int(c["bias"]))]):
        raise ValueError("row program requested physical roles disagree")
    if set(active)=={2}:
        if program.correction is not None:raise ValueError("value-only row program must omit producer")
    else:
        if program.correction is None:raise ValueError("row program correction producer is missing")
        producer=validate_member(program.correction,False)
        for key in producer:
            if key not in ("family","arguments","results","block_threads") and _canonical(producer[key])!=_canonical(c.get(key)):
                raise ValueError("row program producer/consumer policy differs: "+key)
    identity=_checkpoint_identity(tuple(c["shape"]),c["scale"],c["causal"],bias=c["bias"],
        bias_shape=tuple(c.get("bias_shape",())),shape_bounds=tuple(c.get("shape_bounds",())),storage=storage,output_storage=output)
    if program.pair.contract_digest!=identity or f["checkpoint_contract"]!=identity:
        raise ValueError("row program checkpoint identity differs")
    from .attention_shape_contract import attention_dimensions
    dims=attention_dimensions(c["shape"],c.get("shape_bounds",()),runtime_dims)
    _facts(c,True,runtime_dims)
    b,hq,hkv,sq,sk,d,dv=dims
    return dims,((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,dv),(b,hq,sq))


def compile_program(source, active, *, threads=128, launch="packed_v1"):
    from .nvidia_native import AttentionCheckpointPair, _checkpoint_identity, package_scheduled_checkpoint
    from .scheduled_checkpoint import lower_generated_checkpoint
    forward=package_scheduled_checkpoint(lower_generated_checkpoint(source),pipeline_name="tessera-nvidia-pipeline-sm120")
    backward=package_member(source,backward=True,threads=threads,launch=launch)
    c=backward.descriptor.provenance["row_contract"]
    mapping=tuple(c["frontend_argument_indices"])
    physical=tuple(mapping.index(x) for x in active)
    correction=None if set(physical)=={2} else package_member(source,backward=False)
    identity=_checkpoint_identity(tuple(c["shape"]),c["scale"],c["causal"],bias=c["bias"],
        bias_shape=tuple(c.get("bias_shape",())),shape_bounds=tuple(c.get("shape_bounds",())),storage=c["storage"],output_storage=c["output_storage"])
    result=NativeAttentionRowsProgram(AttentionCheckpointPair(forward,backward,identity),correction,physical,mapping)
    validate_program(result)
    return result
