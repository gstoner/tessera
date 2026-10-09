"""Owned gfx1201 native NVFP4 -> lossless storage -> packed matmul chain.

Python validates and submits compiler-produced images. Conversion, permutation,
scale reduction, decoding and matmul arithmetic are owned by MLIR/LLVM.
"""
from __future__ import annotations

import copy
import ctypes as C
from dataclasses import dataclass
import hashlib
import json
import re

import ml_dtypes
import numpy as np

from .native_artifact import BufferArgument
from .rocm_native import ROCMNativePackage
from .rocm_mxfp4_packed_folded import (
    PACKED_FOLDED_PHYSICAL_V1, PACKED_FOLDED_SCALE_PLANE_V1,
    PACKED_FOLDED_TARGET_ABI_V1, PACKED_FOLDED_WEIGHT_LAYOUT_V1,
    _materialize_packed_folded_native, _packed_native_descriptor,
    author_packed_folded_shape_graph,
)
from .rocm_mxfp4_storage import MXFP4_STORAGE_CONTRACT
from .rocm_mxfp4_storage_native import (
    MXFP4StoragePackage, build_mxfp4_storage_graph, package_mxfp4_storage_graph,
)
from .rocm_mxfp4_native import _schedule_hash, _target_string_attr
from .rocm_nvfp4_ingest import nvfp4_requantization_policy
from .rocm_nvfp4_ingest_native import (
    NVFP4IngestPackage, _checked_inputs, build_nvfp4_ingest_graph,
    package_nvfp4_ingest_graph,
)
from .scheduled_matmul import find_tessera_opt, run_tessera_opt


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _shape_receipt(m,n,k,graph,schedule,tile,target):
    return {
        "shape":(n,k),"weight_layout":PACKED_FOLDED_WEIGHT_LAYOUT_V1,
        "scale_layout":PACKED_FOLDED_SCALE_PLANE_V1,
        "physical_contract":PACKED_FOLDED_PHYSICAL_V1,
        "target_abi":PACKED_FOLDED_TARGET_ABI_V1,
        "execution_mode":"folded_row_reference_explicit_approximate",
        "approximate_policy":"explicit_allow",
        "weights_origin":"owned_native_nvfp4_ingest_and_storage",
        "producer_storage_contract":MXFP4_STORAGE_CONTRACT,
        "schedule_hash":_schedule_hash(target,carrier="resident packed Target IR"),
        "graph_ir_sha256":_sha(graph),"schedule_ir_sha256":_sha(schedule),
        "tile_ir_sha256":_sha(tile),"target_ir_sha256":_sha(target),
        # Shape-only compilation makes no assertions about weight content/loss.
        "hsaco_sha256":None,
    }



def _program_digest(data):
    return _sha(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False))



def _native_nvfp4_plan(text):
    """Read a trusted compiler program record; never reconstruct semantic IR."""
    if not isinstance(text, str):
        raise ValueError("native NVFP4 program record must be serialized JSON")
    plan = json.loads(text)
    if (not isinstance(plan,dict) or plan.get("schema") not in {"tessera.native.nvfp4_program.v1","tessera.native.nvfp4_program.v2"}
            or not isinstance(plan.get("source_graph_ir"), str)
            or not isinstance(plan.get("member_graphs"), list)
            or len(plan["member_graphs"]) != 3
            or any(not isinstance(x, str) for x in plan["member_graphs"])
            or not isinstance(plan.get("role_indices"), list)
            or any(type(x) is not int for x in plan["role_indices"])
            or sorted(plan["role_indices"]) != list(range(5))
            or len(plan.get("buffers", [])) != 11
            or [b.get("id") for b in plan["buffers"]] != list(range(11))
            or plan.get("output") != 10):
        raise ValueError("native NVFP4 program ownership record differs")
    roles = plan["role_indices"]
    if ([s.get("inputs") for s in plan.get("steps", [])] !=
            [roles[:3], [5, 6], [roles[3], 8, roles[4], 9]]
            or [s.get("outputs") for s in plan["steps"]] != [[5,6,7], [8,9], [10]]):
        raise ValueError("native NVFP4 program SSA record differs")
    writes,reads = [-1]*11,[-1]*11
    for index,step in enumerate(plan["steps"]):
        for value in step["inputs"]:
            reads[value] = index
        for value in step["outputs"]:
            writes[value] = reads[value] = index
    reads[10] = 3
    widths = {"ui8":1, "f8E4M3FN":1, "f32":4, "f64":8, "bf16":2}
    for index,buffer in enumerate(plan["buffers"]):
        shape = buffer.get("shape")
        size = widths.get(buffer.get("storage"))
        if (not isinstance(shape,list) or not shape or size is None
                or any(type(x) is not int or x <= 0 for x in shape)):
            raise ValueError("native NVFP4 buffer storage/shape record differs")
        for extent in shape:
            size *= extent
        ownership = "readonly_input" if index < 5 else "returned_output" if index == 10 else "private_scratch"
        if (buffer.get("bytes") != size or size > (1<<63)-1
                or buffer.get("ownership") != ownership
                or buffer.get("first_write") != writes[index]
                or buffer.get("last_read") != reads[index]):
            raise ValueError("native NVFP4 buffer capacity/lifetime record differs")
    if plan["schema"].endswith(".v2"):
        active,bound=plan.get("active_m"),plan.get("m_bound")
        if (type(active) is not int or type(bound) is not int or not 0<active<=bound
                or not isinstance(plan.get("original_graph_ir"),str)
                or not plan["original_graph_ir"]
                or len(plan["buffers"][roles[3]]["shape"])!=2
                or len(plan["buffers"][roles[0]]["shape"])!=2
                or plan["buffers"][roles[3]]["shape"][0]!=bound
                or plan["buffers"][roles[3]]["shape"][1]!=2*plan["buffers"][roles[0]]["shape"][1]
                or plan["buffers"][roles[4]]["shape"]!=[bound]
                or plan["buffers"][10]["shape"]!=[bound,plan["buffers"][roles[0]]["shape"][0]]):
            raise ValueError("native NVFP4 bounded row capacity differs")
    return plan

def _portable_stage(graph, schedule, native):
    return copy.deepcopy({
        "graph_ir": graph, "schedule_ir": schedule,
        "tile_ir": native.tile_ir, "target_ir": native.target_ir,
        "image": native.image.to_dict(), "descriptor": native.descriptor.to_dict(),
    })


def _directive_attributes(line):
    """Read canonical serialized attributes for validation, never emit kernel IR."""
    text=line[line.index("{")+1:line.rindex("}")]
    fields=[]; start=0; depth=0; quoted=False; escaped=False
    for i,c in enumerate(text):
        if quoted:
            if escaped: escaped=False
            elif c=="\\": escaped=True
            elif c=='"': quoted=False
        elif c=='"': quoted=True
        elif c in "{[(<": depth+=1
        elif c in "}])>": depth-=1
        elif c=="," and depth==0:
            fields.append(text[start:i].strip()); start=i+1
    fields.append(text[start:].strip())
    if depth or quoted:
        raise ValueError("resident packed consumer malformed Target attributes")
    result={}
    for field in fields:
        pair=field.split(" = ",1)
        key=pair[0]
        if not key or key in result:
            raise ValueError("resident packed consumer duplicate Target attributes")
        result[key]=pair[1] if len(pair)==2 else None
    return result


def _validate_packed_projection(authored, projected, m, n):
    original=next(line for line in authored.splitlines()
                  if "tessera_rocm.scaled_wmma_gemm" in line)
    lines=[line.strip() for line in projected.splitlines() if line.strip()]
    if (len(lines)!=3 or not lines[0].startswith("module attributes {")
            or not lines[1].startswith("tessera_rocm.scaled_wmma_gemm {")
            or lines[2]!="}"):
        raise ValueError("resident packed consumer projected Target structure changed")
    module_before=_directive_attributes(authored.splitlines()[0])
    module_after=_directive_attributes(lines[0])
    module_before.update({
        "tessera.pipeline.arch":'"gfx1201"',
        "tessera.pipeline.backend_codegen":'"rocdl_hsaco"',
        "tessera.pipeline.family":'"matmul"',
        "tessera.pipeline.output":'"target"',
        "tessera.pipeline.schema":'"tessera.executable_pipeline.v1"',
        "tessera.pipeline.target_ir_consumer":'"tessera_rocm"',
        "tessera.pipeline.tile_producer":'"content_addressed_tile"',
    })
    if module_before!=module_after:
        raise ValueError("resident packed consumer projected module contract changed")
    before=_directive_attributes(original)
    after=_directive_attributes(lines[1])
    before.pop("name")
    before.pop("tessera.schedule_hash")
    before.update(m="0 : i64",n="0 : i64",runtime_mn=None,
                  whole_m=str(m%256==0).lower(),whole_n=str(n%64==0).lower())
    after.pop("name",None)
    if before!=after:
        raise ValueError("resident packed consumer projected Target contract changed")


@dataclass(frozen=True)
class ResidentPackedConsumer:
    graph_ir:str
    schedule_ir:str
    package:ROCMNativePackage
    m:int
    n:int
    k:int
    native_plan_json:str|None=None

    def validate(self):
        p=self.package
        image,desc=p.image,p.descriptor
        desc.validate_image(image)
        if (image.target!="rocm_gfx1201" or image.architecture!="gfx1201"
                or image.binary_format!="hsaco" or image.pipeline_name!="tessera-lower-to-rocm"
                or image.target_ir_digest!=_sha(p.target_ir)):
            raise ValueError("resident packed consumer requires its exact gfx1201 HSACO lineage")
        if self.native_plan_json is not None:
            plan = _native_nvfp4_plan(self.native_plan_json)
            if self.graph_ir != plan["member_graphs"][2]:
                raise ValueError("resident packed consumer native member changed")
        else:
            from .rocm_nvfp4_program import build_packed_consumer_module
            expected=build_packed_consumer_module(self.m,self.n,self.k).to_mlir(
                target="rocm_gfx1201",canonical=True)
            if self.graph_ir not in (author_packed_folded_shape_graph(self.m,self.n,self.k),expected):
                raise ValueError("resident packed consumer Graph semantics changed")
        runtime_mn=desc.provenance.get("image_shape_policy")=="runtime_mn_fixed_k"
        authored=desc.provenance.get("authored_target_ir") if runtime_mn else p.target_ir
        if not isinstance(authored,str):
            raise ValueError("resident packed consumer authored Target missing")
        if runtime_mn:
            if desc.provenance.get("authored_target_ir_sha256")!=_sha(authored):
                raise ValueError("resident packed consumer authored Target digest changed")
            _validate_packed_projection(authored,p.target_ir,self.m,self.n)
        tile_hash=_schedule_hash(p.tile_ir,carrier="resident packed Tile IR")
        target_hash=_schedule_hash(authored,carrier="resident packed Target IR")
        if (p.tile_ir.count("tile.scaled_matmul_kernel")!=1
                or p.target_ir.count("tessera_rocm.scaled_wmma_gemm")!=1
                or tile_hash!=target_hash
                or re.findall(r'\bhash = "([0-9a-f]{64})"',self.schedule_ir)!=[target_hash]):
            raise ValueError("resident packed consumer Schedule/Tile/Target lineage changed")
        target=next(line for line in authored.splitlines() if "tessera_rocm.scaled_wmma_gemm" in line)
        for key,value in (
            ("physical_contract",PACKED_FOLDED_PHYSICAL_V1),
            ("package_abi",PACKED_FOLDED_TARGET_ABI_V1),
            ("abi","a_bpacked_sa_scaleplane_d_m_n_k"),
            ("scale_format","e8m0_k32_plus_row_reference"),
        ):
            if _target_string_attr(target,key)!=value:
                raise ValueError("resident packed consumer physical contract changed")
        for key,dimension_value in (("m",self.m),("n",self.n),("k",self.k),
                ("block_m",256),("block_n",64),("stage_k",64),("scale_k",self.k)):
            found=re.search(r"(?<![\w.])"+key+r" = (\d+) : i64",target)
            if found is None or int(found.group(1))!=dimension_value:
                raise ValueError("resident packed consumer Target dimensions changed")
        for key,value in (("output","bf16"),("execution_mode","folded_row_reference_explicit_approximate"),
                ("accum","f32"),("partial_combine","row_reference_after_full_k")):
            if _target_string_attr(target,key)!=value:
                raise ValueError("resident packed consumer Target numerical policy changed")
        image_target=next(line for line in p.target_ir.splitlines() if "tessera_rocm.scaled_wmma_gemm" in line)
        entry=_target_string_attr(image_target,"name")
        receipt=_shape_receipt(self.m,self.n,self.k,self.graph_ir,self.schedule_ir,p.tile_ir,authored)
        if runtime_mn:
            receipt.update(authored_target_ir=authored,authored_target_ir_sha256=_sha(authored),
                           target_ir_sha256=_sha(p.target_ir))
        if desc!=_packed_native_descriptor(image,entry,self.m,self.n,self.k,receipt,runtime_mn=runtime_mn):
            raise ValueError("resident packed consumer descriptor changed")


def package_resident_packed_consumer(m,n,k,*,graph=None,runtime_mn=True,native_plan_json=None):
    graph=(author_packed_folded_shape_graph(m,n,k) if graph is None else
           graph if isinstance(graph,str) else graph.to_mlir(target="rocm_gfx1201",canonical=True))
    tool=find_tessera_opt()
    if tool is None:
        raise RuntimeError("resident packed consumer requires the matching native compiler")
    schedule=run_tessera_opt(tool,graph,"--tessera-graph-to-schedule")
    tile=run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
    target=run_tessera_opt(tool,tile,"--lower-tile-to-rocm=arch=gfx1201")
    receipt=_shape_receipt(m,n,k,graph,schedule,tile,target)
    authored=target
    compiled=_materialize_packed_folded_native(m,n,k,graph,tile,target,receipt,runtime_mn=runtime_mn)
    # Retain static Schedule ancestry separately from the projected image.
    target=compiled.package.target_ir
    receipt=_shape_receipt(m,n,k,graph,schedule,tile,authored if runtime_mn else target)
    if runtime_mn:
        receipt.update(authored_target_ir=authored,authored_target_ir_sha256=_sha(authored),
                       target_ir_sha256=_sha(target))
    entry=compiled.package.descriptor.entry_symbol
    from dataclasses import replace
    package=replace(compiled.package,descriptor=_packed_native_descriptor(
        compiled.package.image,entry,m,n,k,receipt,runtime_mn=runtime_mn))
    result=ResidentPackedConsumer(graph,schedule,package,m,n,k,native_plan_json)
    result.validate()
    return result


@dataclass(frozen=True)
class NVFP4ResidentProgram:
    ingest:NVFP4IngestPackage
    storage:MXFP4StoragePackage
    consumer:ResidentPackedConsumer
    native_plan_json:str|None=None

    def validate(self):
        self.ingest.validate()
        self.storage.validate()
        self.consumer.validate()
        n,k=self.consumer.n,self.consumer.k
        offsets=self.ingest.native.descriptor.provenance["row_offsets"]
        if self.native_plan_json is not None:
            plan = _native_nvfp4_plan(self.native_plan_json)
            if (self.consumer.native_plan_json != self.native_plan_json
                    or [self.ingest.graph_ir, self.storage.graph_ir, self.consumer.graph_ir]
                    != plan["member_graphs"]):
                raise ValueError("resident program native member lineage changed")
            if (plan["schema"].endswith(".v2") and
                    self.consumer.package.descriptor.provenance.get("image_shape_policy")!="runtime_mn_fixed_k"):
                raise ValueError("bounded NVFP4 rows require a runtime-M/N consumer image")
            roles = plan["role_indices"]
            shapes = [b["shape"] for b in plan["buffers"]]
            if (shapes[roles[3]] != [self.consumer.m,k]
                    or shapes[roles[0]] != [n,k//2]
                    or shapes[10] != [self.consumer.m,n]):
                raise ValueError("resident program native buffer dimensions differ")
        else:
            expected_ingest=build_nvfp4_ingest_graph(n,k,offsets,
                numeric_policy=nvfp4_requantization_policy()).to_mlir(
                    target="rocm_gfx1201",canonical=True)
            expected_storage=build_mxfp4_storage_graph(n,k).to_mlir(
                    target="rocm_gfx1201",canonical=True)
            if self.ingest.graph_ir!=expected_ingest or self.storage.graph_ir!=expected_storage:
                raise ValueError("resident program Graph topology or semantic policy changed")
        nk=[n,k]
        if (self.ingest.native.descriptor.provenance["shape_nk"]!=nk
                or self.storage.native.descriptor.provenance["shape_nk"]!=nk):
            raise ValueError("resident converter/storage/consumer shape mismatch")
        if self.storage.native.descriptor.provenance["storage_contract"]!=MXFP4_STORAGE_CONTRACT:
            raise ValueError("resident storage contract differs from packed consumer")

    @property
    def receipt(self):
        self.validate()
        components=(self.ingest.native,self.storage.native,self.consumer.package)
        lossy_steps=nvfp4_requantization_policy()["lossy_steps"]
        if not isinstance(lossy_steps,list) or any(not isinstance(step,str) for step in lossy_steps):
            raise ValueError("NVFP4 policy loss steps must be a list of names")
        return {
            "work_item":"ROCM-NVFP4-INGEST-1",
            "sync_key":"ROCM-INGEST-RESIDENT-2026-10-05",
            "route":"native Graph/Schedule/Tile NVFP4 conversion -> lossless storage -> packed matmul",
            "shape_mnk":[self.consumer.m,self.consumer.n,self.consumer.k],
            "row_offsets":self.ingest.native.descriptor.provenance["row_offsets"],
            "storage_contract":MXFP4_STORAGE_CONTRACT,
            "numeric_policy":nvfp4_requantization_policy(),
            "consumer_numeric_policy":"folded_row_reference_explicit_approximate",
            "lossy_steps":[*lossy_steps,
                "mxfp4_to_e4m3_at_row_reference_in_native_matmul"],
            "component_image_digests":[p.image.image_digest for p in components],
            "component_descriptors":[p.descriptor.to_dict() for p in components],
            "resident_edges":"owned buffers on one private HIP stream, no intermediate reupload",
        }


    def to_dict(self):
        """Persist native images and retained IR; no host weight content is saved."""
        self.validate()
        data = {
            "schema": "tessera.rocm.nvfp4_resident_program.v1",
            "shape_mnk": [self.consumer.m, self.consumer.n, self.consumer.k],
            "stages": [
                _portable_stage(self.ingest.graph_ir, self.ingest.schedule_ir, self.ingest.native),
                _portable_stage(self.storage.graph_ir, self.storage.schedule_ir, self.storage.native),
                _portable_stage(self.consumer.graph_ir, self.consumer.schedule_ir, self.consumer.package),
            ],
        }
        if self.native_plan_json is not None:
            data["schema"] = "tessera.rocm.nvfp4_resident_program.v2"
            data["native_plan_json"] = self.native_plan_json
        return {**data, "program_digest": _program_digest(data)}

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_dict(cls, data):
        """Restore checked native stages without recompilation or GPU access.

        The digest checks integrity, not origin authentication. Artifacts must
        come from a trusted compiler/storage source, like native images.
        """
        from .native_artifact import LaunchDescriptor, NativeImageArtifact
        from .rocm_native import ROCMNativePackage

        native_record = isinstance(data, dict) and data.get("schema") == "tessera.rocm.nvfp4_resident_program.v2"
        fields = {"schema", "shape_mnk", "stages", "program_digest"}
        if native_record:
            fields.add("native_plan_json")
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("resident program schema fields differ")
        data = copy.deepcopy(data)
        if data["schema"] not in {"tessera.rocm.nvfp4_resident_program.v1", "tessera.rocm.nvfp4_resident_program.v2"}:
            raise ValueError("resident program schema version differs")
        shape = data["shape_mnk"]
        if (not isinstance(shape, list) or len(shape) != 3
                or any(type(x) is not int for x in shape)):
            raise ValueError("resident program requires integer M/N/K")
        # The native Graph author validates the full static shape envelope.
        if not native_record:
            author_packed_folded_shape_graph(*shape)
        elif any(x <= 0 for x in shape) or shape[1] % 16 or shape[2] % 64:
            raise ValueError("native resident program requires positive N16/K64")
        stages = data["stages"]
        if not isinstance(stages, list) or len(stages) != 3:
            raise ValueError("resident program requires three ordered stages")
        identity = {key: value for key, value in data.items() if key != "program_digest"}
        if data["program_digest"] != _program_digest(identity):
            raise ValueError("resident program content digest differs")
        restored = []
        for stage in stages:
            if (not isinstance(stage, dict) or set(stage) != {
                    "graph_ir", "schedule_ir", "tile_ir", "target_ir", "image", "descriptor"}
                    or any(not isinstance(stage[key], str) for key in (
                        "graph_ir", "schedule_ir", "tile_ir", "target_ir"))
                    or not isinstance(stage["image"], dict)
                    or not isinstance(stage["descriptor"], dict)):
                raise ValueError("resident program stage schema differs")
            image = NativeImageArtifact.from_dict(stage["image"])
            descriptor = LaunchDescriptor.from_dict(stage["descriptor"])
            native = ROCMNativePackage(stage["tile_ir"], stage["target_ir"], "", image, descriptor)
            restored.append((stage["graph_ir"], stage["schedule_ir"], native))
        program = cls(
            NVFP4IngestPackage(*restored[0]), MXFP4StoragePackage(*restored[1]),
            ResidentPackedConsumer(restored[2][0], restored[2][1], restored[2][2],
                                   shape[0], shape[1], shape[2], data.get("native_plan_json")),
            data.get("native_plan_json"))
        program.validate()
        return program

    @classmethod
    def from_json(cls, text):
        def unique_fields(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("resident program JSON has duplicate fields")
                result[key] = value
            return result
        def invalid_constant(value):
            raise ValueError("resident program JSON has nonfinite constants")
        return cls.from_dict(json.loads(text, object_pairs_hook=unique_fields,
                                       parse_constant=invalid_constant))

    def native_session(self,codes,scales,globals_,a,a_scale,*,reuse=False):
        from .native_resident_nvfp4 import NativeResidentNVFP4
        return NativeResidentNVFP4(self,codes,scales,globals_,a,a_scale,reuse=reuse)

    def session(self,codes,scales,globals_,a,a_scale):
        return ResidentNVFP4Matmul(self,codes,scales,globals_,a,a_scale)


def package_resident_nvfp4_matmul(m,n,k,row_offsets,*,numeric_policy,approximate_policy,runtime_mn=True):
    if numeric_policy!=nvfp4_requantization_policy() or approximate_policy!="explicit_allow":
        raise ValueError("resident NVFP4 matmul requires explicit conversion and approximate consumer policies")
    consumer=package_resident_packed_consumer(m,n,k,runtime_mn=runtime_mn)
    ingest=package_nvfp4_ingest_graph(build_nvfp4_ingest_graph(n,k,row_offsets,numeric_policy=numeric_policy))
    storage=package_mxfp4_storage_graph(build_mxfp4_storage_graph(n,k))
    program=NVFP4ResidentProgram(ingest,storage,consumer)
    program.validate()
    return program


def _activation_inputs(m,k,a,a_scale):
    a,a_scale=np.asarray(a),np.asarray(a_scale)
    if (a.dtype!=np.uint8 or a.shape!=(m,k)
            or a_scale.dtype!=np.float32 or a_scale.shape!=(m,)):
        raise ValueError("resident activation storage/shape differs")
    if (not np.isfinite(a.view(ml_dtypes.float8_e4m3fn)).all()
            or not np.isfinite(a_scale).all() or np.any(a_scale<0)):
        raise ValueError("resident activations and scales must be finite and scales nonnegative")
    return a,a_scale


class ResidentNVFP4Matmul:
    """Private allocations and ordered stream; no borrowed/external buffers."""

    def __init__(self,program,codes,scales,globals_,a,a_scale):
        program.validate()
        # Copy descriptor dictionaries so caller mutation cannot retarget a live session.
        self._program=copy.deepcopy(program)
        self.m,self.n,self.k=program.consumer.m,program.consumer.n,program.consumer.k
        offsets=tuple(program.ingest.native.descriptor.provenance["row_offsets"])
        checked=_checked_inputs(self.n,self.k,offsets,codes,scales,globals_)
        a,a_scale=_activation_inputs(self.m,self.k,a,a_scale)
        self._host_inputs={name:np.array(value,copy=True,order="C") for name,value in zip(
            ("codes","scales","globals","a","a_scale"),[*checked,a,a_scale])}
        for value in self._host_inputs.values():
            value.setflags(write=False)
        self._specs={
            "codes":((self.n,self.k//2),np.dtype(np.uint8)),
            "scales":((self.n,self.k//16),np.dtype(ml_dtypes.float8_e4m3fn)),
            "globals":((len(offsets)-1,),np.dtype(np.float64)),
            "a":((self.m,self.k),np.dtype(np.uint8)),
            "a_scale":((self.m,),np.dtype(np.float32)),
            "packed":((self.n,self.k//2),np.dtype(np.uint8)),
            "exponents":((self.k//32,self.n),np.dtype(np.uint8)),
            "stats":((self.n,self.k//32,2),np.dtype(np.float64)),
            "fragment":((self.n,self.k//2),np.dtype(np.uint8)),
            "plane":((self.k//32+1,self.n),np.dtype(np.uint8)),
            "output":((self.m,self.n),np.dtype(ml_dtypes.bfloat16)),
        }
        self._stage_slots=(("codes","scales","globals","packed","exponents","stats"),
            ("packed","exponents","fragment","plane"),
            ("a","fragment","a_scale","plane","output"))
        self._packages=(self._program.ingest.native,self._program.storage.native,self._program.consumer.package)
        self._stage_scalars: tuple[dict[str,int], ...] = ({}, {}, {"M":self.m,"N":self.n,"K":self.k})
        for package,slots,scalars in zip(self._packages,self._stage_slots,self._stage_scalars):
            bindings=sorted(package.descriptor.buffers,key=lambda b:b.ordinal)
            args={b.name:BufferArgument(b.dtype,self._specs[slot][0],"row_major",
                self._specs[slot][1].itemsize) for b,slot in zip(bindings,slots)}
            package.descriptor.validate_invocation(package.image,args,scalars)
        self._closed=False
        self._closing=False
        self._activations_ready=False
        self._weights_ready=False
        self._output_ready=False
        self._buffers: dict[str,C.c_void_p] = {}
        self._modules=[]
        self._image_storage=[]
        self._functions=[]
        self._events=[]
        self._graphs={}
        self._params=[]
        self._param_values=[]
        self._stream=C.c_void_p()
        from tessera import runtime as rt
        if rt._rocm_live_arch()!="gfx1201":
            raise RuntimeError("resident ingest requires the exact gfx1201 owning device")
        hip=rt._load_hip_for_launch()
        if hip is None:
            raise RuntimeError("resident ingest HIP runtime unavailable")
        self._hip: C.CDLL = hip
        self._check(hip.hipInit(0))
        for symbol,types in {
            "hipGetDevice":[C.POINTER(C.c_int)],"hipSetDevice":[C.c_int],
            "hipStreamCreateWithFlags":[C.POINTER(C.c_void_p),C.c_uint],
            "hipStreamSynchronize":[C.c_void_p],"hipStreamDestroy":[C.c_void_p],
            "hipMemcpyAsync":[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int,C.c_void_p],
            "hipEventCreate":[C.POINTER(C.c_void_p)],"hipEventRecord":[C.c_void_p,C.c_void_p],
            "hipEventSynchronize":[C.c_void_p],"hipEventElapsedTime":[C.POINTER(C.c_float),C.c_void_p,C.c_void_p],
            "hipEventDestroy":[C.c_void_p],
            "hipStreamBeginCapture":[C.c_void_p,C.c_int],
            "hipStreamEndCapture":[C.c_void_p,C.POINTER(C.c_void_p)],
            "hipGraphGetNodes":[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)],
            "hipGraphInstantiateWithFlags":[C.POINTER(C.c_void_p),C.c_void_p,C.c_ulonglong],
            "hipGraphLaunch":[C.c_void_p,C.c_void_p],
            "hipGraphExecDestroy":[C.c_void_p],"hipGraphDestroy":[C.c_void_p],
        }.items():
            getattr(hip,symbol).argtypes=types
            getattr(hip,symbol).restype=C.c_int
        ordinal=C.c_int()
        self._check(hip.hipGetDevice(C.byref(ordinal)))
        self._device_id=ordinal.value
        try:
            self._check(hip.hipStreamCreateWithFlags(C.byref(self._stream),1))
            for package in self._packages:
                blob=C.create_string_buffer(package.image.payload)
                self._image_storage.append(blob)
                module,function=C.c_void_p(),C.c_void_p()
                self._check(hip.hipModuleLoadData(C.byref(module),blob))
                self._modules.append(module)
                self._check(hip.hipModuleGetFunction(C.byref(function),module,package.descriptor.entry_symbol.encode()))
                self._functions.append(function)
            for name,(shape,dtype) in self._specs.items():
                pointer=C.c_void_p()
                size=int(np.prod(shape,dtype=object))*dtype.itemsize
                self._check(hip.hipMalloc(C.byref(pointer),size))
                if not pointer.value or pointer.value in {p.value for p in self._buffers.values()}:
                    raise RuntimeError("resident ingest allocation ownership is not distinct")
                self._buffers[name]=pointer
            for name,array in self._host_inputs.items():
                self._check(hip.hipMemcpyAsync(self._buffers[name],C.c_void_p(array.ctypes.data),
                    array.nbytes,1,self._stream))
            for package,slots,scalars in zip(self._packages,self._stage_slots,self._stage_scalars):
                values: list[C.c_void_p | C.c_int64] = []
                for slot in slots:
                    shape,_=self._specs[slot]
                    pointer=self._buffers[slot]
                    values.extend((C.c_void_p(pointer.value),C.c_void_p(pointer.value),
                        C.c_int64(0),C.c_int64(int(np.prod(shape,dtype=object))),C.c_int64(1)))
                values.extend(C.c_int64(scalars[s.name]) for s in sorted(
                    package.descriptor.scalars,key=lambda s:s.ordinal))
                params=(C.c_void_p*len(values))(*[C.cast(C.byref(v),C.c_void_p) for v in values])
                self._param_values.append(values)
                self._params.append(params)
            self.synchronize()
            self._activations_ready=True
        except BaseException:
            self.close()
            raise

    def _check(self,rc):
        if rc:
            raise RuntimeError(f"resident NVFP4 HIP status {rc}")

    def _ensure_open(self):
        if self._closed or self._closing:
            raise RuntimeError("resident ingest session is closed")
        ordinal=C.c_int()
        self._check(self._hip.hipGetDevice(C.byref(ordinal)))
        if ordinal.value!=self._device_id:
            raise RuntimeError("resident ingest owning device changed")

    def _launch_stage(self,index):
        self._ensure_open()
        geometry=self._packages[index].descriptor.geometry
        self._check(self._hip.hipModuleLaunchKernel(self._functions[index],
            *geometry.grid,*geometry.workgroup,0,self._stream,self._params[index],None))

    def update_activations(self,a,a_scale):
        self._ensure_open()
        inputs=_activation_inputs(self.m,self.k,a,a_scale)
        staging={name:np.array(value,copy=True,order="C") for name,value in zip(
            ("a","a_scale"),inputs)}
        for value in staging.values():
            value.setflags(write=False)
        # Complete prior uploads/launches before replacing the old staging owner.
        self.synchronize()
        self._output_ready=False
        self._activations_ready=False
        self._host_inputs.update(staging)
        for name,array in staging.items():
            self._check(self._hip.hipMemcpyAsync(self._buffers[name],C.c_void_p(array.ctypes.data),
                array.nbytes,1,self._stream))
        self._activations_ready=True

    def ingest(self):
        self._output_ready=False
        self._weights_ready=False
        self._launch_stage(0)
        self._launch_stage(1)
        self._weights_ready=True

    def launch_matmul(self):
        self._ensure_open()
        if not self._weights_ready:
            raise RuntimeError("resident weights require successful ingest before matmul")
        if not self._activations_ready:
            raise RuntimeError("resident activations require successful upload before matmul")
        self._output_ready=False
        self._launch_stage(2)
        self._output_ready=True

    def run_combined(self):
        self.ingest()
        self.launch_matmul()

    def synchronize(self):
        self._ensure_open()
        self._check(self._hip.hipStreamSynchronize(self._stream))

    def _download(self,name):
        self._ensure_open()
        shape,dtype=self._specs[name]
        output=np.empty(shape,dtype=dtype)
        self._check(self._hip.hipMemcpyAsync(C.c_void_p(output.ctypes.data),self._buffers[name],
            output.nbytes,2,self._stream))
        # Retain output until its asynchronous copy completes, even on an exception.
        try:
            self.synchronize()
        except BaseException:
            self._image_storage.append(output)
            raise
        return output

    def read_output(self):
        if not self._output_ready:
            raise RuntimeError("resident output requires successful matmul")
        return self._download("output")

    def diagnostics(self):
        if not self._weights_ready:
            raise RuntimeError("resident diagnostics require successful ingest")
        return {name:self._download(name) for name in ("packed","exponents","stats","fragment","plane")}

    def _timing_callback(self,stage,samples,repeats):
        self._ensure_open()
        if type(samples) is not int or samples<=0 or type(repeats) is not int or repeats<=0:
            raise ValueError("resident timing requires positive samples/repeats")
        callbacks={"ingest":self.ingest,"consumer":self.launch_matmul,"combined":self.run_combined,
            "converter":lambda:self._launch_stage(0),"storage":lambda:self._launch_stage(1)}
        if stage not in callbacks:
            raise ValueError("unknown resident timing stage")
        if stage in {"storage","consumer"} and not self._weights_ready:
            raise RuntimeError("resident timing requires initialized converted weights")
        return callbacks[stage]

    def measure(self,stage,*,samples=3,repeats=10):
        callback=self._timing_callback(stage,samples,repeats)
        events=[]
        try:
            for _ in range(2):
                event=C.c_void_p()
                self._check(self._hip.hipEventCreate(C.byref(event)))
                self._events.append(event);events.append(event)
            result=[]
            for _ in range(samples):
                self._check(self._hip.hipEventRecord(events[0],self._stream))
                for _ in range(repeats):
                    callback()
                self._check(self._hip.hipEventRecord(events[1],self._stream))
                self._check(self._hip.hipEventSynchronize(events[1]))
                elapsed=C.c_float()
                self._check(self._hip.hipEventElapsedTime(C.byref(elapsed),events[0],events[1]))
                result.append(elapsed.value/repeats)
            return result
        finally:
            self.synchronize()
            for event in events:
                self._check(self._hip.hipEventDestroy(event))
                self._events.remove(event)

    def measure_graph(self,stage,*,samples=3,repeats=128):
        """One graph submission per window removes between-node host gaps.

        Capturing and instantiating are outside timing. Results include GPU
        graph dispatch, not an isolated-instruction/kernel-only claim.
        """
        callback=self._timing_callback(stage,samples,repeats)
        self.synchronize()
        hip=self._hip
        key=(stage,repeats)
        if key not in self._graphs:
            graph,executable=C.c_void_p(),C.c_void_p()
            capturing=False
            readiness=(self._weights_ready,self._output_ready)
            try:
                self._check(hip.hipStreamBeginCapture(self._stream,1))
                capturing=True
                for _ in range(repeats):
                    callback()
                rc=hip.hipStreamEndCapture(self._stream,C.byref(graph))
                capturing=False
                self._check(rc)
                nodes=C.c_size_t()
                self._check(hip.hipGraphGetNodes(graph,None,C.byref(nodes)))
                expected=repeats*({"ingest":2,"combined":3}.get(stage,1))
                if nodes.value!=expected:
                    raise RuntimeError("resident graph node census differs from native stages")
                self._check(hip.hipGraphInstantiateWithFlags(C.byref(executable),graph,0))
                self._graphs[key]=(graph,executable,nodes.value)
            except BaseException:
                if capturing:
                    hip.hipStreamEndCapture(self._stream,C.byref(graph))
                if executable.value:
                    hip.hipGraphExecDestroy(executable)
                if graph.value:
                    hip.hipGraphDestroy(graph)
                raise
            finally:
                self._weights_ready,self._output_ready=readiness
        graph,executable,nodes=self._graphs[key]
        events=[]
        try:
            for _ in range(2):
                event=C.c_void_p()
                self._check(hip.hipEventCreate(C.byref(event)))
                self._events.append(event);events.append(event)
            result=[]
            # Untimed first replay establishes initialization and graph upload.
            self._check(hip.hipGraphLaunch(executable,self._stream))
            self.synchronize()
            for _ in range(samples):
                self._check(hip.hipEventRecord(events[0],self._stream))
                self._check(hip.hipGraphLaunch(executable,self._stream))
                self._check(hip.hipEventRecord(events[1],self._stream))
                self._check(hip.hipEventSynchronize(events[1]))
                elapsed=C.c_float()
                self._check(hip.hipEventElapsedTime(C.byref(elapsed),events[0],events[1]))
                result.append({"stage":stage,"repeats":repeats,"graph_nodes":nodes,
                    "host_graph_submissions":1,"window_ms":elapsed.value,
                    "per_iteration_ms":elapsed.value/repeats})
            if stage in {"ingest","combined"}:
                self._weights_ready=True
            if stage in {"consumer","combined"}:
                self._output_ready=True
            return result
        finally:
            self.synchronize()
            for event in events:
                self._check(hip.hipEventDestroy(event));self._events.remove(event)

    def close(self):
        if self._closed:
            return
        hip=self._hip
        ordinal=C.c_int()
        self._check(hip.hipGetDevice(C.byref(ordinal)))
        if ordinal.value!=self._device_id:
            self._check(hip.hipSetDevice(self._device_id))
        try:
            # Never release an allocation/image while its work may still be pending.
            if self._stream.value:
                self._check(hip.hipStreamSynchronize(self._stream))
            self._closing=True
            for key,(graph,executable,nodes) in list(self._graphs.items()):
                if executable.value:
                    self._check(hip.hipGraphExecDestroy(executable));executable.value=None
                if graph.value:
                    self._check(hip.hipGraphDestroy(graph));graph.value=None
                del self._graphs[key]
            for event in list(self._events):
                self._check(hip.hipEventDestroy(event));self._events.remove(event)
            for name,pointer in list(self._buffers.items()):
                self._check(hip.hipFree(pointer));del self._buffers[name]
            for module in list(self._modules):
                self._check(hip.hipModuleUnload(module));self._modules.remove(module)
            if self._stream.value:
                self._check(hip.hipStreamDestroy(self._stream))
                self._stream=C.c_void_p()
            self._image_storage.clear()
            self._param_values.clear()
            self._params.clear()
            self._host_inputs.clear()
            self._closed=True
        finally:
            if ordinal.value!=self._device_id:
                self._check(hip.hipSetDevice(ordinal.value))

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self,*exc):
        self.close()
