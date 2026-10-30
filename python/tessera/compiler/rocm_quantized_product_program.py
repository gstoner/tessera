"""Native SSA packaging for a resident BF16 quantizer and exact scaled product.

C++ owns source admission, partitioning, lifetimes and all numerical lowering.
Python validates the exported frame and marshals existing native member ABIs.
"""
from __future__ import annotations

import base64
import ctypes as C
from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Any

from .native_artifact import LaunchGeometry, ScalarArgument
from .quantized_binding_cache import memoized_validation,memoized_binding
from .native_scaled_program import PreparedScaledProgram, _BoundScaledABI, _Buffer, _StorageSpec, _storage_numpy_dtype
from .rocm_block_quant_native import BlockQuantRecipe, lower_block_quant_text, package_block_quant_recipe, validate_block_quant_runtime_artifact, _frame
from .rocm_fp8_blockscale import BlockScaleProgram, BlockScaleShape, PACKAGE_ABIS, admitted_rows_graph, check_blockscale_target_ir, package_blockscale, verify_blockscale_schedule
from .rocm_native import ROCMNativePackage, _directive_symbol
from .rocm_nvfp4_expansion_program import _MultiOutputStep
from .rocm_nvfp4_expand_native import NVFP4ExpansionPackage, package_nvfp4_expansion
from .rocm_parent_view_native import ParentViewRecipe,package_parent_view_recipe,validate_parent_view_runtime_artifact,_frame as _parent_view_frame,runtime_width_graph,runtime_width_profile
from .scheduled_matmul import find_tessera_opt, run_tessera_opt

_OPTION = "--tessera-autodiff-forward=export-scaled-primal=true"
_SCHEMA = "tessera.native.quantized_product.v1"
_EXPANDED_SCHEMA = "tessera.native.quantized_product.v2"
_ACTIVATED_SCHEMA = "tessera.native.quantized_product.v3"
_NATIVE_SCHEMAS = (_SCHEMA,_EXPANDED_SCHEMA,_ACTIVATED_SCHEMA)
_PACKAGE_SCHEMAS = tuple(f"tessera.rocm.quantized_product_package.v{i}" for i in (1,2,3))
# v4 is the static seven-member MLP; v5 is that MLP at capacity with a row frame.
_MLP_SCHEMAS = tuple(f"tessera.native.quantized_product.v{i}" for i in (4,5,6,7,8))


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _export(tool, source):
    text = run_tessera_opt(tool,source,_OPTION)
    encoded = re.findall(r'tessera.native.quantized_product_json = "([^"\n]+)"',text)
    if len(encoded) != 1:
        raise ValueError("requires one native quantized-product SSA record")
    plan = json.loads(base64.b64decode(encoded[0],validate=True))
    if not isinstance(plan,dict) or plan.get("schema") not in (*_NATIVE_SCHEMAS,*_MLP_SCHEMAS):
        raise ValueError("native quantized-product schema differs")
    return plan


def _frame_shape(plan):
    if isinstance(plan,dict) and plan.get("schema") in _MLP_SCHEMAS:
        from .rocm_quantized_mlp_program import _frame as mlp_frame
        roles,shape,_=mlp_frame(plan)
        return roles,shape
    if not isinstance(plan,dict):
        raise ValueError("native quantized-product frame differs")
    activated = plan.get("schema") == _ACTIVATED_SCHEMA
    expanded = plan.get("schema") in (_EXPANDED_SCHEMA,_ACTIVATED_SCHEMA)
    roots,members,total,output = (4,4,10,9) if activated else (4,3,9,8) if expanded else (3,2,6,5)
    if (set(plan) != {"schema","source_graph_ir","member_graphs","buffers","steps","output"}
            or plan["schema"] not in _NATIVE_SCHEMAS
            or type(plan["output"]) is not int or plan["output"] != output
            or not isinstance(plan["source_graph_ir"],str) or not plan["source_graph_ir"]
            or not isinstance(plan["member_graphs"],list) or len(plan["member_graphs"]) != members
            or not all(isinstance(v,str) and v for v in plan["member_graphs"])
            or not isinstance(plan["steps"],list) or len(plan["steps"]) != members
            or not isinstance(plan["buffers"],list) or len(plan["buffers"]) != total):
        raise ValueError("native quantized-product frame differs")
    steps,rows = plan["steps"],plan["buffers"]
    operations = ["native_block_quant"] + (["tessera.nvfp4_expand"] if expanded else []) + ["tessera.scaled_matmul"] + (["tessera.silu_mul"] if activated else [])
    for i,step in enumerate(steps):
        if (not isinstance(step,dict) or set(step) != {"step","operation","member","inputs","outputs"}
                or type(step["step"]) is not int or step["step"] != i
                or step["operation"] != operations[i]
                or not isinstance(step["member"],str) or not step["member"]
                or not isinstance(step["inputs"],list) or not isinstance(step["outputs"],list)
                or any(type(v) is not int for v in step["inputs"]+step["outputs"])):
            raise ValueError("native quantized-product step encoding differs")
    if len(steps[0]["inputs"]) != 1 or len(steps[2 if expanded else 1]["inputs"]) != 4:
        raise ValueError("native quantized-product step roles differ")
    roles = [steps[0]["inputs"][0]] + (steps[1]["inputs"] if expanded else [steps[1]["inputs"][1],steps[1]["inputs"][3]])
    if sorted(roles) != list(range(roots)):
        raise ValueError("native quantized-product original arguments differ")
    inputs = [[roles[0]],roles[1:],[4,6,5,7]] if expanded else [[roles[0]],[3,roles[1],4,roles[2]]]
    outputs = [[4,5],[6,7],[8]] if expanded else [[3,4],[5]]
    if activated:
        inputs.append([8]);outputs.append([9])
    if [step["inputs"] for step in steps] != inputs or [step["outputs"] for step in steps] != outputs:
        raise ValueError("native quantized-product lost its private producer edges")
    if not all(isinstance(row,dict) for row in rows):
        raise ValueError("native quantized-product buffer rows differ")
    xshape = rows[roles[0]].get("shape")
    bshape = rows[6 if expanded else roles[1]].get("shape")
    if (not isinstance(xshape,list) or len(xshape) != 2
            or not isinstance(bshape,list) or len(bshape) != 2
            or any(type(d) is not int or d<=0 for d in xshape+bshape)):
        raise ValueError("native quantized-product capacity differs")
    m,k = xshape
    n,bk = bshape
    if bk != k or k%16 or m>2147483647 or k>2147483647 or m*(k//16)>2147483647:
        raise ValueError("native quantized-product K16 capacity differs")
    out = rows[8 if activated else output].get("storage")
    if not isinstance(out,str) or out not in {"f32","bf16"}:
        raise ValueError("native quantized-product output storage differs")
    if activated and (out!="bf16" or n%2):
        raise ValueError("native quantized gate/up requires BF16 parent and paired width")
    specs: list[tuple[list[int],str,int,int] | None] = [None]*total
    specs[roles[0]] = ([m,k],"bf16",-1,0)
    if expanded:
        code = rows[roles[1]].get("storage")
        global_shape = rows[roles[3]].get("shape")
        if (code not in ("i8","ui8") or not isinstance(global_shape,list) or len(global_shape)!=1
                or type(global_shape[0]) is not int or not 0<global_shape[0]<=n):
            raise ValueError("native quantized-product expansion roots differ")
        specs[roles[1]] = ([n,k//2],code,-1,1)
        specs[roles[2]] = ([n,k//16],"f8E4M3FN",-1,1)
        specs[roles[3]] = (global_shape,"f64",-1,1)
        specs[4:] = [([m,k],"f8E4M3FN",0,2),([m,k//16],"f32",0,2),
                     ([n,k],"f8E4M3FN",1,2),([k//16,n],"f32",1,2),([m,n],out,2,3)]
        if activated:
            specs.append(([m,n//2],"bf16",3,4))
    else:
        specs[roles[1]] = ([n,k],"f8E4M3FN",-1,1)
        specs[roles[2]] = ([k//16,n],"f32",-1,1)
        specs[3:] = [([m,k],"f8E4M3FN",0,1),([m,k//16],"f32",0,1),([m,n],out,1,2)]
    sizes = {"bf16":2,"f8E4M3FN":1,"f32":4,"i8":1,"ui8":1,"f64":8}
    for i,(row,spec) in enumerate(zip(rows,specs,strict=True)):
        if spec is None:
            raise ValueError("native quantized-product missing buffer role")
        shape,storage,write,read = spec
        byte_count = math.prod(shape)*sizes[storage]
        expected = {"id":i,"shape":shape,"storage":storage,"bytes":byte_count,
                    "ownership":"readonly_input" if i<roots else "returned_output" if i==output else "private_scratch",
                    "first_write":write,"last_read":read}
        if (byte_count>2**63-1 or row != expected
                or any(type(row.get(key)) is not int for key in ("id","bytes","first_write","last_read"))
                or any(type(d) is not int for d in row["shape"])):
            raise ValueError("native quantized-product buffer storage/lifetime differs")
    return roles,BlockScaleShape(m,n,k,16,1,"nk",out)


def _variant_ranges(shape):
    """Maximal M ranges in [1, capacity] where the static rule takes an LDS panel.

    A shape-free product runs every M; its base image is the register panel
    for M >= 1, and each range gets the LDS image the static rule selects
    there (GFX1201-BOUNDED-PER-M-PANEL-20261030). Register panels are not
    split further: 16 vs 32 rows measured mixed across the 32-row band."""
    from .rocm_fp8_blockscale import blockscale_panel_oracle
    ranges: list[list[Any]] = []
    for m in range(1, shape.m + 1):
        panel = blockscale_panel_oracle(shape, min_m=m)
        if panel.staging != "lds":
            continue
        if ranges and ranges[-1][2] == panel and ranges[-1][1] == m - 1:
            ranges[-1][1] = m
        else:
            ranges.append([m, m, panel])
    if len(ranges) > 8:
        raise ValueError("native quantized-product LDS panel ranges exceed eight variants")
    return tuple((lo, hi) for lo, hi, _ in ranges)


@dataclass(frozen=True)
class ProductVariant:
    """An LDS product image for active M in [min_m, max_m] of a shape-free product."""
    min_m: int
    max_m: int
    recipe: BlockScaleProgram
    native: ROCMNativePackage
    checked_target_ir: str


def _compile_product(tool, shape, child, *, arbitrary_m):
    """Graph -> Schedule -> Tile -> image for one product member Graph."""
    schedule = run_tessera_opt(tool,child,"--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
    recipe = BlockScaleProgram(shape,"quantized_product",child,schedule,tile)
    product = package_blockscale(recipe,pipeline_name="tessera-rocm-executable",
                                 project_image_identity=arbitrary_m,arbitrary_m=arbitrary_m)
    checked_target = product.target_ir
    if arbitrary_m:
        # The image is the shape-free projection; retain the static checked
        # Target separately, as the bounded NVFP4 consumer does.
        from .rocm_native import _run_opt
        from .rocm_pipeline import ROCMExecutablePipeline, ROCMInputLevel, ROCMOutputLevel
        config = ROCMExecutablePipeline(family="matmul", arch="gfx1201", input_level=ROCMInputLevel.TILE)
        checked_target = _run_opt(tool, tile, config.pass_pipeline(output=ROCMOutputLevel.TARGET))
    return recipe, product, checked_target, schedule


def _check_product_image(shape, native, checked_target_ir, arbitrary_m):
    """Validate one product image against its checked Target and the consumer ABI."""
    native.descriptor.validate_image(native.image)
    checked = check_blockscale_target_ir(shape,native.tile_ir,checked_target_ir)
    if arbitrary_m:
        from .rocm_native import _shape_free_target_ir
        projection=_shape_free_target_ir(checked_target_ir,
            family="scaled_matmul" if checked["staging"]=="global" else "scaled_matmul_lds",
            directive="tessera_rocm.scaled_wmma_gemm",
            runtime_k=bool(native.descriptor.provenance.get("lds_runtime_k")),arbitrary_m=True)
        if (native.target_ir != projection
                or native.descriptor.provenance.get("arbitrary_m") is not True
                or native.descriptor.provenance.get("lds_whole_m") is not False):
            raise ValueError("native quantized-product arbitrary-M image differs from its checked Target")
    elif (native.target_ir != checked_target_ir
            or native.descriptor.provenance.get("arbitrary_m") is not False):
        raise ValueError("native quantized-product checked Target differs from image-bound Target")
    descriptor = native.descriptor
    if (native.image.target != "rocm_gfx1201" or native.image.architecture != "gfx1201"
            or native.image.binary_format != "hsaco" or native.image.pipeline_name != "tessera-rocm-executable"
            or native.image.target_ir_digest != _digest(native.target_ir)
            or descriptor.abi_id != PACKAGE_ABIS[("nk",shape.output)]
            or descriptor.entry_symbol != _directive_symbol(native.target_ir,"tessera_rocm.scaled_wmma_gemm")
            or descriptor.geometry != LaunchGeometry(policy="rocm_wmma_macro_tile_grid")
            or descriptor.scalars != tuple(ScalarArgument(i+5,name,"int64") for i,name in enumerate(("M","N","K")))
            or descriptor.provenance["shape"] != [shape.m,shape.n,shape.k]
            or descriptor.provenance["scale_k"] != 16 or descriptor.provenance["scale_n"] != 1
            or descriptor.provenance["macro_tile"] != [checked["block_m"],checked["block_n"]]
            or descriptor.provenance["workgroup"] != [32*checked["warps"],1,1]):
        raise ValueError("native quantized-product consumer ABI/geometry differs")
    return checked


@dataclass(frozen=True)
class NativeQuantizedProductProgram:
    program_json: str
    graph_ir: str
    quantizer_recipe: BlockQuantRecipe
    quantizer: ROCMNativePackage
    product_recipe: BlockScaleProgram
    product: ROCMNativePackage
    checked_product_target_ir: str
    binding_digest: str
    expansion: NVFP4ExpansionPackage | None = None
    activation_recipe: ParentViewRecipe | None = None
    activation: ROCMNativePackage | None = None
    # A bounded MLP component's product image guards arbitrary M. Its checked
    # static Target remains the witness; the image is its shape-free projection.
    arbitrary_m: bool = False
    # A K-bounded MLP's gate component also expands its weights through the
    # runtime-K image and contracts through a shape-free (global) product.
    runtime_k: bool = False
    # An N-bounded MLP's down component expands through the runtime-N image.
    runtime_n: bool = False
    # Per-active-M LDS images of a shape-free product (ascending, disjoint).
    variants: tuple[ProductVariant, ...] = ()

    def _identity(self):
        parts = [self.program_json,self.graph_ir,self.quantizer_recipe.graph_ir,
                 self.quantizer_recipe.schedule_ir,self.quantizer_recipe.tile_ir,
                 self.product_recipe.graph_ir,self.product_recipe.schedule_ir,self.product_recipe.tile_ir,
                 self.checked_product_target_ir]
        if self.expansion is not None:
            parts.extend([self.expansion.graph_ir,self.expansion.schedule_ir])
        if self.activation_recipe is not None:
            parts.extend([self.activation_recipe.graph_ir,self.activation_recipe.schedule_ir,self.activation_recipe.tile_ir])
        for native in self.native_stages:
            parts.extend([native.tile_ir,native.target_ir,
                          native.image.image_digest,native.descriptor.descriptor_digest])
        if self.arbitrary_m:
            parts.append("arbitrary_m")
        if self.runtime_k:
            parts.append("runtime_k")
        if self.runtime_n:
            parts.append("runtime_n")
        for v in self.variants:
            parts.extend([str(v.min_m),str(v.max_m),v.recipe.graph_ir,v.recipe.schedule_ir,v.recipe.tile_ir,
                          v.checked_target_ir,v.native.tile_ir,v.native.target_ir,
                          v.native.image.image_digest,v.native.descriptor.descriptor_digest])
        return _digest(json.dumps(parts,separators=(",",":")))

    @property
    def schedule_irs(self):
        stages: tuple[str,...]=(self.quantizer_recipe.schedule_ir,)
        if self.expansion is not None:stages+=(self.expansion.schedule_ir,)
        stages+=(self.product_recipe.schedule_ir,)
        if self.activation_recipe is not None:stages+=(self.activation_recipe.schedule_ir,)
        return stages

    @property
    def native_stages(self):
        stages: tuple[ROCMNativePackage,...]=(self.quantizer,)
        if self.expansion is not None:stages+=(self.expansion.native,)
        stages+=(self.product,)
        if self.activation is not None:stages+=(self.activation,)
        return stages

    @memoized_validation
    def validate(self):
        if self.binding_digest != self._identity():
            raise ValueError("native quantized-product member binding differs")
        plan = json.loads(self.program_json)
        roles,shape = _frame_shape(plan)
        if (plan["schema"] in (_EXPANDED_SCHEMA,_ACTIVATED_SCHEMA)) != (self.expansion is not None):
            raise ValueError("native quantized-product expansion schema/member differs")
        activated=plan["schema"]==_ACTIVATED_SCHEMA
        if (activated!=(self.activation is not None) or activated!=(self.activation_recipe is not None)):
            raise ValueError("native quantized-product activation schema/member differs")
        members = [self.quantizer_recipe.graph_ir]
        if self.expansion is not None:
            self.expansion.validate()
            members.append(self.expansion.graph_ir)
            _,n,k,offsets,_,_ = self.expansion.contract()
            shapes = ([n,k//2],[n,k//16],[len(offsets)-1],[n,k],[k//16,n])
            storages = (plan["buffers"][roles[1]]["storage"],"f8E4M3FN","f64","f8E4M3FN","f32")
            widths = (1,1,8,1,4)
            for slot,dims,storage,width in zip((*roles[1:],6,7),shapes,storages,widths,strict=True):
                row = plan["buffers"][slot]
                if row["shape"]!=dims or row["storage"]!=storage or row["bytes"]!=math.prod(dims)*width:
                    raise ValueError("native quantized-product expansion/shared buffer interface differs")
        members.append(self.product_recipe.graph_ir)
        expected=list(plan["member_graphs"])
        if self.arbitrary_m:
            # The shape-free product's member requests the panel for M >= 1.
            product_index=2 if self.expansion is not None else 1
            expected[product_index]=admitted_rows_graph(
                expected[product_index],global_staging=self.runtime_k or self.runtime_n)
        if self.activation_recipe is not None:
            members.append(self.activation_recipe.graph_ir)
            if self.runtime_n:
                # A runtime-N gate/up product's SiLU reads the runtime half width:
                # its member is the plan's Graph with the runtime-width request.
                expected[-1]=runtime_width_graph(expected[-1])
        if (plan["source_graph_ir"] != self.graph_ir
                or expected != members
                or self.product_recipe.shape != shape
                or self.quantizer.tile_ir != self.quantizer_recipe.tile_ir
                or self.product.tile_ir != self.product_recipe.tile_ir):
            raise ValueError("native quantized-product projected Graph differs")
        from tessera.runtime import RuntimeArtifact
        quant = self.quantizer
        quant_bound=validate_block_quant_runtime_artifact(RuntimeArtifact(
            graph_ir=self.quantizer_recipe.graph_ir,schedule_ir=self.quantizer_recipe.schedule_ir,
            tile_ir=quant.tile_ir,target_ir=quant.target_ir,native_image=quant.image,
            launch_descriptor=quant.descriptor,
            metadata={"target":"rocm_gfx1201","compiler_path":"rocm_block_quant_native_descriptor"}))
        quant_buffers=json.loads(quant_bound.program_json)["buffers"]
        for local,shared in zip(quant_buffers,(plan["buffers"][slot] for slot in (roles[0],len(roles),len(roles)+1)),strict=True):
            if any(local[key]!=shared[key] for key in ("shape","storage","bytes")):
                raise ValueError("native quantized-product producer/shared buffer interface differs")
        if self.activation is not None and self.activation_recipe is not None:
            if self.activation.tile_ir!=self.activation_recipe.tile_ir:
                raise ValueError("native quantized-product activation Tile differs")
            act=self.activation
            act.descriptor.validate_image(act.image)
            bound=validate_parent_view_runtime_artifact(RuntimeArtifact(
                graph_ir=self.activation_recipe.graph_ir,schedule_ir=self.activation_recipe.schedule_ir,
                tile_ir=act.tile_ir,target_ir=act.target_ir,native_image=act.image,
                launch_descriptor=act.descriptor,metadata={"target":"rocm_gfx1201"}))
            _,act_member=_parent_view_frame(act.target_ir)
            if runtime_width_profile(act_member)!=self.runtime_n:
                raise ValueError("native quantized-product activation width policy differs from its expansion")
            for local,shared in zip(json.loads(bound.program_json)["buffers"],(plan["buffers"][8],plan["buffers"][9]),strict=True):
                if any(local[key]!=shared[key] for key in ("shape","storage","bytes")):
                    raise ValueError("native quantized-product activation/shared buffer interface differs")
        native = self.product
        native.descriptor.validate_image(native.image)
        if (type(self.arbitrary_m) is not bool or type(self.runtime_k) is not bool
                or type(self.runtime_n) is not bool):
            raise ValueError("native quantized-product arbitrary-M policy differs")
        if self.runtime_k != (self.expansion is not None and self.expansion.runtime_k):
            raise ValueError("native quantized-product runtime-K expansion policy differs")
        if self.runtime_n != (self.expansion is not None and self.expansion.runtime_n):
            raise ValueError("native quantized-product runtime-N expansion policy differs")
        checked = _check_product_image(shape,native,self.checked_product_target_ir,self.arbitrary_m)
        if (self.runtime_k or self.runtime_n) and (
                not self.arbitrary_m or checked["staging"] != "global"
                or native.descriptor.provenance.get("runtime_shape_image") is not True):
            # Only the global-staged image reads M, N and K entirely at launch.
            raise ValueError("native quantized-product runtime K/N requires the shape-free global-staged product")
        # A shape-free M-only product carries exactly the LDS variants the static
        # rule implies over [1, capacity]; each is its own checked projection.
        ranges = (_variant_ranges(shape) if self.arbitrary_m and not (self.runtime_k or self.runtime_n)
                  else ())
        if tuple((v.min_m,v.max_m) for v in self.variants) != ranges:
            raise ValueError("native quantized-product per-M variant ranges differ from the static rule")
        member = plan["member_graphs"][2 if self.expansion is not None else 1]
        from .rocm_fp8_blockscale import blockscale_panel_oracle
        for v in self.variants:
            panel = blockscale_panel_oracle(shape, min_m=v.min_m)
            if (not isinstance(v,ProductVariant) or v.recipe.shape != shape
                    or v.recipe.graph_ir != admitted_rows_graph(member,min_m=v.min_m)
                    or v.native.tile_ir != v.recipe.tile_ir):
                raise ValueError("native quantized-product per-M variant Graph differs")
            got = _check_product_image(shape,v.native,v.checked_target_ir,True)
            if (got["staging"],got["block_m"],got["block_n"],got["warps"]) != (
                    panel.staging,panel.macro_tile_m,panel.macro_tile_n,panel.warps):
                raise ValueError("native quantized-product per-M variant panel differs from the static rule")

    def to_manifest(self,*,_bounded_member=False):
        from .rocm_nvfp4_resident import _portable_stage
        self.validate()
        if self.arbitrary_m and not _bounded_member:
            # A shape-free product image is portable only inside the bounded
            # MLP record whose row frame admits it; alone it has no row policy.
            raise ValueError("bounded quantized-product components have no standalone portable record")
        stages = [_portable_stage(self.quantizer_recipe.graph_ir,self.quantizer_recipe.schedule_ir,self.quantizer)]
        if self.expansion is not None:
            stages.append(_portable_stage(self.expansion.graph_ir,self.expansion.schedule_ir,self.expansion.native))
        stages.append(_portable_stage(self.product_recipe.graph_ir,self.product_recipe.schedule_ir,self.product))
        if self.activation is not None and self.activation_recipe is not None:
            stages.append(_portable_stage(self.activation_recipe.graph_ir,self.activation_recipe.schedule_ir,self.activation))
        version=3 if self.activation is not None else 2 if self.expansion is not None else 1
        record: dict[str,Any]={"schema":_PACKAGE_SCHEMAS[version-1],
                "program_json":self.program_json,"graph_ir":self.graph_ir,
                "checked_product_target_ir":self.checked_product_target_ir,
                "binding_digest":self.binding_digest,
                "stages":stages}
        if self.arbitrary_m:
            # The flag is a claim only; replay re-derives the shape-free
            # projection from the checked static Target before accepting it.
            record["arbitrary_m"]=True
        if self.runtime_k:
            record["runtime_k"]=True
        if self.runtime_n:
            record["runtime_n"]=True
        if self.variants:
            # Each variant replays exactly as the base product: its checked
            # static Target is retained and its image re-derived from it.
            record["variants"]=[{"min_m":v.min_m,"max_m":v.max_m,
                                 "checked_product_target_ir":v.checked_target_ir,
                                 "stage":_portable_stage(v.recipe.graph_ir,v.recipe.schedule_ir,v.native)}
                                for v in self.variants]
        return record

    @classmethod
    def from_manifest(cls,record,*,_bounded_member=False):
        if isinstance(record,dict) and record.get("schema") in tuple(
                f"tessera.rocm.quantized_product_package.v{i}" for i in (4,5,6,7,8)):
            if _bounded_member:
                raise ValueError("native quantized MLP portable component schemas differ")
            from .rocm_quantized_mlp_program import NativeQuantizedMLPProgram
            return NativeQuantizedMLPProgram.from_manifest(record)
        from .native_artifact import NativeImageArtifact,LaunchDescriptor
        keys={"schema","program_json","graph_ir","checked_product_target_ir","binding_digest","stages"}
        arbitrary_m=isinstance(record,dict) and "arbitrary_m" in record
        runtime_k=isinstance(record,dict) and "runtime_k" in record
        runtime_n=isinstance(record,dict) and "runtime_n" in record
        if arbitrary_m and (not _bounded_member or record["arbitrary_m"] is not True):
            raise ValueError("native quantized-product portable arbitrary-M claim differs")
        if runtime_k and (not arbitrary_m or record["runtime_k"] is not True):
            raise ValueError("native quantized-product portable runtime-K claim differs")
        if runtime_n and (not arbitrary_m or record["runtime_n"] is not True):
            raise ValueError("native quantized-product portable runtime-N claim differs")
        has_variants=isinstance(record,dict) and "variants" in record
        if has_variants and (not arbitrary_m or not isinstance(record["variants"],list) or not record["variants"]):
            raise ValueError("native quantized-product portable per-M variants differ")
        claims=(({"arbitrary_m"} if arbitrary_m else set())|({"runtime_k"} if runtime_k else set())
                |({"runtime_n"} if runtime_n else set())|({"variants"} if has_variants else set()))
        if (not isinstance(record,dict) or set(record)!=keys|claims
                or record["schema"] not in _PACKAGE_SCHEMAS
                or any(not isinstance(record[key],str) for key in ("program_json","graph_ir","checked_product_target_ir","binding_digest"))
                or not isinstance(record["stages"],list) or len(record["stages"])!=_PACKAGE_SCHEMAS.index(record["schema"])+2):
            raise ValueError("native quantized-product portable record differs")
        plan=json.loads(record["program_json"])
        _,shape=_frame_shape(plan)
        version=_PACKAGE_SCHEMAS.index(record["schema"])+1
        expanded,activated=version>=2,version==3
        if plan["schema"]!=_NATIVE_SCHEMAS[version-1]:
            raise ValueError("native quantized-product portable schema/frame differs")
        stages=[]
        for stage in record["stages"]:
            if (not isinstance(stage,dict) or set(stage)!={"graph_ir","schedule_ir","tile_ir","target_ir","image","descriptor"}
                    or any(not isinstance(stage[key],str) for key in ("graph_ir","schedule_ir","tile_ir","target_ir"))):
                raise ValueError("native quantized-product portable member differs")
            native=ROCMNativePackage(stage["tile_ir"],stage["target_ir"],"",
                NativeImageArtifact.from_dict(stage["image"]),LaunchDescriptor.from_dict(stage["descriptor"]))
            stages.append((stage["graph_ir"],stage["schedule_ir"],native))
        first,second=stages[0],stages[2 if expanded else 1]
        expansion=NVFP4ExpansionPackage(*stages[1]) if expanded else None
        activation_recipe=ParentViewRecipe(stages[3][0],stages[3][1],stages[3][2].tile_ir) if activated else None
        activation=stages[3][2] if activated else None
        variants=[]
        for entry in record.get("variants",()):
            stage=entry.get("stage") if isinstance(entry,dict) else None
            if (not isinstance(entry,dict) or set(entry)!={"min_m","max_m","checked_product_target_ir","stage"}
                    or type(entry["min_m"]) is not int or type(entry["max_m"]) is not int
                    or not isinstance(entry["checked_product_target_ir"],str)
                    or not isinstance(stage,dict)
                    or set(stage)!={"graph_ir","schedule_ir","tile_ir","target_ir","image","descriptor"}
                    or any(not isinstance(stage[key],str) for key in ("graph_ir","schedule_ir","tile_ir","target_ir"))):
                raise ValueError("native quantized-product portable per-M variants differ")
            v_native=ROCMNativePackage(stage["tile_ir"],stage["target_ir"],"",
                NativeImageArtifact.from_dict(stage["image"]),LaunchDescriptor.from_dict(stage["descriptor"]))
            variants.append(ProductVariant(entry["min_m"],entry["max_m"],
                BlockScaleProgram(shape,"quantized_product",stage["graph_ir"],stage["schedule_ir"],stage["tile_ir"]),
                v_native,entry["checked_product_target_ir"]))
        program=cls(record["program_json"],record["graph_ir"],
            BlockQuantRecipe(first[0],first[1],first[2].tile_ir),first[2],
            BlockScaleProgram(shape,"quantized_product",second[0],second[1],second[2].tile_ir),second[2],
            record["checked_product_target_ir"],record["binding_digest"],expansion,activation_recipe,activation,
            arbitrary_m=arbitrary_m,runtime_k=runtime_k,runtime_n=runtime_n,variants=tuple(variants))
        program.validate()
        return program

    def prepare(self, inputs, *, runtime_library):
        return PreparedQuantizedProductProgram(self,inputs,runtime_library=runtime_library)


def package_quantized_product_graph(graph_ir,*,arbitrary_m=False,runtime_k=False,runtime_n=False):
    if type(arbitrary_m) is not bool or type(runtime_k) is not bool or type(runtime_n) is not bool:
        raise ValueError("native quantized-product arbitrary-M policy must be a bool")
    if (runtime_k or runtime_n) and not arbitrary_m:
        raise ValueError("native quantized-product runtime K/N requires arbitrary M")
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("native quantized product requires matching tessera-opt")
    plan = _export(tool,graph_ir)
    if plan["schema"] in _MLP_SCHEMAS:
        if arbitrary_m or runtime_k or runtime_n:
            raise ValueError("a bounded MLP derives its component policy from its own frame")
        from .rocm_quantized_mlp_program import package_quantized_mlp_plan
        return package_quantized_mlp_plan(plan,tool)
    _,shape = _frame_shape(plan)
    # Replay normalized source before accepting a package witness.
    if _export(tool,plan["source_graph_ir"]) != plan:
        raise ValueError("native quantized-product normalized source replay differs")
    quantizer_recipe = lower_block_quant_text(plan["member_graphs"][0])
    quantizer = package_block_quant_recipe(quantizer_recipe)
    expanded=plan["schema"] in (_EXPANDED_SCHEMA,_ACTIVATED_SCHEMA)
    activated=plan["schema"]==_ACTIVATED_SCHEMA
    if (runtime_k or runtime_n) and not expanded:
        raise ValueError("native quantized-product runtime K/N requires its NVFP4 expansion")
    expansion = (package_nvfp4_expansion(plan["member_graphs"][1],runtime_k=runtime_k,runtime_n=runtime_n)
                 if expanded else None)
    child = plan["member_graphs"][2 if expanded else 1]
    if arbitrary_m:
        # The shape-free image runs every M down to 1; its panel is chosen there.
        # A runtime-K/N image exists only as the global-staged body.
        child = admitted_rows_graph(child,global_staging=runtime_k or runtime_n)
    recipe, product, checked_target, schedule = _compile_product(tool, shape, child, arbitrary_m=arbitrary_m)
    verify_blockscale_schedule(shape,schedule,min_m=1 if arbitrary_m else None,
                               staging="global" if runtime_k or runtime_n else None)
    variants: tuple[ProductVariant, ...] = ()
    if arbitrary_m and not (runtime_k or runtime_n):
        # One LDS image per range where the static rule takes an LDS panel.
        member = plan["member_graphs"][2 if expanded else 1]
        built = []
        for lo, hi in _variant_ranges(shape):
            v_recipe, v_product, v_checked, v_schedule = _compile_product(
                tool, shape, admitted_rows_graph(member, min_m=lo), arbitrary_m=True)
            verify_blockscale_schedule(shape, v_schedule, min_m=lo)
            built.append(ProductVariant(lo, hi, v_recipe, v_product, v_checked))
        variants = tuple(built)
    activation_recipe,activation=None,None
    if activated:
        graph=plan["member_graphs"][3]
        if runtime_n:
            graph=runtime_width_graph(graph)
        schedule=run_tessera_opt(tool,graph,"--tessera-graph-to-schedule")
        tile=run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
        activation_recipe=ParentViewRecipe(graph,schedule,tile)
        activation=package_parent_view_recipe(activation_recipe,pipeline_name="tessera-rocm-executable")
    program = NativeQuantizedProductProgram(json.dumps(plan,sort_keys=True),plan["source_graph_ir"],
        quantizer_recipe,quantizer,recipe,product,checked_target,"",expansion,activation_recipe,activation,arbitrary_m,runtime_k,runtime_n,variants)
    program = replace(program,binding_digest=program._identity())
    program.validate()
    return program


class PreparedQuantizedProductProgram(PreparedScaledProgram):
    package: NativeQuantizedProductProgram
    _prepare_symbol = "tessera_rocm_program_prepare_multi_output"
    _step_type = _MultiOutputStep

    def __init__(self,package,inputs,*,runtime_library):
        from tessera import runtime as rt
        if rt._rocm_live_arch() != "gfx1201":
            raise RuntimeError("native quantized product requires owning gfx1201")
        super().__init__(package,inputs,runtime_library=runtime_library)

    @memoized_binding
    def _bind_abi(self,package):
        package.validate()
        plan = json.loads(package.program_json)
        roles,shape = _frame_shape(plan)
        rows,native = plan["buffers"],package.native_stages
        blobs = tuple(C.create_string_buffer(p.image.payload) for p in native)
        entries = tuple(p.descriptor.entry_symbol.encode() for p in native)
        steps = (_MultiOutputStep*len(native))()
        for i,p in enumerate(native):
            step,spec = steps[i],plan["steps"][i]
            step.image,step.image_bytes,step.entry = C.cast(blobs[i],C.c_void_p),len(p.image.payload),entries[i]
            step.input_count,step.output_count = len(spec["inputs"]),len(spec["outputs"])
            step.inputs[:step.input_count],step.outputs[:step.output_count] = spec["inputs"],spec["outputs"]
            if i==0:
                _,member = _frame(p.target_ir)
                scalars,geometry = member["scalars"],member["geometry"]
            elif package.expansion is not None and i==1:
                grid,workgroup=p.descriptor.geometry.grid,p.descriptor.geometry.workgroup
                if grid is None or workgroup is None:
                    raise ValueError("native quantized-product expansion geometry differs")
                # The runtime-K image binds its capacity K16 group count.
                # A runtime-K image binds its capacity K16 groups, runtime-N its capacity N.
                contract=package.expansion.contract()
                scalars=([contract[2]//16] if package.expansion.runtime_k else
                         [contract[1]] if package.expansion.runtime_n else [])
                geometry=[*grid,*workgroup]
            elif package.activation is not None and i==len(native)-1:
                _,member=_parent_view_frame(p.target_ir)
                scalars,geometry=member["scalars"],member["geometry"]
            else:
                bm,bn = p.descriptor.provenance["macro_tile"]
                scalars = [shape.m,shape.n,shape.k]
                geometry = [(shape.n+bn-1)//bn,(shape.m+bm-1)//bm,1,*p.descriptor.provenance["workgroup"]]
            step.scalar_count = len(scalars)
            step.scalars[:step.scalar_count],step.geometry[:] = scalars,geometry
        ownership = {"readonly_input":0,"private_scratch":1,"returned_output":2}
        buffers = (_Buffer*len(rows))(*[
            _Buffer(b["bytes"],math.prod(b["shape"]),b["first_write"],b["last_read"],ownership[b["ownership"]],0)
            for b in rows])
        import numpy as np
        dtypes={"i8":np.dtype("uint8"),"ui8":np.dtype("uint8"),"f64":np.dtype("float64")}
        storage = tuple(_StorageSpec(tuple(b["shape"]),b["storage"],b["bytes"],dtypes.get(b["storage"],_storage_numpy_dtype(b["storage"]))) for b in rows)
        return _BoundScaledABI(buffers,steps,blobs,entries,len(roles),storage,(plan["output"],),())

    def _inputs(self,inputs):
        if self.package.expansion is not None:
            from .nvfp4_expansion import checked_expansion_inputs,nvfp4_expansion_policy
            roles,_=_frame_shape(json.loads(self.package.program_json))
            if len(inputs)!=len(roles):
                raise ValueError("native quantized-product expansion requires four roots")
            _,_,_,offsets,_,_=self.package.expansion.contract()
            checked_expansion_inputs(*(inputs[i] for i in roles[1:]),row_offsets=offsets,numeric_policy=nvfp4_expansion_policy(),_scale_bounds=self._nvfp4_scale_bounds())
        return super()._inputs(inputs)

    def invoke_range(self,first,count,repeats=1,*,timed=True):
        if (type(first) is not int or type(count) is not int or first<0 or count<=0
                or first+count>len(self._binding.steps) or type(repeats) is not int
                or not 1<=repeats<=1048576):
            raise ValueError("native quantized-product invocation range differs")
        from .native_scaled_program import _status
        invoke=self.lib.tessera_rocm_program_invoke_range
        invoke.argtypes=[C.c_uint64,C.c_uint32,C.c_uint32,C.c_uint32,C.POINTER(C.c_uint64),C.POINTER(C.c_float)]
        invoke.restype=C.c_int
        generation,elapsed=C.c_uint64(),C.c_float()
        _status(invoke(self.handle,first,count,repeats,C.byref(generation),C.byref(elapsed) if timed else None))
        return generation.value,float(elapsed.value) if timed else None
