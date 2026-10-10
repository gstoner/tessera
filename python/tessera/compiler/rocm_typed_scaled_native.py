"""Canonical typed FP8/MXFP8 primal packaging from the original frontend Graph."""
import math
from dataclasses import dataclass, replace
from .rocm_fp8_blockscale import BlockScaleShape, BlockScaleProgram
from .scheduled_matmul import find_tessera_opt, run_tessera_opt

def requests_typed_scaled(module):
    if len(module.functions)!=1:return False
    fn=module.functions[0]
    if len(fn.body)!=1:return False
    op=fn.body[0]
    args={arg.name:arg.ir_type for arg in fn.args}
    lhs=args.get(op.operands[0].removeprefix("%")) if op.operands else None
    return (op.op_name=="tessera.scaled_matmul" and not op.kwargs.get("physical_contract")
            and len(fn.args)==4 and lhs is not None and lhs.dtype=="fp8_e4m3")


def requests_composed_typed_scaled(module):
    """Recognize frontend product/sum intent; concrete admission is separate."""
    if len(module.functions)!=1:return False
    fn=module.functions[0]
    if not 2<=len(fn.body)<=128:return False
    arguments={arg.name:arg.ir_type for arg in fn.args}
    products=0
    permutations=0
    for op in fn.body:
        if op.op_name=="tessera.add":continue
        if op.op_name=="tessera.transpose":
            permutations+=1
            continue
        if op.op_name!="tessera.scaled_matmul" or len(op.operands)!=4:return False
        lhs=arguments.get(op.operands[0].removeprefix("%"))
        if lhs is None or lhs.dtype!="fp8_e4m3" or op.kwargs.get("physical_contract"):return False
        products+=1
    return products>=2 or (products>=1 and permutations>=1)


def requests_floating_scaled(module):
    """Recognize continuous Graph products; this is not primal admission."""
    if len(module.functions) != 1:
        return False
    fn = module.functions[0]
    args = {arg.name: arg.ir_type for arg in fn.args}
    products = 0
    for op in fn.body:
        if op.op_name in {"tessera.add", "tessera.transpose"}:
            continue
        if (op.op_name != "tessera.scaled_matmul" or len(op.operands) != 4
                or op.kwargs.get("physical_contract")
                or any(args.get(name.removeprefix("%")) is None
                       or args[name.removeprefix("%")].dtype != "fp32"
                       for name in op.operands)):
            return False
        products += 1
    return products > 0

@dataclass(frozen=True)
class _LogicalScaleShape:
    # Semantic group extents do not imply an aligned WMMA primal schedule.
    m: int
    n: int
    k: int
    scale_k: int
    scale_n: int
    weight_layout: str
    output: str

    @property
    def groups(self):
        return (self.k+self.scale_k-1)//self.scale_k

    @property
    def n_groups(self):
        return (self.n+self.scale_n-1)//self.scale_n

def contract(module, *, semantic_only=False):
    floating = semantic_only and requests_floating_scaled(module)
    if not requests_typed_scaled(module) and not floating:return None
    fn=module.functions[0];op=fn.body[0]
    if (len(op.operands)!=4 or len(op.result_names)!=1 or
        len(set(op.operands))!=4 or
        set(x.removeprefix("%") for x in op.operands)!={a.name for a in fn.args} or
        [x.removeprefix("%") for x in fn.return_values]!=op.result_names or
        type(op.kwargs.get("transposeA",False)) is not bool or
        (op.kwargs.get("transposeA",False) is not False and not floating and
         op.kwargs.get("batching") not in (None,"broadcast")) or
        type(op.kwargs.get("transposeB",False)) is not bool or
        op.kwargs.get("batching") not in (None,"shared_rhs_rows","independent_rhs","shared_lhs","broadcast") or
        op.kwargs.get("numeric_policy")!={"accum":"fp32","execution_mode":"exact_per_block"}):
        return None
    layout=op.kwargs.get("scale_layout")
    if not isinstance(layout,dict) or set(layout)!={"granularity","block","format"} or layout["granularity"]!="block":
        return None
    block=layout["block"];fmt=layout["format"]
    if not isinstance(block,(list,tuple)) or len(block)!=2 or any(type(v) is not int or v<=0 for v in block):
        return None
    if fmt not in {"fp32","e8m0"} or (fmt=="e8m0" and list(block)!=[1,32]):return None
    args={a.name:a.ir_type for a in fn.args}
    types=[args[x.removeprefix("%")] for x in op.operands]
    policy=op.kwargs.get("batching")
    if floating:
        if fmt != "fp32" or any(t.dtype != "fp32" for t in types):
            return None
        a,b,sa,sb=types
        if policy in {"shared_lhs","shared_rhs_rows","independent_rhs"}:
            lhs_batched=policy != "shared_lhs"
            rhs_batched=policy != "shared_rhs_rows"
            prefix=a.shape[:-2] if lhs_batched else b.shape[:-2]
            if (not prefix or a.shape[:-2] != (prefix if lhs_batched else ())
                    or sa.shape[:-2] != a.shape[:-2]
                    or b.shape[:-2] != (prefix if rhs_batched else ())
                    or sb.shape[:-2] != b.shape[:-2]):
                return None
        elif policy is None and any(t.rank != 2 for t in types):
            return None
        return _independent_scale_semantics(fn,op,types,block,fmt)
    scalar_plane = (policy is None and all(t.rank == 2 for t in types) and
        (op.kwargs.get("transposeA",False) or
         int(types[0].shape[-1]) % block[1] != 0))
    if policy == "broadcast" or scalar_plane:
        info=_independent_scale_semantics(fn,op,types,block,fmt)
        if semantic_only or info is None:
            return info
        logical,fmt,names,output=info
        # Admission reflects the native WMMA group profile; semantic reverse
        # remains independent of aligned primal constraints.
        if logical.scale_k % 16 or logical.k > 2**63-logical.scale_k:return None
        shape_type=_LogicalScaleShape if logical.k % logical.scale_k else BlockScaleShape
        shape=shape_type(logical.m,logical.n,logical.k,logical.scale_k,
                              logical.scale_n,logical.weight_layout,logical.output)
        return shape,fmt,names,output
    lhs_batched=policy in ("shared_rhs_rows","independent_rhs")
    rhs_batched=policy in ("independent_rhs","shared_lhs")
    batch_shape=types[0].shape[:-2] if lhs_batched else types[1].shape[:-2] if rhs_batched else ()
    if policy is not None and not batch_shape:return None
    ranks=(len(batch_shape)+2 if lhs_batched else 2,len(batch_shape)+2 if rhs_batched else 2,
           len(batch_shape)+2 if lhs_batched else 2,len(batch_shape)+2 if rhs_batched else 2)
    if any(t.rank!=rank or any(not str(v).isdigit() or int(v)<=0 for v in t.shape)
           for t,rank in zip(types,ranks,strict=True)):
        return None
    a,b,sa,sb=types
    if a.dtype!="fp8_e4m3" or b.dtype!="fp8_e4m3":return None
    if sa.dtype!=("fp32" if fmt=="fp32" else "uint8") or sb.dtype!=sa.dtype:return None
    rows,k=map(int,a.shape[-2:])
    batches=math.prod(map(int,batch_shape)) if batch_shape else 1
    if batches*rows>2**63-1 or batches>2**31-1:return None
    nk=op.kwargs.get("transposeB",False)
    n,kb=map(int,b.shape[-2:]) if nk else tuple(map(int,b.shape[-2:][::-1]))
    if kb!=k or (rhs_batched and b.shape[:-2]!=batch_shape):return None
    m=batches*rows if policy=="shared_rhs_rows" else rows
    shape_type = _LogicalScaleShape if semantic_only else BlockScaleShape
    shape=shape_type(m,n,k,block[1],block[0],"nk" if nk else "kn","f32")
    expected_sa=(*map(int,batch_shape),rows,shape.groups) if lhs_batched else (rows,shape.groups)
    expected_sb=(*map(int,batch_shape),shape.groups,shape.n_groups) if rhs_batched else (shape.groups,shape.n_groups)
    if tuple(map(int,sa.shape))!=expected_sa or tuple(map(int,sb.shape))!=expected_sb:
        return None
    return shape,fmt,[x.removeprefix("%") for x in op.operands],op.result_names[0]

def _independent_scale_semantics(fn,op,types,block,fmt):
    # Semantic admission only. Native MLIR owns reductions, addresses and ABI.
    if any(t.rank < 2 or any(not str(v).isdigit() or int(v) <= 0 for v in t.shape)
           for t in types):
        return None
    a,b,sa,sb=types
    if (a.dtype != b.dtype or a.dtype not in {"fp8_e4m3","fp32"}
            or (a.dtype == "fp32" and fmt != "fp32")):
        return None
    if sa.dtype != ("fp32" if fmt == "fp32" else "uint8") or sb.dtype != sa.dtype:
        return None
    m,k=(tuple(map(int,a.shape[-2:][::-1])) if op.kwargs.get("transposeA",False)
         else tuple(map(int,a.shape[-2:])))
    kb,n=(tuple(map(int,b.shape[-2:][::-1])) if op.kwargs.get("transposeB",False)
          else tuple(map(int,b.shape[-2:])))
    if kb != k or tuple(map(int,sa.shape[-2:])) != (m,(k+block[1]-1)//block[1]) or (
            tuple(map(int,sb.shape[-2:])) != ((k+block[1]-1)//block[1],(n+block[0]-1)//block[0])):
        return None
    depth=max(t.rank-2 for t in types)
    prefix=[1]*depth
    for t in types:
        own=[1]*(depth-(t.rank-2))+list(map(int,t.shape[:-2]))
        for axis,extent in enumerate(own):
            if extent != 1 and prefix[axis] != 1 and extent != prefix[axis]:
                return None
            prefix[axis]=max(prefix[axis],extent)
    if (math.prod(prefix)*m > 2**63-1 or math.prod(prefix)>2**31-1 or
            any(math.prod(map(int,t.shape))>2**63-1 for t in types) or
            len(fn.result_types)!=1 or fn.result_types[0].dtype!="fp32" or
            tuple(map(str,fn.result_types[0].shape))!=tuple(map(str,(*prefix,m,n)))):
        return None
    shape=_LogicalScaleShape(m,n,k,block[1],block[0],
                             "nk" if op.kwargs.get("transposeB",False) else "kn","f32")
    return shape,fmt,[x.removeprefix("%") for x in op.operands],op.result_names[0]

def primal_call_module(module):
    """Immutable frontend projection of an ordinary call on an AD owner.

    Numerical operations/types/policies are unchanged. An ordinary call asks
    for its primal; keep the original differentiation request as retained
    metadata rather than an executable paired-export request.
    """
    import copy
    result = copy.deepcopy(module)
    keys = ("tessera.autodiff", "tessera.autodiff.wrt", "tessera.autodiff.wrt_indices")
    for fn in result.functions:
        requested = {key: fn.fn_attrs[key] for key in keys if key in fn.fn_attrs}
        if requested:
            names = {"tessera.autodiff": "mode", "tessera.autodiff.wrt": "wrt",
                     "tessera.autodiff.wrt_indices": "wrt_indices"}
            retained = "{" + ", ".join(names[key]+" = "+value
                                        for key,value in requested.items()) + "}"
            fn.fn_attrs["tessera.primal_call.requested_autodiff"] = retained
        for key in keys:
            fn.fn_attrs.pop(key, None)
    for key in keys:
        result.module_attrs.pop(key, None)
    return result


def supports_floating_scaled_primal(module):
    """Admit continuous Graph intent for the native gfx1201 product consumer."""
    try:
        if not requests_floating_scaled(module):
            return False
        fn = module.functions[0]
        if (any(arg.ir_type.rank > 8 for arg in fn.args) or
                len(fn.result_types) != 1 or fn.result_types[0].rank > 8 or
                any(not str(dim).isdigit() or int(dim) <= 0
                    for dim in fn.result_types[0].shape) or
                math.prod(map(int, fn.result_types[0].shape)) > 2**31-1):
            return False
        if len(fn.body) == 1:
            return contract(module, semantic_only=True) is not None
        return _supports_composed_scaled(module, (), primal=True, floating_reverse=True)
    except (ValueError, TypeError, KeyError, IndexError):
        return False


def supports_floating_scaled_jvp(module, wrt_indices):
    """Continuous operand roles; encoded storage retains its separate gates."""
    if not supports_floating_scaled_primal(module) or not wrt_indices:
        return False
    fn = module.functions[0]
    return (len(set(wrt_indices)) == len(wrt_indices)
            and all(type(index) is int and 0 <= index < len(fn.args)
                    and fn.args[index].ir_type.dtype == "fp32" for index in wrt_indices))


def supports_typed_scaled(module):
    try:return contract(module) is not None
    except (ValueError,TypeError):return False

def supports_scale_transpose(module):
    """Static FP32 scale-adjoint semantic admission; native AD owns reduction."""
    try:
        info = contract(module, semantic_only=True)
        return requests_typed_scaled(module) and info is not None and info[1] == "fp32"
    except (ValueError, TypeError):
        return False


def supports_scaled_reverse(module, wrt_indices=()):
    """Continuous/scale-only reverse semantics, independent of primal WMMA."""
    if len(module.functions) != 1:
        return False
    if len(module.functions[0].body) > 1:
        if not (requests_floating_scaled(module) or requests_composed_typed_scaled(module)):
            return False
        return _supports_composed_scaled(module, wrt_indices, primal=False,
                                        floating_reverse=True)
    try:
        fn = module.functions[0]
        return (contract(module, semantic_only=True) is not None
                and fn.body[0].kwargs["scale_layout"]["format"] == "fp32"
                and len(set(wrt_indices)) == len(wrt_indices)
                and all(type(i) is int and 0 <= i < len(fn.args)
                        and fn.args[i].ir_type.dtype == "fp32" for i in wrt_indices))
    except (ValueError, TypeError, KeyError, IndexError):
        return False

def lower_typed_scaled(module):
    info=contract(module)
    if info is None:raise ValueError("typed scaled primal requires its exact static matrix/scale contract")
    shape,fmt,_,_=info
    projected=replace(module,module_attrs={**module.module_attrs,
        "tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    graph=projected.to_mlir(target="rocm_gfx1201",canonical=True)
    tool=find_tessera_opt()
    if tool is None:raise RuntimeError("typed scaled primal requires matching tessera-opt")
    schedule=run_tessera_opt(tool,graph,"--tessera-graph-to-schedule")
    tile=run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
    return BlockScaleProgram(shape,module.functions[0].name,graph,schedule,tile)

def package_typed_scaled(module,program,*,pipeline_name):
    return _package_native_typed(module,program,pipeline_name=pipeline_name)


def _package_native_typed(module,program,*,pipeline_name):
    """Bind the compiler-owned member image; do not compile a second kernel."""
    import hashlib
    import json
    from .native_scaled_program import package_native_scaled_primal_artifacts
    from .native_artifact import (BufferBinding,LaunchDescriptor,LaunchGeometry,
        NativeEntryPoint,NativeImageArtifact,OrderingSemantics,ScalarArgument,ShapeGuard)
    from .rocm_fp8_blockscale import _scale_profile
    from .rocm_native import (ROCMNativePackage,_driver_selected_device_libraries,
        _version_fingerprint,_rocm_clang,_rocm_path)
    shape,fmt,names,output=contract(module)
    native,target_ir,backend_ir=package_native_scaled_primal_artifacts(program.graph_ir)
    member=json.loads(native.members_json[0])
    _,abi=_scale_profile(shape,fmt)
    tool=find_tessera_opt()
    if tool is None:raise RuntimeError("typed primal requires its selected compiler")
    compiler_fp=_version_fingerprint(tool)
    libraries=_driver_selected_device_libraries(arch="gfx1201")
    identity="|".join(f"{item.logical_name}:{item.content_digest}:{item.link_mode}" for item in libraries)
    clang=_rocm_clang(_rocm_path())
    driver_fp=_version_fingerprint(clang) if clang is not None else "missing"
    toolchain_fp=hashlib.sha256(f"{compiler_fp}|{driver_fp}|gfx1201|{identity}".encode()).hexdigest()
    image=NativeImageArtifact(target="rocm_gfx1201",architecture="gfx1201",
        pipeline_name=pipeline_name,compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(),
        binary_format="hsaco",payload=native.images[0],
        entry_points=(NativeEntryPoint(member["entry"],abi),),compile_state="cold",
        device_libraries=libraries)
    fn=module.functions[0]
    types={arg.name:arg.ir_type for arg in fn.args}
    types[output]=fn.result_types[0]
    bindings=tuple(BufferBinding(slot,name,"output" if slot==4 else "input",
        types[name].dtype,types[name].rank,"row_major",
        1 if types[name].dtype in ("fp8_e4m3","uint8") else 4)
        for slot,name in enumerate((*names,output)))
    descriptor=LaunchDescriptor(image_digest=image.image_digest,entry_symbol=member["entry"],
        abi_id=abi,buffers=bindings,
        scalars=tuple(ScalarArgument(5+i,name,"int64") for i,name in enumerate(("M","N","K"))),
        shape_guards=tuple(ShapeGuard(name,axis,"eq",int(dim))
            for name,t in types.items() for axis,dim in enumerate(t.shape)),
        geometry=LaunchGeometry(grid=tuple(member["geometry"][:3]),
                                workgroup=tuple(member["geometry"][3:])),
        ordering=OrderingSemantics(ordered_submission=True,residency="none",
                                  synchronization=("completion",)),
        provenance={"work_item":"FRONTEND-IR-MEDIUM-1",
            "sync_key":"INDEPENDENT-SCALE-BATCH-2026-10-07",
            "route":"canonical_scheduled_tile_consumer",
            "shape":[shape.m,shape.n,shape.k],"frontend_graph":"original_typed_scaled_graph",
            "native_scaled_primal_program":native.to_manifest(),
            "native_program_arg_names":[arg.name for arg in fn.args],
            "native_program_output_name":output,
            "graph_ir_sha256":hashlib.sha256(program.graph_ir.encode()).hexdigest(),
            "tile_ir_sha256":hashlib.sha256(program.tile_ir.encode()).hexdigest(),
            "target_ir_sha256":image.target_ir_digest,
            "batching":fn.body[0].kwargs.get("batching"),"image_policy":member.get("image_policy")})
    descriptor.validate_image(image)
    return ROCMNativePackage(program.tile_ir,target_ir,backend_ir,image,descriptor)


def supports_composed_scaled_primal(module):
    return (supports_floating_scaled_primal(module) or
            _supports_composed_scaled(module, (), primal=True))


def supports_composed_scale_jvp(module, wrt_indices):
    return (supports_floating_scaled_jvp(module, wrt_indices) or
            _supports_composed_scaled(module, wrt_indices, primal=False))


def _supports_composed_scaled(module, wrt_indices, *, primal, floating_reverse=False):
    """Check frontend product/sum SSA; native AD and codegen own execution."""
    import copy
    if len(module.functions) != 1 or (not primal and not wrt_indices):
        return False
    fn = module.functions[0]
    if not 2 <= len(fn.body) <= 128 or len(fn.result_types) != 1 or len(fn.return_values) != 1:
        return False
    names = {arg.name: arg for arg in fn.args}
    values = {("%" + name): arg.ir_type for name, arg in names.items()}
    scales: set[str] = set()
    used: set[str] = set()
    expected = fn.result_types[0]
    if expected.dtype != "fp32":
        return False
    for op in fn.body:
        if len(op.result_names) != 1 or any(v not in values for v in op.operands):
            return False
        result = op.inferred_type
        if result is None or result.dtype != "fp32":
            return False
        if op.op_name == "tessera.scaled_matmul":
            if len(op.operands) != 4 or any(v.removeprefix("%") not in names for v in op.operands):
                return False
            member = copy.deepcopy(module)
            member_fn = member.functions[0]
            member_fn.args = [copy.deepcopy(names[v.removeprefix("%")]) for v in op.operands]
            member_fn.body = [copy.deepcopy(op)]
            member_fn.result_types = [copy.deepcopy(result)]
            member_fn.return_values = ["%" + op.result_names[0]]
            if (contract(member, semantic_only=True) is None
                    or (requests_floating_scaled(member) and not floating_reverse)):
                return False
            eligible = op.operands if floating_reverse and requests_floating_scaled(member) else op.operands[2:]
            scales.update(v.removeprefix("%") for v in eligible)
            used.update(v.removeprefix("%") for v in op.operands)
        elif op.op_name == "tessera.add":
            if (len(op.operands) != 2 or op.kwargs or op.numeric_policy is not None
                    or any(str(values[v]) != str(result) for v in op.operands)):
                return False
        elif op.op_name == "tessera.transpose":
            if len(op.operands) != 1 or set(op.kwargs) != {"permutation"} or op.numeric_policy is not None:
                return False
            axes = op.kwargs["permutation"]
            source = values[op.operands[0]]
            if (op.operands[0].removeprefix("%") in names or
                    not isinstance(axes, (list, tuple)) or
                    any(type(axis) is not int for axis in axes) or
                    sorted(axes) != list(range(source.rank)) or
                    not 1 <= source.rank <= 8 or source.dtype != "fp32" or
                    tuple(result.shape) != tuple(source.shape[axis] for axis in axes)):
                return False
        else:
            return False
        values["%" + op.result_names[0]] = result
    return (str(values.get(fn.return_values[0])) == str(expected)
            and fn.return_values == ["%" + fn.body[-1].result_names[0]]
            and used == set(names)
            and all(type(i) is int and 0 <= i < len(fn.args)
                    and fn.args[i].name in scales and fn.args[i].ir_type.dtype == "fp32"
                    for i in wrt_indices)
            and len(set(wrt_indices)) == len(wrt_indices))
