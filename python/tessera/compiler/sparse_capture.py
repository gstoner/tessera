"""Explicit 2:4 specialization of a single tracer-owned matrix product.

Selecting this API declares a checked sparse precondition. Ordinary JIT dispatch
is unchanged, and input data is never pruned or used to choose a sparsity pattern.
"""
from dataclasses import dataclass
import hashlib
import re

from .rocm_sparse_runtime import SparseMatmulPackage, _compile_sparse_source


@dataclass(frozen=True)
class CapturedSparseMatmul:
    graph_ir: str
    graph_digest: str
    package: SparseMatmulPackage
    package_digest: str

    def run(self,a,b,**options):
        if hashlib.sha256(self.graph_ir.encode()).hexdigest() != self.graph_digest or self.package.digest != self.package_digest:
            raise ValueError('sparse capture identity changed')
        return self.package.run(a,b,**options)


def native_sparse_source(module, selection="checked_2to4"):
    """Frontend admission of one logical half matmul for the checked 2:4 route;
    returns the Graph IR text carrying the target, arch and sparse policy the
    native lowering consumes. Shared by the isolated-worker capture and the
    scheduled package (public admission, 2026-09-18)."""
    if selection not in {"checked_2to4", "auto_2to4"}:
        raise ValueError("unknown sparse selection policy")
    if module.module_attrs.get("tessera.target") not in (None, '"rocm"') or module.module_attrs.get("tessera.arch") not in (None, '"gfx1201"'):
        raise ValueError("sparse capture requires its owning gfx1201 target")
    if len(module.functions) != 1:
        raise ValueError('sparse capture requires one matrix entry')
    if set(module.module_attrs) - {"tessera.target", "tessera.arch", "tessera.ir.version", "tessera.frontend.authority", "tessera.autodiff", "tessera.autodiff.wrt", "tessera.autodiff.wrt_indices"}:
        raise ValueError(f"sparse capture cannot discard module policies: {module.module_attrs}")
    fn = module.functions[0]
    if len(fn.args) != 2 or len(fn.body) != 1 or len(fn.result_types) != 1:
        raise ValueError('sparse capture requires one isolated matrix product')
    provenance_attrs = {"tessera.frontend.authority", "tessera.structured_cfg.schema", "tessera.structured_cfg.digest", "tessera.structured_cfg.blocks", "tessera.autodiff", "tessera.autodiff.wrt", "tessera.autodiff.wrt_indices"}
    if set(fn.fn_attrs) - provenance_attrs:
        raise ValueError(f'sparse capture does not consume function policies: {fn.fn_attrs}')
    op = fn.body[0]
    if (op.op_name not in {'tessera.matmul','tessera.gemm'} or
        op.operands != ['%'+arg.name for arg in fn.args] or
        fn.return_values != ['%'+op.result]):
        raise ValueError('sparse capture requires ordered A/B and the direct result')
    if any(key not in {'activation','epilogue'} or value not in (None,'none') for key,value in op.kwargs.items()):
        raise ValueError('sparse capture cannot discard a matrix policy or epilogue')
    a,b = [arg.ir_type for arg in fn.args]
    out = fn.result_types[0]
    if out.layout not in (None,"row_major"):
        raise ValueError("sparse capture requires row-major output layout")
    from tessera.dtype import canonicalize_dtype
    parsed = []
    for ty in (a,b,out):
        match = re.fullmatch(r'tensor<([1-9][0-9]*)x([1-9][0-9]*)x(f16|bf16|f32)>',str(ty))
        if match is None:
            raise ValueError('sparse capture requires plain concrete rank-two tensor types')
        parsed.append(match)
    ashape,bshape,oshape = [tuple(map(int,(t[1],t[2]))) for t in parsed]
    ad,bd,od = [{'f16':'fp16','bf16':'bf16','f32':'fp32'}[t[3]] for t in parsed]
    for ty,shape,dtype in zip((a,b,out),(ashape,bshape,oshape),(ad,bd,od),strict=True):
        if ((ty.dtype is not None and canonicalize_dtype(ty.dtype) != dtype) or
            (ty.shape and tuple(map(int,ty.shape)) != shape)):
            raise ValueError('sparse capture tensor metadata disagrees with serialized type')
    if ad not in {'fp16','bf16'} or bd != ad or od not in {ad,'fp32'}:
        raise ValueError('sparse capture supports matching f16/bf16 inputs')
    if ashape[1] != bshape[0] or oshape != (ashape[0],bshape[1]):
        raise ValueError('sparse capture matrix/result shapes disagree')
    for arg in fn.args:
        if arg.effect not in (None,'read') or arg.shard_spec or arg.model_parameter or arg.model_parameter_bytes_bound is not None:
            raise ValueError('sparse capture cannot discard effect/shard/model contracts')
        if arg.layout not in (None,'row_major') or arg.ir_type.layout not in (None,'row_major'):
            raise ValueError('sparse capture requires row-major input layout')
    # Native Graph lowering owns all physical packing and projects its ABI.
    import copy
    native_module = copy.deepcopy(module)
    native_module.module_attrs.update({"tessera.target": '"rocm"',
        "tessera.arch": '"gfx1201"', "tessera.sparse_policy": f'"{selection}"'})
    source = native_module.to_mlir(target='rocm',canonical=True)
    return source


def compile_sparse_graph(module, *, selection="checked_2to4", **compiler_options):
    """Frontend-only admission; the runtime receives the serialized package."""
    if selection not in {"checked_2to4", "auto_2to4"}:
        raise ValueError("unknown sparse selection policy")
    if set(compiler_options) - {"compiler", "llvm_bin", "toolkit"}:
        raise ValueError("sparse capture cannot override the captured arithmetic policy")
    source = native_sparse_source(module, selection)
    from .scheduled_matmul import find_tessera_opt, run_tessera_opt
    compiler = compiler_options.get('compiler') or find_tessera_opt()
    if compiler is None:
        raise RuntimeError('sparse capture requires native Graph verification')
    schedule = run_tessera_opt(compiler,source,'--tessera-graph-to-schedule')
    shape_match = re.search(r'tessera.sparse_shape = array<i64: (\d+), (\d+), (\d+)>',schedule)
    storage = re.search(r'tessera.sparse_storage = "(f16|bf16)"',schedule)
    output = re.search(r'tessera.sparse_output = "(f16|bf16|f32)"',schedule)
    if shape_match is None or storage is None or output is None:
        raise ValueError('native sparse descriptor is missing')
    m,n,k = map(int,shape_match.groups())
    dtype = 'float16' if storage[1] == 'f16' else 'bfloat16'
    package = _compile_sparse_source(schedule,m,n,k,dtype=dtype,
        output_storage=output[1],native_graph_ir=source,**compiler_options)
    return CapturedSparseMatmul(source,hashlib.sha256(source.encode()).hexdigest(),package,package.digest)
