"""Bounded-storage cooperative saved-LSE JVP, lowered from native GPU MLIR.

The physical product consumes a resident forward generation. The automatic
adapter lowers an isolated TangentInterface export; neither path promotes a
performance candidate.
"""
from pathlib import Path
from .native_gpu_storage import build_native_gpu_storage


def source(dims, scale, causal, *, compiler=None, shape_bounds=()):
    import math
    import struct
    from .attention_shape_contract import attention_dimensions, DYNAMIC_DIM
    capacities=attention_dimensions(dims,shape_bounds)
    if (any(d>65536 for d in capacities) or type(causal) is not bool or
            type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0):
        raise ValueError('invalid native attention JVP policy')
    try:
        rounded_scale=struct.unpack('f',struct.pack('f',scale))[0]
    except (OverflowError,struct.error):
        raise ValueError('native attention JVP scale must be representable in fp32') from None
    if not math.isfinite(rounded_scale) or rounded_scale==0:
        raise ValueError('native attention JVP scale must be representable in fp32')
    b,hq,hkv,sq,sk,d,dv=dims
    if hq%hkv or capacities[0]*capacities[1]*capacities[3]>2147483647:
        raise ValueError('invalid native attention JVP head or launch geometry')
    qshape=(b,hq,sq,d)
    kshape=(b,hkv,sk,d)
    vshape=(b,hkv,sk,dv)
    oshape=(b,hq,sq,dv)
    def tensor(shape):
        return "tensor<" + "x".join("?" if d==DYNAMIC_DIM else str(d) for d in shape) + "xf32>"
    bounds_attr=(", tessera.attention_shape_bounds = array<i64: " +
                 ", ".join(map(str,shape_bounds)) + ">") if shape_bounds else ""
    qt,kt,vt,ot,lt=map(tensor,(qshape,kshape,vshape,oshape,(b,hq,sq)))
    graph=f"""module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"{bounds_attr}}} {{
      func.func @attention_jvp(%q: {qt}, %k: {kt}, %v: {vt},
          %dq: {qt}, %dk: {kt}, %dv: {vt}) -> {ot} {{
        %o, %lse = "tessera_attn.checkpoint_forward"(%q, %k, %v)
          {{scale = {rounded_scale:.17e} : f32, causal = {str(causal).lower()}}}
          : ({qt}, {kt}, {vt}) -> ({ot}, {lt})
        %tangent = "tessera_attn.checkpoint_jvp"(%q, %k, %v, %o, %lse, %dq, %dk, %dv)
          {{scale = {rounded_scale:.17e} : f32, causal = {str(causal).lower()}}}
          : ({qt}, {kt}, {vt}, {ot}, {lt}, {qt}, {kt}, {vt}) -> {ot}
        return %tangent : {ot}
      }}
    }}"""
    return _lower_graph(graph,compiler=compiler)


def _lower_graph(graph, *, compiler=None):
    from .scheduled_matmul import find_tessera_opt, run_tessera_opt
    tool=Path(compiler) if compiler is not None else find_tessera_opt()
    if tool is None:
        raise RuntimeError("native attention JVP requires tessera-opt")
    # One native pass manager retains typed SSA between verified stages.
    # Avoid a subprocess and a printed/reparsed Schedule boundary.
    return run_tessera_opt(tool,graph,
        "--pass-pipeline=builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile)")


def materialize(dims, scale, causal, *, compiler, llvm_bin, shape_bounds=()):
    return build_native_gpu_storage(source(dims,scale,causal,compiler=compiler,shape_bounds=shape_bounds),compiler=Path(compiler),
                                    llvm_bin=Path(llvm_bin),backend='nvidia',chip='sm_120')


def materialize_generated(graph_source, dims, scale, causal, *, compiler, llvm_bin, bias_shape=(), shape_bounds=()):
    """Lower the native AD contract of one isolated attention function.

    The compiler verifies the source's complete argument/return mapping and the
    O/LSE producer relation. Physical lowering uses that verified product, with
    inactive tangent slots zeroed regardless of caller data.
    """
    import hashlib
    import json
    import re
    import struct
    from .native_gpu_storage import _decode_image
    from .scheduled_matmul import run_tessera_opt
    product=run_tessera_opt(Path(compiler),graph_source,
                           '--tessera-autodiff-forward=export-attention-jvp')
    fields=re.findall(r'tessera\.autodiff\.attention_jvp_contract = "((?:\\.|[^"\\])*)"',product)
    if len(fields)!=1:
        raise ValueError('native automatic JVP contract is missing or ambiguous')
    contract=json.loads(_decode_image(fields[0]).decode())
    biased=bool(bias_shape)
    from .attention_shape_contract import attention_dimensions, physical_attention_bias_shape, DYNAMIC_DIM
    attention_dimensions(dims,shape_bounds)
    if biased:
        physical_attention_bias_shape(dims,bias_shape)
    # AD's portable JSON sentinel is -1; native checkpoint policy retains
    # MLIR's kDynamic. Convert only at this adapter boundary.
    portable=lambda shape:[-1 if d==DYNAMIC_DIM else d for d in shape]
    expected_schema=(3 if shape_bounds else 1)+int(biased)
    if (contract.get('schema')!=expected_schema or
            contract.get('bias_shape',[])!=portable(bias_shape) or contract.get('dims')!=portable(dims) or
            contract.get('shape_bounds',[])!=list(shape_bounds) or
            (bool(shape_bounds) and contract.get('shape_policy')!='bounded_sequences_v1') or
            contract.get('causal') is not causal or
            struct.pack('f',contract['scale'])!=struct.pack('f',scale)):
        raise ValueError('automatic JVP policy differs from the resident forward generation')
    active=contract.get('active')
    if not isinstance(active,list) or len(active)!=3+int(biased) or any(type(v) is not bool for v in active):
        raise ValueError('automatic JVP tangent mapping is invalid')
    # The native AD product retains the paired output/LSE SSA generation.
    # Only target selection and provenance are supplied by this adapter.
    for key,value in (("tessera.target","nvidia_sm120"),("tessera.arch","sm_120")):
        existing=re.findall(re.escape(key)+r' = "([^"]*)"',product)
        if existing and existing!=[value]:
            raise ValueError("automatic JVP target differs from its package target")
        if not existing:
            if "module attributes {" in product:
                product=product.replace("module attributes {",f'module attributes {{{key} = "{value}", ',1)
            else:
                product=product.replace("module {",f'module attributes {{{key} = "{value}"}} {{',1)
    digest=hashlib.sha256(product.encode()).hexdigest()
    product=product.replace('module attributes {',f'module attributes {{tessera.autodiff.generated_jvp = "{digest}", ',1)
    ir=_lower_graph(product,compiler=compiler)
    return build_native_gpu_storage(ir,compiler=Path(compiler),llvm_bin=Path(llvm_bin),
                                    backend='nvidia',chip='sm_120')
