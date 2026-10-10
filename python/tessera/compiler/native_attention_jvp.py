"""Bounded-storage cooperative saved-LSE JVP, lowered from native GPU MLIR.

The physical product consumes a resident forward generation. The automatic
adapter lowers an isolated TangentInterface export; neither path promotes a
performance candidate.
"""
from pathlib import Path
from .native_gpu_storage import build_native_gpu_storage


def source(dims, scale, causal, *, compiler=None):
    import math
    import struct
    if (len(dims)!=7 or any(type(d) is not int or not 0<d<=65536 for d in dims) or
            type(causal) is not bool or type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0):
        raise ValueError('invalid native attention JVP policy')
    try:
        rounded_scale=struct.unpack('f',struct.pack('f',scale))[0]
    except (OverflowError,struct.error):
        raise ValueError('native attention JVP scale must be representable in fp32') from None
    if not math.isfinite(rounded_scale) or rounded_scale==0:
        raise ValueError('native attention JVP scale must be representable in fp32')
    b,hq,hkv,sq,sk,d,dv=dims
    if hq%hkv or b*hq*sq>2147483647:
        raise ValueError('invalid native attention JVP head or launch geometry')
    qshape=(b,hq,sq,d)
    kshape=(b,hkv,sk,d)
    vshape=(b,hkv,sk,dv)
    oshape=(b,hq,sq,dv)
    if any(math.prod(shape)>((1<<63)-1)//4 for shape in (qshape,kshape,vshape,oshape)):
        raise ValueError('native attention JVP tensor byte extent overflows')
    def tensor(shape):
        return "tensor<" + "x".join(map(str,shape)) + "xf32>"
    qt,kt,vt,ot,lt=map(tensor,(qshape,kshape,vshape,oshape,(b,hq,sq)))
    graph=f"""module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
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


def materialize(dims, scale, causal, *, compiler, llvm_bin):
    return build_native_gpu_storage(source(dims,scale,causal,compiler=compiler),compiler=Path(compiler),
                                    llvm_bin=Path(llvm_bin),backend='nvidia',chip='sm_120')


def materialize_generated(graph_source, dims, scale, causal, *, compiler, llvm_bin, bias_shape=()):
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
    logical=(dims[0],dims[1],dims[3],dims[4])
    if biased and (len(bias_shape)!=4 or any(type(x) is not int or x not in (1,d)
            for x,d in zip(bias_shape,logical,strict=True))):
        raise ValueError('automatic JVP physical bias shape is invalid')
    saved_lse = contract.get('saved_lse', False)
    if type(saved_lse) is not bool:
        raise ValueError('automatic JVP saved-LSE output selection is invalid')
    if (contract.get('schema')!=(3 if saved_lse else 2 if biased else 1) or
            contract.get('bias_shape',[])!=list(bias_shape) or contract.get('dims')!=list(dims) or
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
