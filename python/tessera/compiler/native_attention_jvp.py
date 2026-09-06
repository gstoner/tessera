"""Bounded-storage cooperative saved-LSE JVP, lowered from native GPU MLIR.

The physical product consumes a resident forward generation. The automatic
adapter lowers an isolated TangentInterface export; neither path promotes a
performance candidate.
"""
from pathlib import Path
from .native_gpu_tensor import TensorSpec, IndexSpec
from .native_storage_contract import attach_tensor_contract
from .native_gpu_storage import build_native_gpu_storage


def source(dims, scale, causal):
    from .nvidia_native import _checkpoint_identity
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
    specs=tuple(TensorSpec(name,'fp32',shape,name=='tangent') for name,shape in
        [('q',qshape),('k',kshape),('v',vshape),('primal',oshape),('lse',(b,hq,sq)),
         ('dq',qshape),('dk',kshape),('dv',vshape),('tangent',oshape)])+(IndexSpec('scratch',128,128),)
    args=', '.join('%'+s.name+': !llvm.ptr<1>' for s in specs[:-1])+', %scratch: index'
    lines=[f'''module {{
  gpu.module @attention_jvp {{
    gpu.func @saved_lse_jvp({args}) kernel {{
      %tid = gpu.thread_id x
      %row = gpu.block_id x
      %t = arith.index_cast %tid : index to i64
      %r = arith.index_cast %row : index to i64
      %zero = arith.constant 0 : i64
      %one = arith.constant 1 : i64
      %width = arith.constant 128 : i64
      %sk = arith.constant {sk} : i64
      %sk_index = arith.constant {sk} : index
      %width_index = arith.constant 128 : index
      %D_index = arith.constant {d} : index
      %one_index = arith.constant 1 : index
      %sq = arith.constant {sq} : i64
      %hq = arith.constant {hq} : i64
      %hkv = arith.constant {hkv} : i64
      %ratio = arith.constant {hq//hkv} : i64
      %D = arith.constant {d} : i64
      %DV = arith.constant {dv} : i64
      %scale = arith.constant {scale:.17e} : f32
      %log2e = arith.constant 1.4426950408889634 : f32
      %z = arith.constant 0.0 : f32
      %izero = arith.constant 0 : index
      %leader = arith.cmpi eq, %t, %zero : i64
      %qi = arith.remui %r, %sq : i64
      %bh = arith.divui %r, %sq : i64
      %head = arith.remui %bh, %hq : i64
      %batch = arith.divui %bh, %hq : i64
      %kvhead = arith.divui %head, %ratio : i64
      %bhkv0 = arith.muli %batch, %hkv : i64
      %bhkv = arith.addi %bhkv0, %kvhead : i64
      %kvbase = arith.muli %bhkv, %sk : i64
      %qbase = arith.muli %r, %D : i64
      %obase = arith.muli %r, %DV : i64
      %align = arith.constant {max(sk-sq,0)} : i64
      %limit = arith.addi %qi, %align : i64
      %lp = llvm.getelementptr %lse[%r] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %L = llvm.load %lp : !llvm.ptr<1> -> f32
      %moments = memref.alloca(%scratch) : memref<?xf32>
      "tile.alloc_shared"(%moments) : (memref<?xf32>) -> ()
      %products = memref.alloca(%scratch) : memref<?xf32>
      "tile.alloc_shared"(%products) : (memref<?xf32>) -> ()
      scf.for %col = %zero to %DV step %one : i64 {{
        %acc:2 = scf.for %key_index = %tid to %sk_index step %width_index iter_args(%moment = %z, %product = %z) -> (f32, f32) {{
''']
    lines.append('          %key = arith.index_cast %key_index : index to i64\n')
    lines.append('          %legal = arith.cmpi ule, %key, %limit : i64\n' if causal else '          %legal = arith.constant true\n')
    lines.append('''          %next:2 = scf.if %legal -> (f32, f32) {
            %kr = arith.addi %kvbase, %key : i64
            %kbase = arith.muli %kr, %D : i64
            %dot:2 = scf.for %axis_index = %izero to %D_index step %one_index iter_args(%s = %z, %ds = %z) -> (f32, f32) {
              %axis = arith.index_cast %axis_index : index to i64
              %qindex = arith.addi %qbase, %axis : i64
              %kindex = arith.addi %kbase, %axis : i64
''')
    for name,index in [('q','qindex'),('dq','qindex'),('k','kindex'),('dk','kindex')]:
        lines.append(f'              %{name}p = llvm.getelementptr %{name}[%{index}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32\n'
                     f'              %{name}v = llvm.load %{name}p : !llvm.ptr<1> -> f32\n')
    lines.append('''              %qk = arith.mulf %qv, %kv : f32
              %dqk = arith.mulf %dqv, %kv : f32
              %qdk = arith.mulf %qv, %dkv : f32
              %dscore = arith.addf %dqk, %qdk : f32
              %sn = arith.addf %s, %qk : f32
              %dsn = arith.addf %ds, %dscore : f32
              scf.yield %sn, %dsn : f32, f32
            }
            %score = arith.mulf %dot#0, %scale : f32
            %direction = arith.mulf %dot#1, %scale : f32
            %shifted = arith.subf %score, %L : f32
            %expinput = arith.mulf %shifted, %log2e : f32
            %probability = math.exp2 %expinput : f32
            %vbase = arith.muli %kr, %DV : i64
            %vi = arith.addi %vbase, %col : i64
            %vp = llvm.getelementptr %v[%vi] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
            %dvp = llvm.getelementptr %dv[%vi] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
            %vv = llvm.load %vp : !llvm.ptr<1> -> f32
            %dvv = llvm.load %dvp : !llvm.ptr<1> -> f32
            %pd = arith.mulf %probability, %direction : f32
            %pdv = arith.mulf %pd, %vv : f32
            %pddv = arith.mulf %probability, %dvv : f32
            %both = arith.addf %pdv, %pddv : f32
            %mn = arith.addf %moment, %pd : f32
            %pn = arith.addf %product, %both : f32
            scf.yield %mn, %pn : f32, f32
          } else {
            scf.yield %moment, %product : f32, f32
          }
          scf.yield %next#0, %next#1 : f32, f32
        }
        memref.store %acc#0, %moments[%tid] : memref<?xf32>
        memref.store %acc#1, %products[%tid] : memref<?xf32>
        gpu.barrier
''')
    for stride in (64,32,16,8,4,2,1):
        lines.append(f'''        %s{stride} = arith.constant {stride} : index
        %active{stride} = arith.cmpi ult, %tid, %s{stride} : index
        scf.if %active{stride} {{
          %other{stride} = arith.addi %tid, %s{stride} : index
''')
        for name in ('moments','products'):
            prefix=f'%{name}{stride}'
            lines.append(f'''          {prefix}a = memref.load %{name}[%tid] : memref<?xf32>
          {prefix}b = memref.load %{name}[%other{stride}] : memref<?xf32>
          {prefix}sum = arith.addf {prefix}a, {prefix}b : f32
          memref.store {prefix}sum, %{name}[%tid] : memref<?xf32>
''')
        lines.append('        }\n        gpu.barrier\n')
    lines.append('''        scf.if %leader {
          %m = memref.load %moments[%izero] : memref<?xf32>
          %p = memref.load %products[%izero] : memref<?xf32>
          %oi = arith.addi %obase, %col : i64
          %op = llvm.getelementptr %primal[%oi] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
          %outp = llvm.getelementptr %tangent[%oi] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
          %o = llvm.load %op : !llvm.ptr<1> -> f32
          %correction = arith.mulf %o, %m : f32
          %result = arith.subf %p, %correction : f32
          llvm.store %result, %outp : f32, !llvm.ptr<1>
        }
        gpu.barrier
      }
      gpu.return
    }
  }
}
''')
    text=attach_tensor_contract(''.join(lines),specs,grid=(b*hq*sq,1,1),block=(128,1,1))
    identity=_checkpoint_identity(tuple(dims),scale,causal)
    return text.replace('module attributes {',f'module attributes {{tessera.attention_checkpoint_identity = "{identity}", ',1)


def materialize(dims, scale, causal, *, compiler, llvm_bin):
    return build_native_gpu_storage(source(dims,scale,causal),compiler=Path(compiler),
                                    llvm_bin=Path(llvm_bin),backend='nvidia',chip='sm_120')


def materialize_generated(graph_source, dims, scale, causal, *, compiler, llvm_bin):
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
    if (contract.get('schema')!=1 or contract.get('dims')!=list(dims) or
            contract.get('causal') is not causal or
            struct.pack('f',contract['scale'])!=struct.pack('f',scale)):
        raise ValueError('automatic JVP policy differs from the resident forward generation')
    active=contract.get('active')
    if not isinstance(active,list) or len(active)!=3 or any(type(v) is not bool for v in active):
        raise ValueError('automatic JVP tangent mapping is invalid')
    ir=source(dims,scale,causal)
    for name,enabled in zip(('dq','dk','dv'),active,strict=True):
        if not enabled:
            ir=ir.replace(f'%{name}v = llvm.load %{name}p : !llvm.ptr<1> -> f32',
                          f'%{name}v = arith.constant 0.0 : f32')
    digest=hashlib.sha256(product.encode()).hexdigest()
    ir=ir.replace('module attributes {',f'module attributes {{tessera.autodiff.generated_jvp = "{digest}", ',1)
    return build_native_gpu_storage(ir,compiler=Path(compiler),llvm_bin=Path(llvm_bin),
                                    backend='nvidia',chip='sm_120')
