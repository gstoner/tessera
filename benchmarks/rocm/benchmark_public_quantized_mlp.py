"""Authored two-quantizer NVFP4 gate/up and down-projection Graph."""
from __future__ import annotations
import ml_dtypes
import numpy as np
import tessera as ts
from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy


def make_mlp(m=3,width=32,k=48,n=19):
    groups=m*k//16;down_groups=m*width//16
    divisor=np.full((groups,1),448,np.float32);divisor.flags.writeable=False
    floor=np.full((groups,1),np.finfo(np.float32).tiny,np.float32);floor.flags.writeable=False
    down_divisor=np.full((down_groups,1),448,np.float32);down_divisor.flags.writeable=False
    down_floor=np.full((down_groups,1),np.finfo(np.float32).tiny,np.float32);down_floor.flags.writeable=False
    def mlp(x,packed,weight_scales,globals_,down_packed,down_weight_scales,down_globals):
        blocks=ts.ops.reshape(x,shape=(groups,16))
        wide=ts.ops.cast(blocks,'fp32')
        absolute=ts.ops.absolute(wide)
        peak=ts.ops.reduce(absolute,op='max',axis=1,keepdims=True)
        raw=ts.ops.div(peak,divisor)
        scale=ts.ops.maximum(raw,floor)
        expanded=ts.ops.expand(scale,shape=(groups,16))
        normalized=ts.ops.div(wide,expanded)
        encoded=ts.ops.cast(normalized,'fp8_e4m3')
        codes=ts.ops.reshape(encoded,shape=(m,k))
        scales=ts.ops.reshape(scale,shape=(m,k//16))
        b,sb=ts.ops.nvfp4_expand(packed,weight_scales,globals_,row_offsets=[0,width,2*width],numeric_policy=nvfp4_expansion_policy())
        accumulator=ts.ops.scaled_matmul(codes,b,scales,sb,transposeB=True,
            scale_layout={'granularity':'block','block':[1,16],'format':'fp32'},
            numeric_policy={'accum':'fp32','execution_mode':'exact_per_block'})
        parent=ts.ops.cast(accumulator,'bf16')
        left=ts.ops.slice(parent,start_indices=[0,0],slice_sizes=[m,width])
        right=ts.ops.slice(parent,start_indices=[0,width],slice_sizes=[m,width])
        left_f32=ts.ops.cast(left,'fp32')
        right_f32=ts.ops.cast(right,'fp32')
        gated=ts.ops.silu_mul(left_f32,right_f32)
        activation=ts.ops.cast(gated,'bf16')
        down_blocks=ts.ops.reshape(activation,shape=(down_groups,16))
        down_wide=ts.ops.cast(down_blocks,'fp32')
        down_absolute=ts.ops.absolute(down_wide)
        down_peak=ts.ops.reduce(down_absolute,op='max',axis=1,keepdims=True)
        down_raw=ts.ops.div(down_peak,down_divisor)
        down_scale=ts.ops.maximum(down_raw,down_floor)
        down_expanded=ts.ops.expand(down_scale,shape=(down_groups,16))
        down_normalized=ts.ops.div(down_wide,down_expanded)
        down_encoded=ts.ops.cast(down_normalized,'fp8_e4m3')
        down_codes=ts.ops.reshape(down_encoded,shape=(m,width))
        down_scales=ts.ops.reshape(down_scale,shape=(m,width//16))
        down_b,down_sb=ts.ops.nvfp4_expand(down_packed,down_weight_scales,down_globals,row_offsets=[0,n],numeric_policy=nvfp4_expansion_policy())
        down_accumulator=ts.ops.scaled_matmul(down_codes,down_b,down_scales,down_sb,transposeB=True,
            scale_layout={'granularity':'block','block':[1,16],'format':'fp32'},
            numeric_policy={'accum':'fp32','execution_mode':'exact_per_block'})
        return ts.ops.cast(down_accumulator,'bf16')
    return mlp


def inputs(m,width,k,n,seed=731):
    from benchmarks.rocm.benchmark_public_quantized_gate_up import inputs as gate_inputs
    roots=gate_inputs(m,width,k,seed)
    rng=np.random.default_rng(seed+1)
    return roots+[rng.integers(0,256,size=(n,width//2),dtype=np.uint8),
        rng.uniform(.125,2,size=(n,width//16)).astype(ml_dtypes.float8_e4m3fn),
        np.array([.0123456789],np.float64)]


def run(m,h,k,n,output_dir,repeats):
    import hashlib
    import json
    import os
    from pathlib import Path
    import re
    from statistics import median
    import subprocess
    import time
    from tessera import runtime as rt
    from benchmarks.rocm.quantized_mlp_reference import assert_compiled_chain
    if rt._rocm_live_arch()!='gfx1201':
        raise RuntimeError('requires owning gfx1201')
    info=subprocess.check_output(['/opt/rocm/bin/rocminfo'],text=True)
    identity=re.search(r'Name:\s+gfx1201\s+Uuid:\s+(GPU-\S+)\s+Marketing Name:\s+([^\n]+)',info)
    if identity is None:
        raise RuntimeError('owning device identity unavailable')
    roots=inputs(m,h,k,n)
    fn=ts.jit(target='rocm_gfx1201',native_required=True)(make_mlp(m,h,k,n))
    start=time.perf_counter();first=fn(*roots);cold=(time.perf_counter()-start)*1000
    artifact=fn.runtime_artifact()
    if fn.execution_kind!='native_gpu' or artifact.metadata['compiler_path']!='canonical_rocm_quantized_mlp_program':
        raise RuntimeError('public call did not execute the seven-member program')
    program=fn._rocm_quantized_product_last_program.native
    lib=rt._load_rocm_native_movement_runtime()
    certificate=assert_compiled_chain(program,roots,first,lib._name)
    receipts=list(fn._rocm_quantized_product_last_receipts)
    if len(receipts)!=7:
        raise RuntimeError('missing native member receipts')
    record={'shape_mhkn':[m,h,k,n],'architecture':rt._rocm_live_arch(),
        'device_uuid':identity[1],'device_name':identity[2].strip(),
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'recorder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256(Path(__file__).with_name('quantized_mlp_reference.py').read_bytes()).hexdigest(),
        'compiler_sha256':hashlib.sha256(Path(os.environ['TESSERA_OPT']).read_bytes()).hexdigest(),
        'provider_sha256':hashlib.sha256(Path(lib._name).read_bytes()).hexdigest(),
        'correctness':'per_stage_and_propagation_passed_before_timing','numerical_certificate':certificate,
        'route':'authored Graph SSA -> seven native partitions -> Schedule -> Tile -> ROCm Target -> LLVM/HSACO -> one checked multi-output HIP owner',
        'native_schema':json.loads(program.program_json)['schema'],'binding_digest':program.binding_digest,
        'component_receipts':receipts,'public_cold_ms':cold,'public_warm_samples_ms':[],
        'prepared_invoke_read_samples_ms':[],'ordinary_program_event_samples_ms':[],
        'captured_member_samples_ms':[],
        'member_order':['input_quantizer','gate_up_expansion','gate_up_product_bf16_store','silu_bf16_store',
                        'activation_quantizer','down_expansion','down_product_bf16_store']}
    for _ in range(7):
        start=time.perf_counter()
        for _ in range(repeats):result=fn(*roots)
        record['public_warm_samples_ms'].append((time.perf_counter()-start)*1000/repeats)
    np.testing.assert_array_equal(result,first)
    with program.prepare(roots,runtime_library=lib._name) as owner:
        generation,_=owner.invoke();np.testing.assert_array_equal(owner.read(generation)[0],first)
        for _ in range(7):
            _,members=owner.profile_members(repeats=128)
            record['captured_member_samples_ms'].append(list(members))
            generation,event=owner.invoke(repeats=50,timed=True)
            record['ordinary_program_event_samples_ms'].append(event)
            start=time.perf_counter();generation,_=owner.invoke();held=owner.read(generation)[0]
            record['prepared_invoke_read_samples_ms'].append((time.perf_counter()-start)*1000)
        np.testing.assert_array_equal(held,first)
    for label in ('public_warm','prepared_invoke_read','ordinary_program_event'):
        record[label+'_median_ms']=median(record[label+'_samples_ms'])
    record['captured_member_median_ms']=[median(row[i] for row in record['captured_member_samples_ms']) for i in range(7)]
    record['timing_scope']={'public':'warm frontend, input validation, root copies, owner prepare/invoke/read',
        'prepared':'invoke and returned readback; excludes prepare, compilation and root upload',
        'ordinary_event':'per-program HIP event window; native API already averages repeats; includes enqueue gaps',
        'captured_members':'seven individual graph dispatch windows; excludes capture/instantiation/copies',
        'numerical_diagnostics':'independent math and staged native controls precede timing and are excluded from all timing windows'}
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    stem=f'public-{m}-{h}-{k}-{n}';payload=artifact.to_json()
    (output_dir/(stem+'-artifact.json')).write_text(payload)
    record['artifact_sha256']=hashlib.sha256(payload.encode()).hexdigest()
    (output_dir/(stem+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({key:record[key] for key in ('shape_mhkn','public_warm_median_ms','captured_member_median_ms','ordinary_program_event_median_ms','numerical_certificate')}),flush=True)


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if not 1<=args.repeats<=1024:parser.error('repeats must be between 1 and 1024')
    for shape in ((1,16,16,1),(3,32,48,19),(256,128,12288,256)):
        run(*shape,args.output_dir,args.repeats)
