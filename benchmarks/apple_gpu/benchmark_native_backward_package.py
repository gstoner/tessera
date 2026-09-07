"""Paired native-package versus identical direct ABI; not .mtlpackage evidence."""
from __future__ import annotations
import argparse
import ctypes as ct
import json
import statistics
import time
from pathlib import Path
import numpy as np
from tessera.compiler import apple_native, scheduled_attention_backward
from tessera.compiler.graph_ir import GraphIRModule, GraphIRFunction, IRArg, IRType, IROp
from tessera.compiler.apple_route_selector import live_apple_route_context
from tessera.runtime import RuntimeArtifact, launch
from tessera._apple_gpu_dispatch import bind_registered
from benchmark_attention_backward import _reference


def measure(dtype: str, trials: int, reps: int):
    from ml_dtypes import bfloat16
    spelling, storage = ('f16', np.float16) if dtype == 'fp16' else ('bf16', bfloat16)
    shape = (1, 4, 9, 32)
    kvshape = (1, 2, 19, 32)
    def ty(dims, element, logical):
        return IRType('tensor<'+'x'.join(map(str, dims))+'x'+element+'>', tuple(map(str, dims)), logical)
    q, k = ty(shape, spelling, dtype), ty(kvshape, spelling, dtype)
    bias = ty((1, 4, 9, 19), 'f32', 'fp32')
    results = [ty(d, 'f32', 'fp32') for d in (shape, kvshape, kvshape)]
    feeds = [q, q, k, k, bias]
    module = GraphIRModule(functions=[GraphIRFunction(name='mixed_bias_vjp',
        args=[IRArg(n, t) for n, t in zip(('do','q','k','v','bias'), feeds)],
        result_types=results, body=[IROp(result='dq,dk,dv', op_name='tessera.flash_attn_bwd',
        operands=['%do','%q','%k','%v','%bias'], operand_types=list(map(str, feeds)),
        result_type='('+', '.join(map(str, results))+')', kwargs={'scale':0.25,'causal':True})],
        return_values=['%dq','%dk','%dv'])])
    start=time.perf_counter_ns()
    scheduled=scheduled_attention_backward.lower_scheduled_attention_backward(module,target='apple_gpu')
    package=apple_native.package_scheduled_attention_backward(scheduled,pipeline_name='tessera-lower-to-apple_gpu')
    cold=time.perf_counter_ns()-start
    artifact=RuntimeArtifact(metadata={'target':'apple_gpu'},native_image=package.image,
        launch_descriptor=package.descriptor,tile_ir=package.tile_ir,target_ir=package.target_ir)
    rng=np.random.default_rng(797)
    arrays={n:rng.normal(scale=.2,size=s).astype(storage) for n,s in
            [('q',shape),('k',kvshape),('v',kvshape),('do',shape)]}
    arrays['bias']=rng.normal(scale=.05,size=(1,4,9,19)).astype(np.float32)
    arrays.update({n:np.zeros(s,np.float32) for n,s in [('dq',shape),('dk',kvshape),('dv',kvshape)]})
    expected=_reference(*[arrays[n].astype(np.float64).reshape((-1,9 if n in ('q','do') else 19,32))
                          for n in ('q','k','v','do')],q_heads=4,kv_heads=2,scale=.25,causal=True,
                        bias=arrays['bias'].astype(np.float64).reshape(4,9,19))
    fn=bind_registered(package.descriptor.entry_symbol)
    if fn is None: raise RuntimeError('mixed bias runtime unavailable')
    word=ct.POINTER(ct.c_uint16); floating=ct.POINTER(ct.c_float)
    def direct():
        status=fn(*[arrays[n].ctypes.data_as(word) for n in ('q','k','v','do')],
            *[arrays[n].ctypes.data_as(floating) for n in ('bias','dq','dk','dv')],
            4,4,2,9,19,32,.25,1,0,0.,2)
        if status != 1: raise RuntimeError('direct non-native dispatch')
    def packaged():
        result=launch(artifact,arrays)
        if not result['ok'] or result['execution_kind']!='native_gpu': raise RuntimeError(str(result))
    samples={route:[] for route in ('direct','package')}
    errors={route:[] for route in samples}
    calls={'direct':direct,'package':packaged}
    for trial in range(trials+1):
        for route in (('direct','package') if trial%2==0 else ('package','direct')):
            elapsed=[]
            for _ in range(reps):
                begin=time.perf_counter_ns(); calls[route](); elapsed.append(time.perf_counter_ns()-begin)
            for name, ref in zip(('dq','dk','dv'),expected):
                got=arrays[name].reshape(ref.shape)
                np.testing.assert_allclose(got,ref,rtol=.005,atol=.001)
                errors[route].append(float(np.max(np.abs(got-ref))))
            if trial: samples[route].append(int(statistics.median(elapsed)))
    return [dict(backend='apple_gpu',op='compiled_attention_backward',shape='b1_hq4_hkv2_sq9_sk19_d32_c1_bias_fp32',
        dtype=spelling,device='apple7',route=route,native_dispatched=True,numerically_validated=True,
        cold_compile_ns=cold,package_image_digest=package.image.image_digest,
        schedule_digest=scheduled.schedule_digest,tile_digest=scheduled.tile_digest,
        max_abs_error=max(errors[route]),reps=reps*trials,trials=trials,
        latency_ms=statistics.median(times)/1e6,telemetry=dict(
            paired_trial_end_to_end_medians_ns=times,end_to_end_median_ns=int(statistics.median(times)),
            device_time_median_ns=None,paired_trial_device_medians_ns=None,device_time_coverage=0.,
            timing_source='host_complete_native_call',resources={'scope':'native_package_descriptor_or_direct_identical_ABI'}))
        for route,times in samples.items()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trials',type=int,default=7)
    parser.add_argument('--reps',type=int,default=10)
    args=parser.parse_args()
    if args.trials < 2 or args.reps < 1: parser.error('positive repetitions and at least two trials required')
    rows=[r for dtype in ('fp16','bf16') for r in measure(dtype,args.trials,args.reps)]
    args.output.write_text(json.dumps(dict(schema_version=1,selection_scope='package_subgraph',
        package_kind='native_image_launch_descriptor',context=live_apple_route_context().as_mapping(),runs=rows),indent=2)+'\n')

if __name__=='__main__': main()
