"""Bounded scheduled GEMM/attention workloads with independent NumPy oracles."""
import hashlib
import numpy as np


def prepare(backend, workload):
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
    from tessera.compiler import scheduled_matmul, scheduled_attention, nvidia_native, rocm_native
    from tessera import runtime as rt
    target = 'nvidia_sm120' if backend == 'nvidia' else 'rocm_gfx1151'
    native = nvidia_native if backend == 'nvidia' else rocm_native
    pipeline = 'tessera-nvidia-pipeline-sm120' if backend == 'nvidia' else 'tessera-lower-to-rocm'
    rng = np.random.default_rng(748)

    def ty(shape, half=False):
        return IRType('tensor<' + 'x'.join(map(str, shape)) + ('xf16>' if half else 'xf32>'),
                      tuple(map(str, shape)), 'fp16' if half else 'fp32')

    if workload == 'gemm':
        shapes = [(32, 64), (64, 48)]
        names, output_shape, half = ['a', 'b'], (32, 48), True
        kwargs, opname = {'activation': 'none'}, 'tessera.matmul'
    elif workload == 'attention':
        half = backend == 'rocm'
        d = 64 if half else 4
        shapes = [(1, 2, 8, d), (1, 2, 12, d), (1, 2, 12, d)]
        names, output_shape = ['q', 'k', 'v'], (1, 2, 8, d)
        kwargs, opname = {'scale': d ** -.5, 'causal': False}, 'tessera.flash_attn'
    else:
        raise ValueError('unknown native matrix workload')
    types = [ty(s, half) for s in shapes]
    out_type = ty(output_shape)
    module = GraphIRModule(functions=[GraphIRFunction(name='benchmark_' + workload,
        args=[IRArg(n,t) for n,t in zip(names,types,strict=True)], result_types=[out_type],
        body=[IROp(result='o',op_name=opname,operands=['%'+n for n in names],
                   operand_types=[str(t) for t in types],result_type=str(out_type),kwargs=kwargs)],
        return_values=['%o'])])
    lower = scheduled_matmul.lower_scheduled_matmul if workload == 'gemm' else scheduled_attention.lower_scheduled_attention
    scheduled = lower(module, target=target)
    package_fn = native.package_scheduled_matmul if workload == 'gemm' else native.package_scheduled_attention
    package = package_fn(scheduled, pipeline_name=pipeline)
    values = [rng.uniform(-.3,.3,s).astype(np.float16 if half else np.float32) for s in shapes]
    layouts = {b.name:b.layout for b in package.descriptor.buffers}
    values = [np.asfortranarray(v) if layouts[n] == 'col_major' else np.ascontiguousarray(v)
              for n,v in zip(names,values,strict=True)]
    if workload == 'gemm':
        expected = values[0].astype(np.float64) @ values[1].astype(np.float64)
    else:
        q,k,v = [a.astype(np.float64) for a in values]
        scores = (q @ k.swapaxes(-1,-2)) * kwargs['scale']
        probs = np.exp(scores - scores.max(axis=-1,keepdims=True))
        expected = (probs / probs.sum(axis=-1,keepdims=True)) @ v
    out = np.empty(output_shape,np.float32)
    runtime = rt.RuntimeArtifact(metadata={'target':target},native_image=package.image,
        launch_descriptor=package.descriptor,tile_ir=package.tile_ir,target_ir=package.target_ir)
    arguments = dict(zip(names,values,strict=True), o=out)
    if workload == 'gemm':
        arguments.update(M=32,N=48,K=64)
    else:
        arguments.update(zip(('B','Hq','Hkv','Sq','Sk','D','Dv'),scheduled.dims,strict=True))

    def run():
        result = rt.launch(runtime, arguments)
        if not result.get('ok') or result.get('execution_kind') != 'native_gpu':
            raise RuntimeError('native matrix execution failed: ' + str(result))
        np.testing.assert_allclose(out,expected,rtol=3e-3,atol=3e-4)
        return float(np.max(np.abs(out-expected)))

    identity = dict(workload=workload,backend=backend,shape=[list(s) for s in shapes],
        dtype='fp16' if half else 'fp32',artifact=scheduled.schedule_digest,
        image=package.image.image_digest,tile_sha256=hashlib.sha256(package.tile_ir.encode()).hexdigest())
    return run, identity
