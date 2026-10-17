"""Old/new producer metadata compatibility; actual gfx1201 execution required."""
from pathlib import Path
import hashlib,json,sys
import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
v=Path('/home/angstorms/scratch/rocm-native-gate-up-views-20261016')
assert rt._rocm_live_arch()=='gfx1201'
phase=sys.argv[1];results=[]
for name,adt,bdt in [('f16',np.float16,np.float16),('mixed_fp8',ml_dtypes.float8_e4m3fn,ml_dtypes.float8_e5m2)]:
    m,k,n=17,31,19
    if phase=='capture':
        rng=np.random.default_rng(964)
        a=rng.uniform(-.5,.5,(m,k)).astype(adt);b=rng.uniform(-.5,.5,(k,n)).astype(bdt)
        expected=a.astype(np.float32)@b.astype(np.float32)
        f=ts.from_text('def product(a,b):\n    return tessera.ops.matmul(a,b,output_dtype="fp32")\n',target='rocm_gfx1201')
        np.testing.assert_allclose(f(a,b),expected,rtol=1e-4,atol=1e-3)
        artifact=f.runtime_artifact();assert artifact.native_image.pipeline_name=='tessera-lower-to-rocm'
        (v/f'pre-resolution-{name}.json').write_text(artifact.to_json())
        np.savez(v/f'pre-resolution-{name}.npz',a=a.view(np.uint8).reshape(-1),b=b.view(np.uint8).reshape(-1),expected=expected)
        results.append({'case':name,'producer':artifact.native_image.pipeline_name,'payload_sha256':hashlib.sha256(artifact.native_image.payload).hexdigest()})
    else:
        original=rt.RuntimeArtifact.from_json((v/f'pre-resolution-{name}.json').read_text())
        data=np.load(v/f'pre-resolution-{name}.npz');a=data['a'].view(adt).reshape(m,k);b=data['b'].view(bdt).reshape(k,n)
        inputs=[x.name for x in original.launch_descriptor.buffers if x.direction=='input'];output=next(x.name for x in original.launch_descriptor.buffers if x.direction=='output')
        out=np.empty((m,n),np.float32)
        receipt=rt.launch(original,{'buffers':{inputs[0]:a,inputs[1]:b,output:out},'scalars':{x.name:{'M':m,'N':n,'K':k}[x.name] for x in original.launch_descriptor.scalars}})
        assert receipt['ok'] and receipt['execution_kind']=='native_gpu',receipt
        np.testing.assert_allclose(out,data['expected'],rtol=1e-4,atol=1e-3)
        f=ts.from_text('def product(a,b):\n    return tessera.ops.matmul(a,b,output_dtype="fp32")\n',target='rocm_gfx1201')
        np.testing.assert_allclose(f(a,b),data['expected'],rtol=1e-4,atol=1e-3)
        new=f.runtime_artifact();assert new.native_image.pipeline_name=='tessera-rocm-executable'
        assert new.native_image.payload==original.native_image.payload
        results.append({'case':name,'legacy_replay':'native_gpu_numerically_passed','old_producer':original.native_image.pipeline_name,'new_producer':new.native_image.pipeline_name,'payload_sha256':hashlib.sha256(new.native_image.payload).hexdigest(),'physical_bytes_unchanged':True})
(v/f'resolution-compatibility-{phase}.json').write_text(json.dumps({'architecture':'gfx1201','scope':'canonical producer metadata and legacy serialized replay; no schedule/performance promotion','rows':results},indent=2)+'\n')
print(json.dumps(results,indent=2))
