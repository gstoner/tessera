import json
import numpy as np
import tessera as ts
from tessera import Tensor

@ts.jit(target="apple_gpu")
def rope(x: ts.f32[4,8], theta: ts.f32[4,8]) -> ts.f32[4,8]:
    return ts.ops.ntk_rope(x, theta, scale=2.0)

@ts.jit(target="apple_gpu")
def verify(tokens: "tensor<4xi32>", logits: ts.f32[4,8]) -> ts.f32[4,8]:
    return ts.ops.target_verify(tokens, logits)

rows=[]
x=np.linspace(-.75,.75,32,dtype=np.float32).reshape(4,8)
theta=np.linspace(-.3,.6,32,dtype=np.float32).reshape(4,8)
for name,fn,args in [('ntk_rope',rope,(x,theta)),('target_verify',verify,(np.arange(4,dtype=np.int32),x))]:
    row={'operation':name}
    try:
        out=fn(*args)
        row['output_shape']=list(np.asarray(out).shape)
        artifact=fn.runtime_artifact()
        row['metadata']=artifact.metadata
        row['has_native_image']=artifact.native_image is not None
        row['has_launch_descriptor']=artifact.launch_descriptor is not None
    except Exception as exc:
        row['error']=str(exc);row['error_type']=type(exc).__name__
    rows.append(row)
print(json.dumps(rows,indent=2,default=str))
