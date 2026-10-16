"""Independent FP64 oracle and authored frontend for source-active JVP evidence."""
import numpy as np
import ml_dtypes
import tessera as ts


def inputs(packet,profile='mixed'):
    m,n,k=packet['shape_mnk'];dtype=np.float16 if packet['storage']=='f16' else ml_dtypes.bfloat16
    rng=np.random.default_rng(551)
    x=rng.normal(0,2,(m,k));dx=rng.normal(0,0.5,(m,k))
    if profile=='overflow':
        x[:]=64;x[:,1::2]=-64
        dx[:]=np.resize([0.015625,-0.015625,0.03125,-0.03125],x.shape)
    elif profile=='zero':x[:]=0
    elif profile=='threshold':
        x*=1e-5;x[:,::3]=0
    b=np.random.default_rng(552).normal(0,0.25,(k,n))
    return x.astype(dtype),b.astype(dtype),dx.astype(dtype)

def oracle(x,b,dx,epsilon=1e-5):
    storage=x.dtype
    wide=x.astype(np.float64);seed=dx.astype(np.float64);rhs=b.astype(np.float64)
    inv=1/np.sqrt(np.mean(wide*wide,axis=1,keepdims=True)+epsilon)
    z=wide*inv;rounded=z.astype(storage)
    primal=rounded.copy()
    primal[rounded<0]=storage.type(0)
    if storage!=np.dtype(ml_dtypes.bfloat16):
        primal[rounded==0]=storage.type(0)
    tangent=((seed-z*np.mean(seed*z,axis=1,keepdims=True))*inv).astype(storage)
    tangent=np.where(rounded>0,tangent,storage.type(0)).astype(storage)
    expected=[];bounds=[]
    for edge in [primal,tangent]:
        value=edge.astype(np.float64)
        upper=np.nextafter(edge,storage.type(np.inf));lower=np.nextafter(edge,storage.type(-np.inf))
        assert upper.dtype==storage and lower.dtype==storage
        up=upper.astype(np.float64);down=lower.astype(np.float64)
        ulp=np.maximum(np.abs(up-value),np.abs(value-down))
        magnitude=np.abs(value)@np.abs(rhs)
        count=x.shape[1];gamma=count*np.finfo(np.float32).eps/(1-count*np.finfo(np.float32).eps)
        expected.append(value@rhs)
        # Prefix proof allows one adjacent storage value; propagate that error
        # through the independent FP64 product, plus FP32 accumulation gamma.
        bounds.append(ulp@np.abs(rhs)+gamma*magnitude+1e-7)
    return expected,bounds

def product(a,b):
    return ts.ops.matmul(ts.ops.relu(ts.ops.rmsnorm(a,eps=1e-5,
        numeric_policy={'accum':'fp32'})),b,output_dtype='fp32')

def product_reversed(b,a):
    return ts.ops.matmul(ts.ops.relu(ts.ops.rmsnorm(a,eps=1e-5,
        numeric_policy={'accum':'fp32'})),b,output_dtype='fp32')
