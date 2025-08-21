import ctypes as C
import numpy as np
import os, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu","cuda"], default="cpu")
parser.add_argument("--mma", action="store_true", help="Prefer WMMA path for GEMM (CUDA only)")
args = parser.parse_args()

libname = "libtessera_runtime.so"
cand = None
for p in [os.getcwd()] + os.environ.get("LD_LIBRARY_PATH","").split(":"):
    if not p: continue
    libp = os.path.join(p, libname)
    if os.path.exists(libp):
        cand = libp; break
if cand is None:
    print("ERROR: could not find", libname); sys.exit(2)
lib = C.CDLL(cand)

class TsTensor(C.Structure):
    _fields_ = [("ptr", C.c_void_p),
                ("device", C.c_int),
                ("dtype", C.c_int),
                ("rank", C.c_int),
                ("shape", C.c_longlong*8)]

class TsStream(C.Structure):
    _fields_ = [("impl", C.c_void_p), ("device", C.c_int)]

# Signatures
MatmulAsync = C.CFUNCTYPE(C.c_int, C.POINTER(TsTensor), C.POINTER(TsTensor), C.POINTER(TsTensor), C.POINTER(TsStream))
BatchNormAsync = C.CFUNCTYPE(C.c_int, C.POINTER(TsTensor), C.POINTER(TsTensor), C.POINTER(TsTensor),
                             C.POINTER(TsTensor), C.POINTER(TsTensor), C.POINTER(TsTensor), C.c_float, C.POINTER(TsStream))
AllocHost = C.CFUNCTYPE(C.c_int, C.POINTER(TsTensor), C.c_int, C.c_int, C.POINTER(C.c_longlong))
FreeHost = C.CFUNCTYPE(C.c_int, C.POINTER(TsTensor))
AllocDev = C.CFUNCTYPE(C.c_int, C.POINTER(TsTensor), C.c_int, C.c_int, C.c_int, C.POINTER(C.c_longlong))
FreeDev = C.CFUNCTYPE(C.c_int, C.POINTER(TsTensor))
H2D = C.CFUNCTYPE(C.c_int, C.POINTER(TsTensor), C.POINTER(TsTensor))
D2H = C.CFUNCTYPE(C.c_int, C.POINTER(TsTensor), C.POINTER(TsTensor))
SetDev = C.CFUNCTYPE(C.c_int, C.c_int)
StrmCreate = C.CFUNCTYPE(C.c_int, C.c_int, C.POINTER(TsStream))
StrmDestroy = C.CFUNCTYPE(C.c_int, C.POINTER(TsStream))
StrmSync = C.CFUNCTYPE(C.c_int, C.POINTER(TsStream))

tesseraMatmulAsync = MatmulAsync(("tesseraMatmulAsync", lib))
tesseraBatchNormAsync = BatchNormAsync(("tesseraBatchNormAsync", lib))
tesseraAllocHost = AllocHost(("tesseraAllocHost", lib))
tesseraFreeHost = FreeHost(("tesseraFreeHost", lib))
tesseraAllocDevice = AllocDev(("tesseraAllocDevice", lib))
tesseraFreeDevice = FreeDev(("tesseraFreeDevice", lib))
tesseraCopyHostToDevice = H2D(("tesseraCopyHostToDevice", lib))
tesseraCopyDeviceToHost = D2H(("tesseraCopyDeviceToHost", lib))
tesseraSetDevice = SetDev(("tesseraSetDevice", lib))
tesseraStreamCreate = StrmCreate(("tesseraStreamCreate", lib))
tesseraStreamDestroy = StrmDestroy(("tesseraStreamDestroy", lib))
tesseraStreamSync = StrmSync(("tesseraStreamSync", lib))

def make_host_tensor(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    t = TsTensor(); shp = (C.c_longlong*8)(*([0]*8))
    for i,d in enumerate(arr.shape): shp[i]=d
    rc = tesseraAllocHost(t, 0, arr.ndim, shp); assert rc==0
    # Copy into host allocation
    dst = (C.c_float * arr.size).from_address(t.ptr)
    dst[:] = arr.reshape(-1)
    return t, arr

def make_dev_tensor(shape, device=0):
    t = TsTensor(); shp = (C.c_longlong*8)(*([0]*8))
    for i,d in enumerate(shape): shp[i]=d
    rc = tesseraAllocDevice(t, device, 0, len(shape), shp); assert rc==0
    return t

def tensor_to_numpy_host(t: TsTensor):
    elems=1
    shape=[]
    for i in range(t.rank):
        d = t.shape[i]
        elems*=d; shape.append(d)
    buf = (C.c_float * elems).from_address(t.ptr)
    arr = np.frombuffer(buf, dtype=np.float32).reshape(shape).copy()
    return arr

def run_matmul(dev="cpu", mma=False):
    M,K,N=128,256,96
    A_h = np.random.randn(M,K).astype(np.float32)
    B_h = np.random.randn(K,N).astype(np.float32)
    ref = A_h @ B_h

    if dev=="cpu":
        A_t, _ = make_host_tensor(A_h)
        B_t, _ = make_host_tensor(B_h)
        C_t, _ = make_host_tensor(np.empty((M,N),np.float32))
        s = TsStream()
        tesseraStreamCreate(-1, C.byref(s))
        rc = tesseraMatmulAsync(C.byref(A_t), C.byref(B_t), C.byref(C_t), C.byref(s)); assert rc==0
        tesseraStreamSync(C.byref(s))
        C_out = tensor_to_numpy_host(C_t)
        tesseraStreamDestroy(C.byref(s))
        tesseraFreeHost(C.byref(A_t)); tesseraFreeHost(C.byref(B_t)); tesseraFreeHost(C.byref(C_t))
        err = np.max(np.abs(C_out-ref)); print("[cpu] GEMM err:", err); assert err < 1e-4
    else:
        if mma: os.environ["TESSERA_MMA"]="1"
        tesseraSetDevice(0)
        A_host, _ = make_host_tensor(A_h)
        B_host, _ = make_host_tensor(B_h)
        A_dev = make_dev_tensor((M,K), 0)
        B_dev = make_dev_tensor((K,N), 0)
        C_dev = make_dev_tensor((M,N), 0)
        rc = tesseraCopyHostToDevice(C.byref(A_host), C.byref(A_dev)); assert rc==0
        rc = tesseraCopyHostToDevice(C.byref(B_host), C.byref(B_dev)); assert rc==0
        s = TsStream(); tesseraStreamCreate(0, C.byref(s))
        rc = tesseraMatmulAsync(C.byref(A_dev), C.byref(B_dev), C.byref(C_dev), C.byref(s)); assert rc==0
        tesseraStreamSync(C.byref(s))
        C_host = make_host_tensor(np.empty((M,N),np.float32))[0]
        rc = tesseraCopyDeviceToHost(C.byref(C_dev), C.byref(C_host)); assert rc==0
        C_out = tensor_to_numpy_host(C_host)
        # cleanup
        tesseraStreamDestroy(C.byref(s))
        tesseraFreeDevice(C.byref(A_dev)); tesseraFreeDevice(C.byref(B_dev)); tesseraFreeDevice(C.byref(C_dev))
        tesseraFreeHost(C.byref(A_host)); tesseraFreeHost(C.byref(B_host)); tesseraFreeHost(C.byref(C_host))
        err = np.max(np.abs(C_out-ref)); print("[cuda] GEMM err:", err); assert err < (5e-1 if mma else 1e-2)

def run_bn(dev="cpu"):
    N,Cdim=1024,128
    X = np.random.randn(N,Cdim).astype(np.float32)
    mean = np.random.randn(Cdim).astype(np.float32)
    var = np.abs(np.random.randn(Cdim).astype(np.float32)) + 1e-3
    gamma = np.random.randn(Cdim).astype(np.float32)
    beta = np.random.randn(Cdim).astype(np.float32)
    ref = gamma*(X-mean)/np.sqrt(var+1e-5)+beta

    if dev=="cpu":
        X_t,_=make_host_tensor(X); M_t,_=make_host_tensor(mean); V_t,_=make_host_tensor(var)
        G_t,_=make_host_tensor(gamma); B_t,_=make_host_tensor(beta); Y_t,_=make_host_tensor(np.empty_like(X))
        s = TsStream(); tesseraStreamCreate(-1, C.byref(s))
        rc = tesseraBatchNormAsync(C.byref(X_t),C.byref(M_t),C.byref(V_t),C.byref(G_t),C.byref(B_t),C.byref(Y_t), C.c_float(1e-5), C.byref(s)); assert rc==0
        tesseraStreamSync(C.byref(s))
        out = tensor_to_numpy_host(Y_t)
        tesseraStreamDestroy(C.byref(s))
        [tesseraFreeHost(C.byref(t)) for t in (X_t,M_t,V_t,G_t,B_t,Y_t)]
        err = np.max(np.abs(out-ref)); print("[cpu] BN err:", err); assert err < 1e-5
    else:
        tesseraSetDevice(0)
        # host tensors for source
        Xh,_=make_host_tensor(X); Mh,_=make_host_tensor(mean); Vh,_=make_host_tensor(var); Gh,_=make_host_tensor(gamma); Bh,_=make_host_tensor(beta)
        # device tensors
        Xd=make_dev_tensor((N,Cdim),0); Md=make_dev_tensor((Cdim,),0); Vd=make_dev_tensor((Cdim,),0); Gd=make_dev_tensor((Cdim,),0); Bd=make_dev_tensor((Cdim,),0); Yd=make_dev_tensor((N,Cdim),0)
        tesseraCopyHostToDevice(C.byref(Xh), C.byref(Xd)); tesseraCopyHostToDevice(C.byref(Mh), C.byref(Md))
        tesseraCopyHostToDevice(C.byref(Vh), C.byref(Vd)); tesseraCopyHostToDevice(C.byref(Gh), C.byref(Gd))
        tesseraCopyHostToDevice(C.byref(Bh), C.byref(Bd))
        s = TsStream(); tesseraStreamCreate(0, C.byref(s))
        rc = tesseraBatchNormAsync(C.byref(Xd),C.byref(Md),C.byref(Vd),C.byref(Gd),C.byref(Bd),C.byref(Yd), C.c_float(1e-5), C.byref(s)); assert rc==0
        tesseraStreamSync(C.byref(s))
        Yh,_=make_host_tensor(np.empty_like(X))
        tesseraCopyDeviceToHost(C.byref(Yd), C.byref(Yh))
        out = tensor_to_numpy_host(Yh)
        tesseraStreamDestroy(C.byref(s))
        [tesseraFreeDevice(C.byref(t)) for t in (Xd,Md,Vd,Gd,Bd,Yd)]
        [tesseraFreeHost(C.byref(t)) for t in (Xh,Mh,Vh,Gh,Bh,Yh)]
        err = np.max(np.abs(out-ref)); print("[cuda] BN err:", err); assert err < 1e-2

if __name__ == "__main__":
    dev = args.device
    run_matmul(dev, mma=args.mma and dev=="cuda")
    run_bn(dev)
    print("OK:", dev)
