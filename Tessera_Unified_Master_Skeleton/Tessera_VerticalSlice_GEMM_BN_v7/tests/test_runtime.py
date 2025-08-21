import ctypes as C
import numpy as np
import os, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu","cuda"], default="cpu")
parser.add_argument("--mma", action="store_true", help="Prefer WMMA path")
parser.add_argument("--wgmma", action="store_true", help="Prefer WGMMA path (SM90+)")
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

# Bindings
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

def make_host_tensor_empty(shape):
    t = TsTensor(); shp = (C.c_longlong*8)(*([0]*8))
    for i,d in enumerate(shape): shp[i]=d
    rc = tesseraAllocHost(t, 0, len(shape), shp); assert rc==0
    return t

def make_dev_tensor(shape, device=0):
    t = TsTensor(); shp = (C.c_longlong*8)(*([0]*8))
    for i,d in enumerate(shape): shp[i]=d
    rc = tesseraAllocDevice(t, device, 0, len(shape), shp); assert rc==0
    return t

def tensor_to_numpy_host(t: TsTensor):
    elems=1; shape=[]
    for i in range(t.rank):
        d = t.shape[i]; elems*=d; shape.append(d)
    buf = (C.c_float * elems).from_address(t.ptr)
    arr = np.frombuffer(buf, dtype=np.float32).reshape(shape).copy()
    return arr

def run_matmul_cuda(mma=False, wgmma=False):
    tesseraSetDevice(0)
    if mma: os.environ["TESSERA_MMA"]="1"
    if wgmma: os.environ["TESSERA_WGMMA"]="1"

    M,K,N = 256, 256, 256
    A = np.random.randn(M,K).astype(np.float32)
    B = np.random.randn(K,N).astype(np.float32)
    ref = A @ B

    A_h,_=make_host_tensor(A); B_h,_=make_host_tensor(B); C_h = make_host_tensor_empty((M,N))

    A_d=make_dev_tensor((M,K)); B_d=make_dev_tensor((K,N)); C_d=make_dev_tensor((M,N))
    tesseraCopyHostToDevice(C.byref(A_h), C.byref(A_d)); tesseraCopyHostToDevice(C.byref(B_h), C.byref(B_d))

    s = TsStream(); tesseraStreamCreate(0, C.byref(s))
    rc = tesseraMatmulAsync(C.byref(A_d), C.byref(B_d), C.byref(C_d), C.byref(s)); assert rc==0
    tesseraStreamSync(C.byref(s))

    tesseraCopyDeviceToHost(C.byref(C_d), C.byref(C_h))
    C_out = tensor_to_numpy_host(C_h)

    # Cleanup
    tesseraStreamDestroy(C.byref(s))
    [tesseraFreeDevice(C.byref(t)) for t in (A_d,B_d,C_d)]
    [tesseraFreeHost(C.byref(t)) for t in (A_h,B_h,C_h)]

    err = np.max(np.abs(C_out - ref))
    label = "wgmma" if wgmma else ("wmma" if mma else "naive")
    print(f"[cuda-{label}] GEMM err:", err)
    # Relax tolerances for demo kernels and fp16 paths
    tol = 5e-1 if (mma or wgmma) else 1e-2
    assert err < tol

def run_bn_cuda():
    tesseraSetDevice(0)
    N,Cdim=2048,256
    X=np.random.randn(N,Cdim).astype(np.float32)
    mean=np.random.randn(Cdim).astype(np.float32)
    var=np.abs(np.random.randn(Cdim).astype(np.float32))+1e-3
    gamma=np.random.randn(Cdim).astype(np.float32)
    beta=np.random.randn(Cdim).astype(np.float32)
    ref = gamma*(X-mean)/np.sqrt(var+1e-5)+beta

    Xh,_=make_host_tensor(X); Mh,_=make_host_tensor(mean); Vh,_=make_host_tensor(var); Gh,_=make_host_tensor(gamma); Bh,_=make_host_tensor(beta)
    Xd=make_dev_tensor((N,Cdim)); Md=make_dev_tensor((Cdim,)); Vd=make_dev_tensor((Cdim,)); Gd=make_dev_tensor((Cdim,)); Bd=make_dev_tensor((Cdim,)); Yd=make_dev_tensor((N,Cdim))
    tesseraCopyHostToDevice(C.byref(Xh), C.byref(Xd)); tesseraCopyHostToDevice(C.byref(Mh), C.byref(Md))
    tesseraCopyHostToDevice(C.byref(Vh), C.byref(Vd)); tesseraCopyHostToDevice(C.byref(Gh), C.byref(Gd))
    tesseraCopyHostToDevice(C.byref(Bh), C.byref(Bd))

    s = TsStream(); tesseraStreamCreate(0, C.byref(s))
    rc = tesseraBatchNormAsync(C.byref(Xd),C.byref(Md),C.byref(Vd),C.byref(Gd),C.byref(Bd),C.byref(Yd), C.c_float(1e-5), C.byref(s)); assert rc==0
    tesseraStreamSync(C.byref(s))

    Yh = make_host_tensor_empty((N,Cdim))
    tesseraCopyDeviceToHost(C.byref(Yd), C.byref(Yh))
    out = tensor_to_numpy_host(Yh)

    tesseraStreamDestroy(C.byref(s))
    [tesseraFreeDevice(C.byref(t)) for t in (Xd,Md,Vd,Gd,Bd,Yd)]
    [tesseraFreeHost(C.byref(t)) for t in (Xh,Mh,Vh,Gh,Bh,Yh)]

    err = np.max(np.abs(out-ref)); print("[cuda] BN err:", err); assert err < 1e-2

def run_cpu():
    # Quick CPU check
    M,K,N = 64, 96, 32
    A = np.random.randn(M,K).astype(np.float32)
    B = np.random.randn(K,N).astype(np.float32)
    ref = A @ B
    # CPU path uses host tensors and async APIs (no-op stream)
    lib.tesseraAllocHost.restype = C.c_int

    # Reuse helpers above
    from numpy.random import randn
    A_t,_=make_host_tensor(A); B_t,_=make_host_tensor(B); C_t=make_host_tensor_empty((M,N))
    s = TsStream(); lib.tesseraStreamCreate(-1, C.byref(s))
    rc = tesseraMatmulAsync(C.byref(A_t), C.byref(B_t), C.byref(C_t), C.byref(s)); assert rc==0
    lib.tesseraStreamSync(C.byref(s))
    out = tensor_to_numpy_host(C_t)
    lib.tesseraStreamDestroy(C.byref(s))
    [tesseraFreeHost(C.byref(t)) for t in (A_t,B_t,C_t)]
    err = np.max(np.abs(out-ref)); print("[cpu] GEMM err:", err); assert err < 1e-4

if __name__ == "__main__":
    if args.device == "cpu":
        run_cpu()
    else:
        run_matmul_cuda(mma=args.mma, wgmma=args.wgmma)
        run_bn_cuda()
    print("OK:", args.device)
