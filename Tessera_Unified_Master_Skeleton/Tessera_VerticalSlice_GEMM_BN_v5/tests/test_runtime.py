import ctypes as C
import numpy as np
import os, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu","cuda"], default="cpu")
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

MatmulFn = C.CFUNCTYPE(C.c_int, C.POINTER(TsTensor), C.POINTER(TsTensor), C.POINTER(TsTensor))
BatchNormFn = C.CFUNCTYPE(C.c_int, C.POINTER(TsTensor), C.POINTER(TsTensor), C.POINTER(TsTensor),
                          C.POINTER(TsTensor), C.POINTER(TsTensor), C.POINTER(TsTensor), C.c_float)
AllocFn = C.CFUNCTYPE(C.c_int, C.POINTER(TsTensor), C.c_int, C.c_int, C.POINTER(C.c_longlong))
FreeFn = C.CFUNCTYPE(C.c_int, C.POINTER(TsTensor))

tesseraMatmul = MatmulFn(("tesseraMatmul", lib))
tesseraBatchNorm = BatchNormFn(("tesseraBatchNorm", lib))
tesseraAllocHost = AllocFn(("tesseraAllocHost", lib))
tesseraFreeHost = FreeFn(("tesseraFreeHost", lib))

def as_tensor_host(arr: np.ndarray):
    t = TsTensor()
    # Ensure C-contig float32
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    t.ptr = arr.ctypes.data_as(C.c_void_p)
    t.device = -1
    t.dtype = 0
    t.rank = arr.ndim
    shp = (C.c_longlong*8)(*([0]*8))
    for i,d in enumerate(arr.shape): shp[i]=d
    t.shape = shp
    return t, arr

def alloc_dev_tensor(shape):
    # For demo simplicity, we reuse host alloc and assume external code copies to device;
    # in real runtime, you'd have device alloc and cudaMemcpy. Here we just flag device>=0.
    t = TsTensor()
    shp = (C.c_longlong*8)(*([0]*8))
    for i,d in enumerate(shape): shp[i]=d
    arr = np.zeros(shape, dtype=np.float32)
    t.ptr = arr.ctypes.data_as(C.c_void_p)
    t.device = 0
    t.dtype = 0
    t.rank = len(shape); t.shape = shp
    return t, arr

def run_matmul(device="cpu"):
    M,K,N = 64, 96, 48
    A = np.random.randn(M,K).astype(np.float32)
    B = np.random.randn(K,N).astype(np.float32)
    ref = A @ B
    if device=="cpu":
        tA, A_ = as_tensor_host(A)
        tB, B_ = as_tensor_host(B)
        C_out = np.empty((M,N), dtype=np.float32); tC, C_ = as_tensor_host(C_out)
    else:
        tA, A_ = alloc_dev_tensor((M,K))
        tB, B_ = alloc_dev_tensor((K,N))
        tC, C_ = alloc_dev_tensor((M,N))
        # NOTE: demo keeps data on host pointer but marks device=0 so CUDA path is exercised.
        # In a real runtime, these would be device pointers.
        A_[:] = A; B_[:] = B
    rc = tesseraMatmul(tA, tB, tC); assert rc==0, f"rc={rc}"
    err = np.max(np.abs(C_ - ref))
    print(f"[{device}] GEMM max abs err:", err)
    assert err < 1e-2 if device=="cuda" else err < 1e-4

def run_bn(device="cpu"):
    N, Cdim = 128, 64
    X = np.random.randn(N, Cdim).astype(np.float32)
    mean = np.random.randn(Cdim).astype(np.float32)
    var = np.abs(np.random.randn(Cdim).astype(np.float32)) + 1e-3
    gamma = np.random.randn(Cdim).astype(np.float32)
    beta = np.random.randn(Cdim).astype(np.float32)
    ref = gamma*(X-mean)/np.sqrt(var+1e-5)+beta
    if device=="cpu":
        tX, X_ = as_tensor_host(X)
        tM, M_ = as_tensor_host(mean)
        tV, V_ = as_tensor_host(var)
        tG, G_ = as_tensor_host(gamma)
        tB, B_ = as_tensor_host(beta)
        Y = np.empty_like(X); tY, Y_ = as_tensor_host(Y)
    else:
        tX, X_ = alloc_dev_tensor((N,Cdim)); X_[:] = X
        tM, M_ = alloc_dev_tensor((Cdim,)); M_[:] = mean
        tV, V_ = alloc_dev_tensor((Cdim,)); V_[:] = var
        tG, G_ = alloc_dev_tensor((Cdim,)); G_[:] = gamma
        tB, B_ = alloc_dev_tensor((Cdim,)); B_[:] = beta
        tY, Y_ = alloc_dev_tensor((N,Cdim))
    rc = tesseraBatchNorm(tX, tM, tV, tG, tB, tY, C.c_float(1e-5)); assert rc==0, f"rc={rc}"
    err = np.max(np.abs(Y_ - ref))
    print(f"[{device}] BatchNorm max abs err:", err)
    assert err < 1e-2 if device=="cuda" else err < 1e-5

if __name__ == "__main__":
    dev = args.device
    run_matmul(dev)
    run_bn(dev)
    print("OK:", dev)
