import ctypes as C
import numpy as np
import os, sys

# Locate shared library in build dir via LD_LIBRARY_PATH
libname = "libtessera_runtime.so"
paths = [os.getcwd(), os.environ.get("LD_LIBRARY_PATH","")]
cand = None
for base in paths:
    for p in base.split(":"):
        if not p: continue
        libp = os.path.join(p, libname)
        if os.path.exists(libp):
            cand = libp
            break
    if cand: break
if cand is None:
    # fallback: try CWD
    if os.path.exists(libname):
        cand = os.path.abspath(libname)
    else:
        print("ERROR: could not find", libname)
        sys.exit(2)

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

tesseraMatmul = MatmulFn(("tesseraMatmul", lib))
tesseraBatchNorm = BatchNormFn(("tesseraBatchNorm", lib))

def as_tensor(a: np.ndarray):
    t = TsTensor()
    t.ptr = a.ctypes.data_as(C.c_void_p)
    t.device = -1
    t.dtype = 0  # f32
    t.rank = a.ndim
    shp = (C.c_longlong*8)(*([0]*8))
    for i,d in enumerate(a.shape): shp[i] = d
    t.shape = shp
    return t

def run_matmul_test():
    M,K,N = 8, 16, 12
    A = np.random.randn(M,K).astype(np.float32)
    B = np.random.randn(K,N).astype(np.float32)
    C_out = np.empty((M,N), dtype=np.float32)
    tsA, tsB, tsC = as_tensor(A), as_tensor(B), as_tensor(C_out)
    rc = tesseraMatmul(tsA, tsB, tsC)
    assert rc == 0, f"Matmul failed rc={rc}"
    ref = A @ B
    err = np.max(np.abs(C_out - ref))
    print("Matmul max abs err:", err)
    assert err < 1e-4

def run_bn_test():
    N, Cdim = 10, 7
    X = np.random.randn(N, Cdim).astype(np.float32)
    mean = np.random.randn(Cdim).astype(np.float32)
    var = np.abs(np.random.randn(Cdim).astype(np.float32)) + 1e-3
    gamma = np.random.randn(Cdim).astype(np.float32)
    beta = np.random.randn(Cdim).astype(np.float32)
    Y = np.empty_like(X)

    rc = tesseraBatchNorm(as_tensor(X), as_tensor(mean), as_tensor(var),
                          as_tensor(gamma), as_tensor(beta), as_tensor(Y), C.c_float(1e-5))
    assert rc == 0, f"BatchNorm failed rc={rc}"
    ref = gamma*(X-mean)/np.sqrt(var+1e-5)+beta
    err = np.max(np.abs(Y - ref))
    print("BatchNorm max abs err:", err)
    assert err < 1e-5

if __name__ == "__main__":
    run_matmul_test()
    run_bn_test()
    print("OK")
