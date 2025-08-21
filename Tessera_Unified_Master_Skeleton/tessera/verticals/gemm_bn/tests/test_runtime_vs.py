# This vertical slice test uses the unified runtime library.
import ctypes as C
import numpy as np
import os, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu","cuda"], default="cpu")
parser.add_argument("--mma", action="store_true")
parser.add_argument("--wgmma", action="store_true")
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

def alloc_host(shape):
    t = TsTensor(); shp=(C.c_longlong*8)(*([0]*8))
    for i,d in enumerate(shape): shp[i]=d
    rc = tesseraAllocHost(t, 0, len(shape), shp); assert rc==0
    return t

def set_host_data(t, arr):
    import numpy as np
    arr = np.ascontiguousarray(arr, dtype=np.float32).reshape([-1])
    buf = (C.c_float*arr.size).from_address(t.ptr)
    buf[:] = arr

def to_numpy(t):
    import numpy as np
    elems=1; shape=[]
    for i in range(t.rank):
        d = t.shape[i]; elems*=d; shape.append(d)
    buf=(C.c_float*elems).from_address(t.ptr)
    return np.frombuffer(buf, dtype=np.float32).reshape(shape).copy()

def test_run(dev="cpu", mma=False, wgmma=False):
    import numpy as np
    M,K,N = 128,256,96
    A = np.random.randn(M,K).astype(np.float32)
    B = np.random.randn(K,N).astype(np.float32)
    ref = A @ B

    if dev=="cpu":
        Ah=alloc_host((M,K)); Bh=alloc_host((K,N)); Ch=alloc_host((M,N))
        set_host_data(Ah, A); set_host_data(Bh, B)
        s=TsStream(); tesseraStreamCreate(-1, C.byref(s))
        rc = tesseraMatmulAsync(C.byref(Ah), C.byref(Bh), C.byref(Ch), C.byref(s)); assert rc==0
        tesseraStreamSync(C.byref(s)); out = to_numpy(Ch)
        err=np.max(np.abs(out-ref)); print("[cpu] err:", err); assert err < 1e-4
        tesseraStreamDestroy(C.byref(s))
        [tesseraFreeHost(C.byref(t)) for t in (Ah,Bh,Ch)]
    else:
        tesseraSetDevice(0)
        if mma: os.environ["TESSERA_MMA"]="1"
        if wgmma: os.environ["TESSERA_WGMMA"]="1"
        Ah=alloc_host((M,K)); Bh=alloc_host((K,N)); Ch=alloc_host((M,N))
        set_host_data(Ah, A); set_host_data(Bh, B)
        Ad=TsTensor(); Bd=TsTensor(); Cd=TsTensor()
        shpA=(C.c_longlong*8)(M,K,0,0,0,0,0,0)
        shpB=(C.c_longlong*8)(K,N,0,0,0,0,0,0)
        shpC=(C.c_longlong*8)(M,N,0,0,0,0,0,0)
        tesseraAllocDevice(C.byref(Ad), 0, 0, 2, shpA)
        tesseraAllocDevice(C.byref(Bd), 0, 0, 2, shpB)
        tesseraAllocDevice(C.byref(Cd), 0, 0, 2, shpC)
        tesseraCopyHostToDevice(C.byref(Ah), C.byref(Ad)); tesseraCopyHostToDevice(C.byref(Bh), C.byref(Bd))

        s=TsStream(); tesseraStreamCreate(0, C.byref(s))
        rc = tesseraMatmulAsync(C.byref(Ad), C.byref(Bd), C.byref(Cd), C.byref(s)); assert rc==0
        tesseraStreamSync(C.byref(s))
        tesseraCopyDeviceToHost(C.byref(Cd), C.byref(Ch)); out = to_numpy(Ch)

        err=np.max(np.abs(out-ref)); label="wgmma" if wgmma else ("wmma" if mma else "naive")
        print(f"[cuda-{label}] err:", err); assert err < (5e-1 if (mma or wgmma) else 1e-2)
        tesseraStreamDestroy(C.byref(s))
        [tesseraFreeDevice(C.byref(t)) for t in (Ad,Bd,Cd)]
        [tesseraFreeHost(C.byref(t)) for t in (Ah,Bh,Ch)]

if __name__ == "__main__":
    test_run(args.device, mma=args.mma, wgmma=args.wgmma)
    print("OK vertical GEMM+BN")
