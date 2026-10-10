"""Compiler-free native reverse execution and separate timing domains."""
import ctypes as c,hashlib,itertools,json,os,subprocess,time
from pathlib import Path
import numpy as np
import ml_dtypes
from tessera.compiler.native_scaled_program import NativeScaledProgram,PreparedScaledProgram
from tessera import runtime as rt

def make_inputs(row,seed):
    rng=np.random.default_rng(seed)
    ta,tb=row["transposeA"],row["transposeB"]
    suffix=((7,3) if ta else (3,7),(5,7) if tb else (7,5),(3,2),(2,2))
    values=tuple((rng.uniform(-.5,.5,size=(*prefix,*shape)).astype(ml_dtypes.float8_e4m3fn)
                  if index<2 else rng.uniform(.3,1.3,size=(*prefix,*shape)).astype(np.float32))
        for index,(prefix,shape) in enumerate(zip(row["prefixes"],suffix)))
    cot=rng.uniform(-1,1,size=(*row["output_prefix"],3,5)).astype(np.float32)
    return (*values,cot)

def oracle(inputs,row):
    a,b,sa,sb,cot=(x.astype(np.float64) for x in inputs)
    if row["transposeA"]:a=a.swapaxes(-1,-2)
    if row["transposeB"]:b=b.swapaxes(-1,-2)
    prefix=tuple(row["output_prefix"])
    dsa=np.zeros_like(sa);dsb=np.zeros_like(sb)
    def index(x,plane,i,j):
        own=x.shape[:-2];padded=(1,)*(len(prefix)-len(own))+own
        coordinates=tuple(0 if extent==1 else axis for extent,axis in zip(padded,plane))
        return (*coordinates[len(prefix)-len(own):],i,j)
    for plane in np.ndindex(prefix):
        for i,j,g in itertools.product(range(3),range(5),range(2)):
            partial=0.
            for k in range(g*4,min((g+1)*4,7)):
                partial+=a[index(a,plane,i,k)]*b[index(b,plane,k,j)]
            sai=index(sa,plane,i,g);sbi=index(sb,plane,g,j//3)
            weight=partial*cot[(*plane,i,j)]
            dsa[sai]+=weight*sb[sbi]
            dsb[sbi]+=weight*sa[sai]
    return dsa,dsb

def main():
    if rt._rocm_live_arch()!="gfx1201":raise RuntimeError("owning gfx1201 is required")
    hip=c.CDLL("/opt/rocm/lib/libamdhip64.so")
    hip.hipInit(0)
    name=c.create_string_buffer(256);uuid=(c.c_ubyte*16)()
    if hip.hipDeviceGetName(name,256,0) or hip.hipDeviceGetUuid(c.byref(uuid),0):
        raise RuntimeError("device identity query failed")
    payload=json.loads(Path("packages.json").read_text())
    runtime=Path(os.environ["TESSERA_ROCM_NATIVE_PROGRAM_LIB"])
    def forbidden(*args,**kwargs):
        raise AssertionError("serialized reverse replay attempted a compiler subprocess")
    subprocess.run=forbidden
    results=[]
    for number,row in enumerate(payload["rows"]):
        package=NativeScaledProgram.from_manifest(row["package"])
        inputs=make_inputs(row,100+number);expected=oracle(inputs,row)
        with PreparedScaledProgram(package,inputs,runtime_library=str(runtime)) as owner:
            old=None;maximum=0.
            for frame in range(2):
                if frame:
                    inputs=make_inputs(row,500+number);expected=oracle(inputs,row)
                    owner.update(inputs)
                generation,_=owner.invoke();outputs=owner.read(generation)
                if old is not None:
                    try:owner.read(old)
                    except RuntimeError:pass
                    else:raise AssertionError("stale output generation accepted")
                old=generation
                for actual,wanted in zip(outputs,expected,strict=True):
                    np.testing.assert_allclose(actual,wanted,rtol=2e-5,atol=1e-5)
                    maximum=max(maximum,float(np.max(np.abs(actual-wanted))))
            events=[];walls=[]
            for _ in range(5):
                _,elapsed=owner.invoke(repeats=20,timed=True);events.append(elapsed/20)
                start=time.perf_counter()
                for _ in range(5):
                    owner.update(inputs);generation,_=owner.invoke();outputs=owner.read(generation)
                walls.append((time.perf_counter()-start)*1e3/5)
                for actual,wanted in zip(outputs,expected,strict=True):
                    np.testing.assert_allclose(actual,wanted,rtol=2e-5,atol=1e-5)
            results.append({k:row[k] for k in ("mask","prefixes","output_prefix","transposeA","transposeB")}|
                {"correctness":"passed_before_and_after_timing","changed_frame_replay":"passed",
                 "stale_generation":"rejected","max_abs_error":maximum,
                 "native_launch_window_samples_ms":events,"native_launch_window_median_ms":float(np.median(events)),
                 "prepared_host_update_invoke_read_samples_ms":walls,
                 "prepared_host_update_invoke_read_median_ms":float(np.median(walls))})
        print("passed",number+1,"/",len(payload["rows"]),flush=True)
    out={"architecture":"gfx1201","device":name.value.decode(),"device_uuid_hex":bytes(uuid).hex(),
         "compiler_sha256":payload["compiler_sha256"],"runtime_sha256":hashlib.sha256(runtime.read_bytes()).hexdigest(),
         "package_file_sha256":hashlib.sha256(Path("packages.json").read_bytes()).hexdigest(),
         "recorder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "compiler_free_replay":True,"route":"Graph native scale transpose -> Schedule -> Tile structured reduction -> ROCm Target -> LLVM -> HSACO -> native HIP owner",
         "timing_scope":"two-member native launch windows and prepared host update/invoke/read; no isolated kernel or speedup claim",
         "rows":results}
    Path("device-results.json").write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":main()
