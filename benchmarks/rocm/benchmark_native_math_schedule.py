"""Textual Graph -> native Schedule/Tile -> ROCm image math characterization.

This recorder exercises the native compiler product directly. It does not
claim an ordinary @jit/portable package migration or narrow-storage coverage.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import time
import numpy as np
from tessera import runtime as rt
from tessera.compiler import rocm_native as native
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from benchmarks.rocm.benchmark_rocm_gemm_schedule_matrix import DeviceCase


def graph(kind, arch, shape, reverse=False):
    typ = "tensor<"+"x".join(map(str, shape))+"xf32>"
    binary = kind in {"add", "div"}
    args = "%a: "+typ+(", %b: "+typ if binary else "")
    operands = ("%b, %a" if reverse else "%a, %b") if binary else "%a"
    attrs = " {axis = -1 : i64}" if kind in {"cumsum", "cummax"} else ""
    bindings = '["a", "b", "out"]' if binary else '["a", "out"]'
    return (
        'module attributes {tessera.target = "rocm", tessera.arch = "'+arch+
        '", tessera.launch_bindings = '+bindings+'} {\n'
        ' func.func @math('+args+') -> '+typ+' {\n'
        '  %o = "tessera.'+kind+'"('+operands+')'+attrs+' : ('+
        ", ".join([typ]*(2 if binary else 1))+') -> '+typ+'\n'
        '  return %o : '+typ+'\n }\n}'
    ).replace("\\n", "\n")


class MathCase(DeviceCase):
    def __init__(self, hip, image, entry, arrays, scan, output_dtype=None):
        self.hip = hip
        self.mod = C.c_void_p()
        self.devs = []
        self.output = np.empty_like(arrays[0],dtype=output_dtype)
        self.scan = scan
        self.grid = (math.prod(arrays[0].shape[:-1]) if scan else (arrays[0].size+255)//256, 1)
        self.fn = C.c_void_p()
        try:
            self.check(hip.hipModuleLoadData(C.byref(self.mod), image), "module load")
            self.check(hip.hipModuleGetFunction(C.byref(self.fn), self.mod, entry.encode()), "entry")
            values = []
            for array in [*arrays, self.output]:
                dev = C.c_void_p()
                self.check(hip.hipMalloc(C.byref(dev), array.nbytes), "allocate")
                self.devs.append(dev)
                if array is not self.output:
                    self.check(hip.hipMemcpy(dev, C.c_void_p(array.ctypes.data), array.nbytes, 1), "upload")
                values += [C.c_void_p(dev.value), C.c_void_p(dev.value), C.c_int64(0),
                           C.c_int64(array.size), C.c_int64(1)]
            self.dd = self.devs[-1]
            values += ([C.c_int64(math.prod(arrays[0].shape[:-1])), C.c_int64(arrays[0].shape[-1])]
                       if scan else [C.c_int64(arrays[0].size)])
            self.arg_values = values
            self.arg_array = (C.c_void_p*len(values))()
            for i, value in enumerate(values):
                self.arg_array[i] = C.cast(C.byref(value), C.c_void_p)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(rc, stage):
        if rc:
            raise RuntimeError(f"native math {stage} failed: HIP {rc}")

    def launch(self):
        return self.hip.hipModuleLaunchKernel(self.fn, self.grid[0], 1, 1, 256, 1, 1,
                                             0, None, self.arg_array, None)

    def download(self):
        self.check(self.launch(), "launch")
        self.check(self.hip.hipDeviceSynchronize(), "completion")
        self.check(self.hip.hipMemcpy(C.c_void_p(self.output.ctypes.data), self.dd,
                                      self.output.nbytes, 2), "readback")
        return self.output.copy()


def expected(kind, arrays):
    a = arrays[0]
    if kind == "sqrt": return np.sqrt(a)
    if kind == "exp": return np.exp(a)
    if kind == "add": return a+arrays[1]
    if kind == "div": return a/arrays[1]
    if kind == "cumsum": return np.cumsum(a, axis=-1, dtype=np.float32)
    if kind == "cummax": return np.maximum.accumulate(a, axis=-1)
    raise ValueError(kind)


def record(hip, arch, kind, shape, reverse, samples, iterations):
    source = graph(kind, arch, shape, reverse)
    tool = find_tessera_opt()
    schedule = run_tessera_opt(tool, source, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    family = "scan" if kind in {"cumsum", "cummax"} else "scalar_binary" if kind in {"add", "div"} else "scalar_unary"
    directive = "tessera_rocm."+("scan" if family == "scan" else "binary" if family == "scalar_binary" else "unary")
    target, backend, image, compiler, toolchain, libraries, state = native._compile_native_tile_ir(
        tile, directive=directive, family=family, architecture=arch)
    entry = native._directive_symbol(target, directive)
    rng = np.random.default_rng(1151_1201)
    arrays = [rng.uniform(.125, 2, shape).astype(np.float32)]
    if family == "scalar_binary": arrays.append(rng.uniform(.5, 2, shape).astype(np.float32))
    # Target generators consume their canonical ABI in Tile operand-role order.
    if reverse: arrays.reverse()
    oracle = expected(kind, arrays)
    walls = []
    with_case = MathCase(hip, image, entry, arrays, family == "scan")
    try:
        actual = with_case.download()
        np.testing.assert_allclose(actual, oracle, rtol=2e-5, atol=2e-5)
        events = with_case.measure(trials=samples, iterations=iterations, warmup=5)
    finally:
        with_case.close()
    for _ in range(samples):
        start = time.perf_counter_ns()
        case = MathCase(hip, image, entry, arrays, family == "scan")
        try: actual = case.download()
        finally: case.close()
        walls.append((time.perf_counter_ns()-start)/1e6)
        np.testing.assert_allclose(actual, oracle, rtol=2e-5, atol=2e-5)
    return {"kind":kind, "shape":list(shape), "reversed_binary_roles":reverse,
            "correctness":"independent NumPy oracle before events and after each allocating call",
            "max_abs_error":float(np.max(np.abs(actual-oracle))),
            "route":"textual Graph->native Schedule->native Tile->ROCm Target->ROCDL/LLVM->HSACO",
            "entry_symbol":entry, "image_sha256":hashlib.sha256(image).hexdigest(),
            "ir_sha256":{k:hashlib.sha256(v.encode()).hexdigest()
                         for k,v in (("graph",source),("schedule",schedule),("tile",tile),("target",target),("backend",backend))},
            "compiler_fingerprint":compiler,"toolchain_fingerprint":toolchain,
            "device_libraries":[x.to_dict() for x in libraries], "compile_state":state,
            "device_event_samples_ms":events, "device_event_median_ms":median(events),
            "allocating_end_to_end_samples_ms":walls,"allocating_end_to_end_median_ms":median(walls)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["gfx1151","gfx1201"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.samples < 3 or args.iterations < 1: raise ValueError("invalid timing counts")
    live = rt._rocm_live_arch()
    if live != args.architecture: raise RuntimeError(f"owning architecture mismatch: {live}")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0): raise RuntimeError("HIP unavailable")
    ordinal=C.c_int(); name=C.create_string_buffer(256); uuid=C.create_string_buffer(16)
    MathCase.check(hip.hipGetDevice(C.byref(ordinal)), "device ordinal")
    MathCase.check(hip.hipDeviceGetName(name,len(name),ordinal.value), "device name")
    hip.hipDeviceGetUuid.argtypes=[C.c_void_p,C.c_int]
    hip.hipDeviceGetUuid.restype=C.c_int
    MathCase.check(hip.hipDeviceGetUuid(uuid,ordinal.value), "device UUID")
    # Event ctypes signatures are established by the common HIP launch loader.
    rows=[record(hip, live, kind, shape, reverse, args.samples, args.iterations)
          for shape in ((3,17),(2,3,257),(256,1024))
          for kind in ("sqrt","exp","add","div","cumsum","cummax")
          for reverse in ((False,True) if kind in {"add","div"} else (False,))]
    root=Path(__file__).resolve().parents[2]
    sources=("src/compiler/programming_model/lib/NativeROCMMath.h",
             "src/compiler/programming_model/lib/PMPasses.cpp",
             "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp",
             "benchmarks/rocm/benchmark_native_math_schedule.py")
    packet={"compiler_binary_sha256":hashlib.sha256(find_tessera_opt().read_bytes()).hexdigest(),
            "source_sha256":{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
            "schema":"tessera.rocm.native-math-schedule.v1","architecture":live,
            "device":name.value.decode(),"device_ordinal":ordinal.value,"opaque_hip_uuid":uuid.raw.hex(),
            "rows":rows,"storage":"f32",
            "timing_domains":{"device":"HIP events around resident launches; allocation/upload/readback excluded",
                              "end_to_end":"module load/allocate/upload/launch/completion/readback/free/unload; compiler excluded"},
            "scope":"direct compiler-product proof; ordinary JIT/package ABI and f16/bf16 remain open"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")
    print(json.dumps([{"kind":r["kind"],"shape":r["shape"],"reversed":r["reversed_binary_roles"],
                      "device_ms":r["device_event_median_ms"],
                      "wall_ms":r["allocating_end_to_end_median_ms"]} for r in rows],indent=2))

if __name__ == "__main__": main()
