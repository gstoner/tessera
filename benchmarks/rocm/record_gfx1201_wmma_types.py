"""Dense RDNA4 WMMA operand-pair and integer-signedness device proof."""

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np
import ml_dtypes
from benchmarks.rocm.benchmark_rocm_arch_fragments import Case, _source, _lower
from tessera.compiler.native_gpu_storage import _decode_image
from tessera.compiler.rocm_target import AMDArch, wmma_dtype_forms



def pair_source(a_type, b_type, k, signed_a=True, signed_b=True, arch="gfx1201", accum="f32"):
    source = _source(Case(arch, "rdna4_wmma", a_type))
    if a_type == "int4" and k == 16:
        source = source.replace("512xi8", "256xi8").replace("k = 32", "k = 16")
        source = source.replace("[16, 32] : [32, 1]", "[16, 16] : [16, 1]")
        source = source.replace("[32, 16] : [16, 1]", "[16, 16] : [16, 1]")
        source = source.replace("leading_dim = 32", "leading_dim = 16")
    if a_type != b_type:
        types = {"e4m3": "f8E4M3FN", "e5m2": "f8E5M2"}
        old, new = types[a_type], types[b_type]
        source = source.replace(f'%b_mem: memref<256x{old}>', f'%b_mem: memref<256x{new}>')
        begin, end = source.index('      %b_tile ='), source.index('      %a =')
        source = source[:begin] + source[begin:end].replace(old,new) + source[end:]
        lines = source.splitlines()
        lines = [line.replace(f'elem = "{a_type}"',f'elem = "{b_type}"') if line.startswith('!frag_b =') else line for line in lines]
        source = "\n".join(lines)+"\n"
        source = source.replace(f'b = "{a_type}"', f'b = "{b_type}"')
    if a_type.startswith("int"):
        source = source.replace('%d = tile.mma %a, %b, %c {',
            '%d = tile.mma %a, %b, %c {signed_a = '+str(signed_a).lower()+', signed_b = '+str(signed_b).lower()+',')
    if accum in ("f16", "bf16"):
        source = source.replace("f32", accum)
    return source


def record(output):
    hip = ct.CDLL("libamdhip64.so")
    P = ct.c_void_p

    def call(name, types, *args):
        fn = getattr(hip, name)
        fn.argtypes = types
        fn.restype = ct.c_int
        rc = fn(*args)
        if rc:
            raise RuntimeError(f"{name} returned {rc}")

    call("hipInit", [ct.c_uint], 0)
    llvm = Path(os.environ["TESSERA_LLVM_BIN"])
    compiler = Path(os.environ["TESSERA_OPT"])
    agents = subprocess.check_output(["rocminfo"], text=True)
    if not re.search(r"Name:\s+gfx1201\b", agents):
        raise RuntimeError("requires owning gfx1201 host")
    rows = []
    spelling = {"fp16":"f16", "bf16":"bf16", "fp32":"f32", "int32":"i32",
                "fp8_e4m3":"e4m3", "fp8_e5m2":"e5m2", "int8":"int8", "int4":"int4"}
    cases = []
    instructions = {}
    for form in wmma_dtype_forms(AMDArch.GFX_1201):
        instructions[(spelling[form.a], spelling[form.b], form.k, spelling[form.accum])] = form.instruction.lower()
        signs = [(a,b) for a in (True,False) for b in (True,False)] if form.accum == "int32" else [(True,True)]
        cases.extend((spelling[form.a],spelling[form.b],form.k,a,b,spelling[form.accum]) for a,b in signs)
    for dtype, b_dtype, k, signed_a, signed_b, accum in cases:
        formats = {"f16": np.float16, "bf16": ml_dtypes.bfloat16,
                   "e4m3": ml_dtypes.float8_e4m3fn, "e5m2": ml_dtypes.float8_e5m2}
        storage = formats.get(dtype, np.int8 if signed_a else np.uint8)
        b_storage = formats.get(b_dtype, np.int8 if signed_b else np.uint8)
        case = Case("gfx1201", "rdna4_wmma", dtype)
        source = pair_source(dtype,b_dtype,k,signed_a,signed_b,accum=accum)
        lowered, _ = _lower(case, source)
        toolkit = Path(os.environ["ROCM_PATH"]).resolve()
        pipeline = "builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,reconcile-unrealized-casts),rocdl-attach-target{chip=gfx1201},gpu-module-to-binary{toolkit={toolkit}})"
        binary = subprocess.check_output(
            [str(llvm / "mlir-opt"), "--pass-pipeline=" + pipeline.replace("{toolkit}", str(toolkit))],
            input=lowered,
            text=True,
        )
        encoded = re.search(r'bin = "((?:\\.|[^"\\])*)"', binary)
        literals = re.findall(r'"((?:\\.|[^"\\])*)"', binary)
        image = _decode_image(encoded[1] if encoded else literals[-1])
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "wmma.hsaco"
            image_path.write_bytes(image)
            assembly = subprocess.check_output(
                [str(llvm / "llvm-objdump"), "--disassemble", str(image_path)], text=True)
        expected_instruction = instructions[(dtype, b_dtype, k, accum)]
        # LLVM adds an encoding suffix to some mnemonics. Require the full
        # datatype signature, never merely the presence of any WMMA opcode.
        matches = re.findall(r"^\s+(v_wmma_[a-z0-9_]+)\b", assembly, re.M)
        if not any(m == expected_instruction or m.startswith(expected_instruction + "_") for m in matches):
            raise RuntimeError(f"missing emitted {expected_instruction} instruction")
        module, fn = P(), P()
        blob = ct.create_string_buffer(image)
        call("hipModuleLoadData", [ct.POINTER(P), P], ct.byref(module), ct.cast(blob, P))
        call(
            "hipModuleGetFunction",
            [ct.POINTER(P), P, ct.c_char_p],
            ct.byref(fn),
            module,
            b"architecture_fragment_store",
        )
        probes = [(16,16,k,"dyadic"), (13,11,k-3,"dyadic"), (1,7,5,"dyadic")]
        if not dtype.startswith("int"):
            probes += [(16,16,k,"finite_range"), (16,16,k,"nonfinite")]
        for m, n, active_k, probe in probes:
            rng = np.random.default_rng(17)
            a = np.zeros((16, k), dtype=storage)
            b = np.zeros((k, 16), dtype=b_storage)
            bits = 4 if dtype == "int4" else 8
            arange = (-(1 << (bits-1)), 1 << (bits-1)) if signed_a else (0, 1 << bits)
            brange = (-(1 << (bits-1)), 1 << (bits-1)) if signed_b else (0, 1 << bits)
            if not dtype.startswith("int"):
                # Dyadic values exercise different FP8 encodings with an exact oracle.
                arange = brange = (-12, 13)
            a[:m, :active_k] = rng.integers(*arange, (m, active_k))
            b[:active_k, :n] = rng.integers(*brange, (active_k, n))
            if not dtype.startswith("int"):
                a = (a.astype(np.float32)*0.25).astype(storage)
                b = (b.astype(np.float32)*0.25).astype(b_storage)
            if probe == "finite_range":
                limits = {"f16": (65504.0,2.0**-24,2.0**-14),
                          "bf16": (float.fromhex("0x1.fep127"),2.0**-133,2.0**-126),
                          "e4m3": (448.0,2.0**-9,2.0**-6),
                          "e5m2": (57344.0,2.0**-16,2.0**-14)}
                maximum, tiny, normal = limits[dtype]
                a.fill(0); b.fill(0)
                values = [-maximum,-normal,-tiny,-0.0,0.0,tiny,normal,maximum,1,-1,0.5,-0.5,2,-2,4,-4]
                for i,value in enumerate(values):
                    a[i,i] = value
                    b[i,i] = 0.5
            if probe == "nonfinite":
                a.fill(0); b.fill(0)
                a[0,0] = float("nan")
                if dtype != "e4m3":
                    a[1,1], a[2,2] = float("inf"), -float("inf")
                for i in range(16): b[i,i] = 0.5
            result = np.zeros((16, 16), dtype=np.int32 if dtype.startswith("int") else formats.get(accum, np.float32))
            arrays = [np.ascontiguousarray(a), np.ascontiguousarray(b.T), result]
            pointers = []
            values = []
            for array in arrays:
                ptr = P()
                call("hipMalloc", [ct.POINTER(P), ct.c_size_t], ct.byref(ptr), array.nbytes)
                pointers.append(ptr)
                call("hipMemcpy", [P, P, ct.c_size_t, ct.c_int], ptr, P(array.ctypes.data), array.nbytes, 1)
                values.extend([P(ptr.value), P(ptr.value), ct.c_int64(0), ct.c_int64(array.size), ct.c_int64(1)])
            arguments = (P * len(values))(*(ct.cast(ct.pointer(v), P) for v in values))
            call(
                "hipModuleLaunchKernel",
                [P] + [ct.c_uint] * 7 + [P, ct.POINTER(P), ct.POINTER(P)],
                fn,
                1,
                1,
                1,
                32,
                1,
                1,
                0,
                None,
                arguments,
                None,
            )
            call("hipDeviceSynchronize", [])
            call("hipMemcpy", [P, P, ct.c_size_t, ct.c_int], P(result.ctypes.data), pointers[2], result.nbytes, 2)
            for ptr in pointers:
                call("hipFree", [P], ptr)
            with np.errstate(invalid="ignore", over="ignore"):
                exact = a.astype(np.float64) @ b.astype(np.float64)
                expected = exact.astype(result.dtype).astype(np.float64)
            actual = result.astype(np.float64)
            finite = np.isfinite(expected)
            classification = (np.array_equal(np.isnan(actual), np.isnan(expected)) and
                              np.array_equal(np.isposinf(actual), np.isposinf(expected)) and
                              np.array_equal(np.isneginf(actual), np.isneginf(expected)))
            error = float(np.max(np.abs(actual[finite]-expected[finite]), initial=0))
            max_budget = 0.0
            if accum in ("f16", "bf16") and probe == "dyadic":
                # gamma_k for native fused accumulation roundings; no wider-
                # accumulator substitution and no change to f32/integer checks.
                u = 2.0**(-11 if accum == "f16" else -8)
                budget = (k*u/(1-k*u)) * (np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64)))
                max_budget = float(budget.max())
                passed = bool(classification and np.all(np.abs(actual-exact) <= budget))
                reference_model = "native_accumulator_gamma_k"
            else:
                passed = classification and error == 0
                reference_model = "exact_result_and_nonfinite_classification"
            rows.append(
                dict(
                    dtype=dtype, b_dtype=b_dtype, accum=accum, k=k, probe=probe, signed_a=signed_a, signed_b=signed_b,
                    shape=[m, n, active_k],
                    max_abs_error=error,
                    passed=bool(passed), reference_model=reference_model, max_error_bound=max_budget, classification_matches=bool(classification),
                    image_sha256=hashlib.sha256(image).hexdigest(),
                    expected_instruction=expected_instruction,
                    emitted_wmma_instructions=sorted(set(matches)),
                    tile_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                )
            )
        call("hipModuleUnload", [P], module)
    packet = dict(
        chip="gfx1201", promotion_eligible=False,
        board="RX 9070 XT",
        compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
        rows=rows,
        scope="one native WMMA tile, externally zero-padded ragged inputs; no arbitrary-size GEMM or performance promotion",
    )
    output.write_text(json.dumps(packet, indent=2) + "\n")
    if not all(r["passed"] for r in rows):
        raise RuntimeError("matrix numerical comparison failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    record(p.parse_args().output)
