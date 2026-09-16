"""Clifford products as native GPU storage packages (W6.4 GPU route, 2026-09-16).

The kernel *skeleton* -- one thread per multivector, loads, the op, stores --
is written here; the arithmetic is not. The body carries a rank-1
``tessera_clifford.<op>`` on ``tensor<dim x f32>`` values built from the
loaded coefficients, and ``ts-clifford-opt`` (the same GradeFusion +
ExpandProductTable lowering the CPU JIT runs) expands it into the
compile-time Cayley contraction; the arena pipeline's canonicalization folds
``tensor.extract(tensor.from_elements(...))`` back to scalars, leaving a
plain scalar kernel for the NVVM / ROCDL lowering. Replay validation is the
shared one (`replay_arena_ir` byte-compares the expanded source).

Envelope: f32, Cl(p, q, r) with p+q+r <= 4, static ``[..., dim]`` shapes; the
scalar forms (inner, norm) write ``[..., 1]``. No fallback anywhere: a missing
tool or a mismatched replay raises.
"""
from __future__ import annotations

import ctypes as ct
import inspect
import math
import os
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .native_gpu_storage import _run, build_native_gpu_storage, replay_arena_ir
from .native_gpu_tensor import IndexSpec, TensorSpec
from .native_storage_contract import attach_tensor_contract, generate_tensor_binding

#: op -> (arity, scalar result). Mirrors `_jit_boundary.CLIFFORD_JIT_OPS`.
CLIFFORD_GPU_OPS: dict[str, tuple[int, bool]] = {
    "geo_product": (2, False), "wedge": (2, False), "left_contract": (2, False),
    "inner": (2, True), "norm": (1, True), "rotor_sandwich": (2, False),
    "reverse": (1, False), "grade_involute": (1, False), "conjugate": (1, False),
    "hodge_star": (1, False), "grade": (1, False),
}

_BYTE_BOUND = 64 * 1024 * 1024


def find_ts_clifford_opt() -> Path | None:
    """The Clifford lowering driver (built with TESSERA_BUILD_CLIFFORD_BACKEND=ON)."""
    if configured := os.environ.get("TS_CLIFFORD_OPT"):
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    root = Path(__file__).resolve().parents[3]
    if selected := os.environ.get("TESSERA_BUILD_DIR"):
        build = Path(selected).expanduser()
        path = (build if build.is_absolute() else root / build) / "src/solvers/clifford/ts-clifford-opt"
        return path if path.is_file() else None
    path = root / "build/src/solvers/clifford/ts-clifford-opt"
    if path.is_file():
        return path
    found = shutil.which("ts-clifford-opt")
    return Path(found) if found else None


def _check(op: str, shape, algebra, grades):
    if op not in CLIFFORD_GPU_OPS:
        raise ValueError(f"clifford GPU lane has no lowering for {op!r}")
    p, q, r = (int(x) for x in algebra)
    if min(p, q, r) < 0 or p + q + r > 4:
        raise ValueError("clifford GPU lane admits Cl(p, q, r) with 0 <= p+q+r <= 4")
    dim = 1 << (p + q + r)
    if (type(shape) is not tuple or not shape or any(type(d) is not int or d <= 0 for d in shape)
            or shape[-1] != dim or math.prod(shape) * 4 > _BYTE_BOUND):
        raise ValueError(f"clifford GPU lane requires a positive [..., {dim}] shape within the native buffer bound")
    wanted = None
    if grades is not None:
        if op not in ("geo_product", "grade"):
            raise ValueError(f"grades applies to geo_product or grade, not {op}")
        wanted = tuple(sorted({int(g) for g in grades}))
        if not wanted or wanted[0] < 0 or wanted[-1] > p + q + r:
            raise ValueError("grades must be a non-empty subset of 0..n")
    elif op == "grade":
        raise ValueError("grade requires the grades to keep")
    return (p, q, r), dim, wanted


def clifford_gpu_skeleton(op: str, shape, *, algebra=(3, 0, 0), grades=None):
    """The unexpanded kernel: thread mapping + rank-1 Clifford op on tensors."""
    (p, q, r), dim, wanted = _check(op, shape, algebra, grades)
    arity, scalar = CLIFFORD_GPU_OPS[op]
    count = math.prod(shape[:-1])
    names = [f"a{i}" for i in range(arity)]
    out_shape = (*shape[:-1], 1) if scalar else shape
    specs = tuple(TensorSpec(n, "fp32", shape, False) for n in names) + (
        TensorSpec("out", "fp32", out_shape, True), IndexSpec("scratch", 1, 1))
    attrs = f'algebra = [{p}, {q}, {r}], dtype = "fp32"'
    if wanted is not None:
        key = "grades" if op == "grade" else "tessera.clifford.output_grades"
        attrs += f", {key} = [{', '.join(map(str, wanted))}]"
    args = ", ".join(f"%{n}: !llvm.ptr<1>" for n in names)
    loads = []
    for n in names:
        loads.append(f"    %{n}_base = arith.muli %i, %c{dim} : i64")
        for k in range(dim):
            loads.append(f"    %{n}_{k}i = arith.addi %{n}_base, %k{k} : i64")
            loads.append(f"    %{n}_{k}p = llvm.getelementptr %{n}[%{n}_{k}i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32")
            loads.append(f"    %{n}_{k} = llvm.load %{n}_{k}p : !llvm.ptr<1> -> f32")
        elems = ", ".join(f"%{n}_{k}" for k in range(dim))
        loads.append(f"    %{n}_mv = tensor.from_elements {elems} : tensor<{dim}xf32>")
    operands = ", ".join(f"%{n}_mv" for n in names)
    types = ", ".join([f"tensor<{dim}xf32>"] * arity)
    width = 1 if scalar else dim
    result_ty = f"tensor<{width}xf32>"
    stores = [f"    %out_base = arith.muli %i, %c{width} : i64"]
    for k in range(width):
        stores.append(f"    %o_{k} = tensor.extract %res[%x{k}] : {result_ty}")
        stores.append(f"    %o_{k}i = arith.addi %out_base, %k{k} : i64")
        stores.append(f"    %o_{k}p = llvm.getelementptr %out[%o_{k}i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32")
        stores.append(f"    llvm.store %o_{k}, %o_{k}p : f32, !llvm.ptr<1>")
    consts = "\n".join(f"   %k{k} = arith.constant {k} : i64\n   %x{k} = arith.constant {k} : index" for k in range(dim))
    body = "\n".join(loads + [
        f'    %res = "tessera_clifford.{op}"({operands}) {{{attrs}}} : ({types}) -> {result_ty}'] + stores)
    source = f"""module {{
 gpu.module @native_clifford {{
  gpu.func @clifford_{op}({args}, %out: !llvm.ptr<1>, %scratch: index) kernel attributes {{known_block_size = array<i32: 128, 1, 1>}} {{
   %marker = memref.alloca(%scratch) : memref<?xf32>
   "tile.alloc_shared"(%marker) : (memref<?xf32>) -> ()
   %block = gpu.block_id x
   %thread = gpu.thread_id x
   %bw = arith.constant 128 : index
   %limit = arith.constant {count} : index
   %base = arith.muli %block, %bw : index
   %index = arith.addi %base, %thread : index
   %inside = arith.cmpi ult, %index, %limit : index
   %c{dim} = arith.constant {dim} : i64
   %c1 = arith.constant 1 : i64
{consts}
   scf.if %inside {{
    %i = arith.index_cast %index : index to i64
{body}
   }}
   gpu.return
  }}
 }}
}}"""
    return attach_tensor_contract(source, specs, grid=((count + 127) // 128, 1, 1), block=(128, 1, 1)), specs


def expand_clifford_source(source: str, *, clifford_opt: Path) -> str:
    """Run the Clifford lowering (with rotor-sandwich expansion) on the skeleton."""
    expanded = _run(Path(clifford_opt), "--allow-unregistered-dialect", "--tessera-clifford-grade-fusion",
                    "--tessera-clifford-expand-product-table=expand-rotor-sandwich=true", source=source)
    if "tessera_clifford." in expanded:
        raise ValueError("clifford GPU lane: an op survived the native lowering")
    return expanded


def bind_clifford_gpu(op: str, shape, *, compiler, llvm_bin, backend, chip, algebra=(3, 0, 0), grades=None,
                      clifford_opt=None):
    """Package one Clifford op for `(backend, chip)` and return its native tensor call."""
    skeleton, specs = clifford_gpu_skeleton(op, shape, algebra=algebra, grades=grades)
    tool = Path(clifford_opt) if clifford_opt is not None else find_ts_clifford_opt()
    if tool is None or not tool.is_file():
        raise ValueError("clifford GPU lane requires ts-clifford-opt (TESSERA_BUILD_CLIFFORD_BACKEND=ON)")
    source = expand_clifford_source(skeleton, clifford_opt=tool)
    compiler = Path(compiler)
    package = build_native_gpu_storage(source, compiler=compiler, llvm_bin=Path(llvm_bin), backend=backend, chip=chip)
    if package.arena_ir != replay_arena_ir(compiler, source):
        raise ValueError("clifford GPU lane native replay disagrees")
    if "tensor." in package.arena_ir or "tessera_clifford." in package.arena_ir:
        raise ValueError("clifford GPU lane: tensor ops survived into the device kernel")
    return generate_tensor_binding(package, inspect.Signature([
        inspect.Parameter(s.name, inspect.Parameter.POSITIONAL_ONLY) for s in specs]))


# ---------------------------------------------------------------------------
# Execution from host arrays.
# ---------------------------------------------------------------------------

_CONTEXT_LOCK = threading.Lock()
_CONTEXTS: dict[str, bool] = {}


def ensure_device_context(backend: str) -> None:
    """Make device 0 current for this process once (the bound package launches
    on the caller's active context and never creates one)."""
    with _CONTEXT_LOCK:
        if _CONTEXTS.get(backend):
            return
        cuda = backend == "nvidia"
        driver = ct.CDLL("libcuda.so.1" if cuda else "libamdhip64.so")
        P = ct.c_void_p

        def call(name, types, *args):
            fn = getattr(driver, name)
            fn.argtypes, fn.restype = types, ct.c_int
            status = fn(*args)
            if status:
                raise RuntimeError(f"clifford GPU lane: {name} failed with status {status}")
        if cuda:
            call("cuInit", [ct.c_uint], 0)
            context = P()
            call("cuDevicePrimaryCtxRetain", [ct.POINTER(P), ct.c_int], ct.byref(context), 0)
            call("cuCtxSetCurrent", [P], context)
        else:
            call("hipInit", [ct.c_uint], 0)
            call("hipSetDevice", [ct.c_int], 0)
        _CONTEXTS[backend] = True


class CliffordDeviceProgram:
    """One packaged Clifford op, runnable from host arrays.

    Device buffers are allocated per call through the bound package's own
    driver handle and freed before returning; the output comes back as a new
    host array. Correctness evidence only -- per-call transfers are not a
    performance path.
    """
    def __init__(self, binding, op: str):
        self.binding, self.op = binding, op
        self.package = binding.package
        self.bound = binding.package.bind()
        binding._bound = self.bound
        cuda = self.package.backend == "nvidia"
        P, S = ct.c_void_p, ct.c_size_t

        def bind(cu, hip, types):
            fn = getattr(self.bound._driver, cu if cuda else hip)
            fn.argtypes, fn.restype = types, ct.c_int
            return fn
        self._alloc = bind("cuMemAlloc_v2", "hipMalloc", [ct.POINTER(P), S])
        self._free = bind("cuMemFree_v2", "hipFree", [P])
        self._to_device = bind("cuMemcpyHtoD_v2", "hipMemcpyHtoD", [P, P, S])
        self._to_host = bind("cuMemcpyDtoH_v2", "hipMemcpyDtoH", [P, P, S])
        self.specs = tuple(s for s in binding.specs if isinstance(s, TensorSpec))
        self._lock = threading.RLock()

    def run(self, *arrays) -> np.ndarray:
        inputs = [np.ascontiguousarray(np.asarray(a, dtype=np.float32)) for a in arrays]
        if len(inputs) != len(self.specs) - 1:
            raise ValueError(f"clifford {self.op} takes {len(self.specs) - 1} operand(s)")
        for spec, value in zip(self.specs[:-1], inputs, strict=True):
            if value.shape != tuple(spec.shape):
                raise ValueError(f"clifford GPU lane admits {spec.name} of shape {tuple(spec.shape)}")
        out_spec = self.specs[-1]
        out = np.empty(tuple(out_spec.shape), np.float32)
        pointers: list[ct.c_void_p] = []
        with self._lock:
            try:
                views = []
                for value in [*inputs, out]:
                    pointer = ct.c_void_p()
                    self.bound._check(self._alloc(ct.byref(pointer), max(value.nbytes, 4)))
                    pointers.append(pointer)
                    views.append(SimpleNamespace(__cuda_array_interface__=dict(
                        version=3, shape=value.shape, typestr="<f4", data=(pointer.value, False))))
                for value, pointer in zip(inputs, pointers[:-1], strict=True):
                    self.bound._check(self._to_device(pointer, value.ctypes.data, value.nbytes))
                self.binding(*views, 1)
                self.bound._check(self.bound._sync())
                self.bound._check(self._to_host(out.ctypes.data, pointers[-1], out.nbytes))
            finally:
                for pointer in reversed(pointers):
                    if pointer.value:
                        self._free(pointer)
        return out

    def close(self):
        self.bound.close()


_PROGRAMS: dict[tuple, CliffordDeviceProgram] = {}
_PROGRAM_LOCK = threading.Lock()


def clifford_gpu_program(op: str, shape, *, backend: str, chip: str, compiler, llvm_bin,
                         algebra=(3, 0, 0), grades=None, clifford_opt=None) -> CliffordDeviceProgram:
    """Compile (once per process) and return the device program for one op/shape."""
    key = (op, tuple(shape), backend, chip, tuple(int(x) for x in algebra),
           None if grades is None else tuple(sorted({int(g) for g in grades})), str(compiler), str(llvm_bin))
    with _PROGRAM_LOCK:
        program = _PROGRAMS.get(key)
        if program is None:
            ensure_device_context(backend)
            binding = bind_clifford_gpu(op, tuple(shape), compiler=compiler, llvm_bin=llvm_bin, backend=backend,
                                        chip=chip, algebra=algebra, grades=grades, clifford_opt=clifford_opt)
            program = _PROGRAMS[key] = CliffordDeviceProgram(binding, op)
        return program


def package_clifford_native(op: str, shape, *, target: str, algebra=(3, 0, 0), grades=None):
    """A runtime artifact for the native GPU route (rows ``rocm`` /
    ``rocm_clifford_native_compiled`` and ``nvidia_sm120`` /
    ``nvidia_clifford_native_compiled``). The kernel is compiled at launch on
    the owning host; a missing toolchain fails the launch, never falls back."""
    if target not in ("rocm", "nvidia_sm120"):
        raise ValueError("clifford native GPU route targets rocm or nvidia_sm120")
    (p, q, r), _dim, wanted = _check(op, tuple(int(d) for d in shape), algebra, grades)
    from ..runtime import RuntimeArtifact
    path = "rocm_clifford_native_compiled" if target == "rocm" else "nvidia_clifford_native_compiled"
    return RuntimeArtifact(metadata={
        "target": target, "compiler_path": path, "executable": True,
        "kernel_id": f"clifford_native_{op}_cl{p}{q}{r}_" + "x".join(map(str, shape)),
        "op": f"clifford_{op}", "clifford_op": op, "algebra": (p, q, r), "shape": tuple(int(d) for d in shape),
        "grades": wanted, "dtype": "f32",
    })

