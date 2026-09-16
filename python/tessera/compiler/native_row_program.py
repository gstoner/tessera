"""A ``[rows, features]`` tensor row program as one cooperative device kernel.

The compiler's ``tessera-row-program-to-gpu`` pass turns a function over
``[rows, features]`` tensors (parallel ``linalg.generic`` bodies, feature-axis
``linalg.reduce``, small uniform integer vectors, ``scf.for``) into one
``gpu.func``: one block per row, one lane per feature, loop-carried state in
registers, ordered shared-memory reductions. This module runs that pass through
one ``tessera-opt`` invocation, attaches the tensor contract, packages the
kernel with ``build_native_gpu_storage`` and returns a host-array runner. It is
the seam every row-shaped domain loop (EBM Langevin first) packages through;
no Python-emitted arithmetic.
"""
from __future__ import annotations

import inspect
import re
import threading
from pathlib import Path
from typing import Sequence

from .native_gpu_storage import _run, build_native_gpu_storage, replay_arena_ir
from .native_gpu_tensor import IndexSpec, TensorSpec
from .native_host_program import HostArrayProgram, ensure_device_context
from .native_storage_contract import attach_tensor_contract, generate_tensor_binding

ROW_PROGRAM_PASS = "--tessera-row-program-to-gpu"
MAX_FEATURES = 1024


def row_program_kernel(module_text: str, *, entry: str, backend: str, compiler, passes: Sequence[str] = ()) -> tuple[str, int]:
    """Run ``passes`` then the row-program emitter; return (kernel text, lanes)."""
    if backend not in ("nvidia", "rocm"):
        raise ValueError("row program device route targets nvidia or rocm")
    kernel = _run(Path(compiler), *passes, f"{ROW_PROGRAM_PASS}=backend={backend} entry={entry}", source=module_text)
    block = re.search(r"known_block_size = array<i32: (\d+), 1, 1>", kernel)
    if block is None or kernel.count("gpu.func ") != 1 or "gpu.func @row_program(" not in kernel:
        raise ValueError("row program device route: the compiler did not produce the row-program kernel")
    return kernel, int(block[1])


def row_program_device_source(module_text: str, *, entry: str, specs: Sequence[TensorSpec], rows: int,
                              backend: str, compiler, passes: Sequence[str] = ()) -> tuple[str, tuple]:
    """The packaged kernel source with its tensor contract (grid = rows, block = lanes)."""
    kernel, lanes = row_program_kernel(module_text, entry=entry, backend=backend, compiler=compiler, passes=passes)
    full = tuple(specs) + (IndexSpec("scratch", 1, 1),)
    return attach_tensor_contract(kernel, full, grid=(int(rows), 1, 1), block=(lanes, 1, 1)), full


def bind_row_program(source: str, specs, *, compiler, llvm_bin, backend: str, chip: str):
    """Package for (backend, chip); replay must agree; return the native tensor call."""
    compiler = Path(compiler)
    package = build_native_gpu_storage(source, compiler=compiler, llvm_bin=Path(llvm_bin), backend=backend, chip=chip)
    if package.arena_ir != replay_arena_ir(compiler, source):
        raise ValueError("row program device route: native replay disagrees")
    return generate_tensor_binding(package, inspect.Signature([
        inspect.Parameter(s.name, inspect.Parameter.POSITIONAL_ONLY) for s in specs]))


_PROGRAMS: dict = {}
_LOCK = threading.Lock()


def row_program_device(module_text: str, *, entry: str, specs: Sequence[TensorSpec], rows: int, backend: str,
                       chip: str, compiler, llvm_bin, passes: Sequence[str] = (), name: str = "row program") -> HostArrayProgram:
    """Compile once per process and return the host-array runner for the program."""
    key = (module_text, entry, tuple(passes), backend, chip, str(compiler), str(llvm_bin))
    with _LOCK:
        program = _PROGRAMS.get(key)
        if program is None:
            ensure_device_context(backend)
            source, full = row_program_device_source(module_text, entry=entry, specs=specs, rows=rows,
                                                     backend=backend, compiler=compiler, passes=passes)
            program = _PROGRAMS[key] = HostArrayProgram(
                bind_row_program(source, full, compiler=compiler, llvm_bin=llvm_bin, backend=backend, chip=chip), name)
        return program


def row_normalize_module(rows: int, features: int) -> str:
    """``y = x / sqrt(sum_f x^2)`` per row: the smallest program that exercises the
    ordered shared-memory reduction (a feature-axis ``linalg.reduce`` whose
    result is broadcast back over the lanes)."""
    r, f = int(rows), int(features)
    return f"""#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0, 0)>
module {{
  func.func @row_normalize(%x: tensor<{r}x{f}xf32>) -> tensor<{r}x{f}xf32> {{
    %zero = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<{r}x{f}xf32>
    %sq = linalg.generic {{indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}}
        ins(%x, %x : tensor<{r}x{f}xf32>, tensor<{r}x{f}xf32>) outs(%e : tensor<{r}x{f}xf32>) {{
    ^bb0(%a: f32, %b: f32, %o: f32):
      %m = arith.mulf %a, %b : f32
      linalg.yield %m : f32
    }} -> tensor<{r}x{f}xf32>
    %re = tensor.empty() : tensor<{r}xf32>
    %init = linalg.fill ins(%zero : f32) outs(%re : tensor<{r}xf32>) -> tensor<{r}xf32>
    %s = linalg.reduce ins(%sq : tensor<{r}x{f}xf32>) outs(%init : tensor<{r}xf32>) dimensions = [1]
      (%in: f32, %acc: f32) {{
        %add = arith.addf %in, %acc : f32
        linalg.yield %add : f32
      }}
    %n = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}}
        ins(%s : tensor<{r}xf32>) outs(%re : tensor<{r}xf32>) {{
    ^bb0(%v: f32, %o: f32):
      %sq2 = math.sqrt %v : f32
      linalg.yield %sq2 : f32
    }} -> tensor<{r}xf32>
    %nx = tensor.expand_shape %n [[0, 1]] output_shape [{r}, 1] : tensor<{r}xf32> into tensor<{r}x1xf32>
    %y = linalg.generic {{indexing_maps = [#map, #row, #map], iterator_types = ["parallel", "parallel"]}}
        ins(%x, %nx : tensor<{r}x{f}xf32>, tensor<{r}x1xf32>) outs(%e : tensor<{r}x{f}xf32>) {{
    ^bb0(%a: f32, %d: f32, %o: f32):
      %q = arith.divf %a, %d : f32
      linalg.yield %q : f32
    }} -> tensor<{r}x{f}xf32>
    return %y : tensor<{r}x{f}xf32>
  }}
}}
"""
