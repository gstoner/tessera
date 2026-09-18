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

#: The emitter's math admission table, mirrored from `kMathAdmission` in
#: `src/transforms/lib/RowProgramToGPUPass.cpp` (drift-gated by
#: `tests/unit/test_row_program_math_admission.py`). A `math.*` op outside it is
#: refused by the pass rather than passed through, because on both device routes
#: it reaches a vendor library whose default accuracy nobody here measured.
#:
#: ``rounding_explicit`` — realized as a correctly-rounded call, exact whatever
#: the vendor default is. ``bit_exact`` — sign/bit manipulation, exact by
#: construction. ``measured`` — the vendor default, admitted because
#: ``benchmarks/record_row_program_math_precision.py`` has measured it against
#: the host on the owning device.
ADMITTED_MATH: dict[str, str] = {
    "math.sqrt": "rounding_explicit",
    "math.absf": "bit_exact",
    "math.exp": "measured",
    "math.log": "measured",
    "math.cos": "measured",
}

#: Refused with a recorded reason rather than left unmentioned, because both were
#: admitted until the audit measured them on gfx1151 (2026-09-16): `math.tanh`
#: ships a kernel whose body is one `s_endpgm` (the launch writes nothing), and
#: `math.log1p` is computed as log(1 + x), losing exactly the accuracy near zero
#: that log1p exists for. See `kMathAdmission` in the pass for the detail.
REFUSED_MATH: dict[str, str] = {
    "math.tanh": "the packaged ROCm image contains no kernel body (__ocml_tanh_f32)",
    "math.log1p": "the device computes log(1 + x), so it is inaccurate near zero",
}

#: The host reference for each admitted op, and the input domain the audit
#: sweeps it over. The domain is per-op because the interesting disagreements
#: are domain-specific: argument reduction for `cos`, the decades-wide range
#: and the near-1 cancellation for `log`, the overflow shoulder for `exp`.
MATH_AUDIT_DOMAINS: dict[str, tuple[tuple[float, float], ...]] = {
    "math.sqrt": ((0.0, 1.0), (1.0, 1e6), (1e-30, 1e-20)),
    "math.absf": ((-1e6, 1e6),),
    "math.exp": ((-87.0, 88.0), (-1.0, 1.0), (-1e-6, 1e-6)),
    "math.log": ((1e-30, 1.0), (1.0, 1e30), (0.5, 2.0)),
    "math.cos": ((-3.14159265, 3.14159265), (-100.0, 100.0), (-1e6, 1e6)),
}


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


def sphere_langevin_step_module(rows: int, features: int) -> str:
    """One Riemannian Langevin step on S^{f-1} per row, the EBM sphere
    front door's device program (2026-09-18):

        gt = gs - <gs, x> x        (tangent projection of the scaled gradient)
        nt = ns - <ns, x> x        (tangent projection of the scaled noise)
        y  = x - gt + nt           (Euler-Maruyama)
        out = y / |y|, or x when |y| < 1e-12   (retract; the host's guard)

    ``gs = eta * grad`` and ``ns = sqrt(2 eta T) * noise`` are folded on the
    host, so the kernel carries no scalar operands: three feature-axis
    reductions (two dots and the norm), the affine step, and the
    rounding-explicit ``math.sqrt`` of the retraction, one lane per feature.
    Mirrors ``ebm.geo_sampling.sphere_langevin_step``'s numpy path in f32."""
    r, f = int(rows), int(features)
    return f"""#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0, 0)>
#vec = affine_map<(d0) -> (d0)>
module {{
  func.func @sphere_langevin_step(%x: tensor<{r}x{f}xf32>, %gs: tensor<{r}x{f}xf32>, %ns: tensor<{r}x{f}xf32>) -> tensor<{r}x{f}xf32> {{
    %zero = arith.constant 0.000000e+00 : f32
    %one = arith.constant 1.000000e+00 : f32
    %tiny = arith.constant 1.000000e-12 : f32
    %e = tensor.empty() : tensor<{r}x{f}xf32>
    %re = tensor.empty() : tensor<{r}xf32>
    %init = linalg.fill ins(%zero : f32) outs(%re : tensor<{r}xf32>) -> tensor<{r}xf32>
    %gx = linalg.generic {{indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}}
        ins(%gs, %x : tensor<{r}x{f}xf32>, tensor<{r}x{f}xf32>) outs(%e : tensor<{r}x{f}xf32>) {{
    ^bb0(%a: f32, %b: f32, %o: f32):
      %m = arith.mulf %a, %b : f32
      linalg.yield %m : f32
    }} -> tensor<{r}x{f}xf32>
    %dg = linalg.reduce ins(%gx : tensor<{r}x{f}xf32>) outs(%init : tensor<{r}xf32>) dimensions = [1]
      (%in: f32, %acc: f32) {{
        %add = arith.addf %in, %acc : f32
        linalg.yield %add : f32
      }}
    %nx = linalg.generic {{indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}}
        ins(%ns, %x : tensor<{r}x{f}xf32>, tensor<{r}x{f}xf32>) outs(%e : tensor<{r}x{f}xf32>) {{
    ^bb0(%a: f32, %b: f32, %o: f32):
      %m = arith.mulf %a, %b : f32
      linalg.yield %m : f32
    }} -> tensor<{r}x{f}xf32>
    %dn = linalg.reduce ins(%nx : tensor<{r}x{f}xf32>) outs(%init : tensor<{r}xf32>) dimensions = [1]
      (%in: f32, %acc: f32) {{
        %add = arith.addf %in, %acc : f32
        linalg.yield %add : f32
      }}
    %dgx = tensor.expand_shape %dg [[0, 1]] output_shape [{r}, 1] : tensor<{r}xf32> into tensor<{r}x1xf32>
    %dnx = tensor.expand_shape %dn [[0, 1]] output_shape [{r}, 1] : tensor<{r}xf32> into tensor<{r}x1xf32>
    %y = linalg.generic {{indexing_maps = [#map, #map, #map, #row, #row, #map], iterator_types = ["parallel", "parallel"]}}
        ins(%x, %gs, %ns, %dgx, %dnx : tensor<{r}x{f}xf32>, tensor<{r}x{f}xf32>, tensor<{r}x{f}xf32>, tensor<{r}x1xf32>, tensor<{r}x1xf32>) outs(%e : tensor<{r}x{f}xf32>) {{
    ^bb0(%xv: f32, %gv: f32, %nv: f32, %dgv: f32, %dnv: f32, %o: f32):
      %gp = arith.mulf %dgv, %xv : f32
      %gt = arith.subf %gv, %gp : f32
      %np = arith.mulf %dnv, %xv : f32
      %nt = arith.subf %nv, %np : f32
      %s1 = arith.subf %xv, %gt : f32
      %yv = arith.addf %s1, %nt : f32
      linalg.yield %yv : f32
    }} -> tensor<{r}x{f}xf32>
    %yy = linalg.generic {{indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}}
        ins(%y, %y : tensor<{r}x{f}xf32>, tensor<{r}x{f}xf32>) outs(%e : tensor<{r}x{f}xf32>) {{
    ^bb0(%a: f32, %b: f32, %o: f32):
      %m = arith.mulf %a, %b : f32
      linalg.yield %m : f32
    }} -> tensor<{r}x{f}xf32>
    %ss = linalg.reduce ins(%yy : tensor<{r}x{f}xf32>) outs(%init : tensor<{r}xf32>) dimensions = [1]
      (%in: f32, %acc: f32) {{
        %add = arith.addf %in, %acc : f32
        linalg.yield %add : f32
      }}
    %nrm = linalg.generic {{indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}}
        ins(%ss : tensor<{r}xf32>) outs(%re : tensor<{r}xf32>) {{
    ^bb0(%v: f32, %o: f32):
      %q = math.sqrt %v : f32
      linalg.yield %q : f32
    }} -> tensor<{r}xf32>
    %nrmx = tensor.expand_shape %nrm [[0, 1]] output_shape [{r}, 1] : tensor<{r}xf32> into tensor<{r}x1xf32>
    %out = linalg.generic {{indexing_maps = [#map, #map, #row, #map], iterator_types = ["parallel", "parallel"]}}
        ins(%y, %x, %nrmx : tensor<{r}x{f}xf32>, tensor<{r}x{f}xf32>, tensor<{r}x1xf32>) outs(%e : tensor<{r}x{f}xf32>) {{
    ^bb0(%yv: f32, %xv: f32, %d: f32, %o: f32):
      %degenerate = arith.cmpf olt, %d, %tiny : f32
      %safe = arith.select %degenerate, %one, %d : f32
      %q = arith.divf %yv, %safe : f32
      %r = arith.select %degenerate, %xv, %q : f32
      linalg.yield %r : f32
    }} -> tensor<{r}x{f}xf32>
    return %out : tensor<{r}x{f}xf32>
  }}
}}
"""


def row_unary_math_module(rows: int, features: int, op: str) -> str:
    """``y[r, f] = <op>(x[r, f])``: the smallest row program that isolates one
    ``math.*`` op, so the audit recorder measures that op and nothing else."""
    if op not in ADMITTED_MATH:
        raise ValueError(f"{op} is not in the emitter's math admission table")
    r, f = int(rows), int(features)
    return f"""#map = affine_map<(d0, d1) -> (d0, d1)>
module {{
  func.func @unary_math(%x: tensor<{r}x{f}xf32>) -> tensor<{r}x{f}xf32> {{
    %e = tensor.empty() : tensor<{r}x{f}xf32>
    %y = linalg.generic {{indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}}
        ins(%x : tensor<{r}x{f}xf32>) outs(%e : tensor<{r}x{f}xf32>) {{
    ^bb0(%a: f32, %o: f32):
      %v = {op} %a : f32
      linalg.yield %v : f32
    }} -> tensor<{r}x{f}xf32>
    return %y : tensor<{r}x{f}xf32>
  }}
}}
"""


def declared_math(kernel_text: str) -> tuple[str, ...]:
    """The `tessera.row_program.math` plan the emitted kernel declares."""
    found = re.search(r"tessera\.row_program\.math = \[([^\]]*)\]", kernel_text)
    if found is None:
        return ()
    return tuple(sorted(entry.strip().strip('"') for entry in found[1].split(",") if entry.strip()))
