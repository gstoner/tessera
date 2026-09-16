"""The traceable quadratic energy loop through the MLIR/LLVM backbone
(W4-PRODUCT-1 / AD-SOLVER-IFT-1 acceptance, 2026-09-16).

The energy is a Graph IR function, ``E(y, x) = 0.5 * sum((x - y)^2, axis=-1)``,
marked ``tessera.autodiff = "reverse"``. Inside ``libtessera_jit`` the paired
autodiff pass derives ``@E__bwd`` (the compiler's gradient, not a hand
formula), the EBM lowering turns each ``tessera_ebm.langevin_step`` into
``y - eta * grad + sqrt(2 eta T) * z`` with ``z`` drawn on the device side of
the lane from Philox-4x32-10 / Box-Muller inside a ``linalg.generic``, and the
whole ``scf.for`` of K steps compiles to one function: no per-step host
gradient or noise transfers. ``reference_langevin_loop`` is the declared RNG
policy mirrored bit-for-bit in numpy (the standalone ``philox_4x32_10``).

Envelope: f32, static ``[rows, features]`` shapes, euclidean manifold, key as
two int64 words. No fallback: a library without the EBM lane raises.
"""
from __future__ import annotations

import math
import threading
from typing import Sequence

import numpy as np

from tessera import _jit_boundary as jb

_BYTE_BOUND = 64 * 1024 * 1024


def has_native_langevin() -> bool:
    """Whether libtessera_jit was built with the EBM lane (TESSERA_BUILD_EBM_BACKEND)."""
    if not jb.is_available():
        return False
    lib = jb._load()
    sym = getattr(lib, "tessera_jit_has_ebm", None)
    if sym is None:
        return False
    import ctypes
    sym.restype, sym.argtypes = ctypes.c_int, []
    return bool(sym())


def _check(shape, eta, temperature, steps):
    shape = tuple(int(d) for d in shape)
    if len(shape) != 2 or any(d <= 0 for d in shape) or math.prod(shape) * 4 > _BYTE_BOUND:
        raise ValueError("native langevin admits a positive static [rows, features] f32 state within the buffer bound")
    if not (isinstance(eta, (int, float)) and math.isfinite(eta) and eta > 0):
        raise ValueError("native langevin requires eta > 0")
    if not (isinstance(temperature, (int, float)) and math.isfinite(temperature) and temperature >= 0):
        raise ValueError("native langevin requires temperature >= 0")
    if type(steps) is not int or steps < 1 or steps > 1 << 20:
        raise ValueError("native langevin requires 1 <= steps <= 2**20")
    return shape


ENERGIES = ("quadratic", "huber", "softplus")
MANIFOLDS = ("euclidean", "sphere")
HUBER_DELTA = 1.0


def _check_kinds(energy: str, manifold: str):
    if energy not in ENERGIES:
        raise ValueError(f"native langevin energies are {ENERGIES}; got {energy!r}")
    if manifold not in MANIFOLDS:
        raise ValueError(f"native langevin manifolds are {MANIFOLDS} (bivector fails closed); got {manifold!r}")


def energy_function_text(energy: str, rows: int, features: int) -> str:
    """The Graph IR energy E(y, x) -> per-row energies, marked for reverse-mode.

    quadratic: 0.5 * sum (x - y)^2      (gradient y - x)
    huber:     sum huber_delta(y - x)   (gradient clip(y - x, -delta, delta))
    softplus:  sum softplus(y - x)      (gradient sigmoid(y - x))
    Each exercises one more adjoint on the Graph IR path (N1); the paired pass
    must produce @E__bwd with no custom_adjoint_call or the lowering refuses.
    """
    st = f"tensor<{rows}x{features}xf32>"
    en = f"tensor<{rows}xf32>"
    head = (f"  func.func @energy(%y: {st}, %x: {st}) -> {en}\n"
            f'      attributes {{tessera.autodiff = "reverse"}} {{\n')
    if energy == "quadratic":
        body = (f'    %d = "tessera.sub"(%x, %y) : ({st}, {st}) -> {st}\n'
                f'    %sq = "tessera.mul"(%d, %d) : ({st}, {st}) -> {st}\n'
                f'    %s = "tessera.reduce"(%sq) {{axis = 1 : i64, kind = "sum"}} : ({st}) -> {en}\n'
                f"    %half = arith.constant dense<5.000000e-01> : {en}\n"
                f'    %e = "tessera.mul"(%s, %half) : ({en}, {en}) -> {en}\n'
                f"    return %e : {en}\n")
    elif energy == "huber":
        body = (f'    %h = "tessera.loss.huber"(%y, %x) {{delta = {float(HUBER_DELTA)!r} : f64, reduction = "none"}} : ({st}, {st}) -> {st}\n'
                f'    %e = "tessera.reduce"(%h) {{axis = 1 : i64, kind = "sum"}} : ({st}) -> {en}\n'
                f"    return %e : {en}\n")
    else:
        body = (f'    %d = "tessera.sub"(%y, %x) : ({st}, {st}) -> {st}\n'
                f'    %p = "tessera.softplus"(%d) : ({st}) -> {st}\n'
                f'    %e = "tessera.reduce"(%p) {{axis = 1 : i64, kind = "sum"}} : ({st}) -> {en}\n'
                f"    return %e : {en}\n")
    return head + body + "  }\n"


def langevin_loop_module(shape, *, eta: float, temperature: float, steps: int,
                         manifold: str = "euclidean", energy: str = "quadratic") -> str:
    """The Graph-level program: an energy + a K-step Langevin loop.

    "sphere" carries the integrator's per-row i32 status word through the loop
    (OR of every step: bit 0 entry precondition violated, bit 1 retraction
    underflow) and returns it as a third result.
    """
    rows, features = _check(shape, eta, temperature, steps)
    _check_kinds(energy, manifold)
    st = f"tensor<{rows}x{features}xf32>"
    en = f"tensor<{rows}xf32>"
    stt = f"tensor<{rows}xi32>"
    attrs = (f"energy_fn = @energy, eta = {float(eta)!r} : f64, temperature = {float(temperature)!r} : f64, "
             f'manifold = "{manifold}"')
    common = ("    %c0 = arith.constant 0 : index\n"
              "    %c1 = arith.constant 1 : index\n"
              f"    %steps = arith.constant {steps} : index\n")
    if manifold == "sphere":
        loop = (f"  func.func @tessera_jit_ebm_langevin_loop(%y0: {st}, %x: {st}, %key0: tensor<2xi64>) -> ({st}, tensor<2xi64>, {stt}) {{\n"
                + common +
                f"    %ok = arith.constant dense<0> : {stt}\n"
                f"    %r:3 = scf.for %t = %c0 to %steps step %c1 iter_args(%y = %y0, %key = %key0, %status = %ok) -> ({st}, tensor<2xi64>, {stt}) {{\n"
                f'      %n:3 = "tessera_ebm.langevin_step"(%y, %key, %x) {{ {attrs} }}\n'
                f"          : ({st}, tensor<2xi64>, {st}) -> ({st}, tensor<2xi64>, {stt})\n"
                f"      %acc = arith.ori %status, %n#2 : {stt}\n"
                f"      scf.yield %n#0, %n#1, %acc : {st}, tensor<2xi64>, {stt}\n"
                "    }\n"
                f"    return %r#0, %r#1, %r#2 : {st}, tensor<2xi64>, {stt}\n"
                "  }\n")
    else:
        loop = (f"  func.func @tessera_jit_ebm_langevin_loop(%y0: {st}, %x: {st}, %key0: tensor<2xi64>) -> ({st}, tensor<2xi64>) {{\n"
                + common +
                f"    %r:2 = scf.for %t = %c0 to %steps step %c1 iter_args(%y = %y0, %key = %key0) -> ({st}, tensor<2xi64>) {{\n"
                f'      %n:2 = "tessera_ebm.langevin_step"(%y, %key, %x) {{ {attrs} }}\n'
                f"          : ({st}, tensor<2xi64>, {st}) -> ({st}, tensor<2xi64>)\n"
                f"      scf.yield %n#0, %n#1 : {st}, tensor<2xi64>\n"
                "    }\n"
                f"    return %r#0, %r#1 : {st}, tensor<2xi64>\n"
                "  }\n")
    energy_call = (f"  func.func @tessera_jit_ebm_energy(%y: {st}, %x: {st}) -> {en} {{\n"
                   f'    %e = "tessera_ebm.energy"(%x, %y) {{ energy_fn = @energy }} : ({st}, {st}) -> {en}\n'
                   f"    return %e : {en}\n"
                   "  }\n")
    return "module {\n" + energy_function_text(energy, rows, features) + energy_call + loop + "}\n"


def _require():
    if not has_native_langevin():
        raise jb.TesseraJitError("libtessera_jit was built without the EBM lane "
                                 "(configure with -DTESSERA_BUILD_EBM_BACKEND=ON)")


def native_quadratic_energy(y, x, *, energy: str = "quadratic") -> np.ndarray:
    """Per-row energy through the lane (the energy op lowers to a call)."""
    _require()
    y = np.ascontiguousarray(y, dtype=np.float32); x = np.ascontiguousarray(x, dtype=np.float32)
    if y.shape != x.shape:
        raise jb.TesseraJitError("energy requires equal state and context shapes")
    shape = _check(y.shape, 1.0, 0.0, 1)
    handle = jb.compile_module(langevin_loop_module(shape, eta=1.0, temperature=0.0, steps=1, energy=energy))
    try:
        out = np.empty((shape[0],), np.float32)
        jb.invoke(handle, "tessera_jit_ebm_energy", [y, x], out)
        return out
    finally:
        jb.destroy(handle)


def native_langevin_loop(y0, x, key: Sequence[int], *, eta: float, temperature: float, steps: int,
                         manifold: str = "euclidean", energy: str = "quadratic"):
    """K Langevin steps as one compiled function.

    Returns ``(y_K, next_key)``, or ``(y_K, next_key, status)`` on the sphere
    (per-row i32: 0 ok; 1 entry |x| != 1; 2 retraction underflow, state kept).
    ``key`` is two int64 words (S4 RNGKey words).
    """
    _require()
    y0 = np.ascontiguousarray(y0, dtype=np.float32); x = np.ascontiguousarray(x, dtype=np.float32)
    if y0.shape != x.shape:
        raise jb.TesseraJitError("langevin requires equal state and context shapes")
    shape = _check(y0.shape, eta, temperature, steps)
    _check_kinds(energy, manifold)
    key_arr = np.ascontiguousarray(np.asarray(key, dtype=np.int64).reshape(2))
    handle = jb.compile_module(langevin_loop_module(shape, eta=eta, temperature=temperature, steps=steps,
                                                    manifold=manifold, energy=energy))
    try:
        out = np.empty(shape, np.float32)
        next_key = np.empty((2,), np.int64)
        outputs = [out, next_key]
        if manifold == "sphere":
            outputs.append(np.empty((shape[0],), np.int32))
        jb.invoke(handle, "tessera_jit_ebm_langevin_loop", [y0, x, key_arr], outputs)
        return tuple(outputs)
    finally:
        jb.destroy(handle)


def _philox_normals(shape, key0: int, key1: int) -> np.ndarray:
    """Standard normals of the declared policy: Philox key = (lo32, hi32) of
    key0; counter = (flat index, lo32, hi32 of key1, 0); z = sqrt(-2 ln u0)
    cos(2 pi u1) with u = (word + 0.5) * 2^-32 on words 0 and 1, computed in f64
    and rounded to f32 once."""
    from tessera.compiler.philox import philox_4x32_10
    philox_key = np.array([key0 & 0xFFFFFFFF, (key0 >> 32) & 0xFFFFFFFF], np.uint32)
    s0, s1 = np.uint32(key1 & 0xFFFFFFFF), np.uint32((key1 >> 32) & 0xFFFFFFFF)
    z = np.empty(math.prod(shape), np.float32)
    for i in range(z.size):
        words = philox_4x32_10(np.array([np.uint32(i), s0, s1, np.uint32(0)], np.uint32), philox_key)
        u0 = (float(words[0]) + 0.5) * 2.0 ** -32
        u1 = (float(words[1]) + 0.5) * 2.0 ** -32
        z[i] = np.float32(math.sqrt(-2.0 * math.log(u0)) * math.cos(2.0 * math.pi * u1))
    return z.reshape(shape)


def reference_energy(y, x, *, energy: str = "quadratic") -> np.ndarray:
    """Per-row energies of the admitted energies in numpy (f32)."""
    y = np.asarray(y, np.float32); x = np.asarray(x, np.float32); d = (y - x).astype(np.float32)
    if energy == "quadratic":
        return (0.5 * np.sum(d * d, axis=1)).astype(np.float32)
    if energy == "huber":
        a = np.abs(d); delta = np.float32(HUBER_DELTA)
        return np.sum(np.where(a <= delta, 0.5 * d * d, delta * (a - 0.5 * delta)), axis=1).astype(np.float32)
    if energy == "softplus":
        return np.sum(np.maximum(d, 0) + np.log1p(np.exp(-np.abs(d))), axis=1).astype(np.float32)
    raise ValueError(energy)


def reference_gradient(y, x, *, energy: str = "quadratic") -> np.ndarray:
    """dE/dy of the admitted energies (the compiler derives these natively)."""
    d = (np.asarray(y, np.float32) - np.asarray(x, np.float32)).astype(np.float32)
    if energy == "quadratic":
        return d
    if energy == "huber":
        return np.clip(d, -HUBER_DELTA, HUBER_DELTA).astype(np.float32)
    if energy == "softplus":
        return (1.0 / (1.0 + np.exp(-d.astype(np.float64)))).astype(np.float32)
    raise ValueError(energy)


def _row_dot_sequential(a, b):
    """Per-row f32 dot product accumulated in feature order — the declared
    reduction order of the sphere integrator's projections and norms."""
    out = np.empty(a.shape[0], np.float32)
    for r in range(a.shape[0]):
        acc = np.float32(0.0)
        for v, w in zip(a[r], b[r]):
            acc = np.float32(np.float32(v * w) + acc)
        out[r] = acc
    return out


SPHERE_ENTRY_TOL = 2.0e-3    # on |x|^2 (the reference's | |x| - 1 | <= 1e-3)
SPHERE_UNDERFLOW = 1.0e-12   # on |y|^2


def reference_langevin_loop(y0, x, key: Sequence[int], *, eta: float, temperature: float, steps: int,
                            manifold: str = "euclidean", energy: str = "quadratic"):
    """The declared policy in numpy, bit-for-bit for the noise and the
    euclidean step; next key = (key[0], key[1] + 1) per step.

    Sphere (geo_sampling.sphere_langevin_step per row, with the same noise):
    g_t = g - <g, x> x ; xi_t = xi - <xi, x> x ; y = x - eta g_t + s xi_t ;
    x' = y / |y|; the dot products and norms are sequential f32 row sums. A
    row whose |x|^2 leaves [1 - 2e-3, 1 + 2e-3] on entry sets status bit 0; a
    row whose |y|^2 < 1e-12 keeps x and sets bit 1. Returns a third value, the
    OR of the per-row status over the steps, on the sphere.
    """
    y = np.ascontiguousarray(y0, dtype=np.float32).copy(); x = np.ascontiguousarray(x, dtype=np.float32)
    shape = _check(y.shape, eta, temperature, steps)
    _check_kinds(energy, manifold)
    k = [int(v) for v in np.asarray(key, dtype=np.int64).reshape(2)]
    scale = math.sqrt(2.0 * float(eta) * float(temperature))
    status = np.zeros(shape[0], np.int32)
    for _ in range(steps):
        grad = reference_gradient(y, x, energy=energy)
        z = _philox_normals(shape, k[0], k[1]) if scale > 0.0 else None
        if manifold == "euclidean":
            y = (y - np.float32(eta) * grad).astype(np.float32)
            if z is not None:
                y = (y + np.float32(scale) * z).astype(np.float32)
        else:
            n0 = _row_dot_sequential(y, y)
            status |= np.where(np.abs(n0 - np.float32(1.0)) > np.float32(SPHERE_ENTRY_TOL), 1, 0).astype(np.int32)
            gt = (grad - _row_dot_sequential(grad, y)[:, None] * y).astype(np.float32)
            step = (y - np.float32(eta) * gt).astype(np.float32)
            if z is not None:
                zt = (z - _row_dot_sequential(z, y)[:, None] * y).astype(np.float32)
                step = (step + np.float32(scale) * zt).astype(np.float32)
            n2 = _row_dot_sequential(step, step)
            under = n2 < np.float32(SPHERE_UNDERFLOW)
            status |= np.where(under, 2, 0).astype(np.int32)
            norm = np.sqrt(np.where(under, np.float32(1.0), n2)).astype(np.float32)
            y = np.where(under[:, None], y, (step / norm[:, None]).astype(np.float32)).astype(np.float32)
        k[1] += 1
    if manifold == "sphere":
        return y, np.array(k, np.int64), status
    return y, np.array(k, np.int64)


def package_ebm_langevin_cpu(shape, *, eta: float, temperature: float, steps: int,
                             manifold: str = "euclidean", energy: str = "quadratic"):
    """A runtime artifact for the loop (row ``cpu`` / ``cpu_ebm_langevin_llvm_jit``)."""
    shape = _check(shape, eta, temperature, steps)
    _check_kinds(energy, manifold)
    _require()
    from tessera.runtime import RuntimeArtifact
    return RuntimeArtifact(metadata={
        "target": "cpu", "compiler_path": "cpu_ebm_langevin_llvm_jit", "executable": True,
        "kernel_id": f"ebm_langevin_{energy}_{manifold}_{shape[0]}x{shape[1]}_k{steps}",
        "op": "ebm_langevin_loop", "shape": shape, "eta": float(eta), "temperature": float(temperature),
        "steps": int(steps), "dtype": "f32", "manifold": manifold, "energy": energy,
    })


# ---------------------------------------------------------------------------
# Device route (EBM_NATIVE_LOOP_ARCHITECTURE.md, realized as the row-program
# emitter): the same lowered loop becomes one cooperative kernel -- one block
# per row, one lane per feature, the K steps and the Philox draw inside the
# kernel, state in registers -- packaged by build_native_gpu_storage.
# ---------------------------------------------------------------------------

DEVICE_ENTRY = "tessera_jit_ebm_langevin_loop"
_DEVICE_PIPELINE = ("--tessera-autodiff-paired", "--tessera-ebm-canonicalize", "--tessera-ebm-lower-langevin",
                    "--tessera-to-linalg", "--inline", "--convert-elementwise-to-linalg", "--canonicalize", "--cse")


def langevin_device_source(shape, *, eta: float, temperature: float, steps: int, backend: str, compiler,
                           manifold: str = "euclidean", energy: str = "quadratic"):
    """Run the whole chain in one tessera-opt invocation and attach the tensor contract."""
    from tessera.compiler.native_gpu_tensor import TensorSpec
    from tessera.compiler.native_row_program import MAX_FEATURES, row_program_device_source
    rows, feats = _check(shape, eta, temperature, steps)
    _check_kinds(energy, manifold)
    if feats > MAX_FEATURES:
        raise ValueError(f"langevin device route admits at most {MAX_FEATURES} features per row (one lane each)")
    specs: tuple[TensorSpec, ...] = (
        TensorSpec("y0", "fp32", (rows, feats), False), TensorSpec("x", "fp32", (rows, feats), False),
        TensorSpec("key", "int64", (2,), False), TensorSpec("y", "fp32", (rows, feats), True),
        TensorSpec("next_key", "int64", (2,), True))
    if manifold == "sphere":
        # The per-row status word the sphere integrator reports.
        specs = specs + (TensorSpec("status", "int32", (rows,), True),)
    source, full = row_program_device_source(
        langevin_loop_module((rows, feats), eta=eta, temperature=temperature, steps=steps, manifold=manifold,
                             energy=energy), entry=DEVICE_ENTRY,
        specs=specs, rows=rows, backend=backend, compiler=compiler, passes=_DEVICE_PIPELINE)
    if "tessera_ebm." in source.split("gpu.func @row_program(", 1)[1]:
        raise ValueError("langevin device route: an EBM op survived the lowering")
    return source, full


def bind_ebm_langevin_gpu(shape, *, eta, temperature, steps, compiler, llvm_bin, backend, chip,
                          manifold: str = "euclidean", energy: str = "quadratic"):
    """Package the loop for (backend, chip) and return its native tensor call."""
    from tessera.compiler.native_row_program import bind_row_program
    source, specs = langevin_device_source(shape, eta=eta, temperature=temperature, steps=steps,
                                           backend=backend, compiler=compiler, manifold=manifold, energy=energy)
    return bind_row_program(source, specs, compiler=compiler, llvm_bin=llvm_bin, backend=backend, chip=chip)


_PROGRAMS: dict = {}
_PROGRAM_LOCK = threading.Lock()


def ebm_langevin_program(shape, *, eta, temperature, steps, backend, chip, compiler, llvm_bin,
                         manifold: str = "euclidean", energy: str = "quadratic"):
    """Compile (once per process) and return the device program for the loop."""
    from tessera.compiler.native_host_program import HostArrayProgram, ensure_device_context
    key = (tuple(shape), float(eta), float(temperature), int(steps), backend, chip, str(compiler), str(llvm_bin),
           manifold, energy)
    with _PROGRAM_LOCK:
        program = _PROGRAMS.get(key)
        if program is None:
            ensure_device_context(backend)
            binding = bind_ebm_langevin_gpu(shape, eta=eta, temperature=temperature, steps=steps,
                                            compiler=compiler, llvm_bin=llvm_bin, backend=backend, chip=chip,
                                            manifold=manifold, energy=energy)
            program = _PROGRAMS[key] = HostArrayProgram(binding, f"ebm langevin loop ({energy}, {manifold})")
        return program


def native_langevin_loop_device(y0, x, key, *, eta, temperature, steps, backend, chip, compiler, llvm_bin,
                                manifold: str = "euclidean", energy: str = "quadratic"):
    """K Langevin steps as one device launch; returns (y_K, next_key) or, on
    the sphere, (y_K, next_key, status)."""
    program = ebm_langevin_program(np.asarray(y0).shape, eta=eta, temperature=temperature, steps=steps,
                                   backend=backend, chip=chip, compiler=compiler, llvm_bin=llvm_bin,
                                   manifold=manifold, energy=energy)
    return tuple(program.run(y0, x, np.asarray(key, dtype=np.int64).reshape(2)))


def package_ebm_langevin_native(shape, *, eta: float, temperature: float, steps: int, target: str,
                                manifold: str = "euclidean", energy: str = "quadratic"):
    """A runtime artifact for the device route (rows ``rocm`` /
    ``rocm_ebm_langevin_native_compiled``, ``nvidia_sm120`` /
    ``nvidia_ebm_langevin_native_compiled``); compiled at launch on the owning host."""
    if target not in ("rocm", "nvidia_sm120"):
        raise ValueError("langevin native device route targets rocm or nvidia_sm120")
    shape = _check(shape, eta, temperature, steps)
    _check_kinds(energy, manifold)
    from tessera.runtime import RuntimeArtifact
    path = "rocm_ebm_langevin_native_compiled" if target == "rocm" else "nvidia_ebm_langevin_native_compiled"
    return RuntimeArtifact(metadata={
        "target": target, "compiler_path": path, "executable": True,
        "kernel_id": f"ebm_langevin_native_{energy}_{manifold}_{shape[0]}x{shape[1]}_k{steps}",
        "op": "ebm_langevin_loop", "shape": shape, "eta": float(eta), "temperature": float(temperature),
        "steps": int(steps), "dtype": "f32", "manifold": manifold, "energy": energy,
    })

