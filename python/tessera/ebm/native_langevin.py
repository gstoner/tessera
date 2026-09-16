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


def langevin_loop_module(shape, *, eta: float, temperature: float, steps: int) -> str:
    """The Graph-level program: quadratic energy + a K-step Langevin loop."""
    rows, features = _check(shape, eta, temperature, steps)
    st = f"tensor<{rows}x{features}xf32>"
    en = f"tensor<{rows}xf32>"
    return f"""module {{
  func.func @quadratic_energy(%y: {st}, %x: {st}) -> {en}
      attributes {{tessera.autodiff = "reverse"}} {{
    %d = "tessera.sub"(%x, %y) : ({st}, {st}) -> {st}
    %sq = "tessera.mul"(%d, %d) : ({st}, {st}) -> {st}
    %s = "tessera.reduce"(%sq) {{axis = 1 : i64, kind = "sum"}} : ({st}) -> {en}
    %half = arith.constant dense<5.000000e-01> : {en}
    %e = "tessera.mul"(%s, %half) : ({en}, {en}) -> {en}
    return %e : {en}
  }}
  func.func @tessera_jit_ebm_energy(%y: {st}, %x: {st}) -> {en} {{
    %e = "tessera_ebm.energy"(%x, %y) {{ energy_fn = @quadratic_energy }} : ({st}, {st}) -> {en}
    return %e : {en}
  }}
  func.func @tessera_jit_ebm_langevin_loop(%y0: {st}, %x: {st}, %key0: tensor<2xi64>) -> ({st}, tensor<2xi64>) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %steps = arith.constant {steps} : index
    %r:2 = scf.for %t = %c0 to %steps step %c1 iter_args(%y = %y0, %key = %key0) -> ({st}, tensor<2xi64>) {{
      %n:2 = "tessera_ebm.langevin_step"(%y, %key, %x)
          {{ energy_fn = @quadratic_energy, eta = {float(eta)!r} : f64, temperature = {float(temperature)!r} : f64, manifold = "euclidean" }}
          : ({st}, tensor<2xi64>, {st}) -> ({st}, tensor<2xi64>)
      scf.yield %n#0, %n#1 : {st}, tensor<2xi64>
    }}
    return %r#0, %r#1 : {st}, tensor<2xi64>
  }}
}}
"""


def _require():
    if not has_native_langevin():
        raise jb.TesseraJitError("libtessera_jit was built without the EBM lane "
                                 "(configure with -DTESSERA_BUILD_EBM_BACKEND=ON)")


def native_quadratic_energy(y, x) -> np.ndarray:
    """Per-row ``0.5 * ||x - y||^2`` through the lane (the energy op lowers to a call)."""
    _require()
    y = np.ascontiguousarray(y, dtype=np.float32); x = np.ascontiguousarray(x, dtype=np.float32)
    if y.shape != x.shape:
        raise jb.TesseraJitError("energy requires equal state and context shapes")
    shape = _check(y.shape, 1.0, 0.0, 1)
    handle = jb.compile_module(langevin_loop_module(shape, eta=1.0, temperature=0.0, steps=1))
    try:
        out = np.empty((shape[0],), np.float32)
        jb.invoke(handle, "tessera_jit_ebm_energy", [y, x], out)
        return out
    finally:
        jb.destroy(handle)


def native_langevin_loop(y0, x, key: Sequence[int], *, eta: float, temperature: float, steps: int):
    """K Langevin steps on the quadratic energy as one compiled function.

    Returns ``(y_K, next_key)``; ``key`` is two int64 words (S4 RNGKey words).
    """
    _require()
    y0 = np.ascontiguousarray(y0, dtype=np.float32); x = np.ascontiguousarray(x, dtype=np.float32)
    if y0.shape != x.shape:
        raise jb.TesseraJitError("langevin requires equal state and context shapes")
    shape = _check(y0.shape, eta, temperature, steps)
    key_arr = np.ascontiguousarray(np.asarray(key, dtype=np.int64).reshape(2))
    handle = jb.compile_module(langevin_loop_module(shape, eta=eta, temperature=temperature, steps=steps))
    try:
        out = np.empty(shape, np.float32)
        next_key = np.empty((2,), np.int64)
        jb.invoke(handle, "tessera_jit_ebm_langevin_loop", [y0, x, key_arr], [out, next_key])
        return out, next_key
    finally:
        jb.destroy(handle)


def reference_langevin_loop(y0, x, key: Sequence[int], *, eta: float, temperature: float, steps: int):
    """The declared policy in numpy, bit-for-bit: Philox key = (lo32, hi32) of
    key[0]; counter = (flat index, lo32, hi32 of key[1], 0); z = sqrt(-2 ln u0)
    cos(2 pi u1) with u = (word + 0.5) * 2^-32 on words 0 and 1 (computed in
    f64, rounded to f32 once); next key = (key[0], key[1] + 1); the gradient of
    the quadratic energy is y - x."""
    from tessera.compiler.philox import philox_4x32_10
    y = np.ascontiguousarray(y0, dtype=np.float32).copy(); x = np.ascontiguousarray(x, dtype=np.float32)
    shape = _check(y.shape, eta, temperature, steps)
    k = [int(v) for v in np.asarray(key, dtype=np.int64).reshape(2)]
    scale = math.sqrt(2.0 * float(eta) * float(temperature))
    philox_key = np.array([k[0] & 0xFFFFFFFF, (k[0] >> 32) & 0xFFFFFFFF], np.uint32)
    for _ in range(steps):
        grad = y - x
        y = (y - np.float32(eta) * grad).astype(np.float32)
        if scale > 0.0:
            s0, s1 = np.uint32(k[1] & 0xFFFFFFFF), np.uint32((k[1] >> 32) & 0xFFFFFFFF)
            z = np.empty(math.prod(shape), np.float32)
            for i in range(z.size):
                words = philox_4x32_10(np.array([np.uint32(i), s0, s1, np.uint32(0)], np.uint32), philox_key)
                u0 = (float(words[0]) + 0.5) * 2.0 ** -32
                u1 = (float(words[1]) + 0.5) * 2.0 ** -32
                z[i] = np.float32(math.sqrt(-2.0 * math.log(u0)) * math.cos(2.0 * math.pi * u1))
            y = (y + np.float32(scale) * z.reshape(shape)).astype(np.float32)
        k[1] += 1
    return y, np.array(k, np.int64)


def package_ebm_langevin_cpu(shape, *, eta: float, temperature: float, steps: int):
    """A runtime artifact for the loop (row ``cpu`` / ``cpu_ebm_langevin_llvm_jit``)."""
    shape = _check(shape, eta, temperature, steps)
    _require()
    from tessera.runtime import RuntimeArtifact
    return RuntimeArtifact(metadata={
        "target": "cpu", "compiler_path": "cpu_ebm_langevin_llvm_jit", "executable": True,
        "kernel_id": f"ebm_langevin_quadratic_{shape[0]}x{shape[1]}_k{steps}",
        "op": "ebm_langevin_loop", "shape": shape, "eta": float(eta), "temperature": float(temperature),
        "steps": int(steps), "dtype": "f32",
    })
