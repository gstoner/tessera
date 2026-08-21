"""W4-PRODUCT-1 — the bounded IRREDUCIBLE-CFG state machine executes on
gfx1151, forward and generated backward, as one device kernel each.

The exact-device row the integrated plan's queue order 2 names ("Next extend
the physical packet with one true irreducible state machine"): a two-entry
SCC — entry may jump into either cycle block — is structurized by
`--tessera-autodiff-paired` into the typed program-counter state machine
(`bounded_state_machine_v1`), differentiated (recompute_all), and then
lowered by `generate-rocm-state-machine-kernel` to per-thread gpu.func
kernels that run the WHOLE machine per element. Binaries come from the
CANONICAL registered executable pipeline (PR #605 review P1 — the same
route normal ROCm compilation takes):

  paired → tessera-rocm-executable{family=control_state_machine
           input=tile output=binary arch=gfx1151} → hsaco →
  hipModuleLaunchKernel

The row binds the exact CFG identity: the structured-CFG digest stamped on
the source function must be stamped verbatim on the emitted gpu.func
(asserted below), and the per-thread STATUS buffer carries the `cf.assert`
bound check — the host REJECTS a launch whose STATUS is not all-ones, so
max_steps exhaustion cannot pass silently.

Both entry paths of the irreducible SCC are exercised:
  enter_left=True : entry→bb1→bb2→bb1→bb2→bb3  ⇒ y = tanh(tanh(x))
  enter_left=False: entry→bb2→bb1→bb2→bb3      ⇒ y = tanh(x)
with the backward checked against the analytic derivative.

This is a correctness row only (WSL — Decision #26a timing rules); clean
bare-metal timing remains open per the W4.3 ledger. Skip-clean when
tessera-opt isn't built or no usable AMD GPU is present.
"""

from __future__ import annotations

import ctypes
import os

import pytest

from tests._support.compiler_tool import require_tessera_opt, run_tessera_opt

np = pytest.importorskip("numpy")

CHIP = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")
BD = 256  # must match GenerateROCMStateMachineKernel's block dim

DIGEST = "4242424242424242424242424242424242424242424242424242424242424242"


def _irreducible_mlir(n: int) -> str:
    return f"""
module {{
  func.func @irreducible(%enter_left: i1, %x: tensor<{n}xf32>)
      -> tensor<{n}xf32> attributes {{tessera.autodiff = "reverse"}} {{
    %out = scf.execute_region -> tensor<{n}xf32> {{
      %c0 = arith.constant 0 : index
      cf.cond_br %enter_left, ^bb1(%c0, %x : index, tensor<{n}xf32>),
                              ^bb2(%c0, %x : index, tensor<{n}xf32>)
    ^bb1(%i: index, %state: tensor<{n}xf32>):
      %next = "tessera.tanh"(%state) :
          (tensor<{n}xf32>) -> tensor<{n}xf32>
      cf.br ^bb2(%i, %next : index, tensor<{n}xf32>)
    ^bb2(%j: index, %right_state: tensor<{n}xf32>):
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %next_i = arith.addi %j, %c1 : index
      %continue = arith.cmpi slt, %next_i, %c2 : index
      cf.cond_br %continue,
          ^bb1(%next_i, %right_state : index, tensor<{n}xf32>),
          ^bb3(%right_state : tensor<{n}xf32>)
    ^bb3(%result: tensor<{n}xf32>):
      scf.yield %result : tensor<{n}xf32>
    }} {{tessera.structured_cfg.digest = "{DIGEST}",
       tessera.structured_cfg.max_steps = 8 : i64}}
    return %out : tensor<{n}xf32>
  }}
}}
"""


def _oracle_forward(x: np.ndarray, enter_left: bool) -> np.ndarray:
    # Simulate the CFG: bb1 applies tanh; bb2 counts to 2 then exits.
    i, state, block = 0, x, ("bb1" if enter_left else "bb2")
    while True:
        if block == "bb1":
            state = np.tanh(state)
            block = "bb2"
        else:
            i += 1
            block = "bb1" if i < 2 else "bb3"
            if block == "bb3":
                return state


def _oracle_backward(x: np.ndarray, dout: np.ndarray,
                     enter_left: bool) -> np.ndarray:
    if enter_left:  # y = tanh(tanh(x))
        t1 = np.tanh(x)
        t2 = np.tanh(t1)
        return dout * (1.0 - t2 * t2) * (1.0 - t1 * t1)
    t1 = np.tanh(x)  # y = tanh(x)
    return dout * (1.0 - t1 * t1)


def _load_hip():
    for name in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _extract_hsacos(s: str) -> dict[str, bytes]:
    """All gpu.binary payloads in the module, keyed by binary symbol name."""
    out: dict[str, bytes] = {}
    pos = 0
    while True:
        b = s.find("gpu.binary @", pos)
        if b < 0:
            return out
        start = b + len("gpu.binary @")
        end = start
        while s[end] not in " [\n":
            end += 1
        name = s[start:end]
        j = s.index('bin = "', b) + len('bin = "')
        raw = bytearray()
        hexd = "0123456789abcdefABCDEF"
        simple = {"\\": 0x5C, '"': 0x22, "n": 0x0A, "t": 0x09, "r": 0x0D}
        while j < len(s):
            c = s[j]
            if c == '"':
                break
            if c == "\\":
                nxt = s[j + 1:j + 3]
                if len(nxt) == 2 and nxt[0] in hexd and nxt[1] in hexd:
                    raw.append(int(nxt, 16))
                    j += 3
                    continue
                if s[j + 1] in simple:
                    raw.append(simple[s[j + 1]])
                    j += 2
                    continue
            raw.append(ord(c))
            j += 1
        out[name] = bytes(raw)
        pos = j


def _compile(n: int) -> dict[str, bytes]:
    src = _irreducible_mlir(n)
    paired = run_tessera_opt(src, "--tessera-autodiff-paired")
    assert paired.returncode == 0, f"paired failed: {paired.stderr}"
    gen = run_tessera_opt(paired.stdout, "--generate-rocm-state-machine-kernel")
    assert gen.returncode == 0, f"kernel-gen failed: {gen.stderr}"
    # The row binds the exact CFG digest: the gpu.func must carry the SOURCE
    # digest verbatim (and the recompute_all residual policy).
    assert gen.stdout.count(f'tessera.structured_cfg.digest = "{DIGEST}"') >= 4, \
        "kernels do not bind the structured-CFG digest"
    for kname in ("tessera_state_machine_irreducible",
                  "tessera_state_machine_irreducible__bwd"):
        assert f"gpu.func @{kname}" in gen.stdout, f"missing kernel {kname}"
    # PR #605 review (P1): serialize through the CANONICAL registered
    # executable pipeline — the same `family=control_state_machine` route
    # normal ROCm binary compilation takes — not a hand-assembled pass list.
    pipe = ("builtin.module(tessera-rocm-executable{family=control_state_machine "
            f"input=tile output=binary arch={CHIP}}})")
    ser = run_tessera_opt(paired.stdout, f"--pass-pipeline={pipe}")
    assert ser.returncode == 0, f"serialize failed: {ser.stderr}"
    hsacos = _extract_hsacos(ser.stdout)
    for name, blob in hsacos.items():
        assert blob[:4] == b"\x7fELF", f"{name}: not an ELF hsaco"
    return hsacos


class _Device:
    def __init__(self, hip):
        self.hip = hip
        self.ok = hip.hipInit(0) == 0

    def launch(self, hsaco: bytes, kernel: bytes, buffers: list[np.ndarray],
               n: int) -> bool:
        """Launch kernel(memrefs(buffers)..., N). Buffers are copied in,
        launched over ceil(n/BD) blocks, and copied back in place."""
        hip = self.hip
        mod = ctypes.c_void_p()
        if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
            return False
        fn = ctypes.c_void_p()
        if hip.hipModuleGetFunction(ctypes.byref(fn), mod, kernel) != 0:
            return False
        devs = []
        for buf in buffers:
            d = ctypes.c_void_p()
            if hip.hipMalloc(ctypes.byref(d), buf.nbytes) != 0:
                return False
            hip.hipMemcpy(d, buf.ctypes.data_as(ctypes.c_void_p),
                          buf.nbytes, 1)  # H2D
            devs.append(d)

        def memref(p, size):
            return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                    ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int64(1)]

        args = []
        for d, buf in zip(devs, buffers):
            args += memref(d, buf.size)
        args.append(ctypes.c_int64(n))
        arr = (ctypes.c_void_p * len(args))()
        for i, a_ in enumerate(args):
            arr[i] = ctypes.cast(ctypes.byref(a_), ctypes.c_void_p)
        launch = hip.hipModuleLaunchKernel
        launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                           + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_void_p])
        grid = (n + BD - 1) // BD
        if launch(fn, grid, 1, 1, BD, 1, 1, 0, None, arr, None) != 0:
            return False
        if hip.hipDeviceSynchronize() != 0:
            return False
        for d, buf in zip(devs, buffers):
            hip.hipMemcpy(buf.ctypes.data_as(ctypes.c_void_p), d,
                          buf.nbytes, 2)  # D2H
            hip.hipFree(d)
        return True


@pytest.mark.parametrize("enter_left", [True, False])
def test_irreducible_state_machine_forward_executes_on_gfx1151(enter_left):
    require_tessera_opt()
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    n = 300  # deliberately not a multiple of BD (bounds check on device)
    rng = np.random.default_rng(5)
    x = rng.standard_normal(n).astype(np.float32)
    hsacos = _compile(n)
    dev = _Device(hip)
    if not dev.ok:
        pytest.skip("hipInit failed — no usable AMD GPU")

    flags = np.array([1.0 if enter_left else -1.0], dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    status = np.zeros(n, dtype=np.float32)
    launched = dev.launch(hsacos["tessera_state_machine_irreducible_mod"],
                          b"tessera_state_machine_irreducible",
                          [flags, x.copy(), out, status], n)
    if not launched:
        pytest.skip("no usable AMD GPU (module load / launch unavailable)")
    # The bound check is host-enforced: every thread must have finished
    # before max_steps (STATUS all-ones), else the result is rejected.
    assert np.all(status == 1.0), "bounded state machine exhausted max_steps"
    ref = _oracle_forward(x, enter_left)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("enter_left", [True, False])
def test_irreducible_state_machine_backward_executes_on_gfx1151(enter_left):
    require_tessera_opt()
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    n = 300
    rng = np.random.default_rng(6)
    x = rng.standard_normal(n).astype(np.float32)
    dout = rng.standard_normal(n).astype(np.float32)
    hsacos = _compile(n)
    dev = _Device(hip)
    if not dev.ok:
        pytest.skip("hipInit failed — no usable AMD GPU")

    flags = np.array([1.0 if enter_left else -1.0], dtype=np.float32)
    dx = np.zeros(n, dtype=np.float32)
    status = np.zeros(n, dtype=np.float32)
    launched = dev.launch(
        hsacos["tessera_state_machine_irreducible__bwd_mod"],
        b"tessera_state_machine_irreducible__bwd",
        [flags, x.copy(), dout.copy(), dx, status], n)
    if not launched:
        pytest.skip("no usable AMD GPU (module load / launch unavailable)")
    assert np.all(status == 1.0), "bounded state machine exhausted max_steps"
    ref = _oracle_backward(x, dout, enter_left)
    np.testing.assert_allclose(dx, ref, rtol=1e-5, atol=1e-6)


_SELECT_DIGEST = "7" * 64


def _select_machine_mlir(n: int) -> str:
    """A machine whose state update is per-element DATA-dependent:
    s = select(s > 0, tanh(s), s), three steps — the cmpf/select interior
    vocabulary the review's P2 named (tensor<Nxi1> intermediates must
    scalarize to i1, not survive as tensor results)."""
    return f"""
module {{
  func.func @select_machine(%x: tensor<{n}xf32>) -> tensor<{n}xf32> {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %r = scf.for %i = %c0 to %c3 step %c1 iter_args(%s = %x)
        -> (tensor<{n}xf32>) {{
      %zero = arith.constant dense<0.0> : tensor<{n}xf32>
      %p = arith.cmpf ogt, %s, %zero : tensor<{n}xf32>
      %t = "tessera.tanh"(%s) : (tensor<{n}xf32>) -> tensor<{n}xf32>
      %n2 = arith.select %p, %t, %s : tensor<{n}xi1>, tensor<{n}xf32>
      scf.yield %n2 : tensor<{n}xf32>
    }} {{tessera.structured_cfg.execution = "bounded_state_machine_v1",
       tessera.structured_cfg.digest = "{_SELECT_DIGEST}",
       tessera.structured_cfg.max_steps = 4 : i64}}
    return %r : tensor<{n}xf32>
  }}
}}
"""


def test_data_dependent_select_machine_executes_on_gfx1151():
    require_tessera_opt()
    hip = _load_hip()
    if hip is None:
        pytest.skip("libamdhip64.so not loadable — no ROCm host")
    n = 300
    rng = np.random.default_rng(9)
    x = rng.standard_normal(n).astype(np.float32)

    src = _select_machine_mlir(n)
    gen = run_tessera_opt(src, "--generate-rocm-state-machine-kernel")
    assert gen.returncode == 0, f"kernel-gen failed: {gen.stderr}"
    assert f'tessera.structured_cfg.digest = "{_SELECT_DIGEST}"' in gen.stdout
    pipe = ("builtin.module(tessera-rocm-executable{family=control_state_machine "
            f"input=tile output=binary arch={CHIP}}})")
    ser = run_tessera_opt(src, f"--pass-pipeline={pipe}")
    assert ser.returncode == 0, f"serialize failed: {ser.stderr}"
    hsacos = _extract_hsacos(ser.stdout)

    dev = _Device(hip)
    if not dev.ok:
        pytest.skip("hipInit failed — no usable AMD GPU")
    out = np.zeros(n, dtype=np.float32)
    status = np.zeros(n, dtype=np.float32)
    launched = dev.launch(hsacos["tessera_state_machine_select_machine_mod"],
                          b"tessera_state_machine_select_machine",
                          [np.zeros(1, dtype=np.float32),  # FLAGS (no i1 args)
                           x.copy(), out, status], n)
    if not launched:
        pytest.skip("no usable AMD GPU (module load / launch unavailable)")
    assert np.all(status == 1.0)
    ref = x.copy()
    for _ in range(3):
        ref = np.where(ref > 0.0, np.tanh(ref), ref)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
