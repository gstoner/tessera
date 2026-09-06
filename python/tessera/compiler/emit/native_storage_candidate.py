"""Arbiter adapter generated from a native package's compiler-preserved ABI."""
from __future__ import annotations
import ctypes as ct
from contextvars import ContextVar
from .candidate import Candidate, Tier, register_candidate, register_op_kind, unregister_candidate
from ..native_storage_contract import generate_tensor_binding

OP_NATIVE_STORAGE = 'native_storage'


class NativeStorageCandidate(Candidate):
    op = OP_NATIVE_STORAGE
    tier = Tier.EMITTED

    def __init__(self, package, signature, oracle):
        self.binding = generate_tensor_binding(package, signature)
        self.target = package.backend
        self.name = 'native-storage-' + package.binding_digest
        self.oracle = oracle
        self.native_runs = 0
        self._probe_runs: ContextVar[int] = ContextVar(self.name, default=0)
        self.closed = False

    def available(self):
        if self.closed:
            return False
        try:
            driver = ct.CDLL('libcuda.so.1' if self.target == 'nvidia' else 'libamdhip64.so')
            if self.target == 'nvidia':
                context = ct.c_void_p()
                query = driver.cuCtxGetCurrent
                query.argtypes, query.restype = [ct.POINTER(ct.c_void_p)], ct.c_int
                return query(ct.byref(context)) == 0 and context.value is not None
            count = ct.c_int()
            query = driver.hipGetDeviceCount
            query.argtypes, query.restype = [ct.POINTER(ct.c_int)], ct.c_int
            return query(ct.byref(count)) == 0 and count.value > 0
        except (OSError, AttributeError):
            return False

    def applies_to(self, region):
        return not self.closed and region == self.binding.package.binding_digest

    def applies_to_inputs(self, region, *inputs):
        if not self.applies_to(region):
            return False
        try:
            self.binding.prepare(*inputs)
            return True
        except (ValueError, TypeError, KeyError):
            return False

    def run(self, region, *inputs, **kwargs):
        if not self.applies_to(region):
            raise ValueError('native candidate region differs from its pinned package')
        result = self.binding(*inputs, **kwargs)
        self.native_runs += 1
        self._probe_runs.set(self._probe_runs.get() + 1)
        return result, f'{self.target}_native_storage'

    def close(self):
        self.binding.close()
        self.closed = True
        unregister_candidate(self)


def _verify(candidate, region, *, atol=1e-5, seed=0):
    if not isinstance(candidate, NativeStorageCandidate):
        return False
    before = candidate._probe_runs.get()
    try:
        verdict = candidate.oracle(candidate, region, atol=atol, seed=seed)
        return verdict is True and candidate._probe_runs.get() > before
    except Exception:
        return False


def register_native_storage_candidate(package, signature, oracle):
    """Generate and register a Tier-2 candidate; the arbiter still gates it.

    The caller supplies the operation's numerical oracle. A callback that never
    executes this native candidate cannot attest it. No winner is preselected.
    """
    candidate = NativeStorageCandidate(package, signature, oracle)
    register_op_kind(OP_NATIVE_STORAGE, _verify)
    register_candidate(candidate)
    return candidate
