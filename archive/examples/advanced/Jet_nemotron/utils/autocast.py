# utils/autocast.py
"""Mixed-precision autocast shim for Tessera-like API (scaffold).

Usage:
    from tessera_jetnemotron.utils.autocast import Autocast
    with Autocast(enabled=True, compute_dtype="fp16", accum_dtype="fp32"):
        loss = forward_loss(...)
"""
from contextlib import contextmanager

class Autocast:
    def __init__(self, enabled=True, compute_dtype="fp16", accum_dtype="fp32"):
        self.enabled = enabled
        self.compute_dtype = compute_dtype
        self.accum_dtype = accum_dtype

    def __enter__(self):
        # Placeholder: wire to tessera.effects.autocast if available
        # e.g., effects.autocast(dtype=self.compute_dtype, accum=self.accum_dtype).__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Placeholder: exit underlying autocast if used
        return False  # do not suppress exceptions
