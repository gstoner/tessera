"""Resident ROCm attention in a replaceable process; no device pointers cross IPC."""
from __future__ import annotations

import os

import numpy as np

from .isolated_attention import IsolatedAttentionTape, _UNCERTAIN, serve
from .resident_rocm_attention import prepare_attention_cotangent

__all__ = ["IsolatedROCmAttentionTape", "_UNCERTAIN"]


class IsolatedROCmAttentionTape(IsolatedAttentionTape):
    """Synchronous public host-array owner with bounded worker response waits.

    The child's resident tape retains immutable primals and saved LSE. Recovery
    requires confirmed process death; replacement always repeats zero/nonzero VJP
    health check. This is not a global device-health certificate or device reset.
    The spawn/lease/relay body lives in ``isolated_attention.IsolatedAttentionTape``
    (shared with the CUDA owner since 2026-09-18); this class contributes the HIP
    worker, the ROCm program payload and the split-reduced health oracle.
    """
    def __init__(self, program, buffers, *, device=0, timeout_seconds=30.0):
        self.program = program
        self._buffers = {name: np.array(value, copy=True, order='C') for name, value in buffers.items()}
        forward = {b.name for b in program.descriptors[0].buffers if b.direction == 'input'}
        for name in forward:
            if name not in self._buffers or not np.all(np.isfinite(self._buffers[name])):
                raise ValueError("isolated attention requires finite captured inputs")
        names = [b.name for b in program.descriptors[1].buffers if b.layout != 'program_workspace' and b.name not in forward]
        if len(names) != 1 or names[0] not in self._buffers:
            raise ValueError('attention isolation requires one declared cotangent')
        self._cotangent_name = names[0]
        self._shape, self._dtype = self._buffers[names[0]].shape, self._buffers[names[0]].dtype
        for value in self._buffers.values():
            value.setflags(write=False)
        super().__init__(device=device, timeout_seconds=timeout_seconds)

    def _worker_target(self):
        return _worker

    def _payload(self):
        return (self.program, self._buffers, self.device, self._cotangent_name)

    def _ready_token(self):
        return self.program.image.image_digest

    def _prepare_cotangent(self, cotangent, casting):
        return prepare_attention_cotangent(cotangent, self._shape, self._dtype, casting)

    def _replacement_kwargs(self):
        return dict(program=self.program, buffers=self._buffers)


def _worker(connection, program, buffers, device, cotangent_name):
    try:
        from tessera import runtime as rt
        from .resident_rocm_attention import ResidentROCmAttentionTape
        hip = rt._load_hip_for_launch()
        if hip is None or hip.hipSetDevice(device) != 0:
            raise RuntimeError('attention worker cannot select its HIP device')
        with ResidentROCmAttentionTape(program, buffers) as tape:
            _check_health(tape, program, buffers, cotangent_name)
            serve(connection, tape, program.image.image_digest, lambda t, value: t.backward(value))
    except BaseException as error:
        try:
            connection.send(('error', type(error).__name__, str(error)))
            connection.close()
        finally:
            os._exit(1)


def _check_health(tape, program, buffers, cotangent_name):
    # Workload-scoped health probe also initializes saved O/LSE. No
    # readiness acknowledgement precedes native execution and copyback.
    probe = tape.backward(np.zeros_like(buffers[cotangent_name]))
    if not all(np.all(value == 0) for value in probe):
        raise RuntimeError('attention zero-VJP health probe failed')
    # A zero derivative cannot detect a missing/uninitialized saved
    # checkpoint. Check a nonzero cotangent against the shared oracle
    # before admitting this workload on either initial or replacement
    # workers. This is not permission to reuse the old device context.
    from .attention_contract import reference_attention_backward_split_reduced
    inputs = [b.name for b in program.descriptors[0].buffers if b.direction == 'input' and b.layout != 'program_workspace']
    q, k, v = (buffers[name] for name in inputs[:3])
    provenance = program.descriptors[0].provenance
    cotangent = np.ones_like(buffers[cotangent_name])
    expected = reference_attention_backward_split_reduced(cotangent, q, k, v,
        bias=buffers[inputs[3]] if len(inputs) == 4 else None,
        **{name: provenance[name] for name in ('split_count', 'scale', 'causal', 'window_left', 'window_right', 'softcap', 'dropout_p', 'dropout_seed')})
    actual = tape.backward(cotangent)
    if len(actual) != 3 or not all(np.allclose(a, e, rtol=.04, atol=.003, equal_nan=False) for a, e in zip(actual, expected, strict=True)):
        raise RuntimeError('attention nonzero-VJP health probe failed')
