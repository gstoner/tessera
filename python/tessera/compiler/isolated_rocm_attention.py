"""Resident attention in a replaceable process; no device pointers cross IPC."""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import threading

import numpy as np

from .native_driver_isolation import DriverIsolationLease, SpawnedProcessBoundary
from .resident_rocm_attention import prepare_attention_cotangent

_UNCERTAIN: set = set()


class IsolatedROCmAttentionTape:
    """Synchronous public host-array owner with bounded worker response waits.

    The child's resident tape retains immutable primals and saved LSE. Recovery
    requires confirmed process death; replacement always repeats zero/nonzero VJP
    health check. This is not a global device-health certificate or device reset.
    """
    def __init__(self,program,buffers,*,device=0,timeout_seconds=30.0):
        if type(device) is not int or not 0 <= device < 2**31:
            raise ValueError('attention isolation requires a nonnegative device ordinal')
        if type(timeout_seconds) not in (int,float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('attention isolation requires a finite positive timeout')
        self.program,self.device,self.timeout = program,device,float(timeout_seconds)
        self._buffers = {name: np.array(value,copy=True,order='C') for name,value in buffers.items()}
        forward = {b.name for b in program.descriptors[0].buffers if b.direction == 'input'}
        for name in forward:
            if name not in self._buffers or not np.all(np.isfinite(self._buffers[name])):
                raise ValueError("isolated attention requires finite captured inputs")
        names = [b.name for b in program.descriptors[1].buffers if b.layout != 'program_workspace' and b.name not in forward]
        if len(names) != 1 or names[0] not in self._buffers:
            raise ValueError('attention isolation requires one declared cotangent')
        self._shape,self._dtype = self._buffers[names[0]].shape,self._buffers[names[0]].dtype
        for value in self._buffers.values():
            value.setflags(write=False)
        self._lock = threading.RLock()
        self.failed = self.closed = False
        context = mp.get_context('spawn')
        parent,child = context.Pipe()
        self._connection = parent
        self._process = context.Process(target=_worker,args=(child,program,self._buffers,device,names[0]),daemon=True)
        try:
            self._process.start()
        except BaseException:
            parent.close();child.close()
            raise
        child.close()
        self.lease = DriverIsolationLease(SpawnedProcessBoundary(self._process),
            context_identity=f'attention-worker-{self._process.pid}',timeout_seconds=min(self.timeout,5.0))
        try:
            if self._receive() != ('ready',program.image.image_digest):
                raise RuntimeError('attention worker health admission failed')
        except BaseException:
            self._poison()
            self.recover()
            raise

    def _receive(self):
        if not self._connection.poll(self.timeout):
            raise TimeoutError('attention worker response deadline exceeded')
        return self._connection.recv()

    def _poison(self):
        self.failed = True
        self.lease.mark_uncertain()
        _UNCERTAIN.add(self)

    def backward(self,cotangent,*,casting='no'):
        with self._lock:
            if self.failed or self.closed:
                raise RuntimeError('attention worker is closed or uncertain')
            value = prepare_attention_cotangent(cotangent,self._shape,self._dtype,casting)
            try:
                self._connection.send(('backward',value))
                message = self._receive()
                if message[0] != 'result':
                    raise RuntimeError(f'attention worker failed: {message}')
                return message[1]
            except BaseException:
                self._poison()
                raise

    def recover(self):
        with self._lock:
            if not self.failed:
                raise ValueError('attention recovery requires an uncertain worker')
            self.lease.recover()
            self._connection.close()
            self.closed = True
            _UNCERTAIN.discard(self)

    def replacement(self):
        with self._lock:
            if not self.closed or not self.failed or not self.lease.reusable:
                raise ValueError('attention replacement requires confirmed worker death')
            return type(self)(self.program,self._buffers,device=self.device,timeout_seconds=self.timeout)

    def close(self):
        with self._lock:
            if self.closed:
                return
            if self.failed:
                self.recover()
                return
            try:
                self._connection.send(('close',))
                if self._receive() != ('closed',):
                    raise RuntimeError('attention worker close failed')
                self._process.join(self.timeout)
                if self._process.exitcode != 0:
                    raise RuntimeError('attention worker teardown unconfirmed')
                self._connection.close()
                self.closed = True
            except BaseException:
                self._poison()
                raise

    def __enter__(self):
        return self

    def __exit__(self,exc_type,exc,tb):
        try:
            try:
                self.close()
            except BaseException:
                if self.failed:
                    self.recover()
                raise
        except BaseException as error:
            if exc is None:
                raise
            exc.add_note(f'attention isolated cleanup incomplete: {error}')
        return False


def _worker(connection,program,buffers,device,cotangent_name):
    try:
        from tessera import runtime as rt
        from .resident_rocm_attention import ResidentROCmAttentionTape
        hip = rt._load_hip_for_launch()
        if hip is None or hip.hipSetDevice(device) != 0:
            raise RuntimeError('attention worker cannot select its HIP device')
        with ResidentROCmAttentionTape(program,buffers) as tape:
            _check_health(tape,program,buffers,cotangent_name)
            connection.send(('ready',program.image.image_digest))
            while True:
                message = connection.recv()
                if message[0] == 'close':
                    break
                if message[0] != 'backward':
                    raise ValueError('unknown attention worker request')
                connection.send(('result',tape.backward(message[1])))
        connection.send(('closed',))
        connection.close()
    except BaseException as error:
        try:
            connection.send(('error',type(error).__name__,str(error)))
            connection.close()
        finally:
            os._exit(1)


def _check_health(tape,program,buffers,cotangent_name):
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
    q,k,v = (buffers[name] for name in inputs[:3])
    provenance = program.descriptors[0].provenance
    cotangent = np.ones_like(buffers[cotangent_name])
    expected = reference_attention_backward_split_reduced(cotangent,q,k,v,
        bias=buffers[inputs[3]] if len(inputs) == 4 else None,
        **{name: provenance[name] for name in ('split_count','scale','causal','window_left','window_right','softcap','dropout_p','dropout_seed')})
    actual = tape.backward(cotangent)
    if len(actual) != 3 or not all(np.allclose(a,e,rtol=.04,atol=.003,equal_nan=False) for a,e in zip(actual,expected,strict=True)):
        raise RuntimeError('attention nonzero-VJP health probe failed')
