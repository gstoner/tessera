"""Resident attention in a replaceable process, for any backend; no device
pointers cross IPC.

The owner spawns a worker, admits it only after a zero/nonzero VJP health
probe inside the child, relays host-array cotangents over a pipe with a
bounded wait, and treats a missed deadline or a broken pipe as an *uncertain*
worker: recovery needs confirmed process death (``DriverIsolationLease``) and
a replacement re-runs the health probe on a fresh process. Everything a
backend contributes is a module-level worker function (spawn-picklable), the
payload it needs, the readiness token it answers with, and how a cotangent is
prepared; ``IsolatedROCmAttentionTape`` and ``IsolatedCUDAAttentionTape``
supply those. This is not a global device-health certificate or a device
reset.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import threading
from typing import Any, Callable

from .native_driver_isolation import DriverIsolationLease, SpawnedProcessBoundary

#: Owners whose worker died or stalled and whose death is not yet confirmed.
_UNCERTAIN: set = set()


class IsolatedAttentionTape:
    """Synchronous host-array owner with bounded worker response waits."""

    def __init__(self, *, device: int = 0, timeout_seconds: float = 30.0) -> None:
        if type(device) is not int or not 0 <= device < 2**31:
            raise ValueError('attention isolation requires a nonnegative device ordinal')
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('attention isolation requires a finite positive timeout')
        self.device, self.timeout = device, float(timeout_seconds)
        self._lock = threading.RLock()
        self.failed = self.closed = False
        context = mp.get_context('spawn')
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(target=self._worker_target(), args=(child, *self._payload()), daemon=True)
        try:
            self._process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        self.lease = DriverIsolationLease(SpawnedProcessBoundary(self._process),
            context_identity=f'attention-worker-{self._process.pid}', timeout_seconds=min(self.timeout, 5.0))
        try:
            if self._receive() != ('ready', self._ready_token()):
                raise RuntimeError('attention worker health admission failed')
        except BaseException:
            self._poison()
            self.recover()
            raise

    # -- backend hooks -------------------------------------------------------
    def _worker_target(self) -> Callable[..., None]:
        """The spawn target, ``worker(connection, *payload)``; resolved at
        spawn time so a test can substitute the module-level function."""
        raise NotImplementedError

    def _payload(self) -> tuple:
        """Everything the worker needs after the connection, the device ordinal
        included wherever the backend's worker signature puts it."""
        raise NotImplementedError

    def _ready_token(self) -> Any:
        raise NotImplementedError

    def _prepare_cotangent(self, cotangent: Any, casting: str) -> Any:
        raise NotImplementedError

    def _replacement_kwargs(self) -> dict:
        raise NotImplementedError

    # -- protocol -----------------------------------------------------------
    def _receive(self):
        if not self._connection.poll(self.timeout):
            raise TimeoutError('attention worker response deadline exceeded')
        return self._connection.recv()

    def _poison(self):
        self.failed = True
        self.lease.mark_uncertain()
        _UNCERTAIN.add(self)

    def backward(self, cotangent, *, casting='no'):
        with self._lock:
            if self.failed or self.closed:
                raise RuntimeError('attention worker is closed or uncertain')
            value = self._prepare_cotangent(cotangent, casting)
            try:
                self._connection.send(('backward', value))
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
            return type(self)(**self._replacement_kwargs(), device=self.device, timeout_seconds=self.timeout)

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

    def __exit__(self, exc_type, exc, tb):
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


def serve(connection, tape, ready_token, run_backward):
    """The worker's request loop once its resident tape is admitted."""
    connection.send(('ready', ready_token))
    while True:
        message = connection.recv()
        if message[0] == 'close':
            break
        if message[0] != 'backward':
            raise ValueError('unknown attention worker request')
        connection.send(('result', run_backward(tape, message[1])))
    connection.send(('closed',))
    connection.close()
