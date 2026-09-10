"""Opt-in process-owned native ANN execution with checked host result transport.

No device pointer crosses the process boundary. A failed/late request poisons
its worker; replacement requires confirmed process death, never a driver retry.
"""
from __future__ import annotations

import math
import os
import ctypes as ct
import multiprocessing as mp
import subprocess
import threading
import time

import numpy as np

from .native_driver_isolation import DriverIsolationLease, IsolationRecovery


def _ann_worker(connection, pair, input_bound, absolute_budget):
    try:
        backend = pair.original.backend
        driver = ct.CDLL('libcuda.so.1' if backend == 'nvidia' else 'libamdhip64.so')
        def call(name, types, *args):
            fn = getattr(driver, name)
            fn.argtypes, fn.restype = types, ct.c_int
            status = fn(*args)
            if status:
                raise RuntimeError(f'isolated worker context initialization failed: {name} {status}')
        if backend == 'nvidia':
            call('cuInit', [ct.c_uint], 0)
            context = ct.c_void_p()
            call('cuDevicePrimaryCtxRetain', [ct.POINTER(ct.c_void_p), ct.c_int], ct.byref(context), 0)
            call('cuCtxSetCurrent', [ct.c_void_p], context)
        elif backend == 'rocm':
            call('hipInit', [ct.c_uint], 0)
            call('hipSetDevice', [ct.c_int], 0)
        else:
            raise ValueError('isolated worker requires CUDA or HIP')
        _serve(connection, pair, input_bound, absolute_budget)
    except BaseException as error:
        try:
            connection.send(('error', type(error).__name__, str(error)))
        finally:
            connection.close()


def _serve(connection, pair, input_bound, absolute_budget):
    runner = None
    try:
        runner = pair.bind(input_bound=input_bound, absolute_budget=absolute_budget)
        # Admission exercises transfers, both admitted kernels and the independent
        # analytic oracle inside this process. The parent startup deadline also
        # bounds a driver call that never returns during these probes.
        amplitude = min(float(input_bound), 1.0)
        probes = [np.zeros(runner.shape, np.float32),
                  np.linspace(-amplitude, amplitude, int(np.prod(runner.shape)),
                              dtype=np.float32).reshape(runner.shape)]
        if runner.verify(probes) is not True:
            raise RuntimeError('isolated device health probe did not verify')
        connection.send(('ready', pair.original.binding_digest, pair.transformed.binding_digest,
                         'numerical-health-v1'))
        while True:
            message = connection.recv()
            if message[0] == 'close':
                runner.close()
                runner = None
                connection.send(('closed',))
                return
            _, request, value, transformed = message
            output = runner.run(value, transformed=transformed)
            connection.send(('result', request, output))
    except BaseException as error:
        try:
            connection.send(('error', type(error).__name__, str(error)))
        except (OSError, EOFError):
            pass
        connection.close()
        os._exit(1)  # Avoid destructor-driven driver retries after uncertainty.
    finally:
        # On failure, do not retry a driver operation in this process. Process
        # exit is the isolation boundary; the parent still must join it.
        connection.close()


class _ProcessBoundary:
    def __init__(self, process):
        self.process = process

    def poll(self):
        return self.process.exitcode

    def terminate(self):
        self.process.terminate()

    def kill(self):
        self.process.kill()

    def wait(self, timeout=None):
        self.process.join(timeout)
        if self.process.exitcode is None:
            raise subprocess.TimeoutExpired('native ANN worker', timeout)
        return self.process.exitcode


# Retain uncertain channels/process owners even if the caller loses its handle.
_UNCERTAIN_WORKERS: set[IsolatedNativeANN] = set()


class IsolatedNativeANN:
    """Device zero, one outstanding request; only confirmed host results escape."""
    def __init__(self, pair, *, input_bound, absolute_budget, timeout_seconds=30.0):
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('isolated ANN requires a finite positive timeout')
        self.pair, self.timeout = pair, float(timeout_seconds)
        self.input_bound, self.absolute_budget = input_bound, absolute_budget
        self._lock = threading.RLock()
        self._pending = None
        self._recovery = None
        self._next = 0
        self.closed = False
        self.failed = False
        context = mp.get_context('spawn')
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(target=_ann_worker, args=(child, pair, input_bound, absolute_budget), daemon=True)
        self._process.start()
        child.close()
        self.lease = DriverIsolationLease(_ProcessBoundary(self._process),
            context_identity=f'ann-worker-{self._process.pid}', timeout_seconds=min(self.timeout, 5.0))
        try:
            if not parent.poll(self.timeout):
                raise TimeoutError('isolated ANN startup timed out')
            ready = parent.recv()
            if ready != ('ready', pair.original.binding_digest, pair.transformed.binding_digest, 'numerical-health-v1'):
                raise RuntimeError(f'isolated ANN startup failed: {ready}')
        except BaseException:
            self._poison()
            # Startup failures have no public owner yet: reclaim the worker
            # here; retain it in quarantine if confirmed teardown fails.
            self.recover()
            raise

    def _poison(self):
        self.failed = True
        self.lease.mark_uncertain()
        _UNCERTAIN_WORKERS.add(self)

    def _ready(self):
        if self.closed or self.failed:
            raise RuntimeError('isolated ANN worker is closed or uncertain')

    def submit(self, value, *, transformed=False):
        with self._lock:
            self._ready()
            if self._pending is not None:
                raise ValueError('isolated ANN already has an outstanding request')
            if type(transformed) is not bool:
                raise ValueError('ANN variant must be boolean')
            value = np.array(value, copy=True, order='C')
            # Shape/domain checks also execute in the owning worker. Bound IPC
            # to the supported ANN envelope; no arbitrary pointer transport.
            from .native_ann import _affine
            shape = _affine(self.pair.logical.original)[1]
            if (value.dtype != np.float32 or value.ndim != 2 or value.size > 512
                    or value.shape != shape or not np.isfinite(value).all()
                    or np.any(np.abs(value.astype(np.float64)) > self.input_bound)):
                raise ValueError('isolated ANN input violates the admitted shape or domain')
            self._next += 1
            self._pending = (self._next, time.monotonic() + self.timeout)
            try:
                self._connection.send(('run', self._next, value, transformed))
            except BaseException:
                self._poison()
                raise
            return self._next

    def poll(self):
        with self._lock:
            self._ready()
            if self._pending is None:
                raise ValueError('isolated ANN has no pending result')
            try:
                if time.monotonic() >= self._pending[1]:
                    raise TimeoutError('isolated ANN completion is uncertain: deadline expired')
                if not self._connection.poll():
                    if self._process.exitcode is not None:
                        raise TimeoutError('isolated ANN completion is uncertain')
                    return None
                message = self._connection.recv()
                if len(message) != 3 or message[0] != 'result' or message[1] != self._pending[0]:
                    raise RuntimeError(f'isolated ANN completion failed: {message}')
                output = message[2]
                from .native_ann import _affine, _output_shape
                expected = _output_shape(_affine(self.pair.logical.original))
                if not isinstance(output, np.ndarray) or output.dtype != np.float32 or output.shape != expected or not np.isfinite(output).all():
                    raise RuntimeError('isolated ANN returned invalid output')
                self._pending = None
                return output
            except BaseException:
                self._poison()
                raise

    def run(self, value, *, transformed=False):
        self.submit(value, transformed=transformed)
        while True:
            output = self.poll()
            if output is not None:
                return output
            time.sleep(.001)

    def replacement(self):
        """Admit a fresh, numerically probed worker after confirmed teardown.

        This tests the selected workload on device zero now; it does not prove
        global driver health or reset a device. Failure never reuses this worker.
        """
        with self._lock:
            if not self.closed or not self.failed or not self.lease.reusable:
                raise ValueError('replacement requires confirmed uncertain-worker teardown')
            return type(self)(self.pair, input_bound=self.input_bound,
                              absolute_budget=self.absolute_budget,
                              timeout_seconds=self.timeout)

    def recover_async(self):
        """Start bounded process teardown; poll_recovery finalizes host ownership."""
        with self._lock:
            if not self.failed:
                raise ValueError('healthy ANN worker does not require recovery')
            if self._recovery is None:
                self._recovery = IsolationRecovery.submit(self.lease, owner=self)
            return self._recovery

    def poll_recovery(self):
        with self._lock:
            if self._recovery is None:
                raise ValueError('no asynchronous recovery admitted')
            if not self._recovery.poll():
                return False
            self._finish_recovery()
            return True

    def _finish_recovery(self):
        self._connection.close()
        self._pending = None
        self.closed = True
        _UNCERTAIN_WORKERS.discard(self)

    def recover(self):
        with self._lock:
            if not self.failed:
                raise ValueError('healthy ANN worker does not require recovery')
            if self._recovery is not None:
                if not self._recovery.poll():
                    raise ValueError('asynchronous recovery is still pending')
            else:
                self.lease.recover()
            self._finish_recovery()

    def close(self):
        with self._lock:
            if self.closed:
                return
            self._ready()
            if self._pending is not None:
                raise ValueError('complete the pending ANN request before close')
            try:
                self._connection.send(('close',))
                if not self._connection.poll(self.timeout) or self._connection.recv() != ('closed',):
                    raise TimeoutError('isolated ANN close is uncertain')
                self._process.join(self.timeout)
                if self._process.exitcode != 0:
                    raise RuntimeError('isolated ANN worker exit is unconfirmed')
                self._connection.close()
                self.closed = True
            except BaseException:
                self._poison()
                raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            with self._lock:
                if self._pending is not None:
                    self._poison()
                if self.failed:
                    self.recover()
                else:
                    self.close()
        except BaseException as cleanup_error:
            # Unconfirmed cleanup remains quarantined, without replacing the
            # exception that caused the caller to leave its ownership scope.
            if exc is None:
                raise
            exc.add_note(f'isolated ANN cleanup incomplete: {cleanup_error}')
        return False
