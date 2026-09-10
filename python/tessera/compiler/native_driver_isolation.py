"""Recovery boundary for device resources owned by an isolated process."""
from __future__ import annotations

import threading
import math
import subprocess


class DriverIsolationLease:
    def __init__(self, process, *, context_identity: str, timeout_seconds: float = 5.0):
        if not context_identity:
            raise ValueError("driver isolation requires a context identity")
        self.process = process
        self.context_identity = context_identity
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("driver isolation timeout must be finite and positive")
        self.timeout_seconds = float(timeout_seconds)
        self._uncertain = False
        self._recovered = False
        self._lock = threading.Lock()

    def mark_uncertain(self) -> None:
        with self._lock:
            self._uncertain = True

    def recover(self) -> None:
        """Destroy the isolation boundary once; process death is reclamation proof."""
        with self._lock:
            if not self._uncertain:
                raise ValueError("driver isolation recovery requires an uncertain outcome")
            if self._recovered:
                return
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=self.timeout_seconds)
            if self.process.poll() is None:
                raise RuntimeError("driver isolation process did not terminate")
            self._recovered = True

    def reconcile_death(self) -> bool:
        """Observe a late exit without retrying termination or driver operations."""
        with self._lock:
            if not self._uncertain:
                raise ValueError("driver isolation recovery requires an uncertain outcome")
            if self._recovered:
                return True
            if self.process.poll() is None:
                return False
            self._recovered = True
            return True

    @property
    def reusable(self) -> bool:
        return self._recovered and self.process.poll() is not None


_RECOVERY_SLOTS = threading.BoundedSemaphore(8)
_RECOVERIES: set[IsolationRecovery] = set()
_RECOVERY_LOCK = threading.Lock()


class IsolationRecovery:
    """Bounded off-thread process teardown, retaining dependent owners on failure.

    A completed ticket proves process death only, never device health. Failed
    tickets retain their admission slot and owners; repeated failures cannot
    create an unlimited teardown backlog. No driver operation is retried.
    """
    def __init__(self, lease, owner):
        self.lease, self.owner = lease, owner
        self._slot = _RECOVERY_SLOTS
        self.done = threading.Event()
        self.error = None
        self._finalize_lock = threading.Lock()
        self._released = False

    @classmethod
    def submit(cls, lease, *, owner):
        if not _RECOVERY_SLOTS.acquire(blocking=False):
            return None
        ticket = cls(lease, owner)
        with _RECOVERY_LOCK:
            _RECOVERIES.add(ticket)
        try:
            threading.Thread(target=ticket._run, daemon=True,
                             name='tessera-isolation-recovery').start()
        except BaseException:
            with _RECOVERY_LOCK:
                _RECOVERIES.remove(ticket)
            _RECOVERY_SLOTS.release()
            raise
        return ticket

    def _run(self):
        try:
            self.lease.recover()
        except BaseException as error:
            self.error = error
        finally:
            if self.error is None:
                self._release_owner()
            self.done.set()

    def _release_owner(self):
        with self._finalize_lock:
            if not self._released:
                self.owner = None
                self.error = None
                with _RECOVERY_LOCK:
                    _RECOVERIES.discard(self)
                self._slot.release()
                self._released = True

    def poll(self):
        if not self.done.is_set():
            return False
        with self._finalize_lock:
            if self._released:
                return True
        if self.error is not None:
            if not self.lease.reconcile_death():
                raise RuntimeError('isolation teardown unconfirmed; owner retained') from self.error
            self._release_owner()
        return True
