"""Recovery boundary for device resources owned by an isolated process."""
from __future__ import annotations

import threading
import subprocess


class DriverIsolationLease:
    def __init__(self, process, *, context_identity: str, timeout_seconds: float = 5.0):
        if not context_identity:
            raise ValueError("driver isolation requires a context identity")
        self.process = process
        self.context_identity = context_identity
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
            raise ValueError("driver isolation timeout must be positive")
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

    @property
    def reusable(self) -> bool:
        return self._recovered and self.process.poll() is not None
