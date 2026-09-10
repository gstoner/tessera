"""Bounded admission for off-thread driver unloads after device completion.

Polling never waits for the unload worker. A stalled driver retains its module,
context owner and admission slot; it cannot create an unbounded worker backlog.
The driver operation itself has no cancellation or latency guarantee.
"""
import threading

_SLOTS = threading.BoundedSemaphore(8)
_LIVE: set["ModuleRetirement"] = set()
_LOCK = threading.Lock()


class ModuleRetirement:
    def __init__(self, owner, *, owns_slot=False):
        self.owner = owner
        self.done = threading.Event()
        self.error = None
        self.phase = 'queued'
        self.context_exit_error = None
        self._cleanup_retryable = False
        self._retry_lock = threading.Lock()
        self._slot = _SLOTS
        self._owns_slot = owns_slot
        self.isolation = getattr(owner, '_isolation_lease', None)

    @classmethod
    def retain_failure(cls, owner, error, phase):
        """Record an uncertain synchronous outcome without starting a worker."""
        ticket=cls(owner)
        ticket.error=error
        ticket.phase=phase
        with _LOCK:
            _LIVE.add(ticket)
        ticket.done.set()
        return ticket

    @classmethod
    def submit(cls, owner):
        if not _SLOTS.acquire(blocking=False):
            return None
        ticket = cls(owner, owns_slot=True)
        with _LOCK:
            _LIVE.add(ticket)
        try:
            threading.Thread(target=ticket._run, daemon=True, name='tessera-module-retirement').start()
        except BaseException:
            with _LOCK:
                _LIVE.remove(ticket)
            _SLOTS.release()
            raise
        return ticket

    def _run(self):
        try:
            self.phase = 'context_enter'
            self.owner._enter_unload_context()
            try:
                self.phase = 'driver_unload'
                self.owner._check(self.owner._unload(self.owner._module))
                # Unload succeeded: never retry it if filesystem cleanup fails.
                self.owner._module_unloaded = True
            except BaseException as error:
                self.error = error
            finally:
                try:
                    if self.error is None:
                        self.phase = 'context_exit'
                    self.owner._leave_unload_context()
                except BaseException as error:
                    self.context_exit_error = error
                    if self.error is None:
                        raise
            if self.error is not None:
                return
            # No driver action remains. Only this phase may be safely retried.
            self._cleanup_retryable = True
            self.phase = 'filesystem_cleanup'
            self.owner._directory.cleanup()
            self._cleanup_retryable = False
        except BaseException as error:
            self.error = error
        finally:
            # Failed unloads stay reachable; their outcome cannot authorize a
            # retry or resource reuse. Successful tickets can release the slot.
            with self._retry_lock:
                if self.error is None:
                    with _LOCK:
                        _LIVE.remove(self)
                    if self._owns_slot:
                        self._slot.release()
                self.done.set()

    def retry_cleanup(self):
        """Retry filesystem cleanup off-thread, never a failed driver action."""
        with self._retry_lock:
            if not self.done.is_set() or self.error is None or not self._cleanup_retryable:
                raise ValueError('only completed post-unload cleanup failures are retryable')
            previous=self.error
            self.error=None
            self.done.clear()
            def cleanup():
                try:
                    self.owner._directory.cleanup()
                    self._cleanup_retryable=False
                except BaseException as error:
                    self.error=error
                finally:
                    with self._retry_lock:
                        if self.error is None:
                            with _LOCK:_LIVE.remove(self)
                            if self._owns_slot:
                                self._slot.release()
                        self.done.set()
            try:
                threading.Thread(target=cleanup,daemon=True,name='tessera-module-cleanup').start()
            except BaseException:
                self.error=previous
                self.done.set()
                raise
        return self

    def recover_isolation(self):
        """Replace an uncertain context only after its owning process dies."""
        with self._retry_lock:
            if not self.done.is_set() or self.error is None or self.isolation is None:
                raise ValueError('uncertain retirement has no isolation recovery boundary')
            self.isolation.mark_uncertain()
            self.isolation.recover()
            self.owner._module = type(self.owner._module)()
            self.owner._directory.cleanup()
            self.error = None
            self.phase = 'isolation_recovered'
            with _LOCK:
                _LIVE.discard(self)
            if self._owns_slot:
                self._slot.release()
                self._owns_slot = False
            return True

    def poll(self):
        with self._retry_lock:
            if not self.done.is_set():
                return False
            if self.error is not None:
                detail = '; context exit also failed' if self.context_exit_error is not None and self.phase == 'driver_unload' else ''
                raise RuntimeError(f'native module retirement failed in {self.phase}{detail}; owner retained') from self.error
            return True
