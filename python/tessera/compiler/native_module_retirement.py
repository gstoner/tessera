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
    def __init__(self, owner):
        self.owner = owner
        self.done = threading.Event()
        self.error = None

    @classmethod
    def submit(cls, owner):
        if not _SLOTS.acquire(blocking=False):
            return None
        ticket = cls(owner)
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
            self.owner._enter_unload_context()
            try:
                self.owner._check(self.owner._unload(self.owner._module))
                # Unload succeeded: never retry it if filesystem cleanup fails.
                self.owner._module_unloaded = True
                self.owner._directory.cleanup()
            finally:
                self.owner._leave_unload_context()
        except BaseException as error:
            self.error = error
        finally:
            self.done.set()
            # Failed unloads stay reachable; their outcome cannot authorize a
            # retry or resource reuse. Successful tickets can release the slot.
            if self.error is None:
                with _LOCK:
                    _LIVE.remove(self)
                _SLOTS.release()

    def poll(self):
        if not self.done.is_set():
            return False
        if self.error is not None:
            raise RuntimeError('native module retirement failed; owner retained') from self.error
        return True
