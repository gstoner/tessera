"""Private final-mark receipts and bounded off-thread pool destruction."""
import threading

_SLOTS = threading.BoundedSemaphore(4)
_LIVE: set["PoolTeardown"] = set()
_LOCK = threading.Lock()


class MarkFinalization:
    def __init__(self, pool, receipt):
        self.pool, self.receipt, self.result = pool, receipt, None

    def poll(self):
        with self.pool._lock:
            if self.result is None:
                self.result = self.receipt.poll()
                if self.result is None:
                    return False
                if self.result[0] not in (0, 2):
                    self.pool._poison_objects()
                    raise RuntimeError('unknown native finalization status')
                self.pool._pending_finalization = None
                self.pool._free_receipts.append(self.receipt)
                if self.result[0] == 0:
                    self.pool._mark_active = False
            if self.result[0] != 0:
                raise ValueError('marking is incomplete or graph closure is invalid')
            return True


class PoolTeardown:
    def __init__(self, pool):
        self.pool = pool
        self.done = threading.Event()
        self.error = None
        self.worker = None

    @classmethod
    def submit(cls, pool):
        if not _SLOTS.acquire(blocking=False):
            raise ValueError('pool teardown worker capacity exhausted')
        ticket = cls(pool)
        pool._teardown = ticket
        with _LOCK:
            _LIVE.add(ticket)
        try:
            threading.Thread(target=ticket._run, daemon=True, name='tessera-heap-teardown').start()
        except BaseException:
            pool._teardown = None
            with _LOCK:
                _LIVE.remove(ticket)
            _SLOTS.release()
            raise
        return ticket

    def _run(self):
        self.worker = threading.get_ident()
        try:
            self.pool.native._enter_unload_context()
            try:
                self.pool.close()
            finally:
                self.pool.native._leave_unload_context()
        except BaseException as error:
            self.error = error
            self.pool._poison_objects()
        finally:
            if self.error is None:
                with _LOCK:
                    _LIVE.remove(self)
                _SLOTS.release()
            self.done.set()

    def poll(self):
        if not self.done.is_set():
            return False
        if self.error is not None:
            raise RuntimeError('uncertain pool teardown retained; process recovery required') from self.error
        return True
