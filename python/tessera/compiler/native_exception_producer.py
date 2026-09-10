"""Native host exception allocation ABI; no Python allocator fallback.

The C++ runtime owns storage, generations, roots and collection. This bounded
producer does not implement GPU allocation or automatic source-exception lowering.
"""
import ctypes as ct
import threading


class NativeExceptionProducer:
    def __init__(self, library, *, capacity=16, payload_capacity=1024):
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        self._library = ct.CDLL(str(library))
        p, u, h = ct.c_void_p, ct.c_uint32, ct.c_uint64
        self._functions = {}
        signatures = {
            'create': [u,u,ct.POINTER(p)], 'destroy': [p],
            'alloc': [p,u,p,u,h,h,ct.c_int,ct.POINTER(h)],
            'edges': [p,h,h,h], 'root': [p,h,ct.c_int],
            'collect': [p,ct.POINTER(u)],
            'read': [p,h,ct.POINTER(u),ct.POINTER(h),ct.POINTER(h),p,u,ct.POINTER(u)],
        }
        for name, args in signatures.items():
            fn = getattr(self._library, 'tsr_exception_heap_' + name)
            fn.argtypes, fn.restype = args, None if name == 'destroy' else ct.c_int
            self._functions[name] = fn
        self._check(self._functions['create'](self._uint(capacity,65536),
                    self._uint(payload_capacity,1<<28),ct.byref(self._handle)))

    @staticmethod
    def _uint(value, limit=(1<<64)-1):
        if type(value) is not int or not 0 <= value <= limit:
            raise ValueError('invalid native exception integer')
        return value

    @staticmethod
    def _check(status):
        if status == 1:
            raise ValueError('invalid native exception argument or stale handle')
        if status in (2,3):
            raise MemoryError('native exception heap capacity exhausted')
        if status:
            raise RuntimeError('unknown native exception heap status')

    def _call(self, name, *args):
        if not self._handle.value:
            raise ValueError('native exception producer is closed')
        self._check(self._functions[name](self._handle,*args))

    def allocate(self, kind, payload, *, cause=0, context=0, root=False):
        if not isinstance(payload,bytes) or type(root) is not bool:
            raise ValueError('native exception payload/root must be bytes/bool')
        with self._lock:
            handle = ct.c_uint64()
            self._call('alloc',self._uint(kind,(1<<32)-1),payload,
                       self._uint(len(payload),(1<<32)-1),self._uint(cause),
                       self._uint(context),int(root),ct.byref(handle))
            return handle.value

    def set_edges(self, handle, *, cause=0, context=0):
        with self._lock:
            self._call('edges',self._uint(handle),self._uint(cause),self._uint(context))

    def root(self, handle, enabled=True):
        if type(enabled) is not bool:
            raise ValueError('native exception root must be bool')
        with self._lock:
            self._call('root',self._uint(handle),int(enabled))

    def collect(self):
        with self._lock:
            count = ct.c_uint32()
            self._call('collect',ct.byref(count))
            return count.value

    def read(self, handle):
        with self._lock:
            handle = self._uint(handle)
            kind,size = ct.c_uint32(),ct.c_uint32()
            cause,context = ct.c_uint64(),ct.c_uint64()
            if not self._handle.value:
                raise ValueError('native exception producer is closed')
            status = self._functions['read'](self._handle,handle,ct.byref(kind),ct.byref(cause),
                                            ct.byref(context),None,0,ct.byref(size))
            if status != 2:
                self._check(status)
            payload = ct.create_string_buffer(size.value)
            self._call('read',handle,ct.byref(kind),ct.byref(cause),ct.byref(context),
                       payload,size.value,ct.byref(size))
            return kind.value, payload.raw[:size.value], cause.value, context.value

    def close(self):
        with self._lock:
            if self._handle.value:
                self._functions['destroy'](self._handle)
                self._handle = ct.c_void_p()

    def __enter__(self):
        if not self._handle.value:
            raise ValueError('native exception producer is closed')
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        if getattr(self,'_handle',None) and self._handle.value:
            self.close()
