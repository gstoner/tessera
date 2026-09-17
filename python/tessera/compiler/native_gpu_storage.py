"""Bound native GPU image + checked host sizing companion, without Graph re-entry.

This raw device-pointer interface owns neither tensor allocation nor scheduling.
Synchronous calls and event-owned stream submissions consume one immutable pair; the native GPU
loader enforces image compatibility with the caller's current device/context.
"""
from __future__ import annotations

import base64
import ctypes as ct
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading

# Retain asynchronous buffer owners even if callers drop their completion handle.
# Explicit wait/close retires entries; forgetting both retains storage safely.
_LIVE_SUBMISSIONS: set = set()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# `llvm-objdump` is the kernel-body check's disassembler (below); it was missing
# from this set, so a caller's `/usr/lib/llvm-23/bin/llvm-objdump` was used as
# given and raised FileNotFoundError on Tajasarus, whose LLVM lives in a
# toolchain prefix (review of 2026-09-17).
_LLVM_COMPANIONS = frozenset({"mlir-opt", "mlir-translate", "llc", "llvm-link", "clang", "clang++", "llvm-objdump"})


def _resolve_tool(tool: Path) -> Path:
    """A caller-supplied LLVM companion path that does not exist on this host
    (callers historically pass the Ubuntu ``/usr/lib/llvm-23/bin``) resolves to
    the fleet-matched LLVM 23 tool of the same name; anything else is used as
    given so a wrong compiler path still fails loudly."""
    tool = Path(tool)
    if tool.exists() or tool.name not in _LLVM_COMPANIONS:
        return tool
    from .llvm_tools import find_llvm_tool
    found = find_llvm_tool(tool.name)
    return found if found is not None else tool


def _run(tool: Path, *args: str, source: str | None = None) -> str:
    return subprocess.check_output([str(_resolve_tool(tool)), *args], input=source, text=True,
                                   stderr=subprocess.PIPE, timeout=120)


def _block(text: str, pattern: str) -> str:
    blocks = re.findall(pattern, text, re.M | re.S)
    if len(blocks) != 1:
        raise ValueError('native storage package requires one kernel and sizing companion')
    return 'module {\n' + blocks[0] + '\n}\n'


def _decode_image(literal: str) -> bytes:
    simple = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"'}
    def decode(match):
        escape = match[1]
        if len(escape) == 2:
            return chr(int(escape, 16))
        if escape not in simple:
            raise ValueError('invalid native image escape')
        return simple[escape]
    return re.sub(r'\\([0-9A-Fa-f]{2}|.)', decode, literal).encode('latin1')


@dataclass(frozen=True)
class NativeGPUStoragePackage:
    backend: str
    chip: str
    entry: str
    sizer: str
    abi: tuple[str, ...]
    arena_ir: str
    image: bytes
    host_library: bytes
    compiler_digest: str
    llvm_digest: str
    binding_digest: str

    def _payload(self) -> dict:
        data = asdict(self)
        del data['binding_digest']
        for key in ('image', 'host_library'):
            data[key] = base64.b64encode(data[key]).decode('ascii')
        return data

    def _digest(self) -> str:
        return _sha(json.dumps(self._payload(), sort_keys=True, separators=(',', ':')).encode())

    def validate(self) -> None:
        if (self.backend, self.chip) not in (('nvidia', 'sm_120'), ('rocm', 'gfx1151'), ('rocm', 'gfx1201')) or not self.image or not self.host_library:
            raise ValueError('invalid native storage package binding')
        if not self.abi or any(t not in ('pointer', 'index') for t in self.abi):
            raise ValueError('unsupported native storage ABI')
        if self._digest() != self.binding_digest:
            raise ValueError('native kernel/sizer package binding disagrees')

    def to_json(self) -> str:
        self.validate()
        return json.dumps({**self._payload(), 'binding_digest': self.binding_digest}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str, *, expected_digest: str) -> NativeGPUStoragePackage:
        data = json.loads(text)
        for key in ('image', 'host_library'):
            data[key] = base64.b64decode(data[key], validate=True)
        data['abi'] = tuple(data['abi'])
        package = cls(**data)
        package.validate()
        if package.binding_digest != expected_digest:
            raise ValueError('native storage package differs from pinned identity')
        return package

    def bind(self) -> BoundNativeGPUStorage:
        self.validate()
        return BoundNativeGPUStorage(self)


# The one arena pipeline every native GPU storage package is built with and
# every consumer replays against. Since 100a2980 the packager also expands
# low-precision conversions before canonicalization; six consumers kept a
# hand-copied four-pass replay and refused every package on device with
# "disagrees with native replay" (found 2026-09-15). Spell the pipeline here
# only -- tests/unit/test_arena_replay_pipeline.py fails a hand copy.
ARENA_PIPELINE = ('--allow-unregistered-dialect', '--tessera-tile-buffer-reuse',
                  '--tessera-tile-buffer-arena', '--tessera-expand-lowp-conversions', '--canonicalize')


def replay_arena_ir(compiler: Path, source: str) -> str:
    """Replay ``source`` through the packager's arena pipeline for validation."""
    return _run(Path(compiler), *ARENA_PIPELINE, source=source)


def build_native_gpu_storage(source: str, *, compiler: Path, llvm_bin: Path,
                             backend: str, chip: str, toolkit: Path | None = None) -> NativeGPUStoragePackage:
    if (backend, chip) not in (('nvidia', 'sm_120'), ('rocm', 'gfx1151'), ('rocm', 'gfx1201')):
        raise ValueError('native storage target is not validated')
    if re.search(r'\btessera\.denormal_mode\s*=', source):
        raise ValueError('explicit denormal policy currently requires the Apple arena consumer')
    binary_pass = _binary_pass(toolkit)
    arena = replay_arena_ir(compiler, source)
    device = _block(arena, r'^  gpu.module .*?^  }')
    host = _block(arena, r'^  func.func @__tessera_shared_bytes_.*?^  }')
    signatures = re.findall(r'gpu.func @([\w]+)\(([^)]*)\) kernel', device)
    if len(signatures) != 1:
        raise ValueError('native storage package requires exactly one kernel')
    entry, arguments = signatures[0]
    types = [arg.split(':', 1)[1].strip() for arg in arguments.split(',')]
    if any(t not in ('!llvm.ptr<1>', 'index') for t in types):
        raise ValueError('native storage ABI admits device pointers and index scalars only')
    symbol = re.search(r'tile.dynamic_shared_size = @([\w]+)', device)
    if symbol is None or f'func.func @{symbol[1]}(' not in host:
        raise ValueError('kernel lacks its compiler-generated sizing companion')
    if backend == 'rocm' and 'nvgpu.' in device:
        raise ValueError('NVGPU producer has no ROCm lowering')
    target = 'nvvm' if backend == 'nvidia' else 'rocdl'
    metadata = 'expand-strided-metadata,lower-affine,' if 'memref.subview' in device else ''
    pipeline = ('builtin.module(gpu.module(' + metadata + 'convert-nvgpu-to-nvvm,convert-scf-to-cf,'
                f'convert-gpu-to-{target},convert-math-to-llvm,reconcile-unrealized-casts),'
                f'{target}-attach-target{{chip={chip}}},{binary_pass})')
    binary = _run(llvm_bin / 'mlir-opt', '--pass-pipeline=' + pipeline, source=device)
    if binary.count('#gpu.object<') != 1:
        raise ValueError('expected exactly one native GPU image')
    literals = re.findall(r'"((?:\\.|[^"\\])*)"', binary)
    encoded = re.search(r'bin = "((?:\\.|[^"\\])*)"', binary)
    literal = encoded[1] if encoded else literals[-1]
    image = _decode_image(literal)
    _reject_bodyless_image(image, backend, entry, llvm_bin)
    lowered = _run(llvm_bin / 'mlir-opt', '--convert-to-llvm', '--reconcile-unrealized-casts', source=host)
    native = _run(llvm_bin / 'mlir-translate', '--mlir-to-llvmir', source=lowered)
    with tempfile.TemporaryDirectory(prefix='tessera-native-sizer-') as tmp:
        directory = Path(tmp)
        (directory / 'sizer.ll').write_text(native)
        _run(llvm_bin / 'clang', '-shared', '-fPIC', '-O2', str(directory / 'sizer.ll'),
             '-o', str(directory / 'sizer.so'))
        host_library = (directory / 'sizer.so').read_bytes()
    package = NativeGPUStoragePackage(backend, chip, entry, symbol[1],
        tuple('pointer' if t.startswith('!llvm.ptr') else 'index' for t in types),
        arena, image, host_library, _sha(compiler.read_bytes()),
        _sha(_resolve_tool(llvm_bin / 'mlir-opt').read_bytes()), '')
    return NativeGPUStoragePackage(**{**asdict(package), 'binding_digest': package._digest()})


def _reject_bodyless_image(image: bytes, backend: str, entry: str, llvm_bin: Path) -> None:
    """Refuse an image whose kernel does not store anything.

    Found 2026-09-16 on gfx1151: a `math.tanh` row program serialized to a kernel
    whose entire body was one `s_endpgm`. The launch succeeded, wrote nothing,
    and the caller read back whatever was in the output buffer -- all zeros, a
    plausible-looking answer. The same module serialized with `format=isa`
    contained the correct implementation, so the body was lost in the binary
    path; `__ocml_tanh_f32` is the trigger, and nothing in the packager noticed.

    Every kernel in this ABI writes at least one output buffer, so "the
    disassembly contains no global store" is a sound refusal rather than a
    heuristic. Only the AMDGPU image is checked: `llvm-objdump` from the matched
    LLVM disassembles amdgcn, and the NVIDIA cubin needs `nvdisasm`, which is not
    a matched-LLVM tool -- so the NVVM route is *not* covered by this guard and
    an equivalent silent loss there would still ship. That gap is named in
    `docs/audit/backend/nvidia/todo.md` rather than papered over.
    """
    if backend != 'rocm':
        return
    objdump = _resolve_tool(llvm_bin / 'llvm-objdump')
    with tempfile.TemporaryDirectory(prefix='tessera-native-image-') as tmp:
        path = Path(tmp) / 'kernel.hsaco'
        path.write_bytes(image)
        result = subprocess.run([str(objdump), '-d', '--triple=amdgcn-amd-amdhsa', str(path)],
                                capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError('native storage package: the image could not be disassembled for validation: '
                         + result.stderr.strip()[:400])
    body = [line for line in result.stdout.splitlines() if line.startswith('\t')]
    if not any('global_store' in line or 'buffer_store' in line or 'flat_store' in line for line in body):
        raise ValueError(
            f'native storage package: the packaged kernel @{entry} stores nothing ('
            f'{len(body)} instructions), so a launch would write no output and the caller would '
            'read back whatever was in the buffer. A device-library call whose body the binary '
            'serialization drops produces exactly this image; serialize the same module with '
            'gpu-module-to-binary{format=isa} to see whether the implementation survives there.')


def _binary_pass(toolkit: Path | None) -> str:
    """Pin serialization to the requested toolkit without pass-option injection."""
    if toolkit is None:
        return 'gpu-module-to-binary'
    path = toolkit.resolve(strict=True)
    if not path.is_dir() or re.fullmatch(r'[/A-Za-z0-9_.+-]+', str(path)) is None:
        raise ValueError('toolkit must be a directory with a plain absolute path')
    return f'gpu-module-to-binary{{toolkit={path}}}'


class BoundNativeGPUStorage:
    """Raw-pointer launches on the caller's active native context."""
    def __init__(self, package: NativeGPUStoragePackage):
        package.validate()
        self.package = package
        self._directory = tempfile.TemporaryDirectory(prefix='tessera-bound-sizer-')
        path = Path(self._directory.name) / 'sizer.so'
        path.write_bytes(package.host_library)
        self._library = ct.CDLL(str(path))
        self._size = getattr(self._library, package.sizer)
        self._types = tuple(ct.c_void_p if t == 'pointer' else ct.c_int64 for t in package.abi)
        self._size.argtypes, self._size.restype = list(self._types), ct.c_int64
        cuda = package.backend == 'nvidia'
        self._driver = ct.CDLL('libcuda.so.1' if cuda else 'libamdhip64.so')
        P, U = ct.c_void_p, ct.c_uint

        def bind(cu: str, hip: str, types: list):
            fn = getattr(self._driver, cu if cuda else hip)
            fn.argtypes, fn.restype = types, ct.c_int
            return fn

        context_type = P if cuda else ct.c_int
        context = context_type()
        self._check(bind('cuCtxGetCurrent', 'hipGetDevice', [ct.POINTER(context_type)])(ct.byref(context)))
        set_context = bind('cuCtxSetCurrent', 'hipSetDevice', [context_type])
        self._enter_unload_context = lambda: self._check(set_context(context))
        self._leave_unload_context = (lambda: self._check(set_context(P()))) if cuda else (lambda: None)
        self._module_retirement = None
        self._closing = False
        self._unload = bind('cuModuleUnload', 'hipModuleUnload', [P])
        self._sync = bind('cuCtxSynchronize', 'hipDeviceSynchronize', [])
        self._launch = bind('cuLaunchKernel', 'hipModuleLaunchKernel', [P] + [U] * 7 + [P, ct.POINTER(P), ct.POINTER(P)])
        self._stream_sync = bind('cuStreamSynchronize', 'hipStreamSynchronize', [P])
        self._event_create = bind('cuEventCreate', 'hipEventCreateWithFlags', [ct.POINTER(P), U])
        self._event_record = bind('cuEventRecord', 'hipEventRecord', [P, P])
        self._event_sync = bind('cuEventSynchronize', 'hipEventSynchronize', [P])
        self._event_query = bind('cuEventQuery', 'hipEventQuery', [P])
        self._event_destroy = bind('cuEventDestroy_v2', 'hipEventDestroy', [P])
        self._stream_wait = bind('cuStreamWaitEvent', 'hipStreamWaitEvent', [P, P, U])
        self._pending: list[NativeSubmission] = []
        self._untracked_inflight = False
        self._lock = threading.RLock()
        self._module, self._function = P(), P()
        blob = ct.create_string_buffer(package.image)
        self._check(bind('cuModuleLoadData', 'hipModuleLoadData', [ct.POINTER(P), P])(ct.byref(self._module), ct.cast(blob, P)))
        try:
            self._check(bind('cuModuleGetFunction', 'hipModuleGetFunction', [ct.POINTER(P), P, ct.c_char_p])(
                ct.byref(self._function), self._module, package.entry.encode()))
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check(status: int) -> None:
        if status:
            raise RuntimeError(f'native GPU driver status {status}')

    def _arguments(self, arguments: tuple[int, ...]) -> list:
        if len(arguments) != len(self._types):
            raise ValueError('native kernel argument count disagrees')
        for kind, value in zip(self.package.abi, arguments, strict=True):
            if type(value) is not int or value < 0 or value > ((1 << 63) - 1):
                raise ValueError('native arguments require nonnegative representable integers')
            if kind == 'pointer' and value == 0:
                raise ValueError('native device pointer must be non-null')
        return [t(v) for t, v in zip(self._types, arguments, strict=True)]

    def _launch_size(self, arguments, grid, block):
        if not self._module or getattr(self, '_closing', False):
            raise ValueError('native storage binding is closed or retiring')
        values = self._arguments(arguments)
        if len(grid) != 3 or len(block) != 3 or any(type(v) is not int or not 0 < v < (1 << 32) for v in grid + block):
            raise ValueError('invalid native launch geometry')
        count = self._size(*values)
        if count < 0 or count > (1 << 31) - 1:
            raise ValueError('native sizing companion rejected the launch extent')
        return values, count

    def submit(self, arguments: tuple[int, ...], *, grid: tuple[int, int, int],
               block: tuple[int, int, int], stream: int,
               producer_streams: tuple[int, ...] = (), keepalive: tuple = ()) -> NativeSubmission:
        """Enqueue on a caller-owned stream; the returned event owns completion.

        Raw pointer callers must order overlapping accesses themselves. Tensor
        bindings add allocation-conflict dependencies before using this method.
        """
        if type(stream) is not int or not 0 < stream < (1 << 64):
            raise ValueError('submission requires a non-null native stream')
        if any(type(s) is not int or not 0 < s < (1 << 64) for s in producer_streams):
            raise ValueError('invalid producer stream')
        with self._lock:
            values, count = self._launch_size(arguments, grid, block)
            events = []
            try:
                for producer in sorted(set(producer_streams) - {stream}):
                    event = ct.c_void_p()
                    self._check(self._event_create(ct.byref(event), 2))  # disable timing
                    events.append(event)
                    self._check(self._event_record(event, ct.c_void_p(producer)))
                    self._check(self._stream_wait(ct.c_void_p(stream), event, 0))
                done = ct.c_void_p()
                self._check(self._event_create(ct.byref(done), 2))
                events.append(done)
                P = ct.c_void_p
                argv = (P * len(values))(*(ct.cast(ct.byref(v), P) for v in values))
                self._check(self._launch(self._function, *grid, *block, count, P(stream), argv, None))
                self._check(self._event_record(done, P(stream)))
                ticket = NativeSubmission(self, done, events, keepalive, count)
                self._pending.append(ticket)
                return ticket
            except BaseException:
                # Recording may fail after a launch. Retain owners until the
                # stream itself proves completion, even if that wait fails too.
                ticket = NativeSubmission(self, None, events, keepalive, count, stream)
                self._pending.append(ticket)
                ticket.wait()
                raise

    def launch(self, arguments: tuple[int, ...], *, grid: tuple[int, int, int],
               block: tuple[int, int, int]) -> int:
        with self._lock:
            values, count = self._launch_size(arguments, grid, block)
            P = ct.c_void_p
            argv = (P * len(values))(*(ct.cast(ct.byref(v), P) for v in values))
            self._untracked_inflight = True
            self._check(self._launch(self._function, *grid, *block, count, None, argv, None))
            self._check(self._sync())
            self._untracked_inflight = False
            return count

    def retry_retirement_cleanup(self):
        """Retry only known post-unload filesystem failures without blocking."""
        with self._lock:
            retirement=getattr(self,'_module_retirement',None)
            if retirement is None:raise ValueError('native module has no retirement ticket')
            retirement.retry_cleanup()

    def _unload_synchronously(self, *, synchronize):
        # Once a driver action starts, failure does not authorize a retry.
        # Keep an externally reachable ticket even if the caller drops self.
        self._closing=True
        phase='device_synchronize' if synchronize else 'driver_unload'
        try:
            if synchronize:
                self._check(self._sync())
                self._untracked_inflight=False
            phase='driver_unload'
            self._check(self._unload(self._module))
            self._module=ct.c_void_p()
        except BaseException as error:
            from .native_module_retirement import ModuleRetirement
            self._module_retirement=ModuleRetirement.retain_failure(self,error,phase)
            raise

    def close_if_complete(self, *, defer_unload=False) -> bool:
        """Unload only after every tracked launch proves completion by query.

        Used by scoped owners that never export untracked kernel work. Driver
        module unload itself is not claimed to have bounded host latency.
        """
        with self._lock:
            retirement = getattr(self, '_module_retirement', None)
            if retirement is not None:
                if not retirement.poll():
                    return False
                self._module = ct.c_void_p()
                return True
            if getattr(self,"_untracked_inflight",False):
                return False
            if not all(ticket.poll() for ticket in tuple(self._pending)):
                return False
            if self._module and (defer_unload or getattr(self, '_closing', False)):
                from .native_module_retirement import ModuleRetirement
                self._closing = True
                self._module_retirement = ModuleRetirement.submit(self)
                return False
            if self._module:
                self._unload_synchronously(synchronize=False)
            self._directory.cleanup()
            return True

    def close(self) -> None:
        with self._lock:
            if getattr(self, '_closing', False):
                if not self.close_if_complete(defer_unload=True):
                    raise RuntimeError('native module retirement is pending; poll completion')
                return
            for ticket in tuple(self._pending):
                ticket.wait()
            if self._module:
                self._unload_synchronously(synchronize=True)
            self._directory.cleanup()

    def __enter__(self) -> BoundNativeGPUStorage:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class NativeSubmission:
    """A native completion event retaining buffers until completion is observed."""
    def __init__(self, owner, event, events, keepalive, dynamic_bytes, stream=None):
        self._owner, self._event, self._events = owner, event, events
        self._keepalive = keepalive
        self.dynamic_bytes = dynamic_bytes
        self.done = False
        self._completed = False
        self._stream = stream
        _LIVE_SUBMISSIONS.add(self)

    def wait_on(self, stream: int) -> None:
        if type(stream) is not int or not 0 < stream < (1 << 64):
            raise ValueError('dependency requires a non-null native stream')
        with self._owner._lock:
            if not self.done and not self._completed:
                if self._event is None:
                    raise RuntimeError("dependency completion is unproven; explicit wait or isolated recovery required")
                else:
                    self._owner._check(self._owner._stream_wait(ct.c_void_p(stream), self._event, 0))

    def poll(self) -> bool:
        """Observe completion without blocking; errors retain all owners."""
        with self._owner._lock:
            if self.done:
                return True
            if self._completed:
                self._finish_completion()
                return True
            if self._event is None:
                return False  # Failed event recording needs an explicit wait.
            status = self._owner._event_query(self._event)
            if status == 600:  # CUDA_ERROR_NOT_READY / hipErrorNotReady
                return False
            self._owner._check(status)
            self._completed = True
            self._finish_completion()  # Query success is already a completion proof.
            return True

    def wait(self) -> int:
        with self._owner._lock:
            if not self.done:
                if not self._completed:
                    if self._event is None:
                        self._owner._check(self._owner._stream_sync(ct.c_void_p(self._stream)))
                    else:
                        self._owner._check(self._owner._event_sync(self._event))
                    self._completed = True
                self._finish_completion()
            return self.dynamic_bytes

    def _finish_completion(self):
        # Caller holds the owner lock and has observed successful completion.
        # Failed event destruction retains the remaining events and resources.
        while self._events:
            self._owner._check(self._owner._event_destroy(self._events[-1]))
            self._events.pop()
        self._keepalive = ()
        self.done = True
        self._owner._pending.remove(self)
        _LIVE_SUBMISSIONS.discard(self)
