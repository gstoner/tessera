"""Materialize the bounded Apple arena route through the native MLIR compiler.

The native host LLVM companion remains authoritative for dynamic bytes. This
artifact API does not promote a route or dispatch through CUDA/HIP bindings.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from .native_gpu_storage import _run, _block, _decode_image


@dataclass(frozen=True)
class AppleNativeArena:
    msl: str
    host_llvm_ir: str
    sizer: str
    arena_ir: str
    compiler_digest: str

    @property
    def denormal_mode(self):
        values = re.findall(r'tessera.apple.denormal_mode = "([^"]+)"', self.arena_ir)
        return values[0] if len(values) == 1 else 'unspecified'

    @property
    def digest(self):
        import json
        from dataclasses import asdict
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def materialize_apple_arena(source: str, *, compiler: Path, llvm_bin: Path) -> AppleNativeArena:
    arena = _run(compiler, '--allow-unregistered-dialect', '--tessera-tile-buffer-reuse',
                 '--tessera-tile-buffer-arena=emit-apple-msl=true', source=source)
    shaders = re.findall(r'tessera.apple.arena_msl = "((?:\\.|[^"\\])*)"', arena)
    sizers = re.findall(r'tessera.apple.arena_sizer = @([\w]+)', arena)
    if len(shaders) != 1 or len(sizers) != 1:
        raise ValueError('Apple native compiler output lacks its shader/sizer pair')
    host = _block(arena, r'^  func.func @__tessera_shared_bytes_.*?^  }')
    lowered = _run(llvm_bin / 'mlir-opt', '--convert-to-llvm', '--reconcile-unrealized-casts', source=host)
    native = _run(llvm_bin / 'mlir-translate', '--mlir-to-llvmir', source=lowered)
    return AppleNativeArena(_decode_image(shaders[0]).decode(), native, sizers[0], arena,
                            hashlib.sha256(compiler.read_bytes()).hexdigest())


@dataclass(frozen=True)
class AppleArenaPackage:
    """Pinned arm64 companion/bridge plus its compiler-owned shader contract."""
    artifact: AppleNativeArena
    library: bytes
    bridge_digest: str
    binding_digest: str

    @property
    def arena_ir(self):
        return self.artifact.arena_ir

    @property
    def abi(self):
        return _apple_abi(self.artifact)[1]

    def _payload(self):
        import base64
        from dataclasses import asdict
        return dict(artifact=asdict(self.artifact), library=base64.b64encode(self.library).decode(),
                    bridge_digest=self.bridge_digest)

    def _digest(self):
        import json
        return hashlib.sha256(json.dumps(self._payload(), sort_keys=True).encode()).hexdigest()

    def validate(self):
        if not self.library or self._digest() != self.binding_digest:
            raise ValueError('Apple shader/companion package identity disagrees')
        _apple_abi(self.artifact)

    def to_json(self):
        import json
        self.validate()
        return json.dumps(dict(**self._payload(), binding_digest=self.binding_digest), sort_keys=True)

    @classmethod
    def from_json(cls, text: str, *, expected_digest: str):
        import base64
        import json
        data = json.loads(text)
        data['artifact'] = AppleNativeArena(**data['artifact'])
        data['library'] = base64.b64decode(data['library'], validate=True)
        package = cls(**data)
        package.validate()
        if package.binding_digest != expected_digest:
            raise ValueError('Apple arena package differs from pinned identity')
        return package

    def bind(self, *, queue=None):
        return BoundAppleArena(self, queue=queue)


def _apple_abi(artifact):
    mode = artifact.denormal_mode
    if mode not in ('unspecified', 'gradual', 'flush_to_zero'):
        raise ValueError('unsupported Apple denormal policy')
    if mode != 'unspecified' and f'// tessera.denormal_mode={mode}\n' not in artifact.msl:
        raise ValueError('Apple shader denormal policy disagrees with native artifact')
    # This is the native emitter's bounded ABI, not a Graph IR reconstruction.
    kernels = re.findall(r'kernel void (\w+)\(\n(.*?)threadgroup uchar\* arena \[\[threadgroup\(0\)\]\]',
                         artifact.msl, re.S)
    if len(kernels) != 1:
        raise ValueError('Apple arena requires one native shader entry')
    entry, header = kernels[0]
    lines = header.splitlines()
    kinds = []
    for i, line in enumerate(lines):
        match = re.fullmatch(r'(device uchar\*|constant long&) v\d+ \[\[buffer\(' + str(i) + r'\)\]\],', line)
        if not match:
            raise ValueError('Apple arena shader ABI is unsupported')
        kinds.append('pointer' if match[1] == 'device uchar*' else 'index')
    if not kinds or len(kinds) > 31 or not re.fullmatch(r'__tessera_shared_bytes_\w+', artifact.sizer):
        raise ValueError('Apple arena sizing ABI is unsupported')
    return entry, tuple(kinds)


def build_apple_arena_package(artifact: AppleNativeArena) -> AppleArenaPackage:
    """Materialize on the owning Mac; the LLVM compiler owns the size formula."""
    import platform
    import subprocess
    import tempfile
    from dataclasses import replace
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Apple arena packaging requires an arm64 Mac toolchain')
    _apple_abi(artifact)
    bridge = Path(__file__).with_name('apple_arena_bridge.mm')
    with tempfile.TemporaryDirectory(prefix='tessera-apple-arena-build-') as tmp:
        directory = Path(tmp)
        # LLVM 23 annotates intrinsics with this optimization-only promise.
        # Older Apple Clang cannot parse it; omitting the promise is conservative
        # and preserves the compiler-owned sizing program and its pinned source.
        companion = re.sub(r'(?<= )nocreateundeforpoison(?= )', '', artifact.host_llvm_ir)
        (directory / 'sizer.ll').write_text(companion)
        subprocess.run(['xcrun', 'clang', '-O2', '-c', str(directory / 'sizer.ll'),
                        '-o', str(directory / 'sizer.o')], check=True, capture_output=True, timeout=120)
        subprocess.run(['xcrun', 'clang++', '-dynamiclib', '-std=c++17', '-fobjc-arc',
                        '-framework', 'Metal', '-framework', 'Foundation', str(bridge),
                        str(directory / 'sizer.o'), '-o', str(directory / 'arena.dylib')],
                       check=True, capture_output=True, timeout=120)
        package = AppleArenaPackage(artifact, (directory / 'arena.dylib').read_bytes(),
                                   hashlib.sha256(bridge.read_bytes()).hexdigest(), '')
    return replace(package, binding_digest=package._digest())


class BoundAppleArena:
    """Explicit resident-buffer ABI; callers own tensor extents and scheduling.

    No automatic Graph/JIT selection is implied. Calls are synchronous on the
    existing Tessera queue; external concurrent producers must synchronize first.
    """
    def __init__(self, package: AppleArenaPackage, *, queue=None):
        import ctypes as ct
        import platform
        import tempfile
        import threading
        from tessera.runtime import apple_gpu_device_handle, apple_gpu_command_queue_handle
        package.validate()
        if platform.system() != 'Darwin' or platform.machine() != 'arm64':
            raise RuntimeError('Apple arena binding requires an arm64 Mac')
        self.package = package
        entry, self.abi = _apple_abi(package.artifact)
        self._lock = threading.RLock()
        self._handle = None
        self._directory = tempfile.TemporaryDirectory(prefix='tessera-bound-apple-arena-')
        path = Path(self._directory.name) / 'arena.dylib'
        path.write_bytes(package.library)
        self._library = ct.CDLL(str(path))
        self._size = getattr(self._library, '_mlir_ciface_' + package.artifact.sizer)
        self._size.argtypes = [ct.c_void_p if t == 'pointer' else ct.c_int64 for t in self.abi]
        self._size.restype = ct.c_int64
        self._create = self._library.tessera_arena_create
        self._create.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_char_p, ct.c_char_p, ct.c_char_p]
        self._create.restype = ct.c_void_p
        self._launch = self._library.tessera_arena_launch
        self._launch.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint64),
                                ct.POINTER(ct.c_uint64), ct.c_uint, ct.c_int64,
                                ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_uint64), ct.c_char_p]
        self._launch.restype = ct.c_int
        self._destroy = self._library.tessera_arena_destroy
        self._destroy.argtypes = [ct.c_void_p]
        self._destroy.restype = None
        error = ct.create_string_buffer(1024)
        self._queue = queue
        if queue is None:
            self._handle = self._create(apple_gpu_device_handle(), apple_gpu_command_queue_handle(),
                                       package.artifact.msl.encode(), entry.encode(), error)
        else:
            if not isinstance(queue, AppleArenaQueue):
                raise ValueError('Apple arena requires an owning Metal queue')
            with queue._lock:
                if not queue.handle:
                    raise RuntimeError('Metal queue is closed')
                self._handle = self._create(apple_gpu_device_handle(), queue.handle,
                                           package.artifact.msl.encode(), entry.encode(), error)
        if not self._handle:
            self._directory.cleanup()
            raise RuntimeError(error.value.decode())

    def last_device_interval(self):
        """Completed command's Metal GPU timestamps, in seconds; not wall time."""
        import ctypes as ct
        import math
        with self._lock:
            if not self._handle:
                raise RuntimeError('Apple arena binding is closed')
            start, end = ct.c_double(), ct.c_double()
            read = self._library.tessera_arena_last_gpu_interval
            read.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
            read.restype = None
            read(self._handle, ct.byref(start), ct.byref(end))
            if not all(math.isfinite(t) for t in (start.value, end.value)) or not 0 < start.value < end.value:
                raise RuntimeError('Metal command GPU timestamps unavailable')
            return start.value, end.value

    def launch(self, arguments, *, grid, block):
        import ctypes as ct
        from tessera.runtime import DeviceTensor
        with self._lock:
            if not self._handle:
                raise RuntimeError('Apple arena binding is closed')
            arguments = tuple(arguments)  # retain Python owners throughout submission
            if len(arguments) != len(self.abi):
                raise ValueError('Apple arena argument count disagrees')
            for dims in (grid, block):
                if len(dims) != 3 or any(type(v) is not int or not 0 < v <= (1 << 32) - 1 for v in dims):
                    raise ValueError('Apple arena requires positive 32-bit launch dimensions')
            values: list[int] = []
            lengths: list[int] = []
            sizer_args: list[int | None] = []
            buffers = set()
            for kind, value in zip(self.abi, arguments, strict=True):
                if kind == 'index':
                    if type(value) is not int or not 0 <= value < (1 << 63):
                        raise ValueError('Apple arena index must be a nonnegative signed integer')
                    values.append(value)
                    lengths.append(0)
                    sizer_args.append(value)
                else:
                    if not isinstance(value, DeviceTensor) or value._freed or not value._owns or not value.is_metal():
                        raise ValueError('Apple arena requires live owning resident Metal tensors')
                    if value.dtype.str != '<f4' or value.nbytes <= 0:
                        raise ValueError('Apple arena requires nonempty f32 tensors')
                    pointer = value.mtl_buffer()
                    if not pointer or pointer in buffers:
                        raise ValueError('Apple arena buffer is missing or aliased')
                    buffers.add(pointer)
                    values.append(pointer)
                    lengths.append(value.nbytes)
                    # Native sizing must never dereference tensor arguments.
                    sizer_args.append(None)
            size = self._size(*sizer_args)
            if size < 0:
                raise ValueError('Apple arena native sizing rejected the launch')
            count = len(values)
            error = ct.create_string_buffer(1024)
            status = self._launch(self._handle, (ct.c_uint8 * count)(*[int(t == 'index') for t in self.abi]),
                                  (ct.c_uint64 * count)(*values), (ct.c_uint64 * count)(*lengths), count, size,
                                  (ct.c_uint64 * 3)(*grid), (ct.c_uint64 * 3)(*block), error)
            if status:
                raise (TimeoutError if status == 2 else RuntimeError)(error.value.decode())
            return size

    def close(self):
        with self._lock:
            if self._handle:
                self._destroy(self._handle)
                self._handle = None
            self._directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class AppleTensorCall:
    """Typed synchronous Apple binding from the compiler-preserved manifest."""
    def __init__(self, package, signature):
        import threading
        from .native_storage_contract import read_tensor_contract, tensor_contract_specs
        from .native_gpu_tensor import validate_tensor_signature, TensorSpec
        data = read_tensor_contract(package)
        self.specs = tensor_contract_specs(data)
        self.grid, self.block = tuple(data['grid']), tuple(data['block'])
        validate_tensor_signature(package.abi, signature, self.specs, self.grid, self.block)
        from tessera.dtype import canonicalize_dtype
        if any(isinstance(s, TensorSpec) and canonicalize_dtype(s.dtype) != 'fp32' for s in self.specs):
            raise ValueError('Apple native arena admits f32 tensor contracts only')
        self.package, self.signature = package, signature
        self._lock = threading.RLock()
        self._bound = None

    def prepare(self, *args, **kwargs):
        from .native_gpu_tensor import IndexSpec, TensorSpec
        from tessera.runtime import DeviceTensor
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        indices = {}
        for spec in self.specs:
            if isinstance(spec, IndexSpec):
                value = values[spec.name]
                if type(value) is not int or not spec.minimum <= value <= spec.maximum:
                    raise ValueError(f'{spec.name} violates Apple index bounds')
                indices[spec.name] = value
        def resolve(dim):
            return indices[dim] if isinstance(dim, str) else dim
        outputs, pointers = [], set()
        for spec in self.specs:
            if isinstance(spec, TensorSpec):
                value = values[spec.name]
                if not isinstance(value, DeviceTensor) or value._freed or not value._owns:
                    raise ValueError('Apple typed binding requires owning live DeviceTensor arguments')
                if value.dtype.str != '<f4' or any(type(d) is not int for d in value.shape) or value.shape != tuple(resolve(d) for d in spec.shape):
                    raise ValueError(f'{spec.name} Apple tensor shape/dtype disagrees')
                pointer = value.mtl_buffer()
                if not value.is_metal() or not pointer or pointer in pointers:
                    raise ValueError('Apple typed binding requires distinct resident buffers')
                pointers.add(pointer)
                if spec.writable:
                    outputs.append(value)
        grid, block = tuple(map(resolve, self.grid)), tuple(map(resolve, self.block))
        if any(v <= 0 for v in grid + block):
            raise ValueError('Apple launch dimensions must be positive')
        return tuple(values[s.name] for s in self.specs), grid, block, outputs

    def __call__(self, *args, **kwargs):
        with self._lock:
            raw, grid, block, outputs = self.prepare(*args, **kwargs)
            if self._bound is None:
                self._bound = self.package.bind()
            self._bound.launch(raw, grid=grid, block=block)
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def close(self):
        with self._lock:
            if self._bound is not None:
                self._bound.close()
                self._bound = None


class AppleArenaQueue:
    """Owning Metal queue with explicit producer-event/consumer-wait ordering.

    Use a package's own bridge; events are single-generation retained objects.
    A consumer's synchronous launch supplies the bounded completion wait.
    """
    def __init__(self, package, *, runtime_queue=False):
        import ctypes as ct
        from tessera.runtime import apple_gpu_device_handle, apple_gpu_command_queue_handle
        self._owner = package.bind()
        self._library = self._owner._library
        self._release = self._library.tessera_arena_object_release
        self._release.argtypes = [ct.c_void_p]
        self._release.restype = None
        create = self._library.tessera_arena_queue_create
        create.argtypes = [ct.c_void_p]
        create.restype = ct.c_void_p
        self._owned = not runtime_queue
        self.handle = apple_gpu_command_queue_handle() if runtime_queue else create(apple_gpu_device_handle())
        if not self.handle:
            self._owner.close()
            raise RuntimeError('Metal command queue unavailable')
        self._signal = self._library.tessera_arena_queue_signal
        self._signal.argtypes = [ct.c_void_p, ct.c_char_p]
        self._signal.restype = ct.c_void_p
        self._wait = self._library.tessera_arena_queue_wait
        self._wait.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_char_p]
        self._wait.restype = ct.c_int
        import threading
        self._lock = threading.RLock()

    def fill_bytes(self, tensor, value):
        import ctypes as ct
        from tessera.runtime import DeviceTensor
        if not isinstance(tensor, DeviceTensor) or tensor._freed or not tensor._owns or not tensor.is_metal():
            raise ValueError('Metal fill requires a live owning tensor')
        if type(value) is not int or not 0 <= value <= 255:
            raise ValueError('Metal fill value must be a byte')
        fill = self._library.tessera_arena_queue_fill
        fill.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint64, ct.c_uint8, ct.c_char_p]
        fill.restype = ct.c_int
        with self._lock:
            if not self.handle:
                raise RuntimeError('Metal queue is closed')
            error = ct.create_string_buffer(1024)
            if fill(self.handle, tensor.mtl_buffer(), tensor.nbytes, value, error):
                raise RuntimeError(error.value.decode())
            return self.signal()

    def signal(self):
        import ctypes as ct
        with self._lock:
            if not self.handle:
                raise RuntimeError('Metal queue is closed')
            error = ct.create_string_buffer(1024)
            event = self._signal(self.handle, error)
            if not event:
                raise RuntimeError(error.value.decode())
            return AppleArenaFence(self, event)

    def wait_for(self, fence):
        import ctypes as ct
        if not isinstance(fence, AppleArenaFence):
            raise ValueError('Metal queue requires an owning fence')
        # Keep fence release and native wait submission mutually exclusive.
        with fence._lock, self._lock:
            if not self.handle or not fence.handle:
                raise RuntimeError('Metal queue/fence is closed')
            error = ct.create_string_buffer(1024)
            if self._wait(self.handle, fence.handle, error):
                raise RuntimeError(error.value.decode())

    def close(self):
        with self._lock:
            if self.handle and self._owned:
                self._release(self.handle)
            self.handle = None
            self._owner.close()


class AppleArenaFence:
    def __init__(self, queue, handle):
        import threading
        import ctypes as ct
        self._queue, self.handle = queue, handle
        self._release = queue._library.tessera_arena_fence_release
        self._release.argtypes = [ct.c_void_p]
        self._release.restype = None
        self._lock = threading.RLock()

    def close(self):
        with self._lock:
            if self.handle:
                self._release(self.handle)
                self.handle = None
