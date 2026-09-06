"""Compiler-produced primal/derivative storage pairs, with explicit AD lineage."""
import hashlib
import inspect
import json
import re
import threading
from .native_gpu_storage import _run, _decode_image, build_native_gpu_storage
from .native_gpu_tensor import TensorSpec, IndexSpec
from .native_storage_contract import attach_tensor_contract, generate_tensor_binding


def materialize_storage_pair(source, *, mode, compiler, llvm_bin, backend, chip=None):
    """Produce a typed native package (or Apple export) from a fresh AD request."""
    if mode not in ('forward', 'reverse'):
        raise ValueError('native pair mode must be forward or reverse')
    if 'tessera.frontend.authority = "tracer"' not in source:
        raise ValueError('native pair requires tracer-owned source')
    flag = 'forward' if mode == 'forward' else 'paired'
    prefix = 'native_jvp_' if mode == 'forward' else 'native_vjp_'
    generated = _run(compiler, '--allow-unregistered-dialect',
                     '--tessera-autodiff-' + flag + '=emit-storage-child=true', source=source)
    def integer(field):
        match = re.search(r'tessera\.' + prefix + field + r' = (\d+) : i64', generated)
        if not match:
            raise ValueError('native pair lacks compiler ABI dimensions')
        return int(match[1])
    width, count = integer('width'), integer('inputs')
    def widths(field, count):
        match = re.search(r'tessera\.' + prefix + field + r' = \[([^\]]*)\]', generated)
        if match is None:
            raise ValueError('native pair lacks per-argument widths')
        values = [int(v) for v in re.findall(r'(\d+)(?: : i64)?', match[1])]
        if len(values) != count or any(v not in (1, width) for v in values):
            raise ValueError('native pair widths disagree')
        return values
    input_widths, output_widths = widths('input_widths', count), widths('output_widths', 2)
    match = re.search(r'tessera\.' + prefix + r'pair = "((?:\\.|[^"\\])*)"', generated)
    if match is None:
        raise ValueError('native pair lacks paired-program lineage')
    paired = _decode_image(match[1]).decode()
    recipe, replaced = re.subn(r'^module attributes .* \{$', 'module {', generated, count=1, flags=re.M)
    if replaced != 1:
        raise ValueError('native pair module is malformed')
    specs = tuple(TensorSpec(f'arg{i}', 'fp32', (input_widths[i],)) for i in range(count)) + (
        TensorSpec('primal', 'fp32', (output_widths[0],), True),
        TensorSpec('derivative', 'fp32', (output_widths[1],), True), IndexSpec('n', width, width))
    recipe = attach_tensor_contract(recipe, specs, grid=(1, 1, 1), block=(width, 1, 1))
    contract = dict(schema=1, mode=mode, source_digest=hashlib.sha256(source.encode()).hexdigest(),
                    paired_ir=paired, paired_digest=hashlib.sha256(paired.encode()).hexdigest(),
                    output_order=['primal', 'tangent' if mode == 'forward' else 'cotangent'])
    encoded = json.dumps(contract, sort_keys=True).replace('\\', '\\5C').replace('"', '\\22')
    recipe = recipe.replace('module attributes {', 'module attributes {tessera.native_ad_contract = "' + encoded + '", ', 1)
    if backend == 'apple':
        from .apple_native_arena import materialize_apple_arena
        return materialize_apple_arena(recipe, compiler=compiler, llvm_bin=llvm_bin)
    return build_native_gpu_storage(recipe, compiler=compiler, llvm_bin=llvm_bin, backend=backend, chip=chip)


class NativeStoragePair:
    def __init__(self, package):
        package.validate()
        matches = re.findall(r'tessera\.native_ad_contract = "((?:\\.|[^"\\])*)"', package.arena_ir)
        if len(matches) != 1:
            raise ValueError('native pair requires one compiler AD contract')
        data = json.loads(_decode_image(matches[0]).decode())
        if set(data) != {'schema', 'mode', 'source_digest', 'paired_ir', 'paired_digest', 'output_order'} or type(data.get('schema')) is not int or data['schema'] != 1 or data.get('mode') not in ('forward', 'reverse'):
            raise ValueError('unsupported native AD contract')
        if hashlib.sha256(data['paired_ir'].encode()).hexdigest() != data['paired_digest']:
            raise ValueError('native paired-program identity disagrees')
        expected = ['primal', 'tangent' if data['mode'] == 'forward' else 'cotangent']
        if data['output_order'] != expected:
            raise ValueError('native paired output order disagrees')
        from .native_storage_contract import read_tensor_contract
        args = read_tensor_contract(package)['arguments']
        if [a['name'] for a in args if a.get('writable')] != ['primal', 'derivative']:
            raise ValueError('native paired writable ABI disagrees')
        self.signature = inspect.Signature([inspect.Parameter(a['name'], inspect.Parameter.POSITIONAL_OR_KEYWORD) for a in args])
        from .apple_native_arena import AppleArenaPackage, AppleTensorCall
        self.binding = AppleTensorCall(package, self.signature) if isinstance(package, AppleArenaPackage) else generate_tensor_binding(package, self.signature)
        from types import MappingProxyType
        data['output_order'] = tuple(data['output_order'])
        self.package, self.contract = package, MappingProxyType(data)
        self._frames = []
        self._frame_lock = threading.RLock()
        self._closed = False

    def __call__(self, *args, **kwargs):
        with self._frame_lock:
            if self._closed:
                raise ValueError('native storage pair is closed')
            return tuple(self.binding(*args, **kwargs))

    def capture(self, value):
        """Capture a persistent device snapshot for this native reverse product."""
        from .native_device_tape import NativeDeviceTape
        with self._frame_lock:
            if self._closed:
                raise ValueError('native storage pair is closed')
            frame = NativeDeviceTape(self, value)
            self._frames = [active for active in self._frames if not active.closed]
            self._frames.append(frame)
            return frame

    def close(self):
        # Publish closure before releasing frames. Do not hold this lock while
        # acquiring frame locks: backward/child calls take them in the reverse
        # order, and must be able to observe closure and unwind.
        with self._frame_lock:
            self._closed = True
            frames = tuple(self._frames)
        for frame in frames:
            frame.close()
        with self._frame_lock:
            self._frames.clear()
            self.binding.close()
