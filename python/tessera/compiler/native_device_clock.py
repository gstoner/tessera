"""Compiler-built device-clock marker kernels (sync WSL-TIMING-ADMISSION-2026-09-26).

A calibration window launches the marker, then N launches of the **exact clean
image** under test, then the marker again, all on one stream. Both marker
launches share one two-word span buffer: the ``--tessera-device-clock-span``
pass makes the marker do ``umin(span[0], clock)`` at its start and
``umax(span[1], clock)`` at its end, so the span runs from the first marker's
start to the second marker's end -- the whole window on the device's
constant-rate clock, independent of the host event API.

Why markers rather than instrumenting the measured kernel: on the gfx1151
serial SSD kernel (2026-09-26) any memory operation stamped at the kernel's
start changed LLVM's optimization of it (2512 -> ~1230 instructions, 2.4x
faster), so an "instrumented twin" timed a different program. A marker leaves
the measured image byte-identical; its own cost (two tiny launches per window)
is what the instrumented/clean ratio in the calibration packet bounds.

The marker is compiled through the same MLIR route as native storage packages
(`tessera-opt` pass -> ROCDL -> `gpu-module-to-binary`), never from source
text in a Python emitter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

from .native_gpu_storage import _binary_pass, _decode_image, _resolve_tool, _run, _sha

MARKER_ENTRY = "tessera_device_clock_marker"

_MARKER_MODULE = f"""module attributes {{gpu.container_module}} {{
  gpu.module @{MARKER_ENTRY} {{
    gpu.func @{MARKER_ENTRY}() kernel {{
      gpu.return
    }}
  }}
}}
"""


@dataclass(frozen=True)
class DeviceClockMarker:
    backend: str
    chip: str
    entry: str
    image: bytes
    compiler_digest: str
    llvm_digest: str

    @property
    def image_sha256(self) -> str:
        return hashlib.sha256(self.image).hexdigest()


def build_device_clock_marker(*, compiler: Path, llvm_bin: Path, backend: str,
                              chip: str, toolkit: Path | None = None) -> DeviceClockMarker:
    """Compile and validate the marker for one exact target."""
    if (backend, chip) not in (('rocm', 'gfx1151'), ('rocm', 'gfx1201')):
        raise ValueError(
            'device-clock marker is validated for ROCm gfx1151/gfx1201 only; the '
            'NVIDIA %globaltimer marker is owed on Super-Bear (sync '
            'WSL-TIMING-ADMISSION-2026-09-26)')
    compiler, llvm_bin = Path(compiler), Path(llvm_bin)
    stamped = _run(compiler, f'--tessera-device-clock-span=backend={backend}', source=_MARKER_MODULE)
    if 'tessera.device_clock_span' not in stamped:
        raise ValueError('device-clock span pass did not stamp the marker kernel')
    pipeline = ('builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,'
                'convert-math-to-llvm,reconcile-unrealized-casts),'
                f'rocdl-attach-target{{chip={chip}}},{_binary_pass(toolkit)})')
    binary = _run(llvm_bin / 'mlir-opt', '--pass-pipeline=' + pipeline, source=stamped)
    if binary.count('#gpu.object<') != 1:
        raise ValueError('expected exactly one device-clock marker image')
    encoded = re.search(r'bin = "((?:\\.|[^"\\])*)"', binary)
    image = _decode_image(encoded[1] if encoded else re.findall(r'"((?:\\.|[^"\\])*)"', binary)[-1])
    _require_clock_and_atomics(image, llvm_bin)
    return DeviceClockMarker(backend, chip, MARKER_ENTRY, image, _sha(compiler.read_bytes()),
                             _sha(_resolve_tool(llvm_bin / 'mlir-opt').read_bytes()))


def _require_clock_and_atomics(image: bytes, llvm_bin: Path) -> None:
    """The marker must read the realtime counter and update the span atomically.

    Stricter than the storage packager's store check (a marker writes only
    through atomics): if either is missing, the marker would time nothing.
    """
    objdump = _resolve_tool(llvm_bin / 'llvm-objdump')
    with tempfile.TemporaryDirectory(prefix='tessera-clock-marker-') as tmp:
        path = Path(tmp) / 'marker.hsaco'
        path.write_bytes(image)
        result = subprocess.run([str(objdump), '-d', '--triple=amdgcn-amd-amdhsa', str(path)],
                                capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError('device-clock marker could not be disassembled: ' + result.stderr.strip()[:300])
    text = result.stdout
    reads = text.count('MSG_RTN_GET_REALTIME')
    atomics = len(re.findall(r'global_atomic_\w+_(?:b64|x2|u64)', text))
    if reads != 2 or atomics < 2:
        raise ValueError(
            f'device-clock marker must read the realtime counter twice and update the span '
            f'atomically; found {reads} clock reads and {atomics} 64-bit global atomics')


__all__ = ["DeviceClockMarker", "MARKER_ENTRY", "build_device_clock_marker"]
