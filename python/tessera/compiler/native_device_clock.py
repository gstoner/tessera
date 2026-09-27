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
start changed LLVM's optimization of it (2512 instructions -> 924 when stamped
at the block start, ~1230 when placed after the entry allocas; the 924 form ran
2.4x faster), so an "instrumented twin" timed a different program. A marker leaves
the measured image byte-identical; its own cost (two tiny launches per window)
is what the instrumented/clean ratio in the calibration packet bounds.

The marker is compiled through the same MLIR route as native storage packages
(`tessera-opt` pass -> ROCDL/NVVM -> `gpu-module-to-binary`), never from source
text in a Python emitter.

Validated targets, each on its own device (evidence never transfers):

* ``rocm`` gfx1151 / gfx1201 -- ``llvm.readsteadycounter`` ->
  ``s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME``; ticks at
  ``hipDeviceAttributeWallClockRate``.
* ``nvidia`` sm_120 -- ``llvm.nvvm.read.ptx.sreg.globaltimer`` -> SASS
  ``CS2R Rn, SR_GLOBALTIMERLO`` (a 64-bit nanosecond counter), span updates as
  64-bit ``REDG.E.MIN/MAX.64``. Validated on The-Super-Bear (RTX 5070, WSL2,
  driver 610.88, 2026-09-26): the span agreed with CUDA events within the 5%
  band once a window is long enough to dominate the per-window offset, and
  the counter's granularity was measured, not assumed (sync
  ``NVIDIA-GLOBALTIMER-MARKER-2026-09-26``; packet
  ``benchmarks/baselines/sm120_ssd_calibrated_pairs_20260926/``).
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


#: ``(backend, chip)`` pairs whose marker has exact-device validation. Explicit,
#: not a prefix match: a new part is refused until it has its own proof.
VALIDATED_MARKER_TARGETS: tuple[tuple[str, str], ...] = (
    ('rocm', 'gfx1151'), ('rocm', 'gfx1201'), ('nvidia', 'sm_120'),
)


def build_device_clock_marker(*, compiler: Path, llvm_bin: Path, backend: str,
                              chip: str, toolkit: Path | None = None) -> DeviceClockMarker:
    """Compile and validate the marker for one exact target."""
    if (backend, chip) not in VALIDATED_MARKER_TARGETS:
        raise ValueError(
            'device-clock marker is validated for ROCm gfx1151/gfx1201 and NVIDIA '
            f'sm_120 only; got {backend!r}/{chip!r} (each target needs its own '
            'exact-device validation; sync NVIDIA-GLOBALTIMER-MARKER-2026-09-26)')
    compiler, llvm_bin = Path(compiler), Path(llvm_bin)
    stamped = _run(compiler, f'--tessera-device-clock-span=backend={backend}', source=_MARKER_MODULE)
    if 'tessera.device_clock_span' not in stamped:
        raise ValueError('device-clock span pass did not stamp the marker kernel')
    target = 'nvvm' if backend == 'nvidia' else 'rocdl'
    pipeline = (f'builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-{target},'
                'convert-math-to-llvm,reconcile-unrealized-casts),'
                f'{target}-attach-target{{chip={chip}}},{_binary_pass(toolkit)})')
    binary = _run(llvm_bin / 'mlir-opt', '--pass-pipeline=' + pipeline, source=stamped)
    if binary.count('#gpu.object<') != 1:
        raise ValueError('expected exactly one device-clock marker image')
    encoded = re.search(r'bin = "((?:\\.|[^"\\])*)"', binary)
    image = _decode_image(encoded[1] if encoded else re.findall(r'"((?:\\.|[^"\\])*)"', binary)[-1])
    if backend == 'nvidia':
        _require_globaltimer_and_atomics(image)
    else:
        _require_clock_and_atomics(image, llvm_bin)
    return DeviceClockMarker(backend, chip, MARKER_ENTRY, image, _sha(compiler.read_bytes()),
                             _sha(_resolve_tool(llvm_bin / 'mlir-opt').read_bytes()))


def _require_globaltimer_and_atomics(image: bytes) -> None:
    """The NVIDIA twin of :func:`_require_clock_and_atomics`, on the SASS.

    The check reads the cubin the driver will load, not the PTX: ``cuobjdump
    --dump-sass`` must show exactly two ``SR_GLOBALTIMERLO`` reads (``CS2R`` of
    the 64-bit ``%globaltimer`` pair) and at least two 64-bit global atomic
    min/max updates of the span. ``cuobjdump`` is a toolkit tool, not a
    matched-LLVM one; without it the marker is refused rather than shipped
    unchecked, since an unchecked marker could time nothing.
    """
    from .native_gpu_storage import _cuda_disassembler
    tool = _cuda_disassembler()
    if tool is None:
        raise ValueError('the NVIDIA device-clock marker needs cuobjdump (CUDA toolkit) to '
                         'verify its %globaltimer reads; source scripts/_nvidia_env.sh')
    with tempfile.TemporaryDirectory(prefix='tessera-clock-marker-') as tmp:
        path = Path(tmp) / 'marker.fatbin'
        path.write_bytes(image)
        result = subprocess.run([str(tool), '--dump-sass', str(path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError('device-clock marker could not be disassembled: '
                         + (result.stderr or result.stdout).strip()[:300])
    text = result.stdout
    reads = len(re.findall(r'\bCS2R\s+R\d+,\s*SR_GLOBALTIMERLO\b', text))
    # Exactly one start update (MIN into span[0]) and one end update (MAX into
    # span[1]): two MINs would leave the end unwritten and still count two
    # atomics (review).
    mins = len(re.findall(r'\b(?:RED|ATOM)G?\.E\.MIN\.64\b', text))
    maxs = len(re.findall(r'\b(?:RED|ATOM)G?\.E\.MAX\.64\b', text))
    if reads != 2 or mins != 1 or maxs != 1:
        raise ValueError(
            f'device-clock marker must read %globaltimer twice and update the span '
            f'with one 64-bit MIN and one 64-bit MAX; found {reads} SR_GLOBALTIMERLO '
            f'reads, {mins} MIN and {maxs} MAX atomics')


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


__all__ = ["DeviceClockMarker", "MARKER_ENTRY", "VALIDATED_MARKER_TARGETS", "build_device_clock_marker"]
