"""The owning architecture of a native quantized MLP timing recorder.

GFX1151-QUANTIZED-MLP-TIMING-20261101. The quantized MLP recorders were written
for gfx1201 and refused every other device. gfx1151 has its own exact-device
proof since slice 4 (GFX1151-QUANTIZED-MLP-20261030), so each recorder now
records on whichever owning architecture is live and labels the packet with
that architecture. Nothing here lets one architecture's packet stand for the
other: the device label, target and source Graph all come from the live device.
"""
import os
import re
import shutil
import subprocess

from tessera import runtime as rt

#: Architectures with exact-device proof of the quantized MLP (mirrors
#: tests/device/rocm/test_native_quantized_mlp_bounded_rows.py).
OWNING_ARCHES = ("gfx1201", "gfx1151")

DEVICES = {
    "gfx1201": "gfx1201 (RX 9070 XT, Tajasarus, WSL2)",
    "gfx1151": "gfx1151 (Radeon 8060S, Strix Halo, Princess-Luna, WSL2)",
}

#: How every recorder's device numbers are produced (Decision #12).
ROUTE = ("authored Graph SSA -> seven native partitions -> Schedule -> Tile -> ROCm Target -> "
         "LLVM/HSACO -> one checked multi-output HIP owner")
HIP_EVENT_TIMING = "hip_event_owner_window"
HOST_WALL_TIMING = "host_wall_perf_counter"


def owning_arch():
    """The live owning architecture; refuses any other device by name."""
    arch = rt._rocm_live_arch()
    if arch not in OWNING_ARCHES:
        raise SystemExit(f"requires an owning {'/'.join(OWNING_ARCHES)} device, not {arch}")
    return arch


def target(arch):
    if arch not in OWNING_ARCHES:
        raise ValueError(f"no owning quantized MLP target for {arch}")
    return "rocm_" + arch


def device(arch):
    return DEVICES[arch]


def source(m, h, k, n, arch):
    """The authored MLP Graph text, retargeted to ``arch``."""
    from tests.unit.test_native_quantized_mlp_program import source as gfx1201_source

    text = gfx1201_source(m, h, k, n)
    if arch == "gfx1201":
        return text
    target(arch)
    if text.count('"rocm_gfx1201"') != 1 or text.count('tessera.arch = "gfx1201"') != 1:
        raise ValueError("the authored MLP Graph no longer names exactly one gfx1201 target")
    return text.replace('"rocm_gfx1201"', f'"rocm_{arch}"').replace(
        'tessera.arch = "gfx1201"', f'tessera.arch = "{arch}"')


def check_sync_key(sync_key, arch):
    """Refuse a sync key that names a different owning architecture."""
    for other in OWNING_ARCHES:
        if other != arch and other.upper() in sync_key.upper():
            raise SystemExit(f"sync key {sync_key} names {other}, but the live device is {arch}")
    return sync_key


def identity(arch):
    """(uuid, marketing name) of the live owning agent, from rocminfo."""
    tool = shutil.which("rocminfo") or os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "rocminfo")
    info = subprocess.check_output([tool], text=True)
    found = re.search(rf"Name:\s+{arch}\s+Uuid:\s+(GPU-\S+)\s+Marketing Name:\s+([^\n]+)", info)
    if found is None:
        raise RuntimeError(f"owning {arch} device identity unavailable from {tool}")
    return found[1], found[2].strip()


def add_arch_arguments(parser, default_sync_key):
    """``--sync-key`` (checked against the live arch) and ``--device-lock``."""
    parser.add_argument("--sync-key", default=default_sync_key)
    parser.add_argument("--device-lock", default=None,
                        help="flock path held around this run; recorded and checked before timing")


def device_lock(path):
    """Whether the shared device lock at ``path`` is held around this recorder.

    Run the recorder as ``flock <path> python ...``. A fresh non-blocking
    exclusive ``flock`` from here then fails because the wrapper holds it; when
    it succeeds nobody holds the lock and the recorder refuses to time.
    """
    import errno
    import fcntl

    if not path:
        return None
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                return {"path": path, "held": True}
            raise
        fcntl.flock(handle, fcntl.LOCK_UN)
    raise SystemExit(f"device lock {path} is not held; run the recorder under flock {path}")
