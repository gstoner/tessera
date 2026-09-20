#!/usr/bin/env python3
"""Compare the toolchains on THIS box against Tessera's pins, and move them together.

Why this exists
---------------
The pins are exact, not floors (see cmake/TesseraToolchainPins.cmake). Exactness
is what makes a result from one box comparable to another's, and it is what the
sm_120 Lion lane needed: nvcc 13.4 emits PTX 9.4 while driver 610.88 JITs only
<= 9.3, so "a newer toolkit is fine" cost a week of opaque rc=3. The price of
exactness is that a fleet upgrade must touch every declaration of the version,
and there are FIVE independent ones. Editing four of five is the failure this
script removes.

What it will not do
-------------------
It never proposes a pin for a toolchain this host does not have. Bumping the
CUDA pin from a box with no CUDA would be asserting a version nobody measured --
the same claim-integrity rule as running device work on the box with the device.
Run --write on the box that HAS the toolkit; each toolchain is independent, so
a CUDA bump on The-Super-Bear and a ROCm bump on Tajasarus compose cleanly.

After a bump
------------
Every performance row recorded under the old pin is now evidence about a
toolchain that is no longer pinned (Decision #11). Re-measure, do not re-stamp.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Detection. Every probe returns None when the toolchain is absent -- never a
# guess, and never a default.
# --------------------------------------------------------------------------
def _run(cmd: list[str]) -> str | None:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        r = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return (r.stdout + r.stderr) if r.returncode == 0 else None


def _grab(text: str | None, pattern: str) -> str | None:
    if not text:
        return None
    m = re.search(pattern, text)
    return m.group(1) if m else None


def detect_cuda_toolkit() -> str | None:
    return _grab(_run(["nvcc", "--version"]), r"release (\d+\.\d+)")


def detect_ptx_isa() -> str | None:
    """The .version nvcc actually emits -- asked, not inferred from the toolkit."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None
    src = ROOT / "build" / ".tessera_ptx_probe.cu"
    try:
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("__global__ void k(){}\n")
        r = subprocess.run([nvcc, "-ptx", "-o", "-", str(src)],
                           capture_output=True, text=True, timeout=60)
        return _grab(r.stdout, r"\.version\s+(\d+\.\d+)") if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        src.unlink(missing_ok=True)


def detect_cuda_driver() -> str | None:
    return _grab(_run(["nvidia-smi", "--query-gpu=driver_version",
                       "--format=csv,noheader"]), r"(\d+\.\d+)")


def detect_hip() -> str | None:
    return _grab(_run(["hipconfig", "--version"]), r"^(\d+\.\d+)")


def detect_rocm() -> str | None:
    """ROCm release, which is NOT the HIP version (10.0 vs 7.15)."""
    for probe in (["rocminfo"], ["hipconfig", "--full"]):
        v = _grab(_run(probe), r"ROCm[- ]?[Vv]ersion[: ]+(\d+\.\d+)")
        if v:
            return v
    for marker in (Path("/opt/rocm/.info/version"), Path("/opt/rocm/core/.info/version")):
        if marker.is_file():
            v = _grab(marker.read_text(), r"^(\d+\.\d+)")
            if v:
                return v
    return None


def detect_llvm() -> str | None:
    out = _run(["llvm-config", "--version"])
    if out is None:
        for cand in ("/opt/homebrew/opt/llvm/bin/llvm-config", "/usr/lib/llvm-23/bin/llvm-config"):
            if Path(cand).is_file():
                out = _run([cand, "--version"])
                if out:
                    break
    return _grab(out, r"(\d+\.\d+\.\d+)")


def detect_macos() -> str | None:
    return _grab(_run(["sw_vers", "-productVersion"]), r"^(\d+\.\d+)")


def detect_metal() -> str | None:
    """Metal/MSL release, from the macOS release that carries it.

    The `metal` compiler reports a build number (32023.921), not an MSL
    version, so the MSL release is derived from the OS: Metal 4.0 shipped with
    macOS 26, Metal 4.1 with macOS 27.
    """
    os_ver = detect_macos()
    if os_ver is None:
        return None
    major = int(os_ver.split(".")[0])
    if major < 26:
        return None
    return f"4.{major - 26}"


# --------------------------------------------------------------------------
# Pin sites. Every declaration of a version, so a bump cannot land in four of
# five places.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Site:
    path: str
    pattern: str   # one group: the version text to replace


@dataclass(frozen=True)
class Pin:
    key: str
    label: str
    detect: object
    sites: list[Site] = field(default_factory=list)
    note: str = ""


PINS: tuple[Pin, ...] = (
    Pin("cuda", "CUDA Toolkit", detect_cuda_toolkit, [
        Site("python/tessera/compiler/gpu_target.py",
             r'(?m)^(TESSERA_TARGET_CUDA_TOOLKIT: str = ")([0-9.]+)(")'),
        Site("src/collectives/include/tessera/Dialect/Collective/Runtime/AdapterVersionPin.h",
             r'(?m)^(#define TESSERA_TARGET_CUDA_TOOLKIT\s+")([0-9.]+)(")'),
        Site("cmake/TesseraToolchainPins.cmake",
             r'(set\(TESSERA_REQUIRED_CUDA_VERSION\s+")([0-9.]+)(")'),
    ]),
    Pin("ptx", "PTX ISA (emitted by nvcc)", detect_ptx_isa, [
        Site("python/tessera/compiler/gpu_target.py",
             r'(?m)^(TESSERA_TARGET_PTX_ISA: str = ")([0-9.]+)(")'),
        Site("src/collectives/include/tessera/Dialect/Collective/Runtime/AdapterVersionPin.h",
             r'(?m)^(#define TESSERA_TARGET_PTX_ISA\s+")([0-9.]+)(")'),
        Site("cmake/TesseraToolchainPins.cmake",
             r'(set\(TESSERA_REQUIRED_PTX_ISA\s+")([0-9.]+)(")'),
    ], note="the TOOLKIT's ISA; the driver may JIT an older one -- see "
            "TESSERA_TARGET_DRIVER_JIT_PTX_ISA, which this script does NOT touch"),
    Pin("driver", "CUDA driver", detect_cuda_driver, [
        Site("python/tessera/compiler/gpu_target.py",
             r'(?m)^(TESSERA_TARGET_CUDA_DRIVER_MIN: str = ")([0-9.]+)(")'),
        Site("cmake/TesseraToolchainPins.cmake",
             r'(set\(TESSERA_REQUIRED_CUDA_DRIVER\s+")([0-9.]+)(")'),
    ]),
    Pin("rocm", "ROCm", detect_rocm, [
        Site("python/tessera/compiler/rocm_target.py",
             r'(?m)^(TESSERA_TARGET_ROCM: str = ")([0-9.]+)(")'),
        Site("src/collectives/include/tessera/Dialect/Collective/Runtime/AdapterVersionPin.h",
             r'(?m)^(#define TESSERA_TARGET_ROCM\s+")([0-9.]+)(")'),
        Site("cmake/TesseraToolchainPins.cmake",
             r'(set\(TESSERA_REQUIRED_ROCM_VERSION\s+")([0-9.]+)(")'),
    ]),
    Pin("hip", "HIP", detect_hip, [
        Site("python/tessera/compiler/rocm_target.py",
             r'(?m)^(TESSERA_TARGET_HIP: str = ")([0-9.]+)(")'),
        Site("src/collectives/include/tessera/Dialect/Collective/Runtime/AdapterVersionPin.h",
             r'(?m)^(#define TESSERA_TARGET_HIP\s+")([0-9.]+)(")'),
        Site("cmake/TesseraToolchainPins.cmake",
             r'(set\(TESSERA_REQUIRED_HIP_VERSION\s+")([0-9.]+)(")'),
    ], note="tessera_pin_rocm() takes THIS number, not the ROCm one"),
    Pin("llvm", "LLVM/MLIR", detect_llvm, [
        Site("cmake/TesseraToolchainPins.cmake",
             r'(set\(TESSERA_REQUIRED_LLVM_VERSION\s+")([0-9.]+)(")'),
    ], note="exact to the PATCH: MLIR's C++ API moves between patch releases"),
    Pin("metal", "Metal / MSL", detect_metal, [
        Site("python/tessera/compiler/apple_target.py",
             r'(?m)^(TESSERA_TARGET_METAL: str = ")([0-9.]+)(")'),
    ]),
)


def _current(site: Site) -> str | None:
    text = (ROOT / site.path).read_text()
    m = re.search(site.pattern, text)
    return m.group(2) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="update the pins this host can actually measure")
    ap.add_argument("--only", metavar="KEY",
                    help="restrict to one pin (%s)" % ", ".join(p.key for p in PINS))
    args = ap.parse_args()

    pins = [p for p in PINS if not args.only or p.key == args.only]
    if args.only and not pins:
        print(f"unknown pin {args.only!r}", file=sys.stderr)
        return 2

    drift, absent, edits = [], [], []
    print(f"{'pin':<10} {'pinned':>10} {'this host':>12}   status")
    print("-" * 78)
    for pin in pins:
        found = pin.detect()
        currents = {s.path: _current(s) for s in pin.sites}
        distinct = {v for v in currents.values() if v is not None}
        pinned = next(iter(distinct)) if len(distinct) == 1 else "/".join(sorted(distinct)) or "?"

        if len(distinct) > 1:
            status = "DECLARATIONS DISAGREE"
            drift.append(pin)
        elif found is None:
            status = "not on this host -- not checked"
            absent.append(pin)
        elif found == pinned:
            status = "ok"
        else:
            status = "DRIFT"
            drift.append(pin)
            for s in pin.sites:
                edits.append((s, pinned, found))
        print(f"{pin.key:<10} {pinned:>10} {str(found or '-'):>12}   {status}")
        if pin.note:
            print(f"{'':<10} {'':>10} {'':>12}   note: {pin.note}")

    missing = [(p, s) for p in pins for s in p.sites if _current(s) is None]
    if missing:
        print("\nPIN SITES THAT NO LONGER MATCH THEIR PATTERN (this script would "
              "silently skip them -- fix the pattern):")
        for p, s in missing:
            print(f"  {p.key}: {s.path}")
        return 2

    if args.write:
        if not edits:
            print("\nnothing to write: every pin this host can measure already matches")
            return 0
        print()
        touched: dict[str, int] = {}
        for site, old, new in edits:
            path = ROOT / site.path
            text = path.read_text()
            text, n = re.subn(site.pattern, lambda m: m.group(1) + new + m.group(3), text)
            path.write_text(text)
            touched[site.path] = touched.get(site.path, 0) + n
            print(f"  {site.path}: {old} -> {new}")
        print(f"\nupdated {sum(touched.values())} declaration(s) in {len(touched)} file(s).")
        print("Now, and this is the part that is easy to skip:")
        print("  * re-measure every performance row taken under the old pin "
              "(Decision #11: a measurement is only valid for the toolchain that "
              "produced it), and")
        print("  * run this on the OTHER boxes -- each toolchain is pinned once "
              "for the whole fleet, so a box left behind now fails its configure.")
        return 0

    if absent:
        print(f"\n{len(absent)} pin(s) not measurable here ({', '.join(p.key for p in absent)}); "
              "check those on the box that has the toolkit.")
    if drift:
        print(f"\nDRIFT in: {', '.join(p.key for p in drift)}. Re-run with --write "
              "on this box to move them.")
        return 1
    print("\nevery pin this host can measure matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
