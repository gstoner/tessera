#!/usr/bin/env python3
"""Sprint G-9 + H-8 (2026-05-11) — NCCL / RCCL symbol resolution probe.

Hardware-free check: dlopens libnccl.so (CUDA 13.2 U1) and librccl.so
(ROCm 7.2.4) and resolves the symbol surface Tessera's adapters link
against (`ncclAllReduce`, `ncclReduceScatter`, `ncclAllGather`,
`ncclSend`, `ncclRecv`, `ncclCommInitRank`, `ncclGetVersion`).  No GPU
or multi-process collective execution is required — just a successful
symbol resolution.

If either library is absent, the corresponding section is skipped.

Usage:
    python scripts/probe_collective_libs.py

Exit codes:
    0 — all available libs resolved cleanly (or both libs absent)
    1 — at least one expected symbol unresolved
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys


TESSERA_TARGET_NCCL_MIN = (2, 22)
TESSERA_TARGET_RCCL_MIN = (2, 22)


EXPECTED_SYMBOLS = (
    "ncclAllReduce",
    "ncclReduceScatter",
    "ncclAllGather",
    "ncclSend",
    "ncclRecv",
    "ncclCommInitRank",
    "ncclGetVersion",
    "ncclGetErrorString",
)


def _try_load(*candidates: str) -> ctypes.CDLL | None:
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    # ctypes.util.find_library as a last resort
    for name in candidates:
        base = name.lstrip("lib").rsplit(".so", 1)[0]
        path = ctypes.util.find_library(base)
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                pass
    return None


def _probe(lib: ctypes.CDLL, label: str, min_version: tuple[int, int]) -> bool:
    ok = True
    missing: list[str] = []
    for sym in EXPECTED_SYMBOLS:
        if not hasattr(lib, sym):
            missing.append(sym)
            ok = False

    if missing:
        print(f"  {label}: ✗ missing symbols: {missing}")
    else:
        print(f"  {label}: ✓ all {len(EXPECTED_SYMBOLS)} expected symbols resolved")

    # Version check via ncclGetVersion(int*).
    try:
        fn = lib.ncclGetVersion
        fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
        fn.restype = ctypes.c_int  # ncclResult_t
        version = ctypes.c_int(0)
        rc = fn(ctypes.byref(version))
        if rc == 0:
            major = version.value // 10000
            minor = (version.value // 100) % 100
            patch = version.value % 100
            print(f"    {label} version: {major}.{minor}.{patch}")
            if (major, minor) < min_version:
                print(f"    ✗ version {major}.{minor} < required {min_version}")
                ok = False
        else:
            print(f"    {label} ncclGetVersion returned rc={rc}")
    except Exception as e:
        print(f"    {label} ncclGetVersion probe failed: {e}")

    return ok


def main() -> int:
    print("Sprint G-9 + H-8 — NCCL / RCCL symbol resolution probe")
    print("======================================================")

    # NCCL — CUDA 13.2 U1 bundles NCCL 2.22+.
    nccl = _try_load("libnccl.so.2", "libnccl.so")
    if nccl is None:
        print("  NCCL: not installed (libnccl.so absent) — skipping")
        nccl_ok = True
    else:
        nccl_ok = _probe(nccl, "NCCL", TESSERA_TARGET_NCCL_MIN)

    print()

    # RCCL — ROCm 7.2.4 bundles RCCL 2.22+.
    rccl = _try_load("librccl.so.1", "librccl.so")
    if rccl is None:
        print("  RCCL: not installed (librccl.so absent) — skipping")
        rccl_ok = True
    else:
        rccl_ok = _probe(rccl, "RCCL", TESSERA_TARGET_RCCL_MIN)

    print()

    if not (nccl_ok and rccl_ok):
        print("FAILED — one or more collective libs missing symbols or below pin.")
        return 1

    if nccl is None and rccl is None:
        print("OK — neither NCCL nor RCCL installed; build will use mock collectives.")
    else:
        print("OK — all installed collective libs resolve cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
