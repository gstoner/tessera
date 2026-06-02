#!/usr/bin/env python3
"""Regenerate Tessera-authored ``.mtlpackage`` fixtures (PK8c).

These are *production* packaged kernels Tessera authors itself from the
MPSGraph lane (PK8), committed under ``tests/fixtures/apple_gpu/`` so the
``PACKAGED_PRODUCTION_KERNELS`` manifest rows reference a real on-disk
artifact. This script makes them reproducible on any Apple-Silicon /
macOS-14+ host — re-run it to rebuild after an SDK bump.

    python3 scripts/author_apple_packages.py [--check]

``--check`` authors into a temp dir and diffs structure (manifest +
mpsgraphpackage presence) against the committed fixtures, without
overwriting — useful as a drift smoke. Default (no flag) (re)writes the
committed fixtures in place.

Caveat: an MPSGraph-serialized package targets the current platform's
latest deployment version; it may not load on an *older* macOS. The
validating test (``test_apple_mlpkg_pk8c.py``) skips when packaged ML is
unavailable and tolerates a load failure as a portability skip rather than
a hard failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

# The canonical Tessera-authored production packages. Keep small + fixed so
# the committed binary is stable and the numerical compare is cheap.
#   name -> (kind, spec)
#     kind "matmul": spec = (M, K, N)
#     kind "chain":  spec = (chain_name, dims_tuple)
FIXTURES = {
    "tessera_authored_matmul_8x8x8": ("matmul", (8, 8, 8)),
    "tessera_authored_matmul_softmax_4x6x5": (
        "chain", ("matmul_softmax", (4, 6, 5))),
}

FIXTURE_DIR = REPO / "tests" / "fixtures" / "apple_gpu"


def _author_one(name, kind, spec, out_dir):
    from tessera import apple_mlpkg as mp

    pkg = out_dir / f"{name}.mtlpackage"
    if kind == "matmul":
        m, k, n = spec
        ok = mp.author_matmul_package(str(pkg), m, k, n)
    elif kind == "chain":
        chain, dims = spec
        ok = mp.author_chain_package(str(pkg), chain, list(dims))
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return pkg, ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="author into a temp dir + verify structure only")
    args = ap.parse_args()

    from tessera import apple_mlpkg as mp

    if not mp.packaged_ml_available():
        reason = mp.packaged_ml_skip_reason() or "packaged ML unavailable"
        print(f"SKIP: cannot author on this host — {reason}")
        # Not an error: off-Darwin / pre-macOS-26 simply can't author.
        return 0

    if args.check:
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="pk8c_check_"))
        ok_all = True
        for name, (kind, spec) in FIXTURES.items():
            pkg, ok = _author_one(name, kind, spec, tmp)
            committed = FIXTURE_DIR / f"{name}.mtlpackage"
            structure_ok = (
                ok
                and (pkg / "manifest.json").is_file()
                and (pkg / "library.mpsgraphpackage").is_dir()
            )
            present = committed.is_dir()
            print(f"  {name}: author={'ok' if ok else 'FAIL'} "
                  f"structure={'ok' if structure_ok else 'FAIL'} "
                  f"committed={'present' if present else 'MISSING'}")
            ok_all &= structure_ok and present
        return 0 if ok_all else 1

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    rc = 0
    for name, (kind, spec) in FIXTURES.items():
        pkg, ok = _author_one(name, kind, spec, FIXTURE_DIR)
        if ok:
            print(f"  wrote {pkg.relative_to(REPO)}")
        else:
            print(f"  FAILED to author {name}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
