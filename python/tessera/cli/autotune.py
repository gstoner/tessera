"""Autotuning command for Tessera foundation compiler paths."""

from __future__ import annotations

from . import prof


def main(argv=None) -> int:
    args = ["--autotune"]
    if argv:
        args.extend(argv)
    return prof.main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
