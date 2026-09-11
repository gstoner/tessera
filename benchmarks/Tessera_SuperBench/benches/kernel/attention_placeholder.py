#!/usr/bin/env python3
"""Retired synthetic timer; never emit fabricated attention measurements."""


def main():
    raise SystemExit(
        "The sleep-based attention placeholder is retired. "
        "Use flashattn_tessera.py for reference/artifact measurements, "
        "or an owning-device package benchmark for native execution."
    )


if __name__ == "__main__":
    main()
