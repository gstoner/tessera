#!/usr/bin/env python3
"""Fail a CI proof step whose pytest run executed nothing it was meant to.

A proof step that builds a tool and then runs tests which ``skipif`` the tool
is absent can come back green having proven nothing: a wrong build path, a
renamed binary or a missing linker turns every test into a skip and pytest
exits 0. That is the hollow-green shape the 2026-09-27 CI audit removed from
the lit and rocm-serialize lanes (sync ``FOUNDATION-BATCH-2-2026-09-27``).

Reads a JUnit XML file written by ``pytest --junitxml`` and exits non-zero when

* no testcase passed, or
* any testcase failed or errored (pytest already fails the step; repeated
  here so the check stands alone), or
* any testcase was skipped and its reason does not contain one of the
  ``--allow-skip`` substrings -- the skips a lane is *expected* to report,
  named in the workflow so a reviewer can see them.

Usage::

    python scripts/ci_require_executed.py results.xml \
        --allow-skip "no ROCm device bitcode"
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def evaluate(xml_path: Path, allowed: list[str]) -> list[str]:
    """Return a list of problems; empty means the proof actually ran."""

    problems: list[str] = []
    try:
        # The file is pytest's own --junitxml output from the same job.
        root = ET.parse(xml_path).getroot()  # noqa: S314
    except (OSError, ET.ParseError) as exc:
        return [f"cannot read JUnit results {xml_path}: {exc}"]

    passed = 0
    for case in root.iter("testcase"):
        name = f"{case.get('classname', '')}::{case.get('name', '')}"
        skipped = case.find("skipped")
        failure = case.find("failure")
        error = case.find("error")
        if failure is not None or error is not None:
            problems.append(f"{name}: failed/errored")
        elif skipped is not None:
            reason = (skipped.get("message") or "") + " " + (skipped.text or "")
            if not any(token in reason for token in allowed):
                problems.append(f"{name}: unexpected skip ({reason.strip()!r})")
        else:
            passed += 1
    if passed == 0:
        problems.append("no testcase passed -- the proof step executed nothing")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("junit_xml", type=Path)
    parser.add_argument(
        "--allow-skip",
        action="append",
        default=[],
        metavar="SUBSTRING",
        help="a skip reason substring this lane is expected to report",
    )
    args = parser.parse_args(argv)
    problems = evaluate(args.junit_xml, args.allow_skip)
    if problems:
        for problem in problems:
            print(f"::error ::{problem}", file=sys.stderr)
        return 1
    print(f"{args.junit_xml}: every selected proof test executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
