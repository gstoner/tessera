#!/usr/bin/env python3
"""Bind a validation command to an expected checkout and detect tree changes.

Capture on the intended tree before copying/switching checkouts, then pass
--expect to run. Without an external expectation the receipt establishes only
which tree ran, not whether that tree was the one the caller intended.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root: Path) -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
    ).decode().split("\0")
    files = {name: digest(root / name) if (root / name).is_file() else None
             for name in sorted(set(names)) if name}
    return {"head": head, "files": files}


def verify(root: Path, expected: dict, actual: dict) -> None:
    if expected.get("head") and expected["head"] != actual["head"]:
        raise ValueError(f"validation checkout mismatch: expected {expected['head']}, got {actual['head']}")
    for name, value in expected["files"].items():
        path = root / name
        if path.is_absolute() and not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"validation manifest path escapes repository: {name}")
        observed = digest(path) if path.is_file() else None
        if observed != value:
            raise ValueError(f"validation source mismatch: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--expect", type=Path, default=os.environ.get("TESSERA_EXPECTED_TREE"))
    parser.add_argument("--expect-head", default=os.environ.get("TESSERA_EXPECTED_HEAD") or os.environ.get("GITHUB_SHA"))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    before = snapshot(args.root)
    if args.capture:
        args.capture.write_text(json.dumps(before, sort_keys=True, indent=2) + "\n")
        return 0
    if args.expect:
        verify(args.root, json.loads(Path(args.expect).read_text()), before)
    if args.expect_head:
        verify(args.root, {"head": args.expect_head, "files": {}}, before)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide --capture or a validation command after --")
    tree_hash = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    print(f"Validation tree: {before['head']} content={tree_hash}", flush=True)
    env = dict(os.environ, TESSERA_VALIDATION_TREE_ACTIVE="1")
    env["PYTHONPATH"] = str(args.root.resolve() / "python") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(command, cwd=args.root, env=env, check=False)
    after = snapshot(args.root)
    stable = before == after
    receipt = {"tree": before, "content_digest": tree_hash, "command": command,
               "exit_code": result.returncode, "stable": stable,
               "expectation_bound": bool(args.expect or args.expect_head)}
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    if not stable:
        print("Validation tree changed during execution; result is invalid.", file=sys.stderr)
        return 1
    return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
