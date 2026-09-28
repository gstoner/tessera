"""Compare a re-recorded corpus with the prior one and restore the prior order.

Sync AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27. Usage:

    python summarize_rerecord.py --prior OLD.json --new NEW.json --output OUT.json \
        [--devices nvidia:sm_120] [--ops paged_kv_decode ...]

``OUT.json`` is ``NEW.json`` with its records in the prior file's order (new
keys appended in sorted order), so the committed diff shows only what was
re-measured. With ``--devices``/``--ops`` the rows of ``NEW`` outside that
selection are replaced by the prior file's rows byte-for-byte (a recorder that
rewrites the whole corpus -- the gfx1151 paged-KV recorder, run on a box where
the sm_120 rows load as stale -- must not move other devices' rows).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    bucket = row.get("bucket")
    return (row["device"], row["target"], row["op"],
            tuple(bucket) if bucket is not None else None, row["dtype"],
            row.get("timing", "end_to_end"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="*", default=None)
    parser.add_argument("--ops", nargs="*", default=None)
    args = parser.parse_args(argv)
    prior = json.loads(args.prior.read_text())
    new = json.loads(args.new.read_text())
    old = {_key(r): r for r in prior["records"]}
    fresh = {_key(r): r for r in new["records"]}

    def selected(key: tuple[Any, ...]) -> bool:
        return ((args.devices is None or key[0] in args.devices)
                and (args.ops is None or key[2] in args.ops))

    merged = {k: (fresh[k] if selected(k) and k in fresh else old.get(k, fresh.get(k)))
              for k in set(old) | set(fresh)
              if k in old or selected(k)}
    order = [k for k in (_key(r) for r in prior["records"]) if k in merged]
    order += sorted((k for k in merged if k not in old), key=str)
    out = {**new, "records": [merged[k] for k in order]}
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    lost = sorted((k for k in old if k not in merged), key=str)
    added = sorted((k for k in merged if k not in old), key=str)
    changed = [k for k in order if k in old and old[k] != merged[k]]
    print("lost", lost)
    print("new", added)
    print(f"rows changed {len(changed)} unchanged {len(order) - len(changed) - len(added)}")
    print("changed by device/op:",
          dict(collections.Counter((k[0], k[2]) for k in changed)))
    untouched = [k for k in old if k in merged and not selected(k)]
    print("rows outside the selection byte-identical:",
          all(old[k] == merged[k] for k in untouched), len(untouched))
    wins = [(k, old[k]["winner"], merged[k]["winner"]) for k in changed
            if old[k]["winner"] != merged[k]["winner"]]
    print("winner changes", len(wins))
    for k, before, after in wins:
        print(f"   {k[2]} {k[4]} {k[5]} {list(k[3])} {before} -> {after}")
    registry = {"matmul", "fused_region", "attention", "gated_matmul"}
    reg = [merged[k] for k in order if k[0] == "nvidia:sm_120" and k[2] in registry]
    print("sm120 registry rows", len(reg), "selector_eligible",
          sum(bool((r.get("evidence") or {}).get("selector_eligible")) for r in reg))
    unstamped = []
    for k in order:
        row = merged[k]
        stamps = (row.get("evidence") or {}).get("delegate_identities") or {}
        for name in row.get("candidates", {}):
            if not stamps.get(name):
                unstamped.append(f"{k[0]} {k[2]} {list(k[3])} {k[5]}: {name}")
    print("timed candidates with no identity (all rows)", len(unstamped))
    for line in unstamped:
        print("  ", line)
    print("sm120 registry rows with unmeasured candidates",
          sum(bool(r.get("unmeasured")) for r in reg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
