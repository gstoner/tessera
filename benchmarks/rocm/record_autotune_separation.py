"""Re-race the committed gfx1151 corpus rows so each carries a separation verdict.

AUTOTUNE-SEPARATION, ROCm half. The 12 committed `rocm:gfx1151` rows predate
#663 and carry `separation: None` -- "never asked", which `corpus_winner`
currently still accepts as a dispatch hint. They cannot be re-raced by simply
re-running a recorder: `load_corpus` makes every key a hit, and
`_record_raced_the_live_field` never consults `separation`, so an
already-present row is returned unchanged. This evicts them first.

Two rules carried from the NVIDIA half, both learned by nearly breaking them:

* **Warm-start always.** `save_corpus` writes the WHOLE cache, so a recorder
  that starts empty deletes every other device's rows. A bare NVIDIA run
  deleted all 12 ROCm rows; the symmetric mistake here would delete 97 NVIDIA
  ones.
* **Diff before committing.** Both near-misses were regenerations that
  SUCCEEDED while producing weaker evidence than they replaced. Nothing fails;
  only a row-and-evidence count catches it, so this prints one.

ROCm-specific hazard, and the reason `--check-timer` exists (on by default):
`_hip_resident_launch_latency` falls back to the host wall clock when HIP
events misbehave on this fleet, and a wall clock is *noisier* than a device
event. That RAISES the noise floor and makes separation harder to reach --
correctly. So an unseparated gfx1151 row may mean the timer degraded rather
than that the kernels are equal, and the two must not be confused.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

#: The one device key this driver recreates. Eviction and the safety count
#: both key on it exactly: matching `startswith("rocm")` would evict another
#: ROCm device's fused rows, and `_summarise` counting every ROCm device as one
#: bucket meant the "did another device lose rows?" guard could not see it --
#: the guard against silent evidence loss had the same blind spot as the loss.
DEVICE = "rocm:gfx1151"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt                              # noqa: E402
from tessera.compiler import fusion as F                       # noqa: E402
from tessera.compiler.emit import autotune as at               # noqa: E402
from tessera.compiler.emit.candidate import OP_FUSED_REGION    # noqa: E402

# Candidate registration is a side effect of importing the backend's emit
# module. Without this the registry is EMPTY and `measured_arbitrate` returns
# None for every shape -- which prints as "no verified candidate", i.e. it
# reads as "this hardware cannot do it" rather than "the driver forgot an
# import". A missing import here produces a confident wrong answer, not an
# error, so it is imported explicitly rather than relied upon transitively.
from tessera.compiler.emit import rocm_hip as _rocm_hip        # noqa: E402,F401


def candidates_live(region, inputs):
    from tessera.compiler.emit.candidate import live_candidates
    return live_candidates(region, OP_FUSED_REGION, "rocm", inputs)


def _summarise(cache: at.MeasureCache) -> dict[str, int]:
    rows = cache.to_dict()["records"]
    inf = float("inf")
    ranked = [r for r in rows
              if len([v for v in r.get("candidates", {}).values() if v != inf]) >= 2]
    return {
        "rows": len(rows),
        "rocm_rows": sum(1 for r in rows if r["device"] == DEVICE),
        "other_rows": sum(1 for r in rows if r["device"] != DEVICE),
        "ranked": len(ranked),
        "separated": sum(1 for r in ranked
                         if (r.get("separation") or {}).get("separated")),
        "unseparated": sum(1 for r in ranked
                           if r.get("separation") is not None
                           and not r["separation"].get("separated")),
        "never_asked": sum(1 for r in ranked if r.get("separation") is None),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", nargs="+", type=int, default=[64, 256, 512, 1024])
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device-repeats", type=int, default=10,
                    help="whole device measurements per candidate; sets the "
                         "noise floor behind each separation verdict")
    ap.add_argument("--dry-run", action="store_true",
                    help="measure and report without writing the corpus")
    ap.add_argument("--check-timer", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="refuse to write device-timed rows unless the ROCm "
                         "timer actually used device events (default: on)")
    args = ap.parse_args()

    from tessera.compiler.emit.candidate import candidates_for
    registered = [c.name for c in candidates_for("rocm", OP_FUSED_REGION)]
    if not registered:
        print("no rocm fused_region candidates are registered -- refusing to "
              "record a corpus that would claim this hardware has none")
        return 1
    print(f"racing {len(registered)} candidates: {', '.join(sorted(registered))}")

    cache = at.MeasureCache()
    at.load_corpus(cache=cache)          # warm-start: never drop other devices
    before = _summarise(cache)

    # Evict only the gfx1151 fused rows this driver owns, so the re-race is a
    # measurement rather than a cache hit. Everything else is left untouched.
    evicted = [k for k in list(cache._store)
               if str(k[0]) == DEVICE and k[2] == OP_FUSED_REGION]
    for key in evicted:
        del cache._store[key]
    # Decision #11: a row recorded under another (or no) toolchain identity is
    # never in `_store` -- `load_dict` holds it in `stale_records()`, where it is
    # already a miss. Count it too, or a v3 corpus reports "evicted 0" while every
    # row it owns is in fact about to be re-raced.
    stale_owned = [k for k in cache.stale_records()
                   if str(k[0]) == DEVICE and k[2] == OP_FUSED_REGION]
    print(f"evicted {len(evicted)} current gfx1151 fused_region rows; "
          f"{len(stale_owned)} stale (pre-toolchain-identity) rows will be "
          f"re-measured and replaced")

    # Refuse to key device rows under a bare target id. `_device_id` falls back
    # to "rocm" when the live device probe fails (no HIP runtime, no build
    # tree), and a row written there is invisible to every gfx1151 lookup.
    live_device = at._device_id("rocm")
    if live_device != DEVICE:
        print(f"REFUSING TO RECORD: the live device key is {live_device!r}, not "
              f"{DEVICE!r}. Source scripts/_rocm_env.sh and run from a tree whose "
              f"build/ (or TESSERA_BUILD_DIR) carries the ROCm runtime.")
        return 1
    from tessera.compiler import kernel_code_identity as KI
    print(f"device key {live_device}; disassembler {KI.find_llvm_objdump()}")

    region = F.FusedRegion(epilogue=("bias", "gelu"))
    rng = np.random.default_rng(0)
    unstamped: list[str] = []
    for size in args.shapes:
        a = rng.standard_normal((size, size)).astype(np.float32)
        b = rng.standard_normal((size, size)).astype(np.float32)
        bias = rng.standard_normal((size,)).astype(np.float32)
        for timing in (at.TIMING_END_TO_END, at.TIMING_DEVICE):
            # No explicit dims: `measured_arbitrate` infers (M, N, K) exactly as
            # `corpus_winner` does for ordinary `run_arbitrated` dispatch. The
            # rows recorded before 2026-09-26 passed dims=(size, size), a 2-D
            # bucket production never looked up, and one that omitted K.
            winner = at.measured_arbitrate(
                region, OP_FUSED_REGION, "rocm", a, b, bias,
                dtype="f16", cache=cache,
                reps=args.reps, warmup=args.warmup, timing=timing,
                device_repeats=args.device_repeats)
            if winner is None:
                print(f"  {size}x{size} {timing}: no verified candidate")
                continue
            dims = at._infer_dims(OP_FUSED_REGION, (a, b, bias))
            key = (DEVICE, "rocm", OP_FUSED_REGION,
                   at.bucket_key(dims, at.SpecPolicy.BUCKET), "f16", timing)
            rec = cache.get(key)
            # A timed candidate that must carry an artifact identity and was not
            # stamped makes the row unservable (every lookup misses on it), so
            # writing it would replace a good row with a dead one.
            live = candidates_live(region, (a, b, bias))
            stamped_names = set((rec.evidence.get("delegate_identities") or {})
                                if rec else ())
            for name in sorted(rec.candidates if rec else {}):
                cand = live.get(name)
                if (cand is not None and cand.requires_artifact_identity()
                        and name not in stamped_names):
                    unstamped.append(f"{size}^3 {timing}: {name}")
            sep = (rec.separation or {}) if rec else {}
            verdict = ("separated" if sep.get("separated")
                       else "NOT separated" if sep else "no verdict")
            print(f"  {size}x{size} {timing:10s}: {winner.name:22s} {verdict}"
                  f"  margin={(sep.get('margin') or 0)*100:.2f}%"
                  f" noise={(sep.get('noise') or 0)*100:.2f}%")
            # Decision #11: the artifact identity each timed candidate was
            # stamped with -- for rocm_wmma_gemm the kernel-code identity of
            # the image timed at this shape, not the tessera-opt binary; for
            # rocm_generic_hip the emitted-source identity (HIP text, its
            # kernel_cache key and the hipcc flags incl. offload arch).
            stamped = (rec.evidence.get("delegate_identities") or {}) if rec else {}
            for name, ident in sorted(stamped.items()):
                if ident.get("identity") == "emitted_source":
                    print(f"      {name}: source="
                          f"{str(ident.get('source_sha256', '-'))[:16]}"
                          f" cache_key={str(ident.get('cache_key', '-'))[:16]}"
                          f" build={ident.get('build', '-')!r}")
                    continue
                print(f"      {name}: stream="
                      f"{str(ident.get('instruction_stream_sha256', '-'))[:16]}"
                      f" data={ident.get('data_sections', '-')}"
                      f" kd={str(ident.get('kernel_descriptor_sha256', '-'))[:16]}"
                      f" n={ident.get('instruction_count', '-')}")

    # The gate this module's docstring promised and did not have. HIP events
    # on this fleet have been measured returning success while writing garbage,
    # and `_hip_resident_launch_latency` then falls back to the host wall clock
    # -- silently. Rows recorded that way would be saved as `timing="device"`
    # while carrying wall-clock numbers: the wrong timing domain, wearing the
    # right label, in published evidence.
    #
    # Documenting a gate that does not exist is worse than not mentioning one,
    # because a reader believes the write was checked. It was checked by hand
    # in a separate probe, which is exactly how the claim got written.
    if unstamped:
        print("\nREFUSING TO WRITE: candidates that require an artifact identity "
              "were timed but not stamped (no disassembler? no image?), so these "
              "rows could never be served:")
        for line in unstamped:
            print(f"  {line}")
        return 1

    source = rt.rocm_last_timer_source()
    if args.check_timer and source != "device_event":
        print(f"\nREFUSING TO WRITE: device rows were timed by {source!r}, not "
              f"'device_event'. A wall-clock fallback is a different timing "
              f"domain and must not be recorded as device timing. Re-run once "
              f"the HIP event path is healthy, or pass --no-check-timer if you "
              f"deliberately want wall-clock rows.")
        return 1
    print(f"\ndevice rows timed by: {source}")

    after = _summarise(cache)
    print("\n           rows  rocm  other  ranked  separated  unseparated  never_asked")
    for label, s in (("before", before), ("after", after)):
        print(f"  {label:7s}{s['rows']:5d}{s['rocm_rows']:6d}{s['other_rows']:7d}"
              f"{s['ranked']:8d}{s['separated']:11d}{s['unseparated']:13d}"
              f"{s['never_asked']:13d}")
    if after["other_rows"] < before["other_rows"]:
        print("\nREFUSING TO WRITE: this run would delete another device's rows.")
        return 1
    still_stale = sorted(
        (k for k in cache.stale_records()
         if str(k[0]) == DEVICE and k[2] == OP_FUSED_REGION), key=str)
    if still_stale:
        # A shape that produced no verified candidate leaves its stale row in
        # place. Writing would look like a full re-record while some rows
        # still select nothing, so say which.
        print(f"\nWARNING: {len(still_stale)} owned gfx1151 fused_region rows "
              f"were not re-measured and remain stale:")
        for key in still_stale:
            print(f"  {key}")
    if args.dry_run:
        print("\n--dry-run: corpus not written")
        return 0
    print(f"\nwrote {at.save_corpus(cache=cache)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
