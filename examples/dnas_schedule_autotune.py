"""Minimal DNAS + schedule-search prototype using tessera.arch.

This example keeps the math tiny and deterministic. It mirrors the compiler
contract: architecture choices live in Graph IR, schedule knobs live in Schedule
IR, and the hardware objective is supplied by a smooth cost model that can later
be trained from autotuner measurements.
"""

from __future__ import annotations

from tessera import arch


def main() -> None:
    attn = arch.MixedOp(
        ["flash_attention", "performer_attention", "multi_query_attention", "gmlp"],
        relax="softmax",
        temperature=2.0,
        name="block0.attn",
    )
    attn.alpha.set([0.2, -0.1, 0.0, 0.4])

    sched = arch.ScheduleSpace(
        {
            "tile_m": [64, 128],
            "tile_n": [128, 256],
            "tile_k": [32, 64],
            "stages": [2, 3, 4],
        },
        temperature=1.5,
    )
    sched.alpha["tile_m"].set([0.0, 1.0])
    sched.alpha["tile_n"].set([0.0, 1.0])
    sched.alpha["tile_k"].set([1.0, 0.0])
    sched.alpha["stages"].set([0.0, 1.0, 0.0])

    features = {
        "flops": 2.0e12,
        "bytes_moved": 1.2e9,
        "params": 45.0e6,
        "seq_len": 4096,
        "bandwidth_gbps": 2000.0,
        "peak_tflops": 312.0,
    }
    cost = arch.hw_cost(features, schedule=sched.current())

    choices_op = arch.argmax({"attn": attn})
    choices_sched = arch.schedule_argmax(sched)
    frozen = arch.specialize({"attn": attn}, choices_op)

    print("gates:", [round(v, 4) for v in attn.gates()])
    print("cost:", round(cost.latency_ms, 4), round(cost.energy, 4), int(cost.memory_bytes))
    print("frozen op:", frozen["attn"])
    print("frozen schedule:", choices_sched)


if __name__ == "__main__":
    main()
