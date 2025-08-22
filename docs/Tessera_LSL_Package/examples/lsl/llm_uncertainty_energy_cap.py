#!/usr/bin/env python3
"""LLM with uncertainty and energy cap LSL example (prototype)."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import tessera.lsl as lsl

spec = lsl.learning_objective(
    task="language_modeling",
    description="next-token prediction with calibrated uncertainty",
    inputs={"tokens":"int[b,t]"}, outputs={"next":"int[b,t]"},
    inductive_bias=["transformer"]
)
spec += lsl.constraint.energy("<0.4J/100tok@H100")
spec += lsl.constraint.latency("<10ms/seq@H100:bs=1")
spec += lsl.capability.uncertainty(outputs=("entropy","mutual_information","mean","std"))
spec += lsl.capability.robustness(shifts=["length:up_to_8k","minor_bpe_noise"])

artifact = lsl.solve(spec, target="nvidia:h100:sm90")
print("Report:", artifact["report"])