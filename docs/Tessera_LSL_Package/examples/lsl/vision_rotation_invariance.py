#!/usr/bin/env python3
"""Vision rotation invariance LSL example (prototype)."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import tessera.lsl as lsl

spec = lsl.learning_objective(
    task="image_classification",
    description="classify images with rotation invariance",
    inputs={"x":"image[b,c,h,w]"}, outputs={"y":"class[b,k]"},
    inductive_bias=["cnn","group-equivariance:so2"]
)
spec += lsl.constraint.latency("<50ms@A100:bs=32")
spec += lsl.constraint.accuracy(">0.95@CIFAR10/val")
spec += lsl.constraint.memory("<6GB")
spec += lsl.capability.few_shot(True, k=16)
spec += lsl.capability.uncertainty(outputs=("mean","std","epistemic","aleatoric"))
spec += lsl.capability.interpretability(methods=("grad_cam","integrated_gradients"))

artifact = lsl.solve(spec, target="nvidia:a100:sm80")
print("Report:", artifact["report"])