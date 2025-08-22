
# examples/rotation_invariance.py
from tessera.lsl import spec, constraint, capability, solve

model = spec.learning_objective(
    task="image_classification",
    description="classify images with rotation invariance",
    inputs={"x":"image[b,c,h,w]"}, outputs={"y":"class[b,k]"},
    inductive_bias=["cnn","group-equivariance:so2"]
)

model += constraint.latency("<50ms@A100:bs=32")
model += constraint.accuracy(">0.95@CIFAR10/val")
model += capability.few_shot(True, k=16)
model += capability.uncertainty(("mean","std","epistemic","aleatoric"))

artifact = solve(model, target="nvidia:a100:sm80")
print(artifact["report"])
