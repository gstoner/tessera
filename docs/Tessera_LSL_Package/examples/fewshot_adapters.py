
# examples/fewshot_adapters.py
from tessera.lsl import spec, constraint, capability, solve

spec_fs = (spec.learning_objective("few_shot_classification",
                inputs={"x":"image[b,c,h,w]"}, outputs={"y":"class[b,k]"})
           + capability.few_shot(True, adapters="lora", lora_rank=16)
           + constraint.memory("<12GB"))

artifact = solve(spec_fs, target="nvidia:a100:sm80")
print(artifact["report"])
