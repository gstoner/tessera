<<<MERGE_START: Tessera_TPU_Shardy_Export>>>
# Shardy-Native Sharding Export

This pass attaches `sdy.mesh` and `sdy.tensor_sharding` attributes so downstream
Shardy/GSPMD passes can consume Tessera’s partitioning intent.

- Module attr: `sdy.mesh = "mesh = {axes=[\"data\",\"model\"], shape=[D,M]}"` (example)
- Per-op attr: `sdy.tensor_sharding = "{sharding = replicated}"` (starter)

Replace these with real mesh/axes inferred from Tessera’s `tile.distribute` and IR attrs.

<<<MERGE_END: Tessera_TPU_Shardy_Export>>>
