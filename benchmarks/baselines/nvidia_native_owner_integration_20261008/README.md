# Native owner integration
Owner W1.1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Sync NVIDIA-NATIVE-OWNER-FOLLOWUP-2026-10-08.

This integration combines the independently proved retained softmax staging and static macro-CTA prepared producer owner. Source identities and pre-integration evidence are recorded separately. The 2.270 median control/candidate ratio is measured public host-call time for the staging change, not a combined-build kernel speedup. Pre-integration macro evidence includes original long BF16 gates, typed/dynamic/portable owner regressions and twelve changed-input/output-lifetime cases. Fresh matching-source combined runtime proof remains pending.

Matching combined runtime builds with SHA256 ca860af9503bbb4733c64d3a1c51634de7f7f1e09a7e3ad7720c8c6f67c58d60. Owning device and contract gates are running. The compiler image stays unchanged.

## Matching-source execution and characterization

448 owning RTX 5070 checks and 397 shared contract/registry gates pass. Twelve separate-stage benchmark profiles pass independent numerics before/after timing. Four K4096 column-major profiles use macro consumers with two/three producers and FP16/BF16 storage. Maximum absolute error across all twelve is 1.48945e-5. Long profile timings (milliseconds): [{"storage": "fp16", "producer_count": 2, "stage_ms": [0.010847999714314938, 0.9917700290679932, 0.057193998247385025], "prepared_host_wall_ms": 1.6368913988117129, "max_abs_error": 1.0340379574813596e-06}, {"storage": "fp16", "producer_count": 3, "stage_ms": [0.010604999959468842, 0.010017000138759613, 0.991320013999939, 0.05705200135707855], "prepared_host_wall_ms": 1.4330739999422804, "max_abs_error": 1.9940613356084214e-06}, {"storage": "bf16", "producer_count": 2, "stage_ms": [0.009393000043928623, 0.9921200275421143, 0.05721199885010719], "prepared_host_wall_ms": 1.4386271010152996, "max_abs_error": 1.4178317542246077e-05}, {"storage": "bf16", "producer_count": 3, "stage_ms": [0.009891999885439873, 0.010975999757647514, 0.9922620058059692, 0.057071998715400696], "prepared_host_wall_ms": 1.4325059979455546, "max_abs_error": 1.4894492963435368e-05}]. Stage events and prepared host wall measure different scopes and are not added into a synthetic whole-program device time. No A/B kernel promotion claim.

The initial recorder incorrectly uploaded column-major RHS as row-major; it failed descriptor validation before the corrected recorder consumed the compiler-owned layout. Both logs are retained. Native images, numerical gates and descriptor checks were preserved.
