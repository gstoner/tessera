#!/usr/bin/env bash
# Capture Nsight Compute route resources for the arbiter routes that had none
# in nvidia_sm120_test5_route_resources.json (AUTOTUNE-SM120-ROUTE-RESOURCES,
# sync AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27). One ncu report per route, each
# holding only that route's launches (profile_route_resources.py), normalized
# by parse_ncu_resources.py and added by build_test5_resource_manifest.py
# --route. Run on The-Super-Bear from the repo root with the venv active and
# scripts/_nvidia_env.sh sourced; device work runs under the timing lock.
#
#   bash benchmarks/nvidia/capture_route_resources.sh OUT_DIR
set -euo pipefail
OUT="${1:?usage: capture_route_resources.sh OUT_DIR}"
NCU="${NCU:-/usr/local/cuda/bin/ncu}"
PY="${PYTHON:-python}"
LOCK="${TESSERA_TIMING_LOCK:-/tmp/tessera-timing.lock}"
mkdir -p "$OUT"

# route  op  storage
ROUTES=(
  "nvidia_mma_fused_tf32 fused_region f32"
  "nvidia_mma_fused_fp8_e4m3 fused_region fp8_e4m3"
  "nvidia_mma_fused_fp8_e5m2 fused_region fp8_e5m2"
  "nvidia_mma_fused_composed_fp8_e4m3 fused_region fp8_e4m3"
  "nvidia_mma_fused_composed_fp8_e5m2 fused_region fp8_e5m2"
  "nvidia_mma_attn_tf32 attention f32"
  "nvidia_mma_attn_fp8_e4m3 attention fp8_e4m3"
  "nvidia_mma_attn_fp8_e5m2 attention fp8_e5m2"
  "nvidia_mma_attn_composed_fp8_e4m3 attention fp8_e4m3"
  "nvidia_mma_attn_composed_fp8_e5m2 attention fp8_e5m2"
  "nvidia_flash_attn attention f16"
  "nvidia_mma_gated_tf32 gated_matmul f32"
  "nvidia_mma_gated_fp8_e4m3 gated_matmul fp8_e4m3"
  "nvidia_mma_gated_fp8_e5m2 gated_matmul fp8_e5m2"
  "nvidia_mma_gated_composed_fp8_e4m3 gated_matmul fp8_e4m3"
  "nvidia_mma_gated_composed_fp8_e5m2 gated_matmul fp8_e5m2"
  "nvidia_gated gated_matmul f32"
)

args=()
for spec in "${ROUTES[@]}"; do
  read -r route op storage <<<"$spec"
  echo "== $route ($op, $storage)"
  flock "$LOCK" "$NCU" --profile-from-start off --set full -f -o "$OUT/$route" \
    "$PY" benchmarks/nvidia/profile_route_resources.py \
      --candidate "$route" --op "$op" --storage "$storage" \
    > "$OUT/$route.ncu.txt" 2>&1
  report="$OUT/$route.ncu-repz"
  [ -f "$report" ] || report="$OUT/$route.ncu-rep"
  "$PY" benchmarks/nvidia/parse_ncu_resources.py "$report" --ncu "$NCU" \
    --output "$OUT/$route.json"
  args+=(--route "$route=$OUT/$route.json")
done
"$PY" benchmarks/nvidia/build_test5_resource_manifest.py \
  --base benchmarks/baselines/nvidia_sm120_test5_route_resources.json \
  "${args[@]}" --output "$OUT/nvidia_sm120_test5_route_resources.json"
echo "wrote $OUT/nvidia_sm120_test5_route_resources.json"
