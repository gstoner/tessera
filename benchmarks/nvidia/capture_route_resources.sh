#!/usr/bin/env bash
# Capture Nsight Compute route resources for the arbiter routes that had none
# in nvidia_sm120_test5_route_resources.json (AUTOTUNE-SM120-ROUTE-RESOURCES,
# sync AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27). One ncu report per route, each
# holding only that route's launches (profile_route_resources.py), normalized
# by parse_ncu_resources.py and added by build_test5_resource_manifest.py
# --route. Run on The-Super-Bear from the repo root with the venv active and
# scripts/_nvidia_env.sh sourced; device work runs under the timing lock.
#
#   bash benchmarks/nvidia/capture_route_resources.sh [--refresh] OUT_DIR
#
# Default: ADD the routes below to the committed manifest. Since
# AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27 committed them, the default run now
# refuses BEFORE any capture (the preflight names the routes already present).
# `--refresh`: RE-CAPTURE exactly these routes and replace only their entries
# (routes, details, and the sources tagged with them); every other route and
# source is kept, and no route ends up with an old and a new capture mixed.
set -euo pipefail
REFRESH=()
if [ "${1:-}" = "--refresh" ]; then
  REFRESH=(--refresh)
  shift
fi
OUT="${1:?usage: capture_route_resources.sh [--refresh] OUT_DIR}"
NCU="${NCU:-/usr/local/cuda/bin/ncu}"
PY="${PYTHON:-python}"
LOCK="${TESSERA_TIMING_LOCK:-/tmp/tessera-timing.lock}"
BASE="${TESSERA_ROUTE_RESOURCES_BASE:-benchmarks/baselines/nvidia_sm120_test5_route_resources.json}"

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

names=()
for spec in "${ROUTES[@]}"; do
  read -r route _ <<<"$spec"
  names+=("$route")
done
# Fail fast: decide whether assembly can succeed before any expensive capture.
"$PY" benchmarks/nvidia/build_test5_resource_manifest.py --base "$BASE" \
  ${REFRESH[@]+"${REFRESH[@]}"} --check-routes "${names[@]}"
mkdir -p "$OUT"

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
"$PY" benchmarks/nvidia/build_test5_resource_manifest.py --base "$BASE" \
  ${REFRESH[@]+"${REFRESH[@]}"} "${args[@]}" --output "$OUT/nvidia_sm120_test5_route_resources.json"
echo "wrote $OUT/nvidia_sm120_test5_route_resources.json"
