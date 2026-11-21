#!/usr/bin/env bash
# Never starts robot hardware. Override asset/data roots without editing recipes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export HOLOMOTION_TRAIN_PREFIX="${HOLOMOTION_TRAIN_PREFIX:-$PWD/.venv-omomo}"
set +u
source train.env
set -u
ROBOT="${1:?Usage: bash run_omomo_box.sh g1|r1 [preflight|train]}"
MODE="${2:-preflight}"
HOLOSOMA_ROOT="${HOLOSOMA_ROOT:-$PWD/../holosoma}"
case "$ROBOT" in
 g1) ROBOT_URDF="${ROBOT_URDF:-$HOLOSOMA_ROOT/src/holosoma/holosoma/data/robots/g1/main_mesh_collision_halfspherehand.urdf}"; MOTION_MANIFEST="${MOTION_MANIFEST:-data/omomo-g1-mesh/manifest.json}" ;;
 r1) ROBOT_URDF="${ROBOT_URDF:-$HOLOSOMA_ROOT/src/holosoma_retargeting/holosoma_retargeting/demo_data/models/r1/r1_26dof.urdf}" ;;
 *) exit 2 ;;
esac
RUN_OUTPUT="${RUN_OUTPUT:-logs/omomo-box/${ROBOT}-${MODE}-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN_OUTPUT"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/not_for_commit/omomo-cache}"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
ARGS=(--manifest "${MOTION_MANIFEST:-data/omomo-$ROBOT/manifest.json}" --robot-urdf "$ROBOT_URDF"
      --output "$RUN_OUTPUT/training" --physics-backend isaac_physx
      --num-envs "${NUM_ENVS:-128}" --buffer-steps 500 --horizon 24
      --wm-batch 256 --pi-batch 512 --pi-microbatch 128 --wm-batches 10
      --world-only-updates 100 --lr 0.0004 --max-hours "${MAX_HOURS:-0.5}"
      --no-capture-graph --seed 42)
case "$MODE" in
 preflight) ARGS+=(--num-envs 2 --preflight-only --warmup-steps 4 --no-compile) ;;
 train) ;;
 *) exit 2 ;;
esac
ARGS+=("${@:3}")
exec "$Train_CONDA_PREFIX/bin/python" -m holomotion.scripts.training.train_supermimic "${ARGS[@]}"
