#!/usr/bin/env bash
set -euo pipefail

# Sweep lowered-fork geometry for the 2026-10-13 16:30:21 BS recording.
#
# The supplied wheelbase is the raised-forks/base value. run_pipeline.py then
# adds the selected lowered-forks offset to camera.x, lidar.x and wheelbase.
# Thus wb=1.390 and offset=0.030 produce an effective wheelbase of 1.420 m.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")

OUT_ROOT=${OUT_ROOT:-/home/optimus/combinations}
RECORDING=${RECORDING:-$PROJECT_ROOT/ros-data/recordings/real/20261013/163021_dephan_m360_bs/canonical}
CONFIG_DIR=${CONFIG_DIR:-$PROJECT_ROOT/fliftbot_map_build/configs/bs_version}
JOBS=${JOBS:-8}
DRY_RUN=${DRY_RUN:-0}

OFFSETS_M=(0.020 0.025 0.030 0.035 0.040 0.045 0.050)
WHEELBASES_M=(1.380 1.385 1.390 1.395 1.400)

run_one() {
    local offset_m=$1
    local wheelbase_m=$2
    local offset_cm
    local wheelbase_tag
    local run_name
    local log_path

    offset_cm=$(awk -v value="$offset_m" 'BEGIN { printf "%g", value * 100 }')
    wheelbase_tag=${wheelbase_m/./_}
    run_name="163021_bs_30m_forks_down_${offset_cm}cm_wb_${wheelbase_tag}"
    log_path="$OUT_ROOT/logs/${run_name}.log"

    local command=(
        python3 "$PROJECT_ROOT/fliftbot_map_build/src/map_build/run_pipeline.py"
        --camera-config "$CONFIG_DIR/camera.config"
        --lidar-config "$CONFIG_DIR/lidar.config"
        --robot-config "$CONFIG_DIR/liftbot.config"
        --lidar-data "$RECORDING/horizon"
        --encoder "$RECORDING/encoder.jsonl"
        --output "$OUT_ROOT"
        --name "$run_name"
        --max-range 30
        --wheelbase "$wheelbase_m"
        --fork-position lowered
        --lowered-forks-offset-m "$offset_m"
        --encoder-speed-sign 1
        --steering-sign 1
        --mapping-yaw 180
        --no-markers
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        command+=(--dry-run)
    fi

    echo "[$(date --iso-8601=seconds)] START offset=${offset_m}m base_wheelbase=${wheelbase_m}m name=${run_name}" | tee "$log_path"
    if "${command[@]}" 2>&1 | tee -a "$log_path"; then
        echo "[$(date --iso-8601=seconds)] OK ${run_name}" | tee -a "$log_path"
    else
        local status=${PIPESTATUS[0]}
        echo "[$(date --iso-8601=seconds)] FAILED status=${status} ${run_name}" | tee -a "$log_path" >&2
        return "$status"
    fi
}

if [[ ${1:-} == "--worker" ]]; then
    [[ $# -eq 3 ]] || { echo "worker needs: offset_m wheelbase_m" >&2; exit 2; }
    mkdir -p "$OUT_ROOT/logs"
    run_one "$2" "$3"
    exit $?
fi

[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be a positive integer" >&2; exit 2; }
for required in \
    "$CONFIG_DIR/camera.config" \
    "$CONFIG_DIR/lidar.config" \
    "$CONFIG_DIR/liftbot.config" \
    "$RECORDING/encoder.jsonl"; do
    [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 2; }
done
[[ -d "$RECORDING/horizon" ]] || { echo "Missing horizon directory: $RECORDING/horizon" >&2; exit 2; }

mkdir -p "$OUT_ROOT/logs"
PLAN="$OUT_ROOT/sweep_plan.tsv"
{
    printf 'lowered_offset_m\tbase_wheelbase_m\teffective_wheelbase_m\n'
    for offset_m in "${OFFSETS_M[@]}"; do
        for wheelbase_m in "${WHEELBASES_M[@]}"; do
            effective=$(awk -v a="$offset_m" -v b="$wheelbase_m" 'BEGIN { printf "%.3f", a + b }')
            printf '%s\t%s\t%s\n' "$offset_m" "$wheelbase_m" "$effective"
        done
    done
} > "$PLAN"

export OUT_ROOT RECORDING CONFIG_DIR JOBS DRY_RUN
echo "Launching 35 experiments with JOBS=$JOBS"
echo "Output: $OUT_ROOT"
echo "Plan:   $PLAN"

args=()
for offset_m in "${OFFSETS_M[@]}"; do
    for wheelbase_m in "${WHEELBASES_M[@]}"; do
        args+=("$offset_m" "$wheelbase_m")
    done
done

printf '%s\0' "${args[@]}" | xargs -0 -n 2 -P "$JOBS" "$SCRIPT_PATH" --worker
echo "All 35 experiments completed successfully."
