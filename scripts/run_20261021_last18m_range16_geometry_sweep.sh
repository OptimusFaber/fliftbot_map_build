#!/usr/bin/env bash
set -euo pipefail

# Geometry sweep for the staged 141857 last-18-minute recording.
#
# For every combination:
#   effective_wheelbase = wheelbase + delta_x
#   effective_camera_x  = source camera.x + delta_x
#   effective_lidar_x   = source lidar.x + delta_x
#
# extra_delta is fixed at zero and is not part of this sweep.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")

SOURCE_RUN=${SOURCE_RUN:-$PROJECT_ROOT/ros-data/results/real/20261019/20261020_144904_141857_last18m_range12m}
SOURCE_INPUT=${SOURCE_INPUT:-$SOURCE_RUN/input}
OUT_ROOT=${OUT_ROOT:-$PROJECT_ROOT/ros-data/results/real/20261021/last18m_range16_geometry_sweep}
JOBS=${JOBS:-10}
MAX_RANGE_M=16
ENCODER_SPEED_SIGN=1

WHEELBASES_M=(1.380 1.385 1.390 1.395 1.400 1.405 1.410)
DELTA_X_M=(0.000 0.010 0.020 0.030 0.040 0.050)

usage() {
    echo "Usage: $0 [--plan | --worker WHEELBASE DELTA_X]"
}

number_tag() {
    awk -v value="$1" 'BEGIN { printf "%04d", int(value * 1000 + 0.5) }'
}

add_values() {
    awk -v a="$1" -v b="$2" 'BEGIN { printf "%.6f", a + b }'
}

prepare_base_configs() {
    python3 - "$SOURCE_RUN/run_parameters.json" "$SOURCE_INPUT/camera.json" "$OUT_ROOT/configs" <<'PY'
import json
import sys
from pathlib import Path

run_parameters = Path(sys.argv[1])
camera_intrinsics = Path(sys.argv[2]).resolve()
config_dir = Path(sys.argv[3])
parameters = json.loads(run_parameters.read_text(encoding="utf-8"))

camera = parameters["camera"]
camera["intrinsics"] = str(camera_intrinsics)

lidar = parameters["lidar"]
lidar["max_range_m"] = 16.0
lidar["encoder_speed_sign"] = 1.0

robot = parameters["robot"]

for name, config in (("camera.config", camera), ("lidar.config", lidar), ("liftbot.config", robot)):
    (config_dir / name).write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
PY
}

run_one() {
    local wheelbase_m=$1
    local delta_x_m=$2
    local effective_wheelbase_m
    local wb_tag
    local dx_tag
    local run_name
    local log_path
    local done_path
    local failed_path

    effective_wheelbase_m=$(add_values "$wheelbase_m" "$delta_x_m")
    wb_tag=$(number_tag "$wheelbase_m")
    dx_tag=$(number_tag "$delta_x_m")
    run_name="last18m_r16_wb${wb_tag}_dx${dx_tag}"
    log_path="$OUT_ROOT/logs/${run_name}.log"
    done_path="$OUT_ROOT/status/${run_name}.done"
    failed_path="$OUT_ROOT/status/${run_name}.failed"

    if [[ -f "$done_path" ]]; then
        echo "SKIP $run_name (already complete)"
        return 0
    fi

    rm -f "$failed_path"
    {
        echo "START $(date --iso-8601=seconds)"
        echo "wheelbase_m=$wheelbase_m"
        echo "delta_x_m=$delta_x_m"
        echo "effective_wheelbase_m=$effective_wheelbase_m"
        echo "encoder_speed_sign=$ENCODER_SPEED_SIGN"

        python3 "$PROJECT_ROOT/fliftbot_map_build/src/map_build/run_pipeline.py" \
            --camera-config "$OUT_ROOT/configs/camera.config" \
            --lidar-config "$OUT_ROOT/configs/lidar.config" \
            --robot-config "$OUT_ROOT/configs/liftbot.config" \
            --lidar-data "$SOURCE_INPUT/horizon" \
            --encoder "$SOURCE_INPUT/encoder.jsonl" \
            --output "$OUT_ROOT/maps" \
            --name "$run_name" \
            --max-range "$MAX_RANGE_M" \
            --wheelbase "$wheelbase_m" \
            --fork-position lowered \
            --lowered-forks-offset-m "$delta_x_m" \
            --encoder-speed-sign "$ENCODER_SPEED_SIGN" \
            --no-markers

        echo "DONE $(date --iso-8601=seconds)"
    } >"$log_path" 2>&1 && {
        printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >"$done_path"
        echo "OK $run_name"
    } || {
        local status=$?
        printf 'failed_at=%s\nexit_status=%s\n' "$(date --iso-8601=seconds)" "$status" >"$failed_path"
        echo "FAILED $run_name status=$status log=$log_path" >&2
        return "$status"
    }
}

validate_inputs() {
    [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be a positive integer" >&2; exit 2; }
    for required in run_parameters.json input/camera.json input/encoder.jsonl; do
        [[ -f "$SOURCE_RUN/$required" ]] || { echo "Missing input: $SOURCE_RUN/$required" >&2; exit 2; }
    done
    [[ -d "$SOURCE_INPUT/horizon" ]] || { echo "Missing input directory: $SOURCE_INPUT/horizon" >&2; exit 2; }
}

write_plan() {
    local source_camera_x
    local source_lidar_x
    source_camera_x=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["camera"]["position_m"]["x"])' "$SOURCE_RUN/run_parameters.json")
    source_lidar_x=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["lidar"]["position_m"]["x"])' "$SOURCE_RUN/run_parameters.json")
    {
        printf 'wheelbase_m\tdelta_x_m\teffective_wheelbase_m\teffective_camera_x_m\teffective_lidar_x_m\n'
        for wheelbase_m in "${WHEELBASES_M[@]}"; do
            for delta_x_m in "${DELTA_X_M[@]}"; do
                printf '%s\t%s\t%s\t%s\t%s\n' \
                    "$wheelbase_m" "$delta_x_m" \
                    "$(add_values "$wheelbase_m" "$delta_x_m")" \
                    "$(add_values "$source_camera_x" "$delta_x_m")" \
                    "$(add_values "$source_lidar_x" "$delta_x_m")"
            done
        done
    } >"$OUT_ROOT/sweep_plan.tsv"
}

if [[ ${1:-} == "--worker" ]]; then
    [[ $# -eq 3 ]] || { usage >&2; exit 2; }
    validate_inputs
    mkdir -p "$OUT_ROOT/configs" "$OUT_ROOT/logs" "$OUT_ROOT/maps" "$OUT_ROOT/status"
    [[ -f "$OUT_ROOT/configs/camera.config" ]] || prepare_base_configs
    run_one "$2" "$3"
    exit $?
fi

if [[ $# -gt 1 ]] || [[ $# -eq 1 && $1 != "--plan" ]]; then
    usage >&2
    exit 2
fi

validate_inputs
mkdir -p "$OUT_ROOT/configs" "$OUT_ROOT/logs" "$OUT_ROOT/maps" "$OUT_ROOT/status"
prepare_base_configs
write_plan

echo "Sweep combinations: 42"
echo "Parallel jobs:      $JOBS"
echo "LiDAR max range:    $MAX_RANGE_M m"
echo "Encoder speed sign: +$ENCODER_SPEED_SIGN"
echo "Output:             $OUT_ROOT"
echo "Plan:               $OUT_ROOT/sweep_plan.tsv"

if [[ ${1:-} == "--plan" ]]; then
    exit 0
fi

args=()
for wheelbase_m in "${WHEELBASES_M[@]}"; do
    for delta_x_m in "${DELTA_X_M[@]}"; do
        args+=("$wheelbase_m" "$delta_x_m")
    done
done

printf '%s\0' "${args[@]}" | xargs -0 -n 2 -P "$JOBS" "$SCRIPT_PATH" --worker
echo "Sweep finished successfully."
