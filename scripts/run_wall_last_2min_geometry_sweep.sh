#!/usr/bin/env bash
set -euo pipefail

# Sweep wheelbase and longitudinal LiDAR geometry for wall_last_2min.
#
# For every combination:
#   effective_wheelbase = wheelbase + delta_x
#   effective_lidar_x   = source_lidar_x + delta_x - extra_delta
#
# Camera geometry is left unchanged.  The pipeline is therefore run with
# fork-position=raised and receives the already calculated effective values.

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")

SOURCE=${SOURCE:-$PROJECT_ROOT/ros-data/results/real/20261016/wall_last_2min}
OUT_ROOT=${OUT_ROOT:-$PROJECT_ROOT/ros-data/results/real/20261016/wall_last_2min_geometry_sweep_60m}
JOBS=${JOBS:-10}
MAX_RANGE_M=${MAX_RANGE_M:-60}
ENCODER_SPEED_SIGN=1

WHEELBASES_M=(1.380 1.385 1.390 1.395 1.400 1.405 1.410)
DELTA_X_M=(0.000 0.010 0.020 0.030 0.040 0.050)
EXTRA_DELTA_M=(0.000 0.010 0.020 0.030)

usage() {
    echo "Usage: $0 [--plan | --worker WHEELBASE DELTA_X EXTRA_DELTA]"
}

number_tag() {
    awk -v value="$1" 'BEGIN { printf "%04d", int(value * 1000 + 0.5) }'
}

calculate() {
    awk -v a="$1" -v b="$2" 'BEGIN { printf "%.6f", a + b }'
}

calculate_lidar_x() {
    awk -v base="$1" -v dx="$2" -v extra="$3" \
        'BEGIN { printf "%.6f", base + dx - extra }'
}

source_lidar_x() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["position_m"]["x"])' "$SOURCE/lidar.config"
}

run_one() {
    local wheelbase_m=$1
    local delta_x_m=$2
    local extra_delta_m=$3
    local effective_wheelbase_m
    local effective_lidar_x_m
    local wb_tag
    local dx_tag
    local extra_tag
    local run_name
    local lidar_config
    local log_path
    local done_path
    local failed_path

    effective_wheelbase_m=$(calculate "$wheelbase_m" "$delta_x_m")
    effective_lidar_x_m=$(calculate_lidar_x "$(source_lidar_x)" "$delta_x_m" "$extra_delta_m")
    wb_tag=$(number_tag "$wheelbase_m")
    dx_tag=$(number_tag "$delta_x_m")
    extra_tag=$(number_tag "$extra_delta_m")
    run_name="wall_last2m_r60_wb${wb_tag}_dx${dx_tag}_extra${extra_tag}"
    lidar_config="$OUT_ROOT/configs/lidar_${run_name}.config"
    log_path="$OUT_ROOT/logs/${run_name}.log"
    done_path="$OUT_ROOT/status/${run_name}.done"
    failed_path="$OUT_ROOT/status/${run_name}.failed"

    if [[ -f "$done_path" ]]; then
        echo "SKIP $run_name (already complete)"
        return 0
    fi

    python3 -c '
import json, sys
source, target, lidar_x, max_range = sys.argv[1:]
with open(source, encoding="utf-8") as stream:
    config = json.load(stream)
config["position_m"]["x"] = float(lidar_x)
config["max_range_m"] = float(max_range)
config["encoder_speed_sign"] = 1.0
with open(target, "w", encoding="utf-8") as stream:
    json.dump(config, stream, indent=2, ensure_ascii=False)
    stream.write("\n")
' "$SOURCE/lidar.config" "$lidar_config" "$effective_lidar_x_m" "$MAX_RANGE_M"

    rm -f "$failed_path"
    {
        echo "START $(date --iso-8601=seconds)"
        echo "wheelbase_base_m=$wheelbase_m"
        echo "delta_x_m=$delta_x_m"
        echo "extra_delta_m=$extra_delta_m"
        echo "effective_wheelbase_m=$effective_wheelbase_m"
        echo "effective_lidar_x_m=$effective_lidar_x_m"
        echo "encoder_speed_sign=$ENCODER_SPEED_SIGN"

        python3 "$PROJECT_ROOT/fliftbot_map_build/src/map_build/run_pipeline.py" \
            --camera-config "$SOURCE/camera.config" \
            --lidar-config "$lidar_config" \
            --robot-config "$SOURCE/liftbot.config" \
            --lidar-data "$SOURCE/horizon" \
            --encoder "$SOURCE/encoder.jsonl" \
            --output "$OUT_ROOT/maps" \
            --name "$run_name" \
            --max-range "$MAX_RANGE_M" \
            --wheelbase "$effective_wheelbase_m" \
            --encoder-speed-sign "$ENCODER_SPEED_SIGN" \
            --fork-position raised \
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
    for required in camera.config camera_intrinsics.json lidar.config liftbot.config encoder.jsonl; do
        [[ -f "$SOURCE/$required" ]] || { echo "Missing input: $SOURCE/$required" >&2; exit 2; }
    done
    [[ -d "$SOURCE/horizon" ]] || { echo "Missing input directory: $SOURCE/horizon" >&2; exit 2; }
}

write_plan() {
    local base_lidar_x
    base_lidar_x=$(source_lidar_x)
    {
        printf 'wheelbase_m\tdelta_x_m\textra_delta_m\teffective_wheelbase_m\teffective_lidar_x_m\n'
        for wheelbase_m in "${WHEELBASES_M[@]}"; do
            for delta_x_m in "${DELTA_X_M[@]}"; do
                for extra_delta_m in "${EXTRA_DELTA_M[@]}"; do
                    printf '%s\t%s\t%s\t%s\t%s\n' \
                        "$wheelbase_m" "$delta_x_m" "$extra_delta_m" \
                        "$(calculate "$wheelbase_m" "$delta_x_m")" \
                        "$(calculate_lidar_x "$base_lidar_x" "$delta_x_m" "$extra_delta_m")"
                done
            done
        done
    } >"$OUT_ROOT/sweep_plan.tsv"
}

if [[ ${1:-} == "--worker" ]]; then
    [[ $# -eq 4 ]] || { usage >&2; exit 2; }
    validate_inputs
    mkdir -p "$OUT_ROOT/configs" "$OUT_ROOT/logs" "$OUT_ROOT/maps" "$OUT_ROOT/status"
    run_one "$2" "$3" "$4"
    exit $?
fi

if [[ $# -gt 1 ]] || [[ $# -eq 1 && $1 != "--plan" ]]; then
    usage >&2
    exit 2
fi

validate_inputs
mkdir -p "$OUT_ROOT/configs" "$OUT_ROOT/logs" "$OUT_ROOT/maps" "$OUT_ROOT/status"
write_plan

echo "Sweep combinations: 168"
echo "Parallel jobs:      $JOBS"
echo "Output:             $OUT_ROOT"
echo "Plan:               $OUT_ROOT/sweep_plan.tsv"

if [[ ${1:-} == "--plan" ]]; then
    exit 0
fi

args=()
for wheelbase_m in "${WHEELBASES_M[@]}"; do
    for delta_x_m in "${DELTA_X_M[@]}"; do
        for extra_delta_m in "${EXTRA_DELTA_M[@]}"; do
            args+=("$wheelbase_m" "$delta_x_m" "$extra_delta_m")
        done
    done
done

printf '%s\0' "${args[@]}" | xargs -0 -n 3 -P "$JOBS" "$SCRIPT_PATH" --worker
echo "Sweep finished successfully."
