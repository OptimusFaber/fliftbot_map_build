#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/optimus/ros-projects"
BUILD_ROOT="$PROJECT_ROOT/fliftbot_map_build"
RECORDING="$PROJECT_ROOT/ros-data/recordings/real/20261013/141857_dephan_m360_bs"
RESULTS_ROOT="$PROJECT_ROOT/ros-data/results/20261020"
CANONICAL="$RESULTS_ROOT/_canonical/141857_last18m"

cd "$PROJECT_ROOT"

if [[ -f "$BUILD_ROOT/.venv/bin/activate" ]]; then
    source "$BUILD_ROOT/.venv/bin/activate"
fi
if [[ -f /opt/ros/humble/setup.bash ]]; then
    set +u
    source /opt/ros/humble/setup.bash
    set -u
fi

mkdir -p "$RESULTS_ROOT" "$(dirname "$CANONICAL")"

# Create the canonical slice once and reuse it for all three LiDAR ranges.
if [[ ! -d "$CANONICAL/horizon" || ! -f "$CANONICAL/encoder.jsonl" ]]; then
    if [[ -d "$CANONICAL" ]] && [[ -n "$(find "$CANONICAL" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Incomplete non-empty canonical directory: $CANONICAL" >&2
        echo "Move or remove it, then run this script again." >&2
        exit 1
    fi

    read -r START_UNIX END_UNIX < <(
        python3 - "$RECORDING/canonical/recording_metadata.json" <<'PY'
import json
import sys

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
end = float(metadata["lidar"]["last_ts"])
start = max(float(metadata["lidar"]["first_ts"]), end - 18 * 60)
print(f"{start:.6f} {end:.6f}")
PY
    )

    python3 fliftbot_map_build/scripts/slice_canonical_interval.py \
        --recording "$RECORDING/canonical" \
        --start-unix "$START_UNIX" \
        --end-unix "$END_UNIX" \
        --output "$CANONICAL"
fi

export RESULTS_ROOT CANONICAL

printf '%s\n' 12 16 18 | xargs -P 3 -n 1 bash -c '
set -euo pipefail
lidar_range="$1"

python3 fliftbot_map_build/src/map_build/run_pipeline.py \
    --camera-config "$CANONICAL/camera.config" \
    --lidar-config "$CANONICAL/lidar.config" \
    --robot-config "$CANONICAL/liftbot.config" \
    --lidar-data "$CANONICAL/horizon" \
    --encoder "$CANONICAL/encoder.jsonl" \
    --output "$RESULTS_ROOT" \
    --name "141857_last18m_range${lidar_range}m" \
    --wheelbase 1.335 \
    --fork-position lowered \
    --lowered-forks-offset-m 0.05 \
    --max-range "$lidar_range" \
    --encoder-speed-sign 1 \
    --cartographer-config fliftbot_2d_offline.lua \
    --use-odometry \
    --no-markers
' _

echo "Last-18-minute maps finished. Results: $RESULTS_ROOT"
