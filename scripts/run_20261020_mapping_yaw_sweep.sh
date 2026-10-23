#!/usr/bin/env bash
set -euo pipefail

# Test Cartographer mapping yaw on the last 18 minutes of recording 141857.
# Four maps (-90, 0, 90, 180 deg) are built in parallel.

PROJECT_ROOT="/home/optimus/ros-projects"
BUILD_ROOT="$PROJECT_ROOT/fliftbot_map_build"
RECORDING="$PROJECT_ROOT/ros-data/recordings/real/20261013/141857_dephan_m360_bs"
RESULTS_ROOT="$PROJECT_ROOT/ros-data/results/20261020/yaw_sweep_last18m_range12m"
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

# Build an independent canonical interval so the sweep is reproducible.
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

with open(sys.argv[1], encoding="utf-8") as stream:
    metadata = json.load(stream)

recording_start = float(metadata["lidar"]["first_ts"])
recording_end = float(metadata["lidar"]["last_ts"])
interval_start = max(recording_start, recording_end - 18 * 60)
print(f"{interval_start:.6f} {recording_end:.6f}")
PY
    )

    python3 fliftbot_map_build/scripts/slice_canonical_interval.py \
        --recording "$RECORDING/canonical" \
        --start-unix "$START_UNIX" \
        --end-unix "$END_UNIX" \
        --output "$CANONICAL"
fi

# Guard against stale canonical data created before the corrected LiDAR X.
python3 - "$CANONICAL/lidar.config" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as stream:
    config = json.load(stream)

actual = float(config["position_m"]["x"])
expected = 1.048
if abs(actual - expected) > 1e-9:
    raise SystemExit(
        f"{path}: expected base LiDAR X={expected:.3f} m, got {actual:.3f} m"
    )
PY

export RESULTS_ROOT CANONICAL

printf '%s\n' -90 0 90 180 | xargs -P 4 -n 1 bash -c '
set -euo pipefail
yaw="$1"

if [[ "$yaw" == -* ]]; then
    yaw_name="minus${yaw#-}"
else
    yaw_name="$yaw"
fi

python3 fliftbot_map_build/src/map_build/run_pipeline.py \
    --camera-config "$CANONICAL/camera.config" \
    --lidar-config "$CANONICAL/lidar.config" \
    --robot-config "$CANONICAL/liftbot.config" \
    --lidar-data "$CANONICAL/horizon" \
    --encoder "$CANONICAL/encoder.jsonl" \
    --output "$RESULTS_ROOT" \
    --name "141857_last18m_range12m_mapping_yaw_${yaw_name}deg" \
    --wheelbase 1.335 \
    --fork-position lowered \
    --lowered-forks-offset-m 0.05 \
    --max-range 12 \
    --encoder-speed-sign 1 \
    --mapping-yaw "$yaw" \
    --cartographer-config fliftbot_2d_offline.lua \
    --use-odometry \
    --no-markers
' _

# Produce a compact machine-readable/text summary after all runs finish.
python3 - "$RESULTS_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for stats_path in sorted(root.glob("*/map/run_stats.json")):
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    run = stats_path.parent.parent
    parameters = json.loads((run / "run_parameters.json").read_text(encoding="utf-8"))
    yaw = parameters["lidar"]["mapping_orientation_deg"]["yaw"]
    rows.append(
        {
            "mapping_yaw_deg": yaw,
            "run": str(run.resolve()),
            "scans": stats.get("scans"),
            "constraints": stats.get("constraints"),
            "odom_samples": stats.get("odom_samples"),
            "wheelbase_m": stats.get("wheelbase_m"),
            "encoder_speed_sign": stats.get("encoder_speed_sign"),
            "map_png": str((run / "map/map.png").resolve()),
            "trajectory_png": str((run / "map_with_trajectory.png").resolve()),
        }
    )

summary = {
    "schema": "fliftbot.mapping_yaw_sweep.v1",
    "recording": "141857_dephan_m360_bs last 18 minutes",
    "max_range_m": 12,
    "effective_camera_x_m": 1.127,
    "effective_lidar_x_m": 1.098,
    "effective_wheelbase_m": 1.385,
    "runs": rows,
}
(root / "yaw_sweep_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print("mapping_yaw  constraints  scans  run")
for row in sorted(rows, key=lambda item: item["mapping_yaw_deg"]):
    print(
        f"{row['mapping_yaw_deg']:>11}  "
        f"{str(row['constraints']):>11}  "
        f"{str(row['scans']):>5}  "
        f"{row['run']}"
    )
print(f"Summary: {root / 'yaw_sweep_summary.json'}")
PY

echo "Yaw sweep finished. Results: $RESULTS_ROOT"
