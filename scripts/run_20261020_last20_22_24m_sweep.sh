#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/optimus/ros-projects"
BUILD_ROOT="$PROJECT_ROOT/fliftbot_map_build"
RECORDING="$PROJECT_ROOT/ros-data/recordings/real/20261013/141857_dephan_m360_bs"
RESULTS_ROOT="$PROJECT_ROOT/ros-data/results/20261020"
CANONICAL_ROOT="$RESULTS_ROOT/_canonical"
JOBS_FILE="$RESULTS_ROOT/last20_22_24m_jobs.txt"

cd "$PROJECT_ROOT"

if [[ -f "$BUILD_ROOT/.venv/bin/activate" ]]; then
    source "$BUILD_ROOT/.venv/bin/activate"
fi
if [[ -f /opt/ros/humble/setup.bash ]]; then
    set +u
    source /opt/ros/humble/setup.bash
    set -u
fi

mkdir -p "$RESULTS_ROOT" "$CANONICAL_ROOT"

# Create the three canonical intervals once.
python3 - "$RECORDING" "$CANONICAL_ROOT" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

recording = Path(sys.argv[1])
canonical_root = Path(sys.argv[2])
metadata = json.loads(
    (recording / "canonical/recording_metadata.json").read_text(encoding="utf-8")
)
recording_start = float(metadata["lidar"]["first_ts"])
recording_end = float(metadata["lidar"]["last_ts"])
slicer = Path("fliftbot_map_build/scripts/slice_canonical_interval.py")

for minutes in (20, 22, 24):
    name = f"141857_last{minutes}m"
    output = canonical_root / name
    if (output / "horizon").is_dir() and (output / "encoder.jsonl").is_file():
        print(f"Reusing canonical interval: {name}")
        continue
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Incomplete non-empty canonical directory: {output}\n"
            "Move or remove this directory, then start the script again."
        )
    start = max(recording_start, recording_end - minutes * 60)
    subprocess.run(
        [
            sys.executable,
            str(slicer),
            "--recording", str(recording / "canonical"),
            "--start-unix", f"{start:.6f}",
            "--end-unix", f"{recording_end:.6f}",
            "--output", str(output),
        ],
        check=True,
    )
PY

: > "$JOBS_FILE"
for minutes in 20 22 24; do
    for lidar_range in 12 16; do
        printf '%s\t%s\t%s\n' \
            "$CANONICAL_ROOT/141857_last${minutes}m" \
            "141857_last${minutes}m_range${lidar_range}m" \
            "$lidar_range" >> "$JOBS_FILE"
    done
done

export RESULTS_ROOT

# Six jobs are launched simultaneously.
xargs -P 6 -n 3 bash -c '
set -euo pipefail
canonical="$1"
name="$2"
lidar_range="$3"

python3 fliftbot_map_build/src/map_build/run_pipeline.py \
    --camera-config "$canonical/camera.config" \
    --lidar-config "$canonical/lidar.config" \
    --robot-config "$canonical/liftbot.config" \
    --lidar-data "$canonical/horizon" \
    --encoder "$canonical/encoder.jsonl" \
    --output "$RESULTS_ROOT" \
    --name "$name" \
    --wheelbase 1.335 \
    --fork-position lowered \
    --lowered-forks-offset-m 0.05 \
    --max-range "$lidar_range" \
    --encoder-speed-sign 1 \
    --cartographer-config fliftbot_2d_offline.lua \
    --use-odometry \
    --no-markers
' _ < "$JOBS_FILE"

echo "Last-20/22/24-minute maps finished. Results: $RESULTS_ROOT"
