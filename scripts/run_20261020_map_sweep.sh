#!/usr/bin/env bash
set -euo pipefail

# Usage: ./fliftbot_map_build/scripts/run_20261020_map_sweep.sh [parallel_jobs]
# Default: 6 simultaneous mapping processes.
PARALLEL_JOBS="${1:-6}"

if ! [[ "$PARALLEL_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "parallel_jobs must be a positive integer" >&2
    exit 2
fi

PROJECT_ROOT="/home/optimus/ros-projects"
BUILD_ROOT="$PROJECT_ROOT/fliftbot_map_build"
RECORDINGS_ROOT="$PROJECT_ROOT/ros-data/recordings/real/20261013"
RESULTS_ROOT="$PROJECT_ROOT/ros-data/results/20261020"
CANONICAL_ROOT="$RESULTS_ROOT/_canonical"
JOBS_FILE="$RESULTS_ROOT/map_sweep_jobs.txt"

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

# Prepare the eleven time intervals once. Existing complete slices are reused.
python3 - "$RECORDINGS_ROOT" "$CANONICAL_ROOT" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

recordings_root = Path(sys.argv[1])
canonical_root = Path(sys.argv[2])
slicer = Path("fliftbot_map_build/scripts/slice_canonical_interval.py")


def bounds(recording: Path) -> tuple[float, float]:
    metadata = json.loads(
        (recording / "canonical/recording_metadata.json").read_text(encoding="utf-8")
    )
    return float(metadata["lidar"]["first_ts"]), float(metadata["lidar"]["last_ts"])


intervals: list[tuple[Path, float, float, str]] = []

first = recordings_root / "141857_dephan_m360_bs"
first_start, first_end = bounds(first)
for minutes in (6, 8, 10, 12, 15):
    intervals.append(
        (first, first_start, min(first_start + minutes * 60, first_end),
         f"141857_first{minutes}m")
    )
for minutes in (15, 12, 10, 8, 6):
    intervals.append(
        (first, max(first_start, first_end - minutes * 60), first_end,
         f"141857_last{minutes}m")
    )

second = recordings_root / "163021_dephan_m360_bs"
second_start, second_end = bounds(second)
intervals.append((second, second_start, second_end, "163021_full"))

for recording, start, end, name in intervals:
    output = canonical_root / name
    if (output / "horizon").is_dir() and (output / "encoder.jsonl").is_file():
        print(f"Reusing canonical interval: {name}")
        continue
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Incomplete non-empty canonical directory: {output}\n"
            "Move or remove this directory, then start the script again."
        )
    subprocess.run(
        [
            sys.executable,
            str(slicer),
            "--recording", str(recording / "canonical"),
            "--start-unix", f"{start:.6f}",
            "--end-unix", f"{end:.6f}",
            "--output", str(output),
        ],
        check=True,
    )
PY

# One line per mapping run: canonical directory, run name, LiDAR range.
: > "$JOBS_FILE"
for interval in \
    141857_first6m 141857_first8m 141857_first10m 141857_first12m 141857_first15m \
    141857_last15m 141857_last12m 141857_last10m 141857_last8m 141857_last6m \
    163021_full
do
    for lidar_range in 16 18 20 24 32; do
        printf '%s\t%s\t%s\n' \
            "$CANONICAL_ROOT/$interval" \
            "${interval}_range${lidar_range}m" \
            "$lidar_range" >> "$JOBS_FILE"
    done
done

export RESULTS_ROOT

xargs -P "$PARALLEL_JOBS" -n 3 bash -c '
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

echo "All mapping jobs finished. Results: $RESULTS_ROOT"
