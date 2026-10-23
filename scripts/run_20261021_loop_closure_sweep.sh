#!/usr/bin/env bash
set -euo pipefail

# Loop-closure diagnostic sweep for the trimmed 2026-10-21 recording.
# In addition to three score thresholds, run two controls at score 0.60:
# LiDAR-only, and odometry with +0.2 deg added to every steering measurement.
# Usage: ./scripts/run_20261021_loop_closure_sweep.sh [parallel_jobs]
# Default: run all five experiments simultaneously.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
PARALLEL_JOBS="${1:-5}"
RECORDING="${RECORDING:-/home/optimus/recordings_1pt_20261021/results/_canonical_trim_to_start_20261021_103044}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/optimus/recordings_1pt_20261021/results/loop_closure_sweep}"
LOG_ROOT="${RESULTS_ROOT}/logs"
JOBS_FILE="${RESULTS_ROOT}/jobs.tsv"

CONFIG_ROOT="${ROOT}/configs/bs_version"
CAMERA_CONFIG="${CONFIG_ROOT}/camera.config"
LIDAR_CONFIG="${CONFIG_ROOT}/lidar.config"
ROBOT_CONFIG="${CONFIG_ROOT}/liftbot.config"
CAMERA_POSITION="/home/optimus/recordings_1pt_20261021/results/20261021_142924_recordings_1pt_20261021/input/camera_position.txt"
CARTOGRAPHER_CONFIG_ROOT="${ROOT}/configs/cartographer"

if ! [[ "${PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "parallel_jobs must be a positive integer" >&2
    exit 2
fi

for path in \
    "${RECORDING}/horizon" \
    "${RECORDING}/encoder.jsonl" \
    "${CAMERA_CONFIG}" \
    "${LIDAR_CONFIG}" \
    "${ROBOT_CONFIG}" \
    "${CAMERA_POSITION}" \
    "${CARTOGRAPHER_CONFIG_ROOT}/fliftbot_2d_loop_0625.lua" \
    "${CARTOGRAPHER_CONFIG_ROOT}/fliftbot_2d_loop_0600.lua" \
    "${CARTOGRAPHER_CONFIG_ROOT}/fliftbot_2d_loop_0550.lua"
do
    [[ -e "${path}" ]] || { echo "Required input not found: ${path}" >&2; exit 2; }
done

if [[ "${ALLOW_CONCURRENT:-0}" != "1" ]] && pgrep -f 'fliftbot_cartographer_offline' >/dev/null; then
    echo "Another Cartographer build is still running." >&2
    echo "Wait for it to finish, or set ALLOW_CONCURRENT=1 explicitly." >&2
    exit 2
fi

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
cat > "${JOBS_FILE}" <<'EOF'
fliftbot_2d_loop_0625.lua	trim20_loop_score0625	odom	0.0
fliftbot_2d_loop_0600.lua	trim20_loop_score0600	odom	0.0
fliftbot_2d_loop_0550.lua	trim20_loop_score0550	odom	0.0
fliftbot_2d_loop_0600.lua	trim20_loop_score0600_lidar_only	lidar_only	0.0
fliftbot_2d_loop_0600.lua	trim20_loop_score0600_steering_plus0p2	odom	0.2
EOF

export ROOT PYTHON RECORDING RESULTS_ROOT LOG_ROOT
export CAMERA_CONFIG LIDAR_CONFIG ROBOT_CONFIG CAMERA_POSITION

run_loop_job() {
    local lua_config="$1"
    local name="$2"
    local mode="$3"
    local steering_offset_deg="$4"
    local log="${LOG_ROOT}/${name}.log"
    local result_path_file="${RESULTS_ROOT}/${name}.result_path"
    local odometry_flag="--use-odometry"
    if [[ "${mode}" == "lidar_only" ]]; then
        odometry_flag="--lidar-only"
    elif [[ "${mode}" != "odom" ]]; then
        echo "Unknown mapping mode: ${mode}" >&2
        return 2
    fi

    echo "START ${name}: ${lua_config}, mode=${mode}, steering_offset_deg=${steering_offset_deg}"
    if "${PYTHON}" "${ROOT}/src/map_build/run_pipeline.py" \
        --camera-config "${CAMERA_CONFIG}" \
        --camera-position-file "${CAMERA_POSITION}" \
        --lidar-config "${LIDAR_CONFIG}" \
        --robot-config "${ROBOT_CONFIG}" \
        --lidar-data "${RECORDING}/horizon" \
        --encoder "${RECORDING}/encoder.jsonl" \
        --output "${RESULTS_ROOT}" \
        --name "${name}" \
        --fork-position lowered \
        --lowered-forks-offset-m 0.05 \
        --wheelbase 1.39 \
        --mapping-yaw 180 \
        --max-range 20 \
        --encoder-speed-sign 1 \
        --steering-sign 1 \
        --steering-offset-deg "${steering_offset_deg}" \
        --odometry-time-offset 0 \
        --cartographer-config "${lua_config}" \
        "${odometry_flag}" \
        --no-markers \
        --result-path-file "${result_path_file}" >"${log}" 2>&1
    then
        echo "DONE  ${name} (log: ${log})"
    else
        local status=$?
        echo "FAIL  ${name} (exit ${status}, log: ${log})" >&2
        return "${status}"
    fi
}
export -f run_loop_job

echo "Running five loop-closure tests with ${PARALLEL_JOBS} parallel processes."
xargs -P "${PARALLEL_JOBS}" -n 4 bash -c \
    'run_loop_job "$1" "$2" "$3" "$4"' _ < "${JOBS_FILE}"

"${PYTHON}" - "${RESULTS_ROOT}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for result_file in sorted(root.glob("*.result_path")):
    run = Path(result_file.read_text(encoding="utf-8").strip())
    parameters = json.loads((run / "run_parameters.json").read_text())
    stats = json.loads((run / "map/run_stats.json").read_text())
    poses = json.loads((run / "map/trajectory.json").read_text())["poses"]
    closure = math.hypot(
        poses[-1]["x"] - poses[0]["x"],
        poses[-1]["y"] - poses[0]["y"],
    )
    start_yaw = 2.0 * math.atan2(poses[0]["qz"], poses[0]["qw"])
    end_yaw = 2.0 * math.atan2(poses[-1]["qz"], poses[-1]["qw"])
    heading_error = abs((end_yaw - start_yaw + math.pi) % (2.0 * math.pi) - math.pi)
    path_length = sum(
        math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        for a, b in zip(poses, poses[1:])
    )
    log = root / "logs" / f"{result_file.stem}.log"
    additional = 0
    computations = 0
    pattern = re.compile(r"(\d+) computations resulted in (\d+) additional constraints")
    for match in pattern.finditer(log.read_text(encoding="utf-8", errors="replace")):
        computations += int(match.group(1))
        additional += int(match.group(2))
    rows.append({
        "name": result_file.stem,
        "run": str(run),
        "configuration": parameters["mapping"]["cartographer_config"],
        "use_odometry": parameters["mapping"]["use_odometry"],
        "steering_offset_deg": stats["steering_offset_deg"],
        "nodes": len(poses),
        "constraints_total": stats["constraints"],
        "constraint_computations": computations,
        "additional_constraints": additional,
        "path_length_m": path_length,
        "start_end_distance_m": closure,
        "start_end_heading_difference_deg": math.degrees(heading_error),
    })

(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
print("\nLoop-closure sweep summary")
print("name                                      odom steer_off   added   closure_m   heading_deg   path_m")
for row in rows:
    print(
        f"{row['name']:<41} {str(row['use_odometry']):>5} "
        f"{row['steering_offset_deg']:>9.2f} {row['additional_constraints']:>7} "
        f"{row['start_end_distance_m']:>11.2f} "
        f"{row['start_end_heading_difference_deg']:>13.1f} "
        f"{row['path_length_m']:>8.1f}"
    )
print(f"\nDetailed summary: {root / 'summary.json'}")
PY

echo "Maps: ${RESULTS_ROOT}"
echo "Logs: ${LOG_ROOT}"
