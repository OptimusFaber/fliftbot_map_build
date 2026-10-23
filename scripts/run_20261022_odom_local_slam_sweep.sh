#!/usr/bin/env bash
set -euo pipefail

# Diagnose steering correction, odometry weights, local scan matching and
# global constraint search. At most parallel_jobs maps run simultaneously.
# Usage: ./scripts/run_20261022_odom_local_slam_sweep.sh [parallel_jobs]
# Default: build all six maps simultaneously.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
PARALLEL_JOBS="${1:-6}"
RECORDING="${RECORDING:-/home/optimus/recordings_1pt_20261021/results/_canonical_trim_to_start_20261021_103044}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/optimus/recordings_1pt_20261021/results/odom_local_slam_sweep}"
LOG_ROOT="${RESULTS_ROOT}/logs"
JOBS_FILE="${RESULTS_ROOT}/jobs.tsv"

CONFIG_ROOT="${ROOT}/configs/bs_version"
CAMERA_CONFIG="${CONFIG_ROOT}/camera.config"
LIDAR_CONFIG="${CONFIG_ROOT}/lidar.config"
ROBOT_CONFIG="${CONFIG_ROOT}/liftbot.config"
CAMERA_POSITION="${CAMERA_POSITION:-/home/optimus/recordings_1pt_20261021/results/20261021_142924_recordings_1pt_20261021/input/camera_position.txt}"
CARTOGRAPHER_CONFIG_ROOT="${ROOT}/configs/cartographer"

if ! [[ "${PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "parallel_jobs must be a positive integer" >&2
    exit 2
fi

CONFIGS=(
    fliftbot_2d_exp_odom_w100.lua
    fliftbot_2d_exp_odom_w1k.lua
    fliftbot_2d_exp_odom_w10k.lua
    fliftbot_2d_exp_odom_w100k.lua
    fliftbot_2d_exp_odom_w10k_rt_off.lua
    fliftbot_2d_exp_odom_w10k_rt_narrow.lua
    fliftbot_2d_exp_odom_w10k_maxdist20.lua
    fliftbot_2d_exp_odom_t10k_r1k.lua
    fliftbot_2d_exp_odom_t10k_r100k.lua
    fliftbot_2d_exp_odom_w10k_rt_a05.lua
    fliftbot_2d_exp_odom_w10k_rt_a15.lua
    fliftbot_2d_exp_odom_w10k_rt_a25.lua
    fliftbot_2d_exp_odom_w10k_rt_l005.lua
    fliftbot_2d_exp_odom_w10k_rt_l030.lua
    fliftbot_2d_exp_odom_w10k_global_a20.lua
    fliftbot_2d_exp_odom_w10k_global_a90.lua
    fliftbot_2d_exp_odom_w10k_maxdist30.lua
    fliftbot_2d_exp_odom_w10k_score050.lua
    fliftbot_2d_exp_odom_w10k_score060.lua
)

for path in \
    "${RECORDING}/horizon" \
    "${RECORDING}/encoder.jsonl" \
    "${CAMERA_CONFIG}" \
    "${LIDAR_CONFIG}" \
    "${ROBOT_CONFIG}" \
    "${CAMERA_POSITION}"
do
    [[ -e "${path}" ]] || { echo "Required input not found: ${path}" >&2; exit 2; }
done
for config in "${CONFIGS[@]}"; do
    path="${CARTOGRAPHER_CONFIG_ROOT}/${config}"
    [[ -e "${path}" ]] || { echo "Required config not found: ${path}" >&2; exit 2; }
done

if [[ "${ALLOW_CONCURRENT:-0}" != "1" ]] && pgrep -f 'fliftbot_cartographer_offline' >/dev/null; then
    echo "Another Cartographer build is still running." >&2
    echo "Wait for it to finish, or set ALLOW_CONCURRENT=1 explicitly." >&2
    exit 2
fi

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
cat > "${JOBS_FILE}" <<'EOF'
fliftbot_2d_exp_odom_w100.lua	01_weight_w100_s1p0	1.0
fliftbot_2d_exp_odom_w10k.lua	01_weight_w10k_s1p0	1.0
fliftbot_2d_exp_odom_w100k.lua	01_weight_w100k_s1p0	1.0
fliftbot_2d_exp_odom_w10k.lua	02_steering_s0p6	0.6
fliftbot_2d_exp_odom_w10k.lua	02_steering_s0p8	0.8
fliftbot_2d_exp_odom_w10k.lua	02_steering_s0p9	0.9
fliftbot_2d_exp_odom_w10k.lua	02_steering_s1p1	1.1
fliftbot_2d_exp_odom_w10k.lua	02_steering_s1p2	1.2
fliftbot_2d_exp_odom_w10k.lua	02_steering_s1p4	1.4
fliftbot_2d_exp_odom_w100.lua	02_grid_w100_s0p6	0.6
fliftbot_2d_exp_odom_w100.lua	02_grid_w100_s0p8	0.8
fliftbot_2d_exp_odom_w100.lua	02_grid_w100_s0p9	0.9
fliftbot_2d_exp_odom_w100.lua	02_grid_w100_s1p1	1.1
fliftbot_2d_exp_odom_w100.lua	02_grid_w100_s1p2	1.2
fliftbot_2d_exp_odom_w100.lua	02_grid_w100_s1p4	1.4
fliftbot_2d_exp_odom_w1k.lua	02_grid_w1k_s0p6	0.6
fliftbot_2d_exp_odom_w1k.lua	02_grid_w1k_s0p8	0.8
fliftbot_2d_exp_odom_w1k.lua	02_grid_w1k_s0p9	0.9
fliftbot_2d_exp_odom_w1k.lua	02_grid_w1k_s1p0	1.0
fliftbot_2d_exp_odom_w1k.lua	02_grid_w1k_s1p1	1.1
fliftbot_2d_exp_odom_w1k.lua	02_grid_w1k_s1p2	1.2
fliftbot_2d_exp_odom_w1k.lua	02_grid_w1k_s1p4	1.4
fliftbot_2d_exp_odom_w100k.lua	02_grid_w100k_s0p6	0.6
fliftbot_2d_exp_odom_w100k.lua	02_grid_w100k_s0p8	0.8
fliftbot_2d_exp_odom_w100k.lua	02_grid_w100k_s0p9	0.9
fliftbot_2d_exp_odom_w100k.lua	02_grid_w100k_s1p1	1.1
fliftbot_2d_exp_odom_w100k.lua	02_grid_w100k_s1p2	1.2
fliftbot_2d_exp_odom_w100k.lua	02_grid_w100k_s1p4	1.4
fliftbot_2d_exp_odom_t10k_r1k.lua	03_rotation_r1k	1.0
fliftbot_2d_exp_odom_t10k_r100k.lua	03_rotation_r100k	1.0
fliftbot_2d_exp_odom_w10k_rt_off.lua	04_local_rt_off	1.0
fliftbot_2d_exp_odom_w10k_rt_a05.lua	04_local_angle05	1.0
fliftbot_2d_exp_odom_w10k_rt_narrow.lua	04_local_angle10	1.0
fliftbot_2d_exp_odom_w10k_rt_a15.lua	04_local_angle15	1.0
fliftbot_2d_exp_odom_w10k_rt_a25.lua	04_local_angle25	1.0
fliftbot_2d_exp_odom_w10k_rt_l005.lua	04_local_linear005	1.0
fliftbot_2d_exp_odom_w10k_rt_l030.lua	04_local_linear030	1.0
fliftbot_2d_exp_odom_w10k_global_a20.lua	05_global_angle20	1.0
fliftbot_2d_exp_odom_w10k.lua	05_global_angle45	1.0
fliftbot_2d_exp_odom_w10k_global_a90.lua	05_global_angle90	1.0
fliftbot_2d_exp_odom_w10k_maxdist20.lua	05_global_distance20	1.0
fliftbot_2d_exp_odom_w10k_maxdist30.lua	05_global_distance30	1.0
fliftbot_2d_exp_odom_w10k_score050.lua	06_score050	1.0
fliftbot_2d_exp_odom_w10k_score060.lua	06_score060	1.0
EOF

export ROOT PYTHON RECORDING RESULTS_ROOT LOG_ROOT
export CAMERA_CONFIG LIDAR_CONFIG ROBOT_CONFIG CAMERA_POSITION

run_mapping_job() {
    local lua_config="$1"
    local name="$2"
    local steering_offset_deg="$3"
    local log="${LOG_ROOT}/${name}.log"
    local result_path_file="${RESULTS_ROOT}/${name}.result_path"

    echo "START ${name}: ${lua_config}, steering_offset=${steering_offset_deg} deg"
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
        --use-odometry \
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
export -f run_mapping_job

JOB_COUNT="$(wc -l < "${JOBS_FILE}")"
echo "Running ${JOB_COUNT} mapping tests with at most ${PARALLEL_JOBS} parallel processes."
xargs -P "${PARALLEL_JOBS}" -n 3 bash -c \
    'run_mapping_job "$1" "$2" "$3"' _ < "${JOBS_FILE}"

"${PYTHON}" - "${RESULTS_ROOT}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
pattern = re.compile(r"(\d+) computations resulted in (\d+) additional constraints")
for result_file in sorted(root.glob("*.result_path")):
    run = Path(result_file.read_text(encoding="utf-8").strip())
    parameters = json.loads((run / "run_parameters.json").read_text())
    stats = json.loads((run / "map/run_stats.json").read_text())
    poses = json.loads((run / "map/trajectory.json").read_text())["poses"]
    closure = math.hypot(poses[-1]["x"] - poses[0]["x"], poses[-1]["y"] - poses[0]["y"])
    start_yaw = 2.0 * math.atan2(poses[0]["qz"], poses[0]["qw"])
    end_yaw = 2.0 * math.atan2(poses[-1]["qz"], poses[-1]["qw"])
    heading_error = abs((end_yaw - start_yaw + math.pi) % (2.0 * math.pi) - math.pi)
    path_length = sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) for a, b in zip(poses, poses[1:]))
    log = root / "logs" / f"{result_file.stem}.log"
    computations = additional = 0
    for match in pattern.finditer(log.read_text(encoding="utf-8", errors="replace")):
        computations += int(match.group(1))
        additional += int(match.group(2))
    rows.append({
        "name": result_file.stem,
        "run": str(run),
        "configuration": parameters["mapping"]["cartographer_config"],
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
print("\nOdometry/local-SLAM sweep summary")
print("name                                  steer  added  total  closure_m  heading_deg  path_m")
for row in rows:
    print(f"{row['name']:<37} {row['steering_offset_deg']:>5.1f} {row['additional_constraints']:>5} {row['constraints_total']:>6} "
          f"{row['start_end_distance_m']:>10.2f} {row['start_end_heading_difference_deg']:>12.1f} "
          f"{row['path_length_m']:>7.1f}")
print(f"\nDetailed summary: {root / 'summary.json'}")
PY

echo "Maps: ${RESULTS_ROOT}"
echo "Logs: ${LOG_ROOT}"
