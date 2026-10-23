#!/usr/bin/env bash
set -euo pipefail

# Seven-run comparison for recordings_1pt_20261021.  The existing
# full/offline/20 m result is the eighth (baseline) cell of the 2x2x2 matrix.
#
# Usage:
#   ./scripts/run_20261021_route_profile_range_sweep.sh [parallel_jobs]
#
# Optional overrides:
#   SOURCE_CANONICAL, RESULTS_ROOT, TRIM_END_UNIX,
#   OFFLINE_CONFIG, ONLINE_CONFIG, PYTHON

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
PARALLEL_JOBS="${1:-3}"

SOURCE_CANONICAL="${SOURCE_CANONICAL:-/home/optimus/recordings_1pt_20261021/canonical}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/optimus/recordings_1pt_20261021/results}"
TRIM_END_UNIX="${TRIM_END_UNIX:-1792578643.9}"
TRIMMED_CANONICAL="${RESULTS_ROOT}/_canonical_trim_to_start_20261021_103044"
EXPERIMENT_ROOT="${RESULTS_ROOT}/route_profile_range_sweep"
JOBS_FILE="${EXPERIMENT_ROOT}/jobs.tsv"
LOG_ROOT="${EXPERIMENT_ROOT}/logs"

# fliftbot_2d.lua is the base live/online profile.  The offline profile adds
# online correlative scan matching and raises odometry weights from 0.25 to 0.5.
OFFLINE_CONFIG="${OFFLINE_CONFIG:-fliftbot_2d_offline.lua}"
ONLINE_CONFIG="${ONLINE_CONFIG:-fliftbot_2d.lua}"

CONFIG_ROOT="${ROOT}/configs/bs_version"
CAMERA_CONFIG="${CONFIG_ROOT}/camera.config"
LIDAR_CONFIG="${CONFIG_ROOT}/lidar.config"
ROBOT_CONFIG="${CONFIG_ROOT}/liftbot.config"
BASELINE="${RESULTS_ROOT}/20261021_142924_recordings_1pt_20261021"
CAMERA_POSITION="${BASELINE}/input/camera_position.txt"
CONFIGURATION_DIR="${ROOT}/configs/cartographer"

if ! [[ "${PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "parallel_jobs must be a positive integer" >&2
    exit 2
fi

for path in \
    "${SOURCE_CANONICAL}/horizon" \
    "${SOURCE_CANONICAL}/encoder.jsonl" \
    "${CAMERA_CONFIG}" \
    "${LIDAR_CONFIG}" \
    "${ROBOT_CONFIG}" \
    "${CAMERA_POSITION}" \
    "${CONFIGURATION_DIR}/${OFFLINE_CONFIG}" \
    "${CONFIGURATION_DIR}/${ONLINE_CONFIG}"
do
    [[ -e "${path}" ]] || { echo "Required input not found: ${path}" >&2; exit 2; }
done

mkdir -p "${EXPERIMENT_ROOT}" "${LOG_ROOT}"

# Determine the exact source start from the canonical metadata.  The selected
# end is the closest-return point found in the current Cartographer trajectory
# (about 25:49 into the recording, leaving about 4:11 trimmed).
SOURCE_START_UNIX="$(${PYTHON} - "${SOURCE_CANONICAL}" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads((Path(sys.argv[1]) / "recording_metadata.json").read_text())
print(metadata["lidar"]["first_ts"])
PY
)"

if [[ -f "${TRIMMED_CANONICAL}/interval_report.json" ]]; then
    "${PYTHON}" - "${TRIMMED_CANONICAL}/interval_report.json" "${TRIM_END_UNIX}" <<'PY'
import json
import math
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
expected = float(sys.argv[2])
actual = float(report["requested_end_unix"])
if not math.isclose(actual, expected, abs_tol=1e-6):
    raise SystemExit(
        f"Existing trimmed recording ends at {actual}, expected {expected}. "
        "Use a different RESULTS_ROOT or move the existing trimmed directory."
    )
print(f"Reusing trimmed canonical recording: {sys.argv[1]}")
PY
elif [[ -e "${TRIMMED_CANONICAL}" ]] && [[ -n "$(find "${TRIMMED_CANONICAL}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Incomplete non-empty trimmed directory: ${TRIMMED_CANONICAL}" >&2
    echo "Move it aside and run this script again." >&2
    exit 2
else
    "${PYTHON}" "${ROOT}/scripts/slice_canonical_interval.py" \
        --recording "${SOURCE_CANONICAL}" \
        --start-unix "${SOURCE_START_UNIX}" \
        --end-unix "${TRIM_END_UNIX}" \
        --output "${TRIMMED_CANONICAL}"
fi

# recording, profile label, Lua file, range, result name
# The already-built full/offline/20 m baseline is deliberately omitted.
cat > "${JOBS_FILE}" <<EOF
${SOURCE_CANONICAL}	offline	${OFFLINE_CONFIG}	100	full_offline_range100m
${SOURCE_CANONICAL}	online	${ONLINE_CONFIG}	20	full_online_range20m
${SOURCE_CANONICAL}	online	${ONLINE_CONFIG}	100	full_online_range100m
${TRIMMED_CANONICAL}	offline	${OFFLINE_CONFIG}	20	trim251s_offline_range20m
${TRIMMED_CANONICAL}	online	${ONLINE_CONFIG}	20	trim251s_online_range20m
${TRIMMED_CANONICAL}	offline	${OFFLINE_CONFIG}	100	trim251s_offline_range100m
${TRIMMED_CANONICAL}	online	${ONLINE_CONFIG}	100	trim251s_online_range100m
EOF

export ROOT PYTHON EXPERIMENT_ROOT LOG_ROOT
export CAMERA_CONFIG LIDAR_CONFIG ROBOT_CONFIG CAMERA_POSITION

run_mapping_job() {
    local recording="$1"
    local profile="$2"
    local lua_config="$3"
    local lidar_range="$4"
    local name="$5"
    local log="${LOG_ROOT}/${name}.log"

    echo "START ${name}: profile=${profile}, range=${lidar_range}, recording=${recording}"
    if "${PYTHON}" "${ROOT}/src/map_build/run_pipeline.py" \
        --camera-config "${CAMERA_CONFIG}" \
        --camera-position-file "${CAMERA_POSITION}" \
        --lidar-config "${LIDAR_CONFIG}" \
        --robot-config "${ROBOT_CONFIG}" \
        --lidar-data "${recording}/horizon" \
        --encoder "${recording}/encoder.jsonl" \
        --output "${EXPERIMENT_ROOT}" \
        --name "${name}" \
        --fork-position lowered \
        --lowered-forks-offset-m 0.05 \
        --wheelbase 1.39 \
        --mapping-yaw 180 \
        --max-range "${lidar_range}" \
        --encoder-speed-sign 1 \
        --steering-sign 1 \
        --odometry-time-offset 0 \
        --cartographer-config "${lua_config}" \
        --use-odometry \
        --no-markers >"${log}" 2>&1
    then
        echo "DONE  ${name} (log: ${log})"
    else
        local status=$?
        echo "FAIL  ${name} (exit ${status}, log: ${log})" >&2
        return "${status}"
    fi
}
export -f run_mapping_job

echo "Running seven jobs with up to ${PARALLEL_JOBS} simultaneous processes."
echo "Existing baseline: ${BASELINE}"
echo "Experiment output: ${EXPERIMENT_ROOT}"

xargs -P "${PARALLEL_JOBS}" -n 5 bash -c \
    'run_mapping_job "$1" "$2" "$3" "$4" "$5"' _ < "${JOBS_FILE}"

echo "All seven mapping jobs completed."
echo "Results: ${EXPERIMENT_ROOT}"
echo "Logs:    ${LOG_ROOT}"
