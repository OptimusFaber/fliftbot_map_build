#!/usr/bin/env bash
set -euo pipefail

# Three identical Cartographer runs with different route intervals.
# Parameters: steering offset +0.975 deg, wheelbase 1.41 m, LiDAR range 20 m.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
RECORDING="${RECORDING:-/home/optimus/recordings_1pt_20261021/results/_canonical_trim_to_start_20261021_103044}"
RESULTS="${RESULTS:-/home/optimus/recordings_1pt_20261021/results/trimmed_0975_wb141}"
CONFIG_ROOT="${ROOT}/configs/bs_version"
MAP_CONFIG="${MAP_CONFIG:-fliftbot_2d_test.lua}"

mkdir -p "${RESULTS}"

NAMES=(trim_last5 trim_first50 trim_last45)
declare -A PIDS TOTALS STATES

read -r FIRST LAST <<EOF
$(${PYTHON} - "${RECORDING}" <<'PY'
import sys, glob, numpy as np
files=sorted(glob.glob(sys.argv[1] + "/horizon/*.npz"))
stamps=[float(np.load(f)["stamp"]) for f in files]
print(stamps[0], stamps[-1])
PY
)
EOF

stage_interval() {
  local name="$1" start="$2" end="$3"
  local input="${RESULTS}/inputs_${name}"
  [[ ! -e "${input}" ]] || { echo "Input already exists: ${input}" >&2; exit 2; }
  "${PYTHON}" "${ROOT}/scripts/slice_canonical_interval.py" \
    --recording "${RECORDING}" --start-unix "${start}" --end-unix "${end}" --output "${input}" \
    > "${RESULTS}/${name}_slice.log" 2>&1
  TOTALS["${name}"]="$(find "${input}/horizon" -maxdepth 1 -type f -name '*.npz' | wc -l)"
}

launch_run() {
  local name="$1"
  local input="${RESULTS}/inputs_${name}"
  (
    exec "${PYTHON}" "${ROOT}/src/map_build/run_pipeline.py" \
      --camera-config "${CONFIG_ROOT}/camera.config" \
      --camera-position-file /home/optimus/recordings_1pt_20261021/results/20261021_142924_recordings_1pt_20261021/input/camera_position.txt \
      --lidar-config "${CONFIG_ROOT}/lidar.config" \
      --robot-config "${CONFIG_ROOT}/liftbot.config" \
      --lidar-data "${input}/horizon" --encoder "${input}/encoder.jsonl" \
      --output "${RESULTS}" --name "${name}" --fork-position raised \
      --wheelbase 1.41 --lidar-x 1.098 --lidar-y 0.248 --lidar-z 1.961 \
      --mapping-yaw 180 --max-range 20 --encoder-speed-sign 1 --steering-sign 1 \
      --steering-offset-deg 0.975 --odometry-time-offset 0 \
      --cartographer-config "${MAP_CONFIG}" --use-odometry --no-markers \
      > "${RESULTS}/${name}.log" 2>&1
  ) &
  PIDS["${name}"]=$!
  STATES["${name}"]="running"
}

stage_interval trim_last5 "${FIRST}" "$(awk -v x="${LAST}" 'BEGIN {printf "%.9f", x-300}')"
stage_interval trim_first50 "$(awk -v x="${FIRST}" 'BEGIN {printf "%.9f", x+50}')" "${LAST}"
stage_interval trim_last45 "${FIRST}" "$(awk -v x="${LAST}" 'BEGIN {printf "%.9f", x-45}')"

# Compile once before parallel launch. Otherwise three wrappers may try to
# rebuild runner.cc simultaneously after the progress instrumentation changes.
"${ROOT}/src/map_build/run_standalone_cartographer.sh" --help \
  >/dev/null 2>&1 || true

for name in "${NAMES[@]}"; do launch_run "${name}"; done

bar() {
  local percent="$1" width=30 filled empty
  filled=$((percent * width / 100)); empty=$((width - filled))
  printf '%*s' "${filled}" '' | tr ' ' '#'
  printf '%*s' "${empty}" '' | tr ' ' '-'
}

while :; do
  running=0
  if [[ -t 1 ]]; then printf '\033[2J\033[H'; fi
  printf 'Cartographer progress (full output is stored in %s/*.log)\n\n' "${RESULTS}"
  for name in "${NAMES[@]}"; do
    log="${RESULTS}/${name}.log"
    done_scans=0
    if [[ -f "${log}" ]]; then
      done_scans="$(awk -F'[ =]' '/PROGRESS scans=/ {n=$3} END {print n+0}' "${log}")"
      constraints="$(awk '/computations resulted in [0-9]+ additional constraints/ {for(i=1;i<=NF;i++) if($i=="additional") n+=$(i-1)} END {print n+0}' "${log}")"
    else
    constraints=0
    fi
    total="${TOTALS[${name}]}"
    if [[ -f "${log}" ]]; then
      # The runner emits these markers after feeding has actually ended.
      fed="$(awk '/fed scans=[0-9]+ odom=/ {for(i=1;i<=NF;i++) if($i ~ /^scans=/) {sub("scans=", "", $i); print $i}}' "${log}" | tail -1)"
      [[ -n "${fed}" ]] && done_scans="${fed}"
    fi
    percent=$((total > 0 ? done_scans * 100 / total : 0))
    status="feeding"
    if [[ -f "${log}" ]] && grep -q 'exported$' "${log}"; then
      status="done"
      STATES["${name}"]="done"
    elif [[ -f "${log}" ]] && grep -q 'optimized, serializing' "${log}"; then
      status="serializing"
    elif [[ -f "${log}" ]] && grep -q 'finished, optimizing' "${log}"; then
      status="optimizing"
    elif [[ -f "${log}" ]] && grep -q 'fed scans=' "${log}"; then
      status="finalizing"
    fi
    if [[ "${STATES[${name}]}" == running ]] && kill -0 "${PIDS[${name}]}" 2>/dev/null; then
      running=$((running + 1))
      if ((percent >= 100)); then percent=99; [[ "${status}" == feeding ]] && status="finalizing"; fi
    else
      if [[ "${STATES[${name}]}" == running ]]; then
        wait "${PIDS[${name}]}" 2>/dev/null && STATES["${name}"]="done" || STATES["${name}"]="failed"
      fi
      status="${STATES[${name}]}"
      [[ "${status}" == done ]] && percent=100
    fi
    printf '%-13s [%s] %3d%%  scans %5d/%-5d  constraints %-4d  %s\n' \
      "${name}" "$(bar "${percent}")" "${percent}" "${done_scans}" "${total}" "${constraints}" "${status}"
  done
  ((running == 0)) && break
  sleep 2
done

echo "All three runs finished. Results: ${RESULTS}"
