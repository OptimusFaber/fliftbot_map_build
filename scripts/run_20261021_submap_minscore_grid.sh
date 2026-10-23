#!/usr/bin/env bash
set -euo pipefail

# 5 x 4 grid: submap size versus loop-closure minimum score.
# At most MAX_PARALLEL maps are built concurrently (default: 8).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
RECORDING="${RECORDING:-/home/optimus/recordings_1pt_20261021/results/_canonical_trim_to_start_20261021_103044}"
RESULTS="${RESULTS:-/home/optimus/recordings_1pt_20261021/results/submap_minscore_grid_0990_wb141}"
BASE_CONFIG="${ROOT}/configs/cartographer"
CONFIG_DIR="${RESULTS}/lua"
MAX_PARALLEL="${MAX_PARALLEL:-8}"

SUBMAPS=(10 15 20 30 45)
SCORES=(0.48 0.50 0.525 0.55)
NAMES=()

mkdir -p "${RESULTS}" "${CONFIG_DIR}"
cp "${BASE_CONFIG}"/*.lua "${CONFIG_DIR}/"

for submap in "${SUBMAPS[@]}"; do
  for score in "${SCORES[@]}"; do
    score_tag="${score//./p}"
    name="sm${submap}_score${score_tag}"
    NAMES+=("${name}")
    {
      printf 'include "fliftbot_2d_test.lua"\n'
      printf 'TRAJECTORY_BUILDER_2D.submaps.num_range_data = %s\n' "${submap}"
      printf 'POSE_GRAPH.constraint_builder.sampling_ratio = 1.0\n'
      printf 'POSE_GRAPH.constraint_builder.min_score = %s\n' "${score}"
      printf 'return options\n'
    } > "${CONFIG_DIR}/${name}.lua"
  done
done

bar() {
  local percent="$1" width=24 filled empty
  filled=$((percent * width / 100)); empty=$((width - filled))
  printf '%*s' "${filled}" '' | tr ' ' '#'
  printf '%*s' "${empty}" '' | tr ' ' '-'
}

monitor() {
  local total
  total="$(find "${RECORDING}/horizon" -maxdepth 1 -type f -name '*.npz' | wc -l)"
  while :; do
    [[ -t 1 ]] && printf '\033[2J\033[H'
    printf 'submap/min_score grid — full Cartographer output: %s/*.log\n\n' "${RESULTS}"
    for name in "${NAMES[@]}"; do
      local log="${RESULTS}/${name}.log" scans=0 constraints=0 percent=0 status="queued" fed="" opt=""
      if [[ -f "${log}" ]]; then
        scans="$(awk -F'[ =]' '/PROGRESS scans=/ {n=$3} END {print n+0}' "${log}")"
        fed="$(awk '/fed scans=[0-9]+ odom=/ {for(i=1;i<=NF;i++) if($i ~ /^scans=/) {sub("scans=", "", $i); print $i}}' "${log}" | tail -1)"
        [[ -n "${fed}" ]] && scans="${fed}"
        # Score histogram Count is cumulative and therefore must not be summed.
        constraints="$(awk '/^Count: [0-9]+/ {n=$2} END {print n+0}' "${log}")"
        if grep -q 'exported$' "${log}"; then status="done"; percent=100
        elif grep -q 'optimized, serializing' "${log}"; then status="serializing"; percent=99
        elif grep -q 'finished, optimizing' "${log}"; then
          opt="$(grep -aoE 'Optimizing: [0-9]+([.][0-9]+)?%' "${log}" | tail -1 | sed -E 's/Optimizing: ([0-9]+([.][0-9]+)?)%/\1/')"
          if [[ -n "${opt}" ]]; then
            percent="$(awk -v p="${opt}" 'BEGIN {printf "%d", p}')"
            status="optimizing ${opt}%"
          else
            percent=0
            status="optimizing"
          fi
        elif grep -q 'fed scans=' "${log}"; then status="finalizing"; percent=99
        else status="feeding"; percent=$((total > 0 ? scans * 100 / total : 0)); fi
      fi
      printf '%-19s [%s] %3d%% scans %5d/%-5d constraints %-5d %s\n' \
        "${name}" "$(bar "${percent}")" "${percent}" "${scans}" "${total}" "${constraints}" "${status}"
    done
    sleep 3
  done
}

monitor &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT INT TERM

run_one() {
  local name="$1"
  exec "${PYTHON}" "${ROOT}/src/map_build/run_pipeline.py" \
    --camera-config "${ROOT}/configs/bs_version/camera.config" \
    --camera-position-file /home/optimus/recordings_1pt_20261021/results/20261021_142924_recordings_1pt_20261021/input/camera_position.txt \
    --lidar-config "${ROOT}/configs/bs_version/lidar.config" \
    --robot-config "${ROOT}/configs/bs_version/liftbot.config" \
    --lidar-data "${RECORDING}/horizon" --encoder "${RECORDING}/encoder.jsonl" \
    --output "${RESULTS}" --name "${name}" --fork-position raised \
    --wheelbase 1.41 --lidar-x 1.098 --lidar-y 0.248 --lidar-z 1.961 \
    --mapping-yaw 180 --max-range 20 --encoder-speed-sign 1 --steering-sign 1 \
    --steering-offset-deg 0.990 --odometry-time-offset 0 \
    --cartographer-config "${name}.lua" --cartographer-config-dir "${CONFIG_DIR}" \
    --use-odometry --no-markers
}

active=0
failures=0
for name in "${NAMES[@]}"; do
  (run_one "${name}") > "${RESULTS}/${name}.log" 2>&1 &
  active=$((active + 1))
  if ((active >= MAX_PARALLEL)); then
    if ! wait -n; then failures=$((failures + 1)); fi
    active=$((active - 1))
  fi
done
while ((active > 0)); do
  if ! wait -n; then failures=$((failures + 1)); fi
  active=$((active - 1))
done

kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
trap - EXIT INT TERM
echo "Grid finished: ${RESULTS}; failed jobs: ${failures}"
((failures == 0))
