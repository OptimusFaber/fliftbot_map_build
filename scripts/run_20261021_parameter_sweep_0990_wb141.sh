#!/usr/bin/env bash
set -euo pipefail

# Full-route one-factor Cartographer sweep.
# Fixed: steering offset +0.990 deg, wheelbase 1.41 m, raised forks,
# LiDAR (1.098, 0.248, 1.961) m, max range 20 m, fliftbot_2d_test.lua.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
RECORDING="${RECORDING:-/home/optimus/recordings_1pt_20261021/results/_canonical_trim_to_start_20261021_103044}"
RESULTS="${RESULTS:-/home/optimus/recordings_1pt_20261021/results/parameter_sweep_0990_wb141}"
BASE_CONFIG="${ROOT}/configs/cartographer"
CONFIG_DIR="${RESULTS}/lua"
CONFIG_NAME="fliftbot_2d_test.lua"
MAX_PARALLEL="${MAX_PARALLEL:-10}"

mkdir -p "${RESULTS}" "${CONFIG_DIR}"
cp "${BASE_CONFIG}"/*.lua "${CONFIG_DIR}/"

make_config() {
  local name="$1" override="$2"
  {
    printf 'include "fliftbot_2d_test.lua"\n'
    printf '%s\n' "${override}"
    printf 'return options\n'
  } > "${CONFIG_DIR}/${name}.lua"
}

make_config baseline ' -- baseline: values from fliftbot_2d_test.lua'
for value in 0.3 0.6 1.0; do
  tag=${value//./p}
  make_config "sampling_${tag}" "POSE_GRAPH.constraint_builder.sampling_ratio = ${value}"
done
for value in 0.60 0.55 0.45; do
  tag=${value//./p}
  make_config "minscore_${tag}" "POSE_GRAPH.constraint_builder.min_score = ${value}"
done
for value in 5 10 15; do
  make_config "maxdist_${value}" "POSE_GRAPH.constraint_builder.max_constraint_distance = ${value}.0"
done
for value in 0.25 5 100 1000; do
  tag=${value//./p}
  make_config "odomw_${tag}" "POSE_GRAPH.optimization_problem.odometry_translation_weight = ${value}; POSE_GRAPH.optimization_problem.odometry_rotation_weight = ${value}"
done
for value in 15 45 90; do
  make_config "submap_${value}" "TRAJECTORY_BUILDER_2D.submaps.num_range_data = ${value}"
done

NAMES=(baseline sampling_0p3 sampling_0p6 sampling_1p0 minscore_0p60 minscore_0p55 minscore_0p45 maxdist_5 maxdist_10 maxdist_15 odomw_0p25 odomw_5 odomw_100 odomw_1000 submap_15 submap_45 submap_90)

monitor() {
  local total="$(find "${RECORDING}/horizon" -maxdepth 1 -type f -name '*.npz' | wc -l)"
  while :; do
    if [[ -t 1 ]]; then printf '\033[2J\033[H'; fi
    printf 'Cartographer parameter sweep (logs: %s/*.log)\n\n' "${RESULTS}"
    for name in "${NAMES[@]}"; do
      local log="${RESULTS}/${name}.log" scans=0 constraints=0 percent=0 status="queued"
      if [[ -f "${log}" ]]; then
        scans="$(awk -F'[ =]' '/PROGRESS scans=/ {n=$3} END {print n+0}' "${log}")"
        fed="$(awk '/fed scans=[0-9]+ odom=/ {for(i=1;i<=NF;i++) if($i ~ /^scans=/) {sub("scans=", "", $i); print $i}}' "${log}" | tail -1)"
        [[ -n "${fed}" ]] && scans="${fed}"
        constraints="$(awk '/computations resulted in [0-9]+ additional constraints/ {for(i=1;i<=NF;i++) if($i=="additional") n+=$(i-1)} END {print n+0}' "${log}")"
        if grep -q 'exported$' "${log}"; then status="done"; percent=100
        elif grep -q 'optimized, serializing' "${log}"; then status="serializing"; percent=99
        elif grep -q 'finished, optimizing' "${log}"; then status="optimizing"; percent=99
        elif grep -q 'fed scans=' "${log}"; then status="finalizing"; percent=99
        else status="feeding"; fi
        if ((percent == 0)); then percent=$((total > 0 ? scans * 100 / total : 0)); fi
      fi
      printf '%-15s %3d%%  scans %5d/%-5d  constraints %-4d  %s\n' "${name}" "${percent}" "${scans}" "${total}" "${constraints}" "${status}"
    done
    sleep 3
  done
}

monitor &
monitor_pid=$!
active=0
for name in "${NAMES[@]}"; do
  output="${RESULTS}/map_${name}"
  [[ ! -e "${output}" ]] || { echo "Output already exists: ${output}" >&2; exit 2; }
  (
    exec "${PYTHON}" "${ROOT}/src/map_build/run_pipeline.py" \
      --camera-config "${ROOT}/configs/bs_version/camera.config" \
      --camera-position-file /home/optimus/recordings_1pt_20261021/results/20261021_142924_recordings_1pt_20261021/input/camera_position.txt \
      --lidar-config "${ROOT}/configs/bs_version/lidar.config" \
      --robot-config "${ROOT}/configs/bs_version/liftbot.config" \
      --lidar-data "${RECORDING}/horizon" --encoder "${RECORDING}/encoder.jsonl" \
      --output "${RESULTS}" --name "map_${name}" --fork-position raised \
      --wheelbase 1.41 --lidar-x 1.098 --lidar-y 0.248 --lidar-z 1.961 \
      --mapping-yaw 180 --max-range 20 --encoder-speed-sign 1 --steering-sign 1 \
      --steering-offset-deg 0.990 --odometry-time-offset 0 \
      --cartographer-config "${name}.lua" --cartographer-config-dir "${CONFIG_DIR}" \
      --use-odometry --no-markers
  ) > "${RESULTS}/${name}.log" 2>&1 &
  active=$((active + 1))
  if ((active >= MAX_PARALLEL)); then
    wait -n
    active=$((active - 1))
  fi
done
wait
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
echo "Sweep finished: ${RESULTS}"
