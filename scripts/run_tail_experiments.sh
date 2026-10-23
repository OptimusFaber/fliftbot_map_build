#!/usr/bin/env bash
set -euo pipefail

# Build maps from the last 6, 8 and 10 minutes of a canonical recording.
# The source is never modified; selected NPZ files are hard-linked when possible.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
RECORDING="${RECORDING:-/home/optimus/ros-projects/ros-data/recordings/real/20261013/141857_dephan_m360_bs/canonical}"
OUTPUT="${OUTPUT:-/home/optimus/ros-projects/ros-data/results/real/20261013/tail_experiments_60m_141857}"
WHEELBASE="${WHEELBASE:-1.39}"
ENCODER_SPEED_SIGN="${ENCODER_SPEED_SIGN:-1}"
STEERING_SIGN="${STEERING_SIGN:-1}"
MAPPING_YAW="${MAPPING_YAW:-180}"
CONFIG_DIR="${CONFIG_DIR:-${ROOT}/configs/bs_version}"
CAMERA_CONFIG="${CAMERA_CONFIG:-${CONFIG_DIR}/camera.config}"
LIDAR_CONFIG="${LIDAR_CONFIG:-${CONFIG_DIR}/lidar_60m.config}"
ROBOT_CONFIG="${ROBOT_CONFIG:-${CONFIG_DIR}/liftbot.config}"

for CONFIG in "${CAMERA_CONFIG}" "${LIDAR_CONFIG}" "${ROBOT_CONFIG}"; do
  [[ -f "${CONFIG}" ]] || { echo "Configuration not found: ${CONFIG}" >&2; exit 2; }
done

mkdir -p "${OUTPUT}/recordings"

for MINUTES in 6 8 10; do
  TAG="tail_${MINUTES}min"
  TAIL="${OUTPUT}/recordings/${TAG}"
  if [[ -e "${TAIL}/tail_report.json" ]]; then
    echo "${TAG}: canonical tail already exists, keeping it"
  else
    "${PYTHON}" "${ROOT}/scripts/slice_canonical_tail.py" \
      --recording "${RECORDING}" \
      --minutes "${MINUTES}" \
      --output "${TAIL}"
  fi

  "${PYTHON}" "${ROOT}/src/map_build/build_from_canonical.py" \
    --recording "${TAIL}" \
    --camera-config "${CAMERA_CONFIG}" \
    --lidar-config "${LIDAR_CONFIG}" \
    --robot-config "${ROBOT_CONFIG}" \
    --output "${OUTPUT}/maps" \
    --name "${TAG}" \
    --wheelbase "${WHEELBASE}" \
    --encoder-speed-sign "${ENCODER_SPEED_SIGN}" \
    --steering-sign "${STEERING_SIGN}" \
    --mapping-yaw "${MAPPING_YAW}"
done

echo "Results: ${OUTPUT}/maps"
